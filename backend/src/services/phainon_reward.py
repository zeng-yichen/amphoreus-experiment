"""V2a reward function — extracted as a reusable module.

Predicts where a candidate LinkedIn post will fall in a specific
creator's own engagement distribution, given their post-era history
plus temporal + semantic-neighbor context.

Calibration evidence (2026-04-24):
  * Mark Hensley (Hensley Biostats), V2a-truth Spearman: +0.426
    → Phainon's full-mode reward gate is calibrated for this creator.
  * Sachil (Innovo): -0.366. Andrew Ettinger (Hume): -0.121.
    → Reward signal is anti-predictive or near-zero. Skip the gate.
  * Mark Schwartz / Heather Adkins / Grady Joseph: insufficient
    post-era data (<20 posts) — calibration deferred until they
    accumulate enough Virio-authored posts.

The module is deliberately framework-agnostic — pass in the data
arrays, get back a (band, reasoning, cost_usd) tuple. The Phainon
orchestrator decides whether to USE the band as a gate (full mode)
or ignore it (diverse / warm-start modes).

Output contract: integer band 1-5 + prose reasoning + cost in USD.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REWARD_MODEL          = "claude-opus-4-6"
_REWARD_MAX_TOKENS     = 400
# Opus 4.6 pricing per million tokens (matches irontomb.py + the calibration scripts).
_INPUT_COST_PER_MTOK   = 15.0
_OUTPUT_COST_PER_MTOK  = 75.0

# Body caps applied before sending to Opus — bound the input token budget.
_MAX_BODY_CHARS_CTX    = 1800
_MAX_BODY_CHARS_TGT    = 4000

_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    return (
        "You predict where a LinkedIn post will fall in a specific creator's "
        "OWN engagement distribution. You are given: (a) the creator's recent "
        "posts with reaction counts and timing, (b) the candidate post's top-3 "
        "nearest semantic neighbors from that history, and (c) the candidate "
        "post itself with its timing metadata.\n\n"
        "Bands:\n"
        "  1 = bottom 20% of this creator's posts\n"
        "  2 = 21-40%\n"
        "  3 = 41-60% (median)\n"
        "  4 = 61-80%\n"
        "  5 = top 20% (strongest engagement)\n\n"
        "Reason from the creator's own patterns, not from what's 'objectively "
        "good LinkedIn writing.' Cadence matters: a post published on a rest "
        "day with a fresh audience behaves differently than the 4th post in 7 "
        "days. Semantic neighbors tell you what happened the last time this "
        "creator wrote on this angle.\n\n"
        "Output strict JSON: {\"band\": <1-5>, \"reasoning\": \"<one sentence; "
        "cite content pattern OR timing OR neighbor outcome as your anchor>\"}. "
        "No preamble, no markdown."
    )


def _render_timestamp(iso: str) -> str:
    """``2026-04-20T16:00:23+00:00`` → ``2026-04-20 Mon 16:00 UTC``"""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return iso[:10]
    return f"{dt.strftime('%Y-%m-%d')} {_DOW[dt.weekday()]} {dt.strftime('%H:%M')} UTC"


def _gap_days(prev_iso: Optional[str], this_iso: str) -> Optional[float]:
    if not prev_iso or not this_iso:
        return None
    try:
        a = datetime.fromisoformat(prev_iso.replace("Z", "+00:00"))
        b = datetime.fromisoformat(this_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    return (b - a).total_seconds() / 86400


def _user_prompt(
    context_posts:       list[dict],
    target_body:         str,
    target_posted_at:    Optional[str],
    target_prev_post_at: Optional[str],
    cadence_7d:          Optional[int],
    neighbors:           list[tuple[float, dict]],
) -> str:
    lines: list[str] = []
    if cadence_7d is not None:
        lines.append(f"CREATOR CADENCE: {cadence_7d} posts in trailing 7d (incl. target)\n")

    lines.append("CREATOR'S RECENT POSTS (chronological — date · day-of-week · time · gap · reactions):")
    prev_iso = None
    for p in context_posts:
        iso = p.get("posted_at") or ""
        ts = _render_timestamp(iso)
        gap = _gap_days(prev_iso, iso)
        gap_str = f"+{gap:.1f}d" if gap is not None else "—"
        rx = p.get("total_reactions") or 0
        body = (p.get("post_text") or "").strip()[:_MAX_BODY_CHARS_CTX]
        lines.append(f"\n--- [{ts}, gap {gap_str}] {rx} reactions ---")
        lines.append(body)
        prev_iso = iso

    lines.append("\n" + "=" * 70)
    lines.append("CANDIDATE'S TOP-3 SEMANTIC NEIGHBORS IN THIS CREATOR'S HISTORY:")
    if not neighbors:
        lines.append("(no neighbor data available)")
    else:
        for sim, np_ in neighbors:
            iso = np_.get("posted_at") or ""
            ts = _render_timestamp(iso)[:16]
            rx = np_.get("total_reactions") or 0
            hook = (np_.get("post_text") or "").split("\n", 1)[0][:120]
            lines.append(f"  [{ts}] sim={sim:.2f}  {rx:>4} rx  \"{hook}\"")

    lines.append("\n" + "=" * 70)
    lines.append("CANDIDATE TO PREDICT:")
    if target_posted_at:
        ts = _render_timestamp(target_posted_at)
        gap = _gap_days(target_prev_post_at, target_posted_at)
        gap_str = f"+{gap:.1f}d" if gap is not None else "—"
        lines.append(f"Timing: {ts}, gap {gap_str} since last post")
    else:
        lines.append("Timing: hypothetical (no posted_at — Phainon-generated candidate)")
    lines.append("Body:")
    lines.append((target_body or "")[:_MAX_BODY_CHARS_TGT])

    lines.append("\nRespond with JSON only: "
                 "{\"band\": 1-5, \"reasoning\": \"<one sentence>\"}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic client (lazy)
# ---------------------------------------------------------------------------

_CLIENT = None
def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_candidate(
    *,
    candidate_text:      str,
    context_posts:       list[dict],
    neighbors:           list[tuple[float, dict]],
    target_posted_at:    Optional[str] = None,
    target_prev_post_at: Optional[str] = None,
    cadence_7d:          Optional[int] = None,
) -> tuple[Optional[int], Optional[str], float]:
    """Predict the engagement band for ``candidate_text`` given the creator's
    history. Returns ``(band, reasoning, cost_usd)``.

    Failure handling: returns ``(None, None, cost_so_far)`` on any error.
    Caller should treat None-band as "score unavailable, don't rank."
    """
    if not candidate_text or not (candidate_text or "").strip():
        return None, None, 0.0

    sys_prompt = _system_prompt()
    usr_prompt = _user_prompt(
        context_posts        = context_posts,
        target_body          = candidate_text,
        target_posted_at     = target_posted_at,
        target_prev_post_at  = target_prev_post_at,
        cadence_7d           = cadence_7d,
        neighbors            = neighbors,
    )
    try:
        resp = _client().messages.create(
            model=_REWARD_MODEL,
            max_tokens=_REWARD_MAX_TOKENS,
            system=sys_prompt,
            messages=[{"role": "user", "content": usr_prompt}],
        )
    except Exception:
        # Network blips, rate limits — let caller retry/skip
        return None, None, 0.0

    text = resp.content[0].text if resp.content else ""
    m = re.search(r"\{[^{}]*\"band\"[^{}]*\}", text, flags=re.DOTALL)
    if not m:
        return None, text[:200], _cost_of(resp)
    try:
        payload = json.loads(m.group(0))
        band = int(payload["band"])
        reasoning = str(payload.get("reasoning", ""))[:400]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, text[:200], _cost_of(resp)

    return band, reasoning, _cost_of(resp)


def _cost_of(resp) -> float:
    """Dollar cost of one Anthropic call from the usage block."""
    u = getattr(resp, "usage", None)
    if u is None:
        return 0.0
    return (
        (u.input_tokens  / 1e6) * _INPUT_COST_PER_MTOK
        + (u.output_tokens / 1e6) * _OUTPUT_COST_PER_MTOK
    )
