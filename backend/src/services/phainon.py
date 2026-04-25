"""Phainon — exemplar generator for Stelle's bundle.

Named after the Deliverer in the Honkai Star Rail Amphoreus arc, who
explored 33 million cycles against the Destruction. Our analog is
much smaller: per tracked FOC, generate N candidate drafts, optionally
score them against a calibrated reward model, persist the top
exemplars, and let Stelle retrieve them at draft-write time.

Three operating modes per FOC, distinguishable in ``creator_exemplars.mode``:

  * ``full``       — N=30 candidates, ALL scored by the V2a reward model,
                     top-K kept. Use when the reward Spearman ≥ 0.4 on
                     ground-truth-era calibration. Stelle treats these as
                     ranked exemplars.
  * ``diverse``    — N=15 candidates, NO scoring. Use when reward is
                     anti-predictive or near-zero (Sachil, Andrew). Top-K
                     ranking would mislead, so we just store the candidates
                     as-is and Stelle treats them as "directions Phainon
                     explored." Surfaces variety, not ranking.
  * ``warm_start`` — N=15 candidates, NO scoring, generated from the FOC's
                     full pre+post-Virio history. Use when post-Virio data
                     is too thin (<20 posts). Carries a caveat that voice
                     may include pre-Virio tone.

Bitter-Lesson posture
---------------------
* Reward function is an Opus forward pass — no trained classifier.
* Generation is Opus forward passes with angle/temperature variation —
  no fine-tuning, no SFT.
* Transfer to Stelle is via retrieval (Stelle reads exemplars in her
  prompt) — no weights crossed.
* Calibration is measurement-only; the gate is per-creator and
  re-measurable monthly.

Cost envelope
-------------
Per weekly run across the 6-creator prototype:
  full mode (1 creator):      $10-12 = (30 generations + 30 scorings)
  diverse mode (2 creators):  $1.50-2 each = $3-4
  warm_start mode (3 creators): $1.50-2 each = $4.50-6
Total: ~$20/week ≈ $85/month.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — hardcoded prototype roster
# ---------------------------------------------------------------------------

@dataclass
class _CreatorConfig:
    handle:        str
    company_label: str
    company_uuid:  Optional[str]   # for ``creator_exemplars.company_id``
    onboarding:    str             # YYYY-MM-DD
    mode:          str             # 'full' | 'diverse' | 'warm_start'
    n_candidates:  int
    k_top:         Optional[int] = None   # only used in 'full' mode

# Hardcoded for the prototype. Promote to a Supabase table once we
# decide the prototype delivers value.
PHAINON_PROTOTYPE_CREATORS: list[_CreatorConfig] = [
    _CreatorConfig(
        handle="markdavidhensley", company_label="Hensley Biostats",
        company_uuid="b980b0ea-0a43-4418-9726-3bbe10603993",
        onboarding="2026-02-11", mode="full",
        n_candidates=30, k_top=10,
    ),
    _CreatorConfig(
        handle="sachilv", company_label="InnovoCommerce",
        company_uuid="3888886d-b19c-400c-8d02-fb93bd849987",
        onboarding="2026-01-20", mode="diverse",
        n_candidates=15,
    ),
    _CreatorConfig(
        handle="andrewettinger23", company_label="Hume AI",
        company_uuid="9abcb96e-8b35-4963-bb1c-1b01c146a1c4",
        onboarding="2026-02-13", mode="diverse",
        n_candidates=15,
    ),
    _CreatorConfig(
        handle="mark-schwartz-27b5613", company_label="Trimble (Mark)",
        company_uuid="6b39e696-8d97-4651-83d8-79483903b4de",
        onboarding="2026-03-30", mode="warm_start",
        n_candidates=15,
    ),
    _CreatorConfig(
        handle="heather-adkins", company_label="Trimble (Heather)",
        company_uuid="6b39e696-8d97-4651-83d8-79483903b4de",
        onboarding="2026-04-06", mode="warm_start",
        n_candidates=15,
    ),
    _CreatorConfig(
        handle="gradyjoseph", company_label="TerraFort",
        company_uuid="430970aa-cbb5-419e-85f9-1982fd1d9f4f",
        onboarding="2026-04-06", mode="warm_start",
        n_candidates=15,
    ),
]


# Generation knobs.
#
# Phainon routes generation + scoring through the Claude CLI subprocess
# (``mcp_bridge.claude_cli.cli_single_shot``), same path Stelle / Cyrene
# / Castorice use. That means generation runs on the operator's Claude
# Pro/Team subscription, NOT on metered Anthropic API credits.
# Direct-API ``anthropic.Anthropic()`` was the original wiring; switched
# 2026-04-25 after the API credit balance ran out partway through the
# first prototype batch.
_GEN_MODEL              = "opus"   # CLI accepts "sonnet" | "opus"
_GEN_MAX_TOKENS         = 1200
_GEN_TIMEOUT_S          = 180
_GEN_HISTORY_FOR_VOICE  = 12       # how many prior posts to show as voice context

# The angle pool — cycle through these so candidates are varied.
# Each angle is structural / framing-level, not topical.
_ANGLES = [
    "personal anecdote with a concrete moment",
    "operational vignette — describe a specific working scene",
    "contrarian hot take backed by an observation",
    "data-driven thinkpiece with a single sharp point",
    "milestone reflection — gratitude-coded, not promotional",
    "question-led exploration — open with a real question",
    "client-story reveal — one composite anonymized example",
    "tooling or process critique — what's broken, why",
    "industry-trend observation grounded in field experience",
    "concrete how-to with a single counterintuitive insight",
]


# ---------------------------------------------------------------------------
# Run aggregate
# ---------------------------------------------------------------------------

@dataclass
class PhainonRunResult:
    creators_processed: int            = 0
    creators_skipped:   int            = 0
    candidates_total:   int            = 0
    candidates_scored:  int            = 0
    cost_usd:           float          = 0.0
    duration_seconds:   float          = 0.0
    errors:             list[str]      = field(default_factory=list)
    per_creator:        dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_phainon(creators: Optional[list[_CreatorConfig]] = None) -> PhainonRunResult:
    """One Phainon pass. Generates exemplars for every prototype creator
    and persists to ``creator_exemplars``.

    ``creators`` defaults to ``PHAINON_PROTOTYPE_CREATORS``. Pass a subset
    to scope a single-creator smoke run.
    """
    t0 = time.time()
    result = PhainonRunResult()
    if creators is None:
        creators = PHAINON_PROTOTYPE_CREATORS

    from backend.src.db.amphoreus_supabase import _get_client, is_configured
    if not is_configured():
        result.errors.append("Amphoreus Supabase not configured")
        return result
    sb = _get_client()
    if sb is None:
        result.errors.append("Amphoreus Supabase client unavailable")
        return result

    for cfg in creators:
        try:
            per = _run_one_creator(sb, cfg)
        except Exception as exc:
            logger.exception("[phainon] creator %s failed", cfg.handle)
            result.errors.append(f"{cfg.handle}: {str(exc)[:200]}")
            result.creators_skipped += 1
            continue

        result.creators_processed += 1
        result.candidates_total   += per["candidates_total"]
        result.candidates_scored  += per["candidates_scored"]
        result.cost_usd           += per["cost_usd"]
        result.per_creator[cfg.handle] = per
        logger.info(
            "[phainon] %s mode=%s candidates=%d scored=%d cost=$%.2f",
            cfg.handle, cfg.mode, per["candidates_total"],
            per["candidates_scored"], per["cost_usd"],
        )

    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(
        "[phainon] run complete: creators=%d/%d candidates=%d scored=%d "
        "cost=$%.2f duration=%.1fs errors=%d",
        result.creators_processed, len(creators),
        result.candidates_total, result.candidates_scored,
        result.cost_usd, result.duration_seconds, len(result.errors),
    )
    return result


# ---------------------------------------------------------------------------
# Per-creator orchestration
# ---------------------------------------------------------------------------

def _run_one_creator(sb, cfg: _CreatorConfig) -> dict[str, Any]:
    batch_id = str(_uuid.uuid4())
    onboarding_dt = datetime.fromisoformat(cfg.onboarding + "T00:00:00+00:00")

    # Pull creator history. ``warm_start`` uses the full mirror; ``full``
    # and ``diverse`` use post-onboarding only so voice context is
    # Virio-era only (no pre-Virio promotional voice contamination).
    all_posts = _load_posts(sb, cfg.handle)
    if cfg.mode == "warm_start":
        history = all_posts
    else:
        history = [p for p in all_posts
                   if (_parse_ts(p.get("posted_at")) or onboarding_dt) >= onboarding_dt]
    history.sort(key=lambda p: _parse_ts(p.get("posted_at"))
                 or datetime.min.replace(tzinfo=timezone.utc))

    if not history:
        return {
            "candidates_total": 0, "candidates_scored": 0, "cost_usd": 0.0,
            "skipped_reason": "no history", "mode": cfg.mode,
        }

    # Voice context for generation: most-recent N posts in the history.
    voice_ctx = history[-_GEN_HISTORY_FOR_VOICE:]

    # Generate candidates
    generations: list[dict] = []
    cost = 0.0
    for i in range(cfg.n_candidates):
        angle = _ANGLES[i % len(_ANGLES)]
        try:
            draft, gen_cost = _generate_candidate(cfg, voice_ctx, angle)
        except Exception as exc:
            logger.warning(
                "[phainon] generation failed (%s, angle=%r): %s",
                cfg.handle, angle[:30], exc,
            )
            continue
        cost += gen_cost
        if not draft or not draft.strip():
            continue
        generations.append({
            "draft": draft.strip(),
            "angle": angle,
            "gen_cost": gen_cost,
        })

    # Score (only for ``full`` mode + scoring helpers below)
    scored = []
    if cfg.mode == "full" and generations:
        from backend.src.services.phainon_reward import score_candidate
        # Embed neighbor pool once for the creator's history
        from backend.src.services.post_bundle import _embed_creator_posts
        creator_pool_for_neighbors = {p.get("provider_urn") or f"_idx_{i}": p
                                       for i, p in enumerate(history)}
        try:
            embeddings = _embed_creator_posts(creator_pool_for_neighbors)
        except Exception:
            embeddings = {}
        for g in generations:
            neighbors = _top_neighbors_for_text(
                g["draft"], history, embeddings,
            )
            band, reasoning, score_cost = score_candidate(
                candidate_text       = g["draft"],
                context_posts        = voice_ctx,
                neighbors            = neighbors,
                target_posted_at     = None,            # hypothetical
                target_prev_post_at  = (history[-1].get("posted_at") if history else None),
                cadence_7d           = _rolling_cadence(history, len(history) - 1),
            )
            cost += score_cost
            g["band"] = band
            g["reasoning"] = reasoning
            g["score_cost"] = score_cost
            scored.append(g)
        # Rank top-K by predicted band; ties broken by reasoning length
        # (heuristic: more-elaborated reasoning often means more confident
        # high-band call).
        scored.sort(
            key=lambda x: (-(x.get("band") or 0), -len(x.get("reasoning") or "")),
        )
        keep = scored[: (cfg.k_top or len(scored))]
    else:
        # Diverse / warm-start: store all generations, no rank
        keep = generations

    # Persist
    rows = []
    for rank, g in enumerate(keep, start=1):
        rows.append({
            "creator_handle": cfg.handle,
            "company_id":     cfg.company_uuid,
            "company_label":  cfg.company_label,
            "onboarding_date": cfg.onboarding,
            "mode":           cfg.mode,
            "batch_id":       batch_id,
            "angle":          g.get("angle"),
            "exemplar_text":  g["draft"],
            "predicted_band": g.get("band") if cfg.mode == "full" else None,
            "reasoning":      g.get("reasoning") if cfg.mode == "full" else None,
            "rank_within_batch": rank if cfg.mode == "full" else None,
            "creator_post_count_at_gen": len(history),
            "cost_usd":       round(g.get("gen_cost", 0) + g.get("score_cost", 0), 5),
        })
    if rows:
        try:
            sb.table("creator_exemplars").insert(rows).execute()
        except Exception as exc:
            logger.exception("[phainon] persist failed for %s", cfg.handle)
            return {
                "candidates_total": len(generations),
                "candidates_scored": len(scored),
                "cost_usd": cost,
                "persist_error": str(exc)[:300],
                "mode": cfg.mode,
            }

    return {
        "candidates_total": len(generations),
        "candidates_scored": len(scored),
        "cost_usd": round(cost, 4),
        "history_n": len(history),
        "kept_n": len(rows),
        "mode": cfg.mode,
        "batch_id": batch_id,
    }


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

def _generate_candidate(
    cfg: _CreatorConfig,
    voice_ctx: list[dict],
    angle: str,
) -> tuple[str, float]:
    """One Opus generation via Claude CLI subprocess. Returns
    ``(draft_text, cost_usd)``. Cost is always 0.0 — the CLI runs on
    the operator's Claude subscription, not metered API credits;
    detailed token usage is recorded by ``cli_single_shot`` itself
    via ``_record_cli_usage``."""
    # System prompt — voice instructions + output contract
    sys_prompt = (
        f"You are drafting a LinkedIn post for {cfg.company_label}. "
        f"You write in the creator's authentic voice. Reference the "
        f"creator's recent posts as voice grounding — match their cadence, "
        f"vocabulary, opener style, paragraph structure, and topical "
        f"register. Do NOT copy any one prior post; produce a NEW draft "
        f"that fits their voice but covers a different beat."
        f"\n\n"
        f"Output strict JSON: {{\"draft\": \"<the post body>\", "
        f"\"hook\": \"<the first line>\"}}. "
        f"No preamble, no markdown, no discussion."
    )

    # User prompt — voice context + angle directive
    lines = [
        f"CREATOR'S RECENT POSTS (chronological, with reactions for context):\n"
    ]
    for p in voice_ctx:
        iso = (p.get("posted_at") or "")[:10]
        rx = p.get("total_reactions") or 0
        body = (p.get("post_text") or "").strip()[:1500]
        lines.append(f"--- [{iso}] {rx} rx ---")
        lines.append(body)
        lines.append("")

    lines.append("=" * 60)
    lines.append("\nDIRECTIVE FOR THIS DRAFT:")
    lines.append(f"  Angle: {angle}")
    lines.append("  Length: 200-450 words. Real working insights only — no")
    lines.append("  hype, no buzzwords, no closing call-to-action.")
    lines.append("  Avoid topics already covered in the recent history above.")
    lines.append("\nProduce the JSON now.")

    usr_prompt = "\n".join(lines)

    from backend.src.mcp_bridge.claude_cli import cli_single_shot

    text = cli_single_shot(
        prompt=usr_prompt,
        system_prompt=sys_prompt,
        model=_GEN_MODEL,
        max_tokens=_GEN_MAX_TOKENS,
        timeout=_GEN_TIMEOUT_S,
    )
    if text is None:
        raise RuntimeError("cli_single_shot returned None (CLI subprocess failed)")

    # Tolerant JSON extraction — same as before
    m = re.search(r"\{[^{}]*\"draft\"[^{}]*\}", text, flags=re.DOTALL)
    if not m:
        m = re.search(r"\{.*?\"draft\".*?\}", text, flags=re.DOTALL)
    if m:
        try:
            payload = json.loads(m.group(0))
            return str(payload.get("draft", "")), 0.0
        except Exception:
            pass
    # Fallback: treat the full text as the draft (model went off-format)
    return text.strip()[:5000], 0.0


# ---------------------------------------------------------------------------
# Helpers (lightweight; we deliberately avoid importing more from post_bundle
# than necessary to keep this module self-contained for tests)
# ---------------------------------------------------------------------------

def _parse_ts(iso):
    if not iso: return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except Exception:
        return None


def _load_posts(sb, creator_handle: str) -> list[dict]:
    rows: list[dict] = []
    offset, page = 0, 1000
    while True:
        resp = (
            sb.table("linkedin_posts")
              .select("provider_urn,post_text,posted_at,total_reactions,total_comments,total_reposts,is_company_post")
              .eq("creator_username", creator_handle)
              .or_("is_company_post.is.null,is_company_post.eq.false")
              .not_.is_("post_text", "null").not_.is_("posted_at", "null")
              .order("posted_at", desc=False)
              .range(offset, offset + page - 1)
              .execute()
        )
        d = resp.data or []
        rows.extend(d)
        if len(d) < page: break
        offset += page
    # Filter to non-trivial bodies + recorded reactions
    return [
        r for r in rows
        if (r.get("post_text") or "").strip() and len((r.get("post_text") or "").strip()) >= 40
        and r.get("total_reactions") is not None
    ]


def _rolling_cadence(posts: list[dict], idx: int) -> int:
    """Posts in the trailing 7 days from posts[idx]'s timestamp, inclusive."""
    if not posts or idx < 0:
        return 0
    anchor = _parse_ts(posts[idx].get("posted_at"))
    if anchor is None: return 0
    cutoff = anchor.timestamp() - 7 * 86400
    ct = 0
    for j in range(idx + 1):
        t = _parse_ts(posts[j].get("posted_at"))
        if t is None: continue
        if cutoff <= t.timestamp() <= anchor.timestamp():
            ct += 1
    return ct


def _top_neighbors_for_text(
    text: str,
    history: list[dict],
    embeddings: dict[str, list[float]],
    k: int = 3,
) -> list[tuple[float, dict]]:
    """For a candidate text, find its top-k semantic neighbors in the
    creator's history using ``post_bundle._embed_single_text`` for the
    candidate + the precomputed history embeddings."""
    if not embeddings:
        return []
    try:
        from backend.src.services.post_bundle import _embed_single_text
    except Exception:
        return []
    q = _embed_single_text(text)
    if not q:
        return []
    scored: list[tuple[float, dict]] = []
    for p in history:
        urn = p.get("provider_urn") or f"_idx_{history.index(p)}"
        emb = embeddings.get(urn)
        if not emb or len(emb) != len(q):
            continue
        # text-embedding-3-small is L2-normalized → dot = cosine
        dot = 0.0
        for x, y in zip(q, emb):
            dot += x * y
        scored.append((dot, p))
    scored.sort(key=lambda t: -t[0])
    return scored[:k]


# ---------------------------------------------------------------------------
# CLI — manual single-creator smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        for c in (Path(__file__).resolve().parents[3] / ".env",
                  Path.cwd() / ".env"):
            if c.exists():
                load_dotenv(c); break
    except Exception:
        pass

    # Optional: --handle <handle> to scope to one creator
    handle_arg = None
    if "--handle" in _sys.argv:
        handle_arg = _sys.argv[_sys.argv.index("--handle") + 1]
    creators = (
        [c for c in PHAINON_PROTOTYPE_CREATORS if c.handle == handle_arg]
        if handle_arg else None
    )
    r = run_phainon(creators)
    print(json.dumps({
        "creators_processed": r.creators_processed,
        "creators_skipped":   r.creators_skipped,
        "candidates_total":   r.candidates_total,
        "candidates_scored":  r.candidates_scored,
        "cost_usd":           round(r.cost_usd, 4),
        "duration_seconds":   r.duration_seconds,
        "errors":             r.errors,
        "per_creator":        {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                                for k, v in r.per_creator.items()},
    }, indent=2, default=str))
