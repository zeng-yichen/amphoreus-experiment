"""Stelle-based rewrite — replaces ``demiurge.CyreneStyleRewriter``.

The previous rewriter was a single Claude Opus call with a templated
4-step XML prompt. No voice substrate, no critic check, no iteration.
The model rewrote the draft using its generic "Cyrene the copyeditor"
prior — clean text, but disconnected from this client's actual voice
and from any audience-reaction signal.

This module replaces it with a Stelle-grade flow scoped to "rewrite
this single post addressing these comments":

    1. Pull this client's voice substrate via post_bundle (engagement-
       desc published posts, transcripts via Aglaea's surface — same
       substrate Stelle reads at generation time).
    2. Generate the rewrite via Opus, grounded in the substrate +
       operator comments. Single shot to start; the iteration below
       refines if needed.
    3. Run the rewrite through Irontomb (``simulate_flame_chase_journey``).
       If gestalt is positive AND every anchor reads as recognition or
       neutral, accept. If anchors flag specific lines, refine those
       spans and re-critique.
    4. Cap at ``_MAX_REWRITE_CYCLES`` iterations to bound cost.
    5. Run the final through Aglaea (``evaluate_client_comfort``) for
       voice-fidelity verification. If Aglaea flags spans, return them
       to the operator alongside the rewrite — operator decides
       whether to apply or re-rewrite.
    6. Return ``{final_post, irontomb_reactions, aglaea}`` so the API
       endpoint can persist the rewrite to ``content`` and surface
       critic provenance to the operator.

This is NOT a Stelle subprocess. It's the same critic-loop semantics
without the agent-loop scaffolding. Stelle's full agent loop is
overkill for rewrites (no topic discovery, no transcript mining
needed — operator already gave the topic + specific feedback). The
critic loop is the valuable piece.

Module-level singleton inside the API process; never raises (returns
a stub on any failure so the caller can degrade gracefully).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import anthropic

logger = logging.getLogger(__name__)

_REWRITE_MODEL = "claude-opus-4-6"
_REWRITE_MAX_TOKENS = 4096
# Iteration budget. One pass through Irontomb costs ~5-10s; capping at
# 3 keeps the operator-facing latency under ~30s for the worst case
# while still letting the rewrite refine on flagged spans.
_MAX_REWRITE_CYCLES = 3

# Per-Stelle-bundle char cap when feeding into the rewriter prompt.
# The full bundle for a creator with 100 calibration posts is ~80KB;
# we don't need that much context for a single post-rewrite. 30KB is
# enough to capture the engagement-desc top of the bundle.
_BUNDLE_CHAR_CAP = 30_000


_REWRITE_SYSTEM = """\
You are rewriting a single LinkedIn draft for a specific creator. Your
output text replaces the draft as the operator-facing display
version; it should match this creator's voice and address every
unresolved operator comment.

You have access to the creator's voice substrate below — published
posts ordered by reactions desc (top of the list = what actually
landed for this audience; bottom = what flopped). Pattern-match the
top performers for voice/structure cues, NOT the flat low-reaction
posts.

Hard rules:
  - Preserve every factual claim from the original draft.
  - Address every operator comment unless one explicitly contradicts
    a stronger signal in the substrate (rare).
  - Match the creator's voice from the substrate, not your generic
    "polished LinkedIn copywriter" prior.
  - No filler ("In today's world..."). No engineered closer cliches
    ("X isn't Y. It's Z."). Match the creator's natural sentence
    rhythm and closer style as visible in their high-engagement posts.

Output ONLY the rewritten post text. No XML wrappers, no preamble,
no explanation. Just the post body."""


_REFINE_SYSTEM = """\
You are refining a draft that just received critic feedback from
Irontomb (the LinkedIn-audience reaction simulator). Specific lines
got flagged. Rewrite ONLY the flagged spans, leaving the rest
untouched. Address each flag.

Output ONLY the refined post text. No XML, no preamble."""


def _build_voice_substrate(
    company: str, user_id: Optional[str]
) -> str:
    """Pull the engagement-desc bundle for voice context. Caller-side
    fallback to empty string on any failure — the rewrite still runs
    without voice substrate, just with weaker grounding."""
    try:
        from backend.src.services.post_bundle import build_post_bundle
        bundle = build_post_bundle(
            company,
            user_id=user_id,
            sort_by="engagement",
        )
        if bundle and len(bundle) > _BUNDLE_CHAR_CAP:
            bundle = bundle[:_BUNDLE_CHAR_CAP] + "\n\n[... bundle truncated for rewrite context ...]"
        return bundle or ""
    except Exception as exc:
        logger.debug("[stelle_rewrite] bundle build failed: %s", exc)
        return ""


def _format_feedback(prior_feedback: list[dict]) -> str:
    """Render operator comments as a bulletted block. Inline comments
    quote the selected text; post-wide are plain."""
    if not prior_feedback:
        return ""
    lines: list[str] = ["## Operator comments to address:"]
    for fb in prior_feedback:
        body = (fb.get("body") or "").strip()
        if not body:
            continue
        sel = (fb.get("selected_text") or "").strip()
        author = (fb.get("author_name") or fb.get("author_email") or "operator").strip()
        if sel:
            lines.append(f'- {author} on "{sel[:160]}":')
            lines.append(f"  → {body}")
        else:
            lines.append(f"- {author}: {body}")
    return "\n".join(lines) + "\n"


def _invoke_opus(system: str, user: str) -> str:
    """Single Opus call. Routes through CLI when ``AMPHOREUS_USE_CLI=true``,
    falls through to the Anthropic SDK otherwise — same routing pattern
    as the rest of the codebase."""
    try:
        from backend.src.mcp_bridge.claude_cli import use_cli, cli_single_shot
        if use_cli():
            txt = cli_single_shot(
                user, model="opus",
                max_tokens=_REWRITE_MAX_TOKENS,
                system_prompt=system,
                timeout=120,
            )
            return (txt or "").strip()
    except Exception as exc:
        logger.debug("[stelle_rewrite] CLI path failed, falling back to SDK: %s", exc)

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_REWRITE_MODEL,
            max_tokens=_REWRITE_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as exc:
        logger.warning("[stelle_rewrite] Opus call failed: %s", exc)
        return ""


def _critic_pass(company: str, draft_text: str) -> dict[str, Any]:
    """Run Irontomb on the draft and return her verdict.
    Caller-side fallback to empty dict on any failure."""
    try:
        from backend.src.mcp_bridge.claude_cli import (
            use_cli, simulate_flame_chase_journey_cli,
        )
        if use_cli():
            return simulate_flame_chase_journey_cli(company, draft_text) or {}
    except Exception as exc:
        logger.debug("[stelle_rewrite] Irontomb CLI failed: %s", exc)

    try:
        from backend.src.agents.irontomb import simulate_flame_chase_journey
        return simulate_flame_chase_journey(company, draft_text) or {}
    except Exception as exc:
        logger.warning("[stelle_rewrite] Irontomb call failed: %s", exc)
        return {}


def _is_landing(reaction: dict) -> bool:
    """Heuristic for "Irontomb thinks this lands." Gestalt has no
    obvious negative tokens AND every anchor reaction reads as
    positive or neutral. Conservative — false negatives are fine
    (will trigger another refine cycle); false positives cost a bad
    final."""
    if not reaction:
        return False
    gestalt = (reaction.get("reaction") or "").lower()
    neg_in_gestalt = any(
        bad in gestalt
        for bad in ("scrolled", "eyeroll", "fuck", "drifted", "flat",
                    "didn't pull", "didactic", "boring", "manufactured")
    )
    if neg_in_gestalt:
        return False
    anchors = reaction.get("anchors") or []
    if not anchors:
        return False
    for a in anchors:
        ar = (a.get("reaction") or "").lower()
        if any(bad in ar for bad in (
            "scrolled", "eyeroll", "fuck", "didactic", "manufactured",
            "engineered", "gpt", "drifted", "didn't pull", "feels assembled",
            "feels like a sales line", "thumb moves", "smells crafted",
        )):
            return False
    return True


def _refine_against_irontomb(
    draft: str, reaction: dict
) -> str:
    """Build a refine-prompt that quotes the flagged anchor spans and
    asks Opus to rewrite only those spans."""
    anchors = reaction.get("anchors") or []
    flagged: list[dict] = []
    for a in anchors:
        ar = (a.get("reaction") or "").lower()
        if any(bad in ar for bad in (
            "scrolled", "eyeroll", "fuck", "didactic", "manufactured",
            "engineered", "gpt", "drifted", "didn't pull",
        )):
            flagged.append(a)
    if not flagged:
        return draft  # no flags to act on; bail unchanged

    gestalt = reaction.get("reaction") or ""
    flag_lines: list[str] = [
        "Irontomb (LinkedIn-audience reaction simulator) returned:",
        f"  Gestalt: {gestalt}",
        "  Flagged spans:",
    ]
    for a in flagged:
        q = (a.get("quote") or "").strip()
        ar = (a.get("reaction") or "").strip()
        flag_lines.append(f'    - "{q}" → {ar}')

    user = (
        "<draft>\n"
        + draft
        + "\n</draft>\n\n"
        + "\n".join(flag_lines)
        + "\n\nRewrite ONLY the flagged spans. Leave everything else "
          "untouched. Output ONLY the refined post text."
    )
    refined = _invoke_opus(_REFINE_SYSTEM, user)
    return refined or draft


def rewrite_post_via_stelle_loop(
    *,
    company: str,
    user_id: Optional[str],
    post_text: str,
    prior_feedback: Optional[list[dict]] = None,
    style_instruction: str = "",
) -> dict[str, Any]:
    """Stelle-grade rewrite of a single post.

    Args:
      company:           company slug or UUID (for substrate lookup).
      user_id:           target FOC UUID (per-FOC scoping).
      post_text:         the existing draft text to rewrite.
      prior_feedback:    list of unresolved draft_feedback rows
                         (operator inline + post-wide comments). The
                         API endpoint pulls these from Supabase.
      style_instruction: optional free-text from the operator (rarely
                         used — usually the comments are the
                         instruction).

    Returns:
      ``{final_post: str, irontomb_reactions: list[dict],
         aglaea: dict | None, cycles: int, _model: str}``
      ``final_post`` is empty on total failure; the API endpoint
      should NOT persist an empty rewrite (callers check).
    """
    if not (post_text or "").strip():
        return {"final_post": "", "_error": "post_text is empty"}

    bundle = _build_voice_substrate(company, user_id)
    feedback_block = _format_feedback(prior_feedback or [])

    initial_user = (
        ("=== VOICE SUBSTRATE (read this for voice cues) ===\n"
         + bundle + "\n\n" if bundle else "")
        + "<original_draft>\n"
        + post_text
        + "\n</original_draft>\n\n"
        + feedback_block
        + (f"\n## Style note from operator:\n{style_instruction}\n"
           if style_instruction.strip() else "")
        + "\nRewrite this draft. Output ONLY the rewritten post body."
    )

    rewrite = _invoke_opus(_REWRITE_SYSTEM, initial_user)
    if not rewrite:
        return {
            "final_post": "",
            "_error": "initial rewrite call failed",
        }

    # Iterate against Irontomb — refine flagged spans up to N cycles.
    irontomb_reactions: list[dict] = []
    cycles_used = 0
    for i in range(_MAX_REWRITE_CYCLES):
        reaction = _critic_pass(company, rewrite)
        irontomb_reactions.append(reaction)
        cycles_used = i + 1
        if _is_landing(reaction):
            break
        # Refine against the flagged anchors and re-critique
        refined = _refine_against_irontomb(rewrite, reaction)
        if not refined or refined == rewrite:
            # Couldn't improve; ship what we have.
            break
        rewrite = refined

    # Aglaea gate — final voice-fidelity check.
    aglaea_result: dict | None = None
    try:
        from backend.src.agents.aglaea import evaluate_client_comfort
        aglaea_result = evaluate_client_comfort(
            rewrite,
            user_id=user_id,
            company_slug=company,
        )
    except Exception as exc:
        logger.warning("[stelle_rewrite] Aglaea call failed: %s", exc)
        aglaea_result = {"_error": f"aglaea failed: {str(exc)[:200]}"}

    return {
        "final_post": rewrite,
        "irontomb_reactions": irontomb_reactions,
        "aglaea": aglaea_result,
        "cycles": cycles_used,
        "_model": _REWRITE_MODEL,
    }
