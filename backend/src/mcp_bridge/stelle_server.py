#!/usr/bin/env python3
"""MCP server exposing Stelle's custom tools for Claude CLI.

The Claude CLI already provides filesystem (Read/Write/Edit/Bash/Grep/Glob)
and web (WebSearch/WebFetch) tools natively. This server exposes only the
custom tools that don't map to built-in CLI capabilities:

  - query_observations — scored post history with engagement + reactors
  - query_top_engagers — aggregated top ICP engagers
  - search_linkedin_corpus — 200K+ LinkedIn post corpus
  - execute_python — sandboxed Python with pre-loaded observations
  - write_result — terminal tool with structural validation

Irontomb (simulate_flame_chase_journey) is deliberately NOT exposed
to Stelle during generation. Previously, requiring Stelle to iterate
against Irontomb's engagement predictions collapsed her writing toward
Irontomb's taste bias (precedent-favored, LinkedIn-average). Stelle
now writes authentically; Irontomb evaluates her final drafts
post-hoc (see _process_result in stelle.py) so its predictions can
be calibrated against real engagement without distorting the writing.

Launched as a subprocess by Claude CLI via --mcp-config.
Reads config from environment variables:
  STELLE_COMPANY — client slug
  STELLE_USE_CLI_IRONTOMB — "1" to run Irontomb through CLI too (default,
    used by post-hoc evaluator, not by Stelle directly)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.src.mcp_bridge.server import MCPServer

_COMPANY = os.environ.get("STELLE_COMPANY", "")
_USE_CLI_IRONTOMB = os.environ.get("STELLE_USE_CLI_IRONTOMB", "1") in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Stateful per-run tracking
# ---------------------------------------------------------------------------
_simulate_call_count = 0
_simulate_results: list[dict] = []

# Aglaea (check_client_comfort) calls keyed by draft hash. Used by the
# submit_draft gate (2026-05-01) to refuse persistence when no comfort
# call has fired on the EXACT content being submitted. Forces Stelle to
# run one final voice-fidelity check on whatever she's actually
# shipping, even if she ran comfort on an earlier iteration.
_comfort_results: dict[str, dict] = {}

# Pre-load scored observations on startup
_scored_observations: list[dict] = []
_client_median_engagement: float | None = None
_client_median_impressions: int | None = None


def _init_observations():
    """Load scored observations and compute median engagement."""
    global _scored_observations, _client_median_engagement, _client_median_impressions
    if not _COMPANY:
        return
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(_COMPANY) or {}
        _scored_observations = [
            o for o in state.get("observations", [])
            if o.get("status") in ("scored", "finalized")
        ]

        # Compute median engagement
        rates = []
        impressions = []
        for obs in _scored_observations:
            raw = (obs.get("reward") or {}).get("raw_metrics", {})
            imp = raw.get("impressions", 0)
            react = raw.get("reactions", 0)
            if imp > 0:
                rates.append(react / imp * 1000)
                impressions.append(imp)
        rates.sort()
        impressions.sort()
        if rates:
            mid = len(rates) // 2
            _client_median_engagement = (
                rates[mid] if len(rates) % 2
                else (rates[mid - 1] + rates[mid]) / 2
            )
        if impressions:
            mid = len(impressions) // 2
            _client_median_impressions = int(
                impressions[mid] if len(impressions) % 2
                else (impressions[mid - 1] + impressions[mid]) / 2
            )
    except Exception as e:
        print(f"Warning: could not load observations: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------
server = MCPServer("stelle-tools")


# ---------------------------------------------------------------------------
# Tool: query_observations
# ---------------------------------------------------------------------------
def _handle_query_observations(args: dict) -> str:
    from backend.src.agents.analyst import _tool_query_observations
    return _tool_query_observations(args, _scored_observations)

# query_observations intentionally NOT registered. Observation data is
# now injected up-front via memory/post-history.md at workspace stage
# time (see _build_observation_digest in stelle.py). Same raw data,
# delivered as a readable file — no tool call required.


# ---------------------------------------------------------------------------
# Tool: query_top_engagers
# ---------------------------------------------------------------------------
def _handle_query_top_engagers(args: dict) -> str:
    if not _COMPANY:
        return json.dumps({"error": "company not set"})
    limit = min(args.get("limit", 20), 50)
    try:
        from backend.src.db.local import get_top_icp_engagers
        engagers = get_top_icp_engagers(_COMPANY, limit=limit)
        return json.dumps({"count": len(engagers), "engagers": engagers}, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)[:200]})

# query_top_engagers intentionally NOT registered. Went unused in
# production (5 calls across 9 sessions). We want Stelle optimizing for
# engagement first; ICP-specific scoring is a downstream concern.


# ---------------------------------------------------------------------------
# Tool: search_linkedin_corpus
# ---------------------------------------------------------------------------
def _handle_search_corpus(args: dict) -> str:
    from backend.src.agents.analyst import _tool_search_linkedin_bank
    return _tool_search_linkedin_bank(args)

# search_linkedin_corpus intentionally NOT registered. Went unused in
# production (0 calls across 9 sessions). LinkedIn-wide corpus pulls
# Stelle's taste toward LinkedIn-average polish and away from the
# client's own voice; the reference distribution she should learn from
# is her client's past top performers, which lives in
# memory/post-history.md.


# ---------------------------------------------------------------------------
# Tool: execute_python
# ---------------------------------------------------------------------------
def _handle_execute_python(args: dict) -> str:
    from backend.src.agents.analyst import _tool_execute_python
    try:
        from backend.src.utils.post_embeddings import get_post_embeddings
        emb = get_post_embeddings(_COMPANY)
    except Exception:
        emb = None
    return _tool_execute_python(args, _scored_observations, embeddings=emb)

# execute_python intentionally NOT registered. Went nearly unused in
# production (4 calls across 9 sessions). Stelle's job is writing, not
# data analysis; observation data is delivered via
# memory/post-history.md already.


# ---------------------------------------------------------------------------
# Tool: simulate_flame_chase_journey
# ---------------------------------------------------------------------------
def _handle_simulate(args: dict) -> str:
    global _simulate_call_count
    _simulate_call_count += 1

    draft = args.get("draft_text", "")
    if not draft:
        return json.dumps({"_error": "draft_text is required"})

    try:
        if _USE_CLI_IRONTOMB:
            from backend.src.mcp_bridge.claude_cli import simulate_flame_chase_journey_cli
            result = simulate_flame_chase_journey_cli(_COMPANY, draft)
        else:
            from backend.src.agents.irontomb import simulate_flame_chase_journey
            result = simulate_flame_chase_journey(_COMPANY, draft)
    except Exception as e:
        return json.dumps({"_error": f"simulate failed: {str(e)[:200]}"})

    _dh = result.get("_draft_hash", "")
    _simulate_results.append({"draft_hash": _dh, "result": result})

    # --- Gradient signal ---
    pred_eng = result.get("engagement_prediction", 0) or 0
    pred_imp = result.get("impression_prediction", 0) or 0
    gradient: dict[str, Any] = {}

    if _client_median_engagement is not None:
        delta = pred_eng - _client_median_engagement
        gradient["client_median_engagement"] = round(_client_median_engagement, 2)
        gradient["predicted_engagement"] = round(pred_eng, 2)
        gradient["delta_vs_median"] = round(delta, 2)
        if delta < 0:
            gradient["signal"] = (
                f"BELOW median by {abs(delta):.1f}. "
                f"This draft would underperform baseline. Revise and re-simulate."
            )
        else:
            gradient["signal"] = (
                f"ABOVE median by {delta:.1f}. Predicted to outperform baseline."
            )

    if _client_median_impressions is not None:
        gradient["client_median_impressions"] = _client_median_impressions
        gradient["predicted_impressions"] = pred_imp

    # Trajectory
    prev_preds = [
        sr["result"].get("engagement_prediction", 0) or 0
        for sr in _simulate_results[:-1]
        if sr["draft_hash"] == _dh
    ]
    if prev_preds:
        gradient["revision_trajectory"] = [round(p, 2) for p in prev_preds] + [round(pred_eng, 2)]
        improvement = pred_eng - prev_preds[-1]
        gradient["last_revision_delta"] = round(improvement, 2)
        if abs(improvement) < 0.5 and len(prev_preds) >= 2:
            gradient["plateau_detected"] = True
            gradient["plateau_note"] = (
                "Engagement prediction has plateaued. Ship if above median, "
                "or try a fundamentally different hook/angle if below."
            )

    if gradient:
        result["_gradient"] = gradient

    return json.dumps(result, default=str)

# simulate_flame_chase_journey intentionally NOT registered. Irontomb
# has been unplugged from Stelle's generation loop — she writes
# authentically, and Irontomb runs post-hoc on her final drafts (see
# _process_result in stelle.py). The _handle_simulate function above
# is kept dormant in case we want to expose it again as an optional
# sanity-check tool in the future, but it is not part of Stelle's
# toolbelt right now.



# ---------------------------------------------------------------------------
# Tool: get_reader_reaction (Irontomb rough-reader adversarial loop)
# ---------------------------------------------------------------------------
def _handle_get_reader_reaction(args: dict) -> str:
    global _simulate_call_count
    _simulate_call_count += 1
    draft = args.get("draft_text", "")
    if not draft:
        return json.dumps({"_error": "draft_text is required"})

    # 2026-04-29: per-run draft-hash cache.
    #
    # Irontomb's verdicts are stochastic — the same draft critiqued
    # twice can return contradictory line-level reactions ("Vibes."
    # called "that's the line" on iter N, "every AI post does this"
    # on iter N+1). Stelle iterating on minor closer tweaks would
    # trigger fresh critic calls on a substantially-unchanged draft
    # and get whipsawed by contradictory feedback, sometimes
    # abandoning a draft her FIRST critique had rated positively
    # (Andrew Track 3 / voice-AI / Appen weekend, run f5158db7).
    #
    # Memoize on the exact draft hash. If Stelle calls
    # get_reader_reaction with the SAME text twice in a session,
    # return the cached verdict. ~10 LOC fix; no taxonomy change;
    # any actual edit to the draft yields a fresh verdict because
    # the hash changes.
    from backend.src.agents.irontomb import _draft_hash as _hash_draft
    _dh_in = _hash_draft(draft)
    for prior in _simulate_results:
        if prior.get("draft_hash") == _dh_in:
            cached = dict(prior.get("result") or {})
            cached["_cache_hit"] = True
            # Re-emit as enriched response (trajectory built below path
            # is bypassed — cached hit means Stelle didn't actually edit
            # the draft, so the trajectory is the same as last time).
            return json.dumps(cached, default=str)

    try:
        from backend.src.mcp_bridge.claude_cli import use_cli
        if _USE_CLI_IRONTOMB and use_cli():
            from backend.src.mcp_bridge.claude_cli import simulate_flame_chase_journey_cli
            result = simulate_flame_chase_journey_cli(_COMPANY, draft)
        else:
            from backend.src.agents.irontomb import simulate_flame_chase_journey
            result = simulate_flame_chase_journey(_COMPANY, draft)
    except Exception as e:
        return json.dumps({"_error": f"reader call failed: {str(e)[:200]}"})

    # Build a trajectory snapshot: the last 5 reactions from this session,
    # each with the draft's first line and length so Stelle can distinguish
    # "recent reactions on the post I'm iterating on now" from "earlier
    # reactions on a different post." Lets her see whether her edits are
    # moving the signal, not just the latest data point.
    trajectory: list[dict] = []
    for prior in _simulate_results[-5:]:
        pr_result = prior.get("result", {}) or {}
        pr_draft = prior.get("draft", "") or ""
        # Show the gestalt + every anchor (quote + reaction) so Stelle
        # sees WHICH spans got which feedback, not just a single
        # collapsed pointer. Lets her track localized signal across
        # iterations: e.g. "the closer was anchored negative on iter 1,
        # I rewrote it, anchor disappeared on iter 2 — closer is now
        # fine, leave it; the middle just acquired a negative anchor,
        # focus there next."
        pr_anchors = pr_result.get("anchors")
        if not isinstance(pr_anchors, list):
            pr_anchors = []
        trajectory.append({
            "draft_first_line": pr_draft.split("\n")[0][:80] if pr_draft else "",
            "draft_len": len(pr_draft),
            "reaction": pr_result.get("reaction", ""),
            "anchors":  pr_anchors,
        })

    # Persist THIS call so future invocations can see it as prior history.
    _dh = result.get("_draft_hash", "")
    _simulate_results.append({"draft_hash": _dh, "draft": draft, "result": result})

    # Compose the return: current reaction + trajectory so Stelle can
    # read both in one response.
    enriched = dict(result)
    if trajectory:
        enriched["_prior_reactions"] = trajectory

    # Fire-and-forget convergence log. See services/convergence_log.py
    # for the dataset-building rationale. Never raises; the critic's
    # return value is unchanged regardless of whether this succeeds.
    try:
        from backend.src.services.convergence_log import log_irontomb_call
        log_irontomb_call(_COMPANY, draft, enriched)
    except Exception:
        pass

    return json.dumps(enriched, default=str)


server.register(
    name="get_reader_reaction",
    description=(
        "Send a draft to Irontomb, a rough-reader simulator. Returns "
        "a gestalt reader-voice reaction PLUS a list of inline anchors "
        "(at least 1; one per reader-state-change moment, or one on "
        "the line that represents the texture for uniformly-meh "
        "drafts).\n\n"
        "Response shape:\n"
        "  reaction — under-15-word GESTALT reaction (net effect after "
        "the whole draft).\n"
        "  anchors  — list of {quote, reaction}. quote = verbatim 3-15 "
        "words from the draft that triggered the shift; reaction = "
        "short reader-voice response to that span. Always at least 1.\n\n"
        "Use anchors as a localized gradient: revise spans the reader "
        "anchored negatively, leave spans they didn't anchor or "
        "anchored positively. Don't rewrite what's working."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "draft_text": {"type": "string"},
        },
        "required": ["draft_text"],
    },
    handler=_handle_get_reader_reaction,
)


# ---------------------------------------------------------------------------
# Tool: write_result (terminal, with guards)
# ---------------------------------------------------------------------------
def _handle_write_result(args: dict) -> str:
    raw_json = args.get("result_json", "")
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return json.dumps({"_error": f"Invalid JSON: {e}"})

    # Validate output structure
    from backend.src.agents.stelle import _validate_output
    passed, val_errors, val_warnings = _validate_output(parsed)
    if not passed:
        return json.dumps({
            "_error": "Validation failed",
            "errors": val_errors,
            "warnings": val_warnings,
        })

    # (Irontomb gates removed — simulate_flame_chase_journey is no
    # longer part of Stelle's toolbelt during generation. Final-post
    # Irontomb evaluation happens post-hoc in _process_result.)

    # Write the result to a known location so the caller can read it
    result_path = os.path.join(_PROJECT_ROOT, ".stelle_cli_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False, default=str)

    return json.dumps({
        "accepted": True,
        "n_posts": len(parsed.get("posts", [])),
        "result_path": result_path,
        "warnings": val_warnings,
    })

server.register(
    name="write_result",
    description=(
        "Submit your final posts (ends the session). Validates output "
        "structure. Pass the full output as a JSON string in result_json."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "result_json": {
                "type": "string",
                "description": "The full output JSON as a string",
            },
        },
        "required": ["result_json"],
    },
    handler=_handle_write_result,
)


# ---------------------------------------------------------------------------
# Tool: submit_draft (per-draft persistence to local_posts → Posts tab)
# ---------------------------------------------------------------------------
# CLI-mode Stelle needs parity with native-mode Stelle's submit_draft tool.
# Without this, posts generated under AMPHOREUS_USE_CLI never land in the
# local_posts table and the Posts tab on amphoreus.app stays empty.
#
# 2026-05-01: Castorice is now wired into the CLI path (was bypassed
# previously — Stelle's why_post arg went straight through and operators
# saw Stelle-style audit text in the operator-facing why_post field
# instead of Castorice's Cyrene-brief-grounded strategic-fit verdict).
# Both native and CLI paths now run through the same
# ``apply_castorice_to_submit_args`` helper, which maps:
#   args.why_post / args.process_notes  → forwarded.process_notes
#   Castorice strategic_fit_note         → forwarded.why_post
#   Castorice fact-check report          → forwarded.fact_check_report
#   Castorice citation strings           → forwarded.citation_comments
def _handle_submit_draft(args: dict) -> str:
    if not _COMPANY:
        return "Error: STELLE_COMPANY env var not set"
    from pathlib import Path as _Path
    from backend.src.agents.stelle import (
        _dispatch_submit_draft,
        apply_castorice_to_submit_args,
    )

    args = args or {}
    content = (args.get("content") or "").strip()
    if not content:
        return "Error: content is required"

    # 2026-05-01: Aglaea gate. submit_draft refuses to persist unless
    # check_client_comfort has fired on this EXACT content this session.
    # Reason: prior runs shipped 4-of-6 posts with zero comfort checks
    # because Stelle's prompt instruction ("until BOTH critics pass")
    # was advisory; she'd skip Aglaea under time pressure or after
    # Irontomb passed. Now structurally enforced.
    #
    # The hash is on the FINAL content the operator will see, so an
    # earlier comfort call on iter-2 doesn't satisfy a submit on iter-3.
    # Stelle has to run comfort on whatever she's actually shipping.
    try:
        from backend.src.agents.irontomb import _draft_hash as _hash_draft
        _dh = _hash_draft(content)
    except Exception:
        _dh = None

    if _dh and _dh not in _comfort_results:
        return (
            "Error: submit_draft REFUSED — no check_client_comfort call "
            "has fired on this exact draft text this session. Run "
            "check_client_comfort(draft_text=<final content>) first, "
            "then retry submit_draft. Aglaea's voice-fidelity check is "
            "non-optional; the operator-facing review surface depends "
            "on it. Earlier comfort calls on prior iterations don't "
            "count — the gate hashes the FINAL content."
        )

    try:
        forwarded = apply_castorice_to_submit_args(_COMPANY, args)
    except Exception as e:
        # apply_castorice never raises by design, but guard anyway —
        # if Castorice itself fails fully, fall through to raw dispatch
        # so the draft still persists.
        logger.warning("[stelle_server] Castorice wrap failed: %s", e)
        forwarded = dict(args)

    # The native dispatcher pulls company from env (STELLE_COMPANY_KEYWORD /
    # DATABASE_COMPANY_ID), so we just forward args + a workspace root.
    # Root is only used for the markdown mirror path; CWD-relative is fine.
    try:
        return _dispatch_submit_draft(_Path("."), forwarded)
    except Exception as e:
        return f"Error: submit_draft failed: {e}"


server.register(
    name="submit_draft",
    description=(
        "Persist one final draft to the Posts tab (local_posts table, "
        "reviewable at amphoreus.app/ghostwriter/{company}). Call this once "
        "per finished post. Returns a confirmation + draft_id.\n\n"
        "Args:\n"
        "  user_slug (required): which FOC user this post is for\n"
        "  content (required): the final markdown post text\n"
        "  scheduled_date (optional): YYYY-MM-DD slot (tomorrow or later)\n"
        "  publication_order (optional): 1, 2, 3… for multi-post runs\n"
        "  why_post (optional): rationale shown alongside the draft"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "user_slug": {"type": "string"},
            "content": {"type": "string"},
            "scheduled_date": {"type": "string"},
            "publication_order": {"type": "integer"},
            "why_post": {"type": "string"},
        },
        "required": ["user_slug", "content"],
    },
    handler=_handle_submit_draft,
)


# ---------------------------------------------------------------------------
# check_client_comfort — Aglaea gate (brand-safety + voice fit)
# ---------------------------------------------------------------------------
# In API mode this tool lives as a Python tool_handlers entry in
# stelle.py:3913. In CLI mode Stelle can't reach that handler, so the
# tool was invisible — she couldn't call Aglaea at all. Register it
# here as an MCP tool so CLI-mode Stelle has the same draft-gating
# affordance. Handler just delegates to the same Python function the
# API path uses, keeping the semantics identical across modes.

def _handle_check_client_comfort(args: dict) -> str:
    import json as _json
    import os as _os
    draft = (args.get("draft_text") or args.get("draft") or "").strip()
    if not draft:
        return _json.dumps({"_error": "draft_text is required"})
    user_slug = (_os.environ.get("DATABASE_USER_SLUG") or "").strip() or None
    company = _COMPANY
    try:
        from backend.src.agents.aglaea import evaluate_client_comfort
        result = evaluate_client_comfort(
            draft,
            user_slug=user_slug,
            company_slug=company,
        )
        # Track this call for the submit_draft gate. Hash the exact
        # draft text so submit_draft can verify comfort fired on
        # *this* content (not a stale earlier iteration).
        try:
            from backend.src.agents.irontomb import _draft_hash as _hash_draft
            _comfort_results[_hash_draft(draft)] = {
                "result": result,
                "draft_first_line": draft.split("\n", 1)[0][:80],
                "draft_len": len(draft),
            }
        except Exception:
            pass
        # Fire-and-forget convergence log. See services/convergence_log.py
        # for the dataset-building rationale. Never raises; the critic's
        # return value is unchanged regardless of whether this succeeds.
        try:
            from backend.src.services.convergence_log import log_aglaea_call
            log_aglaea_call(company, draft, result)
        except Exception:
            pass
        return _json.dumps(result, default=str)
    except Exception as e:
        return _json.dumps({"_error": f"aglaea failed: {str(e)[:200]}"})


# ---------------------------------------------------------------------------
# retrieve_similar_posts — cross-creator corpus precedence search
# ---------------------------------------------------------------------------
# Registered in both CLI mode (here) and API mode (stelle.py::_TOOL_HANDLERS).
# Backed by the ~390k-post Jacquard mirror + post_embeddings pgvector index;
# lets Stelle escape the FOC's local posting basin by querying cross-creator
# patterns grounded in real engagement numbers.

def _handle_retrieve_similar_posts(args: dict) -> str:
    import json as _json
    try:
        from backend.src.services.post_retrieval import retrieve_similar_posts
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": f"import failed: {exc}"})
    query = (args.get("query") or "").strip()
    if not query:
        return _json.dumps({"count": 0, "posts": [], "error": "query is required"})
    k = int(args.get("k") or 10)
    k = max(1, min(k, 50))
    min_reactions = int(args.get("min_reactions") or 0)
    exclude_creator = args.get("exclude_creator") or None
    try:
        rows = retrieve_similar_posts(
            query=query, k=k, min_reactions=min_reactions,
            exclude_creator=exclude_creator,
        )
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": str(exc)[:400]})
    return _json.dumps({"count": len(rows), "posts": rows}, default=str)


server.register(
    name="retrieve_similar_posts",
    description=(
        "Semantic search over the 390k-post LinkedIn corpus (Jacquard mirror "
        "+ OpenAI text-embedding-3-small vectors). Returns real posts with "
        "engagement numbers — use it to ground new angles in proven structures "
        "from other creators, escape a single FOC's local basin, or find "
        "precedent for a risky take.\n\n"
        "Express any content-type filtering directly in the natural-language "
        "query (e.g. 'first-person narrative about failure, NOT a product "
        "announcement'). No hand-labeled archetype filter.\n\n"
        "Args:\n"
        "  query (required): natural-language description of what to search for\n"
        "  k (optional, default 10, max 50): number of posts to return\n"
        "  min_reactions (optional): only return posts at or above this count\n"
        "  exclude_creator (optional): LinkedIn username to omit (e.g. the current client)"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query":              {"type": "string"},
            "k":                  {"type": "integer"},
            "min_reactions":      {"type": "integer"},
            "exclude_creator":    {"type": "string"},
        },
        "required": ["query"],
    },
    handler=_handle_retrieve_similar_posts,
)


# ---------------------------------------------------------------------------
# retrieve_recent_peer_bangers — temporal + engagement-ranked retrieval
# ---------------------------------------------------------------------------
# Same corpus as retrieve_similar_posts, but filtered to posts from the last
# N days and re-ranked by engagement within that window. Grounds Stelle in
# what's landing with the audience RIGHT NOW rather than all-time-great
# reference posts that may use stale hook patterns. See
# services/post_retrieval.py::retrieve_recent_peer_bangers for mechanics.

def _handle_retrieve_recent_peer_bangers(args: dict) -> str:
    import json as _json
    try:
        from backend.src.services.post_retrieval import retrieve_recent_peer_bangers
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": f"import failed: {exc}"})
    query = (args.get("query") or "").strip()
    if not query:
        return _json.dumps({"count": 0, "posts": [], "error": "query is required"})
    k = int(args.get("k") or 5)
    k = max(1, min(k, 20))
    lookback_days = int(args.get("lookback_days") or 14)
    lookback_days = max(1, min(lookback_days, 90))
    min_reactions = int(args.get("min_reactions") or 50)
    exclude_creator = args.get("exclude_creator") or None
    try:
        rows = retrieve_recent_peer_bangers(
            query=query,
            lookback_days=lookback_days,
            min_reactions=min_reactions,
            k=k,
            exclude_creator=exclude_creator,
        )
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": str(exc)[:400]})
    return _json.dumps(
        {"count": len(rows), "lookback_days": lookback_days, "posts": rows},
        default=str,
    )


server.register(
    name="retrieve_recent_peer_bangers",
    description=(
        "Like retrieve_similar_posts, but filtered to posts from the last "
        "N days (default 14) and re-ranked by engagement_score within that "
        "window rather than by similarity. Use this to ground a draft in "
        "what's landing with the audience RIGHT NOW — hook patterns, "
        "topical framings, structural moves that worked this week. "
        "LinkedIn's feed has no memory at multi-year timescales, so "
        "all-time bangers are often poor current reference.\n\n"
        "Express any content-type filtering directly in the query "
        "(e.g. 'first-person narrative, NOT product promo').\n\n"
        "Args:\n"
        "  query (required): natural-language description of the kind of "
        "reference post you're looking for\n"
        "  lookback_days (optional, default 14, max 90): temporal window\n"
        "  min_reactions (optional, default 50): absolute engagement floor\n"
        "  k (optional, default 5, max 20): number of results\n"
        "  exclude_creator (optional): always pass the current client here"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query":              {"type": "string"},
            "lookback_days":      {"type": "integer"},
            "min_reactions":      {"type": "integer"},
            "k":                  {"type": "integer"},
            "exclude_creator":    {"type": "string"},
        },
        "required": ["query"],
    },
    handler=_handle_retrieve_recent_peer_bangers,
)


server.register(
    name="check_client_comfort",
    description=(
        "Policy / brand-safety / voice-fit gate (Aglaea). Evaluates whether "
        "the target FOC user would be comfortable publishing the draft AS-IS. "
        "Returns a structured verdict {pass|soften|rewrite, concerns[], "
        "suggested_edits?, summary}. Use this BEFORE submit_draft on any "
        "post that touches sensitive territory (first-person claims, numbers, "
        "political/legal ground, comparisons to named competitors). Cheap — "
        "single Claude call, no tool loop.\n\n"
        "Args:\n"
        "  draft_text (required): the draft content to evaluate"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "draft_text": {"type": "string"},
        },
        "required": ["draft_text"],
    },
    handler=_handle_check_client_comfort,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _COMPANY:
        print("Error: STELLE_COMPANY env var not set", file=sys.stderr)
        sys.exit(1)
    _init_observations()
    server.run()
