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
        trajectory.append({
            "draft_first_line": pr_draft.split("\n")[0][:80] if pr_draft else "",
            "draft_len": len(pr_draft),
            "reaction": pr_result.get("reaction", ""),
            "anchor": pr_result.get("anchor", ""),
        })

    # Persist THIS call so future invocations can see it as prior history.
    _dh = result.get("_draft_hash", "")
    _simulate_results.append({"draft_hash": _dh, "draft": draft, "result": result})

    # Compose the return: current reaction + trajectory so Stelle can
    # read both in one response.
    enriched = dict(result)
    if trajectory:
        enriched["_prior_reactions"] = trajectory
    return json.dumps(enriched, default=str)


server.register(
    name="get_reader_reaction",
    description=(
        "Send a draft to Irontomb, a rough-reader simulator. Returns "
        "{reaction, anchor} — a short visceral reader-voice reaction and "
        "a pointer to where in the post the reader reacted. Not a critique. "
        "Stelle interprets and revises."
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _COMPANY:
        print("Error: STELLE_COMPANY env var not set", file=sys.stderr)
        sys.exit(1)
    _init_observations()
    server.run()
