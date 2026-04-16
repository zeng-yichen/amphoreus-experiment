#!/usr/bin/env python3
"""MCP server exposing Stelle's custom tools for Claude CLI.

The Claude CLI already provides filesystem (Read/Write/Edit/Bash/Grep/Glob)
and web (WebSearch/WebFetch) tools natively. This server exposes only the
custom tools that don't map to built-in CLI capabilities:

  - query_observations — scored post history with engagement + reactors
  - query_top_engagers — aggregated top ICP engagers
  - search_linkedin_corpus — 200K+ LinkedIn post corpus
  - execute_python — sandboxed Python with pre-loaded observations
  - simulate_flame_chase_journey — Irontomb audience simulation
  - write_result — terminal tool with validation guards

Stateful within a run: tracks simulate call count and results for
the write_result guards (iteration discipline, would_stop_scrolling,
engagement gradient).

Launched as a subprocess by Claude CLI via --mcp-config.
Reads config from environment variables:
  STELLE_COMPANY — client slug
  STELLE_USE_CLI_IRONTOMB — "1" to run Irontomb through CLI too (default)
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

server.register(
    name="query_observations",
    description=(
        "Inspect this client's scored post history. Returns every scored "
        "post with draft text, published text (read side-by-side for client "
        "preferences), engagement metrics (reactions/comments/reposts/impressions), "
        "icp_match_rate, and per-post reactor list with ICP scores. "
        "Filters: min_reward, max_reward, limit, summary_only."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "min_reward": {"type": "number"},
            "max_reward": {"type": "number"},
            "limit": {"type": "integer"},
            "summary_only": {"type": "boolean"},
        },
    },
    handler=_handle_query_observations,
)


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

server.register(
    name="query_top_engagers",
    description=(
        "Get aggregated top engagers across all scored posts, ranked by "
        "ICP fit x engagement count. Returns name, headline, company, "
        "icp_score, engagement_count, posts_engaged."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max engagers (default 20, max 50)"},
        },
    },
    handler=_handle_query_top_engagers,
)


# ---------------------------------------------------------------------------
# Tool: search_linkedin_corpus
# ---------------------------------------------------------------------------
def _handle_search_corpus(args: dict) -> str:
    from backend.src.agents.analyst import _tool_search_linkedin_bank
    return _tool_search_linkedin_bank(args)

server.register(
    name="search_linkedin_corpus",
    description=(
        "Search 200K+ real LinkedIn posts. Modes: 'keyword' (exact text) "
        "or 'semantic' (meaning-based). Returns post text, engagement "
        "metrics, creator info."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["keyword", "semantic"]},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["query", "mode"],
    },
    handler=_handle_search_corpus,
)


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

server.register(
    name="execute_python",
    description=(
        "Run Python with pre-loaded obs (scored observations), numpy, "
        "scipy, sklearn, pandas. Use print() for output. 60s timeout."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    },
    handler=_handle_execute_python,
)


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

server.register(
    name="simulate_flame_chase_journey",
    description=(
        "MANDATORY before submitting any post. Send a draft through "
        "Irontomb, the adversarial audience simulator. Returns "
        "engagement_prediction, impression_prediction, would_stop_scrolling, "
        "and a _gradient block showing predicted vs client median engagement. "
        "Irontomb is your loss function — minimize the gap between predicted "
        "and median engagement through revision."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "draft_text": {"type": "string", "description": "The full draft post text"},
        },
        "required": ["draft_text"],
    },
    handler=_handle_simulate,
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

    # Guard 1: iteration discipline
    n_posts = len(parsed.get("posts", []))
    min_required = max(3 * n_posts, 3)
    if _simulate_call_count < min_required:
        return json.dumps({
            "_error": (
                f"Iteration discipline: {_simulate_call_count} simulate calls "
                f"but need {min_required} (3 per post). Go back and iterate."
            ),
        })

    # Guard 2: would_stop_scrolling
    failed_posts = []
    for pi, post in enumerate(parsed.get("posts", []), 1):
        pt = (post.get("text") or "").strip()
        if not pt:
            continue
        ph = hashlib.sha256(pt.encode("utf-8")).hexdigest()[:16]
        last = None
        for sr in reversed(_simulate_results):
            if sr["draft_hash"] == ph:
                last = sr["result"]
                break
        if last and last.get("would_stop_scrolling") is False:
            iv = (last.get("inner_voice") or "")[:120]
            failed_posts.append(f"Post {pi}: would_stop_scrolling=False ({iv})")

    if failed_posts:
        return json.dumps({
            "_error": "Scroll-stop check failed",
            "failed_posts": failed_posts,
            "instruction": "Revise hooks and re-simulate until would_stop_scrolling is True.",
        })

    # Stamp Irontomb predictions onto each post
    for post in parsed.get("posts", []):
        pt = (post.get("text") or "").strip()
        if not pt:
            continue
        ph = hashlib.sha256(pt.encode("utf-8")).hexdigest()[:16]
        for sr in reversed(_simulate_results):
            if sr["draft_hash"] == ph:
                r = sr["result"]
                post["predicted_engagement"] = r.get("engagement_prediction")
                post["predicted_impressions"] = r.get("impression_prediction")
                break

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
        "structure, enforces iteration discipline (min 3 simulate calls "
        "per post), and checks would_stop_scrolling. Pass the full output "
        "as a JSON string in result_json."
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
