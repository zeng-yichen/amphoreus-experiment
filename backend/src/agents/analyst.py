"""Analyst — convergence-objective engagement predictor via tool-use agent.

The analyst's job is to build a working model that predicts engagement for
a specific client, or prove it can't yet. Not to report findings. Not to
explore patterns. To build something that works.

It has 11 statistical tools spanning three data scales (client, cross-client,
LinkedIn-wide) and decides what to run and in what order. No predetermined
analysis sequence. The prompt defines the goal; the model figures out how.

Each weekly run improves on the previous run's model. The convergence isn't
just within a single run — it's across runs over weeks as observation count
grows. The full finding and model history is preserved so the agent can
assess its own progress.

Usage:
    from backend.src.agents.analyst import run_analysis

    result = run_analysis("innovocommerce")
    # → model + findings stored in memory/{company}/analyst_findings.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_ANALYST_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 16384
_MAX_TURNS = 80  # generous ceiling — the prompt says "as many as needed"


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
# Engagement Predictor

Your task is to build a model that predicts engagement scores for this \
client's posts. You have tools to test features, fit regressions, and \
validate predictions. You are not done until your model achieves \
LOO R² > 0.3 on the client's historical data, or you have exhausted all \
hypotheses and can explain why prediction at that accuracy is not yet \
possible with the available data. If your first approach doesn't work, \
try a different one. You may run as many tool calls as needed.

## The Standard

You hand the model to the client's ghostwriter. The ghostwriter uses it \
to decide which of 5 candidate post ideas to write first. If the model \
ranks them in an order that matches actual engagement outcomes, you \
succeeded. If the rankings are no better than random, you failed.

**Don't come back until you can predict engagement accurately.** If you \
can't reach LOO R² > 0.3, come back with:
- The best model you could build and its honest validation metrics
- Every approach you tried and why each failed
- What specific data the client would need to collect to make prediction \
  possible (be concrete: "label each post's opening hook style" not \
  "collect more data")

## Tools

You have 11 tools spanning client data (n=30-50), cross-client data \
(22+ clients), and LinkedIn-wide data (200K+ posts). Use whatever you \
need. Call tools in whatever order makes sense. There is no predetermined \
analysis sequence.

You'll see how many turns you have left in each tool result. Budget your \
time — if you have 10 turns left, stop exploring and validate your best \
model.

## Constraints

- **Validate on held-out data.** In-sample R² proves nothing at small n. \
  LOO R² is the metric that matters. If your model fits training data \
  beautifully but LOO R² is negative, it's worthless.
- **Check confounders.** A correlation might be driven by something else. \
  Use `compute_partial_correlation` before trusting a feature.
- **Be honest about sample size.** If a subgroup has n<10, say so.
- **Don't report noise as signal.** If you test 15 hypotheses, expect \
  1-2 to look meaningful by chance.

## Self-assessment

If the user message shows a prior run's model, your job is to beat it. \
If the prior model achieved LOO R² = 0.06, try to beat 0.06. If you \
can't improve it, explain what you tried. Models that improve across \
weekly runs demonstrate real learning.

## Output

Use `store_finding` for two things:

1. **type="finding"**: discoveries that explain what drives engagement. \
   Store as you go — these are the "why" behind the model.

2. **type="model"**: your best validated model. Store LAST. Include: \
   features used, approach, coefficients, LOO R², predicted vs actual \
   rankings, and whether the model is ready to score candidate posts.
"""


# ------------------------------------------------------------------
# Tool definitions (Claude tool-use schema)
# ------------------------------------------------------------------

_TOOLS = [
    {
        "name": "query_observations",
        "description": (
            "Query the client's scored observation history. Returns a JSON list "
            "of observations matching the filters. Each observation has: "
            "post_hash, topic_tag, format_tag, source_segment_type, reward "
            "(with immediate, depth, reach, eng_rate, raw_metrics), posted_at, "
            "char_count (derived from post body), edit_similarity, "
            "cyrene_composite, cyrene_iterations, and post body text. "
            "Use this to explore the data before forming hypotheses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic_filter": {
                    "type": "string",
                    "description": "If set, only return observations with this topic_tag (exact match).",
                },
                "format_filter": {
                    "type": "string",
                    "description": "If set, only return observations with this format_tag (exact match).",
                },
                "min_reward": {
                    "type": "number",
                    "description": "If set, only return observations with reward.immediate >= this value.",
                },
                "max_reward": {
                    "type": "number",
                    "description": "If set, only return observations with reward.immediate <= this value.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max observations to return (default 50). Set lower for summaries, higher for full analysis.",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort field: 'reward' (default), 'posted_at', 'char_count'.",
                },
                "sort_order": {
                    "type": "string",
                    "description": "'desc' (default) or 'asc'.",
                },
                "summary_only": {
                    "type": "boolean",
                    "description": "If true, return aggregate stats instead of individual observations: count, mean/std reward, topic distribution, format distribution, date range.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compute_correlation",
        "description": (
            "Compute Spearman rank correlation between two numeric sequences. "
            "Returns the correlation coefficient (-1 to +1) and the number of "
            "data points. Use for continuous-vs-continuous relationships "
            "(e.g., char_count vs reward, posting_hour vs reward)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x_field": {
                    "type": "string",
                    "description": "Field name for x values. Can be: 'char_count', 'edit_similarity', 'posting_hour', 'posting_day', 'cyrene_composite', 'cyrene_iterations', or 'reward'.",
                },
                "y_field": {
                    "type": "string",
                    "description": "Field name for y values. Same options as x_field.",
                },
                "topic_filter": {
                    "type": "string",
                    "description": "Optional: restrict to observations with this topic_tag.",
                },
                "format_filter": {
                    "type": "string",
                    "description": "Optional: restrict to observations with this format_tag.",
                },
            },
            "required": ["x_field", "y_field"],
        },
    },
    {
        "name": "compute_effect_size",
        "description": (
            "Compare mean engagement reward between two groups defined by a "
            "categorical split. Returns Cohen's d, mean of each group, and "
            "sample sizes. Use for categorical-vs-engagement comparisons "
            "(e.g., 'hot take' vs 'storytelling' format, or 'topic A' vs 'topic B')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_field": {
                    "type": "string",
                    "description": "Categorical field to split on: 'topic_tag', 'format_tag', 'source_segment_type', or 'posting_day'.",
                },
                "group_a_value": {
                    "type": "string",
                    "description": "Value for group A (e.g., 'hot take').",
                },
                "group_b_value": {
                    "type": "string",
                    "description": "Value for group B (e.g., 'storytelling'). If omitted, group B = all observations NOT in group A.",
                },
            },
            "required": ["group_field", "group_a_value"],
        },
    },
    {
        "name": "compute_partial_correlation",
        "description": (
            "Compute partial correlation between a target variable and reward, "
            "controlling for a list of confounders. Returns the partial correlation, "
            "the marginal (uncontrolled) correlation, and sample size. "
            "Use this to check whether an apparent relationship is real or "
            "explained by other variables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_field": {
                    "type": "string",
                    "description": "The field to test: 'char_count', 'posting_hour', 'posting_day', 'edit_similarity', 'topic_tag', 'format_tag'.",
                },
                "control_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to control for. Same options as target_field.",
                },
            },
            "required": ["target_field", "control_fields"],
        },
    },
    {
        "name": "fit_regression",
        "description": (
            "Fit a ridge regression predicting engagement reward from a set of "
            "features. Returns R², LOO R², and per-feature coefficients. "
            "Use for multivariate prediction: which features jointly predict "
            "engagement after accounting for each other?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature fields to include. Options: 'char_count', 'posting_hour', 'posting_day', 'edit_similarity', 'topic_tag', 'format_tag'.",
                },
                "ridge_alpha": {
                    "type": "number",
                    "description": "Ridge regularization strength (default 1.0). Higher = more regularization.",
                },
            },
            "required": ["feature_fields"],
        },
    },
    {
        "name": "build_transition_matrix",
        "description": (
            "Build a first-order Markov transition matrix from a sequence of "
            "categorical values (e.g., topic_tag or format_tag over time). "
            "Returns P(next | previous) with counts and mean reward per transition. "
            "Use to discover sequence effects: does topic A perform better "
            "after topic B than after topic C?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence_field": {
                    "type": "string",
                    "description": "Categorical field to build transitions on: 'topic_tag' or 'format_tag'.",
                },
            },
            "required": ["sequence_field"],
        },
    },
    {
        "name": "embed_and_compare",
        "description": (
            "Embed two texts and compute cosine similarity. Use to test "
            "semantic hypotheses: are high-performing posts more similar to "
            "each other than to low-performing posts? Does a specific opening "
            "style cluster with success?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_a": {
                    "type": "string",
                    "description": "First text to embed (can be a post body excerpt or a description).",
                },
                "text_b": {
                    "type": "string",
                    "description": "Second text to embed.",
                },
            },
            "required": ["text_a", "text_b"],
        },
    },
    {
        "name": "store_finding",
        "description": (
            "Store an analytical finding OR a predictive model for this client. "
            "Two types:\n\n"
            "type='finding': An analytical discovery that explains what drives "
            "engagement. Store as you discover them.\n\n"
            "type='model': Your best predictive model specification with "
            "validation metrics. Store LAST, after validation. Include: what "
            "features, what approach, LOO R² or Spearman, predicted vs actual "
            "rankings, and whether the model is ready to score candidate posts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "'finding' (analytical discovery) or 'model' (predictive model specification).",
                },
                "claim": {
                    "type": "string",
                    "description": "For findings: a clear, actionable claim. For models: a summary of the model (e.g., 'Ridge regression on format_tag + char_count achieves Spearman 0.25 on LOO predictions').",
                },
                "evidence": {
                    "type": "string",
                    "description": "For findings: statistical evidence. For models: validation metrics (LOO R², Spearman, n) and predicted vs actual post rankings.",
                },
                "confidence": {
                    "type": "string",
                    "description": "'strong' / 'suggestive' / 'weak' / 'insufficient_data'.",
                },
                "model_spec": {
                    "type": "object",
                    "description": "For type='model' only. The model specification: features used, approach (regression/heuristic/ensemble), coefficients if applicable, ridge_alpha, any other parameters needed to reproduce the model.",
                },
                "model_ready": {
                    "type": "boolean",
                    "description": "For type='model' only. Can this model be used to score candidate posts? true if validation metrics meet minimum quality, false otherwise.",
                },
                "model_readiness_explanation": {
                    "type": "string",
                    "description": "For type='model' only. Why the model is or isn't ready. What data would improve it.",
                },
            },
            "required": ["claim", "evidence", "confidence"],
        },
    },
    {
        "name": "search_linkedin_posts",
        "description": (
            "Search a database of 200K+ real LinkedIn posts by keyword. Returns "
            "posts ranked by engagement score with metrics (reactions, comments, "
            "reposts, engagement_score). Use this to benchmark the client's "
            "performance against the broader LinkedIn ecosystem, discover what "
            "topics/formats/hooks perform well industry-wide, and find patterns "
            "across thousands of posts that the client's 30-50 observations "
            "can't reveal. This is your most statistically powerful tool — "
            "use it to validate hypotheses from the client data against "
            "large-sample LinkedIn-wide evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search query. Use specific domain terms relevant to the client's industry (e.g., 'clinical trial protocol', 'ecommerce CAC', 'remote team culture'). More specific = better results.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max posts to return (default 15, max 30).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_linkedin_semantic",
        "description": (
            "Semantic search over 200K+ LinkedIn posts by meaning, not keywords. "
            "Finds conceptually similar posts even when different words are used. "
            "Use this for abstract queries: 'vulnerability-based leadership hooks', "
            "'contrarian takes on industry trends', 'data-driven storytelling'. "
            "Returns posts ranked by semantic similarity with engagement metrics. "
            "Combine with search_linkedin_posts for comprehensive discovery."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "A natural-language description of the type of content to find. Be descriptive — 'posts that open with a specific failure story and pivot to a framework' works better than 'failure stories'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max posts to return (default 10, max 20).",
                },
            },
            "required": ["concept"],
        },
    },
    {
        "name": "query_cross_client_data",
        "description": (
            "Access aggregated learning data from all 22+ Amphoreus clients. "
            "Three data sources:\n"
            "  - 'patterns': universal engagement patterns that hold across 3+ clients "
            "(e.g., 'opening with concrete numbers predicts +0.42 reward lift')\n"
            "  - 'hooks': top-performing hooks across all clients with engagement scores "
            "and style classifications (number_led, personal_story, contrarian, etc.)\n"
            "  - 'principles': learned quality principles discovered from top vs bottom "
            "performers across the full client portfolio\n\n"
            "Use this to compare the current client's patterns against what works "
            "across the portfolio. Findings from 22 clients × 30+ posts each "
            "are far more statistically robust than any single client's data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_source": {
                    "type": "string",
                    "description": "'patterns' (universal engagement patterns), 'hooks' (top-performing hooks with style tags), or 'principles' (quality principles). Pick one.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max items to return (default 15).",
                },
                "hook_style_filter": {
                    "type": "string",
                    "description": "For 'hooks' source only: filter by hook style. Options: 'number_led', 'personal_story', 'question', 'contrarian', 'icp_callout', 'story_climax', 'declarative'.",
                },
            },
            "required": ["data_source"],
        },
    },
]


# ------------------------------------------------------------------
# Tool dispatch — thin wrappers around existing statistical primitives
# ------------------------------------------------------------------

def _dispatch_tool(tool_name: str, tool_input: dict, company: str,
                   observations: list[dict], run_id: str = "") -> str:
    """Route a tool call to the appropriate handler. Returns JSON string."""
    try:
        if tool_name == "query_observations":
            return _tool_query_observations(tool_input, observations)
        elif tool_name == "compute_correlation":
            return _tool_compute_correlation(tool_input, observations)
        elif tool_name == "compute_effect_size":
            return _tool_compute_effect_size(tool_input, observations)
        elif tool_name == "compute_partial_correlation":
            return _tool_compute_partial_correlation(tool_input, observations)
        elif tool_name == "fit_regression":
            return _tool_fit_regression(tool_input, observations)
        elif tool_name == "build_transition_matrix":
            return _tool_build_transition_matrix(tool_input, observations)
        elif tool_name == "embed_and_compare":
            return _tool_embed_and_compare(tool_input)
        elif tool_name == "store_finding":
            return _tool_store_finding(tool_input, company, run_id=run_id)
        elif tool_name == "search_linkedin_posts":
            return _tool_search_linkedin_posts(tool_input)
        elif tool_name == "search_linkedin_semantic":
            return _tool_search_linkedin_semantic(tool_input)
        elif tool_name == "query_cross_client_data":
            return _tool_query_cross_client_data(tool_input)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        return json.dumps({"error": f"{tool_name} failed: {str(e)[:300]}"})


def _extract_numeric_field(obs: dict, field: str) -> Optional[float]:
    """Extract a numeric value from an observation by field name."""
    if field == "reward":
        return obs.get("reward", {}).get("immediate")
    elif field == "char_count":
        body = obs.get("posted_body") or obs.get("post_body") or ""
        return float(len(body)) if body else None
    elif field == "posting_hour":
        ts = obs.get("posted_at") or ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return float(dt.hour)
            except Exception:
                pass
        return None
    elif field == "posting_day":
        ts = obs.get("posted_at") or ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return float(dt.weekday())
            except Exception:
                pass
        return None
    elif field == "edit_similarity":
        v = obs.get("edit_similarity", -1)
        return float(v) if v >= 0 else None
    elif field == "cyrene_composite":
        v = obs.get("cyrene_composite")
        return float(v) if v is not None else None
    elif field == "cyrene_iterations":
        v = obs.get("cyrene_iterations")
        return float(v) if v is not None else None
    else:
        # Try direct access for any field
        v = obs.get(field)
        if isinstance(v, (int, float)):
            return float(v)
        return None


def _filter_observations(observations: list[dict], tool_input: dict) -> list[dict]:
    """Apply standard filters from tool_input to observations."""
    filtered = list(observations)
    topic = tool_input.get("topic_filter")
    if topic:
        filtered = [o for o in filtered if o.get("topic_tag") == topic]
    fmt = tool_input.get("format_filter")
    if fmt:
        filtered = [o for o in filtered if o.get("format_tag") == fmt]
    min_r = tool_input.get("min_reward")
    if min_r is not None:
        filtered = [o for o in filtered if (o.get("reward", {}).get("immediate", 0)) >= min_r]
    max_r = tool_input.get("max_reward")
    if max_r is not None:
        filtered = [o for o in filtered if (o.get("reward", {}).get("immediate", 0)) <= max_r]
    return filtered


def _tool_query_observations(tool_input: dict, observations: list[dict]) -> str:
    filtered = _filter_observations(observations, tool_input)
    limit = tool_input.get("limit", 50)
    sort_by = tool_input.get("sort_by", "reward")
    sort_order = tool_input.get("sort_order", "desc")
    reverse = sort_order == "desc"

    if tool_input.get("summary_only"):
        rewards = [o.get("reward", {}).get("immediate", 0) for o in filtered]
        from collections import Counter
        topics = Counter(o.get("topic_tag", "?") for o in filtered)
        formats = Counter(o.get("format_tag", "?") for o in filtered)
        dates = [o.get("posted_at", "") for o in filtered if o.get("posted_at")]
        import math
        mean_r = sum(rewards) / len(rewards) if rewards else 0
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)) if len(rewards) > 1 else 0
        return json.dumps({
            "count": len(filtered),
            "reward_mean": round(mean_r, 4),
            "reward_std": round(std_r, 4),
            "reward_min": round(min(rewards), 4) if rewards else None,
            "reward_max": round(max(rewards), 4) if rewards else None,
            "topic_distribution": dict(topics.most_common()),
            "format_distribution": dict(formats.most_common()),
            "date_range": {"earliest": min(dates) if dates else None,
                           "latest": max(dates) if dates else None},
        }, indent=2)

    # Sort
    if sort_by == "reward":
        filtered.sort(key=lambda o: o.get("reward", {}).get("immediate", 0), reverse=reverse)
    elif sort_by == "posted_at":
        filtered.sort(key=lambda o: o.get("posted_at", ""), reverse=reverse)
    elif sort_by == "char_count":
        filtered.sort(key=lambda o: len(o.get("posted_body", o.get("post_body", ""))), reverse=reverse)

    # Compact representation for the model
    results = []
    for o in filtered[:limit]:
        body = (o.get("posted_body") or o.get("post_body") or "")
        r = o.get("reward", {})
        results.append({
            "topic_tag": o.get("topic_tag"),
            "format_tag": o.get("format_tag"),
            "source_segment_type": o.get("source_segment_type"),
            "reward": r.get("immediate"),
            "impressions": r.get("raw_metrics", {}).get("impressions"),
            "comments": r.get("raw_metrics", {}).get("comments"),
            "reactions": r.get("raw_metrics", {}).get("reactions"),
            "char_count": len(body),
            "edit_similarity": o.get("edit_similarity", -1),
            "posted_at": o.get("posted_at", "")[:10],  # date only
            "opening": body[:200] if body else "",
        })
    return json.dumps(results, indent=2)


def _tool_compute_correlation(tool_input: dict, observations: list[dict]) -> str:
    filtered = _filter_observations(observations, tool_input)
    x_field = tool_input["x_field"]
    y_field = tool_input["y_field"]

    pairs = []
    for o in filtered:
        x = _extract_numeric_field(o, x_field)
        y = _extract_numeric_field(o, y_field)
        if x is not None and y is not None:
            pairs.append((x, y))

    if len(pairs) < 3:
        return json.dumps({"error": f"Insufficient data: {len(pairs)} pairs (need ≥3)",
                           "x_field": x_field, "y_field": y_field})

    from backend.src.utils.correlation_analyzer import _spearman_correlation
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    corr = _spearman_correlation(xs, ys)
    return json.dumps({
        "spearman_correlation": round(corr, 4),
        "n": len(pairs),
        "x_field": x_field,
        "y_field": y_field,
    })


def _tool_compute_effect_size(tool_input: dict, observations: list[dict]) -> str:
    field = tool_input["group_field"]
    val_a = tool_input["group_a_value"]
    val_b = tool_input.get("group_b_value")

    group_a = [o.get("reward", {}).get("immediate", 0) for o in observations if o.get(field) == val_a]
    if val_b:
        group_b = [o.get("reward", {}).get("immediate", 0) for o in observations if o.get(field) == val_b]
    else:
        group_b = [o.get("reward", {}).get("immediate", 0) for o in observations if o.get(field) != val_a]

    if len(group_a) < 2 or len(group_b) < 2:
        return json.dumps({"error": f"Insufficient data: group_a n={len(group_a)}, group_b n={len(group_b)}"})

    import math
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / max(len(group_a) - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / max(len(group_b) - 1, 1)
    pooled_sd = math.sqrt(
        ((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b)
        / max(len(group_a) + len(group_b) - 2, 1)
    )
    d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

    return json.dumps({
        "cohens_d": round(d, 4),
        "group_a": {"value": val_a, "n": len(group_a), "mean_reward": round(mean_a, 4)},
        "group_b": {"value": val_b or f"NOT {val_a}", "n": len(group_b), "mean_reward": round(mean_b, 4)},
    })


def _tool_compute_partial_correlation(tool_input: dict, observations: list[dict]) -> str:
    target = tool_input["target_field"]
    controls = tool_input.get("control_fields", [])

    from backend.src.utils.causal_filter import (
        _build_feature_matrix, _partial_correlation, _spearman,
    )

    all_fields = [target] + [c for c in controls if c != target]
    feature_names, X, y = _build_feature_matrix(observations)

    if not feature_names or target not in feature_names:
        return json.dumps({"error": f"Field '{target}' not found in data. Available: {feature_names}"})

    target_idx = feature_names.index(target)

    # Marginal
    marginal = _spearman([row[target_idx] for row in X], y)
    # Partial
    partial = _partial_correlation(X, y, target_idx)

    return json.dumps({
        "target": target,
        "controls": controls,
        "marginal_correlation": round(marginal, 4),
        "partial_correlation": round(partial, 4),
        "n": len(y),
    })


def _tool_fit_regression(tool_input: dict, observations: list[dict]) -> str:
    features = tool_input.get("feature_fields", [])
    alpha = tool_input.get("ridge_alpha", 1.0)

    from backend.src.utils.causal_filter import _build_feature_matrix

    feature_names, X, y = _build_feature_matrix(observations)
    if not feature_names:
        return json.dumps({"error": "No features could be extracted"})

    # Filter to requested features
    keep_idx = [i for i, f in enumerate(feature_names) if f in features]
    if not keep_idx:
        return json.dumps({"error": f"Requested features not found. Available: {feature_names}"})

    X_filtered = [[row[i] for i in keep_idx] for row in X]
    used_names = [feature_names[i] for i in keep_idx]

    from backend.src.utils.engagement_predictor import _ridge_fit, _leave_one_out_residuals, _normalize_features

    X_norm, means, stds = _normalize_features(X_filtered)
    X_with_intercept = [row + [1.0] for row in X_norm]
    coefficients = _ridge_fit(X_with_intercept, y, lam=alpha)

    # LOO R²
    loo_resids = _leave_one_out_residuals(X_norm, y, lam=alpha)
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum(r ** 2 for r in loo_resids)
    loo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # In-sample R²
    preds = [sum(c * x for c, x in zip(coefficients, row)) for row in X_with_intercept]
    ss_res_train = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2_train = 1.0 - ss_res_train / ss_tot if ss_tot > 0 else 0.0

    feature_coefficients = {
        name: round(coefficients[i], 4)
        for i, name in enumerate(used_names)
    }

    return json.dumps({
        "features": used_names,
        "coefficients": feature_coefficients,
        "intercept": round(coefficients[-1], 4),
        "r_squared_train": round(r2_train, 4),
        "r_squared_loo": round(loo_r2, 4),
        "n": len(y),
        "ridge_alpha": alpha,
    })


def _tool_build_transition_matrix(tool_input: dict, observations: list[dict]) -> str:
    field = tool_input["sequence_field"]

    # Sort chronologically
    sorted_obs = sorted(
        [o for o in observations if o.get(field) and o.get("posted_at")],
        key=lambda o: o.get("posted_at", ""),
    )

    from collections import defaultdict
    transitions = defaultdict(lambda: defaultdict(lambda: {"count": 0, "rewards": []}))

    for i in range(1, len(sorted_obs)):
        prev = sorted_obs[i - 1].get(field)
        nxt = sorted_obs[i].get(field)
        if prev and nxt:
            reward = sorted_obs[i].get("reward", {}).get("immediate", 0)
            transitions[prev][nxt]["count"] += 1
            transitions[prev][nxt]["rewards"].append(reward)

    matrix = {}
    for prev, nexts in transitions.items():
        total = sum(v["count"] for v in nexts.values())
        matrix[prev] = {}
        for nxt, data in nexts.items():
            rs = data["rewards"]
            matrix[prev][nxt] = {
                "count": data["count"],
                "probability": round(data["count"] / total, 3),
                "mean_reward": round(sum(rs) / len(rs), 4),
            }

    return json.dumps({
        "field": field,
        "n_observations": len(sorted_obs),
        "n_unique_values": len(set(o.get(field) for o in sorted_obs if o.get(field))),
        "transitions": matrix,
    }, indent=2)


def _tool_embed_and_compare(tool_input: dict) -> str:
    text_a = tool_input["text_a"]
    text_b = tool_input["text_b"]

    from backend.src.agents.lola import _embed_texts, _cosine_similarity
    embs = _embed_texts([text_a[:8000], text_b[:8000]])
    if len(embs) != 2:
        return json.dumps({"error": "Embedding failed"})
    sim = _cosine_similarity(embs[0], embs[1])
    return json.dumps({
        "cosine_similarity": round(sim, 4),
        "text_a_preview": text_a[:100],
        "text_b_preview": text_b[:100],
    })


def _tool_store_finding(tool_input: dict, company: str, run_id: str = "") -> str:
    findings_path = vortex.memory_dir(company) / "analyst_findings.json"
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        existing = json.loads(findings_path.read_text(encoding="utf-8")) if findings_path.exists() else {"findings": [], "runs": []}
    except Exception:
        existing = {"findings": [], "runs": []}

    entry_type = tool_input.get("type", "finding")

    finding = {
        "type": entry_type,
        "claim": tool_input["claim"],
        "evidence": tool_input["evidence"],
        "confidence": tool_input.get("confidence", "suggestive"),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Model-specific fields
    if entry_type == "model":
        finding["model_spec"] = tool_input.get("model_spec", {})
        finding["model_ready"] = tool_input.get("model_ready", False)
        finding["model_readiness_explanation"] = tool_input.get("model_readiness_explanation", "")

    existing["findings"].append(finding)

    tmp = findings_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(findings_path)

    label = "model" if entry_type == "model" else f"finding #{sum(1 for f in existing['findings'] if f.get('type') != 'model')}"
    return json.dumps({"status": "stored", "type": entry_type, "label": label,
                        "total_entries": len(existing["findings"])})


# ------------------------------------------------------------------
# LinkedIn-wide and cross-client tools
# ------------------------------------------------------------------

def _tool_search_linkedin_posts(tool_input: dict) -> str:
    """Keyword search over 200K+ LinkedIn posts in Supabase."""
    query = tool_input.get("query", "")
    limit = min(tool_input.get("limit", 15), 30)
    if not query:
        return json.dumps({"error": "query is required"})

    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")
    if not sb_url or not sb_key:
        return json.dumps({"error": "SUPABASE_URL / SUPABASE_KEY not configured"})

    try:
        import httpx
        # Use the longest keyword for the primary ilike filter, then post-filter
        keywords = [w.strip() for w in query.split() if len(w.strip()) >= 3]
        if not keywords:
            return json.dumps({"error": "Query too short — use 3+ character words"})
        keywords.sort(key=len, reverse=True)
        primary = keywords[0]

        resp = httpx.get(
            f"{sb_url}/rest/v1/linkedin_posts",
            params={
                "select": "hook,post_text,posted_at,creator_username,"
                          "total_reactions,total_comments,total_reposts,"
                          "engagement_score,is_outlier",
                "post_text": f"ilike.*{primary}*",
                "is_company_post": "eq.false",
                "order": "engagement_score.desc",
                "limit": str(limit * 5),  # overfetch for keyword filtering
            },
            headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        rows = resp.json()

        # Post-filter: all keywords must match
        results = []
        for row in rows:
            text = (row.get("post_text") or "").lower()
            if all(kw.lower() in text for kw in keywords):
                results.append(row)
            if len(results) >= limit:
                break

        if not results:
            return json.dumps({"query": query, "results": [], "count": 0,
                               "note": "No posts matched all keywords"})

        # Compact format for the model
        compact = []
        for r in results:
            eng = r.get("engagement_score") or 0
            compact.append({
                "creator": r.get("creator_username", "?"),
                "date": (r.get("posted_at") or "")[:10],
                "reactions": r.get("total_reactions", 0),
                "comments": r.get("total_comments", 0),
                "reposts": r.get("total_reposts", 0),
                "engagement": round(eng / 100, 2) if eng else 0,
                "is_outlier": r.get("is_outlier", False),
                "hook": (r.get("hook") or "")[:200],
                "text": (r.get("post_text") or "")[:800],
            })
        return json.dumps({"query": query, "count": len(compact), "results": compact}, indent=2)

    except Exception as e:
        return json.dumps({"error": f"LinkedIn search failed: {str(e)[:200]}"})


def _tool_search_linkedin_semantic(tool_input: dict) -> str:
    """Semantic search over LinkedIn posts via Pinecone vector index."""
    concept = tool_input.get("concept", "")
    limit = min(tool_input.get("limit", 10), 20)
    if not concept:
        return json.dumps({"error": "concept is required"})

    pc_key = os.environ.get("PINECONE_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")

    if not pc_key or not oai_key:
        return json.dumps({"error": "PINECONE_API_KEY / OPENAI_API_KEY not configured"})

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        # Embed the query
        oai = OpenAI()
        pc = Pinecone(api_key=pc_key)
        idx = pc.Index("linkedin-posts")
        stats = idx.describe_index_stats()
        dims = stats.get("dimension", 1536)

        resp = oai.embeddings.create(input=[concept], model="text-embedding-3-small",
                                      dimensions=dims)
        vec = resp.data[0].embedding

        # Query Pinecone
        results = idx.query(vector=vec, top_k=limit, namespace="v2",
                            include_metadata=True)
        matches = results.get("matches", [])
        if not matches:
            return json.dumps({"concept": concept, "results": [], "count": 0})

        # Hydrate from Supabase if possible
        urns = [m["id"] for m in matches]
        posts_by_urn = {}
        if sb_url and sb_key:
            import httpx
            urn_filter = ",".join(f'"{u}"' for u in urns[:limit])
            try:
                r = httpx.get(
                    f"{sb_url}/rest/v1/linkedin_posts",
                    params={
                        "select": "provider_urn,hook,post_text,posted_at,creator_username,"
                                  "total_reactions,total_comments,total_reposts,"
                                  "engagement_score,is_outlier",
                        "provider_urn": f"in.({urn_filter})",
                    },
                    headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}"},
                    timeout=30.0,
                )
                r.raise_for_status()
                for row in r.json():
                    if row.get("provider_urn"):
                        posts_by_urn[row["provider_urn"]] = row
            except Exception:
                pass

        compact = []
        for m in matches:
            row = posts_by_urn.get(m["id"])
            sim = m.get("score", 0)
            if row:
                eng = row.get("engagement_score") or 0
                compact.append({
                    "similarity": round(sim, 3),
                    "creator": row.get("creator_username", "?"),
                    "date": (row.get("posted_at") or "")[:10],
                    "reactions": row.get("total_reactions", 0),
                    "comments": row.get("total_comments", 0),
                    "engagement": round(eng / 100, 2) if eng else 0,
                    "hook": (row.get("hook") or "")[:200],
                    "text": (row.get("post_text") or "")[:600],
                })
            else:
                meta = m.get("metadata", {})
                compact.append({
                    "similarity": round(sim, 3),
                    "text": str(meta.get("post_text", meta.get("text", "")))[:600],
                })
        return json.dumps({"concept": concept, "count": len(compact), "results": compact}, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Semantic search failed: {str(e)[:200]}"})


def _tool_query_cross_client_data(tool_input: dict) -> str:
    """Access aggregated learning data from all Amphoreus clients."""
    source = tool_input.get("data_source", "")
    limit = min(tool_input.get("limit", 15), 50)

    our_memory = vortex.our_memory_dir()

    if source == "patterns":
        path = our_memory / "universal_patterns.json"
        if not path.exists():
            return json.dumps({"error": "No universal patterns file found"})
        try:
            patterns = json.loads(path.read_text(encoding="utf-8"))
            # Sort by confidence × evidence
            patterns.sort(
                key=lambda p: p.get("confidence", 0) * p.get("evidence_clients", 1),
                reverse=True,
            )
            compact = []
            for p in patterns[:limit]:
                compact.append({
                    "pattern": p.get("pattern", ""),
                    "confidence": p.get("confidence", 0),
                    "evidence_clients": p.get("evidence_clients", 0),
                    "avg_reward_lift": p.get("avg_reward_lift", 0),
                    "category": p.get("category", ""),
                })
            return json.dumps({"source": "patterns", "count": len(compact),
                               "data": compact}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to load patterns: {e}"})

    elif source == "hooks":
        path = our_memory / "hook_library.json"
        if not path.exists():
            return json.dumps({"error": "No hook library found"})
        try:
            hooks = json.loads(path.read_text(encoding="utf-8"))
            style_filter = tool_input.get("hook_style_filter")
            if style_filter:
                hooks = [h for h in hooks if h.get("hook_style") == style_filter]
            compact = []
            for h in hooks[:limit]:
                compact.append({
                    "hook": h.get("hook", ""),
                    "hook_style": h.get("hook_style", ""),
                    "engagement_score": h.get("engagement_score", 0),
                    "impressions": h.get("impressions", 0),
                    "char_count": h.get("char_count", 0),
                })
            return json.dumps({"source": "hooks", "count": len(compact),
                               "style_filter": style_filter, "data": compact}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to load hooks: {e}"})

    elif source == "principles":
        path = our_memory / "learned_principles.json"
        if not path.exists():
            return json.dumps({"error": "No learned principles found"})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            principles = data.get("principles", [])
            compact = []
            for p in principles[:limit]:
                compact.append({
                    "id": p.get("id", ""),
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "evidence_count": p.get("evidence_count", 0),
                })
            return json.dumps({
                "source": "principles",
                "count": len(compact),
                "source_observations": data.get("source_observations", 0),
                "source_clients": data.get("source_clients", 0),
                "data": compact,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to load principles: {e}"})

    else:
        return json.dumps({"error": f"Unknown data_source: {source}. Use 'patterns', 'hooks', or 'principles'."})


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

def run_analysis(company: str, verbose: bool = False) -> dict:
    """Run the analyst agent for a client.

    Loads the client's scored observations, hands them to the agent with
    the full toolkit, and lets it form and test hypotheses. Findings are
    stored in ``memory/{company}/analyst_findings.json``.

    Returns a summary dict with tool call count, findings count, and cost.
    """
    import anthropic

    # Load observations
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None
    if state is None:
        return {"error": f"No RuanMei state for {company}"}

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("reward", {}).get("immediate") is not None
    ]
    if len(scored) < 10:
        return {"error": f"Insufficient data: {len(scored)} scored observations (need ≥10)"}

    # Build the user message with a data summary to orient the model
    from collections import Counter
    topics = Counter(o.get("topic_tag", "?") for o in scored)
    formats = Counter(o.get("format_tag", "?") for o in scored)
    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    import math
    mean_r = sum(rewards) / len(rewards)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1))

    # Set up findings path FIRST (needed for prior-run context loading).
    _run_id = datetime.now(timezone.utc).isoformat()
    findings_path = vortex.memory_dir(company) / "analyst_findings.json"
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    # Build user message with convergence framing.
    user_message = (
        f"Build a predictive model of engagement for client: {company}\n\n"
        f"Data summary:\n"
        f"- {len(scored)} scored observations\n"
        f"- Reward: mean={mean_r:+.3f}, std={std_r:.3f}, "
        f"range=[{min(rewards):+.3f}, {max(rewards):+.3f}]\n"
        f"- Topics: {dict(topics.most_common())}\n"
        f"- Formats: {dict(formats.most_common())}\n"
        f"- Date range: {scored[0].get('posted_at', '?')[:10]} to "
        f"{scored[-1].get('posted_at', '?')[:10]}\n\n"
        f"You have {_MAX_TURNS} turns. Your goal: build a model that predicts "
        f"engagement (Spearman > 0.3 on LOO validation, or explain why that's "
        f"not achievable yet with {len(scored)} observations).\n\n"
        f"Store findings as you discover them (type='finding'). Store your "
        f"best model LAST (type='model') with validation metrics and "
        f"predicted-vs-actual rankings."
    )

    # Prior-run context: show the agent what it achieved before so it can
    # improve. This is the cross-run learning mechanism.
    try:
        _prior_af = json.loads(findings_path.read_text(encoding="utf-8")) if findings_path.exists() else {}
        _prior_findings = _prior_af.get("findings", [])
        _prior_runs = _prior_af.get("runs", [])

        if _prior_runs:
            _last = _prior_runs[-1]
            _last_rid = _last.get("run_id", "")

            # Show prior model if one exists
            _prior_models = [
                f for f in _prior_findings
                if f.get("type") == "model" and f.get("run_id") == _last_rid
            ]
            _prior_findings_only = [
                f for f in _prior_findings
                if f.get("type") != "model" and f.get("run_id") == _last_rid
            ]

            if _prior_models or _prior_findings_only:
                user_message += (
                    f"\n\nPRIOR RUN ({_last_rid[:10]}, "
                    f"{_last.get('tool_calls', '?')} tools, "
                    f"{_last.get('findings_stored', '?')} entries):\n"
                )

            if _prior_models:
                _pm = _prior_models[-1]
                user_message += (
                    f"\nPrior model: {_pm.get('claim', '')[:300]}\n"
                    f"Metrics: {_pm.get('evidence', '')[:300]}\n"
                    f"Ready: {_pm.get('model_ready', '?')} — "
                    f"{_pm.get('model_readiness_explanation', '')[:200]}\n"
                    f"\nYour job: IMPROVE on this model. If you can't beat it, "
                    f"explain what you tried.\n"
                )

            if _prior_findings_only:
                user_message += "\nPrior findings (validate or update):\n"
                for _pf in _prior_findings_only[:8]:
                    user_message += (
                        f"  [{_pf.get('confidence', '?').upper()}] "
                        f"{_pf.get('claim', '')[:150]}\n"
                    )
    except Exception:
        pass

    # Run the agentic loop.
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": user_message}]

    tool_calls = 0
    findings_stored = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    for turn in range(_MAX_TURNS):
        resp = client.messages.create(
            model=_ANALYST_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=messages,
        )

        # Track tokens
        if hasattr(resp, "usage"):
            total_input_tokens = max(total_input_tokens, resp.usage.input_tokens)
            total_output_tokens += resp.usage.output_tokens

        messages.append({"role": "assistant", "content": resp.content})

        if verbose:
            for block in resp.content:
                if hasattr(block, "text"):
                    print(block.text)

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "end_turn" or not tool_uses:
            break

        # Dispatch tools — include turn budget so the agent can plan.
        remaining_turns = _MAX_TURNS - turn - 1
        tool_results: list[dict] = []
        for tu in tool_uses:
            tool_calls += 1
            if verbose:
                print(f"  → {tu.name}({json.dumps(tu.input)[:120]})")
            result = _dispatch_tool(tu.name, tu.input, company, scored, run_id=_run_id)
            if tu.name == "store_finding":
                findings_stored += 1
            if verbose:
                print(f"    ← {result[:200]}")

            # Append turn budget to every tool result so the agent knows
            # when to stop exploring and start validating.
            result_with_budget = result.rstrip("}") + f', "turns_remaining": {remaining_turns}' + "}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_with_budget,
            })

        messages.append({"role": "user", "content": tool_results})

    elapsed = time.time() - start_time

    # Record run metadata. Finding history is preserved (nothing cleared).
    try:
        data = json.loads(findings_path.read_text(encoding="utf-8")) if findings_path.exists() else {"findings": [], "runs": []}
    except Exception:
        data = {"findings": [], "runs": []}

    # Check if a model was stored this run
    run_findings = [f for f in data.get("findings", []) if f.get("run_id") == _run_id]
    model_entries = [f for f in run_findings if f.get("type") == "model"]

    run_meta = {
        "run_id": _run_id,
        "timestamp": _run_id,
        "model": _ANALYST_MODEL,
        "tool_calls": tool_calls,
        "findings_stored": findings_stored,
        "model_stored": len(model_entries) > 0,
        "model_ready": model_entries[-1].get("model_ready", False) if model_entries else False,
        "turns": turn + 1,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "observation_count": len(scored),
    }
    data["runs"].append(run_meta)

    tmp = findings_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(findings_path)

    logger.info(
        "[analyst] %s: %d tool calls, %d findings, model_stored=%s, "
        "model_ready=%s, %d turns, %.1fs",
        company, tool_calls, findings_stored,
        run_meta["model_stored"], run_meta["model_ready"],
        turn + 1, elapsed,
    )

    return run_meta
