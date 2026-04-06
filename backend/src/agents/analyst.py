"""Analyst — hypothesis-driven engagement analysis via tool-use agent.

Replaces the fixed pipeline's predetermined analysis sequence with an
open-ended agent that forms and tests its own hypotheses about what
predicts engagement for a specific client.

The agent has access to the same statistical primitives the pipeline uses
(correlation, partial correlation, regression, effect size, transition
matrices, embeddings) but decides what to run and in what order. It can
pursue hypotheses the pipeline never encoded.

Design (bitter lesson compliance):
  - No predetermined analysis sequence.
  - The model decides which tools to call and in what order.
  - Tools are composable statistical primitives, not pipeline steps.
  - Findings are stored with evidence and confidence, not as assertions.
  - The fixed pipeline continues to run in parallel. This agent is
    additive — it discovers things the pipeline misses, but doesn't
    replace the pipeline until its findings prove more useful.

Usage:
    from backend.src.agents.analyst import run_analysis

    findings = run_analysis("innovocommerce")
    # → stored in memory/{company}/analyst_findings.json
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
_MAX_TURNS = 40  # safety ceiling on tool-use turns


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an engagement analyst for a LinkedIn ghostwriting agency. Your job is \
to discover what predicts post performance for a specific client.

You have access to the client's scored observation history and a set of \
statistical tools. Use them to form hypotheses, test them, and report findings.

## How to work

1. Start by exploring the data. Use `query_observations` to understand the \
   client's history: how many posts, what topics, what formats, reward \
   distribution, time range.

2. Form hypotheses about what drives engagement. These might be about:
   - Content attributes (topic, format, length, opening style)
   - Temporal patterns (day of week, posting cadence, sequence effects)
   - Interaction effects (does topic X work better in format Y?)
   - Anything else the data suggests

3. Test each hypothesis with the appropriate tool. Use `compute_correlation` \
   for continuous relationships, `compute_effect_size` for categorical \
   comparisons, `compute_partial_correlation` to control for confounders, \
   `fit_regression` for multivariate prediction.

4. Report findings with evidence. Include effect sizes, sample sizes, and \
   honest confidence assessments.

## Statistical discipline

- **Effect size over p-values.** At n<50, almost nothing reaches p<0.05. \
  Report the effect size and let the reader decide if it matters.
- **Multiple comparisons.** If you test 10 hypotheses, expect 1-2 to look \
  significant by chance. Flag this explicitly.
- **Confounders.** When you find a correlation, ask what else could explain it. \
  Use `compute_partial_correlation` to check.
- **Sample size honesty.** If n<15 for a subgroup, say so. Don't report a \
  "finding" from 3 observations.
- **Replication.** A finding that appears in one analysis is a hypothesis. \
  A finding that appears consistently across multiple analyses is knowledge.

## What to report

After your analysis, call `store_finding` for each discovery worth reporting. \
A good finding has:
- A clear, specific claim ("posts tagged 'hot take' after a 'case study' \
  average +0.8 higher reward")
- Evidence (effect size, sample size, correlation value)
- Confidence level ("strong" / "suggestive" / "weak" / "insufficient data")
- Whether it's consistent with or contradicts the fixed pipeline's outputs \
  (which you'll see in the data)

Do NOT report findings that are trivially obvious (e.g., "posts with more \
impressions have higher engagement") or findings with n<5.

## Tools available

You have 8 tools. Each wraps a pure statistical function. Call them in \
whatever order makes sense for your analysis — there is no predetermined \
sequence. You may call the same tool multiple times with different inputs.
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
            "Store an analytical finding for this client. Findings are persisted "
            "to memory and surfaced in the strategy brief. Only store findings "
            "worth acting on — not every test result. Include the evidence "
            "and an honest confidence assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "A clear, specific, actionable claim (e.g., 'posts under 1800 chars average +0.6 higher reward than posts over 2500 chars').",
                },
                "evidence": {
                    "type": "string",
                    "description": "The statistical evidence supporting the claim (effect size, n, correlation, etc.).",
                },
                "confidence": {
                    "type": "string",
                    "description": "'strong' (large effect, adequate n) / 'suggestive' (moderate effect or small n) / 'weak' (small effect or very small n).",
                },
                "contradicts_pipeline": {
                    "type": "boolean",
                    "description": "True if this finding contradicts the fixed pipeline's output for this client.",
                },
                "hypothesis_tested": {
                    "type": "string",
                    "description": "The hypothesis that was tested to produce this finding.",
                },
            },
            "required": ["claim", "evidence", "confidence"],
        },
    },
]


# ------------------------------------------------------------------
# Tool dispatch — thin wrappers around existing statistical primitives
# ------------------------------------------------------------------

def _dispatch_tool(tool_name: str, tool_input: dict, company: str,
                   observations: list[dict]) -> str:
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
            return _tool_store_finding(tool_input, company)
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


def _tool_store_finding(tool_input: dict, company: str) -> str:
    findings_path = vortex.memory_dir(company) / "analyst_findings.json"
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        existing = json.loads(findings_path.read_text(encoding="utf-8")) if findings_path.exists() else {"findings": [], "runs": []}
    except Exception:
        existing = {"findings": [], "runs": []}

    finding = {
        "claim": tool_input["claim"],
        "evidence": tool_input["evidence"],
        "confidence": tool_input.get("confidence", "suggestive"),
        "contradicts_pipeline": tool_input.get("contradicts_pipeline", False),
        "hypothesis_tested": tool_input.get("hypothesis_tested", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    existing["findings"].append(finding)

    tmp = findings_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(findings_path)

    return json.dumps({"status": "stored", "total_findings": len(existing["findings"])})


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

    user_message = (
        f"Analyze engagement patterns for client: {company}\n\n"
        f"Data summary:\n"
        f"- {len(scored)} scored observations\n"
        f"- Reward: mean={mean_r:+.3f}, std={std_r:.3f}, "
        f"range=[{min(rewards):+.3f}, {max(rewards):+.3f}]\n"
        f"- Topics: {dict(topics.most_common())}\n"
        f"- Formats: {dict(formats.most_common())}\n"
        f"- Date range: {scored[0].get('posted_at', '?')[:10]} to "
        f"{scored[-1].get('posted_at', '?')[:10]}\n\n"
        "Start by exploring the data with query_observations, then form and test "
        "hypotheses. Store each actionable finding with store_finding."
    )

    # Also load the fixed pipeline's findings so the agent can compare
    pipeline_context = ""
    causal_path = vortex.memory_dir(company) / "causal_dimensions.json"
    if causal_path.exists():
        try:
            causal = json.loads(causal_path.read_text(encoding="utf-8"))
            pipeline_context += "\n\nFIXED PIPELINE OUTPUTS (for comparison):\n"
            pipeline_context += f"Causal filter (n={causal.get('observation_count', '?')}): "
            for d in causal.get("dimensions", []):
                pipeline_context += (
                    f"{d['dimension']}: marginal={d['marginal_correlation']:+.3f}, "
                    f"partial={d['partial_correlation']:+.3f} → {d['classification']}\n"
                )
        except Exception:
            pass
    if pipeline_context:
        user_message += pipeline_context

    # Run the agentic loop
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

        # Log text output if verbose
        if verbose:
            for block in resp.content:
                if hasattr(block, "text"):
                    print(block.text)

        # Check for tool calls
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "end_turn" or not tool_uses:
            break

        # Dispatch tools
        tool_results: list[dict] = []
        for tu in tool_uses:
            tool_calls += 1
            if verbose:
                print(f"  → {tu.name}({json.dumps(tu.input)[:120]})")
            result = _dispatch_tool(tu.name, tu.input, company, scored)
            if tu.name == "store_finding":
                findings_stored += 1
            if verbose:
                print(f"    ← {result[:200]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    elapsed = time.time() - start_time

    # Record the run metadata in the findings file
    findings_path = vortex.memory_dir(company) / "analyst_findings.json"
    try:
        data = json.loads(findings_path.read_text(encoding="utf-8")) if findings_path.exists() else {"findings": [], "runs": []}
    except Exception:
        data = {"findings": [], "runs": []}

    run_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": _ANALYST_MODEL,
        "tool_calls": tool_calls,
        "findings_stored": findings_stored,
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
        "[analyst] %s: %d tool calls, %d findings, %d turns, %.1fs, "
        "in=%d out=%d tokens",
        company, tool_calls, findings_stored, turn + 1, elapsed,
        total_input_tokens, total_output_tokens,
    )

    return run_meta
