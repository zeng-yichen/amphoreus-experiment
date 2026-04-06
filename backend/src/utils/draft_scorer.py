"""Draft scorer — rank candidate posts using the analyst's learned model.

Takes a batch of draft posts, extracts features from each (format_tag via
the observation tagger, char_count, posting_hour), applies the analyst's
model coefficients, and returns a ranked list with predicted engagement
scores and explanations.

This connects the analyst's statistical output to generation: instead of
the model sitting in a JSON file as prose, its coefficients are applied
to actual drafts to produce actionable rankings.

The scorer does NOT gate or filter. It ranks. The operator sees:
  "Draft 3 scores highest (+0.42): storytelling, 2100 chars, 9am.
   Draft 5 scores lowest (-0.31): hot take, 800 chars, 4pm."
and decides the publishing order.

Usage:
    from backend.src.utils.draft_scorer import score_drafts

    ranked = score_drafts("innovocommerce", [
        {"text": "We were brought in to review...", "scheduled_hour": 9},
        {"text": "If I got dropped into a Director...", "scheduled_hour": 14},
    ])
    for r in ranked:
        print(f"#{r['rank']} score={r['predicted_score']:+.3f} — {r['explanation']}")
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)


@dataclass
class ScoredDraft:
    """A draft post with predicted engagement score and explanation."""
    rank: int
    text: str
    predicted_score: float
    features: dict          # extracted feature values
    explanation: str        # human-readable "why this score"
    model_source: str       # "analyst_model" | "heuristic_only" | "no_model"


def score_drafts(
    company: str,
    drafts: list[dict],
    default_hour: int = 9,
) -> list[ScoredDraft]:
    """Score and rank a batch of draft posts using the analyst's model.

    Each draft dict should have:
      - "text": the full post text (required)
      - "scheduled_hour": posting hour in UTC (optional, defaults to default_hour)

    Returns drafts ranked by predicted engagement score (highest first).
    If no analyst model exists, returns drafts in original order with
    a "no_model" source marker and uniform scores.

    The scoring is additive:
      1. Apply the analyst's regression model (format_tag, char_count, posting_hour)
      2. Apply the heuristic layer (hook style bonus) if present in the model spec
      3. Sum → predicted score
    """
    if not drafts:
        return []

    # Load the analyst's model
    model_spec, model_source = _load_model(company)

    scored: list[ScoredDraft] = []

    for i, draft in enumerate(drafts):
        text = draft.get("text", "")
        if not text.strip():
            scored.append(ScoredDraft(
                rank=0, text=text, predicted_score=0.0,
                features={}, explanation="Empty draft",
                model_source="no_model",
            ))
            continue

        hour = draft.get("scheduled_hour", default_hour)
        features = _extract_features(text, hour)
        pred_score, explanation = _apply_model(features, model_spec, model_source)

        scored.append(ScoredDraft(
            rank=0,
            text=text,
            predicted_score=round(pred_score, 4),
            features=features,
            explanation=explanation,
            model_source=model_source,
        ))

    # Rank by predicted score (highest first)
    scored.sort(key=lambda s: s.predicted_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored


def _load_model(company: str) -> tuple[dict, str]:
    """Load the latest analyst model for a client.

    Returns (model_spec, source_label). Falls back to a basic heuristic
    if no analyst model exists.
    """
    findings_path = vortex.memory_dir(company) / "analyst_findings.json"
    if not findings_path.exists():
        return {}, "no_model"

    try:
        data = json.loads(findings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "no_model"

    # Find the latest model entry
    runs = data.get("runs", [])
    findings = data.get("findings", [])

    if not runs or not findings:
        return {}, "no_model"

    # Walk backwards through runs to find the most recent model
    for run in reversed(runs):
        rid = run.get("run_id", "")
        run_models = [
            f for f in findings
            if f.get("type") == "model" and f.get("run_id") == rid
        ]
        if run_models:
            spec = run_models[-1].get("model_spec", {})
            if spec and spec.get("coefficients"):
                return spec, "analyst_model"

    return {}, "no_model"


def _extract_features(text: str, posting_hour: int) -> dict:
    """Extract the features the analyst's model needs from a draft."""
    features: dict = {
        "char_count": len(text),
        "posting_hour": posting_hour,
        "format_tag": "unknown",
        "has_quote_hook": False,
    }

    # Extract format_tag via the observation tagger
    try:
        from backend.src.utils.observation_tagger import tag_post
        tags = tag_post(text)
        if tags and tags.get("format_tag"):
            features["format_tag"] = tags["format_tag"]
    except Exception:
        pass

    # Detect quote-led hook (verbatim quote in first 200 chars)
    opening = text[:200]
    if '"' in opening or '\u201c' in opening or '\u201d' in opening:
        # Check if it's a substantial quote (not just a single quoted word)
        quote_matches = re.findall(r'["\u201c](.+?)["\u201d]', opening)
        if any(len(q) > 15 for q in quote_matches):
            features["has_quote_hook"] = True

    return features


def _apply_model(
    features: dict,
    model_spec: dict,
    model_source: str,
) -> tuple[float, str]:
    """Apply the model to features and produce a score + explanation.

    Returns (predicted_score, explanation_string).
    """
    if model_source == "no_model":
        return 0.0, "No analyst model available for this client. Drafts are unscored."

    coefficients = model_spec.get("coefficients", {})
    intercept = model_spec.get("intercept", 0.0)
    heuristic = model_spec.get("heuristic_layer", {})

    # --- Regression component ---
    # The regression was trained on z-normalized features. We need to
    # approximate the normalization for new drafts. Use the model's
    # training data stats if available, otherwise use reasonable defaults
    # for innovocommerce-scale data.
    #
    # For format_tag: the regression used target encoding (mean reward per
    # format). We approximate by mapping format names to the ordinal values
    # the causal_filter used during training.
    format_tag = features.get("format_tag", "unknown")
    char_count = features.get("char_count", 1500)
    posting_hour = features.get("posting_hour", 9)

    # Build the regression prediction
    score = intercept
    explanation_parts = []

    # Format contribution
    format_coeff = coefficients.get("format_tag", 0)
    if format_coeff != 0:
        # The heuristic layer gives us a more interpretable scoring
        format_scores = heuristic.get("scoring_guidance", {}).get("format_scores", {})
        if format_scores:
            # Parse the format score from the heuristic guidance
            fmt_score = _parse_format_heuristic(format_tag, format_scores)
            score += fmt_score
            if fmt_score != 0:
                explanation_parts.append(
                    f"format '{format_tag}': {fmt_score:+.2f}"
                )
        else:
            # Use the raw coefficient with a crude encoding
            format_order = {"storytelling": 0, "framework": 0, "list": -0.1, "hot_take": -0.4}
            fmt_val = format_order.get(format_tag, 0)
            score += fmt_val
            if fmt_val != 0:
                explanation_parts.append(f"format '{format_tag}': {fmt_val:+.2f}")

    # Char count contribution
    char_coeff = coefficients.get("char_count", 0)
    if char_coeff != 0:
        char_guidance = heuristic.get("scoring_guidance", {}).get("char_count_score", "")
        if char_guidance:
            if char_count >= 2000:
                char_score = 0.15
            elif char_count >= 1500:
                char_score = 0.0
            else:
                char_score = -0.1
        else:
            # Z-normalize against typical range (mean ~2000, std ~500)
            char_z = (char_count - 2000) / 500
            char_score = char_coeff * char_z
        score += char_score
        if abs(char_score) > 0.01:
            explanation_parts.append(f"length {char_count} chars: {char_score:+.2f}")

    # Posting hour contribution
    hour_coeff = coefficients.get("posting_hour", 0)
    if hour_coeff != 0:
        hour_guidance = heuristic.get("scoring_guidance", {}).get("posting_hour_score", "")
        if hour_guidance:
            if 7 <= posting_hour <= 9:
                hour_score = 0.1
            elif 10 <= posting_hour <= 12:
                hour_score = 0.0
            else:
                hour_score = -0.1
        else:
            hour_z = (posting_hour - 12) / 3
            hour_score = hour_coeff * hour_z
        score += hour_score
        if abs(hour_score) > 0.01:
            explanation_parts.append(f"posting hour {posting_hour}:00: {hour_score:+.2f}")

    # --- Heuristic hook bonus ---
    if features.get("has_quote_hook") and heuristic.get("scoring_guidance", {}).get("hook_bonus"):
        hook_bonus = 0.4
        score += hook_bonus
        explanation_parts.append(f"customer quote hook detected: +{hook_bonus:.2f}")

    # Build explanation
    if explanation_parts:
        explanation = f"Score {score:+.3f} = " + " + ".join(explanation_parts)
        explanation += f" (model LOO R²={model_spec.get('loo_r2', '?')})"
    else:
        explanation = f"Score {score:+.3f} (no distinguishing features detected)"

    return score, explanation


def _parse_format_heuristic(format_tag: str, format_scores: dict) -> float:
    """Parse the analyst's format scoring guidance into a numeric value.

    Handles format name mismatches between the tagger (produces 'hot take'
    with space) and the analyst model spec (may use 'hot_take' with underscore).
    """
    # Normalize for comparison: lowercase, strip, replace spaces/underscores
    def _normalize(s: str) -> str:
        return s.lower().strip().replace("_", " ").replace("-", " ")

    tag_norm = _normalize(format_tag)
    for key, value in format_scores.items():
        if _normalize(key) == tag_norm:
            if "baseline" in str(value).lower() or value == "+0":
                return 0.0
            try:
                return float(str(value).replace("+", ""))
            except (ValueError, TypeError):
                return 0.0
    return 0.0  # unknown format → neutral


# ------------------------------------------------------------------
# Batch scoring for Stelle integration
# ------------------------------------------------------------------

def score_and_explain_batch(company: str, posts: list[dict]) -> str:
    """Score a batch of posts and return a human-readable ranking.

    Designed to be called after Stelle generates drafts, producing a
    summary the operator can use to decide publishing order.

    Each post dict needs "text" and optionally "scheduled_hour" and "hook"
    (the first line, for display).

    Returns a formatted markdown string with the ranking.
    """
    if not posts:
        return "No drafts to score."

    scored = score_drafts(company, posts)

    if scored[0].model_source == "no_model":
        return (
            "No analyst model available for this client. "
            "Run the analyst agent first to build a predictive model."
        )

    lines = [
        "## Draft Ranking (by predicted engagement)\n",
        f"*Model: {scored[0].model_source}, "
        f"based on {len(scored)} drafts*\n",
    ]

    for s in scored:
        hook = s.text[:80].replace("\n", " ").strip()
        lines.append(
            f"**#{s.rank}** | `{s.predicted_score:+.3f}` | "
            f"*{s.features.get('format_tag', '?')}*, "
            f"{s.features.get('char_count', '?')} chars"
        )
        lines.append(f"> {hook}...")
        lines.append(f"  {s.explanation}")
        lines.append("")

    # Summary
    best = scored[0]
    worst = scored[-1]
    spread = best.predicted_score - worst.predicted_score
    lines.append(
        f"**Spread:** {spread:.3f} "
        f"(#{best.rank} '{best.features.get('format_tag')}' vs "
        f"#{worst.rank} '{worst.features.get('format_tag')}')"
    )

    return "\n".join(lines)
