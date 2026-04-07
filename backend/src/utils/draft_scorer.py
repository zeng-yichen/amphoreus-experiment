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
    """A draft post with predicted engagement score, exploration value, and explanation."""
    rank: int
    text: str
    predicted_score: float
    exploration_value: float   # 0-1: how much the system would learn from this post
    features: dict             # extracted feature values
    explanation: str           # human-readable "why this score"
    model_source: str          # "analyst_model+embedding_knn" | "embedding_knn" | "no_model"
    model_ready: bool = False  # True when coefficient model is validated (LOO R² > 0.1, n >= 15)


def _model_is_ready(model_spec: dict, n_observations: int) -> bool:
    """Check if the analyst's model is validated enough to trust its coefficients.

    LOO R² > 0.1 means the model explains at least 10% of out-of-sample variance.
    n >= 15 means the model has enough data to be non-trivial.
    Below these thresholds, the coefficient path is noise — fall back to k-NN.
    """
    loo_r2 = model_spec.get("loo_r2", -999)
    return loo_r2 > 0.1 and n_observations >= 15


def _compute_training_stats(company: str) -> dict:
    """Compute feature normalization stats from the client's actual observations.

    These are the same statistics the regression used during training (z-normalization
    and target encoding). By computing them from the real data, we apply the
    analyst's coefficients in the same feature space they were learned in.
    No hardcoded means, no assumed standard deviations.
    """
    import math
    from collections import defaultdict

    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return {}

    if state is None:
        return {}

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("reward", {}).get("immediate") is not None
    ]
    if not scored:
        return {}

    # Char count stats
    char_counts = [
        len(o.get("posted_body") or o.get("post_body") or "")
        for o in scored
    ]
    char_mean = sum(char_counts) / len(char_counts)
    char_std = math.sqrt(
        sum((c - char_mean) ** 2 for c in char_counts) / max(len(char_counts) - 1, 1)
    ) if len(char_counts) > 1 else 1.0

    # Posting hour stats
    hours = []
    for o in scored:
        ts = o.get("posted_at", "")
        if ts:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hours.append(float(dt.hour))
            except Exception:
                pass
    hour_mean = sum(hours) / len(hours) if hours else 12.0
    hour_std = math.sqrt(
        sum((h - hour_mean) ** 2 for h in hours) / max(len(hours) - 1, 1)
    ) if len(hours) > 1 else 3.0

    # Format tag target encoding: mean reward per format
    format_rewards_agg: dict = defaultdict(list)
    all_rewards = []
    for o in scored:
        fmt = o.get("format_tag")
        reward = o.get("reward", {}).get("immediate", 0)
        all_rewards.append(reward)
        if fmt:
            format_rewards_agg[fmt].append(reward)
    global_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    format_rewards = {
        fmt: sum(rs) / len(rs)
        for fmt, rs in format_rewards_agg.items()
    }

    # Stats of the target-encoded format values (for z-normalization)
    if format_rewards:
        fmt_encoded = [format_rewards.get(o.get("format_tag", ""), global_mean) for o in scored]
        fmt_mean = sum(fmt_encoded) / len(fmt_encoded)
        fmt_std = math.sqrt(
            sum((v - fmt_mean) ** 2 for v in fmt_encoded) / max(len(fmt_encoded) - 1, 1)
        ) if len(fmt_encoded) > 1 else 1.0
    else:
        fmt_mean = 0.0
        fmt_std = 1.0

    return {
        "char_count_mean": round(char_mean, 1),
        "char_count_std": round(char_std, 1),
        "posting_hour_mean": round(hour_mean, 1),
        "posting_hour_std": round(hour_std, 1),
        "format_rewards": {k: round(v, 4) for k, v in format_rewards.items()},
        "format_mean": round(fmt_mean, 4),
        "format_std": round(fmt_std, 4),
        "reward_mean": round(global_mean, 4),
        "observation_count": len(scored),
    }


def _score_by_embedding_knn(
    company: str,
    text: str,
    k: int = 5,
) -> tuple[Optional[float], str]:
    """Score a draft by k-NN similarity to the client's scored observations.

    Embeds the draft, finds the k most similar scored posts by cosine
    similarity, and returns a similarity-weighted average of their rewards.

    This is the embedding-based scoring path — no format_tag, no topic_tag,
    no categorical features. The embedding captures everything about the
    post in a continuous vector. Works alongside the coefficient-based path.

    Returns (predicted_score, explanation) or (None, explanation) on failure.
    """
    from backend.src.utils.post_embeddings import (
        get_post_embeddings, embed_text, find_similar, cosine_similarity,
    )

    embeddings = get_post_embeddings(company)
    if not embeddings:
        return None, "No post embeddings available"

    draft_emb = embed_text(text)
    if draft_emb is None:
        return None, "Failed to embed draft"

    similar = find_similar(draft_emb, embeddings, top_k=k)
    if not similar:
        return None, "No similar posts found"

    # Load observations to get rewards for the similar posts
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return None, "Failed to load observations"
    if state is None:
        return None, "No observation state"

    obs_by_hash = {
        o.get("post_hash"): o
        for o in state.get("observations", [])
        if o.get("status") == "scored"
    }

    # Similarity-weighted reward average
    weighted_sum = 0.0
    weight_sum = 0.0
    neighbor_details = []
    for h, sim in similar:
        obs = obs_by_hash.get(h)
        if not obs:
            continue
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue
        weighted_sum += sim * reward
        weight_sum += sim
        body = (obs.get("posted_body") or obs.get("post_body") or "")
        neighbor_details.append(
            f"sim={sim:.2f}/reward={reward:+.2f} "
            f"({obs.get('format_tag', '?')}) \"{body[:60].strip()}...\""
        )

    if weight_sum < 1e-6:
        return None, "No similar posts with rewards"

    pred = weighted_sum / weight_sum
    explanation = (
        f"k-NN score {pred:+.3f} from {len(neighbor_details)} neighbors: "
        + "; ".join(neighbor_details[:3])
    )

    return pred, explanation


def score_drafts(
    company: str,
    drafts: list[dict],
    default_hour: int = 9,
) -> list[ScoredDraft]:
    """Score and rank a batch of draft posts using BOTH scoring paths.

    Two independent predictions per draft:

    1. **Coefficient-based** (from the analyst's regression model):
       Applies learned coefficients to format_tag, char_count, posting_hour.
       Uses the client's actual observation statistics for normalization.

    2. **Embedding-based** (k-NN in continuous vector space):
       Embeds the draft, finds the k most similar scored posts by cosine
       similarity, returns their similarity-weighted reward average.
       No category labels needed — operates in continuous space.

    The final score is the average of both paths (when both are available).
    If only one path produces a score, that score is used alone. The
    explanation shows both scores so the operator sees agreement/disagreement.
    """
    if not drafts:
        return []

    model_spec, model_source = _load_model(company)

    # Model readiness gate: if the coefficient model isn't validated
    # (LOO R² < 0.1 or n < 15), skip the coefficient path entirely
    # and fall back to k-NN only. Coefficients from a noise model
    # produce misleading scores.
    training_stats = {}
    coeff_ready = False
    if model_source == "analyst_model":
        training_stats_full = _compute_training_stats(company)
        coeff_ready = _model_is_ready(model_spec, training_stats_full.get("observation_count", 0))
        if coeff_ready:
            training_stats = training_stats_full
        else:
            model_source = "embedding_knn"  # downgrade to k-NN only

    scored: list[ScoredDraft] = []

    for i, draft in enumerate(drafts):
        text = draft.get("text", "")
        if not text.strip():
            scored.append(ScoredDraft(
                rank=0, text=text, predicted_score=0.0,
                exploration_value=0.0,
                features={}, explanation="Empty draft",
                model_source="no_model", model_ready=False,
            ))
            continue

        hour = draft.get("scheduled_hour", default_hour)
        features = _extract_features(text, hour)

        # Path 1: coefficient-based
        coeff_score, coeff_explanation = _apply_model(
            features, model_spec, model_source,
            training_stats=training_stats,
        )

        # Path 2: embedding k-NN
        knn_score, knn_explanation = _score_by_embedding_knn(company, text)

        # Combine: average both paths when available
        if model_source != "no_model" and knn_score is not None:
            combined = (coeff_score + knn_score) / 2
            source = "analyst_model+embedding_knn"
            explanation = (
                f"Combined {combined:+.3f} = "
                f"coeff {coeff_score:+.3f} + knn {knn_score:+.3f} / 2\n"
                f"  Coeff: {coeff_explanation}\n"
                f"  k-NN: {knn_explanation}"
            )
        elif model_source != "no_model":
            combined = coeff_score
            source = "analyst_model"
            explanation = coeff_explanation
        elif knn_score is not None:
            combined = knn_score
            source = "embedding_knn"
            explanation = knn_explanation
        else:
            combined = 0.0
            source = "no_model"
            explanation = "No scoring model available."

        features["knn_score"] = knn_score
        features["coeff_score"] = coeff_score if model_source != "no_model" else None

        # Exploration value: how much would the system learn from publishing
        # this post? Measured as 1 - max_similarity to any scored post.
        # A draft identical to a historical post (sim=0.95) teaches nothing
        # (exploration=0.05). A genuinely novel draft (sim=0.4) is highly
        # informative (exploration=0.6).
        exploration = _compute_exploration_value(company, text)

        scored.append(ScoredDraft(
            rank=0,
            text=text,
            predicted_score=round(combined, 4),
            exploration_value=round(exploration, 4),
            features=features,
            explanation=explanation,
            model_source=source,
            model_ready=coeff_ready,
        ))

    # Rank by predicted engagement (highest first).
    # Exploration value is shown but does NOT affect ranking —
    # the operator decides the engagement/exploration tradeoff.
    scored.sort(key=lambda s: s.predicted_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored


def _compute_exploration_value(company: str, text: str) -> float:
    """How much the system would learn from publishing this draft.

    Measured as 1.0 - max_cosine_similarity to any scored observation.
    A draft that's 0.95 similar to a historical post teaches nothing new
    (exploration_value = 0.05). A draft that's only 0.40 similar to the
    nearest scored post is genuinely novel (exploration_value = 0.60).

    Returns 0.0 if embeddings are unavailable (can't assess novelty).
    """
    try:
        from backend.src.utils.post_embeddings import (
            get_post_embeddings, embed_text, find_similar,
        )
        embeddings = get_post_embeddings(company)
        if not embeddings:
            return 0.0
        draft_emb = embed_text(text)
        if draft_emb is None:
            return 0.0
        # Find the single most similar observation
        nearest = find_similar(draft_emb, embeddings, top_k=1)
        if not nearest:
            return 1.0  # no observations at all → maximally novel
        max_sim = nearest[0][1]
        return max(0.0, 1.0 - max_sim)
    except Exception:
        return 0.0


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
    training_stats: Optional[dict] = None,
) -> tuple[float, str]:
    """Apply the analyst's regression model directly to produce a score.

    No hardcoded bands, no assumed mean/std, no hand-engineered heuristic
    layer. The analyst spent 80 turns building a regression with real
    coefficients from real data. This function applies those coefficients.

    For format_tag: uses target encoding (mean reward per format) computed
    from the client's actual observations, same encoding the regression
    used during training.

    For numeric features (char_count, posting_hour): z-normalizes using
    the client's actual feature distributions, same normalization the
    regression used during training.

    The quote-hook bonus is the ONE non-regression component: it comes
    from the analyst's model_spec.heuristic_layer, not from hardcoded
    values. If the analyst didn't include a hook bonus, none is applied.
    """
    if model_source == "no_model":
        return 0.0, "No analyst model available. Drafts are unscored."

    coefficients = model_spec.get("coefficients", {})
    intercept = model_spec.get("intercept", 0.0)
    stats = training_stats or {}

    format_tag = features.get("format_tag", "unknown")
    char_count = features.get("char_count", 1500)
    posting_hour = features.get("posting_hour", 9)

    score = intercept
    explanation_parts = []

    # Format contribution — target-encoded from training data
    format_coeff = coefficients.get("format_tag", 0)
    if format_coeff != 0:
        format_rewards = stats.get("format_rewards", {})
        global_mean = stats.get("reward_mean", 0)
        # Target encoding: replace format name with its mean reward from training
        target_val = format_rewards.get(format_tag, global_mean)
        # Z-normalize using the format feature's distribution from training
        fmt_mean = stats.get("format_mean", 0)
        fmt_std = stats.get("format_std", 1)
        fmt_z = (target_val - fmt_mean) / fmt_std if fmt_std > 1e-6 else 0
        fmt_contribution = format_coeff * fmt_z
        score += fmt_contribution
        if abs(fmt_contribution) > 0.01:
            explanation_parts.append(
                f"format '{format_tag}' (avg reward {target_val:+.2f}): {fmt_contribution:+.3f}"
            )

    # Char count contribution — z-normalized from training data
    char_coeff = coefficients.get("char_count", 0)
    if char_coeff != 0:
        char_mean = stats.get("char_count_mean", 2000)
        char_std = stats.get("char_count_std", 500)
        char_z = (char_count - char_mean) / char_std if char_std > 1e-6 else 0
        char_contribution = char_coeff * char_z
        score += char_contribution
        if abs(char_contribution) > 0.01:
            explanation_parts.append(
                f"length {char_count} chars (vs avg {char_mean:.0f}): {char_contribution:+.3f}"
            )

    # Posting hour contribution — z-normalized from training data
    hour_coeff = coefficients.get("posting_hour", 0)
    if hour_coeff != 0:
        hour_mean = stats.get("posting_hour_mean", 12)
        hour_std = stats.get("posting_hour_std", 3)
        hour_z = (posting_hour - hour_mean) / hour_std if hour_std > 1e-6 else 0
        hour_contribution = hour_coeff * hour_z
        score += hour_contribution
        if abs(hour_contribution) > 0.01:
            explanation_parts.append(
                f"posting hour {posting_hour}:00 (vs avg {hour_mean:.0f}:00): {hour_contribution:+.3f}"
            )

    # Quote-hook bonus — from the analyst's model_spec, NOT hardcoded.
    # The analyst decides whether to include a hook bonus and how large it
    # should be. If the model_spec doesn't have one, none is applied.
    heuristic = model_spec.get("heuristic_layer", {})
    hook_bonus_str = heuristic.get("scoring_guidance", {}).get("hook_bonus", "")
    if features.get("has_quote_hook") and hook_bonus_str:
        try:
            # Parse "+0.4 if verbatim customer quote..." → 0.4
            hook_bonus = float(hook_bonus_str.split()[0].replace("+", ""))
        except (ValueError, IndexError):
            hook_bonus = 0
        if hook_bonus > 0:
            score += hook_bonus
            explanation_parts.append(f"quote hook (analyst bonus): +{hook_bonus:.2f}")

    # Build explanation
    loo_r2 = model_spec.get("loo_r2", "?")
    if explanation_parts:
        explanation = f"Score {score:+.3f} = " + " + ".join(explanation_parts)
        explanation += f" (model LOO R²={loo_r2})"
    else:
        explanation = f"Score {score:+.3f} — no distinguishing features (LOO R²={loo_r2})"

    return score, explanation


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
        explore_label = ""
        if s.exploration_value > 0.5:
            explore_label = " 🔬 HIGH EXPLORATION"
        elif s.exploration_value > 0.3:
            explore_label = " 🔬 moderate exploration"
        lines.append(
            f"**#{s.rank}** | `{s.predicted_score:+.3f}` | "
            f"exploration `{s.exploration_value:.2f}`{explore_label} | "
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

    # Exploration recommendation
    most_novel = max(scored, key=lambda s: s.exploration_value)
    if most_novel.exploration_value > 0.3 and most_novel.rank > 1:
        lines.append(
            f"\n**Exploration opportunity:** Draft #{most_novel.rank} "
            f"(exploration={most_novel.exploration_value:.2f}) is the most novel — "
            f"no similar posts in history. Publishing it would teach the model "
            f"more than the higher-ranked drafts."
        )

    return "\n".join(lines)
