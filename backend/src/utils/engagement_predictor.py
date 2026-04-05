"""Pre-publish engagement predictor.

Given a draft post's feature vector (Cyrene scores, constitutional scores,
alignment, edit_similarity, char count, posting hour/day), predicts how it
will perform against the client's historical engagement distribution.

Uses ridge regression implemented with stdlib math only (no sklearn/numpy).
Model is cached in memory/{company}/engagement_model.json and recomputed
during ordinal_sync.

Usage:
    from backend.src.utils.engagement_predictor import (
        build_engagement_model,
        predict_engagement,
    )

    # During ordinal_sync:
    build_engagement_model("innovocommerce")

    # At prediction time:
    result = predict_engagement("innovocommerce", {
        "char_count": 1200,
        "posting_hour": 14,
        "posting_day": 1,  # 0=Mon..6=Sun
        ...
    })
    if result:
        print(result.predicted_reward, result.top_positive, result.top_negative)
"""

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS = 20
DEFAULT_RIDGE_LAMBDA = 1.0


@dataclass
class PredictionResult:
    """Engagement prediction with actionable diagnostics."""
    predicted_reward: float
    confidence_low: float
    confidence_high: float
    top_positive: list[dict]  # [{"feature": str, "contribution": float}, ...]
    top_negative: list[dict]
    model_r_squared: float
    observation_count: int


# ------------------------------------------------------------------
# Ridge regression with stdlib math
# ------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [_dot(row, vec) for row in mat]


def _transpose(mat: list[list[float]]) -> list[list[float]]:
    if not mat:
        return []
    cols = len(mat[0])
    return [[mat[r][c] for r in range(len(mat))] for c in range(cols)]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = _transpose(b)
    return [[_dot(row, col) for col in bt] for row in a]


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _add_mat(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _scale_mat(mat: list[list[float]], s: float) -> list[list[float]]:
    return [[v * s for v in row] for row in mat]


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b via Gauss-Jordan elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            # Singular — set coefficient to 0
            aug[col][col] = 1.0
            aug[col][n] = 0.0
            continue

        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _ridge_fit(
    X: list[list[float]],
    y: list[float],
    lam: float = DEFAULT_RIDGE_LAMBDA,
) -> list[float]:
    """Fit ridge regression: w = (X^T X + λI)^{-1} X^T y.

    X should include a bias column (column of 1s) if intercept is desired.
    """
    Xt = _transpose(X)
    XtX = _mat_mul(Xt, X)
    n_features = len(XtX)
    reg = _scale_mat(_identity(n_features), lam)
    # Don't regularize intercept (last column if appended)
    A = _add_mat(XtX, reg)
    Xty = _mat_vec(Xt, y)
    return _solve_linear(A, Xty)


# ------------------------------------------------------------------
# Feature extraction from RuanMei observations
# ------------------------------------------------------------------

def _extract_features(obs: dict, feature_names: list[str]) -> Optional[list[float]]:
    """Extract a feature vector from a scored observation.

    Returns None if critical features are missing.
    """
    features = {}

    # Basic features always available
    desc = obs.get("descriptor", {})
    features["char_count"] = desc.get("char_count", 0) or len(
        obs.get("posted_body", obs.get("post_body", ""))
    )

    # Edit similarity
    features["edit_similarity"] = obs.get("edit_similarity", -1.0)
    if features["edit_similarity"] < 0:
        features["edit_similarity"] = 1.0  # assume unedited

    # Posting time features
    posted_at = obs.get("posted_at", "")
    if posted_at:
        try:
            dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            features["posting_hour"] = dt.hour
            features["posting_day"] = dt.weekday()  # 0=Mon
        except (ValueError, TypeError):
            features["posting_hour"] = 12
            features["posting_day"] = 2
    else:
        features["posting_hour"] = 12
        features["posting_day"] = 2

    # Alignment score (from draft_map backfill)
    features["alignment_score"] = obs.get("alignment_score", 0.5)
    if features["alignment_score"] is None:
        features["alignment_score"] = 0.5

    # Cyrene dimension scores
    cyrene_dims = obs.get("cyrene_dimensions", {})
    if isinstance(cyrene_dims, dict):
        for k, v in cyrene_dims.items():
            features[f"cyrene_{k}"] = v if isinstance(v, (int, float)) else 0.0

    # Constitutional results
    const_results = obs.get("constitutional_results", {})
    if isinstance(const_results, dict):
        for k, v in const_results.items():
            if isinstance(v, (int, float)):
                features[f"const_{k}"] = v
            elif isinstance(v, dict) and "score" in v:
                features[f"const_{k}"] = v["score"]
            elif isinstance(v, bool):
                features[f"const_{k}"] = 1.0 if v else 0.0

    # Build the vector in the requested order
    vec = []
    for name in feature_names:
        val = features.get(name)
        if val is None:
            vec.append(0.0)
        else:
            vec.append(float(val))
    return vec


def _discover_feature_names(observations: list[dict]) -> list[str]:
    """Discover all available feature names from scored observations.

    Returns a stable sorted list of feature names present in ≥30% of observations.
    """
    feature_counts: dict[str, int] = {}
    n = len(observations)

    for obs in observations:
        seen = set()

        # Always-present features
        for f in ["char_count", "edit_similarity", "posting_hour", "posting_day", "alignment_score"]:
            seen.add(f)

        # Cyrene dimensions
        cyrene_dims = obs.get("cyrene_dimensions", {})
        if isinstance(cyrene_dims, dict):
            for k in cyrene_dims:
                seen.add(f"cyrene_{k}")

        # Constitutional results
        const_results = obs.get("constitutional_results", {})
        if isinstance(const_results, dict):
            for k in const_results:
                seen.add(f"const_{k}")

        for f in seen:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    # Keep features present in ≥30% of observations (or all if few features)
    threshold = max(1, int(n * 0.3))
    names = sorted(k for k, v in feature_counts.items() if v >= threshold)
    return names


def _normalize_features(
    X: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Z-normalize features. Returns (X_normalized, means, stds)."""
    if not X:
        return X, [], []
    n_features = len(X[0])
    n_samples = len(X)

    means = [0.0] * n_features
    for row in X:
        for j in range(n_features):
            means[j] += row[j]
    means = [m / n_samples for m in means]

    stds = [0.0] * n_features
    for row in X:
        for j in range(n_features):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [math.sqrt(s / max(n_samples - 1, 1)) for s in stds]
    # Avoid division by zero
    stds = [s if s > 1e-10 else 1.0 for s in stds]

    X_norm = []
    for row in X:
        X_norm.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])

    return X_norm, means, stds


# ------------------------------------------------------------------
# Model building
# ------------------------------------------------------------------

def _load_causal_feature_classifications(company: str) -> Optional[dict]:
    """Return {feature_name: classification} from the causal filter output.

    Returns None if the causal_dimensions.json file doesn't exist. The caller
    should interpret missing features as "not evaluated" — those should be
    retained in the model, not dropped.
    """
    causal_path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not causal_path.exists():
        return None
    try:
        data = json.loads(causal_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    classifications = {}
    for d in data.get("dimensions", []):
        name = d.get("dimension", "")
        cls = d.get("classification", "")
        if name:
            classifications[name] = cls
    return classifications or None


def _apply_causal_filter(all_features: list[str], classifications: dict) -> list[str]:
    """Drop features the causal filter classified as inert or confounded.

    Features the filter didn't evaluate (not in ``classifications``) are
    RETAINED — the filter has no opinion on them, so defer to the full model.
    """
    drop_classes = {"inert", "confounded"}
    filtered = []
    for f in all_features:
        cls = classifications.get(f)
        if cls in drop_classes:
            continue  # explicitly classified as not useful
        filtered.append(f)
    return filtered


def _fit_and_evaluate(
    scored: list[dict],
    feature_names: list[str],
) -> Optional[dict]:
    """Fit ridge regression on the given feature set and return model + LOO R²."""
    X_raw: list[list[float]] = []
    y: list[float] = []
    for obs in scored:
        vec = _extract_features(obs, feature_names)
        if vec is None:
            continue
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue
        X_raw.append(vec)
        y.append(reward)

    if len(y) < MIN_OBSERVATIONS:
        return None

    X_norm, means, stds = _normalize_features(X_raw)
    X = [row + [1.0] for row in X_norm]

    coefficients = _ridge_fit(X, y, lam=DEFAULT_RIDGE_LAMBDA)
    intercept = coefficients[-1]
    feature_coefficients = coefficients[:-1]

    loo_residuals = _leave_one_out_residuals(X_norm, y, lam=DEFAULT_RIDGE_LAMBDA)

    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum(r ** 2 for r in loo_residuals)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    residual_mean = sum(loo_residuals) / len(loo_residuals)
    residual_std = math.sqrt(
        sum((r - residual_mean) ** 2 for r in loo_residuals) / max(len(loo_residuals) - 1, 1)
    )

    return {
        "feature_names": feature_names,
        "coefficients": [round(c, 6) for c in feature_coefficients],
        "intercept": round(intercept, 6),
        "means": [round(m, 6) for m in means],
        "stds": [round(s, 6) for s in stds],
        "r_squared": round(r_squared, 4),
        "residual_mean": round(residual_mean, 6),
        "residual_std": round(residual_std, 6),
        "observation_count": len(y),
    }


def build_engagement_model(company: str) -> Optional[dict]:
    """Build and cache an engagement prediction model for a client.

    When ``causal_dimensions.json`` exists (populated by the causal filter,
    B1), attempts to fit a reduced-feature model using only causal + uncertain
    dimensions. Compares LOO R² against the full-feature model and keeps
    whichever is better. Logs the comparison for auditability.

    Returns the model dict or None if insufficient data.
    """
    # Load RuanMei state
    state_path = vortex.memory_dir(company) / "ruan_mei_state.json"
    state = None
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        pass

    if state is None and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if state is None:
        return None

    scored = [o for o in state.get("observations", []) if o.get("status") == "scored"]
    if len(scored) < MIN_OBSERVATIONS:
        logger.debug(
            "[engagement_predictor] %s has %d scored obs (need %d), skipping",
            company, len(scored), MIN_OBSERVATIONS,
        )
        return None

    # Discover all candidate features
    all_features = _discover_feature_names(scored)
    if not all_features:
        logger.debug("[engagement_predictor] No features discovered for %s", company)
        return None

    # Fit the full model
    full_model = _fit_and_evaluate(scored, all_features)
    if full_model is None:
        return None

    # If causal filter output exists, try a filtered model and compare
    causal_classifications = _load_causal_feature_classifications(company)
    filtered_model = None
    filter_note = None

    if causal_classifications is not None:
        filtered_features = _apply_causal_filter(all_features, causal_classifications)
        # Only attempt the filtered fit if it actually drops some features
        if filtered_features and len(filtered_features) < len(all_features):
            filtered_model = _fit_and_evaluate(scored, filtered_features)
            if filtered_model is not None:
                dropped = sorted(set(all_features) - set(filtered_features))
                filter_note = {
                    "full_r_squared": full_model["r_squared"],
                    "filtered_r_squared": filtered_model["r_squared"],
                    "dropped_features": dropped,
                    "dropped_classifications": {
                        f: causal_classifications.get(f, "unevaluated") for f in dropped
                    },
                    "kept_features": filtered_features,
                }
                logger.info(
                    "[engagement_predictor] %s causal filter: full R²=%.4f, "
                    "filtered R²=%.4f (dropped %s)",
                    company, full_model["r_squared"], filtered_model["r_squared"],
                    dropped,
                )

    # Pick the better model. Keep the full model when the filtered one isn't
    # strictly better — don't force feature selection if it hurts accuracy.
    if filtered_model and filtered_model["r_squared"] > full_model["r_squared"]:
        chosen_model = filtered_model
        chosen_model["feature_selection"] = "causal_filter"
        chosen_model["feature_selection_note"] = filter_note
    else:
        chosen_model = full_model
        chosen_model["feature_selection"] = "full"
        if filter_note:
            # Still record the comparison even when we kept the full model
            chosen_model["feature_selection_note"] = filter_note

    chosen_model["computed_at"] = datetime.now(timezone.utc).isoformat()

    # Save
    model_path = vortex.memory_dir(company) / "engagement_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(chosen_model, indent=2), encoding="utf-8")
    tmp.rename(model_path)

    logger.info(
        "[engagement_predictor] Built model for %s: R²=%.4f, %d features, %d obs (%s)",
        company, chosen_model["r_squared"], len(chosen_model["feature_names"]),
        chosen_model["observation_count"], chosen_model["feature_selection"],
    )

    return chosen_model


def _leave_one_out_residuals(
    X_norm: list[list[float]],
    y: list[float],
    lam: float = DEFAULT_RIDGE_LAMBDA,
) -> list[float]:
    """Compute leave-one-out residuals for confidence intervals."""
    n = len(y)
    residuals = []

    for i in range(n):
        X_train = [row + [1.0] for j, row in enumerate(X_norm) if j != i]
        y_train = [y[j] for j in range(n) if j != i]

        if len(y_train) < 3:
            residuals.append(0.0)
            continue

        w = _ridge_fit(X_train, y_train, lam=lam)
        x_test = X_norm[i] + [1.0]
        pred = _dot(w, x_test)
        residuals.append(y[i] - pred)

    return residuals


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_engagement(
    company: str,
    feature_dict: dict,
) -> Optional[PredictionResult]:
    """Predict engagement for a draft post.

    Args:
        company: Client slug.
        feature_dict: Dict of feature_name -> value. Missing features use 0.

    Returns PredictionResult or None if no model available.
    """
    model_path = vortex.memory_dir(company) / "engagement_model.json"
    if not model_path.exists():
        return None

    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    feature_names = model["feature_names"]
    coefficients = model["coefficients"]
    intercept = model["intercept"]
    means = model["means"]
    stds = model["stds"]
    residual_std = model["residual_std"]

    if len(coefficients) != len(feature_names):
        logger.warning("[engagement_predictor] Model corrupted for %s", company)
        return None

    # Build normalized feature vector
    raw_vec = [float(feature_dict.get(f, 0.0)) for f in feature_names]
    norm_vec = [(raw_vec[j] - means[j]) / stds[j] if stds[j] > 1e-10 else 0.0
                for j in range(len(feature_names))]

    # Predict
    predicted = _dot(coefficients, norm_vec) + intercept

    # Confidence interval (±1.96 * residual_std for ~95%)
    ci_half = 1.96 * residual_std
    confidence_low = predicted - ci_half
    confidence_high = predicted + ci_half

    # Feature contributions (coefficient * normalized_value)
    contributions = []
    for i, fname in enumerate(feature_names):
        contrib = coefficients[i] * norm_vec[i]
        contributions.append({"feature": fname, "contribution": round(contrib, 4)})

    # Sort by contribution
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    top_positive = [c for c in contributions if c["contribution"] > 0][:3]
    top_negative = [c for c in contributions if c["contribution"] < 0][-3:]
    top_negative.reverse()  # most negative first

    return PredictionResult(
        predicted_reward=round(predicted, 4),
        confidence_low=round(confidence_low, 4),
        confidence_high=round(confidence_high, 4),
        top_positive=top_positive,
        top_negative=top_negative,
        model_r_squared=model["r_squared"],
        observation_count=model["observation_count"],
    )
