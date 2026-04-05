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

def build_engagement_model(company: str) -> Optional[dict]:
    """Build and cache an engagement prediction model for a client.

    Returns the model dict or None if insufficient data.
    """
    # Load RuanMei state
    state_path = vortex.memory_dir(company) / "ruan_mei_state.json"
    state = None

    # Try SQLite first, fall back to JSON
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

    # Discover features
    feature_names = _discover_feature_names(scored)
    if not feature_names:
        logger.debug("[engagement_predictor] No features discovered for %s", company)
        return None

    # Build X, y matrices
    X_raw = []
    y = []
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

    # Normalize features
    X_norm, means, stds = _normalize_features(X_raw)

    # Add intercept column
    X = [row + [1.0] for row in X_norm]

    # Fit ridge regression
    coefficients = _ridge_fit(X, y, lam=DEFAULT_RIDGE_LAMBDA)

    # Intercept is the last coefficient
    intercept = coefficients[-1]
    feature_coefficients = coefficients[:-1]

    # Compute R² and residual stats via leave-one-out
    loo_residuals = _leave_one_out_residuals(X_norm, y, lam=DEFAULT_RIDGE_LAMBDA)

    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum(r ** 2 for r in loo_residuals)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    residual_mean = sum(loo_residuals) / len(loo_residuals)
    residual_std = math.sqrt(
        sum((r - residual_mean) ** 2 for r in loo_residuals) / max(len(loo_residuals) - 1, 1)
    )

    model = {
        "feature_names": feature_names,
        "coefficients": [round(c, 6) for c in feature_coefficients],
        "intercept": round(intercept, 6),
        "means": [round(m, 6) for m in means],
        "stds": [round(s, 6) for s in stds],
        "r_squared": round(r_squared, 4),
        "residual_mean": round(residual_mean, 6),
        "residual_std": round(residual_std, 6),
        "observation_count": len(y),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save
    model_path = vortex.memory_dir(company) / "engagement_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=2), encoding="utf-8")
    tmp.rename(model_path)

    logger.info(
        "[engagement_predictor] Built model for %s: R²=%.4f, %d features, %d obs",
        company, r_squared, len(feature_names), len(y),
    )

    return model


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
