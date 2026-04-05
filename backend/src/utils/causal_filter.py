"""Causal dimension filter — partial correlation analysis for content state.

For each content state dimension (topic_tag, format_tag, posting_day,
hours_since_last_post, cyrene_dimension_scores, edit_similarity, char_count),
test whether it predicts engagement *after controlling for all other observed
dimensions*. Inspired by PGCR (Partial-Grouped Causal Ranking, ACM 2025,
arXiv:2502.02327) but implemented as a pragmatic approximation.

**Important caveat:** partial correlation is NOT true causal inference. It
controls for *observed* confounders only. Unobserved confounders can still
drive the correlation. Treat the output as "dimensions that predict
engagement after controlling for other measured factors," not "dimensions
that cause engagement."

Classification:
- ``causal``: |partial_correlation| >= 0.15 and p < 0.10 (permutation test)
- ``confounded``: |marginal_correlation| >= 0.15 but |partial_correlation| < 0.10
  (marginal signal vanishes once other factors are controlled for)
- ``inert``: neither marginal nor partial correlation reaches the threshold
- ``uncertain``: middling values — preserved as features but not promoted

Categorical dimensions (topic_tag, format_tag) are encoded via target
encoding: each category is replaced by its mean reward (leave-one-out to
avoid target leakage). This is a lossy but tractable encoding for small n.

Requires 30+ tagged observations. Below this, the test is too noisy.

Usage:
    from backend.src.utils.causal_filter import (
        compute_causal_dimensions,
        get_causal_dimensions,
    )

    # During ordinal_sync (after A1 tagging):
    compute_causal_dimensions("innovocommerce")

    # Read results:
    dims = get_causal_dimensions("innovocommerce")
"""

import json
import logging
import math
import random
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_MIN_OBS_FOR_CAUSAL = 30
_PERMUTATION_SHUFFLES = 500
_CAUSAL_THRESHOLD = 0.15    # |partial correlation| above this → causal candidate
_P_VALUE_THRESHOLD = 0.10   # permutation test significance
_CONFOUNDED_PARTIAL_MAX = 0.10  # partial below this while marginal is high → confounded


# ------------------------------------------------------------------
# Linear algebra primitives (stdlib math, no numpy dependency in the hot path)
# ------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _rank(values: list[float]) -> list[float]:
    """Average-tied ranks, 1-indexed."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rank(x), _rank(y))


def _solve_normal_equations(X: list[list[float]], y: list[float],
                             ridge: float = 1e-4) -> list[float]:
    """Solve (X'X + ridge*I) w = X'y for small dense systems via Gauss-Jordan.

    Returns coefficient vector. Ridge term stabilizes against near-singular
    covariance matrices (common when some features are effectively constant).
    """
    n = len(X)
    if n == 0:
        return []
    p = len(X[0])

    # Build X'X
    xtx = [[0.0] * p for _ in range(p)]
    for row in X:
        for i in range(p):
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    for i in range(p):
        xtx[i][i] += ridge

    # Build X'y
    xty = [0.0] * p
    for row, yi in zip(X, y):
        for i in range(p):
            xty[i] += row[i] * yi

    # Gauss-Jordan with partial pivoting
    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(p):
        pivot_row = col
        for r in range(col + 1, p):
            if abs(aug[r][col]) > abs(aug[pivot_row][col]):
                pivot_row = r
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        for c in range(col, p + 1):
            aug[col][c] /= pivot
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            for c in range(col, p + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][p] for i in range(p)]


def _residualize(target: list[float], controls: list[list[float]]) -> list[float]:
    """Regress ``target`` on ``controls`` (linear) and return residuals.

    Controls should include a leading column of 1s for intercept.
    """
    if not controls or not controls[0]:
        return list(target)
    w = _solve_normal_equations(controls, target)
    residuals = []
    for row, t in zip(controls, target):
        pred = sum(wi * xi for wi, xi in zip(w, row))
        residuals.append(t - pred)
    return residuals


# ------------------------------------------------------------------
# Feature matrix construction
# ------------------------------------------------------------------

def _target_encode(values: list, rewards: list[float]) -> list[float]:
    """Leave-one-out target encoding for categorical values.

    Replaces each category with the mean reward of the OTHER observations
    in that category, avoiding target leakage. Unseen categories get the
    global mean.
    """
    n = len(values)
    global_mean = sum(rewards) / n if n else 0.0

    # Group sums and counts per category
    sums: dict = {}
    counts: dict = {}
    for v, r in zip(values, rewards):
        sums[v] = sums.get(v, 0.0) + r
        counts[v] = counts.get(v, 0) + 1

    encoded = []
    for v, r in zip(values, rewards):
        c = counts[v]
        if c <= 1:
            encoded.append(global_mean)
        else:
            # Leave-one-out mean
            encoded.append((sums[v] - r) / (c - 1))
    return encoded


def _build_feature_matrix(observations: list[dict]) -> tuple[list[str], list[list[float]], list[float]]:
    """Build a numeric feature matrix from tagged observations.

    Returns (feature_names, X, y) where:
    - feature_names is the list of dimension names
    - X is the n × p feature matrix (list of rows)
    - y is the reward vector
    """
    # Extract raw dimensions per observation
    rows: list[dict] = []
    rewards: list[float] = []
    for obs in observations:
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue

        posted_at = obs.get("posted_at") or obs.get("recorded_at", "")
        day = 3  # default Wednesday
        hour = 12
        if posted_at:
            try:
                dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                day = dt.weekday()
                hour = dt.hour
            except Exception:
                pass

        row = {
            "topic_tag": obs.get("topic_tag") or "__none__",
            "format_tag": obs.get("format_tag") or "__none__",
            "posting_day": float(day),
            "posting_hour": float(hour),
            "char_count": float(len(obs.get("posted_body") or obs.get("post_body") or "")),
            "edit_similarity": float(obs.get("edit_similarity", 1.0) if obs.get("edit_similarity", -1) >= 0 else 1.0),
        }
        # Cyrene dimension scores if present (flatten with prefix)
        cdims = obs.get("cyrene_dimensions") or {}
        if isinstance(cdims, dict):
            for k, v in cdims.items():
                if isinstance(v, (int, float)):
                    row[f"cyrene_{k}"] = float(v)

        rows.append(row)
        rewards.append(float(reward))

    if not rows:
        return [], [], []

    # Discover all feature keys present in >= 30% of observations
    key_counts: dict = {}
    for row in rows:
        for k in row:
            key_counts[k] = key_counts.get(k, 0) + 1
    threshold = max(1, int(len(rows) * 0.3))
    feature_names = sorted(k for k, c in key_counts.items() if c >= threshold)

    # Build X. Target-encode categorical columns.
    categorical = {"topic_tag", "format_tag"}
    X: list[list[float]] = [[] for _ in rows]

    for key in feature_names:
        if key in categorical:
            col_values = [row.get(key, "__none__") for row in rows]
            encoded = _target_encode(col_values, rewards)
        else:
            col_values = [float(row.get(key, 0.0)) for row in rows]
            # Standardize numeric columns so ridge regularization is comparable
            m = _mean(col_values)
            sd = math.sqrt(sum((v - m) ** 2 for v in col_values) / max(len(col_values) - 1, 1))
            if sd < 1e-9:
                encoded = [0.0] * len(col_values)
            else:
                encoded = [(v - m) / sd for v in col_values]
        for i, v in enumerate(encoded):
            X[i].append(v)

    return feature_names, X, rewards


# ------------------------------------------------------------------
# Partial correlation with permutation test
# ------------------------------------------------------------------

def _partial_correlation(X: list[list[float]], y: list[float],
                          feature_idx: int) -> float:
    """Partial correlation of feature ``feature_idx`` with y, controlling for all others."""
    if not X or len(X[0]) < 2:
        return _pearson([row[feature_idx] for row in X], y)

    # Target column
    target_col = [row[feature_idx] for row in X]

    # Control columns: all other features + intercept
    control_cols = []
    for row in X:
        ctrl = [1.0]
        for j in range(len(row)):
            if j != feature_idx:
                ctrl.append(row[j])
        control_cols.append(ctrl)

    # Residualize both target and y on controls, then correlate residuals
    resid_target = _residualize(target_col, control_cols)
    resid_y = _residualize(y, control_cols)
    return _pearson(resid_target, resid_y)


def _permutation_pvalue(X: list[list[float]], y: list[float], feature_idx: int,
                         observed: float, shuffles: int = _PERMUTATION_SHUFFLES,
                         seed: int = 42) -> float:
    """Two-sided permutation p-value for a partial correlation.

    Shuffles y and recomputes partial correlation. p-value = fraction of
    shuffles where |shuffled| >= |observed|.
    """
    rng = random.Random(seed)
    n = len(y)
    if n < 3:
        return 1.0

    extreme_count = 0
    abs_observed = abs(observed)

    y_list = list(y)
    for _ in range(shuffles):
        rng.shuffle(y_list)
        shuffled = _partial_correlation(X, y_list, feature_idx)
        if abs(shuffled) >= abs_observed:
            extreme_count += 1

    # Laplace smoothing to avoid p=0
    return (extreme_count + 1) / (shuffles + 1)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def compute_causal_dimensions(company: str, force: bool = False) -> Optional[dict]:
    """Compute marginal + partial correlations for all content state dimensions.

    Returns a dict with the classification per dimension and summary stats,
    or None if insufficient data. Persists to
    ``memory/{company}/causal_dimensions.json``.

    Args:
        company: Client slug.
        force: If True, bypass the observation-count cache check and recompute.
    """
    state = _load_state(company)
    if state is None:
        return None

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("topic_tag")
        and o.get("format_tag")
        and o.get("reward", {}).get("immediate") is not None
    ]

    if len(scored) < _MIN_OBS_FOR_CAUSAL:
        logger.debug(
            "[causal_filter] %s has %d tagged obs (need %d)",
            company, len(scored), _MIN_OBS_FOR_CAUSAL,
        )
        return None

    # Cache short-circuit: if the file exists and the observation count hasn't
    # changed since the last run, return the cached result. The causal filter
    # is deterministic on input data (fixed permutation seed), so recomputing
    # on the same inputs is pure waste — 500 permutations × 6 features adds up
    # across 22+ clients at hourly sync intervals.
    cached_path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not force and cached_path.exists():
        try:
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if cached.get("observation_count") == len(scored):
                logger.debug(
                    "[causal_filter] %s cache hit (n=%d unchanged)",
                    company, len(scored),
                )
                return cached
        except Exception:
            pass

    feature_names, X, y = _build_feature_matrix(scored)
    if not feature_names:
        return None

    results: list[dict] = []

    for idx, fname in enumerate(feature_names):
        col = [row[idx] for row in X]

        marginal = _spearman(col, y)
        partial = _partial_correlation(X, y, idx)

        # Permutation p-value on the partial correlation
        p_value = _permutation_pvalue(X, y, idx, partial)

        # Classify
        abs_partial = abs(partial)
        abs_marginal = abs(marginal)

        if abs_partial >= _CAUSAL_THRESHOLD and p_value < _P_VALUE_THRESHOLD:
            classification = "causal"
        elif abs_marginal >= _CAUSAL_THRESHOLD and abs_partial < _CONFOUNDED_PARTIAL_MAX:
            classification = "confounded"
        elif abs_partial >= _CAUSAL_THRESHOLD:
            classification = "uncertain"  # partial strong but p-value didn't pass
        elif abs_partial >= _CONFOUNDED_PARTIAL_MAX:
            classification = "uncertain"
        else:
            classification = "inert"

        results.append({
            "dimension": fname,
            "marginal_correlation": round(marginal, 4),
            "partial_correlation": round(partial, 4),
            "p_value": round(p_value, 4),
            "classification": classification,
        })

    # Sort: causal first by |partial|, then uncertain, then confounded, then inert
    order = {"causal": 0, "uncertain": 1, "confounded": 2, "inert": 3}
    results.sort(
        key=lambda d: (order.get(d["classification"], 4), -abs(d["partial_correlation"]))
    )

    # Top-3 causal for summary
    top_causal = [
        d for d in results if d["classification"] == "causal"
    ][:3]

    output = {
        "company": company,
        "observation_count": len(scored),
        "feature_count": len(feature_names),
        "dimensions": results,
        "top_causal": top_causal,
        "thresholds": {
            "causal_threshold": _CAUSAL_THRESHOLD,
            "p_value_threshold": _P_VALUE_THRESHOLD,
            "permutation_shuffles": _PERMUTATION_SHUFFLES,
        },
        "disclaimer": (
            "Partial correlation controls for observed confounders only. "
            "This is not true causal inference — unobserved confounders can "
            "still drive the correlation. Treat as 'predictive after controlling "
            "for other measured factors', not 'causal'."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    cached_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cached_path)

    causal_count = sum(1 for d in results if d["classification"] == "causal")
    confounded_count = sum(1 for d in results if d["classification"] == "confounded")
    logger.info(
        "[causal_filter] %s (n=%d, %d dimensions): %d causal, %d confounded, %d inert",
        company, len(scored), len(feature_names),
        causal_count, confounded_count,
        sum(1 for d in results if d["classification"] == "inert"),
    )

    return output


def get_causal_dimensions(company: str) -> Optional[list[dict]]:
    """Load cached causal dimension results for a client."""
    path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("dimensions", [])
    except Exception:
        return None


def _load_state(company: str) -> Optional[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
        if state is not None:
            return state
    except Exception:
        pass

    path = vortex.memory_dir(company) / "ruan_mei_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
