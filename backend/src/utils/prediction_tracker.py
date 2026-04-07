"""Prediction accuracy tracker — does the model actually get better?

After each post is scored with actual engagement, compares the draft
scorer's predicted engagement (recorded at generation time) against the
actual reward. Over time, this produces the validation signal:

  - Is the model's Spearman correlation with actual outcomes > 0?
  - Is prediction accuracy improving as data grows?
  - Which clients' models are trustworthy vs garbage?

Without this, the system has LOO R² from the analyst's training run —
a retrospective estimate. This module tracks TRUE prediction accuracy on
posts the model scored BEFORE they were published. That's the real test.

Storage: memory/{company}/prediction_accuracy.json

Usage:
    from backend.src.utils.prediction_tracker import update_prediction_accuracy

    # Called from ordinal_sync after observations are scored:
    result = update_prediction_accuracy("innovocommerce")
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)


def update_prediction_accuracy(company: str) -> Optional[dict]:
    """Compute prediction accuracy from observations that have both a
    predicted_engagement and an actual scored reward.

    Returns the accuracy report dict, or None if no predictions exist.
    Persists to memory/{company}/prediction_accuracy.json.
    """
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return None
    if state is None:
        return None

    # Find observations that have BOTH a prediction and an actual reward
    pairs = []
    for obs in state.get("observations", []):
        pred = obs.get("predicted_engagement")
        reward = obs.get("reward")
        actual = reward.get("immediate") if isinstance(reward, dict) else None
        if pred is not None and actual is not None and obs.get("status") == "scored":
            pairs.append({
                "post_hash": obs.get("post_hash", "")[:8],
                "predicted": pred,
                "actual": actual,
                "error": round(pred - actual, 4),
                "abs_error": round(abs(pred - actual), 4),
                "scored_at": obs.get("scored_at", ""),
            })

    if not pairs:
        return {
            "company": company,
            "n_predictions": 0,
            "status": "no_predictions",
            "message": (
                "No posts have been generated with prediction tracking active. "
                "Generate posts through Stelle to populate predicted_engagement."
            ),
        }

    # Compute accuracy metrics
    predicted = [p["predicted"] for p in pairs]
    actual = [p["actual"] for p in pairs]
    errors = [p["error"] for p in pairs]
    abs_errors = [p["abs_error"] for p in pairs]

    mean_error = sum(errors) / len(errors)
    mean_abs_error = sum(abs_errors) / len(abs_errors)

    # Spearman rank correlation between predictions and actuals
    spearman = 0.0
    if len(pairs) >= 3:
        try:
            from backend.src.utils.correlation_analyzer import _spearman_correlation
            spearman = _spearman_correlation(predicted, actual)
        except Exception:
            pass

    # Trend: is accuracy improving? Compare first half vs second half
    trend = "insufficient_data"
    if len(pairs) >= 6:
        mid = len(pairs) // 2
        early_errors = [p["abs_error"] for p in pairs[:mid]]
        late_errors = [p["abs_error"] for p in pairs[mid:]]
        early_mae = sum(early_errors) / len(early_errors)
        late_mae = sum(late_errors) / len(late_errors)
        if late_mae < early_mae * 0.8:
            trend = "improving"
        elif late_mae > early_mae * 1.2:
            trend = "degrading"
        else:
            trend = "stable"

    report = {
        "company": company,
        "n_predictions": len(pairs),
        "spearman": round(spearman, 4),
        "mean_error": round(mean_error, 4),
        "mean_abs_error": round(mean_abs_error, 4),
        "trend": trend,
        "pairs": pairs,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "interpretation": _interpret(len(pairs), spearman, trend),
    }

    # Persist
    path = vortex.memory_dir(company) / "prediction_accuracy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)

    logger.info(
        "[prediction_tracker] %s: %d predictions, Spearman=%.3f, MAE=%.3f, trend=%s",
        company, len(pairs), spearman, mean_abs_error, trend,
    )

    return report


def _interpret(n: int, spearman: float, trend: str) -> str:
    if n < 3:
        return (
            f"Only {n} predictions tracked. Need at least 10 to assess model quality. "
            "Generate more posts through Stelle to accumulate prediction data."
        )
    if n < 10:
        return (
            f"{n} predictions tracked (Spearman={spearman:+.3f}). Still early — "
            "need 10+ for a reliable assessment. Trend: {trend}."
        )
    if spearman > 0.3:
        return (
            f"Model is predictive (Spearman={spearman:+.3f}, n={n}). "
            "Predictions rank posts in approximately the right order. "
            f"Trend: {trend}."
        )
    if spearman > 0.1:
        return (
            f"Model has weak signal (Spearman={spearman:+.3f}, n={n}). "
            "Better than random but not reliable for individual post scoring. "
            f"Trend: {trend}."
        )
    if spearman > -0.1:
        return (
            f"Model is not predictive (Spearman={spearman:+.3f}, n={n}). "
            "Predictions are essentially random. The model needs more data "
            "or different features. "
            f"Trend: {trend}."
        )
    return (
        f"Model is anti-predictive (Spearman={spearman:+.3f}, n={n}). "
        "The model's predictions are inversely correlated with actual outcomes. "
        "Something is wrong — investigate the model's assumptions. "
        f"Trend: {trend}."
    )
