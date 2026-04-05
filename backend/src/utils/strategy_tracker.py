"""Strategy brief impact tracker — measure whether data-informed posts perform better.

The learning pipeline produces a strategy brief for each client, and Stelle
injects a truncated version into its dynamic directives. This module answers
the meta-question: does data-informed generation actually produce better posts?

It's observation-only. It does not change any agent behavior based on the
result. The goal is to accumulate enough evidence to validate (or falsify)
the hypothesis that strategy briefs improve post quality, at which point a
future session can decide whether to double down on the injection or
retire it.

Partitioning logic:
- Posts with ``strategy_brief_version`` set → "informed" group
- Posts without (legacy observations or posts generated before the brief
  pipeline existed) → "uninformed" group

The comparison is only reported once both groups have at least 20
observations. Below that, the difference is statistical noise.

Usage:
    from backend.src.utils.strategy_tracker import compute_strategy_brief_impact

    # During ordinal_sync (cross-client step):
    result = compute_strategy_brief_impact()
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_MIN_OBS_PER_GROUP = 20  # below this the comparison is underpowered


def compute_strategy_brief_impact() -> Optional[dict]:
    """Walk all clients, partition observations, and compute the impact metric.

    Writes the result to ``memory/our_memory/strategy_brief_impact.json``.
    Returns the comparison dict, or a dict with ``sufficient_data=False`` when
    either group is below the 20-observation threshold.
    """
    memory_root = vortex.MEMORY_ROOT
    if not memory_root.exists():
        return None

    informed_rewards: list[float] = []
    uninformed_rewards: list[float] = []
    per_client_counts: dict = {}

    for company_dir in sorted(memory_root.iterdir()):
        if not company_dir.is_dir():
            continue
        if company_dir.name.startswith(".") or company_dir.name == "our_memory":
            continue

        state = _load_ruan_mei_state(company_dir.name)
        if state is None:
            continue

        client_informed = 0
        client_uninformed = 0

        for obs in state.get("observations", []):
            if obs.get("status") != "scored":
                continue
            reward = obs.get("reward", {}).get("immediate")
            if reward is None:
                continue

            if obs.get("strategy_brief_version"):
                informed_rewards.append(float(reward))
                client_informed += 1
            else:
                uninformed_rewards.append(float(reward))
                client_uninformed += 1

        if client_informed or client_uninformed:
            per_client_counts[company_dir.name] = {
                "informed": client_informed,
                "uninformed": client_uninformed,
            }

    n_informed = len(informed_rewards)
    n_uninformed = len(uninformed_rewards)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: list[float], mean: float) -> float:
        if len(xs) < 2:
            return 0.0
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))

    mean_informed = _mean(informed_rewards)
    mean_uninformed = _mean(uninformed_rewards)
    std_informed = _std(informed_rewards, mean_informed)
    std_uninformed = _std(uninformed_rewards, mean_uninformed)
    delta = mean_informed - mean_uninformed

    sufficient = n_informed >= _MIN_OBS_PER_GROUP and n_uninformed >= _MIN_OBS_PER_GROUP

    # Cohen's d with pooled SD when both arms have sufficient data
    cohens_d: Optional[float] = None
    if n_informed >= 2 and n_uninformed >= 2:
        pooled_sd = math.sqrt(
            ((n_informed - 1) * std_informed ** 2 + (n_uninformed - 1) * std_uninformed ** 2)
            / max(n_informed + n_uninformed - 2, 1)
        )
        if pooled_sd > 1e-9:
            cohens_d = delta / pooled_sd

    result = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "sufficient_data": sufficient,
        "min_obs_per_group": _MIN_OBS_PER_GROUP,
        "informed": {
            "count": n_informed,
            "mean_reward": round(mean_informed, 4),
            "std_reward": round(std_informed, 4),
        },
        "uninformed": {
            "count": n_uninformed,
            "mean_reward": round(mean_uninformed, 4),
            "std_reward": round(std_uninformed, 4),
        },
        "delta_mean_reward": round(delta, 4),
        "cohens_d": round(cohens_d, 4) if cohens_d is not None else None,
        "per_client_counts": per_client_counts,
        "interpretation": _interpret(sufficient, delta, cohens_d),
    }

    # Persist
    out_path = vortex.our_memory_dir() / "strategy_brief_impact.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(out_path)

    if sufficient:
        logger.info(
            "[strategy_tracker] Impact: informed n=%d mean=%.3f vs "
            "uninformed n=%d mean=%.3f (Δ=%+.3f, d=%s)",
            n_informed, mean_informed, n_uninformed, mean_uninformed,
            delta, f"{cohens_d:+.3f}" if cohens_d is not None else "n/a",
        )
    else:
        logger.info(
            "[strategy_tracker] Insufficient data: informed=%d, uninformed=%d "
            "(need ≥%d per group)",
            n_informed, n_uninformed, _MIN_OBS_PER_GROUP,
        )

    return result


def _interpret(sufficient: bool, delta: float, cohens_d: Optional[float]) -> str:
    """Plain-English interpretation of the comparison. Conservative by design."""
    if not sufficient:
        return (
            "Insufficient data. At least 20 observations per group are required "
            "before the comparison is meaningful. Strategy brief impact cannot "
            "yet be evaluated."
        )
    if cohens_d is None:
        return "Insufficient variance to compute effect size."
    if cohens_d >= 0.2:
        return (
            f"Data-informed posts outperform uninformed posts by "
            f"d={cohens_d:+.3f} (small-to-medium positive effect). "
            "Strategy brief injection appears beneficial."
        )
    if cohens_d <= -0.2:
        return (
            f"Data-informed posts underperform uninformed posts by "
            f"d={cohens_d:+.3f} (small-to-medium negative effect). "
            "Strategy brief injection may be harmful — investigate before doubling down."
        )
    return (
        f"Effect size d={cohens_d:+.3f} is below the 0.2 threshold. "
        "Strategy brief injection has no detectable effect on engagement yet."
    )


def _load_ruan_mei_state(company: str) -> Optional[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
        if state is not None:
            return state
    except Exception:
        pass

    path = vortex.ruan_mei_state_path(company)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
