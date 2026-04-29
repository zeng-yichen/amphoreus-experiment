"""AdaptiveConfig — three-tier cascade for data-driven thresholds.

Generalized pattern: per-client data → cross-client aggregate → hard-coded default.
When insufficient data exists, resolves to exactly the current hard-coded values.
Zero behavioral change on day one.

Usage:
    class MyConfig(AdaptiveConfig):
        MODULE_NAME = "my_module"
        def get_defaults(self) -> dict: return {"threshold": 0.5}
        def sufficient_data(self, company: str) -> bool: ...
        def compute_from_client(self, company: str) -> dict: ...
        def compute_from_aggregate(self) -> dict: ...

    cfg = MyConfig()
    params = cfg.resolve("hensley-biostats")
    # → client-specific if enough data, else aggregate, else {"threshold": 0.5}
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_DEFAULT_RECOMPUTE_INTERVAL = 3600  # 1 hour


class AdaptiveConfig(ABC):
    """Base class for data-driven configuration with three-tier cascade."""

    MODULE_NAME: str = "base"  # Override in subclass — used as cache key

    def __init__(self, recompute_interval: int = _DEFAULT_RECOMPUTE_INTERVAL):
        self._recompute_interval = recompute_interval

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get_defaults(self) -> dict:
        """Return the hard-coded default values (current behavior)."""
        ...

    @abstractmethod
    def sufficient_data(self, company: str) -> bool:
        """Return True if the client has enough data for per-client config."""
        ...

    @abstractmethod
    def compute_from_client(self, company: str) -> dict:
        """Compute config from this client's own data."""
        ...

    @abstractmethod
    def compute_from_aggregate(self) -> dict:
        """Compute config from cross-client aggregate data.

        Return empty dict if aggregate data is also insufficient.
        """
        ...

    # ------------------------------------------------------------------
    # Resolve: three-tier cascade
    # ------------------------------------------------------------------

    def resolve(self, company: str) -> dict:
        """Three-tier cascade: client → aggregate → defaults.

        Checks cache first. Recomputes if stale or missing.
        """
        cached = self._load_cached(company)
        if cached is not None:
            return cached

        if self.sufficient_data(company):
            try:
                config = self.compute_from_client(company)
                config["_tier"] = "client"
                config["_computed_at"] = _now()
                self._save_cached(company, config)
                self._log_computation(company, config)
                logger.info("[%s] Resolved from client data for %s", self.MODULE_NAME, company)
                return config
            except Exception as e:
                logger.warning("[%s] Client compute failed for %s: %s", self.MODULE_NAME, company, e)

        try:
            agg = self.compute_from_aggregate()
            if agg:
                agg["_tier"] = "aggregate"
                agg["_computed_at"] = _now()
                self._save_cached(company, agg)
                self._log_computation(company, agg)
                logger.info("[%s] Resolved from aggregate for %s", self.MODULE_NAME, company)
                return agg
        except Exception as e:
            logger.warning("[%s] Aggregate compute failed: %s", self.MODULE_NAME, e)

        defaults = self.get_defaults()
        defaults["_tier"] = "default"
        defaults["_computed_at"] = _now()
        return defaults

    def recompute(self, company: str) -> dict:
        """Force recompute, ignoring cache."""
        self._invalidate_cache(company)
        return self.resolve(company)

    # ------------------------------------------------------------------
    # Cache — persisted to memory/{company}/adaptive_config.json
    # ------------------------------------------------------------------

    def _config_path(self, company: str) -> Path:
        return P.memory_dir(company) / "adaptive_config.json"

    def _load_cached(self, company: str) -> dict | None:
        path = self._config_path(company)
        if not path.exists():
            return None
        try:
            all_configs = json.loads(path.read_text(encoding="utf-8"))
            entry = all_configs.get(self.MODULE_NAME)
            if entry is None:
                return None

            # Check staleness
            computed_at = entry.get("_computed_at", "")
            if computed_at:
                try:
                    dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
                    age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
                    if age_seconds > self._recompute_interval:
                        return None  # Stale
                except (ValueError, TypeError):
                    return None

            return entry
        except Exception:
            return None

    def _save_cached(self, company: str, config: dict) -> None:
        # 2026-04-29: removed fly-local cache write. Adaptive config
        # is recomputed on each ``resolve()`` call from observations
        # (which live in ruan_mei_state). The disk cache was an
        # optimization; recompute is cheap.
        return None

    def _invalidate_cache(self, company: str) -> None:
        return None


    # ------------------------------------------------------------------
    # History log — REMOVED 2026-04-29 (was append-only JSONL on the
    # fly-local volume; nothing read it back, audit-trail-only.)
    # ------------------------------------------------------------------

    def _log_computation(self, company: str, config: dict) -> None:
        return None


def soft_bound(value: float, history: list[float], default: float, z_threshold: float = 3.0) -> float:
    """Accept any value within z_threshold standard deviations of historical mean.

    If history is empty or too short, accept any value.
    Log a warning (don't clip) if outside range.
    Always returns the learned value — never clips.
    """
    if len(history) < 5:
        return value
    mean = sum(history) / len(history)
    std = (sum((x - mean) ** 2 for x in history) / len(history)) ** 0.5
    if std == 0:
        return value
    z = abs(value - mean) / std
    if z > z_threshold:
        logger.warning(
            "Learned value %.4f is %.1f std from mean (%.4f ± %.4f)",
            value, z, mean, std,
        )
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
