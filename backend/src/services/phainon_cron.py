"""Weekly Phainon cron — Sunday 23:00 UTC.

Runs ``phainon.run_phainon()`` once a week so the 6-creator prototype
produces a fresh batch of exemplars heading into Monday's content
sessions. Same opt-in / lifecycle posture as the other crons:

  ``ENABLE_PHAINON_PROTOTYPE=true``  — turn the loop on
  ``PHAINON_RUN_DOW``                — UTC weekday 0=Mon..6=Sun
                                        (default 6 = Sunday)
  ``PHAINON_RUN_HOUR``               — UTC hour 0-23 (default 23)

A weekly run with ~6 creators × ~20 candidates / candidate scoring is
~$20/week of Opus spend. See ``phainon.py`` for the cost breakdown
per mode.

Manual invocation (no cron, no env flag):
    python -m backend.src.services.phainon
    python -m backend.src.services.phainon --handle markdavidhensley
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_task: Optional[asyncio.Task] = None


def _env_flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("true", "1", "yes")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        if lo <= v <= hi:
            return v
    except ValueError:
        pass
    logger.warning("Invalid %s=%r; falling back to %d", name, raw, default)
    return default


def _next_fire(now: datetime, dow: int, hour: int) -> datetime:
    """Next datetime > now where ``weekday() == dow`` and hour-of-day == ``hour``.

    All math in UTC. Caps at 8 days of forward search (should only ever
    take 0 or 7, never more — guard against logic bugs).
    """
    candidate = datetime(now.year, now.month, now.day, hour, 0, 0, tzinfo=timezone.utc)
    if candidate <= now:
        candidate += timedelta(days=1)
    for _ in range(8):
        if candidate.weekday() == dow and candidate > now:
            return candidate
        candidate += timedelta(days=1)
    # Fallback — shouldn't be reachable
    return now + timedelta(days=7)


async def _run_once() -> None:
    """One scheduled Phainon pass. Threadpool-executed because the Anthropic
    SDK blocks; we don't want to freeze the event loop for 5+ minutes."""
    loop = asyncio.get_running_loop()
    try:
        from backend.src.services.phainon import run_phainon
        result = await loop.run_in_executor(None, run_phainon, None)
        logger.info(
            "[phainon_cron] processed=%d skipped=%d candidates=%d scored=%d "
            "cost=$%.2f duration=%.1fs errors=%d",
            result.creators_processed, result.creators_skipped,
            result.candidates_total, result.candidates_scored,
            result.cost_usd, result.duration_seconds, len(result.errors),
        )
    except Exception:
        logger.exception("[phainon_cron] run raised (non-fatal)")


async def _loop(dow: int, hour: int) -> None:
    logger.info("[phainon_cron] started (DOW=%d hour=%d UTC)", dow, hour)
    try:
        while True:
            now = datetime.now(timezone.utc)
            nxt = _next_fire(now, dow, hour)
            wait_s = max(1.0, (nxt - now).total_seconds())
            logger.info(
                "[phainon_cron] next fire at %s (in %.1fh)",
                nxt.isoformat(), wait_s / 3600,
            )
            await asyncio.sleep(wait_s)
            await _run_once()
    except asyncio.CancelledError:
        logger.info("[phainon_cron] cancelled, exiting")
        raise
    except Exception:
        logger.exception("[phainon_cron] loop exited with unexpected error")


def start_phainon_cron() -> None:
    """Idempotent. Opt-in via ``ENABLE_PHAINON_PROTOTYPE=true``."""
    global _task
    if not _env_flag_on("ENABLE_PHAINON_PROTOTYPE"):
        logger.info(
            "phainon_cron disabled "
            "(set ENABLE_PHAINON_PROTOTYPE=true to enable)."
        )
        return
    if _task is not None and not _task.done():
        logger.debug("phainon_cron already running; skipping start.")
        return

    dow  = _env_int("PHAINON_RUN_DOW",  6, 0, 6)   # default Sunday
    hour = _env_int("PHAINON_RUN_HOUR", 23, 0, 23) # default 23:00 UTC

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("phainon_cron: no running event loop; cannot start.")
        return

    _task = loop.create_task(_loop(dow, hour), name="phainon_cron")


def stop_phainon_cron() -> None:
    global _task
    if _task is None:
        return
    if not _task.done():
        _task.cancel()
    _task = None
