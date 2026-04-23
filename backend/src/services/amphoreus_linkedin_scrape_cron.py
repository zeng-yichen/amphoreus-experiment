"""Weekday-midnight cron for the Amphoreus-owned LinkedIn scrape.

Invokes :func:`amphoreus_linkedin_scrape.run_scrape` at 00:00 UTC on
weekdays (Mon-Fri). Runs inside the FastAPI process via an asyncio task,
same posture as ``jacquard_mirror_cron`` and ``post_embeddings_cron`` so
the opt-in flag + lifecycle hooks are consistent.

Env controls:
  ``ENABLE_AMPHOREUS_LINKEDIN_SCRAPE`` = "true" to turn the loop on
                                          (default: off — opt-in so
                                          local dev / branches stay quiet)
  ``AMPHOREUS_LINKEDIN_SCRAPE_TIMEZONE`` = IANA tz name used to resolve
                                          "weekday midnights" (default:
                                          "UTC"). Applied only at schedule
                                          time; the scrape itself is
                                          timezone-agnostic.

Schedule shape: the next fire time is computed as the next occurrence
of "00:00 in the configured timezone" that falls on a Mon-Fri. Running
once per weeknight means ~5 scrapes/week, bounding Apify cost to
~5 × (profiles × per-profile-price) per week independent of
``interval_hours`` tuning knobs.

Manual one-shot (no cron, no env flag needed):
    python -m backend.src.services.amphoreus_linkedin_scrape
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


def _target_tz() -> timezone:
    """Resolve the configured timezone to a ``tzinfo``. Falls back to UTC
    on any lookup failure — better to fire at the wrong local hour than
    crash the cron loop on import."""
    name = os.environ.get("AMPHOREUS_LINKEDIN_SCRAPE_TIMEZONE", "").strip() or "UTC"
    if name.upper() == "UTC":
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)  # type: ignore[return-value]
    except Exception as exc:
        logger.warning(
            "[amph_li_scrape_cron] timezone %r invalid (%s); falling back to UTC",
            name, exc,
        )
        return timezone.utc


def _next_weekday_midnight(now: datetime, tz) -> datetime:
    """Return the next Mon-Fri 00:00 strictly after ``now``, in ``tz``.

    Concretely: take tonight's midnight in tz; if that's still in the
    future AND falls on a weekday, use it; otherwise roll forward one
    day at a time until both conditions hold. Capped at 7 iterations
    (should never take more than ~3 when starting on a Friday).
    """
    local = now.astimezone(tz)
    # "Midnight" in the local timezone means 00:00 on the NEXT calendar
    # day relative to ``local``. So candidate[0] is tomorrow 00:00.
    candidate = datetime.combine(local.date(), time.min, tzinfo=tz) + timedelta(days=1)
    for _ in range(8):
        if candidate > local and candidate.weekday() < 5:  # 0=Mon … 4=Fri
            return candidate
        candidate += timedelta(days=1)
    # Should be unreachable — 7 days of rolling forward hits a Mon-Fri.
    # Fall back to tomorrow 00:00 UTC so the loop at least makes progress.
    return datetime.combine(
        (now.astimezone(timezone.utc) + timedelta(days=1)).date(),
        time.min, tzinfo=timezone.utc,
    )


async def _run_once() -> None:
    """One scheduled scrape. Executes in a threadpool so Apify HTTP
    calls don't freeze the event loop."""
    loop = asyncio.get_running_loop()
    try:
        from backend.src.services.amphoreus_linkedin_scrape import run_scrape
        result = await loop.run_in_executor(None, run_scrape, None)
        logger.info(
            "[amph_li_scrape_cron] profiles=%d/%d inserted=%d updated=%d "
            "skipped=%d duration=%.1fs errors=%d",
            result.profiles_scraped,
            result.profiles_scraped + result.profiles_failed,
            result.posts_inserted, result.posts_updated, result.posts_skipped,
            result.duration_seconds, len(result.errors),
        )
    except Exception:
        logger.exception("[amph_li_scrape_cron] scrape raised (non-fatal)")


async def _loop() -> None:
    tz = _target_tz()
    logger.info(
        "[amph_li_scrape_cron] started (schedule=weekday 00:00 %s)",
        tz.key if hasattr(tz, "key") else "UTC",
    )
    try:
        while True:
            now = datetime.now(timezone.utc)
            next_fire = _next_weekday_midnight(now, tz)
            wait_s = max(1.0, (next_fire - now).total_seconds())
            logger.info(
                "[amph_li_scrape_cron] next fire at %s (in %.1fh)",
                next_fire.isoformat(), wait_s / 3600,
            )
            await asyncio.sleep(wait_s)
            await _run_once()
    except asyncio.CancelledError:
        logger.info("[amph_li_scrape_cron] cancelled, exiting")
        raise
    except Exception:
        logger.exception("[amph_li_scrape_cron] loop exited with unexpected error")


def start_amphoreus_linkedin_scrape_cron() -> None:
    """Idempotent. Opt-in via ``ENABLE_AMPHOREUS_LINKEDIN_SCRAPE=true``."""
    global _task
    if not _env_flag_on("ENABLE_AMPHOREUS_LINKEDIN_SCRAPE"):
        logger.info(
            "amphoreus_linkedin_scrape_cron disabled "
            "(set ENABLE_AMPHOREUS_LINKEDIN_SCRAPE=true to enable)."
        )
        return
    if _task is not None and not _task.done():
        logger.debug("amphoreus_linkedin_scrape_cron already running; skipping start.")
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(
            "amphoreus_linkedin_scrape_cron: no running event loop; cannot start."
        )
        return

    _task = loop.create_task(_loop(), name="amphoreus_linkedin_scrape_cron")


def stop_amphoreus_linkedin_scrape_cron() -> None:
    global _task
    if _task is None:
        return
    if not _task.done():
        _task.cancel()
    _task = None
