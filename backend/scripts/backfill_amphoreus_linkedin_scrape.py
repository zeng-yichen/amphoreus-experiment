#!/usr/bin/env python3
"""One-shot runner for the Amphoreus-owned LinkedIn scrape.

Use this to:

  * Do an immediate first-pass scrape right after flipping the
    ``ENABLE_AMPHOREUS_LINKEDIN_SCRAPE`` env flag, instead of waiting
    until the next weekday 00:00 UTC tick of the cron.
  * Smoke-test one profile in isolation before turning the cron on:

        python backend/scripts/backfill_amphoreus_linkedin_scrape.py \\
            --username zeng-yichen

  * Re-scrape a subset (e.g. just Virio teammates) without touching
    client FOCs:

        python backend/scripts/backfill_amphoreus_linkedin_scrape.py \\
            --username zeng-yichen --username eric-virio --username ...

No args ⇒ full tracked-creator set (same filter the cron uses).

Requires env: ``APIFY_API_TOKEN``, ``AMPHOREUS_SUPABASE_URL``,
``AMPHOREUS_SUPABASE_KEY``, ``SUPABASE_URL``, ``SUPABASE_KEY``.
Idempotent — the scraper's upsert contract means re-runs refresh
engagement counts without creating duplicate rows.

Exit code is 0 if at least one profile was scraped, 1 otherwise
(nothing happened / setup failed).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--username", action="append", default=[],
        help="LinkedIn username to scrape (repeatable). Omit to scrape all "
             "Virio-serviced FOCs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve the username list but don't hit Apify. Useful to "
             "confirm the tracked-creator set before spending.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("backfill_amphoreus_linkedin_scrape")

    # Load .env + make repo importable (mirrors the other backfill scripts).
    try:
        from dotenv import load_dotenv
        for candidate in (
            Path(__file__).resolve().parents[2] / ".env",
            Path.cwd() / ".env",
        ):
            if candidate.exists():
                load_dotenv(candidate)
                break
    except Exception:
        pass

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        repo_root = Path("/app")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    usernames = args.username or None

    if args.dry_run:
        # Resolve the list without scraping. Uses the same helpers the
        # cron uses, so a dry-run here validates the production
        # tracked-creator narrowing.
        from backend.src.services.jacquard_mirror_sync import (
            _amphoreus_client, _jacquard_client, _load_tracked_creators,
        )
        amph = _amphoreus_client()
        jcq = _jacquard_client()
        resolved = usernames or _load_tracked_creators(jcq, amph)
        logger.info("DRY RUN — would scrape %d profile(s):", len(resolved))
        for u in resolved:
            logger.info("  %s", u)
        return 0

    from backend.src.services.amphoreus_linkedin_scrape import run_scrape
    result = run_scrape(usernames)

    print(json.dumps({
        "profiles_scraped":  result.profiles_scraped,
        "profiles_failed":   result.profiles_failed,
        "posts_inserted":    result.posts_inserted,
        "posts_updated":     result.posts_updated,
        "posts_skipped":     result.posts_skipped,
        "duration_seconds":  result.duration_seconds,
        "errors":            result.errors[:20],
    }, indent=2))

    return 0 if result.profiles_scraped > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
