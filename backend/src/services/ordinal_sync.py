"""Continuous Ordinal sync service — background thread syncing posts/comments/approvals."""

import csv
import logging
import threading
import time
from pathlib import Path

from backend.src.core.config import get_settings
from backend.src.db import vortex

logger = logging.getLogger(__name__)

_SYNC_INTERVAL = 3600  # 1 hour default
_sync_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_sync_loop(interval: int = _SYNC_INTERVAL) -> None:
    """Start the background Ordinal sync loop."""
    global _sync_thread
    if _sync_thread and _sync_thread.is_alive():
        logger.info("Ordinal sync already running")
        return

    _stop_event.clear()

    def _loop():
        while not _stop_event.is_set():
            try:
                sync_all_companies()
            except Exception:
                logger.exception("Ordinal sync cycle failed")
            _stop_event.wait(interval)

    _sync_thread = threading.Thread(target=_loop, daemon=True, name="ordinal-sync")
    _sync_thread.start()
    logger.info("Ordinal sync started (interval=%ds)", interval)


def stop_sync_loop() -> None:
    _stop_event.set()


def sync_all_companies() -> None:
    """Iterate ordinal_auth CSV and sync each company."""
    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.debug("No ordinal_auth_rows.csv found — skipping sync")
        return

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_key = row.get("api_key", "").strip()
                company = row.get("company_id", "").strip() or row.get("provider_org_slug", "").strip()
                if api_key and company:
                    try:
                        sync_company(company, api_key)
                    except Exception:
                        logger.exception("Failed to sync company %s", company)
    except Exception:
        logger.exception("Failed to read ordinal auth CSV")


def sync_company(company: str, api_key: str) -> None:
    """Sync posts, comments, and approvals for a single company."""
    import json
    import requests

    base_url = "https://app.tryordinal.com/api/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    posts = []
    has_more = True
    cursor = None

    while has_more:
        params: dict = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(f"{base_url}/posts", headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            posts.extend(data.get("posts", []))
            has_more = data.get("hasMore", False)
            cursor = data.get("nextCursor")
        except Exception as e:
            logger.warning("Ordinal post fetch failed for %s: %s", company, e)
            has_more = False

    if not posts:
        return

    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()

        for post in posts:
            post_id = post.get("id")
            if not post_id:
                continue

            status_map = {
                "Finalized": "approved",
                "Scheduled": "scheduled",
                "Posted": "posted",
                "ForReview": "review",
                "InProgress": "in_progress",
                "Tentative": "tentative",
                "ToDo": "todo",
                "Blocked": "blocked",
            }

            linkedin_copy = ""
            if post.get("linkedIn"):
                linkedin_copy = post["linkedIn"].get("copy", "")

            sb.table("posts").upsert({
                "id": post_id,
                "post_text": linkedin_copy,
                "hook": linkedin_copy[:100] if linkedin_copy else "",
                "status": status_map.get(post.get("status"), post.get("status", "draft")),
                "post_date": post.get("publishAt"),
            }).execute()

        logger.info("Synced %d posts for %s", len(posts), company)

    except Exception:
        logger.exception("Supabase upsert failed for %s", company)
