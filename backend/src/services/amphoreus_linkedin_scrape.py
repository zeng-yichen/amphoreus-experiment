"""Amphoreus-owned LinkedIn post scraper.

Scrapes recent posts for every Virio-serviced FOC directly from LinkedIn
via Apify's ``apimaestro/linkedin-profile-posts`` actor, and upserts the
results into the same ``linkedin_posts`` mirror table that
``jacquard_mirror_sync`` writes to. Everything downstream
(draft_match_worker, Aglaea's post_bundle, peer retrieval,
post_embeddings) reads from ``linkedin_posts`` unchanged.

Why this exists
---------------
The Jacquard mirror gives us engagement data only as fast as Jacquard
itself scrapes upstream — we have no direct control over that cadence,
and we churn off Ordinal soon which removes the other observation leg.
This module is Amphoreus's own independent LinkedIn scrape, scheduled
on a weekday-midnight cron so fresh reaction/comment counts land every
weeknight without waiting on Jacquard.

Write contract — does NOT fight the Jacquard mirror
---------------------------------------------------
``linkedin_posts`` carries a ``_source`` column ('jacquard' |
'amphoreus'). The Jacquard mirror sync refuses to overwrite
``_source='amphoreus'`` rows, and this scraper reciprocates by never
overwriting identity columns on ``_source='jacquard'`` rows. The three
cases:

  1. **URN not yet in the table.** Insert a full row with
     ``_source='amphoreus'``. Jacquard may later discover this URN and
     flip ``_source='jacquard'`` on the next mirror cycle — that's fine,
     the data converges.
  2. **URN exists, ``_source='jacquard'``.** Update ONLY engagement /
     freshness columns (reaction/comment/repost counts,
     ``*_last_refreshed_at``, ``_last_amphoreus_scraped_at``). Identity
     columns (``post_text``, ``creator_username``, ``posted_at``,
     ``_source``) are untouched — Jacquard owns those.
  3. **URN exists, ``_source='amphoreus'``.** Same as (2), minus the
     ``_source`` concern. Amphoreus keeps full ownership.

This keeps the two scrape paths cleanly addtive: either side falling
over doesn't regress what the other side sees.

Actor notes
-----------
``apimaestro/linkedin-profile-posts`` takes a profile (username or URL)
and returns recent posts with engagement counts. Field names in the
response vary slightly between runs / upstream schema tweaks, so
_shape_item() is defensive — missing fields fall through to None and the
upsert carries whatever we have.

Reshares ARE kept (per design review) — stored as a separate
``linkedin_posts`` row with ``reshared_post_urn`` populated, mirroring
Jacquard's shape.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Actor + HTTP config. Matches the pattern in engager_fetcher.py so
# anyone reading one can orient in the other.
_APIFY_BASE = "https://api.apify.com/v2/acts"
_PROFILE_POSTS_ACTOR = "apimaestro~linkedin-profile-posts"
_APIFY_TIMEOUT_SECONDS = 180  # profile-posts actor can take 1-2min per profile

# Hard cap on profiles per run — protects the wallet if the
# tracked-creator list grows silently. Override via env.
_DEFAULT_MAX_PROFILES_PER_RUN = 50

# Cap posts returned per profile. Most FOCs post < 12/month so 30 is
# plenty, and it means per-profile Apify cost is bounded.
_POSTS_PER_PROFILE = 30


# ---------------------------------------------------------------------------
# Result aggregate — matches the shape run_sync / run_match_back emit so the
# caller can log uniformly.
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    profiles_scraped:   int               = 0
    profiles_failed:    int               = 0
    posts_inserted:     int               = 0
    posts_updated:      int               = 0
    posts_skipped:      int               = 0   # couldn't parse URN
    duration_seconds:   float             = 0.0
    errors:             list[str]         = field(default_factory=list)

    def to_rows_per_table(self) -> dict[str, Any]:
        """Match the ``sync_runs.rows_per_table`` shape used elsewhere."""
        return {
            "linkedin_posts_inserted": self.posts_inserted,
            "linkedin_posts_updated":  self.posts_updated,
            "linkedin_posts_skipped":  self.posts_skipped,
            "profiles_scraped":        self.profiles_scraped,
            "profiles_failed":         self.profiles_failed,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scrape(usernames: Optional[list[str]] = None) -> ScrapeResult:
    """One scrape pass. Safe to call from the cron or as a one-off.

    ``usernames`` — optional subset to scrape. Defaults to the full
    Virio-serviced tracked-creator set (same filter as the Jacquard
    mirror uses, for consistency). Pass a one-element list for a
    smoke-test of a single profile.

    Returns a :class:`ScrapeResult` with insert/update counts and any
    per-profile errors. A hard failure during setup (missing Apify
    token, no Supabase client) returns early with a populated
    ``errors`` list — the cron wrapper logs it and moves on.
    """
    t0 = time.time()
    result = ScrapeResult()

    apify_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not apify_token:
        result.errors.append("APIFY_API_TOKEN not set")
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    # Reuse the canonical tracked-creator set from jacquard_mirror_sync
    # so narrowing changes in one place flow to the other.
    from backend.src.services.jacquard_mirror_sync import (
        _amphoreus_client, _jacquard_client, _load_tracked_creators,
    )

    try:
        amph = _amphoreus_client()
    except Exception as exc:
        result.errors.append(f"amphoreus_client: {exc}")
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    if usernames is None:
        try:
            jcq = _jacquard_client()
            usernames = _load_tracked_creators(jcq, amph)
        except Exception as exc:
            logger.exception("[amph_li_scrape] failed to load tracked creators")
            result.errors.append(f"load_creators: {exc}")
            result.duration_seconds = round(time.time() - t0, 2)
            return result

    if not usernames:
        logger.info("[amph_li_scrape] no tracked creators — nothing to do")
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    max_profiles = _env_int("AMPHOREUS_LINKEDIN_SCRAPE_MAX_PROFILES_PER_RUN",
                             _DEFAULT_MAX_PROFILES_PER_RUN)
    if len(usernames) > max_profiles:
        logger.warning(
            "[amph_li_scrape] trimming creator list %d → %d "
            "(set AMPHOREUS_LINKEDIN_SCRAPE_MAX_PROFILES_PER_RUN to override)",
            len(usernames), max_profiles,
        )
        usernames = list(usernames)[:max_profiles]

    logger.info("[amph_li_scrape] starting scrape for %d profile(s)", len(usernames))

    for username in usernames:
        try:
            per = _scrape_and_upsert_profile(amph, apify_token, username)
        except Exception as exc:
            logger.exception("[amph_li_scrape] profile=%s failed", username)
            result.profiles_failed += 1
            result.errors.append(f"{username}: {str(exc)[:200]}")
            continue
        result.profiles_scraped += 1
        result.posts_inserted   += per["inserted"]
        result.posts_updated    += per["updated"]
        result.posts_skipped    += per["skipped"]

    # Fire the semantic match-back worker so newly-upserted LinkedIn
    # rows pair with Stelle drafts immediately, not on the next 8h
    # Jacquard cycle. ``run_match_back`` is idempotent so running it
    # from both crons (this one + jacquard_mirror_cron) is safe.
    try:
        from backend.src.services.draft_match_worker import run_match_back
        match_summary = run_match_back()
        logger.info(
            "[amph_li_scrape] match-back after scrape: %d new matches",
            match_summary.get("matched", 0),
        )
    except Exception as exc:
        logger.exception("[amph_li_scrape] match-back after scrape failed (non-fatal)")
        result.errors.append(f"match_back: {str(exc)[:200]}")

    result.duration_seconds = round(time.time() - t0, 2)

    # One row in sync_runs per scrape cycle. Matches the shape used by
    # jacquard_mirror_sync.run_sync so the /api/mirror/sync-runs UI
    # surfaces this without changes.
    _log_to_sync_runs(amph, result)

    logger.info(
        "[amph_li_scrape] done: profiles=%d/%d posts_in=%d updated=%d "
        "skipped=%d errors=%d duration=%.1fs",
        result.profiles_scraped,
        result.profiles_scraped + result.profiles_failed,
        result.posts_inserted, result.posts_updated, result.posts_skipped,
        len(result.errors), result.duration_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# Per-profile scrape + upsert
# ---------------------------------------------------------------------------

def _scrape_and_upsert_profile(
    amph, apify_token: str, username: str,
) -> dict[str, int]:
    """Scrape one profile and upsert its posts. Raises on hard failures
    (HTTP 500 from Apify, auth errors) — the caller catches + records."""
    items = _run_profile_posts_actor(apify_token, username)
    if not items:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    shaped: list[dict[str, Any]] = []
    skipped_ct = 0
    for item in items:
        row = _shape_item(item, username)
        if row is None:
            skipped_ct += 1
            continue
        shaped.append(row)

    if not shaped:
        return {"inserted": 0, "updated": skipped_ct, "skipped": skipped_ct}

    # Partition into insert (new URN) vs update (existing URN) by
    # querying current state. One batched SELECT beats N individual
    # existence checks.
    urns = [r["provider_urn"] for r in shaped]
    existing = _fetch_existing_urns(amph, urns)

    ins_rows: list[dict[str, Any]] = []
    upd_rows: list[dict[str, Any]] = []
    for r in shaped:
        if r["provider_urn"] in existing:
            upd_rows.append(_engagement_only_payload(r))
        else:
            ins_rows.append(_full_insert_payload(r))

    inserted = _batch_insert(amph, ins_rows)
    updated  = _batch_update(amph, upd_rows)

    # Append one row per shaped post to linkedin_post_engagement_snapshots
    # so the kinetic trajectory (reactions / comments / reposts over
    # time) is preserved across scrapes. Best-effort — failures here
    # never block the upsert. Written AFTER the upsert so the history
    # row is paired with the latest-known counts on the parent row.
    _write_engagement_snapshots(shaped, scraped_by="amphoreus")

    return {"inserted": inserted, "updated": updated, "skipped": skipped_ct}


def _write_engagement_snapshots(
    shaped_rows: list[dict[str, Any]], *, scraped_by: str,
) -> None:
    """Best-effort time-series writes for a batch of just-scraped rows.

    One ``linkedin_post_engagement_snapshots`` row per URN. PK dedup
    (provider_urn, scraped_at) means two writes at the same microsecond
    silently collapse — fine.
    """
    if not shaped_rows:
        return
    try:
        from backend.src.db.amphoreus_supabase import record_engagement_snapshot
    except Exception:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    for r in shaped_rows:
        urn = (r.get("provider_urn") or "").strip()
        if not urn:
            continue
        record_engagement_snapshot(
            provider_urn=    urn,
            scraped_at=      now_iso,
            total_reactions= r.get("total_reactions"),
            total_comments=  r.get("total_comments"),
            total_reposts=   r.get("total_reposts"),
            scraped_by=      scraped_by,
        )


def _run_profile_posts_actor(token: str, username: str) -> list[dict[str, Any]]:
    """Call apimaestro/linkedin-profile-posts and return the raw items list.

    Apify's run-sync-get-dataset-items endpoint blocks until the actor
    finishes, so we run profiles sequentially. Trading throughput for
    simplicity — a weekday-midnight window has plenty of slack.
    """
    import httpx

    url = f"{_APIFY_BASE}/{_PROFILE_POSTS_ACTOR}/run-sync-get-dataset-items"
    params = {"format": "json", "token": token}
    payload = {
        "username":   username,
        "limit":      _POSTS_PER_PROFILE,
        # Some actor versions use ``profile_url`` instead of ``username`` —
        # sending both defensively; the actor ignores unknown fields.
        "profile_url": f"https://www.linkedin.com/in/{username}/",
    }

    try:
        resp = httpx.post(url, params=params, json=payload, timeout=_APIFY_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[amph_li_scrape] actor call failed for %s: %s", username, exc)
        return []

    # run-sync-get-dataset-items returns a bare list when ?format=json.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
    return []


# ---------------------------------------------------------------------------
# Item → row shaping
# ---------------------------------------------------------------------------

def _shape_item(item: dict[str, Any], username: str) -> Optional[dict[str, Any]]:
    """Normalize one actor-output item into a ``linkedin_posts`` row.

    Field names have drifted across actor versions, so every extraction
    tries a few common aliases and falls back to ``None``. The only
    absolutely required field is ``provider_urn`` — without it we can't
    dedupe and the row would be rejected by the PK anyway.
    """
    if not isinstance(item, dict):
        return None

    # URN extraction. apimaestro/linkedin-profile-posts returns ``urn`` as
    # a dict ``{activity_urn, share_urn, ugcPost_urn}`` plus a sibling
    # ``full_urn`` string like ``"urn:li:activity:<id>"``. Jacquard's
    # ``linkedin_posts.provider_urn`` stores just the numeric activity id
    # (e.g. ``"7450942125170442240"``). We need to match that format so
    # updates dedup against existing rows instead of creating parallel
    # amphoreus-keyed duplicates (the bug from the 2026-04-23 first smoke).
    provider_urn: Optional[str] = None
    urn_raw = item.get("urn")
    if isinstance(urn_raw, dict):
        provider_urn = (
            urn_raw.get("activity_urn")
            or urn_raw.get("share_urn")
            or urn_raw.get("ugcPost_urn")
        )
    elif isinstance(urn_raw, str):
        provider_urn = urn_raw
    # Fallbacks: full_urn → strip the "urn:li:activity:" prefix to land
    # on the numeric id; otherwise try the legacy aliases.
    if not provider_urn:
        full = item.get("full_urn")
        if isinstance(full, str) and ":" in full:
            provider_urn = full.rsplit(":", 1)[-1]
    if not provider_urn:
        provider_urn = _first(item, "provider_urn", "post_urn", "activityUrn", "id")
    if not provider_urn:
        # Older actor versions nest under "post"
        post_block = item.get("post")
        if isinstance(post_block, dict):
            provider_urn = _first(post_block, "urn", "provider_urn", "id")
    if not provider_urn:
        return None
    provider_urn = str(provider_urn).strip()

    post_text = _first(item, "text", "post_text", "content", "commentary") or ""
    post_url  = _first(item, "post_url", "url", "postUrl", "permalink") or ""

    # Engagement counts — actor returns these either at the top level or
    # nested under "stats" / "engagement".
    stats = item.get("stats") or item.get("engagement") or {}
    if not isinstance(stats, dict):
        stats = {}
    total_reactions = _first_int(item, "total_reactions", "reactions_count",
                                  "likes_count", "num_reactions") \
        or _first_int(stats, "total_reactions", "reactions", "likes") \
        or 0
    total_comments  = _first_int(item, "total_comments", "comments_count",
                                  "num_comments") \
        or _first_int(stats, "total_comments", "comments") \
        or 0
    total_reposts   = _first_int(item, "total_reposts", "reposts_count",
                                  "shares_count", "num_reposts") \
        or _first_int(stats, "total_reposts", "reposts", "shares") \
        or 0

    # posted_at. apimaestro returns a dict
    # ``{date: "YYYY-MM-DD HH:MM:SS", relative: "4 hours ago…", timestamp: <ms>}``
    # — we pick the ms timestamp (unambiguous) and fall back to ``date``
    # then whatever other top-level alias might carry it.
    posted_at_raw = item.get("posted_at")
    posted_at: Any = None
    if isinstance(posted_at_raw, dict):
        posted_at = (
            posted_at_raw.get("timestamp")
            or posted_at_raw.get("date")
            or posted_at_raw.get("iso")
        )
    elif posted_at_raw is not None:
        posted_at = posted_at_raw
    if not posted_at:
        posted_at = _first(item, "postedAt", "published_at",
                            "publishedAt", "time_posted", "createdAt")
    posted_at_iso = _parse_iso(posted_at)

    # Author resolution — prefer nested ``author``/``profile`` block,
    # fall back to the username we queried with (we know that much).
    author = item.get("author") or item.get("profile") or {}
    if not isinstance(author, dict):
        author = {}
    creator_username = (
        _first(author, "username", "handle", "public_identifier")
        or username
    )
    creator_first_name  = _first(author, "first_name", "firstName", "given_name")
    creator_last_name   = _first(author, "last_name", "lastName", "family_name")
    creator_headline    = _first(author, "headline", "sub_title")
    creator_profile_url = _first(author, "profile_url", "profileUrl", "url") or \
                           f"https://www.linkedin.com/in/{creator_username}/"

    attachments = item.get("attachments") or item.get("media")
    has_image = bool(item.get("has_image")) or _attachment_has_kind(attachments, "image")
    has_video = bool(item.get("has_video")) or _attachment_has_kind(attachments, "video")

    # Reshare tracking — Jacquard's shape uses ``reshared_post_urn``.
    reshared_post_urn = (
        _first(item, "reshared_post_urn", "reshared_urn", "original_post_urn")
        or _first(item.get("reshare") or {}, "urn", "provider_urn")
        or None
    )

    return {
        "provider_urn":        provider_urn,
        "post_url":            post_url,
        "post_text":           post_text,
        "post_text_length":    len(post_text) if post_text else 0,
        "total_reactions":     total_reactions,
        "total_comments":      total_comments,
        "total_reposts":       total_reposts,
        "has_image":           has_image,
        "has_video":           has_video,
        "attachments":         attachments if isinstance(attachments, (list, dict)) else None,
        "creator_username":    (creator_username or username).lower(),
        "creator_first_name":  creator_first_name,
        "creator_last_name":   creator_last_name,
        "creator_headline":    creator_headline,
        "creator_profile_url": creator_profile_url,
        "posted_at":           posted_at_iso,
        "reshared_post_urn":   reshared_post_urn,
    }


def _full_insert_payload(shaped: dict[str, Any]) -> dict[str, Any]:
    """Build the row for a brand-new linkedin_posts insert.

    Stamps ``_source='amphoreus'`` (marking Amphoreus as the originating
    scraper) and populates all three *_last_refreshed_at timestamps so
    downstream freshness logic doesn't assume stale data.

    Also defaults ``is_company_post=False`` because the
    apimaestro/linkedin-profile-posts actor doesn't return that flag.
    FOC profiles are always personal, never company pages, so False
    is the correct default. Without this stamp, PostgREST filters like
    ``is_company_post=eq.false`` (used by ``post_bundle`` before the
    NULL-tolerant fix landed) would silently drop every Amphoreus-
    scraped row.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = dict(shaped)
    payload["_source"]                     = "amphoreus"
    payload["_last_amphoreus_scraped_at"]  = now_iso
    payload["post_last_refreshed_at"]      = now_iso
    payload["reactions_last_refreshed_at"] = now_iso
    payload["comments_last_refreshed_at"]  = now_iso
    payload.setdefault("is_company_post",  False)
    return payload


def _engagement_only_payload(shaped: dict[str, Any]) -> dict[str, Any]:
    """Build the update payload for an existing row.

    Surgical: ONLY fields that can legitimately change between scrapes
    without us clobbering Jacquard's identity state. ``post_text``,
    ``creator_*``, ``posted_at``, and ``_source`` are deliberately
    excluded — see the module docstring's write contract.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "provider_urn":                 shaped["provider_urn"],  # for .eq() match
        "total_reactions":              shaped["total_reactions"],
        "total_comments":               shaped["total_comments"],
        "total_reposts":                shaped["total_reposts"],
        "post_last_refreshed_at":       now_iso,
        "reactions_last_refreshed_at":  now_iso,
        "comments_last_refreshed_at":   now_iso,
        "_last_amphoreus_scraped_at":   now_iso,
    }


# ---------------------------------------------------------------------------
# Supabase read / write helpers
# ---------------------------------------------------------------------------

def _fetch_existing_urns(amph, urns: list[str]) -> set[str]:
    """Return the subset of ``urns`` already present in linkedin_posts.

    Chunks the IN-list at 100 to avoid PostgREST URL-length 400s (same
    trick jacquard_mirror_sync uses for its meeting joins).
    """
    out: set[str] = set()
    if not urns:
        return out
    _CHUNK = 100
    for i in range(0, len(urns), _CHUNK):
        try:
            rows = (
                amph.table("linkedin_posts")
                    .select("provider_urn")
                    .in_("provider_urn", urns[i:i + _CHUNK])
                    .execute()
                    .data
                or []
            )
        except Exception as exc:
            logger.warning("[amph_li_scrape] existence check failed: %s", exc)
            continue
        for r in rows:
            urn = r.get("provider_urn")
            if urn:
                out.add(urn)
    return out


def _batch_insert(amph, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    # Insert in chunks of 100 — supabase-py serializes to one HTTP
    # request per call and PostgREST caps payload at 1MB by default.
    _CHUNK = 100
    inserted = 0
    for i in range(0, len(rows), _CHUNK):
        chunk = rows[i:i + _CHUNK]
        try:
            amph.table("linkedin_posts").insert(chunk).execute()
            inserted += len(chunk)
        except Exception as exc:
            # A concurrent Jacquard mirror write could race us on a
            # URN — fall back to per-row inserts so one PK collision
            # doesn't poison the whole chunk.
            logger.warning("[amph_li_scrape] chunk insert failed (%s); retrying per-row", exc)
            for r in chunk:
                try:
                    amph.table("linkedin_posts").insert(r).execute()
                    inserted += 1
                except Exception as inner:
                    logger.debug(
                        "[amph_li_scrape] skip insert urn=%s: %s",
                        r.get("provider_urn", "?")[:40], inner,
                    )
    return inserted


def _batch_update(amph, rows: list[dict[str, Any]]) -> int:
    """N per-urn UPDATEs — PostgREST doesn't offer a bulk-update-by-pk
    primitive for heterogeneous payloads. Acceptable: the update list
    per cron tick is ≤ profiles × posts_per_profile = 50 × 30 = 1500
    worst case, which is a few seconds of RTT."""
    if not rows:
        return 0
    updated = 0
    for r in rows:
        urn = r.pop("provider_urn", None)
        if not urn:
            continue
        try:
            amph.table("linkedin_posts").update(r).eq("provider_urn", urn).execute()
            updated += 1
        except Exception as exc:
            logger.debug(
                "[amph_li_scrape] skip update urn=%s: %s",
                urn[:40], exc,
            )
    return updated


def _log_to_sync_runs(amph, result: ScrapeResult) -> None:
    """Write one observability row to sync_runs. Non-fatal on failure."""
    try:
        amph.table("sync_runs").insert({
            "started_at":      (datetime.now(timezone.utc)
                                 .replace(microsecond=0).isoformat()),
            "completed_at":    datetime.now(timezone.utc).isoformat(),
            "status":          "completed" if not result.errors else "completed_with_errors",
            "kind":            "amphoreus_linkedin_scrape",
            "scope":           "global",
            "rows_per_table":  result.to_rows_per_table(),
            "error":           (" | ".join(result.errors))[:2000] if result.errors else None,
            "triggered_by":    "cron",
        }).execute()
    except Exception as exc:
        logger.debug("[amph_li_scrape] sync_runs log failed: %s", exc)


# ---------------------------------------------------------------------------
# Tiny util
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return None


def _first_int(d: dict[str, Any], *keys: str) -> int:
    raw = _first(d, *keys)
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _parse_iso(raw: Any) -> Optional[str]:
    """Coerce various posted_at formats to ISO 8601 UTC.

    Actor output has been observed as ISO strings, ms-epochs, and
    already-parsed datetimes. On any parse failure return None and let
    the row land with a NULL posted_at — better than guessing wrong.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(raw, (int, float)):
        # ms epochs are > 10^12; s epochs < 10^11.
        secs = raw / 1000.0 if raw > 10**11 else float(raw)
        try:
            return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    return None


def _attachment_has_kind(attachments: Any, kind: str) -> bool:
    if not isinstance(attachments, list):
        return False
    kind_re = re.compile(kind, re.IGNORECASE)
    for a in attachments:
        if not isinstance(a, dict):
            continue
        for key in ("type", "media_type", "kind"):
            v = a.get(key)
            if isinstance(v, str) and kind_re.search(v):
                return True
    return False


# ---------------------------------------------------------------------------
# CLI — ``python -m backend.src.services.amphoreus_linkedin_scrape [username]``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        from dotenv import load_dotenv
        from pathlib import Path as _Path
        candidate = _Path(__file__).resolve().parents[3] / ".env"
        if candidate.exists():
            load_dotenv(candidate)
    except Exception:
        pass

    # Optional positional arg: single username to smoke-test.
    usernames_arg = sys.argv[1:] or None
    result = run_scrape(usernames_arg)
    import json
    print(json.dumps({
        "profiles_scraped":  result.profiles_scraped,
        "profiles_failed":   result.profiles_failed,
        "posts_inserted":    result.posts_inserted,
        "posts_updated":     result.posts_updated,
        "posts_skipped":     result.posts_skipped,
        "duration_seconds":  result.duration_seconds,
        "errors":            result.errors[:10],
    }, indent=2))
