"""One-way Jacquard → Amphoreus mirror sync.

Pulls the tables listed in ``_TABLES`` from Jacquard's Supabase and
upserts them into the corresponding mirror tables in Amphoreus's
Supabase. Transcript bodies referenced in the ``meetings.transcript_url``
(GCS) column are copied into our own Supabase Storage bucket so agents
can read them without needing Jacquard's GCS credentials.

Design notes:

* ``_source`` column on every mirror row. Rows written here always get
  ``_source = 'jacquard'``. The sync worker NEVER touches rows stamped
  ``_source = 'amphoreus'`` — those are native writes (operator
  transcript uploads, etc.).
* Incremental by ``updated_at`` / ``created_at`` with a per-table
  watermark persisted in ``mirror_sync_state``. First run is a full
  backfill (watermark absent); subsequent runs only pull the delta.
* Per-table errors are caught and logged — a single broken table never
  kills the whole run. The ``sync_runs`` row captures per-table counts
  and the first error string.
* Idempotent. Safe to re-run mid-cycle — upserts on PK.
* READS ARE NOT YET WIRED TO THIS MIRROR. This module exists to
  populate it; agents still read from Jacquard directly via
  ``jacquard_direct``. The cutover is a later, separate change.

Env:
    SUPABASE_URL / SUPABASE_KEY                      — Jacquard (source)
    AMPHOREUS_SUPABASE_URL / AMPHOREUS_SUPABASE_KEY  — Amphoreus (dest)
    GCS_CREDENTIALS_B64 / GCS_BUCKET                 — Jacquard GCS
    AMPHOREUS_TRANSCRIPTS_BUCKET                     — Supabase Storage
                                                       bucket name (default
                                                       ``transcripts``)

Callers:
    * cron (see ``jacquard_mirror_cron.py``) — every 8h
    * ``POST /api/mirror/sync`` — on-demand, admin-only
    * CLI: ``python -m backend.src.services.jacquard_mirror_sync``
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------
# Each entry describes how to sync one Jacquard table into its mirror:
#
#   source_table : str             — name in Jacquard (and Amphoreus — they match)
#   pk           : str             — primary key column (usually 'id')
#   cursor_col   : str | None      — column used for incremental sync
#                                    (None = table is small, always full-refresh)
#   select_cols  : str             — SELECT clause. Usually "*".
#   scope_by_company : str | None  — company-filter column if set
#                                    (for per-company on-demand syncs)
#
# Order matters for logs (independent tables only — no FKs enforced across mirror).


@dataclass(frozen=True)
class _TableSpec:
    source_table: str
    pk: str = "id"
    cursor_col: Optional[str] = "updated_at"
    select_cols: str = "*"
    scope_by_company: Optional[str] = None  # company column, if any
    # Secondary cursor fallback — some tables don't have updated_at but do
    # have created_at; we use that to bound incremental pulls instead.
    cursor_fallback: Optional[str] = "created_at"
    # Batch size for Supabase range pulls. Default 1000; tune per table
    # where rows are big (transcripts have GCS URLs + extracted text).
    page_size: int = 1000
    # Optional: restrict rows to those whose <creator_col> is in the
    # tracked-creators set. Set on linkedin_posts to keep the mirror
    # scoped to OUR clients (~500 rows) instead of the whole 390k
    # Jacquard corpus.
    creator_col: Optional[str] = None
    # Optional: restrict rows to those whose <post_col> matches a
    # provider_urn from our already-mirrored linkedin_posts. Used for
    # linkedin_reactions / linkedin_comments so they stay scoped to
    # our tracked creators' posts. Requires linkedin_posts to be
    # synced first.
    post_urn_col: Optional[str] = None
    # Optional: rolling-window cutoff in days applied on top of the
    # cursor/watermark. ``min_age_days=180`` on linkedin_posts means we
    # only pull + keep posts from the last 6 months — older voice
    # samples have drifted stylistically and aren't useful for Stelle's
    # voice calibration. Applied against ``cursor_col`` (e.g.
    # ``posted_at >= now() - 180d``).
    min_age_days: Optional[int] = None


_TABLES: list[_TableSpec] = [
    # Identity — small, always refresh
    _TableSpec("user_companies",  cursor_col="updated_at"),
    _TableSpec("users",           cursor_col="updated_at", scope_by_company="company_id"),

    # Meetings + transcripts (transcript BODY copied to Supabase Storage).
    # Both Jacquard's meetings + meeting_participants use non-'id' PKs —
    # see database.types.ts. Upsert on_conflict must match.
    _TableSpec("meetings", pk="provider_transcript_id",
               cursor_col="updated_at", scope_by_company=None),
    _TableSpec("meeting_participants", pk="provider_transcript_id,name",
               cursor_col="created_at"),

    # LinkedIn engagement — the hot paths for Stelle/Cyrene/Irontomb.
    # Scoped to our tracked creators only (via creator_col/post_urn_col):
    # Jacquard's linkedin_posts has 390k rows across every creator in
    # their system; Amphoreus only cares about the ~34 creators our
    # clients face. Their posts + the reactions/comments/profiles on
    # those posts are all we mirror.
    # Rolling 6-month window: older posts are stylistically drifted and
    # not useful for Stelle's voice-calibration. Trim the corpus to the
    # last 180 days so the mirror stays lean (~2k rows instead of ~5k+).
    # BL cleanup 2026-04-22: explicit select_cols excludes Jacquard's
    # hand-labeled taxonomies (hook_type, format_archetype, tone,
    # target_persona, topic_tags, key_entities, abm_category,
    # abm_intent, is_tofu/mofu/bofu/lead_magnet, quality_score). Those
    # columns encoded Jacquard's theory of LinkedIn content structure
    # — we rely on semantic retrieval over post_text instead, so
    # copying them wastes mirror space and invites downstream code
    # paths to start filtering on them again. Keep: engagement numbers
    # (reactions / comments / reposts / engagement_score /
    # is_outlier), authorship, URLs, attachments, timestamps.
    _TableSpec("linkedin_posts", pk="provider_urn", cursor_col="posted_at",
               creator_col="creator_username", min_age_days=180,
               select_cols=(
                   "provider_urn, id, post_url, post_type, post_text, "
                   "post_text_length, hook, creator_first_name, "
                   "creator_last_name, creator_headline, creator_username, "
                   "creator_profile_url, has_image, has_video, total_images, "
                   "attachments, total_reactions, like_reactions, "
                   "love_reactions, celebrate_reactions, insight_reactions, "
                   "support_reactions, funny_reactions, total_comments, "
                   "total_reposts, engagement_score, is_outlier, "
                   "is_company_post, reshared_post_urn, posted_at, "
                   "post_last_refreshed_at, created_at"
                   # updated_at dropped 2026-04-23 — Jacquard's source
                   # linkedin_posts doesn't have this column (only
                   # Amphoreus mirror schema does). Keeping it in the
                   # select list crashed the hourly sync starting
                   # 2026-04-22 with "column linkedin_posts.updated_at
                   # does not exist". cursor_col="posted_at" is the
                   # incremental tracking anchor instead.
               )),
    _TableSpec("linkedin_reactions", cursor_col="created_at",
               post_urn_col="provider_post_urn"),
    _TableSpec("linkedin_comments",  cursor_col="created_at",
               post_urn_col="provider_post_urn"),
    # linkedin_profiles is handled by _sync_reactor_profiles (PK-batch
    # lookup driven by the URNs present in our mirrored reactions +
    # comments). A full-table scan on Jacquard's linkedin_profiles
    # blows past the statement-timeout budget.
    # (intentionally omitted from the main table loop)

    # Research + account context
    _TableSpec("parallel_research_results", cursor_col="created_at", scope_by_company="company_id"),
    _TableSpec("reports",                   cursor_col="completed_at", scope_by_company="company_id"),
    _TableSpec("tone_references",           cursor_col="created_at"),
    _TableSpec("context_files",             cursor_col="created_at", scope_by_company="company_id"),

    # Drafts (edit-history; mirror NEVER writes these, only reads them
    # for Cyrene's timeline view)
    _TableSpec("drafts",          cursor_col="updated_at"),
    _TableSpec("draft_comments",  cursor_col="created_at"),
    _TableSpec("draft_snapshots", cursor_col="created_at"),

    # Ops / context — trigger_log / company_events / tasks have uuid
    # ``id`` PKs. slack_channels uses ``provider_id`` as PK.
    _TableSpec("trigger_log",      cursor_col="created_at", scope_by_company="company_id"),
    _TableSpec("company_events",   cursor_col="created_at", scope_by_company="company_id"),
    _TableSpec("tasks",            cursor_col="created_at", scope_by_company="company_id"),
    _TableSpec("slack_channels",   pk="provider_id", cursor_col="created_at", scope_by_company="company_id"),

    # Ordinal auth keys — one row per client, PK=company_id. No
    # updated_at/created_at columns upstream, so cursor_col=None triggers
    # a full fetch + upsert every sync. 36 rows × 3 cols is cheap.
    # Amphoreus-side ``profile_id`` isn't in Jacquard, so the upsert
    # doesn't touch it (the column simply isn't in the payload).
    _TableSpec("ordinal_auth",     pk="company_id", cursor_col=None, cursor_fallback=None),

    # Content pipeline — Jacquard's pre-LinkedIn "content plan" rows. PK is
    # integer ``id``; cursor on ``last_updated_at`` (Jacquard's column name,
    # not the usual ``updated_at``).
    _TableSpec("posts",            pk="id", cursor_col="last_updated_at"),
]


# ---------------------------------------------------------------------------
# Env + client helpers
# ---------------------------------------------------------------------------

def _env(name: str, *, required: bool = True) -> str:
    v = os.environ.get(name, "").strip()
    if required and not v:
        raise RuntimeError(f"{name} is not set")
    return v


def _jacquard_client():
    from supabase import create_client
    return create_client(_env("SUPABASE_URL"), _env("SUPABASE_KEY"))


def _amphoreus_client():
    from supabase import create_client
    return create_client(_env("AMPHOREUS_SUPABASE_URL"), _env("AMPHOREUS_SUPABASE_KEY"))


# ---------------------------------------------------------------------------
# Watermark persistence — `mirror_sync_state` table
# ---------------------------------------------------------------------------
# One row per (table, scope) with the last cursor we synced through.
# Auto-created on first sync if missing.


def _ensure_sync_state_table(amph) -> None:
    """Create mirror_sync_state if it doesn't exist. Idempotent."""
    # We can't run DDL via the supabase-py client; PostgREST doesn't expose
    # it. Instead we just try a select — if it 404s, log a warning and
    # write watermarks in-memory only (no persistence). The DDL for this
    # table lives in the schema addendum; operator should run it once.
    try:
        amph.table("mirror_sync_state").select("table_name").limit(1).execute()
    except Exception:
        logger.warning(
            "[mirror_sync] mirror_sync_state table missing — "
            "watermarks will reset on every run. Apply schema addendum."
        )


def _load_watermark(amph, table: str, scope: str) -> Optional[str]:
    try:
        resp = (
            amph.table("mirror_sync_state")
                .select("cursor_value")
                .eq("table_name", table)
                .eq("scope", scope)
                .limit(1)
                .execute()
        )
        rows = resp.data or []
        if rows:
            return rows[0].get("cursor_value")
    except Exception as exc:
        logger.debug("[mirror_sync] _load_watermark(%s,%s): %s", table, scope, exc)
    return None


def _save_watermark(amph, table: str, scope: str, cursor_value: str) -> None:
    try:
        amph.table("mirror_sync_state").upsert(
            {
                "table_name": table,
                "scope": scope,
                "cursor_value": cursor_value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="table_name,scope",
        ).execute()
    except Exception as exc:
        logger.debug("[mirror_sync] _save_watermark(%s,%s): %s", table, scope, exc)


# ---------------------------------------------------------------------------
# GCS → Supabase Storage for transcript bodies
# ---------------------------------------------------------------------------
# meetings.transcript_url is a gs:// URL in Jacquard. We download each
# transcript body and upload it to Amphoreus Supabase Storage so the
# mirror is self-contained (no GCS creds needed to read).


def _gcs_download(gcs_url: str) -> Optional[bytes]:
    """Fetch a ``gs://bucket/path`` object using GCS_CREDENTIALS_B64.

    Returns None on any failure — caller logs + moves on; this sync run
    can retry on the next pass.
    """
    if not gcs_url or not gcs_url.startswith("gs://"):
        return None
    try:
        creds_b64 = _env("GCS_CREDENTIALS_B64", required=False)
        if not creds_b64:
            return None
        # Reuse the same parse helper the direct reader uses — keeps GCS
        # bucket/object handling consistent.
        from backend.src.agents.jacquard_direct import (
            _parse_gcs_url,
            _gcs_storage_client,
        )
        parsed = _parse_gcs_url(gcs_url)
        if not parsed:
            return None
        bucket_name, blob_name = parsed
        client = _gcs_storage_client()
        if not client:
            return None
        blob = client.bucket(bucket_name).blob(blob_name)
        return blob.download_as_bytes()
    except Exception as exc:
        logger.warning("[mirror_sync] GCS download failed for %s: %s", gcs_url, exc)
        return None


def _storage_upload(amph, bucket: str, path: str, body: bytes, content_type: str) -> Optional[str]:
    """Upload body to Amphoreus Supabase Storage, return the storage path.

    Uses upsert so re-syncs overwrite identical paths without erroring.
    """
    try:
        amph.storage.from_(bucket).upload(
            path=path,
            file=body,
            file_options={
                "content-type": content_type,
                "upsert": "true",
            },
        )
        return path
    except Exception as exc:
        # Some supabase-py versions raise on existing path even with
        # upsert; fall through with a warning.
        logger.warning("[mirror_sync] storage upload %s/%s failed: %s", bucket, path, exc)
        return None


# ---------------------------------------------------------------------------
# Core sync
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    status: str = "running"            # running|completed|failed
    kind: str = "scheduled"            # scheduled|on_demand
    scope: str = "global"              # global|<company_uuid>
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    rows_per_table: dict[str, int] = field(default_factory=dict)
    gcs_objects_copied: int = 0
    error: Optional[str] = None
    triggered_by: Optional[str] = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_query(
    jcq,
    spec: _TableSpec,
    scope_company_id: Optional[str],
    watermark: Optional[str],
    *,
    tracked_creators: Optional[list[str]] = None,
    tracked_post_urns: Optional[list[str]] = None,
):
    """Build a fresh query for ONE page.

    supabase-py's PostgrestFilterBuilder accumulates query params across
    chained calls; reusing the same ``q`` across ``.range()`` iterations
    produces ``?offset=0&offset=1000&offset=2000&...`` URLs which return
    zero rows after the first page. Rebuilding per page avoids that.
    """
    q = jcq.table(spec.source_table).select(spec.select_cols)
    if scope_company_id and spec.scope_by_company:
        q = q.eq(spec.scope_by_company, scope_company_id)
    if spec.creator_col and tracked_creators:
        q = q.in_(spec.creator_col, tracked_creators)
    if spec.post_urn_col and tracked_post_urns:
        q = q.in_(spec.post_urn_col, tracked_post_urns)
    cursor_col = spec.cursor_col or spec.cursor_fallback
    # Rolling-window floor on the cursor column (e.g. last 6 months of
    # posted_at). Applied as a MAX with the watermark so the sync pulls
    # the newer of the two bounds — if the last sync was yesterday, we
    # don't re-pull 180 days of posts.
    if spec.min_age_days and cursor_col:
        from datetime import timedelta
        window_floor = (
            datetime.now(timezone.utc) - timedelta(days=spec.min_age_days)
        ).isoformat()
        effective_watermark = max(watermark, window_floor) if watermark else window_floor
        q = q.gte(cursor_col, effective_watermark)
    elif watermark and cursor_col:
        q = q.gte(cursor_col, watermark)
    if cursor_col:
        q = q.order(cursor_col)
    return q


# Max number of URNs per `in_()` filter. PostgREST encodes the list as
# `?col=in.(v1,v2,...)`; each LinkedIn URN is ~22 chars and httpx caps URL
# length at ~65k chars. 200 URNs per chunk leaves plenty of headroom even
# after URL-encoding.
_POST_URN_CHUNK = 200

# Per-row size ceiling for any mirrored text column. Rows whose text
# content exceeds this get dropped with a logged warning. Picked
# deliberately above realistic long-form content (3h Zoom ~200 KB,
# 50-page PDF extract ~300 KB, LinkedIn post ~6 KB) so false positives
# are rare, but below the "single row eats Stelle's context window"
# and "adversarial 5 MB prompt-injection payload" thresholds.
_MAX_TEXT_COL_BYTES = 1_048_576  # 1 MB
# Columns we check for size + prompt-injection heuristics. Anything
# else flows through untouched.
_RISKY_TEXT_COLS = (
    "transcript_text", "post_text", "extracted_text", "content",
    "message", "body", "text",
)
# Log-only heuristic phrases. Real prompt-injection attempts are
# trivially paraphrased, so we DON'T block — these are alarm bells that
# surface in logs so operators can spot coordinated attempts. False
# positives are expected (e.g. someone legitimately writing "ignore
# previous advice on X"); blocking would hurt more than it helps.
_PROMPT_INJECTION_SENTINELS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard prior instructions",
    "<|im_start|>",
    "<|im_end|>",
    "\n\nsystem:",
    "\n\nassistant:",
    "you are now",
    "new instructions:",
    "jailbreak",
)


def _row_is_too_big(row: dict) -> tuple[bool, str | None]:
    """Return (too_big, offending_col_and_size). None/None if fine."""
    for col in _RISKY_TEXT_COLS:
        v = row.get(col)
        if isinstance(v, str):
            b = len(v.encode("utf-8", errors="replace"))
            if b > _MAX_TEXT_COL_BYTES:
                return True, f"{col}={b}B"
    return False, None


def _scan_injection_sentinels(row: dict, table: str, row_id: str) -> None:
    """Log at WARN when a row's risky text columns contain any sentinel
    phrase. Never blocks. Designed to be noisy in aggregate (grep-able)
    and quiet on normal content."""
    hits: list[str] = []
    for col in _RISKY_TEXT_COLS:
        v = row.get(col)
        if not isinstance(v, str) or not v:
            continue
        v_low = v.lower()
        for s in _PROMPT_INJECTION_SENTINELS:
            if s in v_low:
                hits.append(f"{col}:{s!r}")
                break  # one hit per column is enough
    if hits:
        logger.warning(
            "[mirror_sync] prompt-injection sentinel hit  table=%s row_id=%s  %s",
            table, row_id, " | ".join(hits),
        )


# Cached per-table column allowlist for the destination (Amphoreus) side.
# Populated lazily on the first upsert per table per process. Used to
# strip columns the mirror schema doesn't have — protects against
# Jacquard schema drift (new column added upstream that we haven't
# mirrored yet, or stale columns lingering in PostgREST's schema cache).
# Without this, a single stray column kills the whole table's batch.
_DEST_COLUMNS_CACHE: dict[str, set[str]] = {}


def _dest_columns(amph, table: str) -> set[str] | None:
    """Return the set of column names the Amphoreus mirror table actually
    has, or None if we can't determine them. Cached per-process.

    Strategy: SELECT * LIMIT 1 — cheap, returns the row dict whose keys
    are the columns. If the table is empty we can't infer; return None
    and the caller skips stripping (fall back to the old "send everything,
    hope it sticks" behaviour, which still surfaces the error).
    """
    if table in _DEST_COLUMNS_CACHE:
        return _DEST_COLUMNS_CACHE[table]
    try:
        sample = amph.table(table).select("*").limit(1).execute().data
    except Exception:
        return None
    if sample:
        cols = set(sample[0].keys())
        _DEST_COLUMNS_CACHE[table] = cols
        return cols
    return None


def _strip_unknown_columns(rows: list[dict], allowed: set[str]) -> tuple[list[dict], set[str]]:
    """Return (cleaned_rows, dropped_keys). Mutates each row to keep
    only the keys in ``allowed``. Records dropped keys for the log so
    we can spot drift early."""
    dropped: set[str] = set()
    cleaned: list[dict] = []
    for r in rows:
        kept = {}
        for k, v in r.items():
            if k in allowed:
                kept[k] = v
            else:
                dropped.add(k)
        cleaned.append(kept)
    return cleaned, dropped


def _upsert_rows(spec, amph, rows: list[dict], result: SyncResult, offset: int) -> int:
    """Stamp + upsert a batch. Returns number of rows written.

    Defensive against schema drift: strips any column the destination
    mirror table doesn't have BEFORE the upsert. PostgREST returns
    PGRST204 ("Could not find the X column ...") for any unknown key,
    aborting the entire batch — even if 99% of the columns line up.
    Stripping unknown keys keeps the sync going, with a per-batch log
    entry so we can notice drift and update the schema deliberately.
    """
    now = _iso_now()
    # Size-cap + sentinel-log pass. Runs before stamping and the
    # schema-drift strip so the logs show raw table/row ids and we
    # don't waste cycles on a row we're about to drop.
    filtered: list[dict] = []
    oversized = 0
    for r in rows:
        row_id = str(
            r.get("id")
            or r.get("provider_urn")
            or r.get("provider_transcript_id")
            or r.get("post_id")
            or "?"
        )
        too_big, offender = _row_is_too_big(r)
        if too_big:
            logger.warning(
                "[mirror_sync] dropped oversized row  table=%s row_id=%s %s (>%dB cap)",
                spec.source_table, row_id, offender, _MAX_TEXT_COL_BYTES,
            )
            oversized += 1
            continue
        _scan_injection_sentinels(r, spec.source_table, row_id)
        filtered.append(r)
    rows = filtered
    if not rows:
        return 0
    for r in rows:
        r["_source"] = "jacquard"
        r["_synced_at"] = now

    allowed = _dest_columns(amph, spec.source_table)
    if allowed:
        rows, dropped = _strip_unknown_columns(rows, allowed)
        if dropped:
            logger.info(
                "[mirror_sync] %s: stripped %d unknown columns from upsert payload: %s",
                spec.source_table, len(dropped), sorted(dropped),
            )

    try:
        amph.table(spec.source_table).upsert(rows, on_conflict=spec.pk).execute()
    except Exception as exc:
        logger.error(
            "[mirror_sync] upsert %s (batch %d-%d, %d rows): %s",
            spec.source_table, offset, offset + len(rows), len(rows), exc,
        )
        if result.error is None:
            result.error = f"{spec.source_table}: {exc}"[:400]
        return 0

    # Engagement-snapshot tap. When this batch belongs to
    # ``linkedin_posts``, append one row per URN to
    # ``linkedin_post_engagement_snapshots`` so the kinetic trajectory
    # is preserved across Jacquard syncs (not just Amphoreus scrapes).
    # Best-effort; failures are swallowed by the helper. Runs AFTER the
    # main upsert so we don't pollute history with rows the upsert
    # rejected.
    if spec.source_table == "linkedin_posts":
        _write_linkedin_engagement_snapshots(rows, now)

    return len(rows)


def _write_linkedin_engagement_snapshots(
    rows: list[dict], scraped_at_iso: str,
) -> None:
    """Append engagement snapshots for a batch of Jacquard-sourced
    linkedin_posts rows. ``now`` timestamp is shared across the batch so
    rows from the same cursor page sort together in the time series.
    """
    try:
        from backend.src.db.amphoreus_supabase import record_engagement_snapshot
    except Exception:
        return
    for r in rows:
        urn = (r.get("provider_urn") or "").strip()
        if not urn:
            continue
        record_engagement_snapshot(
            provider_urn=    urn,
            scraped_at=      scraped_at_iso,
            total_reactions= r.get("total_reactions"),
            total_comments=  r.get("total_comments"),
            total_reposts=   r.get("total_reposts"),
            scraped_by=      "jacquard",
        )


def _sync_one_table(
    spec: _TableSpec,
    jcq,
    amph,
    scope_company_id: Optional[str],
    result: SyncResult,
    *,
    tracked_creators: Optional[list[str]] = None,
    tracked_post_urns: Optional[list[str]] = None,
) -> None:
    scope_label = scope_company_id or "global"
    watermark = _load_watermark(amph, spec.source_table, scope_label)
    cursor_col = spec.cursor_col or spec.cursor_fallback

    # Short-circuit if this table requires a filter set we don't have.
    # Prevents pulling the whole linkedin_reactions universe when we only
    # care about ~500 tracked posts.
    if spec.creator_col and not tracked_creators:
        result.rows_per_table[spec.source_table] = 0
        logger.info(
            "[mirror_sync] %s: skipped (no tracked_creators)", spec.source_table,
        )
        return
    if spec.post_urn_col and not tracked_post_urns:
        result.rows_per_table[spec.source_table] = 0
        logger.info(
            "[mirror_sync] %s: skipped (no tracked_post_urns; linkedin_posts not yet synced?)",
            spec.source_table,
        )
        return

    # For post_urn_col-filtered tables, the `.in_()` clause with 3k+ URNs
    # explodes the URL past httpx's limit. Chunk the URN list into
    # ``_POST_URN_CHUNK``-sized slices and sync each slice independently.
    # Watermark advance is done after the whole table finishes so a mid-
    # chunk failure doesn't skip rows on resume.
    if spec.post_urn_col and tracked_post_urns:
        pulled = 0
        next_cursor: Optional[str] = None
        for chunk_start in range(0, len(tracked_post_urns), _POST_URN_CHUNK):
            chunk = tracked_post_urns[chunk_start : chunk_start + _POST_URN_CHUNK]
            offset = 0
            while True:
                q = _build_query(
                    jcq, spec, scope_company_id, watermark,
                    tracked_creators=tracked_creators,
                    tracked_post_urns=chunk,
                )
                page = q.range(offset, offset + spec.page_size - 1).execute()
                rows = page.data or []
                if not rows:
                    break
                n = _upsert_rows(spec, amph, rows, result, offset)
                pulled += n
                if cursor_col:
                    for r in rows:
                        cv = r.get(cursor_col)
                        if cv and (next_cursor is None or str(cv) > str(next_cursor)):
                            next_cursor = str(cv)
                offset += len(rows)
        result.rows_per_table[spec.source_table] = pulled
        if next_cursor:
            _save_watermark(amph, spec.source_table, scope_label, next_cursor)
        return

    # Unfiltered (or creator-filter-only) pagination path.
    pulled = 0
    next_cursor: Optional[str] = None
    offset = 0
    while True:
        q = _build_query(
            jcq, spec, scope_company_id, watermark,
            tracked_creators=tracked_creators,
            tracked_post_urns=None,
        )
        page = q.range(offset, offset + spec.page_size - 1).execute()
        rows = page.data or []
        if not rows:
            break
        n = _upsert_rows(spec, amph, rows, result, offset)
        pulled += n
        if cursor_col:
            for r in rows:
                cv = r.get(cursor_col)
                if cv and (next_cursor is None or str(cv) > str(next_cursor)):
                    next_cursor = str(cv)
        # NOTE: Supabase caps PostgREST responses at ~1000 rows regardless
        # of the requested range. Exit only on empty.
        offset += len(rows)

    result.rows_per_table[spec.source_table] = pulled
    if next_cursor:
        _save_watermark(amph, spec.source_table, scope_label, next_cursor)


def _sync_reactor_profiles(jcq, amph, result: SyncResult) -> None:
    """Mirror linkedin_profiles by PK lookup — same pattern jacquard_direct
    uses at read time.

    Jacquard's ``linkedin_profiles`` is large enough (tens of millions of
    rows) that an ``ORDER BY created_at`` full scan hits Supabase's
    statement timeout before the first page returns. We don't need a
    full scan: Cyrene's reactor analysis only looks up a profile's
    name/headline/about when that profile actually reacted to or
    commented on one of our tracked posts. Collect the distinct URNs
    from our already-mirrored ``linkedin_reactions`` + ``linkedin_comments``
    and pull only those rows from Jacquard.
    """
    # Distinct profile URNs from mirrored reactions + comments.
    #
    # The offset pagination here was hitting Supabase's default statement
    # timeout on large-offset reads once the mirror accumulated ~55k
    # reactions — PostgREST translates .range() to ``LIMIT/OFFSET`` which
    # gets progressively slower as offset grows. Two changes to make this
    # resilient:
    #   * per-page try/except — one slow page no longer aborts the whole
    #     function; we just proceed with whatever URNs we already have and
    #     catch up on the next run,
    #   * smaller page size (500) — keeps each page under the timeout.
    urns: set[str] = set()
    for table, col in (
        ("linkedin_reactions", "provider_profile_urn"),
        ("linkedin_comments", "provider_profile_urn"),
    ):
        offset = 0
        page = 500
        while True:
            try:
                resp = (
                    amph.table(table)
                        .select(col)
                        .not_.is_(col, "null")
                        .range(offset, offset + page - 1)
                        .execute()
                )
            except Exception as exc:
                logger.warning(
                    "[mirror_sync] linkedin_profiles: %s pagination at offset=%d failed (%s); "
                    "continuing with %d URNs collected so far",
                    table, offset, exc, len(urns),
                )
                break
            rows = resp.data or []
            if not rows:
                break
            urns.update(r.get(col) for r in rows if r.get(col))
            if len(rows) < page:
                break
            offset += len(rows)
    logger.info(
        "[mirror_sync] linkedin_profiles: %d distinct reactor/commenter URNs to fetch",
        len(urns),
    )
    if not urns:
        result.rows_per_table["linkedin_profiles"] = 0
        return

    # PK batch lookup against Jacquard
    urn_list = list(urns)
    pulled = 0
    chunk = 100  # matches jacquard_direct's _POST_URN_CHUNK pattern
    for i in range(0, len(urn_list), chunk):
        batch = urn_list[i : i + chunk]
        try:
            resp = (
                jcq.table("linkedin_profiles")
                   .select("*")
                   .in_("provider_urn", batch)
                   .execute()
            )
        except Exception as exc:
            logger.warning(
                "[mirror_sync] linkedin_profiles chunk %d-%d fetch failed: %s",
                i, i + len(batch), exc,
            )
            continue
        rows = resp.data or []
        if not rows:
            continue
        now = _iso_now()
        for r in rows:
            r["_source"] = "jacquard"
            r["_synced_at"] = now
        # Same schema-drift protection as _upsert_rows.
        allowed = _dest_columns(amph, "linkedin_profiles")
        if allowed:
            rows, dropped = _strip_unknown_columns(rows, allowed)
            if dropped:
                logger.info(
                    "[mirror_sync] linkedin_profiles: stripped %d unknown columns: %s",
                    len(dropped), sorted(dropped),
                )
        try:
            amph.table("linkedin_profiles").upsert(
                rows, on_conflict="provider_urn",
            ).execute()
            pulled += len(rows)
        except Exception as exc:
            logger.warning(
                "[mirror_sync] linkedin_profiles upsert %d rows failed: %s",
                len(rows), exc,
            )
    result.rows_per_table["linkedin_profiles"] = pulled
    logger.info("[mirror_sync] linkedin_profiles: %d rows", pulled)


def _copy_transcript_bodies(amph, result: SyncResult) -> None:
    """Copy any meeting transcript bodies that haven't been mirrored yet
    to Amphoreus Supabase Storage.

    **Full migration contract (2026-04-23)**: every row with a
    ``gcs_transcript_url`` gets its body copied, not just the first
    100. The inner pagination loops until there are no more uncopied
    rows OR we've hit a soft wall-clock budget. Reason: Jacquard's
    GCS is the last piece of Jacquard-owned infrastructure that
    Stelle / Castorice / Irontomb transcript reads still touch;
    closing the gap means the agent transcript-read path is
    100% Amphoreus-owned (Supabase + Supabase Storage). Jacquard can
    rotate / revoke GCS access without breaking reads.

    Jacquard's ``meetings`` table uses ``provider_transcript_id`` as
    its PK and stores the transcript body URL in ``gcs_transcript_url``
    — matches the main-schema DDL.

    Time budget: 10 minutes per sync run. One GCS round-trip is
    ~0.5-2s (network + decode), so ~300-1000 transcripts per cycle
    in the worst case, and full completion on the next cron if the
    wall-clock wins first. Cheap safety guard against a stuck GCS
    response tying up the sync forever.
    """
    bucket = os.environ.get("AMPHOREUS_TRANSCRIPTS_BUCKET", "transcripts")
    PAGE_SIZE = 100
    MAX_COPY_SECONDS = 10 * 60  # 10 min per run
    MAX_ITER = 200               # ~20k rows safety cap per run

    deadline = time.time() + MAX_COPY_SECONDS
    copied_total = 0
    iter_count = 0
    while iter_count < MAX_ITER:
        iter_count += 1
        if time.time() >= deadline:
            logger.info(
                "[mirror_sync] transcript-copy hit time budget at %d rows "
                "after %d pages — remainder picks up next cron",
                copied_total, iter_count,
            )
            break
        try:
            # ``not_.is_("null")`` only excludes true NULLs; empty-string
            # gcs_transcript_url rows slip through and waste a pass. Add
            # an explicit ``neq("")`` so the query only returns rows we
            # can actually migrate. Same for provider_transcript_id —
            # if the PK is empty we can't update the row anyway.
            resp = (
                amph.table("meetings")
                    .select("provider_transcript_id, gcs_transcript_url")
                    .is_("_storage_path", "null")
                    .not_.is_("gcs_transcript_url", "null")
                    .neq("gcs_transcript_url", "")
                    .not_.is_("provider_transcript_id", "null")
                    .neq("provider_transcript_id", "")
                    .limit(PAGE_SIZE)
                    .execute()
            )
        except Exception as exc:
            logger.debug("[mirror_sync] transcript-copy query skipped: %s", exc)
            break

        rows = resp.data or []
        if not rows:
            # Empty page → every row with a gcs_url now has _storage_path.
            # This is the terminal condition we want on a steady state.
            logger.info(
                "[mirror_sync] transcript-copy fully caught up (total this run: %d)",
                copied_total,
            )
            break

        page_copied = 0
        for row in rows:
            if time.time() >= deadline:
                break
            gcs_url = row.get("gcs_transcript_url")
            meeting_id = row.get("provider_transcript_id")
            if not (gcs_url and meeting_id):
                continue
            body = _gcs_download(gcs_url)
            if not body:
                # Common case: the GCS object was deleted or moved.
                # Mark the row with a sentinel so the next page skips
                # it instead of spinning forever. We set _storage_path
                # to empty string (not null) so the query above
                # ``is_("_storage_path", "null")`` passes over it.
                try:
                    amph.table("meetings").update(
                        {"_storage_path": ""}
                    ).eq("provider_transcript_id", meeting_id).execute()
                except Exception:
                    pass
                continue
            mime = "text/markdown"
            path = f"jacquard/{meeting_id}"
            if _storage_upload(amph, bucket, path, body, mime):
                amph.table("meetings").update(
                    {"_storage_path": f"{bucket}/{path}"}
                ).eq("provider_transcript_id", meeting_id).execute()
                page_copied += 1

        copied_total += page_copied
        if page_copied == 0:
            # Guard: page returned rows but none got copied (e.g. all
            # GCS downloads failed). Don't loop forever on the same
            # page — break and let next cron retry.
            logger.warning(
                "[mirror_sync] transcript-copy page returned 0 successful copies "
                "out of %d candidates; ending run early",
                len(rows),
            )
            break

    result.gcs_objects_copied += copied_total


def _load_virio_serviced_company_ids(amph) -> set[str]:
    """Company UUIDs Virio actually writes content for.

    Ground-truth signal: a company is "Virio-serviced" iff Amphoreus has
    generated at least one draft for it (``local_posts`` row exists).
    This is self-bootstrapping — the first time we generate for a new
    client, the next mirror cycle picks them up automatically (≤8h lag).

    Built 2026-04-23 as part of the mirror-scoping cleanup. Before this,
    ``_load_tracked_creators`` returned every ``users.posts_content=true``
    row across Jacquard's multi-tenant users table — which included
    FOCs from OTHER Jacquard customers (areganti / kevinparknz /
    growthspecialist etc.) that Virio has no relationship with. Result
    was ~58% of the mirrored linkedin_posts were non-Virio creators,
    polluting voice-reference reads and wasting mirror space.
    """
    try:
        rows = (
            amph.table("local_posts").select("company")
                .limit(5000).execute().data
            or []
        )
    except Exception as exc:
        logger.warning("[mirror_sync] Virio-company lookup failed: %s", exc)
        return set()
    return {r["company"] for r in rows if r.get("company")}


def _load_tracked_creators(jcq, amph=None) -> list[str]:
    """LinkedIn usernames of every FOC user Virio writes content for.

    Pulled live from Jacquard's ``users`` table with TWO filters:

      1. ``posts_content=true`` — Jacquard's own "we write posts for
         this person" flag.
      2. ``company_id IN <Virio-serviced companies>`` — narrows from
         Jacquard's global multi-tenant FOC set to just the companies
         Amphoreus has actually generated drafts for. Without this,
         we vacuum up other Jacquard customers' FOCs and mirror their
         LinkedIn posts (the 2026-04-22 leakage audit found 11 non-Virio
         creators dominating 58% of mirrored posts — `areganti` alone
         was 23% of the entire mirror).

    ``is_internal`` is deliberately NOT filtered: Virio teammates who
    have Stelle generate their posts need their own LinkedIn history
    mirrored so Stelle can voice-match.

    We extract the username from ``linkedin_url`` with a simple regex —
    Amphoreus doesn't store a dedicated ``linkedin_username`` column.
    """
    import re as _re
    # Resolve Virio-serviced company set first. If we can't reach the
    # Amphoreus mirror to read it, fall back to the old global filter
    # (better to over-mirror than break the sync entirely).
    virio_company_ids: set[str] = set()
    if amph is not None:
        virio_company_ids = _load_virio_serviced_company_ids(amph)
    if not virio_company_ids:
        logger.warning(
            "[mirror_sync] Virio-serviced company set empty — falling back "
            "to global posts_content=true filter (may include other "
            "Jacquard customers' FOCs)"
        )

    try:
        q = (
            jcq.table("users")
               .select("linkedin_url,posts_content,is_internal,company_id")
               .eq("posts_content", True)
               .limit(500)
        )
        if virio_company_ids:
            q = q.in_("company_id", list(virio_company_ids))
        resp = q.execute()
    except Exception as exc:
        logger.warning("[mirror_sync] _load_tracked_creators failed: %s", exc)
        return []
    out: set[str] = set()
    for u in resp.data or []:
        url_v = (u.get("linkedin_url") or "").strip()
        m = _re.search(r"linkedin\.com/in/([^/?#]+)", url_v)
        if m:
            out.add(m.group(1).strip().lower().rstrip("/"))
    logger.info(
        "[mirror_sync] tracked_creators: %d creators from %d Virio-serviced companies",
        len(out), len(virio_company_ids),
    )
    return sorted(out)


def _load_tracked_post_urns(amph, creators: list[str]) -> list[str]:
    """Every provider_urn in our mirrored linkedin_posts — the whitelist
    used to scope linkedin_reactions / linkedin_comments."""
    if not creators:
        return []
    urns: list[str] = []
    offset = 0
    page_size = 1000
    while True:
        resp = (
            amph.table("linkedin_posts")
                .select("provider_urn")
                .in_("creator_username", creators)
                .range(offset, offset + page_size - 1)
                .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        urns.extend(r["provider_urn"] for r in rows if r.get("provider_urn"))
        if len(rows) < page_size:
            break
        offset += page_size
    return urns


def run_sync(
    *,
    kind: str = "scheduled",
    scope_company_id: Optional[str] = None,
    triggered_by: Optional[str] = None,
    skip_tables: Optional[set[str]] = None,
) -> SyncResult:
    """Run one mirror sync cycle. Returns a SyncResult (logged to
    ``sync_runs`` as well for UI visibility)."""
    result = SyncResult(
        kind=kind,
        scope=scope_company_id or "global",
        triggered_by=triggered_by or ("cron" if kind == "scheduled" else None),
    )
    skip_tables = skip_tables or set()

    try:
        jcq = _jacquard_client()
        amph = _amphoreus_client()
    except Exception as exc:
        result.status = "failed"
        result.error = f"client init: {exc}"[:400]
        result.completed_at = time.time()
        logger.exception("[mirror_sync] client init failed")
        _log_sync_run(result)
        return result

    _ensure_sync_state_table(amph)

    # Record a 'running' row up front so cancellations can be detected.
    run_id = _log_sync_run(result)

    # Pass ``amph`` so tracked_creators can be narrowed to Virio-serviced
    # companies only (see _load_tracked_creators docstring + 2026-04-22
    # leakage audit). Without this, the mirror pulls FOCs from Jacquard's
    # whole multi-tenant users table.
    tracked_creators = _load_tracked_creators(jcq, amph=amph)

    try:
        for spec in _TABLES:
            if spec.source_table in skip_tables:
                continue
            # Tables that reference posts need the URN whitelist rebuilt
            # once linkedin_posts has landed — compute it lazily right
            # before the first table that needs it.
            tracked_post_urns: Optional[list[str]] = None
            if spec.post_urn_col:
                tracked_post_urns = _load_tracked_post_urns(amph, tracked_creators)
                logger.info(
                    "[mirror_sync] tracked_post_urns (for %s): %d",
                    spec.source_table, len(tracked_post_urns),
                )
            try:
                _sync_one_table(
                    spec, jcq, amph, scope_company_id, result,
                    tracked_creators=tracked_creators if spec.creator_col else None,
                    tracked_post_urns=tracked_post_urns,
                )
                logger.info(
                    "[mirror_sync] %s: %d rows",
                    spec.source_table, result.rows_per_table.get(spec.source_table, 0),
                )
            except Exception as exc:
                logger.exception("[mirror_sync] table %s crashed", spec.source_table)
                if result.error is None:
                    result.error = f"{spec.source_table}: {exc}"[:400]

        # linkedin_profiles post-pass: derived from reactions+comments URNs,
        # PK-batched, never full-scans the Jacquard source.
        try:
            _sync_reactor_profiles(jcq, amph, result)
        except Exception as exc:
            logger.exception("[mirror_sync] reactor-profiles pass crashed")
            if result.error is None:
                result.error = f"linkedin_profiles: {exc}"[:400]

        _copy_transcript_bodies(amph, result)

        # Semantic draft-match-back. Runs after linkedin_posts is fresh so
        # the latest published posts can be cosine-matched against our
        # Stelle drafts in ``local_posts``. Fully non-fatal — any crash
        # here logs and moves on; next 8h cycle retries.
        try:
            from backend.src.services.draft_match_worker import run_match_back
            match_summary = run_match_back()
            # Surface the new-match count on the run row for easy monitoring.
            result.rows_per_table["draft_matches"] = match_summary.get("matched", 0)
            if match_summary.get("errors"):
                logger.warning(
                    "[mirror_sync] draft_match_worker ok but had %d partial errors: %s",
                    len(match_summary["errors"]),
                    match_summary["errors"][:3],
                )
        except Exception as exc:
            logger.exception("[mirror_sync] draft_match_worker crashed (non-fatal)")
            if result.error is None:
                result.error = f"draft_match_worker: {exc}"[:400]

        result.status = "completed" if result.error is None else "completed_with_errors"
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)[:400]
        logger.exception("[mirror_sync] run failed")
    finally:
        result.completed_at = time.time()
        _log_sync_run(result, run_id=run_id)

    return result


def _log_sync_run(result: SyncResult, run_id: Optional[str] = None) -> Optional[str]:
    """Insert or update a ``sync_runs`` row. Returns the run id so the
    caller can pass it back on completion."""
    try:
        amph = _amphoreus_client()
    except Exception:
        return run_id

    payload = {
        "status": result.status,
        "kind": result.kind,
        "scope": result.scope,
        "rows_per_table": result.rows_per_table,
        "gcs_objects_copied": result.gcs_objects_copied,
        "error": result.error,
        "triggered_by": result.triggered_by,
        "completed_at": datetime.fromtimestamp(result.completed_at, tz=timezone.utc).isoformat() if result.completed_at else None,
    }
    try:
        if run_id:
            amph.table("sync_runs").update(payload).eq("id", run_id).execute()
            return run_id
        payload["id"] = str(uuid.uuid4())
        payload["started_at"] = datetime.fromtimestamp(result.started_at, tz=timezone.utc).isoformat()
        amph.table("sync_runs").insert(payload).execute()
        return payload["id"]
    except Exception as exc:
        logger.debug("[mirror_sync] sync_runs log failed: %s", exc)
        return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-company", default=None, help="UUID to scope sync to one company")
    parser.add_argument("--kind", default="scheduled", choices=("scheduled", "on_demand"))
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated table names to skip (e.g. 'linkedin_reactions,linkedin_comments')",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load .env for local runs
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                break
    except Exception:
        pass

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    result = run_sync(kind=args.kind, scope_company_id=args.scope_company, skip_tables=skip)

    logger.info("DONE status=%s rows=%s storage=%d error=%s",
                result.status, result.rows_per_table, result.gcs_objects_copied, result.error)
    sys.exit(0 if result.status.startswith("completed") else 1)


if __name__ == "__main__":
    main()
