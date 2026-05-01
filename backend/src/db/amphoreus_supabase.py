"""Amphoreus-owned Supabase client — the write path for artifacts
Amphoreus authors (and the read path for anything that consumes them).

Unlike ``jacquard_direct`` (read-only mirror of Jacquard's Supabase),
this module owns tables that Amphoreus alone writes to:

  * ``cyrene_briefs``  — Cyrene's strategic briefs. Written once per
                         Cyrene run, read by Tribbie to bootstrap each
                         interview.
  * (future) ``drafts``, ``drafts_annotations``, etc. — once the full
    SQLite-local_posts migration lands.

Connection env vars (required):
  ``AMPHOREUS_SUPABASE_URL``
  ``AMPHOREUS_SUPABASE_KEY`` (service role — writes need it)

All functions no-op + log-warn when the client isn't configured (local
dev without Supabase, tests). Callers should treat a ``None`` return as
"not persisted to Supabase, fall back to local file."
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_client = None  # lazy-constructed singleton


def is_configured() -> bool:
    """True when both URL + service-role key are set in the environment."""
    return bool(
        os.environ.get("AMPHOREUS_SUPABASE_URL", "").strip()
        and os.environ.get("AMPHOREUS_SUPABASE_KEY", "").strip()
    )


def _get_client():
    """Lazy-construct (and cache) the Amphoreus Supabase client.

    Returns ``None`` if env isn't configured — the caller is expected
    to degrade gracefully (fall back to fly-local file, or skip the
    write entirely).
    """
    global _client
    if _client is not None:
        return _client
    if not is_configured():
        return None
    try:
        from supabase import create_client
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] supabase-py import failed: %s — "
            "check that the package is installed.",
            exc,
        )
        return None
    try:
        _client = create_client(
            os.environ["AMPHOREUS_SUPABASE_URL"].rstrip("/"),
            os.environ["AMPHOREUS_SUPABASE_KEY"],
        )
    except Exception as exc:
        logger.warning("[amphoreus_supabase] create_client failed: %s", exc)
        return None
    return _client


# ---------------------------------------------------------------------------
# deletion_log — audit every data-deletion across Amphoreus
# ---------------------------------------------------------------------------
#
# Single write surface: every call site that deletes content the operator
# might later ask about should call log_deletion(). The helper is
# intentionally permissive — snapshot is optional, write failures don't
# raise — so the audit trail can't become a blocker on the delete itself.
# Schema:
#   (id, entity_type, entity_id, entity_snapshot, deleted_by, reason,
#    path, deleted_at) — see amphoreus_supabase_mirror_schema.sql.


def log_deletion(
    *,
    entity_type: str,
    entity_id: Any,
    entity_snapshot: Optional[dict[str, Any]] = None,
    deleted_by: str = "system",
    reason: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Append a row to ``deletion_log``. Never raises.

    ``entity_type`` is a short tag — 'local_post' | 'meeting' |
    'meeting_participant' | 'context_file' | 'draft_feedback' |
    'storage_blob' | 'transcript_file' | 'wipe_batch'.

    ``entity_snapshot`` is the row content at deletion time (for undo /
    audit); pass ``None`` for file-system or storage deletes where the
    path alone is the record.

    ``deleted_by`` defaults to ``'system'`` for background jobs; HTTP
    callers should pass the operator email resolved from the request's
    auth context.
    """
    sb = _get_client()
    if sb is None:
        return
    try:
        sb.table("deletion_log").insert({
            "entity_type":     entity_type,
            "entity_id":       str(entity_id) if entity_id is not None else "",
            "entity_snapshot": entity_snapshot,
            "deleted_by":      deleted_by,
            "reason":          reason,
            "path":            path,
        }).execute()
    except Exception as exc:
        logger.debug(
            "[deletion_log] insert failed (%s/%s): %s",
            entity_type, entity_id, exc,
        )


# ---------------------------------------------------------------------------
# cyrene_briefs
# ---------------------------------------------------------------------------

def save_cyrene_brief(
    company: str,
    brief: dict[str, Any],
    created_by: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Insert a strategic brief. Returns the saved row on success, None
    on failure (callers log + keep going; Cyrene also still writes the
    fly-local file as a belt-and-braces fallback).

    ``user_id`` scopes the brief to one FOC user at a multi-FOC company
    (2026-04-22 Virio fix — a single company-wide brief can't direct
    19 distinct-role teammates, so Cyrene now runs per-FOC when a user
    is targeted). When ``user_id`` is None the brief is company-wide
    (backward-compatible for single-FOC clients like Hume/Innovo/Flora).
    Requires the schema migration ``ALTER TABLE cyrene_briefs ADD
    COLUMN user_id text`` — the insert only includes the column when
    user_id is set, so unmigrated deployments still work for
    company-wide saves.
    """
    sb = _get_client()
    if sb is None:
        return None
    try:
        row = (
            sb.table("cyrene_briefs")
            .insert({
                "company": company,
                "brief": brief,
                **({"created_by": created_by} if created_by else {}),
                **({"run_id": run_id} if run_id else {}),
                **({"user_id": user_id} if user_id else {}),
            })
            .execute()
            .data
        )
        if row:
            return row[0]
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] save_cyrene_brief(%s, user_id=%s) failed: %s",
            company, user_id, exc,
        )
    return None


def get_client_transcripts(
    company: str, limit: int = 30,
) -> list[dict[str, Any]]:
    """Return all transcripts associated with ``company`` from the
    ``meetings`` table — the same source Stelle/Cyrene/the Transcripts
    tab use. Covers BOTH:

    * Jacquard-scraped meetings (via 3-hop participants→users join)
    * Amphoreus-uploaded meetings (via jsonb company tag)
    * Tribbie-authored transcripts uploaded post-session with
      ``meeting_subtype`` set (these are just Amphoreus-uploaded rows)

    Unified shape per row: ``{filename, text, posted_at,
    duration_seconds, meeting_subtype, source}``. Newest first.

    Used by Tribbie for past-session context (dedup + Whisper vocab
    seed) so Tribbie reads from the single canonical transcript store,
    not a bespoke ``tribbie_transcripts`` silo.
    """
    sb = _get_client()
    if sb is None:
        return []

    # Resolve to (company_uuid, user_id) so we can per-FOC-scope the
    # Jacquard join — same pattern /api/mirror/transcripts uses.
    try:
        from backend.src.lib.company_resolver import resolve_to_company_and_user
        company_uuid, user_id = resolve_to_company_and_user(company)
    except Exception:
        company_uuid, user_id = None, None
    company_uuid = company_uuid or company  # legacy raw-UUID / raw-slug fallback

    # Bucket 1: Amphoreus-uploaded (including Tribbie post-session writes)
    amph_rows: list[dict[str, Any]] = []
    try:
        filter_blob: dict[str, Any] = {"amphoreus_company_id": company_uuid}
        if user_id:
            filter_blob["amphoreus_user_id"] = user_id
        amph_rows = (
            sb.table("meetings")
              .select("provider_transcript_id, name, transcript_text, start_time, created_at, duration_seconds, calendar_attendees, _source")
              .eq("_source", "amphoreus")
              .contains("calendar_attendees", filter_blob)
              .order("created_at", desc=True)
              .limit(max(1, min(limit, 500)))
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.debug("[amphoreus_supabase] get_client_transcripts amph: %s", exc)

    # Bucket 2: Jacquard-sourced via 3-hop join.
    #
    # Previously this whole block required ``user_id`` to be already
    # resolved by the caller. Cyrene calls ``_query_transcript_inventory``
    # with whatever string CYRENE_COMPANY held, and run_cyrene_cli was
    # passing the bare company UUID — so resolve_to_company_and_user
    # returned (uuid, None), Bucket 2 was skipped, and Cyrene saw 1
    # transcript when Hensley actually had 25. Surfaced 2026-04-28.
    #
    # Fix: when user_id wasn't resolved up-front, fall back to
    # collecting EVERY user's email under this company and union the
    # transcripts they participated in. At single-FOC clients
    # (Hensley, Andrew, Sachil) this is identical to per-user scoping
    # because there's only one user. At multi-FOC clients (Trimble,
    # Commenda, Virio) it surfaces the company-wide transcript pool —
    # operator-side context Cyrene needs for cross-FOC strategic
    # awareness even when she's running scoped to one FOC.
    jacq_rows: list[dict[str, Any]] = []
    try:
        emails: list[str] = []
        if user_id:
            email_row = (
                sb.table("users").select("email").eq("id", user_id).limit(1).execute().data
                or []
            )
            emails = [e["email"] for e in email_row if e.get("email")]
        else:
            # Bare-UUID fallback: pull every user under this company and
            # union their transcript participation. Slightly wider than
            # per-FOC mode but never NARROWER — Cyrene needs everything
            # the client/team has been on a call about.
            user_rows = (
                sb.table("users").select("email").eq("company_id", company_uuid).limit(50).execute().data
                or []
            )
            emails = [e["email"] for e in user_rows if e.get("email")]
            if emails:
                logger.debug(
                    "[amphoreus_supabase] get_client_transcripts: bare-UUID "
                    "fallback resolved %d user emails for company=%s",
                    len(emails), company_uuid,
                )
        if emails:
            parts = (
                sb.table("meeting_participants")
                  .select("provider_transcript_id")
                  .in_("email", emails)
                  .limit(2000)
                  .execute()
                  .data
                or []
            )
            tids = sorted({p["provider_transcript_id"] for p in parts if p.get("provider_transcript_id")})
            # Chunk IN-list to avoid PostgREST URL-length 400s.
            _CHUNK = 100
            for i in range(0, len(tids), _CHUNK):
                resp = (
                    sb.table("meetings")
                      .select("provider_transcript_id, name, transcript_text, start_time, created_at, duration_seconds, calendar_attendees, _source")
                      .in_("provider_transcript_id", tids[i:i + _CHUNK])
                      .execute()
                      .data
                    or []
                )
                jacq_rows.extend(resp)
    except Exception as exc:
        logger.debug("[amphoreus_supabase] get_client_transcripts jacq: %s", exc)

    # Dedupe on provider_transcript_id (prefer amphoreus when overlap).
    seen: dict[str, dict] = {}
    for r in amph_rows:
        seen[r["provider_transcript_id"]] = r
    for r in jacq_rows:
        seen.setdefault(r["provider_transcript_id"], r)

    def _shape(r: dict) -> dict:
        attend = r.get("calendar_attendees") or {}
        if not isinstance(attend, dict):
            attend = {}
        return {
            "filename": attend.get("amphoreus_filename") or r.get("name") or "(untitled)",
            "text": r.get("transcript_text") or "",
            "posted_at": r.get("start_time") or r.get("created_at"),
            "duration_seconds": r.get("duration_seconds"),
            "meeting_subtype": attend.get("amphoreus_meeting_subtype"),
            "source": r.get("_source") or "jacquard",
        }

    out = [_shape(r) for r in seen.values() if (r.get("transcript_text") or "").strip()]
    out.sort(key=lambda x: x.get("posted_at") or "", reverse=True)
    return out[:limit]


def get_client_context_files(
    company: str,
    limit: int = 50,
    *,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return operator-uploaded context files for ``company`` from the
    ``context_files`` mirror table (the same store that backs Stelle's
    ``<slug>/context/`` mount via :func:`jacquard_direct.fetch_context_files`).

    Used by Tribbie so Tribbie has read parity with Stelle — positioning
    PDFs, brand decks, one-pagers etc. that live under context/ should
    be available during live content interviews, not just to Stelle.

    **Per-FOC scoping** (``user_id`` kwarg): when set, returns only rows
    that are either explicitly tagged to this user OR company-wide
    (``user_id IS NULL``). Company-wide rows are mostly Jacquard-
    mirrored context (Jacquard has no per-FOC column upstream) and
    Amphoreus-uploaded rows written before per-FOC scoping existed;
    keeping them visible preserves backwards compatibility. When
    ``user_id`` is None (callers that haven't been updated, or
    genuinely company-wide queries) we return every row under the
    company — old behaviour, so nothing breaks.

    Each row shape: ``{filename, text, content_type, size_bytes,
    created_at}``. Empty list on any failure (callers should treat
    context as optional).
    """
    sb = _get_client()
    if sb is None:
        return []

    # Resolve slug → UUID; ``context_files.company_id`` is a UUID.
    try:
        from backend.src.lib.company_resolver import resolve_with_fallback
        company_uuid = resolve_with_fallback(company) or company
    except Exception:
        company_uuid = company

    def _query(with_user_filter: bool):
        q = (
            sb.table("context_files")
              .select("filename, extracted_text, content_type, size_bytes, created_at")
              .eq("company_id", company_uuid)
        )
        if with_user_filter:
            # PostgREST ``or`` syntax: return rows tagged to this user
            # OR company-wide (NULL user_id). The comma inside or_()
            # is treated as disjunction.
            q = q.or_(f"user_id.eq.{user_id},user_id.is.null")
        return (
            q.order("created_at", desc=True)
             .limit(max(1, min(limit, 500)))
             .execute()
             .data
            or []
        )

    try:
        rows = _query(with_user_filter=bool(user_id))
    except Exception as exc:
        msg = str(exc)
        # Tolerate the pre-migration state: if ``user_id`` column
        # doesn't exist yet, fall back to unfiltered read. Log once
        # per call so operators notice they need to run the SQL.
        if user_id and "user_id" in msg and ("column" in msg.lower() or "PGRST204" in msg):
            logger.warning(
                "[amphoreus_supabase] context_files.user_id not yet migrated; "
                "returning company-wide rows (run schema addendum SQL to enable per-FOC scoping)."
            )
            try:
                rows = _query(with_user_filter=False)
            except Exception as retry_exc:
                logger.debug(
                    "[amphoreus_supabase] get_client_context_files(%s) retry failed: %s",
                    company, retry_exc,
                )
                return []
        else:
            logger.debug(
                "[amphoreus_supabase] get_client_context_files(%s) failed: %s",
                company, exc,
            )
            return []

    out: list[dict[str, Any]] = []
    for r in rows:
        text = (r.get("extracted_text") or "").strip()
        if not text:
            continue
        out.append({
            "filename": r.get("filename") or "untitled.txt",
            "text": text,
            "content_type": r.get("content_type"),
            "size_bytes": r.get("size_bytes"),
            "created_at": r.get("created_at"),
        })
    return out


def save_tribbie_transcript(
    company: str,
    *,
    file_name: str,
    text: str,
    started_at: datetime,
    duration_seconds: int,
    segment_count: int,
    source: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Mirror a finalized Tribbie interview transcript to Amphoreus
    Supabase. Returns the saved row on success, ``None`` on failure
    (Supabase not configured, network error, etc.).

    The fly-local .txt file remains the durable source of truth — this
    mirror exists so Cyrene and operators on other machines can read
    the transcript without SSH'ing into the capture host.
    """
    sb = _get_client()
    if sb is None:
        return None
    # Normalize to UTC-aware ISO so Postgres timestamptz has a tz.
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    try:
        row = (
            sb.table("tribbie_transcripts")
            .insert({
                "company": company,
                "file_name": file_name,
                "text": text,
                "started_at": started_at.isoformat(),
                "duration_seconds": duration_seconds,
                "segment_count": segment_count,
                **({"source": source} if source else {}),
            })
            .execute()
            .data
        )
        if row:
            return row[0]
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] save_tribbie_transcript(%s, %s) failed: %s",
            company, file_name, exc,
        )
    return None


def get_latest_cyrene_brief(
    company: str,
    user_id: Optional[str] = None,
    strict_user_only: bool = False,
) -> Optional[dict[str, Any]]:
    """Return the newest brief for ``company`` (the ``brief`` jsonb
    payload, not the row envelope) or ``None`` if none exists / fetch
    fails. Used by Tribbie to bootstrap interviews and by reports to
    show current strategy.

    Cyrene canonicalizes company to its UUID before writing (see
    ``strategy.py::run_cyrene_review`` → ``resolve_with_fallback``).
    Tribbie (and other read-side callers) may pass either the slug or
    the UUID, so we try both. The lookup order is:

      1. As-given (handles both slug-stored and UUID-stored rows).
      2. Resolved (slug → UUID) via ``resolve_with_fallback``.

    ``user_id`` resolves for multi-FOC companies.

    ``strict_user_only`` (default False) controls fallback behavior —
    callers that never want a company-wide brief pass ``True``:

      * strict=True, user_id set → per-user brief ONLY. If no
        per-user brief exists, return ``None``. No company-wide
        fallback. This is the mode Tribbie + report use: a company-
        wide brief at a multi-FOC client is policy-incorrect
        (strategic averaging across divergent roles), so those
        consumers refuse to read it.
      * strict=True, user_id is None → return ``None`` immediately
        (strict mode with no user target is an operator error).
      * strict=False, user_id set → per-user preferred, falls back
        to company-wide if no per-user brief exists. Legacy default
        used by Stelle-adjacent code paths.
      * strict=False, user_id None → return the latest brief
        regardless of user_id. The legacy single-FOC path.
    """
    sb = _get_client()
    if sb is None:
        return None

    # Strict mode without a target user is a caller error — the caller
    # opted into "only read FOC-specific briefs" but didn't provide a
    # FOC. Return None rather than silently returning a company-wide
    # brief.
    if strict_user_only and not user_id:
        logger.debug(
            "[amphoreus_supabase] get_latest_cyrene_brief(%s) strict_user_only "
            "without user_id — returning None",
            company,
        )
        return None

    candidates: list[str] = []
    if company:
        candidates.append(company)
    try:
        from backend.src.lib.company_resolver import resolve_with_fallback
        resolved = resolve_with_fallback(company)
        if resolved and resolved != company:
            candidates.append(resolved)
    except Exception:
        pass

    # Per-user lookup. Note: the user_id column may not exist yet on
    # environments that haven't run the schema migration — the except
    # below catches that and falls through. Code path stays safe
    # either way.
    if user_id:
        for key in candidates:
            try:
                rows = (
                    sb.table("cyrene_briefs")
                    .select("brief")
                    .eq("company", key)
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                    .data
                ) or []
            except Exception as exc:
                logger.debug(
                    "[amphoreus_supabase] per-user cyrene lookup failed (%s, user=%s): %s",
                    key, user_id, exc,
                )
                continue
            if rows:
                return rows[0].get("brief")
        # No per-user brief found.
        if strict_user_only:
            # Caller refuses to fall back to company-wide. Return
            # None so the caller skips the brief block entirely.
            return None
        # Legacy mode: fall through to company-wide read.

    for key in candidates:
        try:
            rows = (
                sb.table("cyrene_briefs")
                .select("brief")
                .eq("company", key)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
                .data
            ) or []
        except Exception as exc:
            logger.warning(
                "[amphoreus_supabase] get_latest_cyrene_brief(%s) failed: %s",
                key, exc,
            )
            continue
        if rows:
            return rows[0].get("brief")
    return None


# ---------------------------------------------------------------------------
# local_posts — Stelle drafts / manual posts / Hyacinthia push targets
# ---------------------------------------------------------------------------
#
# Every op here mirrors the SQLite-backed fn in backend/src/db/local.py.
# The fn pair is called in tandem (dual-write / read-preferred) during the
# migration window — once Supabase is the confirmed source of truth the
# SQLite writes can be dropped.

_LOCAL_POSTS_COLS = (
    "id,company,user_id,content,title,status,why_post,"
    # 2026-04-29 split: process_notes (Stelle's audit trail, hidden by
    # default) + fact_check_report (Castorice fact-check transcript)
    # used to be concatenated into why_post; now own columns. Old rows
    # may have NULL for both — UI tolerates that.
    "process_notes,fact_check_report,citation_comments,"
    "ordinal_post_id,linked_image_id,created_at,pre_revision_content,"
    # 2026-05-01: ``stelle_content`` is the immutable Stelle-final
    # draft for ingestion. ``content`` can drift via operator Edit
    # Save and the Rewrite flow; stelle_content does not. Bundle's
    # InFlight rendering + Aglaea edit_deltas read stelle_content
    # (with fallback to pre_revision_content / content for legacy
    # rows where stelle_content is NULL).
    "stelle_content,"
    "cyrene_score,generation_metadata,scheduled_date,publication_order,"
    # Draft ↔ published pairing (set by draft_match_worker semantically
    # or by /api/posts/{id}/set-publish-date manually). Read by
    # post_bundle to render DELTA (draft → published).
    "matched_provider_urn,matched_at,match_similarity,match_method"
)


def _normalize_local_post_row(row: dict[str, Any]) -> dict[str, Any]:
    """Supabase returns ISO strings for timestamptz; SQLite returns floats.
    Existing callers expect the SQLite float shape, so coerce on the way out.
    """
    if not row:
        return row
    ca = row.get("created_at")
    if isinstance(ca, str):
        try:
            from datetime import datetime as _dt
            # Supabase ISO 8601 (sometimes with microseconds + tz). fromisoformat
            # handles both 3.11+.
            dt = _dt.fromisoformat(ca.replace("Z", "+00:00"))
            row = dict(row)
            row["created_at"] = dt.timestamp()
        except Exception:
            pass
    return row


def insert_local_post(fields: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Insert one row into Supabase local_posts. Returns the saved row
    or None on failure. Caller is responsible for having already
    populated ``id`` (uuid4 string) and ``company``.
    """
    sb = _get_client()
    if sb is None:
        return None
    try:
        row = (
            sb.table("local_posts").insert(fields).execute().data
        )
        return _normalize_local_post_row(row[0]) if row else None
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] insert_local_post(%s) failed: %s",
            fields.get("id"), exc,
        )
        return None


def get_local_post(post_id: str) -> Optional[dict[str, Any]]:
    """Fetch one post by id. None if not found / Supabase unavailable."""
    sb = _get_client()
    if sb is None:
        return None
    try:
        rows = (
            sb.table("local_posts")
            .select(_LOCAL_POSTS_COLS)
            .eq("id", post_id)
            .limit(1)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.warning("[amphoreus_supabase] get_local_post(%s) failed: %s", post_id, exc)
        return None
    return _normalize_local_post_row(rows[0]) if rows else None


def list_local_posts(
    company: Optional[str] = None, limit: int = 50,
) -> list[dict[str, Any]]:
    """Return posts for a company (or all posts if company is None),
    newest-first. Returns an empty list when Supabase is unavailable —
    callers that also consult SQLite will fill in.
    """
    sb = _get_client()
    if sb is None:
        return []
    try:
        q = sb.table("local_posts").select(_LOCAL_POSTS_COLS).order("created_at", desc=True)
        if company:
            q = q.eq("company", company)
        rows = q.limit(max(1, min(limit, 500))).execute().data or []
    except Exception as exc:
        logger.warning("[amphoreus_supabase] list_local_posts failed: %s", exc)
        return []
    return [_normalize_local_post_row(r) for r in rows]


def update_local_post_fields(
    post_id: str, fields: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Update arbitrary fields on a single post."""
    if not fields:
        return get_local_post(post_id)
    sb = _get_client()
    if sb is None:
        return None
    try:
        rows = (
            sb.table("local_posts")
            .update(fields)
            .eq("id", post_id)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] update_local_post_fields(%s) failed: %s",
            post_id, exc,
        )
        return None
    return _normalize_local_post_row(rows[0]) if rows else None


def set_local_post_ordinal_id(
    post_id: str, ordinal_post_id: str,
) -> Optional[dict[str, Any]]:
    """Hyacinthia's post-push update. Setting ordinal_post_id flips the
    row from 'unpushed' → 'pushed', which protects it from the next
    Stelle-run wipe.
    """
    return update_local_post_fields(
        post_id, {"ordinal_post_id": ordinal_post_id, "status": "pushed"}
    )


def delete_local_post(post_id: str) -> bool:
    """Delete a ``local_posts`` row.

    Returns ``True`` on success, ``False`` when Amphoreus Supabase isn't
    configured (``_get_client()`` is ``None`` — caller degrades to
    SQLite-only). **Any Supabase-side failure (RLS policy, FK
    constraint, permission denial, network) raises the underlying
    supabase-py exception up to the caller.**

    Proactively clears rows on tables that reference ``local_posts.id``
    without ``ON DELETE CASCADE`` in the live schema before issuing the
    parent delete. As of 2026-04-23 that's ``draft_convergence_log``
    only (the FK there was added directly in Supabase without a
    cascade — the idempotent schema addendum bundles the cascade
    migration but re-running it on the live instance is harmless; the
    code path here is what makes delete work RIGHT NOW without waiting
    on the SQL paste).

    Other referencing tables (``draft_feedback``, ``local_post_revisions``,
    ``draft_publish_matches``) all carry ``ON DELETE CASCADE`` already,
    so Postgres handles them automatically.
    """
    sb = _get_client()
    if sb is None:
        return False
    # Proactive cascade: wipe rows from non-cascading referrers first.
    try:
        sb.table("draft_convergence_log").delete().eq("local_post_id", post_id).execute()
    except Exception as exc:
        logger.debug(
            "[amphoreus_supabase] pre-delete draft_convergence_log(%s) "
            "skipped (non-fatal): %s", post_id, exc,
        )
    sb.table("local_posts").delete().eq("id", post_id).execute()
    return True


# ---------------------------------------------------------------------------
# local_posts — semantic match-back against Jacquard's linkedin_posts
# ---------------------------------------------------------------------------
#
# After Ordinal is retired, the only way to pair a Stelle draft with its
# published LinkedIn post is semantic cosine match. The columns below are
# added by ``scripts/amphoreus_supabase_mirror_schema.sql`` as ``ALTER TABLE
# local_posts ADD COLUMN ...``:
#
#   embedding            vector(1536)   -- text-embedding-3-small of content
#   matched_provider_urn text           -- linkedin_posts.provider_urn pair
#   matched_at           timestamptz
#   match_similarity     real
#   match_method         text           -- 'semantic' | future: 'ordinal_id'
#
# The match-back worker lives in backend/src/services/draft_match_worker.py.


def embed_text_for_local_post(text: str) -> Optional[list[float]]:
    """Embed a draft body with OpenAI text-embedding-3-small (1536 dims).

    Returns ``None`` on any failure so callers can write the row without
    the embedding. The match-back worker filters to rows with a non-null
    embedding, so un-embedded rows are simply skipped on the next cycle.
    """
    import json as _json
    import urllib.request as _ur
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not text or not text.strip():
        return None
    # Cap at ~32k chars; text-embedding-3-small max is 8191 tokens.
    trimmed = text.strip()[:32_000]
    body = _json.dumps({"model": "text-embedding-3-small", "input": trimmed}).encode("utf-8")
    req = _ur.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _ur.urlopen(req, timeout=30) as r:
            return _json.loads(r.read().decode("utf-8"))["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("[amphoreus_supabase] embed_text_for_local_post failed: %s", exc)
        return None


def set_local_post_embedding(post_id: str, embedding: list[float]) -> bool:
    """Stamp ``embedding`` onto an existing local_posts row.

    Called by the write path in ``backend/src/db/local.py`` after a
    successful dual-write so newly-inserted / content-updated drafts are
    immediately searchable. Also used by the backfill script for
    historical rows.
    """
    if not post_id or not embedding:
        return False
    sb = _get_client()
    if sb is None:
        return False
    try:
        sb.table("local_posts").update({"embedding": embedding}).eq("id", post_id).execute()
        return True
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] set_local_post_embedding(%s) failed: %s",
            post_id, exc,
        )
        return False


def load_local_posts_for_match_back(
    company: str,
    *,
    generated_after_iso: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Candidate drafts for the match-back worker.

    Returns rows for ``company`` with a non-null embedding that haven't
    yet been paired with a LinkedIn post. ``generated_after_iso`` is the
    lower bound on ``created_at`` — pass the lookback window from the
    worker's caller so we don't cosine-compare every draft ever written.
    """
    sb = _get_client()
    if sb is None:
        return []
    try:
        q = (
            sb.table("local_posts")
              .select("id,company,user_id,content,title,created_at,embedding")
              .eq("company", company)
              .is_("matched_provider_urn", "null")
              .not_.is_("embedding", "null")
        )
        if generated_after_iso:
            q = q.gte("created_at", generated_after_iso)
        rows = q.order("created_at", desc=True).limit(500).execute().data or []
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] load_local_posts_for_match_back(%s) failed: %s",
            company, exc,
        )
        return []

    # PostgREST returns pgvector columns as JSON-string representations
    # ("[0.123,-0.456,...]") rather than parsed list[float]. The
    # match-back worker checks isinstance(emb, list) before computing
    # cosine, so unparsed strings get silently skipped. This was a
    # 100%-no-op semantic-match-back bug surfaced 2026-04-28 (every
    # draft in the system had string-typed embeddings → zero pairings).
    # Normalize to list[float] here so the caller sees a uniform type.
    import json as _json_local
    for r in rows:
        emb = r.get("embedding")
        if isinstance(emb, str):
            try:
                # pgvector serializes as "[v1,v2,...]" — valid JSON.
                r["embedding"] = _json_local.loads(emb)
            except Exception as exc:
                logger.debug(
                    "[load_local_posts_for_match_back] embedding parse failed for %s: %s",
                    (r.get("id") or "?")[:8], exc,
                )
                r["embedding"] = None
    return rows


def record_local_post_match(
    *,
    post_id: str,
    matched_provider_urn: str,
    similarity: float,
    method: str,
) -> bool:
    """Stamp a match-back outcome onto a local_posts row.

    Idempotent-ish: if the row already has a ``matched_provider_urn``
    set, the update still fires (the caller is expected to filter
    unmatched rows up front so this normally writes to a blank slate).
    """
    pid = (post_id or "").strip()
    urn = (matched_provider_urn or "").strip()
    if not pid or not urn:
        return False
    sb = _get_client()
    if sb is None:
        return False
    try:
        sb.table("local_posts").update({
            "matched_provider_urn": urn,
            "matched_at":           datetime.now(timezone.utc).isoformat(),
            "match_similarity":     round(float(similarity), 4),
            "match_method":         method,
        }).eq("id", pid).execute()
        return True
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] record_local_post_match(%s→%s) failed: %s",
            pid[:16], urn[:24], exc,
        )
        return False


# ---------------------------------------------------------------------------
# linkedin_post_engagement_snapshots — time-series engagement history
# ---------------------------------------------------------------------------
#
# Written on every scrape upsert (both legs: amphoreus_linkedin_scrape +
# jacquard_mirror_sync). Read by post_bundle to render kinetic trajectory
# for recent posts. Schema in
# ``scripts/amphoreus_supabase_engagement_snapshots_schema.sql``.

def record_engagement_snapshot(
    *,
    provider_urn:    str,
    scraped_at:      str,
    total_reactions: Optional[int]  = None,
    total_comments:  Optional[int]  = None,
    total_reposts:   Optional[int]  = None,
    scraped_by:      Optional[str]  = None,
) -> bool:
    """Append one engagement snapshot. Returns True on write, False on
    any failure — callers treat this as best-effort telemetry, not a
    correctness-critical write. Duplicate PK (same URN + same exact
    ``scraped_at``) is swallowed silently.
    """
    urn = (provider_urn or "").strip()
    if not urn or not scraped_at:
        return False
    sb = _get_client()
    if sb is None:
        return False
    row = {
        "provider_urn":    urn,
        "scraped_at":      scraped_at,
        "total_reactions": int(total_reactions) if total_reactions is not None else None,
        "total_comments":  int(total_comments)  if total_comments  is not None else None,
        "total_reposts":   int(total_reposts)   if total_reposts   is not None else None,
        "scraped_by":      scraped_by,
    }
    try:
        sb.table("linkedin_post_engagement_snapshots").insert(row).execute()
        return True
    except Exception as exc:
        # Duplicate (provider_urn, scraped_at) → PostgREST 409. Quiet.
        # Anything else is noisy but non-fatal.
        msg = str(exc)
        if "23505" not in msg and "duplicate key" not in msg.lower():
            logger.debug(
                "[amphoreus_supabase] record_engagement_snapshot(%s) failed: %s",
                urn[:32], msg[:200],
            )
        return False


def fetch_engagement_trajectories(
    provider_urns:  list[str],
    *,
    since_iso:      Optional[str] = None,
    per_urn_limit:  int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{provider_urn: [snapshot_rows...]}`` for a batch of URNs,
    chronologically ASCENDING (oldest first).

    URNs with no snapshots within ``since_iso`` are absent from the
    result — callers should treat "key missing" as "no history yet,"
    not "empty list." Default lookback is 30 days which is already
    past the render cutoff (14d).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    urns = [u for u in (provider_urns or []) if u]
    if not urns:
        return out
    sb = _get_client()
    if sb is None:
        return out
    if since_iso is None:
        since_iso = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()

    _CHUNK = 100
    for i in range(0, len(urns), _CHUNK):
        try:
            rows = (
                sb.table("linkedin_post_engagement_snapshots")
                  .select("provider_urn, scraped_at, total_reactions, "
                          "total_comments, total_reposts, scraped_by")
                  .in_("provider_urn", urns[i:i + _CHUNK])
                  .gte("scraped_at", since_iso)
                  .order("scraped_at", desc=False)
                  .limit(per_urn_limit * _CHUNK)
                  .execute()
                  .data
                or []
            )
        except Exception as exc:
            logger.debug(
                "[amphoreus_supabase] fetch_engagement_trajectories chunk failed: %s",
                exc,
            )
            continue
        for r in rows:
            urn = r.get("provider_urn")
            if not urn:
                continue
            lst = out.setdefault(urn, [])
            if len(lst) < per_urn_limit:
                lst.append(r)
    return out


def wipe_unpushed_drafts(
    company: str, user_id: Optional[str] = None,
) -> int:
    """Delete every draft row for ``company`` that never reached Ordinal.
    Called at the start of each Stelle run so the Posts tab shows only
    the freshly-generated batch plus the historical pushed posts.

    **User scoping (critical — 2026-04-22 incident fix):**

    When ``user_id`` is provided, the delete is scoped to rows stamped
    for that specific FOC PLUS NULL-user orphans at this company.
    This is the only safe mode for shared-company clients (Virio,
    Trimble, Commenda) — without user scoping, one Stelle run at one
    FOC deletes every other FOC's unpushed drafts, which is exactly
    how Eric's 4 stamped drafts vanished during today's incident.

    When ``user_id`` is None, the delete is company-wide. Only valid
    for single-FOC companies where there's no sibling data to harm.
    Stelle's per-run purge now ALWAYS passes ``user_id`` in multi-FOC
    mode (``DATABASE_USER_UUID`` env) — see ``agents/stelle.py``.

    Filter (matches the SQLite purge_unpushed_drafts semantics):
        status == 'draft'   AND   (ordinal_post_id IS NULL OR '')

    Rows with non-draft statuses (scheduled, posted, failed) are
    preserved regardless of ordinal_post_id because they represent
    operator-state the UI still needs to show.

    Returns the count of rows deleted; 0 when Supabase is unavailable
    (safer than raising — the run continues with stale rows).
    """
    if not company:
        return 0
    sb = _get_client()
    if sb is None:
        return 0
    try:
        q = (
            sb.table("local_posts")
            .delete()
            .eq("company", company)
            .eq("status", "draft")
            .or_("ordinal_post_id.is.null,ordinal_post_id.eq.")
        )
        if user_id:
            # User's own drafts PLUS any NULL-user orphans that were
            # probably written on their behalf before user_id stamping
            # worked reliably. NULL rows belong to nobody, so including
            # them in the user's purge is the right default — nobody
            # else will miss them.
            q = q.or_(f"user_id.eq.{user_id},user_id.is.null")
        deleted = q.execute().data or []
    except Exception as exc:
        logger.warning(
            "[amphoreus_supabase] wipe_unpushed_drafts(%s, user_id=%s) failed: %s",
            company, user_id, exc,
        )
        return 0
    return len(deleted)


def list_cyrene_briefs(company: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the last ``limit`` briefs for a company, newest-first.
    Used by Cyrene's ``query_brief_history`` tool to see the
    trajectory of her own recommendations across cycles. Empty list
    on error (never raises).
    """
    sb = _get_client()
    if sb is None:
        return []

    # Same slug/UUID tolerance as get_latest_cyrene_brief — Cyrene
    # canonicalizes company to UUID before writing, but callers may
    # hand us the slug form.
    candidates: list[str] = []
    if company:
        candidates.append(company)
    try:
        from backend.src.lib.company_resolver import resolve_with_fallback
        resolved = resolve_with_fallback(company)
        if resolved and resolved != company:
            candidates.append(resolved)
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for key in candidates:
        try:
            rows = (
                sb.table("cyrene_briefs")
                .select("id, created_at, brief")
                .eq("company", key)
                .order("created_at", desc=True)
                .limit(max(1, min(limit, 200)))
                .execute()
                .data
            ) or []
        except Exception as exc:
            logger.warning(
                "[amphoreus_supabase] list_cyrene_briefs(%s) failed: %s",
                key, exc,
            )
            continue
        if rows:
            break
    # Flatten: each entry = {id, created_at, ...brief-fields}
    out: list[dict[str, Any]] = []
    for r in rows:
        brief = r.get("brief") or {}
        if not isinstance(brief, dict):
            continue
        merged = dict(brief)
        merged["_row_id"] = r.get("id")
        merged["_created_at"] = r.get("created_at")
        out.append(merged)
    return out
