"""SQLite local database for operational state.

Stores runs, events, fact-check results, eval results, and cache entries.
Supabase handles shared/cloud state — we never modify its schema.
"""

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from backend.src.core.config import get_settings

_DB_LOCK = threading.Lock()
_INITIALIZED = False


def _db_path() -> str:
    settings = get_settings()
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return settings.sqlite_path


@contextmanager
def get_connection():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _DB_LOCK:
        if _INITIALIZED:
            return
        with get_connection() as conn:
            conn.executescript(_SCHEMA)
            _migrate_local_posts_columns(conn)
            _migrate_post_engagers(conn)
        _INITIALIZED = True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    client_slug TEXT NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt TEXT,
    output TEXT,
    error TEXT,
    config_snapshot TEXT,
    started_at REAL,
    completed_at REAL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    event_type TEXT NOT NULL,
    data TEXT,
    timestamp REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id);

CREATE TABLE IF NOT EXISTS fact_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(id),
    post_index INTEGER,
    report TEXT,
    corrected_post TEXT,
    has_corrections INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    grade_results TEXT,
    trace TEXT,
    duration_seconds REAL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);

CREATE TABLE IF NOT EXISTS workspace_snapshots (
    id TEXT PRIMARY KEY,
    client_slug TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id),
    snapshot_path TEXT NOT NULL,
    content_hashes TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_client ON workspace_snapshots(client_slug);

CREATE TABLE IF NOT EXISTS local_posts (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    -- FOC-user UUID (Jacquard users.id). Mirrors how Jacquard's own
    -- drafts table disambiguates: one row per FOC user, not per
    -- company. NULL is legal for historical rows and company-wide
    -- writes that don't scope to a specific user.
    user_id TEXT,
    content TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    -- Operator-facing rationale. Authored by Castorice
    -- (analyze_strategic_fit), capped at ~50-60 words, glanceable in the
    -- Posts UI without expansion. Pre-2026-04-29 rows may contain a
    -- multi-section concatenation of Stelle's prose + Castorice's verdict
    -- + fact-check report; new rows hold ONLY Castorice's verdict.
    why_post TEXT,
    -- 2026-05-01: IMMUTABLE post-creation. The Stelle-final draft as
    -- it was ready to ship at the moment of submit_draft (post-
    -- Castorice fact-check, pre any operator edits or rewrites). This
    -- is the canonical record of "what the agent generated" for
    -- learning ingestion.
    --
    -- ``content`` can drift via operator Edit Save and the Rewrite
    -- flow; ``stelle_content`` does not. Future ingestion (bundle's
    -- InFlight section, Aglaea edit_deltas, RuanMei observation
    -- back-fills, cross-roster substrate, etc.) reads stelle_content,
    -- never content. Operator-facing display continues to use content
    -- so they see the latest edited state.
    --
    -- Legacy rows where this is NULL: fall back to
    -- pre_revision_content (which holds Stelle-pre-Castorice text on
    -- rows where Castorice corrected something), then to content as
    -- a last resort. Forward-only invariant.
    stelle_content TEXT,
    -- Stelle's audit trail: provenance, Irontomb anchor highlights,
    -- comfort score, length stats, decision rationale. Hidden behind a
    -- "Show process notes" expander in the UI; useful for debugging,
    -- not for review-time judgment.
    process_notes TEXT,
    -- Castorice fact-check transcript (what was changed, what was
    -- flagged). Rendered in its own UI expander when present.
    fact_check_report TEXT,
    citation_comments TEXT,
    ordinal_post_id TEXT,
    linked_image_id TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_local_posts_company ON local_posts(company);
-- idx_local_posts_user index is created by _migrate_local_posts_columns
-- after the ALTER TABLE that adds user_id to existing DBs; creating it
-- here would fail for DBs where user_id hasn't been added yet.

CREATE TABLE IF NOT EXISTS ruan_mei_state (
    company TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    last_updated REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS post_engagers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    ordinal_post_id TEXT NOT NULL,
    linkedin_post_url TEXT NOT NULL,
    engager_urn TEXT NOT NULL,
    name TEXT,
    headline TEXT,
    engagement_type TEXT NOT NULL DEFAULT 'reaction',
    fetched_at REAL NOT NULL DEFAULT (unixepoch('now')),
    icp_score REAL,
    current_company TEXT,
    title TEXT,
    location TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_post_engagers_dedup
    ON post_engagers(ordinal_post_id, engager_urn, engagement_type);
CREATE INDEX IF NOT EXISTS idx_post_engagers_company ON post_engagers(company);
CREATE INDEX IF NOT EXISTS idx_post_engagers_post ON post_engagers(ordinal_post_id);

-- Per-LLM-call usage/cost ledger. One row per provider API call, attributed
-- to the authenticated user (from the CF Access middleware ContextVar) and,
-- when available, the client slug the call was made on behalf of. Cost is
-- computed at record-time from a static price table in
-- backend/src/usage/pricing.py; editing that table does not backfill
-- existing rows.
--
-- provider:  "anthropic" | "openai" | "perplexity"
-- call_kind: "messages" | "embeddings" | "chat"
-- user_email / client_slug: nullable for unattributed calls (background
--   tasks like ordinal_sync that run outside any HTTP request).
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    call_kind TEXT NOT NULL,
    user_email TEXT,
    client_slug TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    duration_ms INTEGER,
    error TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_events(user_email, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_client ON usage_events(client_slug, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events(created_at);
"""


def _migrate_post_engagers(conn: sqlite3.Connection) -> None:
    """Create post_engagers table and add columns for DBs that predate them."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS post_engagers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ordinal_post_id TEXT NOT NULL,
            linkedin_post_url TEXT NOT NULL,
            engager_urn TEXT NOT NULL,
            name TEXT,
            headline TEXT,
            engagement_type TEXT NOT NULL DEFAULT 'reaction',
            fetched_at REAL NOT NULL DEFAULT (unixepoch('now')),
            icp_score REAL,
            current_company TEXT,
            title TEXT,
            location TEXT
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_post_engagers_dedup
            ON post_engagers(ordinal_post_id, engager_urn, engagement_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_post_engagers_company
            ON post_engagers(company)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_post_engagers_post
            ON post_engagers(ordinal_post_id)
    """)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(post_engagers)").fetchall()}
    for col, typedef in [("icp_score", "REAL"), ("current_company", "TEXT"),
                         ("title", "TEXT"), ("location", "TEXT")]:
        if col not in cols:
            conn.execute(f"ALTER TABLE post_engagers ADD COLUMN {col} {typedef}")


def _migrate_local_posts_columns(conn: sqlite3.Connection) -> None:
    """Add why_post / citation_comments for existing DBs created before these columns."""
    rows = conn.execute("PRAGMA table_info(local_posts)").fetchall()
    cols = {r[1] for r in rows}
    if "why_post" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN why_post TEXT")
    if "process_notes" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN process_notes TEXT")
    if "fact_check_report" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN fact_check_report TEXT")
    if "citation_comments" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN citation_comments TEXT")
    if "ordinal_post_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN ordinal_post_id TEXT")
    if "linked_image_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN linked_image_id TEXT")
    if "pre_revision_content" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN pre_revision_content TEXT")
    if "stelle_content" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN stelle_content TEXT")
    if "cyrene_score" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN cyrene_score REAL")
    if "generation_metadata" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN generation_metadata TEXT")
    if "scheduled_date" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN scheduled_date TEXT")
    if "publication_order" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN publication_order INTEGER")
    if "user_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN user_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_posts_user ON local_posts(user_id)"
        )


# --- Run helpers ---

def mark_stale_runs_failed() -> int:
    """Mark any jobs left in 'running' state as failed (interrupted by server restart)."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE runs SET status='failed', error='Interrupted by server restart', completed_at=? WHERE status='running'",
            (time.time(),),
        )
        return cursor.rowcount


def create_run(run_id: str, client_slug: str, agent: str, prompt: str | None = None, config: dict | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, client_slug, agent, status, prompt, config_snapshot, started_at) VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (run_id, client_slug, agent, prompt, json.dumps(config) if config else None, time.time()),
        )


def complete_run(run_id: str, output: str | None = None, error: str | None = None) -> None:
    status = "failed" if error else "completed"
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET status=?, output=?, error=?, completed_at=? WHERE id=?",
            (status, output, error, time.time(), run_id),
        )


def get_run(run_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(client_slug: str, limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE client_slug=? ORDER BY created_at DESC LIMIT ?",
            (client_slug, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Event helpers ---

def record_event(run_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO run_events (run_id, event_type, data) VALUES (?, ?, ?)",
            (run_id, event_type, json.dumps(data) if data else None),
        )


def get_run_events(run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM run_events WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run_events_after(run_id: str, after_id: int) -> list[dict]:
    """Return all run_events for this run_id whose id > after_id, ordered by id.

    Used by job_manager.drain_events to poll for new events emitted by a
    detached subprocess (stelle_runner). The `id` column is an
    auto-incrementing primary key, so comparing `id > after_id` gives a
    strictly monotonic cursor without timestamp ambiguity.

    Each row's `data` field is JSON-decoded before returning so the
    caller gets the same shape as an in-memory AgentEvent.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, run_id, event_type, data, timestamp FROM run_events "
            "WHERE run_id = ? AND id > ? ORDER BY id ASC",
            (run_id, after_id),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        raw = d.get("data")
        if isinstance(raw, str) and raw:
            try:
                d["data"] = json.loads(raw)
            except Exception:
                d["data"] = {"raw": raw}
        elif raw is None:
            d["data"] = {}
        out.append(d)
    return out


# --- Fact-check helpers ---

def record_fact_check(run_id: str, post_index: int, report: str, corrected_post: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO fact_checks (run_id, post_index, report, corrected_post, has_corrections) VALUES (?, ?, ?, ?, ?)",
            (run_id, post_index, report, corrected_post, 1 if corrected_post else 0),
        )


# --- Cache helpers ---

def cache_get(key: str) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM cache WHERE key=? AND expires_at > ?",
            (key, time.time()),
        ).fetchone()
        return row["value"] if row else None


def cache_set(key: str, value: str, ttl_seconds: int = 3600) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, time.time() + ttl_seconds),
        )


def cache_cleanup() -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        return cursor.rowcount


# --- Local post helpers ---

def _mirror_to_supabase(fn_name: str, *args, **kwargs) -> None:
    """Fire-and-log secondary write to Amphoreus Supabase.

    Every SQLite-backed ``local_posts`` mutation below also mirrors
    through here so Hyacinthia / the Posts-tab API can read from
    Supabase as the primary source. If Supabase is unreachable (503
    during DDL windows, etc.) we log + continue — SQLite remains the
    safety net until the full cutover.
    """
    try:
        from backend.src.db import amphoreus_supabase as _sb
        fn = getattr(_sb, fn_name, None)
        if fn is None:
            return
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("[local_posts mirror] %s failed: %s", fn_name, exc)


def _record_content_revision(
    post_id: str,
    content: str,
    *,
    source: str,
    author_email: str | None = None,
) -> None:
    """Append a row to ``local_post_revisions`` every time a draft's
    content changes.

    Called from :func:`create_local_post` (source=``stelle_initial``)
    and from every content-mutating update path (edit, revert,
    rewrite). Best-effort — failures log at debug and swallow, because
    the canonical write to ``local_posts`` has already succeeded and
    we'd rather lose a revision-history row than fail the operator's
    save.

    The revisions table lives in Amphoreus Supabase, not SQLite —
    local history isn't useful and would just double-write. If the
    mirror table doesn't exist yet (pre-migration) we silently no-op.
    """
    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
    except Exception:
        return
    if not is_configured():
        return
    try:
        sb = _get_client()
        if sb is None:
            return
        sb.table("local_post_revisions").insert({
            "draft_id": post_id,
            "content": content,
            "source": source,
            "author_email": author_email,
        }).execute()
    except Exception as exc:
        import logging as _l
        _l.getLogger(__name__).debug(
            "[local_posts] revision record skipped (post=%s source=%s): %s",
            post_id, source, exc,
        )


def _mirror_embed_local_post_content(post_id: str, content: str) -> None:
    """Best-effort embed + stamp on the Supabase mirror row.

    Lives alongside the other mirror writes because embedding lives only
    in Supabase (pgvector column); SQLite has no vector type. Runs
    inline so a freshly-inserted draft is immediately searchable by the
    match-back worker, at the cost of one OpenAI embedding call per
    write (~100-300ms). Never raises — silent on any failure.
    """
    if not post_id or not content or not content.strip():
        return
    try:
        from backend.src.db import amphoreus_supabase as _sb
        emb = _sb.embed_text_for_local_post(content)
        if emb is not None:
            _sb.set_local_post_embedding(post_id, emb)
    except Exception as exc:
        logger.debug(
            "[local_posts mirror] embed %s failed: %s", post_id[:12], exc,
        )


def create_local_post(
    post_id: str,
    company: str,
    content: str,
    title: str | None = None,
    status: str = "draft",
    why_post: str | None = None,
    process_notes: str | None = None,
    fact_check_report: str | None = None,
    citation_comments: list[str] | None = None,
    pre_revision_content: str | None = None,
    cyrene_score: float | None = None,
    generation_metadata: dict | None = None,
    publication_order: int | None = None,
    scheduled_date: str | None = None,
    user_id: str | None = None,
    stelle_content: str | None = None,
) -> dict:
    """Insert a draft row.

    ``user_id`` is the Jacquard FOC-user UUID when the draft belongs to a
    specific person (Heather vs Mark within Trimble, Logan vs Sam within
    Commenda). Leave NULL for company-wide writes or legacy callers.

    Dual-write: primary destination is Amphoreus Supabase (``local_posts``
    table), secondary is fly-local SQLite. Reads (``list_local_posts`` /
    ``get_local_post``) prefer Supabase; SQLite is the fallback / safety
    net during the migration window.
    """
    cc_json = json.dumps(citation_comments) if citation_comments else None
    gen_meta_json = json.dumps(generation_metadata) if generation_metadata else None
    # 2026-05-01: ``stelle_content`` is the immutable Stelle-final draft
    # for ingestion. If caller didn't pass it explicitly, default to
    # ``content`` at creation time (the post-Castorice corrected Stelle
    # text) — captures the ingestion-canonical version BEFORE any
    # subsequent edits or rewrites can drift it.
    if stelle_content is None:
        stelle_content = content
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO local_posts (id, company, user_id, content, title, status, why_post, process_notes, fact_check_report, citation_comments, ordinal_post_id, linked_image_id, pre_revision_content, stelle_content, cyrene_score, generation_metadata, publication_order, scheduled_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)",
            (post_id, company, user_id, content, title, status, why_post, process_notes, fact_check_report, cc_json, pre_revision_content, stelle_content, cyrene_score, gen_meta_json, publication_order, scheduled_date),
        )
    # Mirror to Amphoreus Supabase (Posts tab + Hyacinthia source of truth).
    _mirror_to_supabase(
        "insert_local_post",
        {
            "id": post_id,
            "company": company,
            "user_id": user_id,
            "content": content,
            "title": title,
            "status": status,
            "why_post": why_post,
            "process_notes": process_notes,
            "fact_check_report": fact_check_report,
            "citation_comments": cc_json,
            "ordinal_post_id": None,
            "linked_image_id": None,
            "pre_revision_content": pre_revision_content,
            "stelle_content": stelle_content,
            "cyrene_score": cyrene_score,
            "generation_metadata": gen_meta_json,
            "publication_order": publication_order,
            "scheduled_date": scheduled_date,
        },
    )
    # Embed + stamp the embedding column so the draft_match_worker can
    # cosine-pair this draft with whatever LinkedIn post it lands on.
    # Best-effort only — Supabase mirror is already written, and the
    # match-back worker skips rows with NULL embedding without erroring.
    _mirror_embed_local_post_content(post_id, content)
    # Seed the revision history with Stelle's initial write so later
    # reverts and diffs have a baseline. Subsequent content changes get
    # additional rows via update_local_post / update_local_post_fields.
    _record_content_revision(post_id, content, source="stelle_initial")
    return get_local_post(post_id) or {
        "id": post_id,
        "company": company,
        "user_id": user_id,
        "content": content,
        "title": title,
        "status": status,
        "why_post": why_post,
        "process_notes": process_notes,
        "fact_check_report": fact_check_report,
        "citation_comments": cc_json,
        "ordinal_post_id": None,
        "linked_image_id": None,
        "created_at": time.time(),
    }


def list_local_posts(
    company: str | None = None,
    limit: int = 50,
    user_id: str | None = None,
) -> list[dict]:
    """List draft rows.

    Filter semantics:
      - ``user_id`` + ``company``  → rows for this specific FOC user,
        PLUS rows in the same company with ``user_id=NULL`` (legacy
        orphans — see invariant note below).
      - ``user_id`` alone          → strictly per-FOC-user view.
      - ``company`` alone          → company-wide view, mixes all users
        including NULL-user-id orphans. This is the admin view.
      - neither                    → all rows (admin view).

    **Why NULL rows are inclusive here (history — 2026-04-22):**

    A per-FOC query returns BOTH the user's own rows AND NULL-user-id
    rows under the same company, for two reasons:

    1. At single-FOC companies (Overwolf, Hensley, Innovo, Hume AI,
       Flora-until-Weber-was-added), every draft written before
       per-FOC stamping landed with user_id=NULL. The sole FOC is
       unambiguously the owner; strict filtering would silently hide
       every legacy draft. 4+ clients in the database today are in
       this state.

    2. At explicitly-shared multi-FOC companies (Trimble heather/mark,
       Commenda logan/sam — see memory note
       ``project_trimble_shared_ordinal_account.md``), one Ordinal
       account + one FOC pool + shared drafting is the operator
       model. Inclusive NULL filtering matches that.

    The original leak case — Virio, where 4 orphan drafts written via
    bare ``virio`` slug leaked into every one of 19 FOC's posts tabs —
    is now blocked at the **write** path (``submit_draft`` refuses to
    write an orphan when ``DATABASE_COMPANY_ID`` indicates multi-FOC
    mode, see stelle.py). So going forward, NULL-user rows only arise
    at single-FOC companies or legacy rows — both of which inclusive
    filtering handles correctly.

    Hardening path (not shipped today, queued): backfill the 19
    remaining NULL rows to their rightful FOC, then tighten this
    filter back to strict. Until then, **inclusive is the safe
    default** because write-side orphaning is now prevented.

    Read-preferred: Amphoreus Supabase is the primary store. SQLite
    serves as fallback when Supabase returns an empty result.
    """
    # Primary: Amphoreus Supabase
    try:
        from backend.src.db import amphoreus_supabase as _sb
        if _sb.is_configured():
            rows = _sb.list_local_posts(company=company, limit=limit)
            if user_id:
                if company:
                    # Inclusive: this user's rows OR NULL-user rows in
                    # this company. See docstring — safe because write
                    # path now refuses to orphan at multi-FOC companies.
                    rows = [
                        r for r in rows
                        if r.get("user_id") == user_id
                        or not (r.get("user_id") or "").strip()
                    ]
                else:
                    # No company context → can't widen. Strict filter.
                    rows = [r for r in rows if r.get("user_id") == user_id]
            if rows:
                return rows
    except Exception as exc:
        logger.debug("[list_local_posts] Supabase read failed, falling back: %s", exc)

    # Fallback: SQLite
    with get_connection() as conn:
        if user_id and company:
            # Inclusive: user's rows OR NULL-user rows in this company.
            rows = conn.execute(
                "SELECT * FROM local_posts "
                "WHERE company=? AND (user_id=? OR user_id IS NULL OR user_id='') "
                "ORDER BY created_at DESC LIMIT ?",
                (company, user_id, limit),
            ).fetchall()
        elif user_id:
            rows = conn.execute(
                "SELECT * FROM local_posts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        elif company:
            rows = conn.execute(
                "SELECT * FROM local_posts WHERE company=? ORDER BY created_at DESC LIMIT ?",
                (company, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM local_posts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_local_post(post_id: str) -> dict | None:
    """Read-preferred: Amphoreus Supabase, SQLite fallback."""
    try:
        from backend.src.db import amphoreus_supabase as _sb
        if _sb.is_configured():
            row = _sb.get_local_post(post_id)
            if row:
                return row
    except Exception as exc:
        logger.debug("[get_local_post] Supabase read failed, falling back: %s", exc)
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM local_posts WHERE id=?", (post_id,)).fetchone()
        return dict(row) if row else None


def update_local_post(
    post_id: str,
    content: str | None = None,
    status: str | None = None,
    title: str | None = None,
    *,
    revision_source: str = "operator_edit",
    revision_author: str | None = None,
) -> dict | None:
    """Update mutable fields on a draft.

    When ``content`` changes, a ``local_post_revisions`` row is
    appended with ``revision_source`` (default ``operator_edit``).
    Callers that represent a different provenance (Castorice rewrite,
    revert-to-original, rewrite-with-feedback) should pass a specific
    ``revision_source`` so downstream eval can slice the history
    accurately.
    """
    fields, values = [], []
    mirror_fields: dict[str, Any] = {}
    if content is not None:
        fields.append("content=?"); values.append(content); mirror_fields["content"] = content
    if status is not None:
        fields.append("status=?"); values.append(status); mirror_fields["status"] = status
    if title is not None:
        fields.append("title=?"); values.append(title); mirror_fields["title"] = title
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    if mirror_fields:
        _mirror_to_supabase("update_local_post_fields", post_id, mirror_fields)
    if content is not None:
        _mirror_embed_local_post_content(post_id, content)
        _record_content_revision(
            post_id, content,
            source=revision_source,
            author_email=revision_author,
        )
    return get_local_post(post_id)


def update_post_schedule(post_id: str, scheduled_date: str | None) -> dict | None:
    """Update the scheduled publication date for a post (calendar drag-drop)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE local_posts SET scheduled_date=? WHERE id=?",
            (scheduled_date, post_id),
        )
    _mirror_to_supabase(
        "update_local_post_fields", post_id, {"scheduled_date": scheduled_date}
    )
    return get_local_post(post_id)


def list_calendar_posts(company: str, month: str | None = None) -> list[dict]:
    """Return all local_posts for a company, optionally filtered to a month.

    month format: '2026-04' (YYYY-MM). If None, returns all posts.
    Results ordered by scheduled_date (nulls last), then publication_order.
    """
    with get_connection() as conn:
        if month:
            rows = conn.execute(
                """SELECT * FROM local_posts
                   WHERE company = ?
                     AND (scheduled_date LIKE ? OR scheduled_date IS NULL)
                   ORDER BY
                     CASE WHEN scheduled_date IS NULL THEN 1 ELSE 0 END,
                     scheduled_date,
                     publication_order""",
                (company, f"{month}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM local_posts
                   WHERE company = ?
                   ORDER BY
                     CASE WHEN scheduled_date IS NULL THEN 1 ELSE 0 END,
                     scheduled_date,
                     publication_order""",
                (company,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_local_post_fields(
    post_id: str,
    updates: dict[str, Any],
    *,
    revision_source: str = "operator_edit",
    revision_author: str | None = None,
) -> dict | None:
    """Partial update: only keys present in ``updates`` are written (e.g. exclude_unset from PATCH).

    When ``content`` is in ``updates``, a ``local_post_revisions`` row
    is appended — see :func:`update_local_post` for the provenance
    convention.
    """
    col_map = {
        "content": "content",
        "status": "status",
        "title": "title",
        "linked_image_id": "linked_image_id",
        "scheduled_date": "scheduled_date",
        "publication_order": "publication_order",
    }
    fields, values = [], []
    mirror_fields: dict[str, Any] = {}
    for key, col in col_map.items():
        if key not in updates:
            continue
        val = updates[key]
        if key == "linked_image_id":
            val = val or None
        fields.append(f"{col}=?")
        values.append(val)
        mirror_fields[col] = val
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    if mirror_fields:
        _mirror_to_supabase("update_local_post_fields", post_id, mirror_fields)
    if "content" in updates and updates["content"] is not None:
        _mirror_embed_local_post_content(post_id, updates["content"])
        _record_content_revision(
            post_id, updates["content"],
            source=revision_source,
            author_email=revision_author,
        )
    return get_local_post(post_id)


def set_local_post_ordinal_post_id(local_post_id: str, ordinal_post_id: str | None) -> dict | None:
    """Store latest Ordinal workspace post id after a successful push (re-push overwrites).

    Setting ``ordinal_post_id`` to a non-empty value marks the draft
    as pushed — the next Stelle-run wipe will preserve it (only rows
    with NULL/empty ordinal_post_id are wiped).
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE local_posts SET ordinal_post_id=? WHERE id=?",
            (ordinal_post_id, local_post_id),
        )
    # Mirror. Also flip status → 'pushed' in Supabase when we have a
    # non-empty ordinal id so the Posts tab and the wipe filter agree.
    mirror_fields: dict[str, Any] = {"ordinal_post_id": ordinal_post_id}
    if ordinal_post_id:
        mirror_fields["status"] = "pushed"
    _mirror_to_supabase("update_local_post_fields", local_post_id, mirror_fields)
    return get_local_post(local_post_id)


def delete_local_post(post_id: str, *, deleted_by: str = "system", reason: Optional[str] = None) -> None:
    """Delete a draft.

    Dual-surface: SQLite (legacy local cache) + Amphoreus Supabase
    (authoritative, read by the Posts tab). The Supabase delete is
    called **directly** here, not via ``_mirror_to_supabase`` — that
    dispatcher swallows all exceptions at DEBUG level, which used to
    mask Supabase-side failures (RLS, FK constraint, permission) and
    let the HTTP layer report "deleted" while the row persisted.

    Any Supabase-side failure now raises so the HTTP route can surface
    a real error instead of lying to the UI. SQLite-side failures also
    raise (same policy).

    ``log_deletion`` is best-effort — audit-log failure is non-critical
    and explicitly swallowed.
    """
    # Snapshot before delete so the audit log captures the row content.
    snapshot: dict | None = None
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM local_posts WHERE id=?", (post_id,)).fetchone()
        if row is not None:
            snapshot = {k: row[k] for k in row.keys()}
        conn.execute("DELETE FROM local_posts WHERE id=?", (post_id,))
    # Authoritative surface — raise on any Supabase failure.
    from backend.src.db import amphoreus_supabase as _sb
    _sb.delete_local_post(post_id)
    # Audit log — best-effort.
    try:
        from backend.src.db.amphoreus_supabase import log_deletion
        log_deletion(
            entity_type="local_post",
            entity_id=post_id,
            entity_snapshot=snapshot,
            deleted_by=deleted_by,
            reason=reason,
        )
    except Exception:
        pass


def purge_unpushed_drafts(company: str, user_id: str | None = None) -> int:
    """Delete all draft local_posts rows for a company that never reached Ordinal.

    Called at the start of every Stelle run so each new batch starts with a
    clean slate. The rule is simple: if a draft has no ordinal_post_id, it
    was never committed to Ordinal, so per the user's dedup model it
    "doesn't exist" and should not persist across runs.

    **User scoping (2026-04-22 incident fix).** When ``user_id`` is
    supplied, the purge is scoped to that FOC's rows plus any NULL-
    user orphans (which belong to nobody and can safely clear out
    alongside the user's batch). A Stelle run for one Virio FOC no
    longer wipes every other Virio FOC's unpushed drafts. Stelle's
    start-of-run hook always passes ``user_id`` when it's in per-FOC
    mode (``DATABASE_USER_UUID`` env set by stelle_runner).

    Preserves:
      - Rows with a non-empty ordinal_post_id (pushed to Ordinal — even if
        the post later got unpublished, the Ordinal workspace still owns it)
      - Rows whose status is anything other than 'draft' (e.g. 'posted',
        'scheduled', 'failed' — those represent states the operator might
        still need to see)
      - At per-FOC scope: OTHER FOCs' draft rows at the same company.

    Returns the number of rows deleted.
    """
    company = (company or "").strip()
    if not company:
        return 0
    # Snapshot candidates BEFORE deleting so each wiped row lands in
    # deletion_log with its content. Without this, the purge erases
    # history we might want to replay later.
    with get_connection() as conn:
        base_pred = (
            "company = ? AND status = 'draft' "
            "AND (ordinal_post_id IS NULL OR ordinal_post_id = '')"
        )
        if user_id:
            # Per-FOC scope: user's rows OR NULL-user orphans.
            pred = base_pred + " AND (user_id = ? OR user_id IS NULL OR user_id = '')"
            params = (company, user_id)
        else:
            pred = base_pred
            params = (company,)
        snapshots = [
            {k: r[k] for k in r.keys()}
            for r in conn.execute(
                f"SELECT * FROM local_posts WHERE {pred}",
                params,
            ).fetchall()
        ]
        cur = conn.execute(
            f"DELETE FROM local_posts WHERE {pred}",
            params,
        )
        deleted = cur.rowcount or 0
    # Audit every wiped draft. Done after the SQLite delete commits so
    # the log never points to something still present.
    try:
        from backend.src.db.amphoreus_supabase import log_deletion
        for snap in snapshots:
            log_deletion(
                entity_type="local_post",
                entity_id=snap.get("id"),
                entity_snapshot=snap,
                deleted_by="system",
                reason=(
                    f"purge_unpushed_drafts(company={company}, user_id={user_id})"
                    if user_id else f"purge_unpushed_drafts({company})"
                ),
            )
    except Exception:
        pass
    # Also wipe in Amphoreus Supabase so the Posts tab stays consistent
    # with the "fresh batch" UX — posts that Stelle replaces shouldn't
    # linger after the run kicks off. Supabase-side wipe deliberately
    # does NOT re-log — the SQLite path above is the authoritative
    # audit record for a given purge.
    _mirror_to_supabase("wipe_unpushed_drafts", company, user_id=user_id)
    return deleted


# --- RuanMei state helpers ---

def ruan_mei_load(company: str) -> dict | None:
    """Return the stored state dict for a company, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT state_json FROM ruan_mei_state WHERE company=?", (company,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def ruan_mei_save(company: str, state: dict) -> None:
    """Atomically persist RuanMei state for a company (upsert)."""
    state["last_updated"] = state.get("last_updated", "")
    blob = json.dumps(state)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ruan_mei_state (company, state_json, last_updated) VALUES (?, ?, unixepoch('now'))"
            " ON CONFLICT(company) DO UPDATE SET state_json=excluded.state_json, last_updated=excluded.last_updated",
            (company, blob),
        )


# --- Post engager helpers ---

def upsert_engagers(company: str, ordinal_post_id: str, linkedin_post_url: str, engagers: list[dict]) -> int:
    """Insert engager records, ignoring duplicates. Returns count of newly inserted rows."""
    inserted = 0
    with get_connection() as conn:
        for e in engagers:
            urn = e.get("urn") or e.get("id") or e.get("profileId") or ""
            if not urn:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO post_engagers "
                    "(company, ordinal_post_id, linkedin_post_url, engager_urn, name, headline, "
                    "engagement_type, current_company, title, location) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        company,
                        ordinal_post_id,
                        linkedin_post_url,
                        urn,
                        e.get("name") or e.get("firstName", "") + " " + e.get("lastName", ""),
                        e.get("headline") or e.get("occupation") or "",
                        e.get("engagement_type", "reaction"),
                        e.get("current_company") or e.get("companyName") or "",
                        e.get("title") or "",
                        e.get("location") or "",
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception:
                pass
    return inserted


def get_engagers_for_post(ordinal_post_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM post_engagers WHERE ordinal_post_id=?",
            (ordinal_post_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def engagers_fetched_for_post(ordinal_post_id: str) -> bool:
    """Return True if we already have any engager rows for this post."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM post_engagers WHERE ordinal_post_id=? LIMIT 1",
            (ordinal_post_id,),
        ).fetchone()
        return row is not None


def update_engager_icp_scores(ordinal_post_id: str, scores: list[tuple[str, float]]) -> int:
    """Batch-update per-engager ICP scores for a post.

    Args:
        ordinal_post_id: The post these engagers belong to.
        scores: List of (engager_urn, icp_score) pairs. Order-independent.

    Returns:
        Number of rows updated.
    """
    updated = 0
    with get_connection() as conn:
        for urn, score in scores:
            conn.execute(
                "UPDATE post_engagers SET icp_score=? "
                "WHERE ordinal_post_id=? AND engager_urn=?",
                (round(score, 4), ordinal_post_id, urn),
            )
            updated += conn.execute("SELECT changes()").fetchone()[0]
    return updated


def get_top_icp_engagers(company: str, limit: int = 30) -> list[dict]:
    """Return top ICP-scored engagers for a company, aggregated across posts.

    Ranking metric: mean_icp_score * log(1 + engagement_count).
    Rewards both ICP fit and repeat engagement.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                engager_urn,
                MAX(name) AS name,
                MAX(headline) AS headline,
                MAX(current_company) AS current_company,
                MAX(title) AS title,
                MAX(location) AS location,
                AVG(icp_score) AS mean_icp_score,
                COUNT(*) AS engagement_count,
                GROUP_CONCAT(DISTINCT ordinal_post_id) AS post_ids
            FROM post_engagers
            WHERE company = ? AND icp_score IS NOT NULL AND icp_score > 0
            GROUP BY engager_urn
            ORDER BY AVG(icp_score) DESC
            """,
            (company,),
        ).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        mean_score = d["mean_icp_score"] or 0.0
        eng_count = d["engagement_count"] or 1
        d["ranking_score"] = round(mean_score * math.log(1 + eng_count), 4)
        d["posts_engaged"] = (d.pop("post_ids") or "").split(",")
        results.append(d)

    results.sort(key=lambda x: x["ranking_score"], reverse=True)
    return results[:limit]


def get_unscored_engager_post_ids(company: str) -> list[str]:
    """Return ordinal_post_ids that have engagers but no ICP scores yet."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ordinal_post_id
            FROM post_engagers
            WHERE company = ? AND icp_score IS NULL
            """,
            (company,),
        ).fetchall()
    return [r[0] for r in rows]
