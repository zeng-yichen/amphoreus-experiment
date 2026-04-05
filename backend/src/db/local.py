"""SQLite local database for operational state.

Stores runs, events, fact-check results, eval results, and cache entries.
Supabase handles shared/cloud state — we never modify its schema.
"""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
    content TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    why_post TEXT,
    citation_comments TEXT,
    ordinal_post_id TEXT,
    linked_image_id TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_local_posts_company ON local_posts(company);

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
    fetched_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_post_engagers_dedup
    ON post_engagers(ordinal_post_id, engager_urn, engagement_type);
CREATE INDEX IF NOT EXISTS idx_post_engagers_company ON post_engagers(company);
CREATE INDEX IF NOT EXISTS idx_post_engagers_post ON post_engagers(ordinal_post_id);
"""


def _migrate_post_engagers(conn: sqlite3.Connection) -> None:
    """Create post_engagers table for DBs that predate the ICP loop feature."""
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
            fetched_at REAL NOT NULL DEFAULT (unixepoch('now'))
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


def _migrate_local_posts_columns(conn: sqlite3.Connection) -> None:
    """Add why_post / citation_comments for existing DBs created before these columns."""
    rows = conn.execute("PRAGMA table_info(local_posts)").fetchall()
    cols = {r[1] for r in rows}
    if "why_post" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN why_post TEXT")
    if "citation_comments" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN citation_comments TEXT")
    if "ordinal_post_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN ordinal_post_id TEXT")
    if "linked_image_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN linked_image_id TEXT")
    if "pre_revision_content" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN pre_revision_content TEXT")
    if "cyrene_score" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN cyrene_score REAL")
    if "generation_metadata" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN generation_metadata TEXT")


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

def create_local_post(
    post_id: str,
    company: str,
    content: str,
    title: str | None = None,
    status: str = "draft",
    why_post: str | None = None,
    citation_comments: list[str] | None = None,
    pre_revision_content: str | None = None,
    cyrene_score: float | None = None,
    generation_metadata: dict | None = None,
) -> dict:
    cc_json = json.dumps(citation_comments) if citation_comments else None
    gen_meta_json = json.dumps(generation_metadata) if generation_metadata else None
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO local_posts (id, company, content, title, status, why_post, citation_comments, ordinal_post_id, linked_image_id, pre_revision_content, cyrene_score, generation_metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)",
            (post_id, company, content, title, status, why_post, cc_json, pre_revision_content, cyrene_score, gen_meta_json),
        )
    return get_local_post(post_id) or {
        "id": post_id,
        "company": company,
        "content": content,
        "title": title,
        "status": status,
        "why_post": why_post,
        "citation_comments": cc_json,
        "ordinal_post_id": None,
        "linked_image_id": None,
        "created_at": time.time(),
    }


def list_local_posts(company: str | None = None, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        if company:
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
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM local_posts WHERE id=?", (post_id,)).fetchone()
        return dict(row) if row else None


def update_local_post(
    post_id: str,
    content: str | None = None,
    status: str | None = None,
    title: str | None = None,
) -> dict | None:
    fields, values = [], []
    if content is not None:
        fields.append("content=?"); values.append(content)
    if status is not None:
        fields.append("status=?"); values.append(status)
    if title is not None:
        fields.append("title=?"); values.append(title)
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    return get_local_post(post_id)


def update_local_post_fields(post_id: str, updates: dict[str, Any]) -> dict | None:
    """Partial update: only keys present in ``updates`` are written (e.g. exclude_unset from PATCH)."""
    col_map = {
        "content": "content",
        "status": "status",
        "title": "title",
        "linked_image_id": "linked_image_id",
    }
    fields, values = [], []
    for key, col in col_map.items():
        if key not in updates:
            continue
        val = updates[key]
        if key == "linked_image_id":
            val = val or None
        fields.append(f"{col}=?")
        values.append(val)
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    return get_local_post(post_id)


def set_local_post_ordinal_post_id(local_post_id: str, ordinal_post_id: str | None) -> dict | None:
    """Store latest Ordinal workspace post id after a successful push (re-push overwrites)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE local_posts SET ordinal_post_id=? WHERE id=?",
            (ordinal_post_id, local_post_id),
        )
    return get_local_post(local_post_id)


def delete_local_post(post_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM local_posts WHERE id=?", (post_id,))


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
                    "(company, ordinal_post_id, linkedin_post_url, engager_urn, name, headline, engagement_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        company,
                        ordinal_post_id,
                        linkedin_post_url,
                        urn,
                        e.get("name") or e.get("firstName", "") + " " + e.get("lastName", ""),
                        e.get("headline") or e.get("occupation") or "",
                        e.get("engagement_type", "reaction"),
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
