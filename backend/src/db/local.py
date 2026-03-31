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
"""


# --- Run helpers ---

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
