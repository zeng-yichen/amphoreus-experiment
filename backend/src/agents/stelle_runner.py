"""Standalone CLI runner for Stelle.

This module exists so Stelle can be invoked as a detached subprocess from
the FastAPI layer via ``subprocess.Popen([sys.executable, "-m",
"backend.src.agents.stelle_runner", ...], start_new_session=True)``. The
``start_new_session=True`` flag creates a new process group that is not
killed when uvicorn's ``--reload`` hot-reloads the parent process,
which is how the April 11 run was killed mid-post-session.

Contract:

  - Parent (FastAPI) creates a ``runs`` row + ``job_id`` via
    ``job_manager.create_job`` BEFORE spawning this runner
  - Parent spawns this runner with the job_id and company on the command
    line, captures the subprocess PID, returns the job_id to the client
  - Runner updates the run status to 'running' on start
  - Runner executes ``generate_one_shot`` to completion, emitting events
    through a SQLite-backed event callback (writes to ``run_events``)
  - Runner sets final status to 'completed' or 'failed' via
    ``complete_run``
  - Parent's SSE endpoint polls ``run_events`` to stream updates to the
    frontend — no in-memory IPC needed

If the parent process dies mid-run (uvicorn reload, crash, kill), this
runner keeps going because it is in its own session/process group. The
parent on restart can reconnect to the same ``run_id`` via the SSE
endpoint and resume streaming from wherever it left off in ``run_events``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logger = logging.getLogger("stelle_runner")


def _configure_logging(job_id: str, log_dir: Path) -> None:
    """Configure logging so the detached process writes to a predictable log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stelle_runner_{job_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("[stelle_runner] logging to %s (pid=%d)", log_file, os.getpid())


def _make_sqlite_event_callback(job_id: str):
    """Return an event callback that writes events to the run_events SQLite table.

    This is the cross-process replacement for the in-memory job_manager
    queue path: we cannot reach the parent FastAPI's in-memory dict from
    a subprocess, but we CAN write to shared SQLite. The parent's SSE
    endpoint polls run_events for new rows and streams them to the client.
    """
    from backend.src.db import local as db
    from backend.src.core.events import (
        thinking_event, tool_call_event, tool_result_event,
        text_delta_event, error_event, compaction_event, status_event,
    )

    _builders = {
        "thinking":    lambda d: thinking_event(d.get("text", "")),
        "tool_call":   lambda d: tool_call_event(d.get("name", ""), {"summary": d.get("arguments", "")}),
        "tool_result": lambda d: tool_result_event(d.get("name", ""), d.get("result", ""), d.get("is_error", False)),
        "text_delta":  lambda d: text_delta_event(d.get("text", "")),
        "error":       lambda d: error_event(d.get("message", "")),
        "compaction":  lambda _d: compaction_event(),
        "status":      lambda d: status_event(d.get("message", "")),
    }

    def callback(event_type: str, data: dict) -> None:
        builder = _builders.get(event_type)
        if builder is None:
            return
        try:
            ev = builder(data)
            # Persist to SQLite — the parent's SSE endpoint will pick this
            # up on its next poll of run_events
            db.record_event(job_id, ev.type, ev.data)
        except Exception:
            logger.exception("[stelle_runner] event callback failed for type=%s", event_type)

    return callback


def _mark_run_running(job_id: str) -> None:
    """Flip the run record from 'pending' to 'running' with start_at timestamp."""
    from backend.src.db.local import get_connection
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET status='running', started_at=? WHERE id=?",
            (time.time(), job_id),
        )


def _mark_run_completed(job_id: str, output_path: str) -> None:
    from backend.src.db.local import complete_run
    complete_run(job_id, output=output_path, error=None)


def _mark_run_failed(job_id: str, error: str) -> None:
    from backend.src.db.local import complete_run
    complete_run(job_id, output=None, error=error[:2000])


def _snapshot_workspace(company: str, job_id: str) -> None:
    """Create a workspace snapshot tied to this run id, matching the old threaded path.

    Skipped in Lineage mode: the only non-wiped content in the workspace
    is ``scratch/`` (the real data lives on Jacquard and is never
    committed to the workspace dir), so snapshots there capture nothing
    worth snapshotting. Left on, they accumulate one empty tree per run
    under ``snapshots/<job_id>/`` and Stelle wastes cycles exploring
    them on her next run.
    """
    try:
        from backend.src.agents import lineage_fs_client as _lfs
        if _lfs.is_lineage_mode():
            logger.info("[stelle_runner] Lineage mode — skipping workspace snapshot")
            return
    except Exception:
        # Fall through — if the import fails we still want the legacy
        # snapshot behavior for local-mode runs.
        pass
    try:
        from backend.src.services.workspace_manager import create_snapshot
        create_snapshot(company, job_id)
    except Exception:
        logger.exception("[stelle_runner] snapshot failed (non-fatal)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stelle as a detached subprocess.")
    parser.add_argument("--company", required=True, help="Client slug")
    parser.add_argument("--job-id", required=True, help="Pre-created run/job UUID")
    parser.add_argument("--prompt", default=None, help="Optional user prompt override")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--user-email", default=None, help="Authenticated user email for usage attribution")
    parser.add_argument(
        "--log-dir",
        default=".runner-logs",
        help="Directory for the runner's own log file (separate from Stelle's session log)",
    )
    # ------------------------------------------------------------------
    # Data-source plumbing. The env-var names carry a LINEAGE_* prefix
    # for historical reasons (Jacquard's user_companies table, Jacquard
    # FOC-user slugs); they identify the company/user in the shared
    # Supabase schema. See backend/src/_lineage_deprecated/ for the
    # quarantined HTTP-proxy integration.
    # ------------------------------------------------------------------
    parser.add_argument("--company-id", default=None,
                        help="Jacquard user_companies.id UUID for the client being written for.")
    parser.add_argument("--user-slug", "--lineage-user-slug", default=None,
                        dest="user_slug",
                        help="FOC-user slug. When set, the run is user-targeted: "
                             "every draft is attributed to this user and filesystem "
                             "reads are scoped to their subtree. Absent = company-wide.")
    args = parser.parse_args()

    # Ensure our cwd is the project root. When spawned via subprocess from
    # the FastAPI process, cwd is inherited, but be defensive in case this
    # gets called from somewhere unexpected.
    project_root = Path(__file__).resolve().parents[3]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    # Surface data-source inputs as env vars BEFORE we import Stelle so
    # the module-level ``is_lineage_mode()`` check in lineage_fs_client
    # sees them. ContextVars don't cross process boundaries; env vars do.
    if args.company_id:
        os.environ["LINEAGE_COMPANY_ID"] = args.company_id
    if args.user_slug:
        os.environ["LINEAGE_USER_SLUG"] = args.user_slug

    # Always surface the Amphoreus company keyword so the local
    # submit_draft handler can stamp drafts with the right ``company``
    # column in local_posts (independent of lineage mode).
    os.environ["STELLE_COMPANY_KEYWORD"] = args.company

    _configure_logging(args.job_id, project_root / args.log_dir)

    # Set usage attribution ContextVars so every Anthropic call made by
    # this subprocess is recorded with the correct user and client.
    # ContextVars don't cross process boundaries, so we accept them as
    # CLI args from the parent (which reads them from the HTTP middleware).
    from backend.src.usage.context import current_user_email, current_client_slug
    if args.user_email:
        current_user_email.set(args.user_email)
    current_client_slug.set(args.company)

    # Install usage instrumentation in the subprocess (the parent's
    # monkey-patch doesn't carry across process boundaries either).
    try:
        from backend.src.usage import install_instrumentation
        install_instrumentation()
    except Exception:
        logger.warning("[stelle_runner] usage instrumentation install failed (non-fatal)")

    logger.info(
        "[stelle_runner] starting company=%s job_id=%s model=%s prompt_len=%d user=%s",
        args.company, args.job_id, args.model,
        len(args.prompt or ""), args.user_email or "unattributed",
    )

    # Mark the run as running so the parent's SSE endpoint knows it's live
    try:
        _mark_run_running(args.job_id)
    except Exception:
        logger.exception("[stelle_runner] could not mark run as running (non-fatal)")

    # Install SIGTERM handler so the /stop endpoint can cleanly interrupt
    # mid-generation work. Python's default SIGTERM behavior kills the
    # process with no chance to flush the event stream — we convert it to
    # KeyboardInterrupt so the except block below emits a closing event
    # and marks the run failed with a proper reason before exit.
    import signal as _signal
    def _on_sigterm(_signum, _frame):
        logger.info("[stelle_runner] SIGTERM received — raising KeyboardInterrupt")
        raise KeyboardInterrupt("SIGTERM")
    try:
        _signal.signal(_signal.SIGTERM, _on_sigterm)
    except Exception:
        logger.warning("[stelle_runner] could not install SIGTERM handler (non-fatal)")

    # Emit an initial status event so the frontend sees immediate activity
    try:
        from backend.src.db import local as db
        from backend.src.core.events import status_event
        ev = status_event(f"Starting Stelle generation for {args.company}...")
        db.record_event(args.job_id, ev.type, ev.data)
    except Exception:
        logger.exception("[stelle_runner] initial status event emit failed")

    # Run Stelle to completion
    try:
        from backend.src.agents.stelle import generate_one_shot
        from backend.src.db import vortex

        vortex.ensure_dirs(args.company)
        output_dir = vortex.post_dir(args.company)
        output_filepath = str(output_dir / f"{args.company}_posts.md")

        event_callback = _make_sqlite_event_callback(args.job_id)

        result_path = generate_one_shot(
            args.company, args.company, output_filepath,
            prompt=args.prompt, model=args.model, event_callback=event_callback,
        )

        _snapshot_workspace(args.company, args.job_id)

        # Emit a final done event via the event stream.
        # Only send a short summary — the full output is redundant
        # with the posts tab and bloats the terminal stream.
        try:
            from backend.src.db import local as db
            from backend.src.core.events import done_event
            summary = f"Generation complete. Output: {result_path or 'unknown'}"
            ev = done_event(summary)
            db.record_event(args.job_id, ev.type, ev.data)
        except Exception:
            logger.exception("[stelle_runner] done event emit failed")

        _mark_run_completed(args.job_id, result_path or "")
        logger.info("[stelle_runner] completed successfully: %s", result_path)
        return 0

    except KeyboardInterrupt:
        # Clean stop from SIGTERM (user clicked Stop in Lineage UI).
        # Flush a final status+error event pair so the SSE stream closes
        # cleanly and the UI reflects "stopped by user" rather than an
        # abrupt disconnect. Scratch files on fly are preserved as-is.
        logger.info("[stelle_runner] stop requested — flushing closing events")
        try:
            from backend.src.db import local as db
            from backend.src.core.events import status_event, error_event
            db.record_event(
                args.job_id, "status",
                status_event("Stopped by user. Partial scratch preserved on fly.").data,
            )
            db.record_event(
                args.job_id, "error",
                error_event("stopped by user").data,
            )
        except Exception:
            logger.exception("[stelle_runner] failed to flush closing events")
        try:
            _mark_run_failed(args.job_id, "stopped by user")
        except Exception:
            pass
        return 130  # conventional exit code for SIGTERM

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[stelle_runner] crashed: %s\n%s", e, tb)
        try:
            from backend.src.db import local as db
            from backend.src.core.events import error_event
            ev = error_event(f"{type(e).__name__}: {str(e)[:500]}")
            db.record_event(args.job_id, ev.type, ev.data)
        except Exception:
            pass
        _mark_run_failed(args.job_id, f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
