"""Job manager — tracks long-running agent tasks with SSE streaming.

Mirrors Jacquard's in-memory job store pattern. Each job has an event queue
that the SSE endpoint drains.
"""

import contextlib
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable

from backend.src.core.events import AgentEvent
from backend.src.db import local as db

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def create_job(client_slug: str, agent: str, prompt: str | None = None, creator_id: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    event_queue: queue.Queue[AgentEvent] = queue.Queue(maxsize=2000)

    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "client_slug": client_slug,
            "agent": agent,
            "creator_id": creator_id,
            "prompt": prompt,
            "status": "pending",
            "output": None,
            "error": None,
            "event_queue": event_queue,
            "start_time": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    db.create_run(job_id, client_slug, agent, prompt)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _JOBS.get(job_id)


def set_status(job_id: str, status: str, output: str | None = None, error: str | None = None) -> None:
    with _LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id]["status"] = status
        _JOBS[job_id]["updated_at"] = time.time()
        if output is not None:
            _JOBS[job_id]["output"] = output
        if error is not None:
            _JOBS[job_id]["error"] = error

    if status in ("completed", "failed"):
        db.complete_run(job_id, output=output, error=error)


def emit_event(job_id: str, event: AgentEvent) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
    eq = job.get("event_queue")
    if eq:
        with contextlib.suppress(queue.Full):
            eq.put_nowait(event)

    db.record_event(job_id, event.type, event.data)


def drain_events(job_id: str, timeout: float = 30.0):
    """Generator that yields AgentEvents from the job's queue. Used by SSE endpoint."""
    job = get_job(job_id)
    if not job:
        return

    eq: queue.Queue[AgentEvent] = job["event_queue"]
    deadline = time.time() + timeout

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        try:
            event = eq.get(timeout=min(remaining, 1.0))
            yield event
            if event.type in ("done", "error"):
                break
        except queue.Empty:
            fresh = get_job(job_id)
            if fresh and fresh.get("status") in ("completed", "failed"):
                break
            continue


def run_in_background(
    job_id: str,
    target: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
) -> threading.Thread:
    """Execute an agent function in a background thread."""

    def _wrapper():
        with _LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "running"
                _JOBS[job_id]["start_time"] = time.time()

        try:
            result = target(*args, **(kwargs or {}))
            set_status(job_id, "completed", output=str(result) if result else None)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            set_status(job_id, "failed", error=str(e))
            from backend.src.core.events import error_event
            emit_event(job_id, error_event(str(e)))

    thread = threading.Thread(target=_wrapper, daemon=True, name=f"job-{job_id[:8]}")
    thread.start()
    return thread
