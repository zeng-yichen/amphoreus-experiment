"""Ghostwriter API — generate, stream, manage workspaces."""

import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ghostwriter", tags=["ghostwriter"])


class GenerateRequest(BaseModel):
    """Request body for /api/ghostwriter/generate.

    Accepts either ``company`` (historical slug like "hume-andrew") or
    ``companyId`` (Lineage UUID like "9abcb96e-..."). When both are present
    the slug wins. When only companyId is present we treat this as a
    Lineage-mode run and resolve the workspace via HTTP tool calls back
    into virio-api rather than local /data/memory/.
    """

    company: str | None = None
    companyId: str | None = None
    prompt: str | None = None
    model: str = "claude-opus-4-6"


class InlineEditRequest(BaseModel):
    """Inline-edit request body.

    Accepts two shapes for backward compat:
      - Amphoreus-native:  {company, post_text, instruction}
      - Jacquard proxy:    {companyId, draftId, selectedText, instruction,
                            selectionStart?, selectionEnd?}

    The handler normalizes both into (identifier, text, instruction).
    """

    company: str | None = None
    companyId: str | None = None
    draftId: str | None = None
    post_text: str | None = None
    selectedText: str | None = None
    instruction: str


class LinkedInUsernameRequest(BaseModel):
    username: str




@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """Start a ghostwriter generation job as a DETACHED subprocess.

    Accepts either ``company`` (slug) or ``companyId`` (Jacquard
    user_companies UUID). If only a slug is given and Supabase creds are
    configured, the slug is resolved against Jacquard's ``user_companies``
    table by name/domain match.

    Behind CF Access (service-token bypass). Email auth is disabled on
    this deployment — see ``cf_access.py`` for the bypass path.

    The detached subprocess pattern survives uvicorn hot-reloads via
    ``start_new_session=True``. Frontend reconnects to
    ``/stream/{job_id}`` for resumable SSE.
    """
    identifier = req.company or req.companyId
    if not identifier:
        raise HTTPException(
            status_code=400,
            detail="generate requires either 'company' (slug) or 'companyId' (UUID)",
        )

    # Only run the body-based ACL check when we have a real slug and the
    # request came through a user identity (admin bypass in
    # middleware.py:138-145 covers CF service tokens).
    from backend.src.auth.middleware import require_client_body
    if req.company:
        require_client_body(request, req.company)

    # Resolve the slug to a Jacquard user_companies.id so Stelle can
    # read the client's data. If a companyId is already supplied we use
    # it as-is. This is the only external coupling to Jacquard — the
    # data source (Supabase + GCS) lives there, Amphoreus just queries
    # against it. A company not registered in Jacquard will run with
    # empty reads; Stelle produces generic output in that case.
    data_source_configured = bool(
        os.environ.get("GCS_CREDENTIALS_B64", "").strip()
        and os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_KEY", "").strip()
    )
    data_source_active = False
    if data_source_configured:
        resolved_company_id = req.companyId or None
        if not resolved_company_id and req.company:
            # Jacquard's user_companies has no slug column — match on
            # name + domains. Normalize both to collapsed lowercase and
            # prefix-match.
            import re as _re
            amph_slug = req.company.strip().lower()
            amph_first = amph_slug.split("-", 1)[0]
            amph_flat = _re.sub(r"[^a-z0-9]+", "", amph_slug)
            try:
                from backend.src.db.supabase_client import get_supabase
                sb = get_supabase()
                rows = (
                    sb.table("user_companies")
                      .select("id, name, domains")
                      .limit(200)
                      .execute()
                      .data
                    or []
                )
                best = None
                for c in rows:
                    name_flat = _re.sub(r"[^a-z0-9]+", "", (c.get("name") or "").lower())
                    domains_flat = "".join(
                        _re.sub(r"[^a-z0-9]+", "", (d or "").lower())
                        for d in (c.get("domains") or [])
                    )
                    if name_flat == amph_flat:
                        best = c
                        break
                    if amph_first and (amph_first in name_flat or amph_first in domains_flat):
                        if best is None or len(c.get("name") or "") < len(best.get("name") or ""):
                            best = c
                if best:
                    resolved_company_id = best.get("id")
                    logger.info(
                        "[ghostwriter] slug %r → company %r (%s)",
                        amph_slug, best.get("name"), resolved_company_id,
                    )
            except Exception as exc:
                logger.debug(
                    "[ghostwriter] Jacquard slug→id lookup failed for %s: %s",
                    req.company, exc,
                )

        if resolved_company_id:
            data_source_active = True
            req.companyId = resolved_company_id
            logger.info(
                "[ghostwriter] data-source active for company=%s (slug=%s uuid=%s)",
                identifier, req.company, resolved_company_id,
            )

    import subprocess
    import sys
    from pathlib import Path
    from backend.src.usage.context import current_user_email

    user_email = current_user_email.get()

    job_id = job_manager.create_job(
        client_slug=identifier,
        agent="stelle",
        prompt=req.prompt,
        creator_id=user_email,
    )

    # Build the subprocess command. Use the same python interpreter the
    # parent is running under so we inherit the same environment (venv,
    # installed packages, env vars).
    project_root = Path(__file__).resolve().parents[4]
    cmd = [
        sys.executable,
        "-m", "backend.src.agents.stelle_runner",
        "--company", identifier,
        "--job-id", job_id,
        "--model", req.model,
    ]
    if req.prompt:
        cmd.extend(["--prompt", req.prompt])
    if user_email:
        cmd.extend(["--user-email", user_email])
    if data_source_active:
        # Pass the Jacquard company UUID — Stelle needs this to query
        # the right transcripts/engagement/research rows.
        cmd.extend(["--company-id", req.companyId or ""])

    # Subprocess log file (separate from Stelle's session log so we can
    # debug runner-level issues independently)
    log_dir = project_root / ".runner-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stelle_runner_{job_id}.out"

    try:
        # start_new_session=True → os.setsid() in the child → new process
        # group that won't receive SIGTERM from uvicorn's reload handler.
        # stdin is closed because the runner is non-interactive; stdout/
        # stderr go to the log file (also tee'd to the child's own
        # logging FileHandler via _configure_logging).
        log_handle = open(log_file, "wb")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        logger.info(
            "[ghostwriter] spawned detached stelle_runner pid=%d job_id=%s company=%s log=%s",
            proc.pid, job_id, req.company, log_file,
        )
    except Exception as exc:
        logger.exception("[ghostwriter] failed to spawn stelle_runner: %s", exc)
        job_manager.set_status(job_id, "failed", error=f"spawn failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not spawn Stelle runner: {exc}")

    return {"job_id": job_id, "status": "pending", "runner_pid": proc.pid}


@router.get("/sandbox/stream")
async def stream_for_company(companyId: str, after_id: int = 0):
    """Per-company bridge that streams the latest run's events.

    Jacquard's ``GhostwriterTerminal`` opens one long-lived SSE connection
    per company (not per job), because in Pi's architecture the event bus
    was per-company. Amphoreus's native stream is per-job, so this endpoint
    resolves ``companyId`` → most recent run → forwards to the per-job
    SSE generator.

    If no run exists for the company, we emit an ``idle`` status event and
    close. If a run is in-flight, we stream its events (with ``after_id``
    resumption) until ``done``/``error`` or the client disconnects.
    """
    from backend.src.db.local import list_runs

    # Pick the latest *active* run (pending / running). If the only
    # run(s) for this company are terminal (completed / failed), treat
    # as "no active run" — otherwise the SSE connection locks onto a
    # dead job and the user's next click on Generate has nowhere to
    # stream to. The idle-pivot branch below will hand off to the new
    # run as soon as it appears.
    _ACTIVE_STATUSES = {"pending", "running"}
    recent_runs = list_runs(companyId, limit=5)
    active_runs = [r for r in recent_runs if r.get("status") in _ACTIVE_STATUSES]

    if not active_runs:
        # No active run yet for this company. Hold the connection open
        # with SSE COMMENT lines (ignored by the client) while polling
        # for a new run every few seconds. When one appears, PIVOT the
        # same open connection into streaming that run's events — so the
        # user's click on Generate immediately starts filling the
        # terminal without needing to reconnect.
        #
        # Implemented as a sync generator so FastAPI can iterate it on a
        # worker thread — mixing ``async def`` with the sync, blocking
        # ``job_manager.sse_stream`` (which uses ``time.sleep``) starved
        # the event loop and prevented the handoff from ever emitting
        # any events.
        import time as _time
        def _idle_then_pivot():
            yield ": idle - no active run for this company yet\n\n"
            while True:
                # Poll every 3 seconds until an active run appears.
                for _ in range(6):  # 6 × 0.5s so keepalives stay warm
                    _time.sleep(0.5)
                    yield ": keepalive\n\n"
                new_runs = list_runs(companyId, limit=5)
                new_active = [r for r in new_runs if r.get("status") in _ACTIVE_STATUSES]
                if new_active:
                    new_job_id = new_active[0].get("id")
                    if new_job_id:
                        # Hand off to the per-job SSE generator — same
                        # one /stream/{job_id} uses — so the client
                        # sees a continuous stream of events for the
                        # newly-started run.
                        for chunk in job_manager.sse_stream(
                            new_job_id,
                            timeout=3600,
                            heartbeat_interval=15,
                            after_id=after_id,
                        ):
                            yield chunk
                        return
        return StreamingResponse(_idle_then_pivot(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    job_id = active_runs[0].get("id")
    if not job_id:
        raise HTTPException(status_code=500, detail="run row missing id")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/{job_id}")
async def stream_events(job_id: str, after_id: int = 0):
    """SSE endpoint — streams AgentEvents from the job's queue.

    Emits SSE keepalive comments (`: keepalive\\n\\n`) every ~15 seconds
    when no real events are available.  This prevents Cloudflare Access
    and other reverse proxies from killing the connection due to their
    idle-read timeouts (typically 100 s for CF).

    ``after_id`` lets a reconnecting client resume from the last event
    id it saw, so mid-stream drops don't cause replay or loss.
    """
    from backend.src.db.local import get_run
    if not job_manager.get_job(job_id) and not get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/stop")
async def stop_run(request: Request):
    """Kill any running stelle_runner subprocess for the given company.

    Called by Jacquard's ``POST /api/ghostwriter/stop`` proxy when a user
    clicks Stop in the Lineage UI. We find the latest ``running`` run for
    the company, identify its runner PID (via /proc scan for the
    ``--job-id`` CLI arg), and send SIGTERM. Also marks the run as
    ``failed`` with reason "stopped by user" so the SSE stream closes.
    """
    body = await request.json()
    company_id_or_slug = body.get("companyId") or body.get("company") or ""
    if not company_id_or_slug:
        raise HTTPException(status_code=400, detail="companyId required")

    import os
    import signal
    import time as _time
    from backend.src.db.local import list_runs, complete_run

    runs = list_runs(company_id_or_slug, limit=5)
    running = [r for r in runs if r.get("status") == "running"]
    if not running:
        return {"stopped": True, "killed_pids": [], "note": "no running run found"}

    killed: list[dict] = []
    for run in running:
        job_id = run.get("id")
        if not job_id:
            continue
        # Scan /proc for the matching runner subprocess. stelle_runner is
        # spawned with start_new_session=True — it's in its own process
        # group — so SIGTERM to the pid is enough.
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            try:
                with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            except Exception:
                continue
            if "stelle_runner" not in cmd or job_id not in cmd:
                continue
            try:
                os.kill(int(pid_str), signal.SIGTERM)
                killed.append({"pid": int(pid_str), "job_id": job_id})
                logger.info("[stop] SIGTERM sent to stelle_runner pid=%s job_id=%s", pid_str, job_id)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("[stop] kill failed for pid=%s: %s", pid_str, exc)

        # Flip run status so the SSE stream can close and the UI reflects it.
        try:
            complete_run(job_id, output=None, error="stopped by user")
        except Exception:
            logger.warning("[stop] complete_run failed for %s", job_id, exc_info=True)

    return {"stopped": True, "killed_pids": killed, "count": len(killed)}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "output": job.get("output"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


@router.post("/provision")
async def provision_workspace(req: GenerateRequest):
    """Provision a persistent workspace for a client."""
    from backend.src.services.workspace_manager import provision_workspace
    result = provision_workspace(req.company)
    return {"status": "provisioned", "workspace": result}



@router.post("/inline-edit")
async def inline_edit(req: InlineEditRequest, request: Request):
    """Inline text editing via Stelle — returns a job_id for SSE streaming.

    Accepts {company, post_text} (the selection is treated as
    ``post_text`` — Stelle returns a revised version of the passed span).
    """
    identifier = req.company or req.companyId
    text = req.post_text or req.selectedText or ""

    if not identifier:
        raise HTTPException(
            status_code=400,
            detail="inline-edit requires 'company' (slug) or 'companyId' (UUID)",
        )
    if not text:
        raise HTTPException(
            status_code=400,
            detail="inline-edit requires 'post_text' or 'selectedText'",
        )

    job_id = job_manager.create_job(
        client_slug=identifier,
        agent="stelle-inline-edit",
        prompt=req.instruction,
        creator_id=None,
    )

    def _run_edit(jid: str, company: str, post_text: str, instruction: str):
        from backend.src.agents.stelle_adapter import run_inline_edit
        result = run_inline_edit(company, post_text, instruction, job_id=jid)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(
        job_id,
        target=_run_edit,
        args=(job_id, identifier, text, req.instruction),
    )
    return {"job_id": job_id, "status": "pending"}


@router.get("/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """Full event timeline for a specific run."""
    from backend.src.db.local import get_run_events, get_run
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    events = get_run_events(run_id)
    return {"run": run, "events": events}


@router.get("/{company}/runs")
async def get_run_history(company: str, limit: int = Query(20)):
    """Run history for a client."""
    from backend.src.db.local import list_runs
    runs = list_runs(company, limit=limit)
    return {"runs": runs}


@router.post("/{company}/rollback/{run_id}")
async def rollback_to_run(company: str, run_id: str):
    """Rollback workspace to a previous run's snapshot."""
    from backend.src.services.workspace_manager import rollback_snapshot
    rollback_snapshot(company, run_id)
    return {"status": "rolled_back", "run_id": run_id}


@router.get("/{company}/ordinal-users")
async def list_ordinal_users(company: str):
    """Workspace members from Ordinal (for approver picker). Requires API key in ordinal_auth CSV."""
    from backend.src.agents.hyacinthia import Hyacinthia
    raw = Hyacinthia().get_users(company)
    if isinstance(raw, list):
        users = raw
    elif isinstance(raw, dict):
        users = raw.get("users") or raw.get("data") or []
    else:
        users = []
    return {"users": users}


@router.get("/sandbox/{company}/files")
async def browse_workspace_files(company: str, path: str = Query("")):
    """Browse workspace files."""
    from backend.src.services.workspace_manager import list_workspace_files
    files = list_workspace_files(company, path)
    return {"files": files}


@router.get("/{company}/linkedin-username")
async def get_linkedin_username(company: str):
    """Return the stored LinkedIn username for a client, or null if not set."""
    from backend.src.db import vortex
    path = vortex.linkedin_username_path(company)
    if not path.exists():
        return {"username": None}
    return {"username": path.read_text(encoding="utf-8").strip() or None}


@router.post("/{company}/linkedin-username")
async def save_linkedin_username(company: str, req: LinkedInUsernameRequest):
    """Write the LinkedIn username file for a client."""
    from backend.src.db import vortex
    username = req.username.strip().lstrip("@").strip("/").split("/")[-1]
    if not username:
        raise HTTPException(status_code=400, detail="Username must not be empty")
    path = vortex.linkedin_username_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(username, encoding="utf-8")
    return {"status": "saved", "username": username}


# ---------------------------------------------------------------------------
# Calendar — post scheduling interface
# ---------------------------------------------------------------------------


class ScheduleRequest(BaseModel):
    scheduled_date: str | None = None  # ISO date e.g. "2026-04-14" or null to unschedule


class AutoAssignRequest(BaseModel):
    cadence: str = "3pw"  # "3pw" (Mon/Tue/Thu) or "2pw" (Tue/Thu)
    start_date: str | None = None  # ISO date; defaults to next Monday


class PushAllRequest(BaseModel):
    pass  # no body needed; pushes all unpushed posts with scheduled_dates


@router.get("/{company}/calendar")
async def get_calendar(company: str, month: str = Query(None)):
    """Return all posts for a company, optionally filtered to a month.

    Posts include scheduled_date, publication_order, status, hook preview,
    and ordinal_post_id. Suitable for rendering a calendar grid.
    """
    from backend.src.db.local import list_calendar_posts
    posts = list_calendar_posts(company, month=month)
    result = []
    for p in posts:
        content = p.get("content") or ""
        hook = content.split("\n")[0][:120] if content else ""
        result.append({
            "id": p.get("id"),
            "hook": hook,
            "content": content,
            "content_preview": content[:300],
            "status": p.get("status"),
            "scheduled_date": p.get("scheduled_date"),
            "publication_order": p.get("publication_order"),
            "ordinal_post_id": p.get("ordinal_post_id"),
            "created_at": p.get("created_at"),
            "why_post": p.get("why_post"),
        })
    return {"company": company, "month": month, "posts": result}


@router.patch("/{company}/posts/{post_id}/schedule")
async def schedule_post(company: str, post_id: str, req: ScheduleRequest):
    """Update the scheduled publication date for a post (drag-drop on calendar)."""
    from backend.src.db.local import update_post_schedule, get_local_post
    post = get_local_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.get("company") != company:
        raise HTTPException(status_code=403, detail="Post belongs to a different company")
    updated = update_post_schedule(post_id, req.scheduled_date)
    return updated


@router.post("/{company}/calendar/auto-assign")
async def auto_assign_calendar(company: str, req: AutoAssignRequest):
    """Auto-assign unscheduled posts to calendar slots based on cadence.

    Distributes posts in publication_order into the cadence's day slots
    (Mon/Tue/Thu for 3pw, Tue/Thu for 2pw) starting from start_date
    (defaults to next Monday).
    """
    from datetime import date, timedelta
    from backend.src.db.local import list_calendar_posts, update_post_schedule

    posts = list_calendar_posts(company)
    unscheduled = [p for p in posts if not p.get("scheduled_date") and p.get("status") == "draft"]
    unscheduled.sort(key=lambda p: p.get("publication_order") or 999)

    if not unscheduled:
        return {"assigned": 0, "message": "No unscheduled draft posts"}

    # Determine cadence days (0=Mon, 1=Tue, ..., 4=Fri)
    if req.cadence == "2pw":
        cadence_days = {1, 3}  # Tue, Thu
    else:
        cadence_days = {0, 1, 3}  # Mon, Tue, Thu

    # Start from: explicit start_date, OR the day after the last
    # already-scheduled post, OR today — whichever is latest. Then
    # advance to the next cadence day (including today if it's one).
    if req.start_date:
        try:
            cursor = date.fromisoformat(req.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    else:
        today = date.today()
        # Find the latest already-scheduled post date
        scheduled_dates = [
            p["scheduled_date"]
            for p in posts
            if p.get("scheduled_date")
        ]
        if scheduled_dates:
            try:
                last_scheduled = max(date.fromisoformat(d) for d in scheduled_dates)
                # Start the day after the last scheduled post
                cursor = max(today, last_scheduled + timedelta(days=1))
            except Exception:
                cursor = today
        else:
            cursor = today

        # Advance cursor to the next cadence day (including today)
        for _ in range(7):
            if cursor.weekday() in cadence_days:
                break
            cursor += timedelta(days=1)

    assigned = []
    post_idx = 0
    max_search_days = 90  # don't look more than 3 months out

    for _ in range(max_search_days):
        if post_idx >= len(unscheduled):
            break
        if cursor.weekday() in cadence_days:
            post = unscheduled[post_idx]
            date_str = cursor.isoformat()
            update_post_schedule(post["id"], date_str)
            assigned.append({"id": post["id"], "scheduled_date": date_str})
            post_idx += 1
        cursor += timedelta(days=1)

    return {"assigned": len(assigned), "posts": assigned}


@router.post("/{company}/calendar/push-all")
async def push_all_scheduled(company: str):
    """Push all unpushed posts with scheduled_dates to Ordinal.

    For each post with status='draft' and a scheduled_date, calls
    Hyacinthia to push to Ordinal with that date as the publish date.
    Returns the results per post.
    """
    from backend.src.db.local import list_calendar_posts
    from backend.src.agents.hyacinthia import Hyacinthia

    posts = list_calendar_posts(company)
    pushable = [
        p for p in posts
        if p.get("status") == "draft"
        and p.get("scheduled_date")
        and not p.get("ordinal_post_id")
    ]

    if not pushable:
        return {"pushed": 0, "message": "No draft posts with scheduled dates to push"}

    hy = Hyacinthia()
    results = []
    for post in pushable:
        try:
            ordinal_result = hy.push_post(
                company,
                post["id"],
                post["content"],
                scheduled_date=post["scheduled_date"],
                why_post=post.get("why_post"),
            )
            results.append({
                "id": post["id"],
                "status": "pushed",
                "ordinal_post_id": ordinal_result.get("ordinal_post_id"),
                "scheduled_date": post["scheduled_date"],
            })
        except Exception as e:
            results.append({
                "id": post["id"],
                "status": "failed",
                "error": str(e)[:200],
            })

    return {"pushed": sum(1 for r in results if r["status"] == "pushed"), "results": results}


class PushSingleRequest(BaseModel):
    post_id: str


@router.post("/{company}/calendar/push-single")
async def push_single_post(company: str, req: PushSingleRequest):
    """Push one specific post to Ordinal."""
    from backend.src.db.local import get_local_post
    from backend.src.agents.hyacinthia import Hyacinthia

    post = get_local_post(req.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.get("company") != company:
        raise HTTPException(status_code=403, detail="Post belongs to a different company")
    if post.get("ordinal_post_id"):
        return {"id": req.post_id, "status": "already_pushed", "ordinal_post_id": post["ordinal_post_id"]}

    try:
        hy = Hyacinthia()
        ordinal_result = hy.push_post(
            company,
            post["id"],
            post["content"],
            scheduled_date=post.get("scheduled_date"),
            why_post=post.get("why_post"),
        )
        return {
            "id": req.post_id,
            "status": "pushed",
            "ordinal_post_id": ordinal_result.get("ordinal_post_id"),
            "scheduled_date": post.get("scheduled_date"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)[:200]}")


