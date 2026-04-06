"""Ghostwriter API — generate, stream, manage workspaces."""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ghostwriter", tags=["ghostwriter"])


class GenerateRequest(BaseModel):
    company: str
    prompt: str | None = None
    model: str = "claude-opus-4-6"


class InlineEditRequest(BaseModel):
    company: str
    post_text: str
    instruction: str


class LinkedInUsernameRequest(BaseModel):
    username: str


class FeedbackRequest(BaseModel):
    company: str
    original: str
    revised: str


class ClientFeedbackRequest(BaseModel):
    company: str
    text: str
    post_id: str | None = None


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Start a ghostwriter generation job."""
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="stelle",
        prompt=req.prompt,
        creator_id=None,
    )

    def _run(jid: str, company: str, prompt: str | None, model: str):
        from backend.src.agents.stelle_adapter import run_stelle
        from backend.src.services.workspace_manager import create_snapshot
        job_manager.emit_event(jid, status_event(f"Starting generation for {company}..."))
        result = run_stelle(company, prompt, model, job_id=jid)
        try:
            snap_path = create_snapshot(company, jid)
            job_manager.emit_event(jid, status_event(f"Snapshot saved: {snap_path}"))
        except Exception as e:
            logger.warning("Snapshot failed for %s: %s", jid, e)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(
        job_id,
        target=_run,
        args=(job_id, req.company, req.prompt, req.model),
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/stream/{job_id}")
async def stream_events(job_id: str):
    """SSE endpoint — streams AgentEvents from the job's queue."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def _event_generator():
        for event in job_manager.drain_events(job_id, timeout=600):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit edit feedback (before/after pair) — stored in workspace feedback/edits/."""
    from backend.src.services.workspace_manager import save_feedback
    save_feedback(req.company, req.original, req.revised)
    return {"status": "saved"}


@router.post("/client-feedback")
async def submit_client_feedback(req: ClientFeedbackRequest):
    """Submit raw client feedback (Slack messages, email notes, verbal instructions).

    Stored in memory/{company}/feedback/ as a timestamped file. The feedback
    distiller reads these and converts them into learned writing rules that
    get injected into Stelle's system prompt.
    """
    import time as _t
    from backend.src.db import vortex
    feedback_dir = vortex.feedback_dir(req.company)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _t.strftime("%Y%m%d_%H%M%S")
    prefix = f"post_{req.post_id[:8]}_" if req.post_id else ""
    filename = f"{prefix}client_{timestamp}.txt"
    filepath = feedback_dir / filename
    filepath.write_text(req.text, encoding="utf-8")

    return {"status": "saved", "file": filename}


@router.get("/feedback/{company}")
async def list_feedback(company: str):
    """List all feedback files and learned directives for a client."""
    import json
    from backend.src.db import vortex

    # Raw feedback files
    feedback_dir = vortex.feedback_dir(company)
    feedback_files = []
    if feedback_dir.exists():
        for f in sorted(feedback_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.is_file() and f.suffix in (".txt", ".md"):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    feedback_files.append({
                        "name": f.name,
                        "path": str(f.relative_to(feedback_dir)),
                        "preview": text[:300],
                        "full_text": text,
                        "modified": f.stat().st_mtime,
                    })
                except Exception:
                    pass

    # Learned directives (what the distiller extracted from this feedback)
    directives = []
    directives_path = vortex.memory_dir(company) / "learned_directives.json"
    if directives_path.exists():
        try:
            data = json.loads(directives_path.read_text(encoding="utf-8"))
            directives = data.get("directives", [])
        except Exception:
            pass

    return {
        "feedback_files": feedback_files,
        "feedback_count": len(feedback_files),
        "directives": directives,
        "directives_count": len(directives),
    }


@router.post("/inline-edit")
async def inline_edit(req: InlineEditRequest):
    """Inline text editing via Stelle — returns a job_id for SSE streaming."""
    job_id = job_manager.create_job(
        client_slug=req.company,
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
        args=(job_id, req.company, req.post_text, req.instruction),
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


# Manual feedback ingestion endpoint removed: feedback learning happens through
# RuanMei observations (draft → client edits → final post → engagement).
# No separate feedback ingestion pipeline needed.
