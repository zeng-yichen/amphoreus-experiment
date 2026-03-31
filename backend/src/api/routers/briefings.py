"""Briefings API — Aglaea interview prep generation."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/briefings", tags=["briefings"])


class BriefingRequest(BaseModel):
    client_name: str
    company: str


@router.post("/generate")
async def generate_briefing(req: BriefingRequest):
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="aglaea",
        prompt=f"Generate briefing for {req.client_name}",
        creator_id=None,
    )

    def _run(jid: str, client_name: str, company: str):
        from backend.src.agents.aglaea_adapter import run_aglaea
        job_manager.emit_event(jid, status_event(f"Generating briefing for {client_name}..."))
        result = run_aglaea(client_name, company, job_id=jid)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(job_id, target=_run, args=(job_id, req.client_name, req.company))
    return {"job_id": job_id, "status": "pending"}


@router.get("/check/{company}")
async def check_briefing(company: str):
    """Check whether an Aglaea briefing exists for a company."""
    from backend.src.db import vortex
    brief_file = vortex.brief_dir(company) / f"{company}_briefing.md"
    return {"exists": brief_file.exists()}


@router.get("/content/{company}")
async def get_briefing_content(company: str):
    """Return the raw markdown content of the latest Aglaea briefing for a company."""
    from backend.src.db import vortex
    brief_file = vortex.brief_dir(company) / f"{company}_briefing.md"
    if not brief_file.exists():
        raise HTTPException(status_code=404, detail="No briefing found")
    return {"content": brief_file.read_text(encoding="utf-8")}


@router.get("/stream/{job_id}")
async def stream_briefing(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def _gen():
        for event in job_manager.drain_events(job_id, timeout=600):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
