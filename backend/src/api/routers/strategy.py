"""Strategy API — Herta content strategy generation."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategy", tags=["strategy"])


class StrategyRequest(BaseModel):
    company: str
    prompt: str | None = None


@router.post("/generate")
async def generate_strategy(req: StrategyRequest):
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="herta",
        prompt=req.prompt,
        creator_id=None,
    )

    def _run(jid: str, company: str, prompt: str | None):
        from backend.src.agents.herta_adapter import run_herta
        job_manager.emit_event(jid, status_event(f"Generating strategy for {company}..."))
        result = run_herta(company, prompt, job_id=jid)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(job_id, target=_run, args=(job_id, req.company, req.prompt))
    return {"job_id": job_id, "status": "pending"}


@router.get("/stream/{job_id}")
async def stream_strategy(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def _gen():
        for event in job_manager.drain_events(job_id, timeout=600):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/{company}/html")
async def get_strategy_html(company: str):
    """Get the most recent content strategy HTML for a client (JSON response)."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        return {"html": None}
    files = sorted(strategy_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"html": None}
    return {"html": files[0].read_text(encoding="utf-8")}


@router.get("/{company}/view", response_class=HTMLResponse)
async def view_strategy_html(company: str):
    """Serve the most recent content strategy HTML directly for browser rendering."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        raise HTTPException(status_code=404, detail="No content strategy found")
    files = sorted(strategy_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No content strategy HTML found")
    return HTMLResponse(content=files[0].read_text(encoding="utf-8"))


@router.get("/{company}")
async def get_current_strategy(company: str):
    """Get the most recent content strategy for a client."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        return {"strategy": None}
    files = sorted(strategy_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"strategy": None}
    return {"strategy": files[0].read_text(encoding="utf-8"), "path": str(files[0])}
