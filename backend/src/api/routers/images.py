"""Images API — generate, stream, list, and serve assembled images."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])


class GenerateImageRequest(BaseModel):
    company: str
    post_text: str
    model: str = "claude-opus-4-6"


@router.post("/generate")
async def generate_image(req: GenerateImageRequest):
    """Start an image assembly job."""
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="phainon",
        prompt=req.post_text[:200],
        creator_id=None,
    )

    def _run(jid: str, company: str, post_text: str, model: str):
        from backend.src.agents.phainon_adapter import run_phainon
        job_manager.emit_event(jid, status_event(f"Starting image assembly for {company}..."))
        result_path = run_phainon(company, post_text, model, job_id=jid)
        job_manager.emit_event(jid, done_event(result_path or ""))
        return result_path

    job_manager.run_in_background(
        job_id,
        target=_run,
        args=(job_id, req.company, req.post_text, req.model),
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


@router.get("/{company}")
async def list_images(company: str, limit: int = Query(50)):
    """List generated images for a company."""
    from backend.src.db import vortex
    img_dir = vortex.images_dir(company)
    if not img_dir.exists():
        return {"images": []}

    images = []
    for f in sorted(img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta_path = f.with_name(f.stem + "_metadata.json")
        metadata = None
        if meta_path.exists():
            import json
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        images.append({
            "id": f.stem,
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "created_at": f.stat().st_mtime,
            "metadata": metadata,
        })

    return {"images": images}


@router.get("/{company}/{image_id}")
async def serve_image(company: str, image_id: str):
    """Serve an image file."""
    from backend.src.db import vortex
    img_dir = vortex.images_dir(company)

    image_path = img_dir / f"{image_id}.png"
    if not image_path.exists():
        image_path = img_dir / image_id
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=image_path.name,
    )
