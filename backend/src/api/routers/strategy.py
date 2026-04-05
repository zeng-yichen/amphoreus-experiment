"""Strategy API — Herta content strategy generation + learned strategy briefs."""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategy", tags=["strategy"])

# Strategy brief staleness threshold — after this many days, the /brief
# endpoint returns 404 with a hint to refresh.
_BRIEF_STALENESS_DAYS = 7


class StrategyRequest(BaseModel):
    company: str
    prompt: str | None = None


# ---------------------------------------------------------------------------
# Learned strategy brief endpoints (A4 pipeline output)
# ---------------------------------------------------------------------------
# These are declared BEFORE the /{company} catchall routes below so FastAPI
# matches them by literal path prefix ("brief", "topics", "causal") rather
# than treating those words as company slugs.


@router.get("/brief/{company}")
async def get_strategy_brief(company: str):
    """Return the auto-generated strategy brief markdown for a client.

    404 if the brief doesn't exist or is stale (> 7 days old). The staleness
    check keeps the frontend from displaying data that no longer reflects
    current engagement patterns. Clients should call /refresh to regenerate.
    """
    from backend.src.db import vortex
    brief_path = vortex.memory_dir(company) / "strategy_brief.md"
    if not brief_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No strategy brief exists for {company}. Trigger an ordinal "
                "sync or call /api/strategy/brief/{company}/refresh to generate one."
            ),
        )

    mtime = datetime.fromtimestamp(brief_path.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400
    if age_days > _BRIEF_STALENESS_DAYS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Strategy brief for {company} is {age_days:.1f} days old "
                f"(threshold: {_BRIEF_STALENESS_DAYS} days). Call "
                "/api/strategy/brief/{company}/refresh to regenerate."
            ),
        )

    try:
        brief_md = brief_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read brief: {e}")

    return {
        "company": company,
        "brief": brief_md,
        "generated_at": mtime.isoformat(),
        "age_days": round(age_days, 2),
    }


@router.get("/brief/{company}/refresh")
async def refresh_strategy_brief(company: str):
    """Force a fresh strategy brief generation and return the result.

    Calls ``generate_strategy_brief(company, force=True)`` synchronously.
    For clients with sparse data this may return 404 if the brief generator
    has insufficient signal to produce anything useful.
    """
    try:
        from backend.src.utils.strategy_brief import generate_strategy_brief
        brief_md = generate_strategy_brief(company, force=True)
    except Exception as e:
        logger.exception("[strategy] Refresh failed for %s", company)
        raise HTTPException(status_code=500, detail=f"Brief generation failed: {e}")

    if not brief_md:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{company} has insufficient data to generate a meaningful "
                "strategy brief (needs scored observations with tags and a "
                "transition model). Run ordinal sync first."
            ),
        )

    from backend.src.db import vortex
    brief_path = vortex.memory_dir(company) / "strategy_brief.md"
    mtime = datetime.fromtimestamp(brief_path.stat().st_mtime, tz=timezone.utc) \
        if brief_path.exists() else datetime.now(timezone.utc)

    return {
        "company": company,
        "brief": brief_md,
        "generated_at": mtime.isoformat(),
        "age_days": 0.0,
    }


@router.get("/topics/{company}")
async def get_topic_transitions(company: str):
    """Return the raw topic_transitions.json for a client.

    Used by the frontend to render topic transition visualizations
    (Markov graphs, sequence timelines, etc.).
    """
    from backend.src.db import vortex
    path = vortex.memory_dir(company) / "topic_transitions.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No topic transition model for {company}. Needs at least 15 "
                "tagged scored observations. Run ordinal sync first."
            ),
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read transitions: {e}")


@router.get("/causal/{company}")
async def get_causal_dimensions(company: str):
    """Return the raw causal_dimensions.json for a client.

    The partial-correlation analysis that classifies content state dimensions
    as causal / confounded / uncertain / inert. Used by the frontend to show
    which features genuinely predict engagement for a client.
    """
    from backend.src.db import vortex
    path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No causal filter output for {company}. Needs at least 30 "
                "tagged scored observations. Run ordinal sync first."
            ),
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read causal data: {e}")


# ---------------------------------------------------------------------------
# Herta content strategy generation endpoints
# ---------------------------------------------------------------------------


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
