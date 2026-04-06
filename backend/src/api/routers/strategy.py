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


@router.get("/findings/{company}")
async def get_analyst_findings(company: str):
    """Return the analyst agent's findings for a client.

    The analyst runs hypothesis-driven engagement analysis using statistical
    tools + LinkedIn-wide data. Its findings replace the old fixed pipeline's
    strategy brief, topic transitions, and causal dimensions with a single
    adaptive, open-ended analysis.

    404 if no findings exist or the latest run is stale (> 7 days).
    """
    from backend.src.db import vortex
    path = vortex.memory_dir(company) / "analyst_findings.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No analyst findings for {company}. Needs at least 10 "
                "scored observations. Run ordinal sync to trigger the analyst."
            ),
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read findings: {e}")

    runs = data.get("runs", [])
    last_run = runs[-1] if runs else {}
    last_ts = last_run.get("timestamp", "")
    age_days = 0.0
    if last_ts:
        try:
            dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        except Exception:
            pass

    if age_days > _BRIEF_STALENESS_DAYS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analyst findings for {company} are {age_days:.1f} days old "
                f"(threshold: {_BRIEF_STALENESS_DAYS} days). Run ordinal sync "
                "to trigger a fresh analysis."
            ),
        )

    return {
        "company": company,
        "findings": data.get("findings", []),
        "last_run": last_run,
        "age_days": round(age_days, 2),
        "total_runs": len(runs),
    }


@router.get("/findings/{company}/refresh")
async def refresh_analyst_findings(company: str):
    """Run a fresh analyst analysis for a client and return the findings.

    Runs the full analyst agent synchronously (typically 2-4 minutes,
    ~$2-3 in LLM cost). Use sparingly — the analyst is designed to run
    weekly, not on-demand for every request.
    """
    try:
        from backend.src.agents.analyst import run_analysis
        result = run_analysis(company)
    except Exception as e:
        logger.exception("[strategy] Analyst refresh failed for %s", company)
        raise HTTPException(status_code=500, detail=f"Analyst failed: {e}")

    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])

    from backend.src.db import vortex
    path = vortex.memory_dir(company) / "analyst_findings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {"findings": [], "runs": []}

    return {
        "company": company,
        "findings": data.get("findings", []),
        "last_run": result,
        "age_days": 0.0,
        "total_runs": len(data.get("runs", [])),
    }


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
