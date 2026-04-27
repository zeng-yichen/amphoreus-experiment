"""Strategy API — Cyrene strategic reviews + Herta content strategy generation.

Cyrene (strategic growth agent) produces a JSON brief with interview
questions, DM targets, content priorities, ABM targeting, and Stelle
scheduling. Runs on demand via POST /api/strategy/cyrene/{company}.

Herta (content strategy document generator) is the older, static
strategy path — still available at /generate for clients that prefer
a formatted strategy document over Cyrene's live brief.
"""

import json
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


# ---------------------------------------------------------------------------
# Cyrene — strategic growth agent
# ---------------------------------------------------------------------------


@router.post("/cyrene/{company}")
async def run_cyrene_review(company: str, userId: str | None = None):
    """Start a Cyrene strategic review as a background job.

    Cyrene is a turn-based Opus agent that studies the client's full
    engagement history, ICP exposure trends, warm prospects, and
    transcript inventory, then produces a strategic brief with:
    interview questions, DM targets, content priorities, ABM targeting,
    Stelle scheduling, and a self-scheduled next-run trigger.

    ``userId`` query param (optional): scope the brief to one FOC user
    at a multi-FOC company. Required in spirit for any multi-FOC client
    (Virio, Trimble, Commenda) — a single company-wide brief can't
    direct 19 distinct-role teammates, so the brief gets keyed and
    retrieved per user. Omit for single-FOC companies (Hume, Innovo,
    Flora) — runs company-wide as before.

    Returns a job_id; connect to /stream/{job_id} for SSE progress.
    """
    # Canonicalize the company identifier (slug|UUID → UUID) + resolve
    # the target FOC user. Cyrene's rewired query tools need a
    # ``DATABASE_USER_UUID`` to scope handle resolution — at multi-FOC
    # companies (Trimble/Commenda/Virio) running company-wide leaves
    # every tool stuck at "could not resolve creator handle," which is
    # exactly the failure Mark's 2026-04-24 00:50 brief diagnosed.
    #
    # Resolution priority:
    #   (1) explicit ``userId`` query param from the UI
    #   (2) slug-embedded user — ``resolve_to_company_and_user`` handles
    #       per-FOC slugs like ``trimble-heather`` → (company_uuid, user_id)
    #   (3) single-FOC companies (Hume, Innovo, Flora, Hensley, etc.) —
    #       auto-pick the only active FOC
    #   (4) multi-FOC + no hint — auto-pick the most-recently-active FOC
    #       so the run isn't dead-on-arrival; log loudly so we can tell
    #       the UI needs to pass userId for that company
    from backend.src.lib.company_resolver import (
        resolve_with_fallback, resolve_to_company_and_user,
    )
    company_uuid_from_slug, user_id_from_slug = resolve_to_company_and_user(company)
    company = company_uuid_from_slug or resolve_with_fallback(company) or company

    user_id = (
        (userId or "").strip()
        or (user_id_from_slug or "").strip()
        or None
    )

    if not user_id:
        # Auto-pick an active FOC so tools don't flail. Use the users
        # table: posts_content=true rows at this company, order by
        # posts_per_month desc then name to stabilize picks.
        try:
            from backend.src.db.amphoreus_supabase import _get_client
            _sb = _get_client()
            if _sb is not None:
                _foc_rows = (
                    _sb.table("users")
                       .select("id,first_name,last_name,posts_per_month")
                       .eq("company_id", company)
                       .eq("posts_content", True)
                       .execute().data or []
                )
                if _foc_rows:
                    # Most active FOC first; ties broken by name for stability.
                    _foc_rows.sort(
                        key=lambda u: (-(u.get("posts_per_month") or 0),
                                       (u.get("first_name") or "") + (u.get("last_name") or "")),
                    )
                    picked = _foc_rows[0]
                    user_id = picked.get("id")
                    logger.warning(
                        "[Cyrene endpoint] no userId supplied for %s — "
                        "auto-picked FOC %s %s (id=%s) out of %d active FOCs. "
                        "Frontend should pass userId to avoid this ambiguity.",
                        company,
                        (picked.get("first_name") or "").strip(),
                        (picked.get("last_name") or "").strip(),
                        user_id, len(_foc_rows),
                    )
        except Exception as _exc:
            logger.warning(
                "[Cyrene endpoint] FOC auto-pick for %s failed: %s", company, _exc,
            )

    job_id = job_manager.create_job(
        client_slug=company,
        agent="cyrene",
        prompt=None,
        creator_id=None,
    )

    def _run(jid: str, co: str, uid: str | None):
        from backend.src.agents.cyrene import run_strategic_review
        _scope = f"user={uid}" if uid else "company-wide"
        job_manager.emit_event(jid, status_event(
            f"Starting Cyrene strategic review for {co} ({_scope})..."
        ))
        try:
            brief = run_strategic_review(co, user_id=uid)
            if brief.get("_error"):
                job_manager.emit_event(jid, status_event(f"Cyrene failed: {brief['_error']}"))
                return brief
            job_manager.emit_event(jid, done_event(json.dumps(brief, default=str)))
            return brief
        except Exception as e:
            job_manager.emit_event(jid, status_event(f"Cyrene crashed: {e}"))
            raise

    job_manager.run_in_background(job_id, target=_run, args=(job_id, company, user_id))
    return {"job_id": job_id, "status": "pending", "user_id": user_id}


@router.get("/cyrene/{company}/brief")
async def get_cyrene_brief(company: str, userId: str | None = None):
    """Return the most recent Cyrene brief for this FOC from Amphoreus Supabase.

    Source of truth is ``cyrene_briefs`` in Supabase (rewired 2026-04-24).
    The fly-local ``memory/<company>/cyrene_brief.json`` path is retired
    as a primary read — it was a single-point-of-failure and had no
    multi-FOC scoping.

    ``userId`` query param mirrors the POST endpoint's 4-layer
    resolution: explicit → slug-embedded → single-FOC auto-pick →
    multi-FOC auto-pick-most-active. The UI reads the same brief that
    the latest POST /cyrene/{company} run wrote, and at multi-FOC
    clients we key by FOC user_id so each FOC sees their own brief.
    """
    from backend.src.lib.company_resolver import (
        resolve_with_fallback, resolve_to_company_and_user,
    )
    from backend.src.db.amphoreus_supabase import get_latest_cyrene_brief

    # Canonicalize + resolve user (same 4-layer logic as the POST
    # endpoint so GET/POST see the same brief).
    company_uuid_from_slug, user_id_from_slug = resolve_to_company_and_user(company)
    company_uuid = company_uuid_from_slug or resolve_with_fallback(company) or company

    user_id = (
        (userId or "").strip()
        or (user_id_from_slug or "").strip()
        or None
    )

    if not user_id:
        try:
            from backend.src.db.amphoreus_supabase import _get_client
            _sb = _get_client()
            if _sb is not None:
                _foc_rows = (
                    _sb.table("users")
                       .select("id,first_name,last_name,posts_per_month")
                       .eq("company_id", company_uuid)
                       .eq("posts_content", True)
                       .execute().data or []
                )
                if _foc_rows:
                    _foc_rows.sort(
                        key=lambda u: (-(u.get("posts_per_month") or 0),
                                       (u.get("first_name") or "") + (u.get("last_name") or "")),
                    )
                    user_id = _foc_rows[0].get("id")
        except Exception as _exc:
            logger.warning(
                "[Cyrene GET brief] FOC auto-pick for %s failed: %s", company_uuid, _exc,
            )

    # strict_user_only=True when we have a user_id — at multi-FOC clients
    # (e.g. Trimble, Commenda) the absence of a per-FOC brief MUST NOT
    # silently fall back to another FOC's brief. That's how Heather started
    # seeing Mark's brief as her own (2026-04-27 diagnosis).
    brief = get_latest_cyrene_brief(
        company_uuid, user_id=user_id, strict_user_only=bool(user_id),
    )
    if brief is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No Cyrene brief for {company}"
                + (f" (user_id={user_id})" if user_id else "")
                + f". Run POST /api/strategy/cyrene/{company} first."
            ),
        )
    return brief


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
async def stream_strategy(job_id: str, after_id: int = 0):
    from backend.src.db.local import get_run
    if not job_manager.get_job(job_id) and not get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
