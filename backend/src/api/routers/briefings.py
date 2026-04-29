"""Briefings API — reads Cyrene's strategic brief for interview prep."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/briefings", tags=["briefings"])


def _load_cyrene_brief_from_supabase(company: str) -> dict | None:
    """Read the latest Cyrene brief for ``company`` from Supabase.

    2026-04-29: replaces the legacy fly-local fallback. Briefs live
    exclusively in ``cyrene_briefs``. Returns None if Supabase has no
    row for this company (or read fails) — caller surfaces 404.
    """
    try:
        from backend.src.db.amphoreus_supabase import get_latest_cyrene_brief
        return get_latest_cyrene_brief(company, strict_user_only=False)
    except Exception as exc:
        logger.warning("[briefings] Supabase brief lookup failed for %s: %s", company, exc)
        return None


def _brief_to_markdown(brief: dict) -> str:
    """Render a Cyrene brief dict as markdown for ReactMarkdown consumption.

    Brief shape (2026-04-22 BL cleanup):
      * ``prose`` — free-form strategic memo (primary)
      * ``dm_targets`` — structured handoff list
      * ``next_run_trigger`` — when Cyrene should run again

    Legacy schema (``content_priorities``, ``content_avoid``,
    ``icp_exposure_assessment``, ``stelle_timing``) rendered as
    fallback for pre-migration briefs.
    """
    lines: list[str] = []
    company = brief.get("_company", "")
    computed = brief.get("_computed_at", "")
    lines.append(f"# Cyrene Brief — {company}")
    if computed:
        lines.append(f"\n_Computed: {computed}_\n")

    prose = (brief.get("prose") or "").strip()
    if prose:
        lines.append("## Strategic Memo\n")
        lines.append(prose)
        lines.append("")
    else:
        # Legacy fallback — old rigid-schema briefs still render.
        assess = brief.get("icp_exposure_assessment")
        if assess:
            lines.append("## ICP / Reach Assessment\n")
            lines.append(str(assess).strip('"'))
            lines.append("")

        timing = brief.get("stelle_timing")
        if timing:
            lines.append("## Stelle Timing\n")
            lines.append(str(timing).strip('"'))
            lines.append("")

        prios = brief.get("content_priorities") or []
        if prios:
            lines.append("## Content Priorities\n")
            for i, p in enumerate(prios, 1):
                lines.append(f"{i}. {p}")
            lines.append("")

        avoid = brief.get("content_avoid") or []
        if avoid:
            lines.append("## Content Avoid\n")
            for a in avoid:
                lines.append(f"- {a}")
            lines.append("")

    dms = brief.get("dm_targets") or []
    if dms:
        lines.append("## DM Targets\n")
        for d in dms:
            name = d.get("name", "")
            company_ = d.get("company", "unknown")
            icp = d.get("icp_score", "")
            eng = d.get("posts_engaged", "")
            headline = d.get("headline", "")
            angle = d.get("suggested_angle", "")
            lines.append(f"### {name} — {company_} (ICP {icp} · {eng} engagements)")
            if headline:
                lines.append(f"_{headline}_\n")
            lines.append(angle)
            lines.append("")

    abms = brief.get("abm_targets") or []
    if abms:
        lines.append("## ABM Targets\n")
        for a in abms:
            lines.append(f"### {a.get('company','')} — {a.get('name','')}")
            lines.append(a.get("rationale", ""))
            lines.append("")

    assets = brief.get("asset_requests") or []
    if assets:
        lines.append("## Asset Requests\n")
        for r in assets:
            lines.append(f"- {r}")
        lines.append("")

    trig = brief.get("next_run_trigger") or {}
    if trig:
        lines.append("## Next Run Trigger\n")
        if trig.get("condition"):
            lines.append(f"**Condition:** {trig['condition']}\n")
        if trig.get("or_after_days") is not None:
            lines.append(f"**Max days:** {trig['or_after_days']}\n")
        if trig.get("rationale"):
            lines.append(f"**Rationale:** {trig['rationale']}\n")

    return "\n".join(lines)


@router.get("/check/{company}")
async def check_briefing(company: str):
    """Check whether a Cyrene brief exists for a company."""
    return {"exists": _load_cyrene_brief_from_supabase(company) is not None}


@router.get("/content/{company}")
async def get_briefing_content(company: str):
    """Return the Cyrene brief as a markdown string for interview prep."""
    brief = _load_cyrene_brief_from_supabase(company)
    if not brief:
        raise HTTPException(status_code=404, detail="No Cyrene brief found. Run Cyrene first.")
    if isinstance(brief, dict):
        return {"content": _brief_to_markdown(brief)}
    return {"content": str(brief)}
