"""Briefings API — reads Cyrene's strategic brief for interview prep."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/briefings", tags=["briefings"])


def _cyrene_brief_path(company: str) -> Path:
    from backend.src.db import vortex
    return vortex.memory_dir(company) / "cyrene_brief.json"


def _brief_to_markdown(brief: dict) -> str:
    """Render a Cyrene brief dict as markdown for ReactMarkdown consumption."""
    lines: list[str] = []
    company = brief.get("_company", "")
    computed = brief.get("_computed_at", "")
    lines.append(f"# Cyrene Brief — {company}")
    if computed:
        lines.append(f"\n_Computed: {computed}_\n")

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
    return {"exists": _cyrene_brief_path(company).exists()}


@router.get("/content/{company}")
async def get_briefing_content(company: str):
    """Return the Cyrene brief as a markdown string for interview prep."""
    path = _cyrene_brief_path(company)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No Cyrene brief found. Run Cyrene first.")
    brief = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(brief, dict):
        return {"content": _brief_to_markdown(brief)}
    # Already a string (legacy) — return as-is
    return {"content": str(brief)}
