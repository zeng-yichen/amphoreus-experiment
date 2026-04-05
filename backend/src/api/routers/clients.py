"""Clients router — canonical list of Ordinal-backed client slugs."""

from fastapi import APIRouter

from backend.src.db import vortex

router = APIRouter(prefix="/api/clients", tags=["clients"])


@router.get("")
async def list_clients():
    """Return deduplicated provider_org_slug values from ordinal_auth_rows.csv."""
    rows = vortex.list_ordinal_companies()
    seen: set[str] = set()
    clients: list[dict] = []
    for row in rows:
        slug = (row.get("provider_org_slug") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            clients.append({"slug": slug})
    clients.sort(key=lambda c: c["slug"])
    return {"clients": clients}
