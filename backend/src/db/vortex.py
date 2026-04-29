"""Vortex — Centralised path layout for Amphoreus.

    memory/{company}/          — persistent client knowledge
        transcripts/           — raw interview transcripts & documents
        accepted/              — approved posts (style exemplars)
        feedback/              — client feedback & Ordinal comments
        revisions/             — before/after revision pairs
        abm_profiles/          — ABM target briefings
        targets/               — xlsx / data files for ABM target sourcing
        references/            — client-provided URLs, articles, and reference material
        past_posts/            — post history for redundancy checking
        content_strategy/      — content strategy docs
        tmp/                   — ephemeral fetched data

    memory/our_memory/         — shared knowledge (LinkedIn writing guidelines, etc.)

    products/{company}/        — generated artefacts
        post/                  — post markdown files
        brief/                 — briefing markdown files
        images/                — assembled post images
"""

import csv
import logging
import pathlib

_vortex_logger = logging.getLogger(__name__)

try:
    from backend.src.core.config import get_settings
    _settings = get_settings()
    _project_root = pathlib.Path(_settings.data_dir).parent
except Exception:
    _settings = None
    _project_root = pathlib.Path(__file__).resolve().parents[3]

# 2026-04-29: ``MEMORY_ROOT`` points to ephemeral tmpfs.
#
# Pre-2026-04-29, this resolved to ``/data/memory/`` (a Fly volume mount)
# and accumulated dual-write artifacts: cyrene_brief.json,
# depth_weights.json, reward_weights.json, draft_map.json,
# story_inventory.md, feedback/, adaptive_config.json, etc. Most writers
# had a Supabase counterpart written elsewhere; the local mirror was
# either legacy dual-write or regenerable cache. The fly-local volume
# became a debugging hazard — operators would purge a Supabase brief
# but the local file would still get re-read by the next agent run,
# leaking stale state.
#
# v2 redirects MEMORY_ROOT to ``/tmp/_amphoreus_legacy_memory/``. Effects:
#   - Writers continue to work (no surgery in 14 callers needed).
#   - State doesn't persist across container restarts.
#   - The Fly /data/memory volume can be wiped and won't be repopulated.
#   - Source-of-truth state lives exclusively in:
#       * Amphoreus Supabase (cyrene_briefs, local_posts, ruan_mei_state, ...)
#       * SQLite (/data/sqlite/amphoreus.db)
#
# Readers that fall back to MEMORY_ROOT files (linkedin_username,
# ordinal_auth_csv, ICP definition, etc.) get None / empty and fall
# through to defaults — same degradation that would happen on a
# fresh container, by design.
import tempfile
_LEGACY_MEMORY_BUCKET = pathlib.Path(tempfile.gettempdir()) / "_amphoreus_legacy_memory"
try:
    _LEGACY_MEMORY_BUCKET.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
MEMORY_ROOT = _LEGACY_MEMORY_BUCKET
PRODUCTS_ROOT = _project_root / "products"


def memory_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company


def transcripts_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "transcripts"


def accepted_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "accepted"


def feedback_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "feedback"


def revisions_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "revisions"


def abm_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "abm_profiles"


def past_posts_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "past_posts"


def content_strategy_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "content_strategy"


def targets_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "targets"


def references_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "references"


def notes_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "notes"


def tmp_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "tmp"


def post_dir(company: str) -> pathlib.Path:
    return PRODUCTS_ROOT / company / "post"


def castorice_annotated_path(company: str) -> pathlib.Path:
    """Annotated post produced by Castorice's source-annotation pass. Stored locally for CE review."""
    return PRODUCTS_ROOT / company / "post" / "castorice_annotated.md"


def brief_dir(company: str) -> pathlib.Path:
    return PRODUCTS_ROOT / company / "brief"


def images_dir(company: str) -> pathlib.Path:
    return PRODUCTS_ROOT / company / "images"


def ordinal_auth_csv() -> pathlib.Path:
    return MEMORY_ROOT / "ordinal_auth_rows.csv"


def our_memory_dir() -> pathlib.Path:
    return MEMORY_ROOT / "our_memory"


def linkedin_username_path(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "linkedin_username.txt"


def story_inventory_path(company: str) -> pathlib.Path:
    """Persistent cross-session log of stories told/untold. Symlinked into workspace."""
    return MEMORY_ROOT / company / "story_inventory.md"


def draft_map_path(company: str) -> pathlib.Path:
    """JSON map of {post_id -> {original_text, title, generated_at}} for feedback diffing."""
    return MEMORY_ROOT / company / "draft_map.json"


def image_feedback_log_path(company: str) -> pathlib.Path:
    """Append-only log of image revision requests for Phainon (bitter-lesson feedback loop)."""
    return MEMORY_ROOT / company / "image_feedback.jsonl"


def icp_definition_path(company: str) -> pathlib.Path:
    """Per-client ICP definition used to classify post engagers."""
    return MEMORY_ROOT / company / "icp_definition.json"


def workspace_dir(client_slug: str) -> pathlib.Path:
    data_dir = pathlib.Path(_settings.data_dir) if _settings else _project_root / "data"
    return data_dir / "workspaces" / client_slug


def snapshots_dir(client_slug: str) -> pathlib.Path:
    return workspace_dir(client_slug) / "snapshots"


def ensure_dirs(company: str) -> None:
    """Create the full directory tree for a client."""
    for d in (
        transcripts_dir(company),
        accepted_dir(company),
        feedback_dir(company),
        revisions_dir(company),
        abm_dir(company),
        targets_dir(company),
        references_dir(company),
        past_posts_dir(company),
        content_strategy_dir(company),
        tmp_dir(company),
        post_dir(company),
        brief_dir(company),
        images_dir(company),
    ):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Ordinal auth CSV helpers
# ------------------------------------------------------------------

_ORDINAL_BASE = "https://app.tryordinal.com/api/v1"


def list_ordinal_companies() -> list[dict]:
    """Return all rows from ordinal_auth_rows.csv as dicts.

    Each dict has keys: company_id, api_key, provider_org_slug, profile_id.
    """
    path = ordinal_auth_csv()
    if not path.exists():
        return []
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def resolve_profile_id(company: str) -> str:
    """Return the LinkedIn scheduling profile UUID for *company*.

    Reads ordinal_auth_rows.csv, finds the matching row by provider_org_slug.
    If profile_id is already set, returns it immediately.  Otherwise calls
    ``GET /profiles/scheduling`` to auto-resolve, writes the value back into
    the CSV, and returns it.

    Returns empty string when resolution fails.
    """
    path = ordinal_auth_csv()
    if not path.exists():
        return ""

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception:
        return ""

    # 1) Direct slug match (legacy path — CSV row's provider_org_slug
    #    equals the incoming identifier verbatim).
    target_row = None
    for row in rows:
        slug = (row.get("provider_org_slug") or "").strip()
        if slug == company:
            target_row = row
            break

    # 2) Fall back to Jacquard UUID resolution so the new mirror-backed
    #    dropdown slugs (e.g. ``hume-ai-andrew``, which isn't a CSV row)
    #    route to the CSV row sharing the same ``company_id``. All FOCs
    #    under the same company therefore share one Ordinal workspace.
    if target_row is None:
        try:
            from backend.src.lib.company_resolver import resolve_to_uuid
            resolved = resolve_to_uuid(company)
        except Exception:
            resolved = None
        if resolved:
            ruid = resolved.lower()
            for row in rows:
                if (row.get("company_id") or "").strip().lower() == ruid:
                    target_row = row
                    break

    if target_row is None:
        return ""

    existing = (target_row.get("profile_id") or "").strip()
    if existing:
        return existing

    api_key = (target_row.get("api_key") or "").strip()
    if not api_key:
        return ""

    try:
        import httpx

        resp = httpx.get(
            f"{_ORDINAL_BASE}/profiles/scheduling",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        profiles = resp.json()
    except Exception as e:
        _vortex_logger.warning("[vortex] profile_id resolution failed for %s: %s", company, e)
        return ""

    linkedin = [p for p in profiles if p.get("channel") == "LinkedIn"]
    if len(linkedin) != 1:
        _vortex_logger.warning(
            "[vortex] Expected 1 LinkedIn profile for %s, got %d — skipping auto-fill",
            company, len(linkedin),
        )
        return ""

    pid = str(linkedin[0]["id"]).strip()

    # Write back into CSV so future lookups are instant.
    target_row["profile_id"] = pid
    try:
        with open(path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        _vortex_logger.info("[vortex] Auto-filled profile_id for %s: %s", company, pid)
    except Exception as e:
        _vortex_logger.warning("[vortex] Could not write profile_id back to CSV: %s", e)

    return pid
