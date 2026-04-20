"""HTTP-backed filesystem for Stelle when running in Lineage mode.

When the stelle_runner is spawned with ``--lineage-workspace-url`` and
``--lineage-run-token`` (forwarded by the /api/ghostwriter/generate
handler), it sets two env vars:

    LINEAGE_WORKSPACE_URL = https://<virio-api>/api/workspace
    LINEAGE_RUN_TOKEN     = <HS256 JWT bound to {companyId, jobId}>

This module then acts as a drop-in replacement for the four local-disk
tool handlers in ``stelle.py`` (``list_directory``, ``read_file``,
``write_file``, ``edit_file``). Every call issues an HTTPS request against
virio-api's workspace surface instead of touching the local filesystem.

Contract on the virio-api side:

    GET  /api/workspace/list?path=P   → {"entries":[{name,type,path}]}
    GET  /api/workspace/file?path=P   → {"content":"..."}  | 404
    POST /api/workspace/write         → {"ok":true,"path":"..."}
                                        body: {"path": P, "content": "..."}

A ``403`` on write means the path is read-only (transcripts, research,
engagement etc.). Write drafts under ``{user-slug}/posts/...`` — those
map to ``drafts`` table rows via virio-api's ``DraftsFs``.

The run token expires (2h default). Errors surface back as tool-result
strings so Stelle's agent loop can react.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("stelle.workspace_fs")

# Max characters to return in a single read_file call (matches stelle.py)
MAX_TOOL_OUTPUT_CHARS = 80_000


class LineageIngestionError(RuntimeError):
    """Raised when a read against the client data source fails and we
    cannot recover. Deliberately NOT caught by Stelle's tool-dispatch
    try/except — the run aborts instead of proceeding with missing data.

    Only fires when ``is_lineage_mode()`` is True (Supabase creds +
    LINEAGE_COMPANY_ID present) AND the read then fails. Runs without a
    configured data source never raise this.
    """


def is_lineage_mode() -> bool:
    """True when Stelle has enough config to READ client data from
    Jacquard's Supabase + GCS.

    Required:
      - LINEAGE_COMPANY_ID — the Jacquard user_companies.id to query
      - SUPABASE_URL + SUPABASE_KEY — shared Jacquard project creds
      - GCS_CREDENTIALS_B64 — service-account JSON for transcript blobs

    The name is historical — it predates the Lineage-UI integration
    deprecation. Conceptually today this is ``has_client_data_source()``.
    """
    if not os.environ.get("LINEAGE_COMPANY_ID", "").strip():
        return False
    return bool(
        os.environ.get("GCS_CREDENTIALS_B64", "").strip()
        and os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_KEY", "").strip()
    )


def _normalize_path(rel: str) -> str:
    """Strip leading slashes and ``..`` segments before sending to server.

    virio-api's workspace router does its own scoping via the JWT's
    companyId claim, so a caller can't escape the workspace even with
    a malicious path. This is just cosmetic normalization.
    """
    cleaned = rel.lstrip("/").lstrip("\\")
    parts = [p for p in cleaned.split("/") if p and p != "."]
    safe = [p for p in parts if p != ".."]
    return "/".join(safe)


# ---------------------------------------------------------------------------
# Path rewriting — translate Stelle's bare paths into Lineage user-scoped
# paths so Stelle can keep using ``transcripts/foo.txt`` etc. without
# knowing which FOC-user directory they live under.
# ---------------------------------------------------------------------------

# Paths that do NOT get user-slug prefixing — they're shared across all
# users in the Lineage workspace root.
_SHARED_ROOTS = frozenset({".pi", "conversations", "slack", "tasks"})

# User-scoped mount directories that Lineage's workspace-builder
# constructs. Any path whose FIRST segment is a shared-root, OR whose
# second segment (under a user-slug) is one of these, is a Lineage path
# and should be routed through the HTTP proxy. Everything else — ``scratch/``,
# ``notes/``, ``plan.md``, ad-hoc files Stelle invents — is private
# scratch space and stays on the fly-local SandboxFs.
#
# Source of truth: ``jacquard/virio-api/src/services/ghostwriter/workspace-builder.ts``
# (fields of the ``readOnly`` mount dict + the DraftsFs/StrategyFs/NotesFs
# wiring). If Jacquard adds a new mount, add it here.
_USER_MOUNTS = frozenset({
    "transcripts",
    "research",
    "engagement",
    "reports",
    "context",
    "posts",      # includes posts/published and posts/drafts
    "edits",
    "tone",
    "strategy",
    # NB: ``notes`` is NOT mounted as a user subdir today, but Stelle's
    # older prompts reference ``notes/plan.md``. Keep it OUT of this
    # set so those writes land fly-local (which is what we want for
    # scratch).
})


# Cache the list of FOC user slugs per company, populated once per
# process. Filled lazily on the first ``is_lineage_path`` call that
# needs to check a bare-slug path. Avoids a Supabase round-trip on
# every path-routing decision.
_FOC_SLUGS_CACHE: dict[str, frozenset[str]] = {}


def _known_foc_slugs() -> frozenset[str]:
    """Return the set of FOC user slugs for the current company.

    Used by ``is_lineage_path`` to recognize bare-slug paths like
    ``weber-wong`` as Lineage reads (they map to ``_direct_list(slug)``
    which returns the user's subdir listing). Without this, Stelle
    spends cycles bouncing off "directory not found" when she tries to
    drill into a slug she just saw at the workspace root.

    Cached per-process so the lookup is free after the first hit. If
    Supabase is unreachable, returns an empty set (callers fall through
    to the safe denial path).
    """
    company_id = os.environ.get("LINEAGE_COMPANY_ID", "").strip()
    if not company_id:
        return frozenset()
    cached = _FOC_SLUGS_CACHE.get(company_id)
    if cached is not None:
        return cached
    try:
        from backend.src.agents import jacquard_direct as _jd
        users = _jd.list_foc_users(company_id) or []
        slugs = frozenset(u.get("slug") for u in users if u.get("slug"))
    except Exception as exc:
        logger.warning("[lineage_fs] FOC slug cache miss: %s", exc)
        slugs = frozenset()
    _FOC_SLUGS_CACHE[company_id] = slugs
    return slugs


def is_lineage_path(rel: str) -> bool:
    """True if the path targets a Lineage mount, False if it's scratch.

    The routing decision is based on the workspace layout declared in
    ``workspace-builder.ts``. A path is "Lineage" when:
      * First segment is a shared root (``conversations/``, ``slack/``,
        ``tasks/``, ``.pi/``), OR
      * First segment is a known FOC user slug (bare slug path —
        ``weber-wong``, ``weber-wong/``), OR
      * First segment is a user slug AND second segment is a known
        user-scope mount (``transcripts/``, ``research/``, etc.).

    Everything else is scratch — writes go to the fly-local SandboxFs,
    reads try fly-local too. This preserves Stelle's classic pattern of
    iterating drafts as files while keeping Lineage's workspace
    read-only from her perspective.
    """
    norm = _normalize_path(rel)
    if not norm:
        return False
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return False
    first = parts[0]
    if first in _SHARED_ROOTS:
        return True
    slug = _targeted_slug()
    # In user-targeted mode the slug is implicit; user-scope paths may
    # arrive without the slug prefix.
    if slug and first in _USER_MOUNTS:
        return True
    # User-slug-prefixed path (either target slug in user-targeted mode
    # or any slug in company-wide mode). Second segment must be a known
    # user mount.
    if len(parts) >= 2 and parts[1] in _USER_MOUNTS:
        return True
    # Known FOC slug prefix — route anything under it (bare slug,
    # synthesized user-root files like ``post-history.md`` / ``profile.md``,
    # and any other slug-scoped path) to Lineage. Without this branch the
    # allowlist only caught ``<slug>/<known-mount>/...`` and slipped the
    # two synthesized files through to fly-local, where they 404.
    if first in _known_foc_slugs():
        return True
    return False


def is_user_targeted() -> bool:
    """True when the run is targeted at a specific FOC user.

    In this mode every filesystem op is prefixed with the user's slug and
    every draft auto-write attributes to that user. False = company-wide
    mode (the agent picks per draft via explicit ``write_file`` calls).
    """
    return bool(os.environ.get("LINEAGE_USER_SLUG"))


def _targeted_slug() -> str | None:
    """Read the target FOC-user slug from env. None in company-wide mode."""
    slug = os.environ.get("LINEAGE_USER_SLUG")
    return slug if slug else None


def _get_primary_slug() -> str | None:
    """Back-compat helper — returns the target slug when in user-targeted mode."""
    return _targeted_slug()


def _rewrite_path(rel: str) -> str:
    """Turn a Stelle-style path into the Lineage workspace path.

    - In **company-wide mode** (no ``LINEAGE_USER_SLUG``) paths pass through
      unchanged; Stelle sees the full multi-user workspace and is expected
      to call ``write_file("<user-slug>/posts/drafts/<uuid>/content.md", ...)``
      explicitly for each draft.
    - In **user-targeted mode** the path is prefixed with the target user's
      slug (shared roots ``.pi``, ``tasks``, ``slack``, ``conversations``
      remain unprefixed; already-prefixed paths are not double-prefixed).
    """
    norm = _normalize_path(rel)
    if not norm:
        return ""

    slug = _targeted_slug()
    if slug is None:
        # Company-wide mode: Stelle has the raw workspace. Don't rewrite.
        return norm

    first = norm.split("/", 1)[0]
    if first in _SHARED_ROOTS:
        return norm
    if first == slug:
        return norm
    return f"{slug}/{norm}"


# ---------------------------------------------------------------------------
# Direct mode — route reads through jacquard_direct (Supabase + GCS) rather
# than Jacquard's virio-api HTTP surface. Used when GCS_CREDENTIALS_B64 +
# SUPABASE_URL/KEY are configured. Gives us full independence from
# Jacquard's runtime: their API can be down and Stelle still reads fine.
# ---------------------------------------------------------------------------


def _direct_enabled() -> bool:
    """True when the direct Supabase+GCS pipeline is both configured AND
    we have a company to query against (LINEAGE_COMPANY_ID present)."""
    try:
        from backend.src.agents import jacquard_direct as _jd
    except Exception:
        return False
    return bool(
        _jd.is_direct_configured()
        and os.environ.get("LINEAGE_COMPANY_ID", "").strip()
    )


def _direct_list(path: str) -> str | None:
    """Direct-mode listing. Returns the formatted listing string on success,
    None when the path isn't one this backend knows (caller falls through
    to HTTP). Path layout mirrors Jacquard's workspace-builder mounts.
    """
    from backend.src.agents import jacquard_direct as _jd
    company_id = os.environ["LINEAGE_COMPANY_ID"].strip()

    norm = _normalize_path(path)

    # Root or user-slug-only — list FOC users / their subdirs.
    if norm == "":
        users = _jd.list_foc_users(company_id)
        if not users:
            return "(empty directory)"
        return "\n".join(f"  {u['slug']}/" for u in users)

    parts = norm.split("/")
    # Shared roots first.
    if parts[0] == "conversations" and len(parts) == 1:
        return "  trigger-log.jsonl"
    if parts[0] == "tasks" and len(parts) == 1:
        tasks = _jd.fetch_tasks(company_id)
        if not tasks:
            return "(empty directory)"
        return "\n".join(f"  {t.get('id')}.json" for t in tasks)

    # User-scoped paths.
    slug = parts[0]
    user = _jd.resolve_user_by_slug(company_id, slug)
    if not user:
        return None  # unknown slug → let HTTP try

    # <slug>/  → subdir listing + the two synthesized user-root files
    # (post-history.md + profile.md are computed on-demand from the
    # linkedin_posts / linkedin_profiles tables — they don't live as
    # rows anywhere, we render them at read time).
    if len(parts) == 1:
        # NB: ``notes/`` is intentionally NOT advertised — ``is_lineage_path``
        # routes it to fly-local scratch (Stelle's scratch notepad) rather
        # than Lineage's Supabase. Listing it here would send Stelle
        # probing a path that errors with "directory not found".
        subdirs = [
            "transcripts/", "research/", "engagement/", "context/",
            "reports/", "posts/", "edits/", "tone/", "strategy/",
        ]
        lines = [f"  {d}" for d in subdirs]
        lines.append("  post-history.md")
        lines.append("  profile.md")
        return "\n".join(lines)

    sub = parts[1]
    if sub == "transcripts" and len(parts) == 2:
        items = _jd.fetch_meeting_transcripts(
            user["id"], user.get("email"), user.get("is_internal", False)
        )
        return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
    if sub == "research" and len(parts) == 2:
        items = _jd.fetch_parallel_research(company_id, user["id"])
        return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
    if sub == "context" and len(parts) == 2:
        items = _jd.fetch_context_files(company_id, user)
        return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
    if sub == "engagement" and len(parts) == 2:
        files = _jd.fetch_icp_data(user)
        if not files:
            return "(empty directory)"
        return "\n".join(f"  {fn}" for fn in files.keys())
    if sub == "reports" and len(parts) == 2:
        report = _jd.fetch_latest_icp_report(company_id, user["id"])
        return f"  {report['filename']}" if report else "(empty directory)"
    if sub == "tone" and len(parts) == 2:
        items = _jd.fetch_tone_references(user["id"])
        return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
    if sub == "edits" and len(parts) == 2:
        items = _jd.fetch_edit_history(user["id"], limit=50)
        return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
    if sub == "posts":
        if len(parts) == 2:
            return "  published/\n  drafts/"
        if len(parts) == 3 and parts[2] == "published":
            items = _jd.fetch_published_posts(user["id"])
            return "\n".join(f"  {i['filename']}" for i in items) or "(empty directory)"
        if len(parts) == 3 and parts[2] == "drafts":
            # Amphoreus keeps in-flight drafts local; no pre-published drafts
            # live in Jacquard's drafts/ for the agent to inspect.
            return "(empty directory)"

    # notes/ and strategy/ are writable — empty from the direct read's POV
    if sub in ("notes", "strategy") and len(parts) == 2:
        return "(empty directory)"

    return None  # let HTTP take over for anything unknown


def _direct_read(path: str) -> str | None:
    """Direct-mode read. Returns content on success; None when the path
    isn't one this backend handles (caller falls through to HTTP)."""
    from backend.src.agents import jacquard_direct as _jd
    company_id = os.environ["LINEAGE_COMPANY_ID"].strip()

    norm = _normalize_path(path)
    if not norm:
        return None

    parts = norm.split("/")

    # Shared roots
    if parts == ["conversations", "trigger-log.jsonl"]:
        return _jd.fetch_trigger_log(company_id)
    if len(parts) == 2 and parts[0] == "tasks" and parts[1].endswith(".json"):
        task_id = parts[1].removesuffix(".json")
        tasks = _jd.fetch_tasks(company_id)
        match = next((t for t in tasks if str(t.get("id")) == task_id), None)
        if match is not None:
            import json as _json
            return _json.dumps(match, default=str, indent=2)
        return None

    # User-root synthesized files: <slug>/post-history.md and <slug>/profile.md.
    # Not stored anywhere — computed on-demand by rendering the top-reacted
    # linkedin_posts for post-history.md, and the linkedin_profiles row for
    # profile.md. Written to disk by populate_lineage_workspace (CLI path)
    # and served here for the direct-API path so both paths see the same
    # workspace.
    if len(parts) == 2 and parts[1] in ("post-history.md", "profile.md"):
        user = _jd.resolve_user_by_slug(company_id, parts[0])
        if not user:
            return None
        if parts[1] == "post-history.md":
            return _jd.build_post_history_digest(user, top_n=10)
        else:
            return _jd.fetch_profile_md(user)

    if len(parts) < 3:
        return None  # not a file path

    slug, sub = parts[0], parts[1]
    user = _jd.resolve_user_by_slug(company_id, slug)
    if not user:
        return None

    filename = "/".join(parts[2:])

    if sub == "transcripts":
        items = _jd.fetch_meeting_transcripts(
            user["id"], user.get("email"), user.get("is_internal", False)
        )
        match = next((i for i in items if i["filename"] == filename), None)
        return match["content"] if match else None

    if sub == "research":
        items = _jd.fetch_parallel_research(company_id, user["id"])
        match = next((i for i in items if i["filename"] == filename), None)
        return match["content"] if match else None

    if sub == "context":
        items = _jd.fetch_context_files(company_id, user)
        match = next((i for i in items if i["filename"] == filename), None)
        return match["content"] if match else None

    if sub == "engagement":
        files = _jd.fetch_icp_data(user)
        return files.get(filename)

    if sub == "reports":
        report = _jd.fetch_latest_icp_report(company_id, user["id"])
        if report and report["filename"] == filename:
            return report["content"]
        return None

    if sub == "tone":
        items = _jd.fetch_tone_references(user["id"])
        match = next((i for i in items if i["filename"] == filename), None)
        return match["content"] if match else None

    if sub == "edits":
        items = _jd.fetch_edit_history(user["id"])
        match = next((i for i in items if i["filename"] == filename), None)
        return match["content"] if match else None

    if sub == "posts" and len(parts) >= 4 and parts[2] == "published":
        items = _jd.fetch_published_posts(user["id"])
        # filename may contain slashes for posts/published/<file>
        target = "/".join(parts[3:])
        match = next((i for i in items if i["filename"] == target), None)
        return match["content"] if match else None

    return None


# ---------------------------------------------------------------------------
# Tool handlers — same signatures as the local ones in stelle.py so they
# can be swapped in at the _TOOL_HANDLERS dispatch dict.
# ---------------------------------------------------------------------------


def exec_list_directory(_workspace_root: Any, args: dict) -> str:
    raw = args.get("path", "") or ""
    # Listing the root ("") must not prefix — Stelle uses this to discover
    # what's available. Everything else is user-slug rewritten.
    rel = "" if _normalize_path(raw) == "" else _rewrite_path(raw)

    # Direct-mode path — Supabase + GCS, no HTTP hop. When direct is
    # configured (the default), this is the ONLY path; HTTP to virio-api
    # is used only when direct isn't configured (legacy/Jacquard-initiated
    # proxy runs).
    if _direct_enabled():
        try:
            direct = _direct_list(rel)
        except Exception as exc:
            raise LineageIngestionError(
                f"list_directory({rel!r}): direct read failed: {exc}"
            ) from exc
        if direct is None:
            # Path isn't one our direct backend knows — this isn't fatal.
            # Stelle may be probing for a path from her local-mode prompt
            # (``context/research``, ``memory/foo``, etc.) that doesn't
            # exist in Jacquard's workspace. Return an empty listing so
            # she learns "nothing there" and moves on, instead of us
            # killing the run with LineageIngestionError.
            logger.info(
                "[lineage_fs] direct list: %r not recognized — returning empty",
                rel,
            )
            return (
                "(empty directory — this path isn't part of Lineage's workspace. "
                "Valid paths: <user-slug>/{transcripts,research,engagement,"
                "context,reports,posts,edits,tone,strategy}, or shared "
                "{conversations,slack,tasks,.pi}.)"
            )
        return direct

    # HTTP-mode fallback was removed with the Lineage-UI deprecation.
    # Direct reads are now the only path; a _direct_list miss (None) is
    # surfaced above as an empty hint. See _lineage_deprecated/http_client
    # for the removed code.
    raise LineageIngestionError(
        f"list_directory({rel!r}): direct-mode reader not configured "
        "(GCS_CREDENTIALS_B64 + SUPABASE_URL + SUPABASE_KEY required)"
    )


def exec_read_file(_workspace_root: Any, args: dict) -> str:
    rel = args.get("path", "") or ""
    norm = _rewrite_path(rel)
    if not norm:
        return "Error: path is required"

    if _direct_enabled():
        try:
            direct = _direct_read(norm)
        except Exception as exc:
            raise LineageIngestionError(
                f"read_file({rel!r}): direct read failed: {exc}"
            ) from exc
        if direct is None:
            return f"Error: file not found: {rel}"
        content = direct
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
        return content

    raise LineageIngestionError(
        f"read_file({rel!r}): direct-mode reader not configured"
    )


def exec_write_file(_workspace_root: Any, args: dict) -> str:
    """Writes against the data-source workspace are always refused —
    the workspace is read-only (it's the client's Supabase data).
    Scratch writes are dispatched to the fly-local sandbox by the
    caller; this function only runs for paths that resolved to a
    client-data mount.
    """
    rel = args.get("path", "") or ""
    return f"Error: {rel} is read-only (client data source, not scratch)"


def exec_edit_file(_workspace_root: Any, args: dict) -> str:
    rel = args.get("path", "") or ""
    return f"Error: {rel} is read-only (client data source, not scratch)"


def exec_search_files(_workspace_root: Any, args: dict) -> str:
    """Grep-style recursive search — not implemented in direct mode.

    The direct-mode reader materializes files on demand from Supabase
    queries; there's no pre-built index to grep over. Callers should
    list the target directory and read specific files instead.
    """
    return (
        "Error: search_files is not supported in direct-mode reads. "
        "Use list_directory + read_file to explore the workspace instead."
    )


def exec_query_observations(_workspace_root: Any, args: dict) -> str:
    """Direct-mode redirect — query_observations is not wired in
    standalone Amphoreus. Point the caller at the engagement JSON
    files that carry the same data.
    """
    return (
        "Error: query_observations is not available in direct-only mode.\n\n"
        "Use these file paths instead — same data, different packaging:\n"
        "  • `<slug>/post-history.md` — top 10 performers with full text "
        "and engagement metrics. Your baseline.\n"
        "  • `<slug>/engagement/posts.json` — every scored post with raw "
        "engagement numbers + per-reaction breakdown. Filter / summarize "
        "in-context however you need.\n"
        "  • `<slug>/engagement/reactions.json`, `comments.json`, "
        "`profiles.json` — engager-level detail if you need it."
    )


def exec_mention_resolve(_workspace_root: Any, args: dict) -> str:
    """Mention-resolve is not available in direct mode (it used to proxy
    through virio-api's APImaestro-backed resolver; see
    _lineage_deprecated/http_client.py if reconnecting)."""
    return "Error: mention_resolve is not available in direct-only mode"


# ---------------------------------------------------------------------------
# Direct helpers used by the runner's post-write path (not tool handlers)
# ---------------------------------------------------------------------------


# Directories at the workspace root that aren't FOC-user mounts. Used
# by ``resolve_primary_user_slug`` to skip past system mounts.
_SYSTEM_DIRS = frozenset({".pi", "conversations", "slack", "tasks", ".keep"})


def resolve_primary_user_slug() -> str | None:
    """Discover a FOC-user directory in the workspace.

    Supabase-fs mounts one directory per user-with-``posts_content=true``
    under the workspace root. For an MVP Stelle run, we pick the first
    non-system directory — later we can let the /generate request specify
    which user the drafts should be attributed to.
    """
    raw = exec_list_directory(None, {"path": ""})
    if raw.startswith("Error:"):
        return None

    # ``exec_list_directory`` returns human-formatted text like:
    #   "  andrew-cai/\n  emmett-chen-ran/\n  ..."
    for line in raw.splitlines():
        name = line.strip().rstrip("/")
        if not name:
            continue
        if name in _SYSTEM_DIRS:
            continue
        return name
    return None


def write_draft_for_current_run(draft_id: str, content: str) -> tuple[bool, str]:
    """Deprecated — the HTTP-mode DraftsFs write path is gone.

    Used to write ``{slug}/posts/drafts/{draft_id}/content.md`` which
    virio-api's DraftsFs mapped to a Jacquard ``drafts`` row. See
    ``backend/src/_lineage_deprecated/drafts_writer.insert_draft`` for
    a direct-Supabase equivalent if reconnecting.
    """
    return False, "deprecated: HTTP-mode drafts write is gone"