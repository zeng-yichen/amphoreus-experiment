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

import base64
import json
import logging
import os
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger("stelle.lineage_fs")

# Max characters to return in a single read_file call (matches stelle.py)
MAX_TOOL_OUTPUT_CHARS = 80_000


class LineageIngestionError(RuntimeError):
    """Raised when a read against Jacquard's Lineage data fails and we
    cannot recover. Deliberately NOT caught by Stelle's tool-dispatch
    try/except — the run aborts instead of proceeding with missing data.

    Produced when either:
    - direct mode was enabled but both direct AND HTTP fallback failed
    - lineage mode was enabled but the HTTP call errored / 5xx'd / 401'd
    - a read returned a content payload shaped like an error

    Pure-local runs (``is_lineage_mode() == False``) never raise this —
    they use local disk paths where a missing file is just "empty dir"
    or "file not found", not a fatal ingestion failure.
    """

# Shared HTTP client — connection pooling across many tool calls during a run.
# 60s per-request timeout matches Stelle's bash timeout.
_client: httpx.Client | None = None

# Refresh the run token when it has this many seconds (or fewer) left.
# Matches Jacquard's ``REFRESH_GRACE_SECONDS`` of 30 min — we refresh
# well before the grace window to avoid racing with a real expiry.
_REFRESH_WINDOW_SECONDS = 10 * 60

# Serialize concurrent refresh attempts from the same process.
_refresh_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60.0)
    return _client


def is_lineage_mode() -> bool:
    """True when Stelle has enough config to read Jacquard data.

    Required:
      - LINEAGE_COMPANY_ID (which company to query)
      - Either direct-mode (GCS + Supabase) OR HTTP-mode (workspace URL + run token)

    Direct-only deployments (our current default) skip the workspace URL
    and run token — no virio-api dependency, no JWT minting. Amphoreus
    reads from Jacquard's Supabase + GCS using its own service credentials.
    """
    if not os.environ.get("LINEAGE_COMPANY_ID", "").strip():
        return False
    # Direct mode: Supabase + GCS credentials in env.
    direct_ok = bool(
        os.environ.get("GCS_CREDENTIALS_B64", "").strip()
        and os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_KEY", "").strip()
    )
    # HTTP mode (legacy/fallback): workspace URL + run token.
    http_ok = bool(
        os.environ.get("LINEAGE_WORKSPACE_URL", "").strip()
        and os.environ.get("LINEAGE_RUN_TOKEN", "").strip()
    )
    return direct_ok or http_ok


def _base_url() -> str:
    url = os.environ["LINEAGE_WORKSPACE_URL"].rstrip("/")
    return url


def _decode_jwt_exp(token: str) -> int | None:
    """Return the ``exp`` (unix seconds) claim from a JWT without verifying.

    We don't need to verify — we only read ``exp`` to decide whether to
    pre-emptively refresh. Verification happens on the server side.
    Returns None for any malformed token (caller then falls through to
    the old behavior of letting the server reject an expired token).
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Pad to a multiple of 4 for urlsafe_b64decode.
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
        exp = data.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
        return None
    except Exception:
        return None


def _refresh_run_token() -> bool:
    """Ask Jacquard for a fresh run token using our current (possibly
    near-expired) one. Updates ``LINEAGE_RUN_TOKEN`` in-place on success.

    Returns True on success. On failure logs a warning and returns False;
    the caller then proceeds with the old token — the server will 401 if
    it's actually expired.
    """
    with _refresh_lock:
        # Re-check inside the lock — if another thread already refreshed,
        # we don't want to burn a second round-trip.
        current = os.environ.get("LINEAGE_RUN_TOKEN", "")
        exp = _decode_jwt_exp(current)
        if exp is not None and exp - int(time.time()) > _REFRESH_WINDOW_SECONDS:
            return True  # Already refreshed by another thread.

        try:
            resp = _get_client().post(
                f"{_base_url()}/refresh-token",
                headers={
                    "Authorization": f"Bearer {current}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            logger.warning("[lineage_fs] refresh-token network error: %s", exc)
            return False
        if resp.status_code >= 300:
            logger.warning(
                "[lineage_fs] refresh-token %s: %s",
                resp.status_code, resp.text[:200],
            )
            return False
        try:
            data = resp.json()
            new_token = data.get("token")
        except Exception as exc:
            logger.warning("[lineage_fs] refresh-token parse failed: %s", exc)
            return False
        if not isinstance(new_token, str) or not new_token:
            return False
        os.environ["LINEAGE_RUN_TOKEN"] = new_token
        logger.info("[lineage_fs] run token refreshed successfully")
        return True


def _headers() -> dict[str, str]:
    token = os.environ.get("LINEAGE_RUN_TOKEN", "")
    # Pre-emptive refresh: if the token expires within the refresh window,
    # rotate it BEFORE making the call. Skip for refresh-token calls
    # themselves (infinite recursion) — we detect those by the caller
    # already holding the refresh lock via best-effort check.
    if token:
        exp = _decode_jwt_exp(token)
        if exp is not None:
            now = int(time.time())
            if exp - now <= _REFRESH_WINDOW_SECONDS:
                _refresh_run_token()
                token = os.environ.get("LINEAGE_RUN_TOKEN", token)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


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


def is_lineage_path(rel: str) -> bool:
    """True if the path targets a Lineage mount, False if it's scratch.

    The routing decision is based on the workspace layout declared in
    ``workspace-builder.ts``. A path is "Lineage" when:
      * First segment is a shared root (``conversations/``, ``slack/``,
        ``tasks/``, ``.pi/``), OR
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
        subdirs = [
            "transcripts/", "research/", "engagement/", "context/",
            "reports/", "posts/", "edits/", "tone/", "notes/", "strategy/",
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
            raise LineageIngestionError(
                f"list_directory({rel!r}): path not recognized by direct backend"
            )
        return direct

    # HTTP mode — only reachable in legacy deployments where direct isn't
    # configured (no GCS creds). Kept for Jacquard-initiated runs.
    try:
        resp = _get_client().get(
            f"{_base_url()}/list",
            params={"path": f"/{rel}" if rel else ""},
            headers=_headers(),
        )
    except httpx.HTTPError as exc:
        logger.error("[lineage_fs] FATAL list error: %s", exc)
        raise LineageIngestionError(
            f"list_directory({rel!r}): HTTP failed: {exc}"
        ) from exc

    if resp.status_code == 401:
        raise LineageIngestionError(
            f"list_directory({rel!r}): run token invalid or expired — ingestion auth broken"
        )
    if resp.status_code >= 500:
        raise LineageIngestionError(
            f"list_directory({rel!r}): workspace returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code != 200:
        raise LineageIngestionError(
            f"list_directory({rel!r}): workspace returned {resp.status_code}"
        )

    data = resp.json()
    entries = data.get("entries", [])
    if not entries:
        return "(empty directory)"

    lines: list[str] = []
    for e in entries:
        name = e.get("name", "?")
        if e.get("type") == "directory":
            lines.append(f"  {name}/")
        else:
            lines.append(f"  {name}")
    return "\n".join(lines)


def exec_read_file(_workspace_root: Any, args: dict) -> str:
    rel = args.get("path", "") or ""
    norm = _rewrite_path(rel)
    if not norm:
        return "Error: path is required"

    # Direct-mode path — Supabase + GCS, no HTTP hop.
    if _direct_enabled():
        try:
            direct = _direct_read(norm)
        except Exception as exc:
            raise LineageIngestionError(
                f"read_file({rel!r}): direct read failed: {exc}"
            ) from exc
        if direct is None:
            # 404-ish — file not found is soft-failure (Stelle can retry
            # with a different path), but a direct-backend miss means we
            # genuinely don't know where to look.
            return f"Error: file not found: {rel}"
        content = direct
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
        return content

    # HTTP mode fallback (legacy).
    try:
        resp = _get_client().get(
            f"{_base_url()}/file",
            params={"path": f"/{norm}"},
            headers=_headers(),
        )
    except httpx.HTTPError as exc:
        logger.error("[lineage_fs] FATAL read error %s: %s", norm, exc)
        raise LineageIngestionError(
            f"read_file({rel!r}): HTTP failed: {exc}"
        ) from exc

    # 404 is the one non-fatal case: "file genuinely doesn't exist in
    # Lineage" is different from "we couldn't talk to Lineage." Stelle
    # handling a missing file (retrying with a different name) shouldn't
    # kill the run.
    if resp.status_code == 404:
        return f"Error: file not found: {rel}"
    if resp.status_code == 401:
        raise LineageIngestionError(
            f"read_file({rel!r}): run token invalid or expired — ingestion auth broken"
        )
    if resp.status_code != 200:
        raise LineageIngestionError(
            f"read_file({rel!r}): workspace returned {resp.status_code}: {resp.text[:200]}"
        )

    content = resp.json().get("content", "")
    if len(content) > MAX_TOOL_OUTPUT_CHARS:
        content = content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
    return content


def exec_write_file(_workspace_root: Any, args: dict) -> str:
    rel = args.get("path", "") or ""
    content = args.get("content", "") or ""
    norm = _rewrite_path(rel)
    if not norm:
        return "Error: path is required"
    try:
        resp = _get_client().post(
            f"{_base_url()}/write",
            headers=_headers(),
            json={"path": f"/{norm}", "content": content},
        )
    except httpx.HTTPError as exc:
        logger.warning("[lineage_fs] write error %s: %s", norm, exc)
        return f"Error: remote write failed: {exc}"

    if resp.status_code == 403:
        return f"Error: {rel} is read-only in the Lineage workspace"
    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code >= 300:
        return f"Error: remote write {resp.status_code}: {resp.text[:200]}"
    return f"Wrote {len(content)} chars to {rel}"


def exec_edit_file(_workspace_root: Any, args: dict) -> str:
    """Read → replace-exactly-once → write, all via HTTP."""
    rel = args.get("path", "") or ""
    old_text = args.get("old_text", "") or ""
    new_text = args.get("new_text", "") or ""
    if not rel:
        return "Error: path is required"
    if not old_text:
        return "Error: old_text is required"

    # Read
    current = exec_read_file(None, {"path": rel})
    if current.startswith("Error:"):
        return current

    count = current.count(old_text)
    if count == 0:
        return f"Error: old_text not found in {rel}"
    if count > 1:
        return f"Error: old_text matches {count} locations — must be unique"
    updated = current.replace(old_text, new_text, 1)

    # Write
    result = exec_write_file(None, {"path": rel, "content": updated})
    if result.startswith("Error:"):
        return result
    return f"Edited {rel} — replaced {len(old_text)} chars with {len(new_text)} chars"


def exec_search_files(_workspace_root: Any, args: dict) -> str:
    """Grep-style recursive search over the Lineage workspace.

    Mirrors Pi's ``search_files`` tool output format so Stelle sees
    identical results:  ``<path>:<line>: <trimmed-content>``.
    """
    query = args.get("query", "") or ""
    directory = args.get("directory", ".") or "."
    if not query:
        return "Error: query is required"

    # Prepend user slug (targeted mode) or leave as-is (company-wide).
    norm_dir = _rewrite_path(directory if directory != "." else "")

    try:
        resp = _get_client().post(
            f"{_base_url()}/search",
            headers=_headers(),
            json={
                "query": query,
                "path": f"/{norm_dir}" if norm_dir else "",
                "maxResults": 100,
            },
        )
    except httpx.HTTPError as exc:
        logger.warning("[lineage_fs] search error: %s", exc)
        return f"Error: remote search failed: {exc}"

    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code != 200:
        return f"Error: remote search {resp.status_code}: {resp.text[:200]}"

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return "(no matches)"

    lines = [
        f"{r.get('path', '?')}:{r.get('line', '?')}: {r.get('text', '')}"
        for r in results
    ]
    if data.get("truncated"):
        lines.append("... (truncated at 100 matches)")
    return "\n".join(lines)


def exec_bash_lineage_stub(_workspace_root: Any, args: dict) -> str:
    """Lineage-mode replacement for the local ``bash`` tool.

    Pi's bash ran inside the virtual workspace so the agent could invoke
    shell pipelines against real workspace files. Under Lineage mode the
    workspace lives on a remote host; running bash locally on the fly.io
    container would execute against a scratch filesystem the agent has
    no reason to touch.

    Rather than silently succeed with bogus results, we fail loud with
    specific guidance on the tool to use instead.
    """
    cmd = (args.get("command") or "")[:120]
    return (
        "Error: bash is disabled in Lineage mode — the remote workspace "
        "is not reachable via a local shell.\n"
        "Use structured tools instead:\n"
        "  • read/list/edit/write_file for filesystem ops\n"
        "  • search_files for grep-style search\n"
        "  • mention_resolve for LinkedIn URN lookup\n"
        "  • web_search / fetch_url for web calls\n"
        f"(rejected command: {cmd!r})"
    )


def exec_submit_draft(_workspace_root: Any, args: dict) -> str:
    """Submit a FINAL draft — atomic create with all metadata.

    Mirrors the intent of Amphoreus's ``write_file(posts/drafts/<id>/content.md)``
    BUT in one call sets content + scheduled_date + approvers + rationale.
    This avoids the v1/v2 duplicate-draft bug that happens when an agent
    writes a rough pass, then writes a revision as a new file — each
    write-through DraftsFs created its own ``drafts`` row.

    Required:  user_slug (str), content (str)
    Optional:  scheduled_date (YYYY-MM-DD), approver_user_ids (list[uuid]),
               publication_order (int), why_post (str)
    """
    user_slug = args.get("user_slug") or ""
    content = args.get("content") or ""
    if not user_slug:
        return "Error: user_slug is required (pick the FOC user this post is for)"
    if not content:
        return "Error: content is required"

    payload: dict[str, Any] = {"user_slug": user_slug, "content": content}
    if args.get("scheduled_date"):
        payload["scheduled_date"] = args["scheduled_date"]
    if isinstance(args.get("approver_user_ids"), list):
        payload["approver_user_ids"] = args["approver_user_ids"]
    if isinstance(args.get("publication_order"), int):
        payload["publication_order"] = args["publication_order"]
    if args.get("why_post"):
        payload["why_post"] = args["why_post"]

    try:
        resp = _get_client().post(
            f"{_base_url()}/submit-draft",
            headers=_headers(),
            json=payload,
        )
    except httpx.HTTPError as exc:
        logger.warning("[lineage_fs] submit_draft error: %s", exc)
        return f"Error: submit_draft call failed: {exc}"

    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code == 404:
        return resp.text
    if resp.status_code >= 300:
        return f"Error: submit_draft {resp.status_code}: {resp.text[:300]}"

    import json as _json
    out = resp.json()
    d = out.get("draft") or {}
    return (
        "Draft submitted successfully.\n"
        f"  id: {d.get('id')}\n"
        f"  status: {d.get('status')}\n"
        f"  scheduled_date: {d.get('scheduled_date')}\n"
        f"  approver_ids: {d.get('approver_ids')}\n"
        f"  title: {(d.get('title') or '')[:80]}\n"
        "\n"
        "The draft row is now in Lineage's `drafts` table and a "
        "review_draft task has been created for the approver(s)."
    )


def exec_query_observations(_workspace_root: Any, args: dict) -> str:
    """Query scored observations via Lineage's ``POST /api/workspace/observations``.

    Parity with Analyst's ``query_observations`` tool. Args mirror
    Amphoreus exactly: ``min_reward``, ``max_reward``, ``limit``,
    ``summary_only``. Returns the same JSON shape so Stelle's existing
    logic (filtering by reward, reading reward_mean / reward_std, etc.)
    continues to work unchanged.
    """
    body: dict[str, Any] = {}
    for k in ("min_reward", "max_reward", "limit", "summary_only"):
        if k in args:
            body[k] = args[k]

    try:
        resp = _get_client().post(
            f"{_base_url()}/observations",
            headers=_headers(),
            json=body,
        )
    except httpx.HTTPError as exc:
        logger.warning("[lineage_fs] query_observations error: %s", exc)
        return f"Error: observations call failed: {exc}"

    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code != 200:
        return f"Error: observations {resp.status_code}: {resp.text[:200]}"

    # Pass through — shape matches Analyst's query_observations output.
    return resp.text


def exec_mention_resolve(_workspace_root: Any, args: dict) -> str:
    """Resolve a LinkedIn username to a mention URN via Lineage's
    ``POST /api/workspace/mention-resolve``. Returns the same JSON shape
    as Pi's ``mention-resolve`` custom command:
        {"name": ..., "urn": "urn:li:member:...", "url": ...}
    """
    raw = (args.get("username") or "").strip()
    if not raw:
        return "Error: username is required"
    username = raw.lstrip("@")

    try:
        resp = _get_client().post(
            f"{_base_url()}/mention-resolve",
            headers=_headers(),
            json={"username": username},
        )
    except httpx.HTTPError as exc:
        logger.warning("[lineage_fs] mention-resolve error: %s", exc)
        return f"Error: mention-resolve failed: {exc}"

    if resp.status_code == 404:
        return f"Error: could not resolve LinkedIn username {username!r}"
    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code != 200:
        return f"Error: mention-resolve {resp.status_code}: {resp.text[:200]}"

    import json as _json
    return _json.dumps(resp.json(), indent=2)


def create_review_task(draft_id: str, title: str) -> tuple[bool, str]:
    """Mirror Pi's post-run hook: create a ``review_draft`` task row so
    the CS review queue picks up the new draft. No-op in company-wide mode
    (the agent's explicit ``write_file`` may already have issued this,
    and we don't know which user the draft belongs to from here).
    """
    if not is_lineage_mode():
        return False, "skipped: not in Lineage mode"
    try:
        resp = _get_client().post(
            f"{_base_url()}/task",
            headers=_headers(),
            json={
                "type": "review_draft",
                "title": title,
                "priority": "medium",
                "entityType": "draft",
                "entityId": draft_id,
            },
        )
    except httpx.HTTPError as exc:
        return False, f"Error: task creation failed: {exc}"

    if resp.status_code >= 300:
        return False, f"Error: task create {resp.status_code}: {resp.text[:200]}"
    return True, "task created"


# ---------------------------------------------------------------------------
# Direct helpers used by the runner's post-write path (not tool handlers)
# ---------------------------------------------------------------------------


def write_draft(path: str, content: str) -> tuple[bool, str]:
    """Write a draft file to the Lineage workspace.

    Returns (ok, message). Used by the final-output phase of Stelle when
    running in Lineage mode — the path is typically
    ``{user-slug}/posts/{draft_id}/draft.json``; supabase-fs's DraftsFs
    maps that to a Supabase ``drafts`` table row.
    """
    msg = exec_write_file(None, {"path": path, "content": content})
    ok = not msg.startswith("Error:")
    return ok, msg


# Directories at the workspace root that aren't FOC-user mounts. Used
# by ``resolve_primary_user_slug`` to skip past system mounts.
_SYSTEM_DIRS = frozenset({".pi", "conversations", "slack", "tasks", ".keep"})


def resolve_primary_user_slug() -> str | None:
    """Discover a FOC-user directory in the Lineage workspace.

    supabase-fs mounts one directory per user-with-``posts_content=true``
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
    """Attribute a finished post to the user-targeted FOC user.

    Only meaningful in **user-targeted mode** — when ``LINEAGE_USER_SLUG``
    is set, this writes to ``{slug}/posts/drafts/{draft_id}/content.md``
    which DraftsFs parses into a Supabase ``drafts`` row.

    In **company-wide mode** this returns a sentinel ``(False, "skipped: ...")``
    — the caller is expected to skip the auto-write since the agent will
    have issued its own ``write_file`` calls with the proper user prefix.
    """
    slug = _targeted_slug()
    if slug is None:
        return False, "skipped: company-wide run (agent issues its own write_file calls)"
    path = f"{slug}/posts/drafts/{draft_id}/content.md"
    return write_draft(path, content)