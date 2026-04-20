"""HTTP-mode Lineage workspace client (DEPRECATED).

This module is quarantined — nothing in the active codebase imports it.
It preserves the HTTP-mode filesystem client for the Lineage-UI
integration, where Jacquard's virio-api spawned Stelle as a remote
worker and Stelle called back into ``<workspace_url>/list``, ``/file``,
``/write``, ``/search``, ``/submit-draft`` over HTTPS under a per-run
HS256 JWT bound to ``{companyId, jobId}``.

Direct-mode Supabase reads (the active path today) did not use any of
this — see ``workspace_fs.py`` and ``jacquard_direct.py`` for the
current ingestion flow.

To reconnect:
  1. Re-import the ``_http_*`` helpers below into ``workspace_fs.py``
     and re-add the fallback branch after the direct-mode attempts in
     each ``exec_*`` dispatcher (the direct-mode branch currently
     returns ``None`` → empty hint; the old code then tried HTTP).
  2. Restore env-var plumbing in ``stelle_runner.py``
     (``--lineage-workspace-url``, ``--lineage-run-token``) and
     re-add header forwarding in
     ``backend/src/api/routers/ghostwriter.py``.
  3. Re-add ``exec_submit_draft_http`` as a branch inside
     ``_dispatch_submit_draft`` gated on
     ``_lineage_deprecated.ui_detection.is_lineage_ui_initiated()``.

Environment variables consumed:
  LINEAGE_WORKSPACE_URL  base URL of virio-api's /api/workspace
  LINEAGE_RUN_TOKEN      HS256 bearer token, auto-refreshed inside the
                         REFRESH_GRACE_SECONDS window before expiry.
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

logger = logging.getLogger("stelle.lineage_deprecated.http")

MAX_TOOL_OUTPUT_CHARS = 80_000
_REFRESH_WINDOW_SECONDS = 10 * 60
_refresh_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60.0)
    return _client


def _base_url() -> str:
    url = os.environ["LINEAGE_WORKSPACE_URL"].rstrip("/")
    return url


def _decode_jwt_exp(token: str) -> int | None:
    """Return the ``exp`` (unix seconds) claim without verifying — only
    used to decide whether to pre-emptively refresh."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
        exp = data.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
        return None
    except Exception:
        return None


def _refresh_run_token() -> bool:
    """Trade a near-expired run token for a fresh one. Updates
    ``LINEAGE_RUN_TOKEN`` in-place on success. Returns True on success."""
    with _refresh_lock:
        current = os.environ.get("LINEAGE_RUN_TOKEN", "")
        exp = _decode_jwt_exp(current)
        if exp is not None and exp - int(time.time()) > _REFRESH_WINDOW_SECONDS:
            return True
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
            logger.warning("refresh-token network error: %s", exc)
            return False
        if resp.status_code >= 300:
            logger.warning("refresh-token %s: %s", resp.status_code, resp.text[:200])
            return False
        try:
            data = resp.json()
            new_token = data.get("token")
        except Exception as exc:
            logger.warning("refresh-token parse failed: %s", exc)
            return False
        if not isinstance(new_token, str) or not new_token:
            return False
        os.environ["LINEAGE_RUN_TOKEN"] = new_token
        return True


def _headers() -> dict[str, str]:
    """Build outbound headers with auto-refreshed bearer token."""
    token = os.environ.get("LINEAGE_RUN_TOKEN", "")
    if token:
        exp = _decode_jwt_exp(token)
        if exp is not None and exp - int(time.time()) <= _REFRESH_WINDOW_SECONDS:
            _refresh_run_token()
            token = os.environ.get("LINEAGE_RUN_TOKEN", token)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Filesystem HTTP fallbacks — previously lived as the second branch inside
# each ``exec_*`` dispatcher in ``lineage_fs_client.py``. Preserved here as
# standalone helpers so the call shape is self-documenting if you reconnect.
# ---------------------------------------------------------------------------


def _http_list(rel: str) -> str:
    """GET /api/workspace/list?path=/<rel>  →  directory listing."""
    try:
        resp = _get_client().get(
            f"{_base_url()}/list",
            params={"path": f"/{rel}" if rel else ""},
            headers=_headers(),
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"list_directory({rel!r}): HTTP failed: {exc}") from exc

    if resp.status_code == 401:
        raise RuntimeError(f"list_directory({rel!r}): run token invalid or expired")
    if resp.status_code >= 500:
        raise RuntimeError(
            f"list_directory({rel!r}): workspace returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"list_directory({rel!r}): workspace returned {resp.status_code}")

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


def _http_read(rel: str) -> str:
    """GET /api/workspace/file?path=/<rel>  →  file content (str)."""
    try:
        resp = _get_client().get(
            f"{_base_url()}/file",
            params={"path": f"/{rel}"},
            headers=_headers(),
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"read_file({rel!r}): HTTP failed: {exc}") from exc

    if resp.status_code == 404:
        return f"Error: file not found: {rel}"
    if resp.status_code == 401:
        raise RuntimeError(f"read_file({rel!r}): run token invalid or expired")
    if resp.status_code != 200:
        raise RuntimeError(f"read_file({rel!r}): workspace returned {resp.status_code}: {resp.text[:200]}")

    content = resp.json().get("content", "")
    if len(content) > MAX_TOOL_OUTPUT_CHARS:
        content = content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
    return content


def _http_write(rel: str, content: str) -> str:
    """POST /api/workspace/write  {path, content}  →  status string."""
    try:
        resp = _get_client().post(
            f"{_base_url()}/write",
            headers=_headers(),
            json={"path": f"/{rel}", "content": content},
        )
    except httpx.HTTPError as exc:
        return f"Error: remote write failed: {exc}"
    if resp.status_code == 403:
        return f"Error: {rel} is read-only in the Lineage workspace"
    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code >= 300:
        return f"Error: remote write {resp.status_code}: {resp.text[:200]}"
    return f"Wrote {len(content)} chars to {rel}"


def _http_search(query: str, norm_dir: str) -> str:
    """POST /api/workspace/search  {query, path, maxResults}  →  grep-style output."""
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


# ---------------------------------------------------------------------------
# Submit-draft (HTTP) — previously used when Lineage UI initiated the run.
# Active path today lives in ``stelle.py::_dispatch_submit_draft`` and writes
# to Amphoreus ``local_posts``. See ``drafts_writer.py`` for the Jacquard
# drafts-table write path (Supabase direct insert).
# ---------------------------------------------------------------------------


def exec_submit_draft_http(_workspace_root: Any, args: dict) -> str:
    """POST /api/workspace/submit-draft  — atomic draft finalization.

    Body: { user_slug, content, scheduled_date?, approver_user_ids?,
            publication_order?, why_post? }
    """
    user_slug = args.get("user_slug") or ""
    content = args.get("content") or ""
    if not user_slug:
        return "Error: user_slug is required"
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
        return f"Error: submit_draft call failed: {exc}"
    if resp.status_code == 401:
        return "Error: run token invalid or expired"
    if resp.status_code == 404:
        return resp.text
    if resp.status_code >= 300:
        return f"Error: submit_draft {resp.status_code}: {resp.text[:300]}"
    return (
        f"Draft submitted to Jacquard drafts table.\n"
        "The draft is now in Lineage's drafts table (status=review)."
    )


# ---------------------------------------------------------------------------
# Bash stub — when Stelle was running as a Lineage-UI-initiated remote
# worker, bash on the Fly container ran against scratch filesystem that
# didn't reflect the actual (remote) workspace. Fail loud instead.
# ---------------------------------------------------------------------------


def exec_bash_lineage_stub(_workspace_root: Any, args: dict) -> str:
    cmd = (args.get("command") or "")[:120]
    return (
        "bash is disabled while running under the Lineage workspace. "
        "The workspace is remote and your local filesystem has no "
        "reflection of it, so bash commands against scratch cannot see "
        "Jacquard-side data. Use the structured tools instead:\n"
        "  • read/list/edit/write_file for filesystem ops\n"
        "  • search_files for grep-style search\n"
        "  • mention_resolve for LinkedIn URN lookup\n"
        "  • web_search / fetch_url for web calls\n"
        f"(rejected command: {cmd!r})"
    )
