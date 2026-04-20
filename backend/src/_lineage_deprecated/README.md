# _lineage_deprecated

Quarantined code from the Lineage-UI integration. Amphoreus runs
standalone now (amphoreus.app); Stelle no longer accepts incoming runs
from Jacquard's Lineage product via virio-api.

**Jacquard's Supabase is still used as the data source** for transcripts,
engagement, research, context, etc. That's normal database access and
lives in `backend/src/agents/jacquard_direct.py` + `workspace_fs.py` —
not here. This directory is only for the Lineage-UI integration bits
(HTTP proxy, drafts-table writes, Lineage-specific prompt text).

Files in this directory are **not imported by any active code path**.
They exist as reference + easy-to-reconnect starting points.

## Files

### `http_client.py`
HTTP-mode Lineage workspace client. When a run was kicked off from
Lineage's UI, Jacquard's virio-api spawned a Stelle subprocess with
`LINEAGE_WORKSPACE_URL` + `LINEAGE_RUN_TOKEN` environment variables.
Stelle's filesystem operations then routed through this client to the
remote workspace over HTTPS, rather than reading Jacquard's Supabase
directly.

Contents:
- `_refresh_run_token`, `_headers`, `_base_url`, `_decode_jwt_exp` —
  per-request JWT plumbing (HS256, 4h TTL, refresh grace window)
- `_http_list`, `_http_read`, `_http_write`, `_http_edit`, `_http_search`
  — HTTP fallback paths that used to live inside `exec_*` dispatchers
  in `workspace_fs.py` (previously `lineage_fs_client.py`)
- `exec_submit_draft_http` — POST /api/workspace/submit-draft
- `exec_bash_lineage_stub` — refused bash in Lineage mode, pointed the
  agent at structured tools instead

### `drafts_writer.py`
Write path into Jacquard's `drafts` Supabase table. Used by
`submit_draft` when the run was Lineage-UI-initiated so the draft
appeared in Lineage's review UI. Includes:
- `insert_draft(user_id, content, title, scheduled_date, ...)`
- `_text_to_yjs_state(content)` — encodes plaintext as a
  y-prosemirror-compatible Y.Doc update (base64). Required because
  Lineage's TipTap editor renders from `yjs_state`, not the plain
  `content` column.

Requires `pycrdt` (Rust-backed Y-CRDT binding). Moved to optional
extras in `pyproject.toml` under `[lineage-deprecated]`.

### `prompt_overlays.py`
Stelle system-prompt additions that were appended on top of the main
prompt when running in Lineage mode. Three blocks:
- `_LINEAGE_DIRECTIVES_TOOL_OVERRIDES` — tool-semantic differences
  (memory/ paths vs Jacquard workspace, submit_draft shape)
- `_LINEAGE_DIRECTIVES_USER_TARGETED` — workspace layout when scoped
  to one FOC user
- `_LINEAGE_DIRECTIVES_COMPANY_WIDE` — workspace layout with all FOC
  users visible

The workspace-layout content from USER_TARGETED was promoted into
`stelle.py`'s main prompt during deprecation (it describes the actual
Jacquard-sourced workspace Stelle reads today). The rest is preserved
here as historical reference — the specific wording reflects how the
Lineage-UI handoff was framed to operators.

### `ui_detection.py`
`is_lineage_ui_initiated()` — true when `LINEAGE_WORKSPACE_URL` is set
in env. Governed whether `submit_draft` wrote to Jacquard's `drafts`
table (Lineage UI review) or Amphoreus's `local_posts` table
(amphoreus.app review). In standalone Amphoreus this always resolves
to False; it was removed from the active code paths.

## How to reconnect Lineage support

Step-by-step to restore the Lineage-UI integration:

1. **Reinstall pycrdt** — uncomment it or promote from the
   `[lineage-deprecated]` extras group back into the main dependencies
   in `pyproject.toml`.

2. **Re-import `_text_to_yjs_state` + `insert_draft`** into
   `backend/src/agents/jacquard_direct.py`.

3. **Re-import `is_lineage_ui_initiated`** into
   `backend/src/agents/workspace_fs.py`.

4. **Re-import the HTTP client functions** into `workspace_fs.py`.
   Each `exec_*` dispatcher currently only handles direct-mode
   Supabase reads — restore the HTTP fallback branch that used to
   follow. See the `http_client.py` module docstring for shape
   details.

5. **Update `_dispatch_submit_draft` in `stelle.py`** to branch on
   `is_lineage_ui_initiated()`:
   - True  → `jacquard_direct.insert_draft()` (Jacquard drafts table)
   - False → current path (Amphoreus `local_posts`)

6. **Restore CLI args in `stelle_runner.py`**:
   `--lineage-workspace-url`, `--lineage-run-token`. Export them to
   env vars of the same name before importing stelle agent code.

7. **Restore HTTP header forwarding in
   `backend/src/api/routers/ghostwriter.py`**:
   `X-Lineage-Workspace-URL`, `X-Lineage-Run-Token`, `Authorization`
   Bearer. Pipe them into the stelle_runner subprocess.

8. **Re-add the prompt overlay application** in
   `stelle._build_dynamic_directives` — conditionally append
   `_LINEAGE_DIRECTIVES_TOOL_OVERRIDES` (or whatever subset is still
   relevant) when `is_lineage_ui_initiated()` is True.

9. **Matching changes on the Jacquard side** — see
   `/Users/zengyichen/Downloads/amphoreus-integration.bundle` or the
   PR branch `amphoreus-integration` (3 commits: remote worker proxy,
   operations cleanup, SSE retry fix) in the Jacquard repo.
