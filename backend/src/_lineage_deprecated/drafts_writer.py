"""Write path into Jacquard's ``drafts`` Supabase table (DEPRECATED).

Used when Lineage's UI kicked off a Stelle run and the operator wanted
the finished draft to appear in Lineage's review UI alongside
Jacquard-native drafts. Lineage's editor is TipTap over Y.js, so a
valid ``yjs_state`` CRDT blob is required — without it the editor
opens blank regardless of what's in the plain ``content`` column.

Active path today writes to Amphoreus's ``local_posts`` SQLite table
instead. This module is not imported.

To reconnect: import ``insert_draft`` into
``backend/src/agents/jacquard_direct.py`` and wire it into
``stelle._dispatch_submit_draft`` behind an
``is_lineage_ui_initiated()`` branch. Also re-add ``pycrdt`` to the
main dependencies (currently in the ``[lineage-deprecated]`` extras
group of ``pyproject.toml``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("stelle.lineage_deprecated.drafts_writer")


def _text_to_yjs_state(content: str) -> str | None:
    """Encode plain text as a TipTap/y-prosemirror compatible Y.Doc
    update, returned as base64 for insert into ``drafts.yjs_state``.

    Schema observed from real Jacquard drafts:

        Doc
        └── XmlFragment "content"
            ├── <paragraph>first paragraph text</paragraph>
            ├── <paragraph></paragraph>   (blank spacer between blocks)
            ├── <paragraph>second paragraph</paragraph>
            ...

    Splits by ``\\n\\n``, collapses single ``\\n`` inside a block to
    spaces, inserts a blank paragraph between real ones. Returns None
    if ``pycrdt`` isn't importable (main deps may have dropped it).
    """
    try:
        import base64
        from pycrdt import Doc, XmlFragment, XmlElement, XmlText
    except Exception as exc:
        logger.warning("pycrdt unavailable, yjs_state will be NULL: %s", exc)
        return None

    doc = Doc()
    fragment = doc.get("content", type=XmlFragment)
    blocks = content.split("\n\n") if content else [""]
    while blocks and not blocks[0].strip():
        blocks.pop(0)
    while blocks and not blocks[-1].strip():
        blocks.pop()
    if not blocks:
        blocks = [""]
    for i, block in enumerate(blocks):
        body = block.replace("\n", " ").strip()
        p = XmlElement("paragraph")
        # pycrdt: parent must be integrated into the doc before children
        # can be appended. Append ``p`` to fragment first, then its text.
        fragment.children.append(p)
        if body:
            p.children.append(XmlText(body))
        if i < len(blocks) - 1:
            spacer = XmlElement("paragraph")
            fragment.children.append(spacer)

    update = doc.get_update()
    return base64.b64encode(update).decode()


def insert_draft(
    user_id: str,
    content: str,
    title: str | None = None,
    scheduled_date: str | None = None,
    status: str = "review",
    visibility: str = "PUBLIC",
) -> dict[str, Any] | None:
    """Insert a finished draft into Jacquard's ``drafts`` Supabase table.

    Returns the inserted row on success, None on failure. The row shows
    up immediately in Lineage's review UI for the specified ``user_id``.

    Column mapping (verified against real Jacquard rows):
      - user_id         FOC user UUID (from resolve_user_by_slug)
      - content         plain markdown body
      - title           first ~120 chars if not provided
      - status          'review' by default
      - visibility      'PUBLIC' (default LinkedIn visibility)
      - scheduled_date  YYYY-MM-DD is up-converted to timestamptz@12:00 UTC
      - yjs_state       TipTap CRDT blob computed from ``content``

    Does NOT set ``approver_ids``, ``linkedin_connection_id``,
    ``post_id``, ``proposal_id``, ``meeting_id``, or ``publish_content``
    — those are populated by Jacquard-side flows (scheduler, approver
    assignment, LinkedIn push).
    """
    if not user_id or not content:
        return None

    # Lazy import — keeps this module decoupled from jacquard_direct's
    # active surface so a future refactor of that module doesn't silently
    # break the reconnection path.
    from backend.src.agents.jacquard_direct import _sb

    sb = _sb()

    row: dict[str, Any] = {
        "user_id": user_id,
        "content": content,
        "status": status,
        "visibility": visibility,
    }
    if title:
        row["title"] = title[:200]
    else:
        first_line = (content.strip().split("\n", 1)[0] or "")[:120]
        if first_line:
            row["title"] = first_line

    if scheduled_date:
        if len(scheduled_date) == 10:
            row["scheduled_date"] = f"{scheduled_date}T12:00:00+00:00"
        else:
            row["scheduled_date"] = scheduled_date

    yjs_b64 = _text_to_yjs_state(content)
    if yjs_b64:
        row["yjs_state"] = yjs_b64

    try:
        resp = sb.table("drafts").insert(row).execute()
    except Exception as exc:
        logger.warning("insert_draft failed: %s", exc)
        return None

    data = resp.data or []
    return data[0] if data else None
