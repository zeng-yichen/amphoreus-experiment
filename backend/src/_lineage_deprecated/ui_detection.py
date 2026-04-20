"""Lineage-UI-initiation detection (DEPRECATED).

True when the run was kicked off from Lineage's UI (via Jacquard's
virio-api HTTP proxy). In that mode, ``LINEAGE_WORKSPACE_URL`` was set
in env so filesystem callbacks could reach virio-api's
``/api/workspace/*`` endpoints.

This helper governed the ``submit_draft`` write destination:
  - True  → insert into Jacquard's ``drafts`` table (Lineage review UI)
  - False → insert into Amphoreus's ``local_posts`` (amphoreus.app review)

Standalone Amphoreus (the current mode) never sets
``LINEAGE_WORKSPACE_URL``, so this always resolves to False. Removed
from active code.

To reconnect: re-import into ``backend/src/agents/workspace_fs.py``
and add the branch in ``stelle._dispatch_submit_draft`` that calls
``_lineage_deprecated.drafts_writer.insert_draft()`` when True.
"""

from __future__ import annotations

import os


def is_lineage_ui_initiated() -> bool:
    """True when ``LINEAGE_WORKSPACE_URL`` is set in env.

    Only set by Jacquard's virio-api proxy when it spawned a Stelle
    subprocess for a Lineage-UI-initiated run. amphoreus.app runs and
    direct-mode reads never populate it.
    """
    return bool(os.environ.get("LINEAGE_WORKSPACE_URL", "").strip())
