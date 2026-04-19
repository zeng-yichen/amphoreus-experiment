"""Mint HS256 run-JWTs that Jacquard's virio-api workspace accepts.

Amphoreus reads Jacquard's workspace (transcripts, engagement, drafts,
meetings, etc.) on its own initiative — the operator clicks Generate on
Amphoreus's UI, not on Jacquard's Lineage. Since Jacquard isn't the
caller, nobody's minting a run token for us. We mint our own.

Jacquard's workspace verifier (``virio-api/src/lib/run-jwt.ts``) accepts
any HS256 JWT signed with ``GHOSTWRITER_SHARED_SECRET`` that carries
``{companyId, jobId}`` required claims + optional ``userId``. We sign
with the same secret and the tokens verify transparently.

Amphoreus never writes to Jacquard — these tokens are used to read only
(list, file, search, observations, mention-resolve). No submit-draft,
no task creation, no CRUD against Jacquard's state. Writes stay local
to Amphoreus.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import jwt

logger = logging.getLogger(__name__)

# Matches Jacquard's DEFAULT_TTL_SECONDS in virio-api/src/lib/run-jwt.ts.
# 4 hours covers any realistic Stelle run including Claude compaction +
# flame-chase loops with wide margin.
_DEFAULT_TTL_SECONDS = 4 * 60 * 60


def _shared_secret() -> str | None:
    """Return GHOSTWRITER_SHARED_SECRET from env, or None if unset.

    None signals "Jacquard integration not configured" — caller should
    fall back to pure-local mode without calling Jacquard.
    """
    secret = os.environ.get("GHOSTWRITER_SHARED_SECRET", "").strip()
    return secret or None


def is_jacquard_read_configured() -> bool:
    """True when Amphoreus has everything it needs to read from Jacquard.

    Requires both the shared secret (for signing) and a workspace URL
    (endpoint to call). If either is missing, the caller should not set
    ``LINEAGE_WORKSPACE_URL`` / ``LINEAGE_RUN_TOKEN`` on a Stelle
    subprocess — Stelle then runs in pure-local mode.
    """
    return bool(_shared_secret() and os.environ.get("JACQUARD_WORKSPACE_URL", "").strip())


def mint_run_token(
    company_id: str,
    user_id: str | None = None,
    job_id: str | None = None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str | None:
    """Mint an HS256 JWT scoped to one Jacquard company.

    Args:
        company_id:  Jacquard's ``user_companies.id`` UUID. Required.
        user_id:     Jacquard's ``users.id`` UUID. Optional; present for
                     user-targeted runs (omit for company-wide).
        job_id:      Opaque run identifier. Generated when omitted.
        ttl_seconds: Token validity window. Defaults to 4 hours.

    Returns:
        The signed JWT string, or None if the shared secret isn't
        configured. Caller should treat None as "can't reach Jacquard".
    """
    secret = _shared_secret()
    if not secret:
        logger.debug("[jacquard_jwt] GHOSTWRITER_SHARED_SECRET not set — cannot mint")
        return None
    if not company_id:
        logger.warning("[jacquard_jwt] mint_run_token called with empty company_id")
        return None

    now = int(time.time())
    payload: dict[str, object] = {
        "companyId": company_id,
        "jobId": job_id or str(uuid.uuid4()),
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if user_id:
        payload["userId"] = user_id

    return jwt.encode(payload, secret, algorithm="HS256")


def workspace_url() -> str:
    """Return the configured Jacquard workspace base URL (no trailing slash).

    Empty string if not configured.
    """
    return os.environ.get("JACQUARD_WORKSPACE_URL", "").strip().rstrip("/")
