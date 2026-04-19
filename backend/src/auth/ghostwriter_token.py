"""Shared-secret JWT verifier for Jacquard ↔ Amphoreus backend-to-backend calls.

When Amphoreus is reachable over the public internet (see ``fly.toml``'s
``[[services]]`` block), we can't rely on Fly's internal 6PN to gate
traffic. Instead we require every request to carry a run-JWT signed with
``GHOSTWRITER_SHARED_SECRET`` — the same HS256 secret Jacquard's virio-api
uses to mint these tokens (see ``jacquard/virio-api/src/lib/run-jwt.ts``).

Functionally equivalent to Cloudflare Access service-tokens for this
specific integration: both are HS256 shared-secret verification.

Flow:
  1. Jacquard mints a token with claims {companyId, jobId, userId?, exp}
     and sends it as ``Authorization: Bearer <jwt>`` when it proxies a
     request to Amphoreus.
  2. Amphoreus's middleware (``CfAccessAuthMiddleware`` extended) calls
     :func:`verify_ghostwriter_token` on the bearer token.
  3. On success, the request is admitted as an "admin" dev user (the
     tokens are only minted by trusted backend code; the user-ACL layer
     is a browser-traffic concern, not backend-to-backend).

When the shared secret is unset, this verifier is disabled and returns
``None`` for every token — letting the surrounding auth chain fall
through to CF Access or the local-dev bypass as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import jwt

logger = logging.getLogger("amphoreus.auth.ghostwriter")


@dataclass(frozen=True)
class GhostwriterTokenClaims:
    """Claims extracted from a successfully-verified ghostwriter run-JWT."""

    company_id: str
    job_id: str
    user_id: str | None


def _shared_secret() -> str:
    return os.environ.get("GHOSTWRITER_SHARED_SECRET", "").strip()


def is_enabled() -> bool:
    """True when GHOSTWRITER_SHARED_SECRET is configured.

    When disabled, :func:`verify_ghostwriter_token` always returns None,
    which lets the surrounding auth chain fall through unchanged.
    """
    return bool(_shared_secret())


def verify_ghostwriter_token(token: str) -> GhostwriterTokenClaims | None:
    """Verify an HS256 JWT minted by Jacquard's ``mintRunToken``.

    Returns the decoded claims on success, ``None`` on any failure
    (including when the shared secret is not configured). Never raises.

    We deliberately do NOT raise jwt exceptions: the caller uses the
    return value to decide whether to fall through to the next auth
    mechanism (CF Access), rather than 401-ing immediately.
    """
    secret = _shared_secret()
    if not secret or not token:
        return None

    try:
        decoded: Any = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        logger.debug("[ghostwriter-token] expired")
        return None
    except jwt.InvalidTokenError as exc:
        # Wrong signature, bad structure, etc. — silently fall through so
        # the caller can try the next verifier.
        logger.debug("[ghostwriter-token] rejected: %s", exc)
        return None
    except Exception:
        logger.exception("[ghostwriter-token] unexpected verification error")
        return None

    if not isinstance(decoded, dict):
        return None
    company_id = decoded.get("companyId")
    job_id = decoded.get("jobId")
    if not isinstance(company_id, str) or not isinstance(job_id, str):
        return None
    user_id = decoded.get("userId")
    if not isinstance(user_id, str):
        user_id = None

    return GhostwriterTokenClaims(
        company_id=company_id,
        job_id=job_id,
        user_id=user_id,
    )
