"""Amphoreus FastAPI application."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.src.auth.acl import Acl
from backend.src.auth.cf_access import CfAccessVerifier
from backend.src.auth.middleware import (
    AuditLogMiddleware,
    CfAccessAuthMiddleware,
    SupabaseAuthMiddleware,
    require_client_from_path,
)
from backend.src.auth.supabase_auth import build_verifier_from_env as _build_supabase_verifier
from backend.src.core.config import get_settings
from backend.src.db.local import initialize_db, mark_stale_runs_failed

logger = logging.getLogger("amphoreus")

# Single shared instances of the auth components. Instantiated at import time
# so the middleware (which is registered before lifespan runs) and the app
# state (populated during lifespan) reference the same objects.
_settings_singleton = get_settings()

# Auth-mode selector. ``AMPHOREUS_AUTH``:
#   - "cf_access" (default)        — legacy Cloudflare Access gate
#   - "supabase"                   — Google sign-in via Supabase Auth
# The two modes are mutually exclusive; both use the same ACL file so
# we can cut over without touching user entitlements.
AUTH_MODE = os.environ.get("AMPHOREUS_AUTH", "cf_access").strip().lower() or "cf_access"
CF_VERIFIER = CfAccessVerifier(
    team_domain=_settings_singleton.cf_access_team_domain,
    audience=_settings_singleton.cf_access_aud,
)
SUPABASE_VERIFIER = _build_supabase_verifier()
ACL = Acl(path=_settings_singleton.acl_path)


def _startup_catchup() -> None:
    """Run one-time catch-up steps on backend boot.

    Currently a no-op. The previous implementation ran
    ``feedback_distiller.distill_directives`` + ``backfill_active_directives``
    on every startup, which turned client feedback into a hand-authored
    rule list and stamped it onto every observation as ``active_directives``.
    That pattern — distill rules, tag observations with rule IDs, feed the
    tagged observations back to the writer — is a closed loop of
    prescriptive injection. Removed to comply with the Bitter Lesson
    filter: the writer reads raw feedback files directly from
    ``memory/{company}/feedback/`` and decides what matters.
    """
    return


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Amphoreus backend...")
    initialize_db()
    logger.info("SQLite initialized at %s", settings.sqlite_path)

    # Install LLM usage instrumentation — monkey-patches the Anthropic SDK
    # so every messages.create call is recorded to usage_events with the
    # authenticated user attribution pulled from the ContextVar set by the
    # CF Access middleware. Idempotent; safe to call on every startup.
    try:
        from backend.src.usage import install_instrumentation
        install_instrumentation()
    except Exception:
        logger.exception("Failed to install usage instrumentation (non-fatal).")

    # CLI-mode leak guard — log-only tripwire that flags any direct
    # Anthropic API call while AMPHOREUS_USE_CLI=true. See
    # backend/src/mcp_bridge/cli_leak_guard.py. No effect when CLI mode
    # is off. Installed AFTER the usage instrumentation so both wrap
    # the same methods without stomping each other.
    try:
        from backend.src.mcp_bridge.cli_leak_guard import install_leak_guard
        install_leak_guard()
    except Exception:
        logger.exception("Failed to install CLI leak guard (non-fatal).")
    stale = mark_stale_runs_failed()
    if stale:
        logger.info("Marked %d stale 'running' job(s) as failed (server restart).", stale)

    # --- Stage 2 auth: Cloudflare Access verifier + ACL ---
    # Single shared instances live at module level (CF_VERIFIER, ACL) so the
    # middleware (registered below) and the FastAPI deps (which read
    # ``request.app.state.acl``) see the same objects.
    app.state.cf_verifier = CF_VERIFIER
    app.state.supabase_verifier = SUPABASE_VERIFIER
    app.state.acl = ACL
    if CF_VERIFIER.enabled:
        logger.info(
            "CF Access auth ENABLED (team=%s, aud=%s...)",
            settings.cf_access_team_domain,
            settings.cf_access_aud[:8],
        )
        # Eager-fetch JWKS at startup so misconfig fails loudly instead of on first request.
        try:
            CF_VERIFIER._get_jwk_client()  # noqa: SLF001
        except Exception:
            logger.exception("CF Access JWKS fetch failed at startup")
    else:
        logger.warning(
            "CF Access auth DISABLED — CF_ACCESS_TEAM_DOMAIN or CF_ACCESS_AUD is empty. "
            "Every request will be treated as an anonymous dev admin. NEVER run this way in prod."
        )

    try:
        _startup_catchup()
    except Exception:
        logger.exception("Startup catch-up failed (non-fatal).")

    # Ordinal sync loops — DISABLED BY DEFAULT as of 2026-04-18.
    # Stelle now runs primarily in database mode (invoked from Jacquard's
    # virio-api) where engagement data is sourced via Jacquard's Supabase
    # through ``POST /api/workspace/observations``. The local memory/ mirror
    # is no longer the primary data path.
    #
    # Set ``ENABLE_SYNC_LOOPS=true`` to re-enable (legacy local-mode runs,
    # or to refresh memory/{company}/ artifacts that Cyrene still reads).
    # The ordinal_sync code remains intact so manual invocations still work.
    enable_sync = os.getenv("ENABLE_SYNC_LOOPS", "").strip().lower() in ("true", "1", "yes")
    if enable_sync and not CF_VERIFIER.enabled:
        try:
            from backend.src.services.ordinal_sync import start_sync_loop, start_fast_sync_loop, stop_sync_loop
            start_sync_loop()           # hourly: full pipeline
            start_fast_sync_loop()      # 15 min: engagement snapshots for posts <72h old
            logger.info("Ordinal sync loops started (slow=3600s, fast=900s).")
        except Exception:
            logger.exception("Failed to start Ordinal sync loop (non-fatal).")
    elif CF_VERIFIER.enabled:
        logger.info("Ordinal sync loops SKIPPED (Fly — run sync locally if needed via ENABLE_SYNC_LOOPS).")
    else:
        logger.info("Ordinal sync loops SKIPPED (default — set ENABLE_SYNC_LOOPS=true to enable).")

    # post_embeddings nightly refresh — opt-in via ENABLE_POST_EMBEDDINGS_CRON.
    # Keeps the ~390k-post pgvector mirror fresh against Jacquard's
    # linkedin_posts by running scripts/embed_linkedin_posts.py --incremental
    # once a day. Cheap (<$0.05/day typically). No-op if env flag is off.
    try:
        from backend.src.services.post_embeddings_cron import (
            start_post_embeddings_cron,
        )
        start_post_embeddings_cron()
    except Exception:
        logger.exception("Failed to start post_embeddings cron (non-fatal).")

    # Jacquard → Amphoreus mirror sync — opt-in via
    # ENABLE_JACQUARD_MIRROR_SYNC. Pulls the agent-ingested tables from
    # Jacquard's Supabase (+ transcript bodies from GCS to Supabase
    # Storage) every 8h. Reads aren't yet wired to the mirror; this just
    # keeps it populated so the cutover is a no-op when we want it.
    try:
        from backend.src.services.jacquard_mirror_cron import (
            start_jacquard_mirror_cron,
        )
        start_jacquard_mirror_cron()
    except Exception:
        logger.exception("Failed to start jacquard_mirror cron (non-fatal).")

    # (draft_publish_matcher v1 cron removed 2026-04-23 — superseded
    # by (a) the semantic ``draft_match_worker`` that runs after the
    # Amphoreus + Jacquard scrapes and (b) the manual-date pairing
    # endpoint ``POST /api/posts/{id}/set-publish-date``. Both write
    # to ``local_posts.matched_provider_urn`` which is what the bundle
    # reads; v1 wrote to a separate ``draft_publish_matches`` table
    # that had no downstream consumer.)

    # Ordinal-reviewer-comment ingestor — opt-in via
    # ENABLE_ORDINAL_COMMENT_SYNC. Transitional: polls Ordinal's
    # per-post /comments + /inline-comments endpoints and writes
    # reviewer feedback into draft_feedback so Cyrene rewrites see
    # Ordinal-sourced comments. Delete this wiring once Ordinal is
    # retired. See services/ordinal_comment_sync.py.
    try:
        from backend.src.services.ordinal_comment_sync_cron import (
            start_ordinal_comment_sync_cron,
        )
        start_ordinal_comment_sync_cron()
    except Exception:
        logger.exception("Failed to start ordinal_comment_sync cron (non-fatal).")

    # Amphoreus-owned LinkedIn scrape — opt-in via
    # ENABLE_AMPHOREUS_LINKEDIN_SCRAPE. Hits Apify's
    # apimaestro/linkedin-profile-posts actor at 00:00 every weekday
    # for each Virio-serviced FOC, upserting engagement counts into
    # linkedin_posts (coexists with the Jacquard mirror via the
    # _source column). See services/amphoreus_linkedin_scrape.py for
    # the write contract.
    try:
        from backend.src.services.amphoreus_linkedin_scrape_cron import (
            start_amphoreus_linkedin_scrape_cron,
        )
        start_amphoreus_linkedin_scrape_cron()
    except Exception:
        logger.exception("Failed to start amphoreus_linkedin_scrape cron (non-fatal).")

    # Phainon exemplar prototype — opt-in via ENABLE_PHAINON_PROTOTYPE.
    # Weekly (Sunday 23:00 UTC by default) generation of N candidate
    # drafts per FOC in the 6-creator prototype roster, optionally
    # scored by the V2a reward function (only for creators where
    # ground-truth-era Spearman ≥ 0.4). Top exemplars surface in
    # Stelle's post bundle. See services/phainon.py.
    try:
        from backend.src.services.phainon_cron import start_phainon_cron
        start_phainon_cron()
    except Exception:
        logger.exception("Failed to start phainon cron (non-fatal).")

    yield

    # Graceful shutdown. Only call stop_sync_loop if the module was
    # imported by the startup branch above — otherwise the import would
    # re-trigger every shutdown for no reason.
    try:
        from backend.src.services.ordinal_sync import stop_sync_loop as _stop
        _stop()
    except Exception:
        pass
    try:
        from backend.src.services.post_embeddings_cron import (
            stop_post_embeddings_cron,
        )
        stop_post_embeddings_cron()
    except Exception:
        pass
    try:
        from backend.src.services.jacquard_mirror_cron import (
            stop_jacquard_mirror_cron,
        )
        stop_jacquard_mirror_cron()
    except Exception:
        pass
    try:
        from backend.src.services.amphoreus_linkedin_scrape_cron import (
            stop_amphoreus_linkedin_scrape_cron,
        )
        stop_amphoreus_linkedin_scrape_cron()
    except Exception:
        pass
    try:
        from backend.src.services.phainon_cron import stop_phainon_cron
        stop_phainon_cron()
    except Exception:
        pass
    try:
        from backend.src.services.ordinal_comment_sync_cron import (
            stop_ordinal_comment_sync_cron,
        )
        stop_ordinal_comment_sync_cron()
    except Exception:
        pass
    logger.info("Shutting down Amphoreus backend.")


settings = _settings_singleton

# Global FastAPI dependency: enforces per-client ACL on every route that has a
# ``{company}`` path param. Routes that carry ``company`` in the request body
# call ``require_client_body`` explicitly from within the handler.
app = FastAPI(
    title="Amphoreus",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_client_from_path)],
)

# Middleware ordering matters: the outermost wrapper runs last on the way in,
# first on the way out. Starlette applies them in REVERSE order of add_middleware
# calls — so the LAST add is the OUTERMOST wrapper. We want:
#   outermost: CORS (so 401 responses still get CORS headers)
#   middle:    Auth (gates everything — CF Access OR Supabase depending on AUTH_MODE)
#   innermost: AuditLog (sees the final status code, knows which user)
# Therefore add order is: AuditLog → Auth → CORS.
app.add_middleware(AuditLogMiddleware, log_path=settings.audit_log_path)
if AUTH_MODE == "supabase":
    logger.info("Auth mode: SUPABASE (Google sign-in via %s)", _settings_singleton.amphoreus_supabase_url if hasattr(_settings_singleton, 'amphoreus_supabase_url') else os.environ.get("AMPHOREUS_SUPABASE_URL", "<unset>"))
    app.add_middleware(SupabaseAuthMiddleware, verifier=SUPABASE_VERIFIER, acl=ACL)
else:
    logger.info("Auth mode: CF_ACCESS (legacy — set AMPHOREUS_AUTH=supabase to swap)")
    app.add_middleware(CfAccessAuthMiddleware, verifier=CF_VERIFIER, acl=ACL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register routers ---
from backend.src.api.routers import (
    auth,
    briefings,
    clients,
    cs,
    deploy,
    desktop,
    ghostwriter,
    images,
    interview,
    learning,
    mirror,
    posts,
    report,
    strategy,
    transcripts,
    usage,
)

app.include_router(auth.router)
app.include_router(clients.router)
app.include_router(deploy.router)
app.include_router(desktop.router)
app.include_router(ghostwriter.router)
app.include_router(briefings.router)
app.include_router(interview.router)
app.include_router(strategy.router)
app.include_router(posts.router)
app.include_router(images.router)
app.include_router(cs.router)
app.include_router(learning.router)
app.include_router(mirror.router)
app.include_router(report.router)
app.include_router(transcripts.router)
app.include_router(usage.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "amphoreus"}


@app.get("/api/me")
async def me(request: Request):
    """Return the current user's identity + ACL-scoped client list.

    Frontend calls this on page load to populate the auth context and decide
    which client tabs to render in the sidebar. Returns ``allowed_clients: "*"``
    for admins, or an explicit list of slugs for scoped users.

    ``auth_enabled`` reflects whichever verifier is active for the current
    ``AMPHOREUS_AUTH`` mode — the frontend uses it to decide whether to
    hide dev-only UI (Push Code → Fly, etc.) and to know whether a 401
    means "go to /login" (Supabase mode) vs the CF Access redirect.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        return {"email": "", "is_admin": False, "allowed_clients": []}
    is_admin = bool(getattr(request.state, "user_is_admin", False))
    allowed: object = "*" if is_admin else ACL.allowed_clients(user.email)
    if AUTH_MODE == "supabase":
        auth_enabled = SUPABASE_VERIFIER.enabled
    else:
        auth_enabled = CF_VERIFIER.enabled
    return {
        "email": user.email,
        "is_admin": is_admin,
        "allowed_clients": allowed,
        "auth_enabled": auth_enabled,
        "auth_mode": AUTH_MODE,
    }
