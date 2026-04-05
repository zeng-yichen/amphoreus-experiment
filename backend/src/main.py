"""Amphoreus FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.core.config import get_settings
from backend.src.db.local import initialize_db, mark_stale_runs_failed

logger = logging.getLogger("amphoreus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Amphoreus backend...")
    initialize_db()
    logger.info("SQLite initialized at %s", settings.sqlite_path)
    stale = mark_stale_runs_failed()
    if stale:
        logger.info("Marked %d stale 'running' job(s) as failed (server restart).", stale)

    # Start Ordinal sync loop (runs every hour in a background thread).
    # Feeds RuanMei with Ordinal LinkedIn analytics only (no Supabase writes).
    try:
        from backend.src.services.ordinal_sync import start_sync_loop, stop_sync_loop
        start_sync_loop()
        logger.info("Ordinal sync loop started.")
    except Exception:
        logger.exception("Failed to start Ordinal sync loop (non-fatal).")

    yield

    # Graceful shutdown.
    try:
        stop_sync_loop()
    except Exception:
        pass
    logger.info("Shutting down Amphoreus backend.")


app = FastAPI(title="Amphoreus", version="0.1.0", lifespan=lifespan)

settings = get_settings()
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
    desktop,
    ghostwriter,
    images,
    interview,
    learning,
    posts,
    research,
    strategy,
)

app.include_router(auth.router)
app.include_router(clients.router)
app.include_router(desktop.router)
app.include_router(ghostwriter.router)
app.include_router(briefings.router)
app.include_router(interview.router)
app.include_router(strategy.router)
app.include_router(posts.router)
app.include_router(images.router)
app.include_router(research.router)
app.include_router(cs.router)
app.include_router(learning.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "amphoreus"}
