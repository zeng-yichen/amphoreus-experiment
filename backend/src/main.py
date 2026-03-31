"""Amphoreus FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.core.config import get_settings
from backend.src.db.local import initialize_db

logger = logging.getLogger("amphoreus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Amphoreus backend...")
    initialize_db()
    logger.info("SQLite initialized at %s", settings.sqlite_path)
    yield
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
    cs,
    desktop,
    ghostwriter,
    images,
    interview,
    posts,
    research,
    strategy,
)

app.include_router(auth.router)
app.include_router(desktop.router)
app.include_router(ghostwriter.router)
app.include_router(briefings.router)
app.include_router(interview.router)
app.include_router(strategy.router)
app.include_router(posts.router)
app.include_router(images.router)
app.include_router(research.router)
app.include_router(cs.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "amphoreus"}
