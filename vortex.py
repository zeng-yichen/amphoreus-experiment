"""
Vortex — Centralised path layout for Amphoreus.

    memory/{company}/          — persistent client knowledge
        transcripts/           — raw interview transcripts & documents
        accepted/              — approved posts (style exemplars)
        feedback/              — client feedback & Ordinal comments
        revisions/             — before/after revision pairs (pipeline draft → human revision)
        abm_profiles/          — ABM target briefings
        targets/               — xlsx / data files for ABM target sourcing
        past_posts/            — post history for redundancy checking
        content_strategy/      — content strategy docs
        tmp/                   — ephemeral fetched data

    memory/our_memory/         — shared knowledge (LinkedIn writing guidelines, etc.)

    products/{company}/        — generated artefacts
        post/                  — post markdown files
        brief/                 — briefing markdown files
"""

import os
import pathlib

MEMORY_ROOT = pathlib.Path("memory")
PRODUCTS_ROOT = pathlib.Path("products")


def memory_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company


def transcripts_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "transcripts"


def accepted_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "accepted"


def feedback_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "feedback"


def revisions_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "revisions"


def abm_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "abm_profiles"


def past_posts_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "past_posts"


def content_strategy_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "content_strategy"


def targets_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "targets"


def notes_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "notes"


def tmp_dir(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "tmp"


def post_dir(company: str) -> pathlib.Path:
    return PRODUCTS_ROOT / company / "post"


def brief_dir(company: str) -> pathlib.Path:
    return PRODUCTS_ROOT / company / "brief"


def ordinal_auth_csv() -> pathlib.Path:
    return MEMORY_ROOT / "ordinal_auth_rows.csv"


def our_memory_dir() -> pathlib.Path:
    return MEMORY_ROOT / "our_memory"


def linkedin_username_path(company: str) -> pathlib.Path:
    return MEMORY_ROOT / company / "linkedin_username.txt"


def ensure_dirs(company: str) -> None:
    """Create the full directory tree for a client."""
    for d in (
        transcripts_dir(company),
        accepted_dir(company),
        feedback_dir(company),
        revisions_dir(company),
        abm_dir(company),
        targets_dir(company),
        past_posts_dir(company),
        content_strategy_dir(company),
        tmp_dir(company),
        post_dir(company),
        brief_dir(company),
    ):
        d.mkdir(parents=True, exist_ok=True)
