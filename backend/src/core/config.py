"""Application configuration loaded from environment variables."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("amphoreus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


class Settings(BaseSettings):
    # --- LLM providers ---
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    parallel_api_key: str = ""

    # --- Supabase (read/write existing tables only — no schema changes) ---
    supabase_url: str = ""
    supabase_key: str = ""

    # --- Pinecone ---
    pinecone_api_key: str = ""
    pinecone_index: str = ""

    # --- Serper (Google Search API) ---
    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev/search"

    # --- Ordinal ---
    ordinal_api_key: str = ""

    # --- E2B ---
    e2b_api_key: str = ""

    # --- App ---
    allowed_origins: str = "http://localhost:3000"
    jwt_secret: str = ""
    workspace_backend: str = "local"  # "local" or "e2b"
    cache_backend: str = "sqlite"  # "sqlite" or "redis"
    redis_url: str = ""
    data_dir: str = str(PROJECT_ROOT / "data")
    sqlite_path: str = str(PROJECT_ROOT / "data" / "amphoreus.db")

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
