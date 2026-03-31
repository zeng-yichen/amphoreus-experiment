"""LinkedIn data ingestion — per-client fetch of posts/profiles.

Lighter version of Jacquard's linkedin-bank focused on per-client ingestion
rather than at-scale discovery.
"""

import logging
from typing import Any

from backend.src.core.config import get_settings

logger = logging.getLogger(__name__)


def ingest_client_posts(
    linkedin_url: str,
    max_posts: int = 50,
) -> list[dict[str, Any]]:
    """Fetch a client's recent LinkedIn posts and store in Supabase + Pinecone."""
    username = linkedin_url.rstrip("/").split("/")[-1]
    if not username:
        return []

    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()

        result = (
            sb.table("linkedin_posts")
            .select("post_text, posted_at, total_reactions, total_comments, engagement_score, hook")
            .eq("creator_username", username)
            .order("engagement_score", desc=True)
            .limit(max_posts)
            .execute()
        )

        posts = result.data or []
        logger.info("Fetched %d posts for %s from Supabase", len(posts), username)
        return posts

    except Exception as e:
        logger.warning("LinkedIn post fetch failed for %s: %s", username, e)
        return []


def ingest_client_profile(linkedin_url: str) -> dict[str, Any] | None:
    """Fetch a client's LinkedIn profile."""
    username = linkedin_url.rstrip("/").split("/")[-1]
    if not username:
        return None

    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()

        result = (
            sb.table("linkedin_profiles")
            .select("*")
            .eq("username", username)
            .limit(1)
            .execute()
        )

        return result.data[0] if result.data else None

    except Exception as e:
        logger.warning("LinkedIn profile fetch failed for %s: %s", username, e)
        return None


def embed_posts_to_pinecone(posts: list[dict[str, Any]], namespace: str = "posts") -> int:
    """Embed posts into Pinecone for semantic search."""
    settings = get_settings()
    if not settings.pinecone_api_key:
        logger.debug("Pinecone not configured — skipping embedding")
        return 0

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        oai = OpenAI(api_key=settings.openai_api_key)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index)

        vectors = []
        for i, post in enumerate(posts):
            text = post.get("post_text", "")
            if not text:
                continue
            resp = oai.embeddings.create(model="text-embedding-3-small", input=text[:8000])
            vectors.append({
                "id": f"post-{i}-{hash(text) % 10**8}",
                "values": resp.data[0].embedding,
                "metadata": {
                    "text": text[:1000],
                    "engagement": post.get("engagement_score", 0),
                },
            })

        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info("Embedded %d posts to Pinecone", len(vectors))

        return len(vectors)

    except Exception as e:
        logger.warning("Pinecone embedding failed: %s", e)
        return 0
