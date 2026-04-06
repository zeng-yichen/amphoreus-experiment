"""Post embedding cache — continuous vector representations of every scored post.

Every observation gets an embedding of its post body. These embeddings
replace discrete format_tag/topic_tag as the primary feature for the
learning pipeline's internal operations:

  - The analyst's `find_similar_posts` tool uses them for similarity search
  - The analyst's `fit_embedding_regression` tool uses them as regression features
  - The draft scorer's k-NN path uses them to score new drafts by similarity
    to historically high-performing posts

Tags (format_tag, topic_tag) remain as display-only metadata for humans.
The learning pipeline operates on embeddings. This follows the principle:
continuous over categorical, labels only for human display.

Storage: memory/{company}/post_embeddings.json
Embedding model: OpenAI text-embedding-3-small (1536 dims) — same as
alignment_scorer, transcript_scorer, and lola.

Usage:
    from backend.src.utils.post_embeddings import (
        get_post_embeddings,
        embed_text,
    )

    # Get all observation embeddings (lazy backfill on first call)
    embeddings = get_post_embeddings("innovocommerce")
    # → {"post_hash_1": [1536 floats], "post_hash_2": [1536 floats], ...}

    # Embed a new draft for comparison
    draft_emb = embed_text("Your post text here")
"""

import json
import logging
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536


def get_post_embeddings(company: str) -> dict[str, list[float]]:
    """Load the post embedding cache, backfilling any missing scored observations.

    Lazy: only embeds posts not already in the cache. First call for a client
    with 34 unembedded observations makes one batched OpenAI call (~$0.001).
    Subsequent calls return instantly from cache.

    Returns {post_hash: embedding_vector} for all scored observations.
    """
    cache = _load_cache(company)

    # Load scored observations
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return cache
    if state is None:
        return cache

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("post_hash")
        and (o.get("posted_body") or o.get("post_body"))
    ]

    # Find observations not yet embedded
    missing = [o for o in scored if o["post_hash"] not in cache]
    if not missing:
        return cache

    # Batch embed
    texts = [
        (o.get("posted_body") or o.get("post_body") or "")[:8000]
        for o in missing
    ]
    hashes = [o["post_hash"] for o in missing]

    embeddings = _embed_batch(texts)
    if embeddings and len(embeddings) == len(hashes):
        for h, e in zip(hashes, embeddings):
            cache[h] = e
        _save_cache(company, cache)
        logger.info(
            "[post_embeddings] Embedded %d new posts for %s (cache total: %d)",
            len(missing), company, len(cache),
        )

    return cache


def embed_text(text: str) -> Optional[list[float]]:
    """Embed a single text (e.g., a new draft) for similarity comparison.

    Returns None on failure. The caller should handle this gracefully.
    """
    if not text or len(text.strip()) < 10:
        return None
    results = _embed_batch([text[:8000]])
    return results[0] if results else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    import math
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(
    target_embedding: list[float],
    embeddings: dict[str, list[float]],
    top_k: int = 5,
    exclude_hashes: Optional[set] = None,
) -> list[tuple[str, float]]:
    """Find the top_k most similar post hashes by cosine similarity.

    Returns [(post_hash, similarity), ...] sorted by similarity descending.
    """
    exclude = exclude_hashes or set()
    scored = []
    for h, emb in embeddings.items():
        if h in exclude:
            continue
        sim = cosine_similarity(target_embedding, emb)
        scored.append((h, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------
# Cache I/O
# ------------------------------------------------------------------

def _cache_path(company: str):
    return vortex.memory_dir(company) / "post_embeddings.json"


def _load_cache(company: str) -> dict[str, list[float]]:
    path = _cache_path(company)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("embeddings", {})
    except Exception:
        return {}


def _save_cache(company: str, embeddings: dict[str, list[float]]) -> None:
    path = _cache_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": _EMBEDDING_MODEL,
        "dim": _EMBEDDING_DIM,
        "count": len(embeddings),
        "embeddings": embeddings,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI."""
    if not texts:
        return []
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            input=texts,
            model=_EMBEDDING_MODEL,
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.warning("[post_embeddings] Embedding failed: %s", e)
        return []
