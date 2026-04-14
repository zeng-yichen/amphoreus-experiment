"""Observation tagger — DISPLAY-ONLY metadata extraction for human dashboards.

Adds ``topic_tag``, ``source_segment_type``, and ``format_tag`` to RuanMei
observations via a single Sonnet call per post. These tags are used
EXCLUSIVELY for human-facing reporting, dashboards, and display purposes.

**IMPORTANT: No learning subsystem should depend on these tags.**

The entire learning pipeline operates on continuous post embeddings:
  - Topic transitions → embedding trajectory model (continuous directions)
  - Causal filter → PCA components of embeddings
  - Draft scorer → embedding k-NN similarity
  - Content brief → embedding clustering
  - Sequential state → embedding cosine similarity

Human-readable labels compress a 1536-dimensional embedding space into
~15 discrete categories designed by humans. This violates the Bitter
Lesson: the categories encode our theory of what distinctions matter,
not what the data reveals. The tags remain for operator convenience —
humans need interpretable labels to understand what's happening — but
the machine learning infrastructure operates in continuous space.

Usage:
    from backend.src.utils.observation_tagger import tag_post, backfill_client_tags

    # Display-only tagging:
    tags = tag_post(post_body)
    # → {"topic_tag": "regulatory compliance",
    #    "source_segment_type": "specific compliance gap",
    #    "format_tag": "case study"}

    # Batch backfill for dashboards (called from ordinal_sync):
    n = backfill_client_tags("innovocommerce")
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Sonnet for bulk tagging. Haiku is cheaper but the tags it produced were
# visibly lower-quality (over-granular topic labels, inconsistent format naming)
# and quality at the tag layer cascades into every downstream learning step
# (topic transitions, causal filter, strategy brief, segment model). Not the
# place to save pennies. Client is created lazily inside tag_post() so .env
# loading order doesn't matter.
_TAGGER_MODEL = "claude-sonnet-4-6"

_TAGGER_PROMPT_BASE = """\
Analyze this LinkedIn post and return three tags.

POST:
{post_body}
{existing_context}
Return a JSON object with exactly these keys:
- "topic_tag": 2-4 words, coarse-grained domain bucket (e.g., "clinical ai vendors", "site monitoring", "biopharma build vs buy"). Aim for a tag that could plausibly apply to 3-10 posts across a year of content, NOT a per-post unique label. Do NOT use generic labels like "business", "leadership", "productivity", "technology".
- "source_segment_type": 3-6 words describing the type of moment or material the post draws from. Examples: "specific compliance gap observed", "team anecdote about failure", "head-to-head product comparison", "personal decision under uncertainty", "observed customer behavior pattern", "contrarian industry reading".
- "format_tag": 1-2 words, coarse format label. Pick from: "storytelling", "list", "hot take", "case study", "framework", "contrarian", "data breakdown", "essay", "dialogue", "playbook", "announcement". Do NOT combine labels ("contrarian framework" → pick one).

All tags lowercase, no punctuation except spaces. Return ONLY the JSON object, no markdown, no explanation."""


def tag_post(post_body: str, existing_topics: Optional[list[str]] = None,
             existing_formats: Optional[list[str]] = None) -> Optional[dict]:
    """Extract topic_tag, source_segment_type, format_tag from a single post.

    If ``existing_topics`` or ``existing_formats`` are provided, the tagger is
    biased to reuse them when semantically appropriate. This drives convergence
    of the tag vocabulary over a client's history, which is essential for the
    topic transition model and causal filter to have repeated observations.

    Returns a dict with the three tags, or None on failure / insufficient content.
    """
    if not post_body or len(post_body.strip()) < 50:
        return None

    # Build context block that encourages tag reuse without forcing it.
    context_parts = []
    if existing_topics:
        # Show up to 15 most-used existing topic tags.
        top_topics = list(dict.fromkeys(existing_topics))[:15]
        context_parts.append(
            "\nEXISTING topic_tags already used for this client (REUSE one of these "
            "if it fits, only create a new one if the post is about genuinely different "
            "subject matter):\n  "
            + "\n  ".join(f"- {t}" for t in top_topics)
        )
    if existing_formats:
        top_formats = list(dict.fromkeys(existing_formats))[:10]
        context_parts.append(
            "\nEXISTING format_tags already used for this client (REUSE one if it fits):\n  "
            + "\n  ".join(f"- {t}" for t in top_formats)
        )
    existing_context = "\n".join(context_parts) + "\n" if context_parts else ""

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_TAGGER_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": _TAGGER_PROMPT_BASE.format(
                    post_body=post_body[:3000],
                    existing_context=existing_context,
                ),
            }],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        tags = json.loads(raw)
        if not all(k in tags for k in ("topic_tag", "source_segment_type", "format_tag")):
            return None

        return {
            "topic_tag": str(tags["topic_tag"]).strip().lower(),
            "source_segment_type": str(tags["source_segment_type"]).strip().lower(),
            "format_tag": str(tags["format_tag"]).strip().lower(),
        }
    except Exception as e:
        logger.debug("[observation_tagger] Failed to tag post: %s", e)
        return None


def backfill_client_tags(company: str, limit: int = 100) -> int:
    """Tag any scored observations in the client's state that don't have tags yet.

    Idempotent: skips observations that already have ``topic_tag`` populated.
    Capped at ``limit`` observations per call to bound LLM cost per sync cycle.

    Returns the number of observations newly tagged.
    """
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
    except Exception:
        return 0

    needs_tagging = []
    for obs in rm._state.get("observations", []):
        if obs.get("status") not in ("scored", "finalized"):
            continue
        # Idempotent: skip observations that already have ALL THREE tag fields.
        # The tagger always sets all three together in one LLM call, so if
        # topic_tag is set, the others should be too. But we check all three
        # explicitly to handle any partial-tag edge cases.
        if obs.get("topic_tag") and obs.get("format_tag") and obs.get("source_segment_type"):
            continue  # already fully tagged
        body = (obs.get("posted_body") or obs.get("post_body") or "").strip()
        if len(body) < 50:
            continue
        needs_tagging.append(obs)

    if not needs_tagging:
        return 0

    # Seed the vocabulary with any tags already present from prior runs,
    # then grow it as we tag. Tags accumulated during this run are passed
    # back into subsequent tag_post() calls so the tagger converges on a
    # compact vocabulary for this client.
    existing_topics: list[str] = [
        o.get("topic_tag") for o in rm._state.get("observations", [])
        if o.get("topic_tag")
    ]
    existing_formats: list[str] = [
        o.get("format_tag") for o in rm._state.get("observations", [])
        if o.get("format_tag")
    ]

    tagged = 0
    for obs in needs_tagging[:limit]:
        body = (obs.get("posted_body") or obs.get("post_body") or "").strip()
        tags = tag_post(body, existing_topics=existing_topics,
                        existing_formats=existing_formats)
        if tags:
            obs["topic_tag"] = tags["topic_tag"]
            obs["source_segment_type"] = tags["source_segment_type"]
            obs["format_tag"] = tags["format_tag"]
            # Grow the vocabulary so the next iteration biases toward reuse.
            existing_topics.append(tags["topic_tag"])
            existing_formats.append(tags["format_tag"])
            tagged += 1

    if tagged:
        rm._save()
        unique_topics = len({
            o.get("topic_tag") for o in rm._state.get("observations", [])
            if o.get("topic_tag")
        })
        logger.info(
            "[observation_tagger] Tagged %d observations for %s "
            "(%d unique topic tags in history)",
            tagged, company, unique_topics,
        )

    return tagged
