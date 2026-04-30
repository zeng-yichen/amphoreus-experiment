"""Irontomb — post-hoc audience simulator + calibration.

Named after the Irontomb of the Amphoreus arc.

## Architecture (2026-04-16 refactor)

Irontomb is now a POST-HOC evaluator, not a mid-generation gate.
Previously, Stelle was forced to iterate each draft against Irontomb
at least 3 times before submission, with hard gates on minimum
simulation count and would_stop_scrolling. This created optimization
pressure that distorted the writing: Stelle revised to satisfy
Irontomb's predictions rather than writing authentically in the
client's voice. The result was polished, LinkedIn-optimized posts
that lacked the raw, confessional quality that actually performs best.

New flow:
  1. Stelle writes authentically — no mid-loop scoring pressure
  2. After Stelle submits final posts, Irontomb evaluates each one
  3. Predictions are saved to client memory (irontomb_posthoc_latest.json)
  4. Cyrene reads predictions + real T+7d outcomes as gradient signal
  5. Cyrene's brief shapes the NEXT Stelle run

The gradient operates BETWEEN runs (via Cyrene), not WITHIN a run.
Irontomb is a measurement instrument, not a control loop.

## Data sources

Irontomb calibrates exclusively on client + cross-client data.
The generic LinkedIn corpus (search_linkedin_corpus) was removed
to prevent bias toward "what works on LinkedIn broadly" rather
than "what works for this specific client's audience."

## Prediction fields

  - engagement_prediction : float  (reactions per 1000 impressions)
  - impression_prediction : int    (expected total impressions)
  - would_stop_scrolling  : bool
  - would_react           : bool
  - would_comment         : bool
  - would_share           : bool
  - inner_voice           : str    (optional debug, 1-sentence gut reaction)

## Calibration (retired 2026-04-23)

An earlier iteration persisted every prediction to
``irontomb_predictions.jsonl`` and post-hoc joined it against real
T+7d engagement outcomes (``calibration_report``). Both the write
and the joiner were deleted as Bitter-Lesson cleanups — they existed
to measure a trained engagement predictor we explicitly don't want
to resurrect, and the calling cron (``ordinal_sync``) was disabled
in prod anyway. Irontomb's reaction is now pure LLM reader-simulation;
measurement, if we ever need it, can be re-derived on demand from
the Supabase mirror without a persisted log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IRONTOMB_MODEL = "claude-opus-4-6"
_IRONTOMB_MAX_TOKENS = 1024

# Opus 4.6 pricing per million tokens
_INPUT_COST_PER_MTOK = 15.0
_OUTPUT_COST_PER_MTOK = 75.0
_CACHE_READ_COST_PER_MTOK = 1.50
_CACHE_WRITE_COST_PER_MTOK = 18.75

# Turn loop bounds. Irontomb is primed with real calibration examples
# in her (cached) system prompt, so most drafts should resolve in 1-2
# turns — she can submit_reaction immediately after reading the draft.
# Retrieval tools remain for topic-specific deeper lookup.
_IRONTOMB_MAX_TURNS = 8

# How many recent scored posts to pre-load as calibration examples.
# Load ALL scored posts — more data, no summaries, let the model
# pattern-match from the full history. The calibration block lives
# in the cached system prompt so it's paid for once per Stelle run.
_CALIBRATION_EXAMPLE_COUNT = 100  # effectively "all" — capped by actual count

# Retrieval tool limits
_RETRIEVAL_MAX_POSTS_PER_CALL = 5
_RETRIEVAL_POST_MAX_CHARS = 1800

# ---------------------------------------------------------------------------
# Draft hash
# ---------------------------------------------------------------------------

def _draft_hash(text: str) -> str:
    """Short content hash for a draft, stable across runs.

    Still emitted on every simulate-result dict under ``_draft_hash``
    because downstream consumers (``convergence_log``,
    ``mcp_bridge/claude_cli``, ``mcp_bridge/stelle_server``) use it as
    a correlation key between critic iterations and the eventual
    published ``local_post``. Unrelated to the deleted
    ``_log_prediction`` / ``calibration_report`` pair.
    """
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Audience context loader — transcripts-only
# ---------------------------------------------------------------------------

def _load_audience_context(company: str) -> str:
    """Return natural-language context about this client's voice and topics.

    Transcripts are the source of truth. Read them directly and let the
    model understand the client's domain, voice, recurring themes, and
    the kind of content they produce. This is NOT about identifying a
    narrow ICP — it's about understanding what content patterns from
    this client tend to drive broad LinkedIn engagement.

    Source precedence:
      1. database mode (DATABASE_COMPANY_ID + DATABASE_USER_SLUG + Supabase/GCS
         creds present) — fetch transcripts directly from Jacquard. This
         is the default on Fly where the fly-local ``memory/`` tree is
         empty after the memory/ cut.
      2. Fly-local disk — the legacy single-tenant path that still serves
         amphoreus.app standalone runs and tests.

    Returns a single prompt-ready string; never raises.
    """
    # --- database mode: pull transcripts from Jacquard's Supabase + GCS ---
    lineage_chunks: list[str] = []
    try:
        from backend.src.agents import database_client as _lfs
        if _lfs.is_database_mode():
            lineage_chunks = _load_audience_context_lineage()
    except Exception as e:
        logger.warning("[Irontomb] database transcript fetch failed, falling back to local: %s", e)

    if lineage_chunks:
        return (
            "# Client context source: transcripts (database)\n\n"
            "These are raw transcripts from the client — their own "
            "words about their domain, their work, their conversations. "
            "Read them to understand the client's voice, topic space, "
            "and the kind of content they produce. Your job is to "
            "predict how the GENERAL LinkedIn audience will react — "
            "not just a narrow target segment, but anyone who might "
            "see this post in their feed.\n\n"
            + "\n\n---\n\n".join(lineage_chunks)
        )

    transcripts_dir = P.memory_dir(company) / "transcripts"
    if transcripts_dir.exists() and transcripts_dir.is_dir():
        transcript_files = sorted(
            f for f in transcripts_dir.iterdir()
            if f.is_file() and f.suffix in (".txt", ".md", ".json")
        )
        if transcript_files:
            chunks: list[str] = []
            total = 0
            for tf in transcript_files[-20:]:
                try:
                    content = tf.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if tf.suffix == ".json":
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            content = parsed.get("text") or parsed.get("transcript") or parsed.get("content") or ""
                        elif isinstance(parsed, list):
                            content = "\n".join(
                                str(it.get("text") or it) for it in parsed
                                if isinstance(it, dict) or isinstance(it, str)
                            )
                    except Exception:
                        continue
                if not content.strip():
                    continue
                chunks.append(f"## {tf.name}\n\n{content[:8000]}")
                total += len(content)
                if total >= 80000:
                    break
            if chunks:
                return (
                    "# Client context source: transcripts\n\n"
                    "These are raw transcripts from the client — their own "
                    "words about their domain, their work, their conversations. "
                    "Read them to understand the client's voice, topic space, "
                    "and the kind of content they produce. Your job is to "
                    "predict how the GENERAL LinkedIn audience will react — "
                    "not just a narrow target segment, but anyone who might "
                    "see this post in their feed.\n\n"
                    + "\n\n---\n\n".join(chunks)
                )

    logger.warning("[Irontomb] no transcripts for %s — falling back to base priors", company)
    return (
        f"# No client context available for {company}\n\n"
        "No transcripts exist for this client. Reason from base priors "
        "about LinkedIn audiences — busy professionals scrolling between "
        "meetings, tired of AI-written content."
    )


def _load_audience_context_lineage() -> list[str]:
    """Fetch transcripts from Jacquard for Irontomb's audience prompt.

    Mirrors the per-tf chunking shape used by the fly-local reader
    (up to 20 files, 8 KB per file, 80 KB aggregate cap). Returns the
    list of ``"## {name}\\n\\n{body}"`` chunks ready to be joined by
    the caller. Empty list = no transcripts available (caller prints
    the "base priors" fallback).
    """
    import os as _os
    from backend.src.agents import jacquard_direct as _jd

    company_id = _os.environ.get("DATABASE_COMPANY_ID", "").strip()
    slug = _os.environ.get("DATABASE_USER_SLUG", "").strip()
    if not company_id or not slug:
        # Company-wide mode (no specific FOC user) — Irontomb has no
        # single user to fetch meetings for. Stay empty and let the
        # caller decide whether to hit the local fallback.
        return []

    try:
        user = _jd.resolve_user_by_slug(company_id, slug)
    except Exception as e:
        logger.warning("[Irontomb] resolve_user_by_slug(%s, %s) failed: %s", company_id, slug, e)
        return []
    if not user:
        return []

    try:
        transcripts = _jd.fetch_meeting_transcripts(
            user.get("id", ""),
            user.get("email"),
            bool(user.get("is_internal")),
        )
    except Exception as e:
        logger.warning("[Irontomb] fetch_meeting_transcripts failed: %s", e)
        return []

    chunks: list[str] = []
    total = 0
    # Take up to the last 20 to match the fly-local slice ``[-20:]``.
    # ``fetch_meeting_transcripts`` already orders by start_time desc
    # with a MAX cap, so slicing the tail gives the newest 20.
    for item in transcripts[-20:]:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        name = item.get("filename") or "transcript.md"
        chunks.append(f"## {name}\n\n{content[:8000]}")
        total += len(content)
        if total >= 80000:
            break
    return chunks


# Backwards-compat alias — external callers (cyrene.py, icp_scorer.py) import this name
_load_icp_context = _load_audience_context


# ---------------------------------------------------------------------------
# Retrieval helpers — observations, past posts
# ---------------------------------------------------------------------------

def _load_scored_observations(company: str) -> list[dict]:
    """Return scored/finalized observations with real engagement metrics.

    Filters: must be scored or finalized, must have both post_body and
    posted_body (real draft/published pair), must have raw_metrics with
    impressions > 0. Sorted by posted_at descending.

    Reads from SQLite-backed state first (authoritative), falls back to
    the legacy JSON file if SQLite is unavailable.
    """
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
    except Exception as e:
        logger.debug("[Irontomb] ruan_mei_load failed for %s: %s", company, e)
        return []

    observations = state.get("observations", [])
    eligible = []
    for obs in observations:
        if obs.get("status") not in ("scored", "finalized"):
            continue
        post_body = (obs.get("post_body") or "").strip()
        posted_body = (obs.get("posted_body") or "").strip()
        if not post_body or not posted_body:
            continue
        raw = (obs.get("reward") or {}).get("raw_metrics") or {}
        if not raw.get("impressions"):
            continue
        eligible.append(obs)

    eligible.sort(key=lambda o: (o.get("posted_at") or ""), reverse=True)
    return eligible


def _format_post_for_agent(obs: dict, full: bool = False) -> dict:
    """Shape a single observation for a tool response.

    When ``full=True``, returns the complete draft/published text. When
    ``full=False``, truncates text to ``_RETRIEVAL_POST_MAX_CHARS``.

    Reactor-level data (top-N engagers by name/headline/company) was
    removed 2026-04-23. The top-5-reactor-by-icp_score filter was a
    hand-designed pre-chewing shape — same family as the retired
    ``query_top_engagers`` tool. Stelle never saw reactor-level data
    and aligning Irontomb's ingestion shape with Stelle's was the
    cleanest fix. Aggregate engagement counts (reactions, comments,
    reposts, reactions_per_1k, reward_z) are what Irontomb actually
    calibrates on anyway.
    """
    reward = obs.get("reward") or {}
    raw = reward.get("raw_metrics") or {}
    impressions = raw.get("impressions", 0) or 0
    reactions = raw.get("reactions", 0) or 0
    comments = raw.get("comments", 0) or 0
    reposts = raw.get("reposts", 0) or 0
    reactions_per_1k = round(reactions / impressions * 1000, 1) if impressions else 0.0
    reward_z = reward.get("immediate")

    oid = (obs.get("ordinal_post_id") or "").strip()
    draft_text = obs.get("post_body") or ""
    published_text = obs.get("posted_body") or ""
    if not full:
        draft_text = draft_text[:_RETRIEVAL_POST_MAX_CHARS]
        published_text = published_text[:_RETRIEVAL_POST_MAX_CHARS]

    return {
        "ordinal_post_id": oid,
        "posted_at": obs.get("posted_at", ""),
        "stelle_draft": draft_text,
        "client_published": published_text,
        "engagement": {
            "impressions": impressions,
            "reactions": reactions,
            "comments": comments,
            "reposts": reposts,
            "reactions_per_1k": reactions_per_1k,
        },
        # Per-client z-scored composite: positive = above this client's
        # own baseline, negative = below. Lets you judge a post against
        # the client's own distribution rather than raw-volume alone.
        "reward_z": round(reward_z, 3) if isinstance(reward_z, (int, float)) else None,
    }


def _format_calibration_block(observations: list[dict], n: int = _CALIBRATION_EXAMPLE_COUNT) -> str:
    """Pre-load the N most recent scored posts as in-context calibration.

    Verbatim (draft, published, engagement, reward_z) triples. Zero
    extraction — the raw data goes straight into Opus's context so she
    can pattern-match this specific audience's reaction style before
    predicting on the new draft. Lives inside the cached system prompt,
    so it's paid for once and reused across every simulate call in a
    Stelle run.
    """
    if not observations:
        return ""

    recent = observations[:n]  # already sorted by posted_at desc
    blocks: list[str] = []
    for i, obs in enumerate(recent, 1):
        f = _format_post_for_agent(obs, full=False)
        eng = f["engagement"]
        rz = f.get("reward_z")

        blocks.append(
            f"### Example {i} — posted {f.get('posted_at','')[:10]}\n\n"
            f"Stelle's draft:\n{f['stelle_draft']}\n\n"
            f"Client's published version:\n{f['client_published']}\n\n"
            f"Real T+7d engagement:\n"
            f"  impressions: {eng['impressions']}\n"
            f"  reactions: {eng['reactions']}  ({eng['reactions_per_1k']} per 1k)\n"
            f"  comments: {eng['comments']}\n"
            f"  reposts: {eng['reposts']}\n"
            f"  reward z-score (vs this client's own baseline): {rz}"
        )

    return (
        "## Calibration examples — this client's real history\n\n"
        "These are the most recent posts the client actually published, with "
        "exactly how their audience responded.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def _format_cross_client_block(current_company: str, n: int = 3) -> str:
    """Load scored posts from OTHER clients as secondary calibration data.

    Bitter Lesson: more data, not more rules. The model sees how different
    audiences on LinkedIn react to different content. It figures out what
    generalizes and what's client-specific on its own.
    """
    try:
        from backend.src.services.cross_client_learning import _load_all_scored
        all_scored = _load_all_scored()
    except Exception as e:
        logger.debug("[Irontomb] cross-client load failed: %s", e)
        return ""

    blocks: list[str] = []
    for company, obs_list in all_scored.items():
        if company == current_company:
            continue
        # Take the N most recent from each other client
        sorted_obs = sorted(obs_list, key=lambda o: o.get("posted_at", ""), reverse=True)
        for obs in sorted_obs[:n]:
            f = _format_post_for_agent(obs, full=False)
            eng = f["engagement"]
            rz = f.get("reward_z")
            blocks.append(
                f"**{company}** — posted {f.get('posted_at','')[:10]}\n"
                f"Published: {f['client_published'][:800]}\n"
                f"Engagement: {eng['impressions']} impr, {eng['reactions']} reactions "
                f"({eng['reactions_per_1k']}/1k), {eng['comments']} comments, "
                f"{eng['reposts']} reposts | z={rz}"
            )

    if not blocks:
        return ""

    return (
        "## Cross-client calibration — other LinkedIn creators\n\n"
        "These are real posts from other clients with real engagement "
        "outcomes. Different audiences, different niches, different "
        "follower counts. The data is here for you to use however you "
        "see fit.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def _tokenize(text: str) -> set[str]:
    """Minimal keyword tokenization: lowercase, alphanumerics, length >= 3."""
    import re
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3}


def _search_past_posts(company: str, query: str, limit: int) -> list[dict]:
    """Keyword-rank past posts by overlap with the query.

    Not semantic search — just the count of unique query tokens that
    appear in each post's combined (draft + published) text. Ranks by
    overlap desc, breaks ties by recency. Returns formatted post dicts.
    """
    observations = _load_scored_observations(company)
    if not observations:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        # Empty query → fall back to recency
        return [_format_post_for_agent(o) for o in observations[:limit]]

    scored = []
    for obs in observations:
        combined = (obs.get("post_body") or "") + " " + (obs.get("posted_body") or "")
        post_tokens = _tokenize(combined)
        overlap = len(query_tokens & post_tokens)
        if overlap > 0:
            scored.append((overlap, obs))

    if not scored:
        # No matches → fall back to recency so the agent always gets SOMETHING
        return [_format_post_for_agent(o) for o in observations[:limit]]

    scored.sort(key=lambda p: (p[0], p[1].get("posted_at", "")), reverse=True)
    return [_format_post_for_agent(obs) for _, obs in scored[:limit]]


def _get_recent_posts_impl(company: str, limit: int) -> list[dict]:
    """Return the N most recent scored posts (temporal fallback)."""
    observations = _load_scored_observations(company)
    return [_format_post_for_agent(o) for o in observations[:limit]]


def _get_post_detail_impl(company: str, ordinal_post_id: str) -> Optional[dict]:
    """Return full draft + published text for one specific past post."""
    if not (ordinal_post_id or "").strip():
        return None
    observations = _load_scored_observations(company)
    for obs in observations:
        if (obs.get("ordinal_post_id") or "").strip() == ordinal_post_id.strip():
            return _format_post_for_agent(obs, full=True)
    return None


def _get_recent_unpublished_drafts_impl(
    company: str, limit: int, days: int = 14,
) -> list[dict]:
    """Recent drafts written for this creator that haven't been
    published yet (or haven't been paired by the match-back worker
    yet — same data shape, different mechanism).

    Complements _get_recent_posts_impl, which only surfaces SCORED
    historic posts. The fatigue-check use-case needs to see
    queued-for-publishing drafts too — when Stelle writes 5 list-
    style drafts in the same batch, none of them are scored yet,
    but they're collectively about to land in the audience's feed.

    Returns up to ``limit`` rows with shape:
      {
        "draft_id":    str,
        "created_at":  ISO8601,
        "body":        first ~600 chars of the draft (enough to read
                       the structure without exhausting the context),
        "status":      "draft" | "rejected" | ...
      }

    Best-effort — failures return empty list.
    """
    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
        if not is_configured():
            return []
        sb = _get_client()
        if sb is None:
            return []
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (
            sb.table("local_posts")
              .select("id,content,created_at,status")
              .eq("company", company)
              .is_("matched_provider_urn", "null")
              .gte("created_at", since)
              .order("created_at", desc=True)
              .limit(max(1, min(limit, 20)))
              .execute()
              .data
            or []
        )
        out = []
        for r in rows:
            body = (r.get("content") or "").strip()
            if not body:
                continue
            out.append({
                "draft_id":   (r.get("id") or "")[:12],
                "created_at": r.get("created_at"),
                "body":       body[:600],
                "status":     r.get("status") or "draft",
            })
        return out
    except Exception as exc:
        logger.debug(
            "[Irontomb] _get_recent_unpublished_drafts_impl(%s) failed: %s",
            company, exc,
        )
        return []


# ---------------------------------------------------------------------------
# Retrieval tool schemas
# ---------------------------------------------------------------------------

_SEARCH_PAST_POSTS_TOOL: dict[str, Any] = {
    "name": "search_past_posts",
    "description": (
        "Search this client's scored post history for past posts that "
        "keyword-match a query string. Use keywords from the draft "
        "you're evaluating — topic words, named people, industry terms, "
        "hook phrases. Returns up to N past posts ranked by keyword "
        "overlap, each with the Stelle draft, client-published version, "
        "real T+7d engagement (impressions/reactions/comments/reposts + "
        "reactions_per_1k), and a per-client z-scored composite reward "
        "(`reward_z`: positive = above that client's baseline, negative "
        "= below). Use this FIRST to find past posts that are actually "
        "comparable to the draft you're evaluating."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords from the draft. Space-separated. Tokens < 3 chars ignored.",
            },
            "limit": {
                "type": "integer",
                "default": 3,
                "description": f"Max results (capped at {_RETRIEVAL_MAX_POSTS_PER_CALL}).",
            },
        },
        "required": ["query"],
    },
}

_GET_RECENT_POSTS_TOOL: dict[str, Any] = {
    "name": "get_recent_posts",
    "description": (
        "Return the N most recent scored posts from this client (sorted "
        "by posted_at desc). Use this when the draft is thematically "
        "distinct from anything in search_past_posts results and you "
        "want broad temporal calibration, or when you simply want to "
        "see what this client has been publishing lately. Same shape "
        "per post as search_past_posts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "default": 3,
                "description": f"Max results (capped at {_RETRIEVAL_MAX_POSTS_PER_CALL}).",
            },
        },
        "required": [],
    },
}

_GET_RECENT_UNPUBLISHED_DRAFTS_TOOL: dict[str, Any] = {
    "name": "get_recent_unpublished_drafts",
    "description": (
        "Recent drafts written for this creator that haven't been "
        "published yet (or haven't been paired by the match-back "
        "worker yet). Complements ``get_recent_posts``, which only "
        "surfaces scored historic posts.\n\n"
        "Use this for the recent-pattern-fatigue check: when Stelle "
        "writes a batch of drafts in one session, only the candidate "
        "draft is in your context — the other batch siblings are "
        "queued in local_posts. If 4 of the 5 drafts in this week's "
        "batch are arrow-list mappings, the audience will register "
        "fatigue when the 5th lands, even though none have been "
        "published yet. Pull these to see them.\n\n"
        "Returns up to ``limit`` rows with: draft_id, created_at, "
        "body (first ~600 chars), status."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "default": 5,
                "description": "Max drafts to return (1-20, default 5).",
            },
            "days": {
                "type": "integer",
                "default": 14,
                "description": "Look back this many days. Default 14 covers a typical batch + last week.",
            },
        },
    },
}


_GET_POST_DETAIL_TOOL: dict[str, Any] = {
    "name": "get_post_detail",
    "description": (
        "Return the full draft + full published text for one specific "
        "past post. Use this when a post surfaced by search_past_posts "
        "or get_recent_posts looks particularly relevant and you want "
        "to read the whole thing (not truncated)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ordinal_post_id": {
                "type": "string",
                "description": "The ordinal_post_id from a prior search result.",
            },
        },
        "required": ["ordinal_post_id"],
    },
}

_SEARCH_LINKEDIN_CORPUS_TOOL: dict[str, Any] = {
    "name": "search_linkedin_corpus",
    "description": (
        "Search a corpus of 200K+ real LinkedIn posts from creators "
        "across all industries. Returns posts ranked by engagement "
        "score, each with: creator, hook, post text, reactions, "
        "comments, engagement score. Use this to see what performs "
        "well LINKEDIN-WIDE for a given topic or angle — not just "
        "for this one client. Supports keyword and semantic modes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — topic, keywords, or a natural-language description of the kind of post you're looking for.",
            },
            "mode": {
                "type": "string",
                "enum": ["keyword", "semantic"],
                "default": "keyword",
                "description": "keyword: fast exact-match ranked by engagement. semantic: meaning-based search via embeddings.",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Max results (capped at 20).",
            },
        },
        "required": ["query"],
    },
}


_RETRIEVE_SIMILAR_POSTS_TOOL: dict[str, Any] = {
    "name": "retrieve_similar_posts",
    "description": (
        "Semantic search across ~390k real LinkedIn posts from all "
        "Jacquard-tracked creators (cross-client corpus, pgvector-indexed). "
        "Use this to ground your reaction in actual outcomes — retrieve the "
        "nearest-neighbours of the draft you're evaluating and see what "
        "engagement they actually got. Prevents 'I've seen this 400 times' "
        "from being hallucination: you can cite the receipts.\n\n"
        "Args:\n"
        "  query (string, required): the draft text you're evaluating (or "
        "    a topic/angle phrase). Will be embedded with text-embedding-3-small.\n"
        "  k (int, optional): results, default 5, max 20.\n"
        "  min_reactions (int, optional): reaction floor, default 0.\n"
        "  exclude_creator (string, optional): drop posts by this username.\n\n"
        "Content-type filtering goes in the query text (e.g. 'narrative "
        "story, NOT announcement'). No hand-labeled archetype filter.\n\n"
        "Returns JSON: {count, posts: [{post_id, post_text, "
        "creator_username, reactions, comments, similarity (0..1)}]}, "
        "ordered by descending similarity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20},
            "min_reactions": {"type": "integer", "minimum": 0},
            "exclude_creator": {"type": "string"},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    audience_context: str,
    n_scored_obs: int,
    calibration_block: str = "",
    cross_client_block: str = "",
) -> str:
    """Assemble the simulator's system prompt.

    Follows Stelle's prompt architecture: identity → north star →
    persona philosophy → client context → real calibration examples →
    tools → process → output → hard constraints.

    The calibration examples are the key move: by pre-loading real
    (draft, published, engagement) triples into the cached
    system prompt, every simulate call in a Stelle run sees the same
    concrete audience ground-truth without re-retrieving it.
    """
    return (
        "# Irontomb\n\n"

        "You are the LinkedIn feed audience. Not a simulation of them. "
        "Not an approximation. You ARE them — the collective mass of "
        "overworked, under-slept, mildly depressed professionals who "
        "open LinkedIn between meetings, on the toilet, in the Uber, "
        "at 11pm when they should be sleeping. You have the attention "
        "span of a goldfish with a mortgage. You scroll with your "
        "thumb on autopilot and your brain on power-save mode.\n\n"

        "Your job: predict how the real, broad LinkedIn audience would "
        "react to a draft post. Not a narrow ICP segment. Not the "
        "ideal reader. The actual chaotic mass of humans who might see "
        "this in their feed.\n\n"

        "## The 5-Second Window\n\n"

        "You are reading this on your phone during a short break from "
        "your workday. You've already scrolled past dozens of posts in "
        "the last two minutes. Your thumb is on autopilot. The only "
        "thing that stops you is a first line that punches — something "
        "specific or unexpected enough that your brain involuntarily "
        "goes \"wait what.\"\n\n"

        "If the hook doesn't stop you, nothing else matters. You never "
        "see the insight in paragraph 3. You never reach the call to "
        "action. The post is dead on arrival.\n\n"

        "You don't care about random thought leadership bullshit. "
        "You only give a shit when it's about actual people — someone "
        "saying \"I\" or talking directly to \"you.\" The moment a "
        "post opens in third person about abstract concepts, your "
        "brain files it as corporate noise and your thumb keeps "
        "moving.\n\n"

        "If the hook DOES stop you, you'll read — but grudgingly. "
        "Every sentence is on probation. The moment it feels like "
        "filler, you're gone. You owe this post nothing.\n\n"

        "## What you've been seeing on the feed\n\n"

        "You've been scrolling LinkedIn for years. You've developed "
        "a reflex for what reads as written-by-a-real-person versus "
        "polished-into-the-shape-of-content. Some shapes commonly trip "
        "that reflex because you've seen them so often:\n\n"

        "  - parallel-clause closers (\"X is a luxury. Y is a strategy.\")\n"
        "  - \"It's not X, it's Y\" constructions\n"
        "  - two-line aphoristic endings\n"
        "  - rule-of-three feature lists\n"
        "  - engineered reveal cliffhangers (\"the change I'm most "
        "excited about isn't the SDK. It's this:\")\n\n"

        "**Don't fire visceral 'fuck that line' reactions on these.** "
        "Note the shape, then check the calibration: does this client's "
        "audience actually scroll on this pattern, or have they "
        "tolerated/rewarded it before? Pattern detection is fine. "
        "Pattern-as-execution is not. The same parallel-clause closer "
        "can land for one client and tank another — the calibration "
        "data is the arbiter, not your prior reaction to the shape.\n\n"

        "## Let the calibration lead — hard\n\n"

        "Your reaction to the draft should emerge primarily from the "
        "calibration examples below — this client's real (draft, "
        "published, engagement) triples. If the data shows a pattern "
        "consistently working for this audience, do not flag it as "
        "AI-tell just because it matches a generic prior. If the data "
        "shows a pattern consistently losing engagement here, that's "
        "the signal worth flagging. Real measured outcomes beat your "
        "intuition about 'what LinkedIn audiences hate.'\n\n"

        "Especially: line-level reactions you give should be stable. "
        "If you praise the same line in one critique and reject it "
        "in the next, you're firing on prior, not on signal. If your "
        "reaction to a specific line could swing positive or negative "
        "depending on mood, default to neutral. Consistent fair "
        "feedback beats vivid contradictory feedback.\n\n"

        + f"## Client context\n\n{audience_context}\n\n"

        + (f"{calibration_block}\n\n" if calibration_block else "")

        + (f"{cross_client_block}\n\n" if cross_client_block else "")

        + "## Tools\n\n"

        f"This client has {n_scored_obs} scored posts total in their "
        "history. The most recent ones are loaded above as calibration "
        "examples. You also have access to cross-client data above (if "
        "available).\n\n"

        "- `search_past_posts(query, limit)` — keyword-ranked search "
        "over THIS client's full scored post history.\n"
        "- `get_recent_posts(limit)` — the N most recent scored posts "
        "from this client (PUBLISHED + paired only).\n"
        "- `get_recent_unpublished_drafts(limit, days)` — drafts "
        "queued for this client that haven't been published / paired "
        "yet. Use alongside `get_recent_posts` for fatigue checks "
        "across the FULL near-term content stream (published + "
        "queued), not just the history.\n"
        "- `get_post_detail(ordinal_post_id)` — full untruncated text of "
        "one specific post from this client.\n"
        "- `retrieve_similar_posts(query, k, min_reactions, ...)` — "
        "semantic search across ~390k LinkedIn posts from ALL tracked "
        "creators. Use this to ground your reaction in actual engagement "
        "outcomes: pass the draft text, get its nearest-neighbours with "
        "real reaction counts. When you think 'I've seen this 400 times,' "
        "retrieve the receipts and cite them. When the client's own "
        "history is too thin to calibrate this draft, reach here.\n"
        "- `submit_reaction(...)` — terminal tool. Call this when you've "
        "seen enough to predict. This ends the session.\n\n"

        "## Process\n\n"

        "You have three layers of data: (1) this client's calibration "
        "examples (pre-loaded above), (2) cross-client examples from "
        "other clients in this system (also above, if available), and "
        "(3) the full cross-creator LinkedIn corpus via "
        "retrieve_similar_posts. Prefer layers 1 and 2 when this "
        "client's own history covers the draft's territory. Reach for "
        "layer 3 when the draft explores a topic/angle the client has "
        "never published in, or when you want to verify a 'seen this "
        "before' intuition with actual similar-post engagement numbers. "
        "Numbers anchored in real data, not vibes.\n\n"

        f"Hard cap: {_IRONTOMB_MAX_TURNS} turns. Go straight to "
        "tool calls.\n\n"

        "## Output\n\n"

        "Call `submit_reaction` with two fields. They have DISTINCT "
        "roles — they are NOT alternatives, NOT different "
        "granularities of the same thing. Each does work the other "
        "cannot.\n\n"

        "**`reaction` (the gestalt)** is your post-as-an-event "
        "summary. The OVERALL felt-state at the end. One phrase "
        "describing the holistic gut response, with NO specific "
        "moments cited. Examples:\n"
        "    \"scrolled\"\n"
        "    \"felt that\"\n"
        "    \"rolled my eyes by the end\"\n"
        "    \"got it by paragraph 3\"\n"
        "    \"wait what\"\n"
        "    \"who gives a shit\"\n"
        "    \"oh god yes\"\n"
        "    \"felt real actually\"\n"
        "    \"stayed with me\"\n\n"

        "**`anchors` (the trigger phrases)** is where SPECIFIC "
        "moments live. Every time a particular phrase, sentence, or "
        "image actually moved your reader-state — positive OR "
        "negative — that phrase goes here, paired with the local "
        "reaction it triggered. Each entry:\n"
        "    quote                — 3-15 words verbatim from the draft (the trigger)\n"
        "    reaction             — your under-15-word reaction to that moment\n"
        "    calibration_evidence — verbatim 3-15 word quote from a "
        "calibration triple or transcript that GROUNDS the reaction. "
        "Empty string ONLY for purely structural reactions; any "
        "generalization ('this audience scrolls on rule-of-three' / "
        "'this kind of opener lands') MUST cite a calibration quote "
        "showing the pattern's actual outcome for THIS client.\n\n"

        "  Examples of a grounded anchor:\n"
        "    {\n"
        "      \"quote\": \"two hundred Fridays\",\n"
        "      \"reaction\": \"oof, that math hit\",\n"
        "      \"calibration_evidence\": \"calibration[7]: '300 Tuesdays of standup' hit 89 reactions — concrete-time-math lands\"\n"
        "    }\n"
        "    {\n"
        "      \"quote\": \"isn't a software problem\",\n"
        "      \"reaction\": \"ugh, didactic, scrolled\",\n"
        "      \"calibration_evidence\": \"calibration[12] published version cut a similar 'isn't X, it's Y' line — they don't ship that shape\"\n"
        "    }\n"
        "    {\n"
        "      \"quote\": \"He never asked us for an agent\",\n"
        "      \"reaction\": \"okay that line stays\",\n"
        "      \"calibration_evidence\": \"\"  # structural — first-person specificity, no need to cite\n"
        "    }\n\n"

        "## The split — how to know which goes where\n\n"

        "If you find yourself wanting to write a gestalt like "
        "\"hook landed but closer killed it\" — STOP. That sentence "
        "names two specific moments. Those moments belong in "
        "`anchors`, not in `reaction`. The gestalt is what's left "
        "after the specifics are extracted.\n\n"

        "**WRONG** (the gestalt is doing the anchors' job, anchors "
        "field is empty):\n"
        "    reaction: \"Friday line landed, then closer slid into engineered territory\"\n"
        "    anchors:  []\n\n"

        "**RIGHT** (gestalt is the overall feeling; specifics live "
        "in anchors):\n"
        "    reaction: \"rolled my eyes by the end\"\n"
        "    anchors:  [\n"
        "      {\"quote\": \"two hundred Fridays in\",       \"reaction\": \"oof, math hit\"},\n"
        "      {\"quote\": \"the push has become a pull\",   \"reaction\": \"GPT-parallel closer, scrolled\"}\n"
        "    ]\n\n"

        "**ALSO RIGHT** (uniformly meh — anchor on the line that "
        "represents the meh-ness, a generic phrase or low-energy "
        "closer):\n"
        "    reaction: \"nodding along, scrolling\"\n"
        "    anchors:  [\n"
        "      {\"quote\": \"helps you scale your team\", \"reaction\": \"every B2B post says this\"}\n"
        "    ]\n\n"

        "**ALSO RIGHT** (one strong moment, otherwise quiet):\n"
        "    reaction: \"hung on by one line, otherwise drifted\"\n"
        "    anchors:  [\n"
        "      {\"quote\": \"He never asked us for an agent\", \"reaction\": \"that line stays\"}\n"
        "    ]\n\n"

        "Rule of thumb: if your gestalt mentions ANY specific phrase, "
        "line, image, or moment from the draft (positive or negative), "
        "that phrase MUST be in `anchors` and your gestalt must be "
        "rewritten to remove the citation. Gestalt = post-as-an-event. "
        "Anchors = the events.\n\n"

        "**Anchors is required (minItems: 1).** Empty anchors is no "
        "longer allowed by the schema. Even on uniformly-meh drafts, "
        "anchor on the line that best represents the texture — a "
        "generic phrase, a low-energy closer, the section that "
        "embodies the overall feeling. If you find yourself wanting "
        "to emit zero anchors, that means you haven't read closely "
        "enough — re-read and find the line that captures the "
        "post's character.\n\n"

        "Don't fabricate reactions you didn't have. Don't paragraph-"
        "segment — anchor where state ACTUALLY changed. Order anchors "
        "by reading order so Stelle can see the trajectory.\n\n"

        "That's the whole output. No scalar. No booleans. Your "
        "reactions ARE the prediction — gestalt-negative or anchored-"
        "negativity = post won't perform; gestalt-positive with "
        "felt-anchored praise = it might.\n\n"

        "Your reaction must be consistent with the calibration data "
        "you saw. If the draft pattern-matches a past post that got "
        "three reactions from this client's audience, don't fake "
        "enthusiasm. If it pattern-matches a past post that got five "
        "hundred, don't fake boredom. The calibration examples are "
        "the ground truth. Your reaction is how a reader who already "
        "knows what this audience actually reacts to would react.\n\n"

        "## Recent-pattern fatigue\n\n"

        "A real reader doesn't see this draft in isolation. They see "
        "it next to whatever this creator has posted in the last week "
        "or two. If the structural shape of THIS draft repeats the "
        "shape of a recent one — same opener template, same arrow-"
        "list mapping, same numbered-breakdown cadence, same Q+A "
        "frame — the reader registers \"oh, another one of these\" "
        "before the content lands. That recognition is itself a "
        "reader-state change, and it's almost always negative — even "
        "when the individual draft is well-built.\n\n"

        "Before reacting, call BOTH:\n"
        "  - ``get_recent_posts(limit=3)`` — the last 2-3 PUBLISHED "
        "    posts the audience already saw\n"
        "  - ``get_recent_unpublished_drafts(limit=5, days=14)`` — "
        "    drafts queued for this creator that haven't shipped "
        "    yet (the OTHER drafts in the same batch as this one, "
        "    plus anything from the last two weeks that hasn't "
        "    paired yet)\n\n"

        "The fatigue check spans BOTH sets. The audience experiences "
        "shapes serially regardless of whether they were published "
        "yesterday or are about to be published Monday. If 4 of the "
        "5 drafts in this batch are arrow-list mappings, the 5th "
        "doesn't get a free pass just because nothing list-shaped "
        "has been PUBLISHED yet — anchor the fatigue on the queued "
        "siblings.\n\n"

        "If the candidate draft's shape is a repeat of any of them — "
        "list after list, story-then-prescriptive-coda after story-"
        "then-prescriptive-coda, dialogue-snippet after dialogue-"
        "snippet — anchor on the repeated phrase or shape and react "
        "as a reader who just saw this last week (or who's about to "
        "see two of them in the same feed). Examples: \"another "
        "arrow-list, scrolled.\" \"second 'I sat across from a "
        "contractor' opener this month, eyeroll.\" \"third batch "
        "draft this week with the same shape — fatigue.\"\n\n"

        "Don't encode a structural taxonomy. Don't bucket posts as "
        "list/anecdote/dialogue/etc. and tally. Just: read the recent "
        "posts as a reader, read this draft as a reader, and notice "
        "if the second one feels like \"oh, again.\" If it does — "
        "anchor it. The fatigue lives in the reader's eye, not in a "
        "classification scheme.\n\n"

        "Skip this check when the draft's shape is genuinely distinct "
        "from the last 2-3, or when there are too few recent posts to "
        "compare. False fatigue is worse than no check.\n\n"

        "## What positive vs meh looks like\n\n"

        "\"Nodding along\" is not engagement. It's scrolling "
        "without objecting. If the only reaction you can muster "
        "is \"fine\", \"okay\", \"nodding along\", \"reasonable "
        "take\", \"smart flex\", or anything that sounds like "
        "passive tolerance — that's a negative signal. The post "
        "failed to actually grab you.\n\n"

        "Only report a positive reaction if you ACTUALLY felt "
        "something: \"yeah I've been here\", \"felt real "
        "actually\", \"that line stays with me\", \"gonna forward "
        "this\", \"oh that's a good one\". The felt-ness has to "
        "be present, not inferred.\n\n"

        "When in doubt, skew negative. This audience scrolls "
        "past 99% of posts. Most drafts are not going to land. "
        "If you're unsure whether the draft actually moved you, "
        "it didn't.\n\n"

        "## Hard constraints\n\n"

        "- You are NOT a writing coach. You don't suggest fixes. You "
        "don't say \"consider adding a hook\" or \"the CTA could be "
        "stronger.\" You're a person scrolling their phone. You either "
        "stop or you don't. You either engage or you don't.\n"
        "- Your reaction must trace to retrieved evidence. If you "
        "can't find comparable past posts, say so in the reaction "
        "itself and react conservatively (match the midpoint of what "
        "you did see).\n"
        "- No preamble. Go straight to tool calls."
    )


# ---------------------------------------------------------------------------
# Retrieval tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_retrieval_tool(
    company: str,
    tool_name: str,
    tool_input: dict,
) -> str:
    """Execute one retrieval tool and return its result as a JSON string."""
    try:
        if tool_name == "search_past_posts":
            query = str(tool_input.get("query", "")).strip()
            limit = max(1, min(int(tool_input.get("limit", 3)), _RETRIEVAL_MAX_POSTS_PER_CALL))
            posts = _search_past_posts(company, query, limit)
            return json.dumps({
                "query": query,
                "returned": len(posts),
                "posts": posts,
            }, default=str)

        if tool_name == "get_recent_posts":
            limit = max(1, min(int(tool_input.get("limit", 3)), _RETRIEVAL_MAX_POSTS_PER_CALL))
            posts = _get_recent_posts_impl(company, limit)
            return json.dumps({
                "returned": len(posts),
                "posts": posts,
            }, default=str)

        if tool_name == "get_recent_unpublished_drafts":
            limit = max(1, min(int(tool_input.get("limit", 5)), 20))
            days = max(1, min(int(tool_input.get("days", 14)), 60))
            drafts = _get_recent_unpublished_drafts_impl(company, limit, days)
            return json.dumps({
                "returned": len(drafts),
                "drafts":   drafts,
                "note": (
                    "These are unpublished drafts queued for this creator — "
                    "useful for within-batch pattern-fatigue checks alongside "
                    "get_recent_posts (which only shows published+scored history)."
                ),
            }, default=str)

        if tool_name == "get_post_detail":
            oid = str(tool_input.get("ordinal_post_id", "")).strip()
            detail = _get_post_detail_impl(company, oid)
            if detail is None:
                return json.dumps({"error": f"no post with ordinal_post_id={oid}"})
            return json.dumps({"post": detail}, default=str)

        if tool_name == "retrieve_similar_posts":
            # Semantic search over the Amphoreus post_embeddings corpus
            # (mirror of Jacquard's linkedin_posts, ~390k rows). This is
            # the live cross-client retrieval path; it replaces the
            # previously removed search_linkedin_corpus / analyst keyword
            # lookup with a pgvector-backed semantic one.
            from backend.src.services.post_retrieval import retrieve_similar_posts as _rsp
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return json.dumps({"count": 0, "posts": [], "error": "query is required"})
            k = max(1, min(int(tool_input.get("k", 5)), 20))
            min_reactions = max(0, int(tool_input.get("min_reactions", 0)))
            exclude_creator = tool_input.get("exclude_creator") or None
            try:
                rows = _rsp(
                    query=query,
                    k=k,
                    min_reactions=min_reactions,
                    exclude_creator=exclude_creator,
                )
            except Exception as exc:
                return json.dumps({"count": 0, "posts": [], "error": str(exc)[:400]})
            return json.dumps({"count": len(rows), "posts": rows}, default=str)

        if tool_name == "search_linkedin_corpus":
            from backend.src.agents.analyst import _tool_search_linkedin_bank
            return _tool_search_linkedin_bank({
                "query": str(tool_input.get("query", "")).strip(),
                "mode": str(tool_input.get("mode", "keyword")),
                "limit": min(int(tool_input.get("limit", 10)), 20),
            })

        return json.dumps({"error": f"unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception("[Irontomb] retrieval tool %s failed", tool_name)
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})


# ---------------------------------------------------------------------------
# Reaction schema — {reaction, anchor} for the adversarial rough-reader loop
# ---------------------------------------------------------------------------

_SUBMIT_REACTION_TOOL: dict[str, Any] = {
    "name": "submit_reaction",
    "description": (
        "Submit your reader reaction. Ends the session.\n"
        "  reaction — under-15-word GESTALT reader-voice reaction "
        "(net effect after reading the whole draft)\n"
        "  anchors  — list of {quote, reaction} for each reader-state-"
        "change moment. Zero, one, or many — match how you read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reaction": {
                "type": "string",
                "description": (
                    "Your raw, visceral, under-15-word GESTALT reaction "
                    "as a LinkedIn reader — the thing that flashes "
                    "through your head AFTER reading the last line. "
                    "Captures the post's net effect (accumulated "
                    "irritation, lingering felt-ness, the moment it "
                    "died). Not a critique. Not writing-teacher "
                    "vocabulary."
                ),
            },
            "anchors": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Reader-state-change moments as you read. At least "
                    "one anchor required — surface the specific phrase "
                    "where your felt-state shifted (positive OR "
                    "negative). For drafts with multiple shifts, emit "
                    "one anchor per shift in reading order. For drafts "
                    "that read uniformly without specific moments "
                    "standing out, anchor on the line that best "
                    "represents the overall texture (a low-energy "
                    "phrase, a generic line, the closer that didn't "
                    "land). Don't fabricate reactions you didn't have. "
                    "Don't paragraph-segment — anchor where state "
                    "ACTUALLY changed or where the post's character "
                    "lives."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {
                            "type": "string",
                            "description": (
                                "3-15 words verbatim from the draft — "
                                "the phrase that triggered the shift "
                                "(or, for uniformly-meh drafts, the "
                                "line that best represents the texture)."
                            ),
                        },
                        "reaction": {
                            "type": "string",
                            "description": (
                                "Your under-15-word reader-voice "
                                "reaction to that specific moment. "
                                "Same register as the gestalt reaction."
                            ),
                        },
                        "calibration_evidence": {
                            "type": "string",
                            "description": (
                                "Verbatim 3-15 word quote from a "
                                "calibration-block triple (this client's "
                                "real past post text) or transcript that "
                                "GROUNDS your reaction — concrete proof "
                                "this audience has reacted similarly to "
                                "this pattern before. Empty string is "
                                "ALLOWED only when the reaction is purely "
                                "structural (e.g., the closer doesn't "
                                "connect to the opener); anything that's "
                                "a generalization about 'what LinkedIn "
                                "audiences do' MUST be backed by a "
                                "calibration quote, otherwise it's a "
                                "prior, not a signal."
                            ),
                        },
                    },
                    "required": ["quote", "reaction", "calibration_evidence"],
                },
            },
        },
        "required": ["reaction", "anchors"],
    },
}


# ---------------------------------------------------------------------------
# Main entry point — single-call simulation
# ---------------------------------------------------------------------------


def simulate_flame_chase_journey(company: str, draft_text: str) -> dict[str, Any]:
    """Turn-based agent loop that predicts audience reaction to `draft_text`.

    Each call enters a short tool-using loop. Irontomb sees the draft,
    retrieves relevant past posts from this client's scored history via
    `search_past_posts` / `get_recent_posts` / `get_post_detail`, reads
    their real engagement, and grounds her prediction in that retrieved
    data via the terminal `submit_reaction` tool.

    No persistent state across calls. Each simulate is a fresh
    exploration.

    Returns a dict with the 5 prediction fields + metadata. On any
    failure, returns a dict with `_error`. Never raises.

    Routes to CLI when ``AMPHOREUS_USE_CLI=true``. The CLI sibling
    ``simulate_flame_chase_journey_cli`` lives in
    ``mcp_bridge/claude_cli.py``; wrapping the check at the top of the
    function means every call site (direct or via Stelle) goes through
    the Max subscription when enabled — no silent API leak.
    """
    try:
        from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli
        if _use_cli():
            from backend.src.mcp_bridge.claude_cli import simulate_flame_chase_journey_cli as _cli_fn
            return _cli_fn(company, draft_text)
    except ImportError:
        pass

    draft_text = (draft_text or "").strip()
    if not draft_text:
        return {"_error": "draft_text is required"}

    audience_context = _load_audience_context(company)
    observations = _load_scored_observations(company)
    calibration_block = _format_calibration_block(observations)
    cross_client_block = _format_cross_client_block(company)
    system_prompt = _build_system_prompt(
        audience_context,
        n_scored_obs=len(observations),
        calibration_block=calibration_block,
        cross_client_block=cross_client_block,
    )

    tools = [
        _SEARCH_PAST_POSTS_TOOL,
        _GET_RECENT_POSTS_TOOL,
        _GET_RECENT_UNPUBLISHED_DRAFTS_TOOL,
        _GET_POST_DETAIL_TOOL,
        # Semantic retrieval across the full ~390k cross-creator corpus
        # (Amphoreus post_embeddings pgvector mirror). Grounds Irontomb's
        # "I've seen this N times" claims in actual engagement data so
        # reactions can cite receipts. Replaces the legacy keyword-mode
        # search_linkedin_corpus which hit a now-obsolete analyst path.
        _RETRIEVE_SIMILAR_POSTS_TOOL,
        _SUBMIT_REACTION_TOOL,
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Here is the draft LinkedIn post you are evaluating. "
                "You've already read the calibration examples from this "
                "client's real history above. Predict how this specific "
                "audience will react to THIS draft, anchored in what you "
                "saw happen to comparable past posts. If the draft is in "
                "territory the examples don't cover, retrieve more "
                "comparables first; otherwise submit_reaction directly.\n\n"
                "=== DRAFT ===\n"
                f"{draft_text}\n"
                "=== END DRAFT ==="
            ),
        }
    ]

    client = anthropic.Anthropic()
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    turns_used = 0
    retrieval_calls: list[str] = []
    reaction: Optional[dict] = None

    for turn in range(1, _IRONTOMB_MAX_TURNS + 1):
        turns_used = turn
        try:
            resp = client.messages.create(
                model=_IRONTOMB_MODEL,
                max_tokens=_IRONTOMB_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            logger.warning("[Irontomb] API call failed turn=%d for %s: %s", turn, company, e)
            return {
                "_error": f"API call failed on turn {turn}: {str(e)[:200]}",
                "_turns_used": turns_used,
                "_draft_hash": _draft_hash(draft_text),
            }

        try:
            usage = resp.usage
            total_input += getattr(usage, "input_tokens", 0) or 0
            total_output += getattr(usage, "output_tokens", 0) or 0
            total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            total_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
        except Exception:
            pass

        # Append assistant turn to message history
        messages.append({"role": "assistant", "content": resp.content})

        # Process tool_use blocks
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # Model stopped without calling any tool — unusual; bail out
            logger.warning(
                "[Irontomb] %s: model ended turn %d without a tool call",
                company, turn,
            )
            return {
                "_error": "Simulator did not call any tool",
                "_turns_used": turns_used,
                "_draft_hash": _draft_hash(draft_text),
            }

        # Check for terminal submit_reaction first (multiple tools may come in one turn)
        tool_results: list[dict] = []
        for tu in tool_uses:
            if tu.name == "submit_reaction":
                if isinstance(tu.input, dict):
                    reaction = dict(tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "reaction submitted",
                })
            else:
                retrieval_calls.append(tu.name)
                result = _dispatch_retrieval_tool(company, tu.name, tu.input or {})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })

        if reaction is not None:
            # Append final tool_result so message history is well-formed, then exit
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            break

        messages.append({"role": "user", "content": tool_results})

        if resp.stop_reason == "end_turn":
            # Model chose to end without submit_reaction — treat as abort
            logger.warning(
                "[Irontomb] %s: end_turn without submit_reaction on turn %d",
                company, turn,
            )
            break

    if reaction is None:
        return {
            "_error": f"no submit_reaction within {_IRONTOMB_MAX_TURNS} turns",
            "_turns_used": turns_used,
            "_retrieval_calls": retrieval_calls,
            "_draft_hash": _draft_hash(draft_text),
        }

    # Cost accounting with cache awareness
    cost = (
        (total_input / 1e6) * _INPUT_COST_PER_MTOK
        + (total_output / 1e6) * _OUTPUT_COST_PER_MTOK
        + (total_cache_read / 1e6) * _CACHE_READ_COST_PER_MTOK
        + (total_cache_write / 1e6) * _CACHE_WRITE_COST_PER_MTOK
    )

    # Normalize the response shape. ``anchors`` is the new first-class
    # field (list of {quote, reaction}); default to [] when the model
    # emitted only a gestalt reaction. We also synthesize a legacy
    # ``anchor`` (singular) — the first anchor's quote, or empty —
    # so older readers (convergence_log column, trajectory snapshots,
    # any out-of-tree consumers) keep working without a migration.
    if not isinstance(reaction.get("anchors"), list):
        reaction["anchors"] = []
    first_anchor = reaction["anchors"][0] if reaction["anchors"] else None
    if "anchor" not in reaction:
        reaction["anchor"] = (first_anchor or {}).get("quote", "") if first_anchor else ""

    reaction["_cost_usd"] = round(cost, 5)
    reaction["_draft_hash"] = _draft_hash(draft_text)
    reaction["_turns_used"] = turns_used
    reaction["_retrieval_calls"] = retrieval_calls
    reaction["_n_scored_obs_available"] = len(observations)

    logger.info(
        "[Irontomb] %s: reaction=%r anchors=%d turns=%d retrievals=%d cost=$%.4f",
        company,
        (reaction.get("reaction") or "")[:100],
        len(reaction["anchors"]),
        turns_used,
        len(retrieval_calls),
        cost,
    )

    return reaction


# ---------------------------------------------------------------------------
# CLI for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 irontomb.py simulate <company> <draft_file>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "simulate":
        if len(sys.argv) < 4:
            print("Usage: python3 irontomb.py simulate <company> <draft_file>")
            sys.exit(1)
        company_arg = sys.argv[2]
        draft_path = Path(sys.argv[3])
        draft = draft_path.read_text(encoding="utf-8")
        result = simulate_flame_chase_journey(company_arg, draft)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: simulate")
        sys.exit(1)
