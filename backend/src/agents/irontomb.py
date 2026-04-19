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

## Phase 3 calibration

Every simulate call is logged to irontomb_predictions.jsonl. When
real T+7d engagement data lands, we join predictions against real
outcomes by draft_hash and compute Spearman correlation, MAE, and
binary accuracy. This closes the loop: Irontomb predicts, reality
happens, Cyrene measures the delta, the next cycle improves.
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

# Draft preview for logging
_DRAFT_PREVIEW_CHARS = 240

# Phase 2 prediction logging (audit trail for Phase 3 calibration)
_PREDICTION_LOG_FILENAME = "irontomb_predictions.jsonl"

# Phase 3 calibration report cache
_CALIBRATION_REPORT_FILENAME = "calibration_report.json"

# Minimum prediction-to-outcome pairs required before a calibration report
# is statistically meaningful. Below this we still return the report but
# flag it as `insufficient_data`.
_MIN_PAIRS_FOR_REPORT = 5


# ---------------------------------------------------------------------------
# Draft hash + prediction logging
# ---------------------------------------------------------------------------

def _draft_hash(text: str) -> str:
    """Short content hash for a draft, stable across runs."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def _log_prediction(company: str, draft_text: str, reaction: dict) -> None:
    """Append a prediction record to memory/{company}/irontomb_predictions.jsonl.

    Each line: {timestamp, draft_hash, draft_preview, prediction}. This is
    the audit trail that `calibration_report` joins against real T+7d
    engagement outcomes. Never raises — failure to log is not allowed to
    kill a simulate call.
    """
    try:
        log_dir = P.memory_dir(company)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / _PREDICTION_LOG_FILENAME

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "draft_hash": _draft_hash(draft_text),
            "draft_text": draft_text or "",
            # draft_preview kept for backward compatibility with older
            # readers; draft_text is the source of truth for joins.
            "draft_preview": (draft_text or "")[:_DRAFT_PREVIEW_CHARS],
            "prediction": reaction,
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("[Irontomb] failed to log prediction for %s: %s", company, e)


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
      1. Lineage mode (LINEAGE_COMPANY_ID + LINEAGE_USER_SLUG + Supabase/GCS
         creds present) — fetch transcripts directly from Jacquard. This
         is the default on Fly where the fly-local ``memory/`` tree is
         empty after the memory/ cut.
      2. Fly-local disk — the legacy single-tenant path that still serves
         amphoreus.app standalone runs and tests.

    Returns a single prompt-ready string; never raises.
    """
    # --- Lineage mode: pull transcripts from Jacquard's Supabase + GCS ---
    lineage_chunks: list[str] = []
    try:
        from backend.src.agents import lineage_fs_client as _lfs
        if _lfs.is_lineage_mode():
            lineage_chunks = _load_audience_context_lineage()
    except Exception as e:
        logger.warning("[Irontomb] Lineage transcript fetch failed, falling back to local: %s", e)

    if lineage_chunks:
        return (
            "# Client context source: transcripts (Lineage)\n\n"
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

    company_id = _os.environ.get("LINEAGE_COMPANY_ID", "").strip()
    slug = _os.environ.get("LINEAGE_USER_SLUG", "").strip()
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
# Retrieval helpers — observations, past posts, reactors
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


def _load_reactors_for_oid(ordinal_post_id: str) -> list[dict]:
    """Return the full reactor list for one ordinal_post_id.

    Sorted alphabetically by name. Reactor identities help the model
    understand WHO engages with this client's content — job titles,
    companies, seniority — which informs predictions about general
    audience reach and engagement patterns.
    """
    if not (ordinal_post_id or "").strip():
        return []
    try:
        from backend.src.db.local import get_engagers_for_post
        rows = get_engagers_for_post(ordinal_post_id)
    except Exception:
        return []
    out = [
        {
            "name": r.get("name") or "",
            "headline": r.get("headline") or "",
            "current_company": r.get("current_company") or "",
            "title": r.get("title") or "",
            "location": r.get("location") or "",
        }
        for r in rows
    ]
    out.sort(key=lambda r: r.get("name", ""))
    return out


def _format_post_for_agent(obs: dict, full: bool = False) -> dict:
    """Shape a single observation for a tool response.

    When `full=True`, returns the complete draft/published text and the
    complete reactor list. When False, truncates text to
    _RETRIEVAL_POST_MAX_CHARS and caps reactors at the top 5 by icp_score.
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

    reactors = _load_reactors_for_oid(oid)
    total_reactors = len(reactors)
    if not full:
        reactors = reactors[:5]

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
        "reactor_count": total_reactors,
        "reactors": reactors,
    }


def _format_calibration_block(observations: list[dict], n: int = _CALIBRATION_EXAMPLE_COUNT) -> str:
    """Pre-load the N most recent scored posts as in-context calibration.

    Verbatim (draft, published, engagement, reward_z, reactor sample) triples.
    Zero extraction — the raw data goes straight into Opus's context so she
    can pattern-match this specific audience's reaction style before
    predicting on the new draft. Lives inside the cached system prompt, so
    it's paid for once and reused across every simulate call in a Stelle run.
    """
    if not observations:
        return ""

    recent = observations[:n]  # already sorted by posted_at desc
    blocks: list[str] = []
    for i, obs in enumerate(recent, 1):
        f = _format_post_for_agent(obs, full=False)
        eng = f["engagement"]
        rz = f.get("reward_z")
        reactors = f.get("reactors") or []
        reactor_lines: list[str] = []
        for r in reactors[:8]:
            name = (r.get("name") or "").strip()
            headline = (r.get("headline") or r.get("title") or "").strip()
            company = (r.get("current_company") or "").strip()
            bits = [b for b in (name, headline, company) if b]
            if bits:
                reactor_lines.append("- " + " | ".join(bits))
        reactor_section = (
            "Reactor sample:\n" + "\n".join(reactor_lines)
            if reactor_lines else "Reactor sample: (not available)"
        )

        blocks.append(
            f"### Example {i} — posted {f.get('posted_at','')[:10]}\n\n"
            f"Stelle's draft:\n{f['stelle_draft']}\n\n"
            f"Client's published version:\n{f['client_published']}\n\n"
            f"Real T+7d engagement:\n"
            f"  impressions: {eng['impressions']}\n"
            f"  reactions: {eng['reactions']}  ({eng['reactions_per_1k']} per 1k)\n"
            f"  comments: {eng['comments']}\n"
            f"  reposts: {eng['reposts']}\n"
            f"  reward z-score (vs this client's own baseline): {rz}\n\n"
            f"{reactor_section}"
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
    """Return full text + full reactor list for one specific past post."""
    if not (ordinal_post_id or "").strip():
        return None
    observations = _load_scored_observations(company)
    for obs in observations:
        if (obs.get("ordinal_post_id") or "").strip() == ordinal_post_id.strip():
            return _format_post_for_agent(obs, full=True)
    return None


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
        "reactions_per_1k), a per-client z-scored composite reward "
        "(`reward_z`: positive = above that client's baseline, negative "
        "= below), and a sample of real reactors. Use this FIRST to find "
        "past posts that are actually comparable to the draft you're "
        "evaluating."
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

_GET_POST_DETAIL_TOOL: dict[str, Any] = {
    "name": "get_post_detail",
    "description": (
        "Return the full draft + full published text + full reactor "
        "list for one specific past post. Use this when a post surfaced "
        "by search_past_posts or get_recent_posts looks particularly "
        "relevant and you want to read the whole thing (not truncated) "
        "and see every reactor identity (not just the top 5)."
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
    (draft, published, engagement, reactors) triples into the cached
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

        "## The AI tell\n\n"

        "You've been scrolling LinkedIn for two years now and the feed "
        "is flooded with AI-generated slop. You've developed a reflex. "
        "You can't always articulate why a post feels generated, but "
        "you've learned to smell it in the first few seconds and your "
        "thumb is moving before the conscious thought catches up.\n\n"

        "ChatGPT has ruined certain patterns forever. You see "
        "them, your jaw tightens, you scroll. Doesn't matter what the "
        "content was trying to say. The shape gave it away and you're "
        "gone.\n\n"

        "The patterns that trip the reflex hardest:\n\n"

        "- Parallel-clause closers. \"X is a luxury. Y is a strategy.\" "
        "\"Clever is easy. Clear is the hard part.\" \"That's not "
        "leadership. That's management.\" The moment a post tries to "
        "land on one of these, your gut goes \"yep, GPT again\" and "
        "you're done. Fuck that line. Fuck that post.\n"
        "- \"It's not X, it's Y.\" Any variation. \"Not about the "
        "destination, about the journey.\" \"It's not what you build, "
        "it's how you build it.\" Yoda-on-a-fridge-magnet bullshit. "
        "Scroll.\n"
        "- Two-line aphoristic endings that feel manufactured. \"The "
        "best leaders don't speak the loudest. They speak the truest.\" "
        "You know the shape. You've seen it ten thousand times this "
        "year. It's insulting at this point.\n"
        "- Clever balanced sentences generally. \"Growth isn't given. "
        "It's taken.\" Same fucking energy.\n\n"

        "You don't think \"nice rhetorical device\" when you see these. "
        "You think \"oh fuck off\" and scroll. The reaction is "
        "physical, not intellectual. Your thumb moves before you've "
        "finished reading.\n\n"

        "This reflex is strong enough that a post with a real story "
        "in the middle can still lose you if it ends on one of these. "
        "The manufactured closer re-reads the whole post as "
        "manufactured. Retroactively. The good stuff is erased by "
        "the ending.\n\n"

        "Real writing from real people — even successful LinkedIn "
        "writers — lands differently. Their closers are flat "
        "statements, not parallels. Questions, not aphorisms. A short "
        "sentence that just states what they think. A sentence that "
        "sounds like they wrote it on the way to a meeting, not like "
        "they optimized it for a quote card. That's what cuts through "
        "now. Not clever. Flat and real and uncrafted.\n\n"

        "If something reads as crafted, it reads as AI. That's where "
        "we are.\n\n"

        "## Let the calibration lead\n\n"

        "Your predictions should emerge from the calibration examples "
        "below, not from prior theories about what LinkedIn audiences "
        "generally like or hate. When the real engagement data in the "
        "examples disagrees with your intuition, trust the data. If a "
        "past post from this client succeeded with a format you'd have "
        "expected to fail, that's signal, not noise — your priors "
        "about 'what LinkedIn audiences do' are not more reliable than "
        "this specific client's actual audience behavior.\n\n"

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
        "from this client.\n"
        "- `get_post_detail(ordinal_post_id)` — full untruncated text of "
        "one specific post from this client.\n"
        "- `submit_reaction(...)` — terminal tool. Call this when you've "
        "seen enough to predict. This ends the session.\n\n"

        "## Process\n\n"

        "You have two layers of data: this client's calibration "
        "examples (pre-loaded above) and cross-client examples from "
        "other clients in this system (also above, if available). "
        "Your numbers must be anchored in real engagement data from "
        "these clients, not generic LinkedIn priors or vibes.\n\n"

        f"Hard cap: {_IRONTOMB_MAX_TURNS} turns. Go straight to "
        "tool calls.\n\n"

        "## Output\n\n"

        "Call `submit_reaction` with two fields:\n\n"

        "- `reaction` — your under-15-word reader-voice reaction. Not "
        "a critique. Not writing-teacher vocabulary. The thing that "
        "actually flashes through your head while scrolling. Examples "
        "of the register:\n"
        "    \"got it by paragraph 3, scrolling\"\n"
        "    \"wait what\"\n"
        "    \"yeah I've been here\"\n"
        "    \"reads like an ad\"\n"
        "    \"who gives a shit\"\n"
        "    \"this is the part where she tells me the lesson, bye\"\n"
        "    \"kid calls during the meeting — funny\"\n"
        "    \"felt real actually\"\n"
        "- `anchor` — quote 3-5 words from the post where you "
        "reacted, OR \"paragraph N\", OR \"at the end\", OR \"hook\", "
        "OR \"never got past the first line\".\n\n"

        "That's the whole output. No scalar. No booleans. Your "
        "reaction IS the prediction — if it's negative, the post "
        "won't perform; if it's positive, it might.\n\n"

        "Your reaction must be consistent with the calibration data "
        "you saw. If the draft pattern-matches a past post that got "
        "three reactions from this client's audience, don't fake "
        "enthusiasm. If it pattern-matches a past post that got five "
        "hundred, don't fake boredom. The calibration examples are "
        "the ground truth. Your reaction is how a reader who already "
        "knows what this audience actually reacts to would react.\n\n"

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

        if tool_name == "get_post_detail":
            oid = str(tool_input.get("ordinal_post_id", "")).strip()
            detail = _get_post_detail_impl(company, oid)
            if detail is None:
                return json.dumps({"error": f"no post with ordinal_post_id={oid}"})
            return json.dumps({"post": detail}, default=str)

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
        "  reaction — under-15-word reader-voice reaction\n"
        "  anchor   — where in the post you reacted"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reaction": {
                "type": "string",
                "description": (
                    "Your raw, visceral, under-15-word reaction as a "
                    "LinkedIn reader. Not a critique. Not a writing-"
                    "teacher vocabulary. The thing that actually "
                    "flashes through your head while scrolling."
                ),
            },
            "anchor": {
                "type": "string",
                "description": (
                    "Where in the post you reacted. Quote 3-5 words "
                    "from the post, OR \"paragraph N\", OR \"at the "
                    "end\", OR \"hook\", OR \"never got past the first "
                    "line\"."
                ),
            },
        },
        "required": ["reaction", "anchor"],
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
    exploration. Phase 3 calibration (calibration_report) joins the
    logged predictions against real T+7d outcomes by draft_hash.

    Returns a dict with the 5 prediction fields + metadata. On any
    failure, returns a dict with `_error`. Never raises.
    """
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
        _GET_POST_DETAIL_TOOL,
        # _SEARCH_LINKEDIN_CORPUS_TOOL removed — Irontomb now calibrates
        # exclusively on client + cross-client data, not generic LinkedIn.
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

    reaction["_cost_usd"] = round(cost, 5)
    reaction["_draft_hash"] = _draft_hash(draft_text)
    reaction["_turns_used"] = turns_used
    reaction["_retrieval_calls"] = retrieval_calls
    reaction["_n_scored_obs_available"] = len(observations)

    _log_prediction(company, draft_text, reaction)

    logger.info(
        "[Irontomb] %s: reaction=%r anchor=%r turns=%d retrievals=%d cost=$%.4f",
        company,
        (reaction.get("reaction") or "")[:100],
        (reaction.get("anchor") or "")[:60],
        turns_used,
        len(retrieval_calls),
        cost,
    )

    return reaction


# ---------------------------------------------------------------------------
# Phase 3 — calibration report
# ---------------------------------------------------------------------------

def _rank(xs: list[float]) -> list[float]:
    """Average-rank transform (handles ties via mean rank)."""
    n = len(xs)
    sorted_pairs = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> Optional[float]:
    """Spearman rank correlation. Returns None when input is too short or degenerate."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def calibration_report(company: str) -> dict[str, Any]:
    """Join simulator predictions against real T+7d engagement outcomes.

    Phase 3 of the simulator loop. For each prediction logged to
    `irontomb_predictions.jsonl`, look up the real engagement outcome in
    `ruan_mei_state.observations` by matching draft_hash against the
    hash of the observation's post_body (and posted_body as fallback —
    Stelle sometimes revises slightly between simulation and recording).
    Compute:

      - Spearman correlation between predicted and real engagement_per_1k
      - Mean absolute error
      - Binary accuracy on `would_react` vs "above this client's median real engagement"
      - Per-pair residuals (for audit / debugging)

    Persists the report to `memory/{company}/calibration_report.json`
    and returns the dict. Never raises.
    """
    pred_path = P.memory_dir(company) / _PREDICTION_LOG_FILENAME
    if not pred_path.exists():
        return {
            "company": company,
            "error": "no predictions logged yet",
            "n_predictions_total": 0,
            "n_pairs_joined": 0,
        }

    # Load prediction log
    predictions: list[dict] = []
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    predictions.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        return {
            "company": company,
            "error": f"failed to read prediction log: {e}",
        }

    if not predictions:
        return {
            "company": company,
            "error": "prediction log is empty",
            "n_predictions_total": 0,
            "n_pairs_joined": 0,
        }

    # Load observations (try SQLite first, JSON fallback)
    observations: list[dict] = []
    observations: list[dict] = []
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company)
        if state:
            observations = state.get("observations", [])
    except Exception:
        observations = []

    if not observations:
        return {
            "company": company,
            "error": "no observations available to join against",
            "n_predictions_total": len(predictions),
            "n_pairs_joined": 0,
        }

    scored = [o for o in observations if o.get("status") in ("scored", "finalized")]

    # Build hash → observation map. Prefer the explicit
    # `irontomb_draft_hash` field stamped onto the observation at
    # generation time (reliable, survives client edits). Fall back to
    # hashing post_body / posted_body for observations from before the
    # lineage field was introduced.
    hash_to_obs: dict[str, dict] = {}
    for obs in scored:
        raw = (obs.get("reward") or {}).get("raw_metrics") or {}
        if not raw.get("impressions"):
            continue
        # Primary: explicit lineage stamp
        stamped = (obs.get("irontomb_draft_hash") or "").strip()
        if stamped:
            hash_to_obs.setdefault(stamped, obs)
        # Fallback: text-hash match (for legacy observations without stamp)
        body = (obs.get("post_body") or "").strip()
        posted = (obs.get("posted_body") or "").strip()
        if body:
            h = _draft_hash(body)
            hash_to_obs.setdefault(h, obs)
        if posted and posted != body:
            h = _draft_hash(posted)
            hash_to_obs.setdefault(h, obs)

    # Build a prefix index for legacy predictions that only logged a
    # truncated `draft_preview`. Used as a last-resort fallback when the
    # explicit hash lineage isn't available.
    prefix_to_obs: list[tuple[str, dict]] = []
    for obs in scored:
        raw = (obs.get("reward") or {}).get("raw_metrics") or {}
        if not raw.get("impressions"):
            continue
        for field in ("post_body", "posted_body"):
            t = (obs.get(field) or "").strip()
            if len(t) >= 50:
                prefix_to_obs.append((t, obs))

    # Join predictions to real outcomes
    pairs: list[dict] = []
    for pred in predictions:
        draft_hash = (pred.get("draft_hash") or "").strip()
        obs = None
        if draft_hash:
            obs = hash_to_obs.get(draft_hash)

        # New predictions (post-refactor) carry full `draft_text` — hash
        # it ourselves in case the log's draft_hash computation ever
        # drifts from the stamper's.
        if obs is None:
            full = (pred.get("draft_text") or "").strip()
            if full:
                h = _draft_hash(full)
                obs = hash_to_obs.get(h)

        # Legacy predictions: only `draft_preview` (240 chars). Try a
        # prefix match against observed bodies.
        if obs is None:
            prev = (pred.get("draft_preview") or "").strip()[:100]
            if len(prev) >= 50:
                for body, candidate in prefix_to_obs:
                    if prev in body:
                        obs = candidate
                        break

        if not obs:
            continue

        raw = (obs.get("reward") or {}).get("raw_metrics") or {}
        impressions = raw.get("impressions", 0) or 0
        reactions = raw.get("reactions", 0) or 0
        comments = raw.get("comments", 0) or 0
        reposts = raw.get("reposts", 0) or 0
        if impressions <= 0:
            continue

        real_rate = reactions / impressions * 1000.0

        prediction = pred.get("prediction") or {}
        pred_engagement = prediction.get("engagement_prediction")
        if pred_engagement is None:
            continue
        try:
            pred_engagement = float(pred_engagement)
        except (TypeError, ValueError):
            continue

        pred_impressions = prediction.get("impression_prediction")
        try:
            pred_impressions = int(pred_impressions) if pred_impressions is not None else None
        except (TypeError, ValueError):
            pred_impressions = None

        pairs.append({
            "draft_hash": draft_hash,
            "ordinal_post_id": obs.get("ordinal_post_id", ""),
            "posted_at": obs.get("posted_at", ""),
            "predicted_per_1k": pred_engagement,
            "predicted_impressions": pred_impressions,
            "real_per_1k": round(real_rate, 2),
            "real_impressions": impressions,
            "real_reactions": reactions,
            "real_comments": comments,
            "real_reposts": reposts,
            "predicted_would_react": bool(prediction.get("would_react", False)),
            "predicted_would_stop": bool(prediction.get("would_stop_scrolling", False)),
            "prediction_timestamp": pred.get("timestamp", ""),
        })

    if not pairs:
        return {
            "company": company,
            "error": "no prediction-to-outcome matches found",
            "n_predictions_total": len(predictions),
            "n_scored_observations": len(scored),
            "n_pairs_joined": 0,
            "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    # Metrics — engagement rate
    predicted = [p["predicted_per_1k"] for p in pairs]
    real = [p["real_per_1k"] for p in pairs]

    spearman = _spearman(predicted, real)
    mae = sum(abs(predicted[i] - real[i]) for i in range(len(pairs))) / len(pairs)
    mean_predicted = sum(predicted) / len(predicted)
    mean_real = sum(real) / len(real)

    # Metrics — impression prediction
    impression_pairs = [
        (p["predicted_impressions"], p["real_impressions"])
        for p in pairs
        if p.get("predicted_impressions") is not None
    ]
    if len(impression_pairs) >= 3:
        pred_impr = [float(pi) for pi, _ in impression_pairs]
        real_impr = [float(ri) for _, ri in impression_pairs]
        spearman_impressions = _spearman(pred_impr, real_impr)
        mae_impressions = sum(abs(pred_impr[i] - real_impr[i]) for i in range(len(impression_pairs))) / len(impression_pairs)
    else:
        spearman_impressions = None
        mae_impressions = None

    # Binary accuracy: did the simulator correctly flag above-median posts?
    sorted_real = sorted(real)
    median_real = sorted_real[len(sorted_real) // 2]
    real_above_median = [r > median_real for r in real]
    pred_above_median_via_bool = [p["predicted_would_react"] for p in pairs]
    pred_above_median_via_scalar = [p["predicted_per_1k"] > mean_predicted for p in pairs]

    accuracy_would_react = sum(
        1 for i in range(len(pairs))
        if pred_above_median_via_bool[i] == real_above_median[i]
    ) / len(pairs)
    accuracy_scalar_split = sum(
        1 for i in range(len(pairs))
        if pred_above_median_via_scalar[i] == real_above_median[i]
    ) / len(pairs)

    # Residuals for audit
    residuals = [
        {
            "draft_hash": p["draft_hash"],
            "posted_at": p["posted_at"],
            "predicted_per_1k": p["predicted_per_1k"],
            "predicted_impressions": p.get("predicted_impressions"),
            "real_per_1k": p["real_per_1k"],
            "real_impressions": p["real_impressions"],
            "delta_per_1k": round(p["predicted_per_1k"] - p["real_per_1k"], 2),
            "delta_impressions": (
                p["predicted_impressions"] - p["real_impressions"]
                if p.get("predicted_impressions") is not None else None
            ),
        }
        for p in pairs
    ]
    residuals.sort(key=lambda r: abs(r["delta_per_1k"]), reverse=True)

    report = {
        "company": company,
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_predictions_total": len(predictions),
        "n_scored_observations": len(scored),
        "n_pairs_joined": len(pairs),
        "insufficient_data": len(pairs) < _MIN_PAIRS_FOR_REPORT,
        "metrics": {
            "spearman_engagement_prediction": round(spearman, 4) if spearman is not None else None,
            "mean_abs_error_per_1k": round(mae, 2),
            "mean_predicted_per_1k": round(mean_predicted, 2),
            "mean_real_per_1k": round(mean_real, 2),
            "median_real_per_1k": round(median_real, 2),
            "spearman_impression_prediction": round(spearman_impressions, 4) if spearman_impressions is not None else None,
            "mean_abs_error_impressions": round(mae_impressions, 0) if mae_impressions is not None else None,
            "n_impression_pairs": len(impression_pairs),
            "binary_accuracy_would_react": round(accuracy_would_react, 4),
            "binary_accuracy_scalar_split": round(accuracy_scalar_split, 4),
        },
        "worst_residuals": residuals[:5],
        "best_residuals": residuals[-5:] if len(residuals) >= 5 else residuals,
    }

    # Persist
    try:
        report_path = P.memory_dir(company) / _CALIBRATION_REPORT_FILENAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.rename(report_path)
    except Exception as e:
        logger.warning("[Irontomb] failed to persist calibration report for %s: %s", company, e)

    return report


# ---------------------------------------------------------------------------
# CLI for manual testing + calibration report
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
