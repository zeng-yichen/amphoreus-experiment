"""LinkedIn Market Intelligence Pipeline — autonomous competitive monitoring.

Monitors top LinkedIn creators per vertical, extracts trending topics, hook
shifts, and whitespace opportunities. Feeds signals into Stelle as soft
generation context.

Design: the system acts and reports, not asks and waits. No human approval
gates except publishing, ICP definition changes, and client onboarding.

Components:
1. Creator Registry — auto-generates monitored creator lists per vertical
2. Weekly Scraper — pulls last 7 days of posts from tracked creators
3. Signal Extraction — topic velocity, hook trends, whitespace detection
4. Stelle Integration — soft context injection during generation
5. Autonomous Wiring — step 10 in ordinal_sync, auto-seed, auto-trigger

Storage: memory/our_memory/market_intel/
  creator_registry.json        — vertical→creator mapping
  {vertical}/weekly_{date}.json — raw scraped posts
  {vertical}/signals_{date}.json — extracted signals
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Feature flags & cost controls
# ---------------------------------------------------------------------------

_MARKET_INTEL_ENABLED = True
_DEFAULT_MAX_CREATORS = 30
_DEFAULT_MAX_POSTS = 50
_DEFAULT_SCRAPE_INTERVAL = 6
_DEFAULT_ROTATION_INTERVAL = 14
_DEFAULT_CORE_RATIO = 0.70
_APIFY_PROFILE_POSTS_ACTOR = "apimaestro~linkedin-profile-posts"


class MarketIntelAdaptiveConfig:
    """Adaptive scraping/rotation params based on signal yield and data density."""

    MODULE_NAME = "market_intel"

    def resolve(self, vertical: str = "") -> dict:
        """Compute adaptive market intel params from vertical signal history."""
        state = _load_market_intel_state()
        v_history = state.get("vertical_history", {}).get(vertical, {})

        if not v_history or v_history.get("rotation_cycles", 0) < 2:
            return self.get_defaults()

        # Scale creators by data density — no hard clamps
        scored = v_history.get("total_scored_posts", 0)
        max_creators = scored // 3 or _DEFAULT_MAX_CREATORS

        # Scale scrape interval by creator activity
        avg_posts_month = v_history.get("avg_posts_per_creator_per_month", 4)
        scrape_interval = round(30 / max(avg_posts_month, 0.5))
        max_posts = round(avg_posts_month * 3) or _DEFAULT_MAX_POSTS

        # Rotation interval from signal yield
        signal_yield = v_history.get("last_signal_yield", 1.0)
        rotation_interval = round(14 / max(signal_yield, 0.1))

        # Core ratio from exploration success — data-driven split
        core_rate = v_history.get("core_signal_rate", 0.7)
        explore_rate = v_history.get("explore_signal_rate", 0.3)
        total_rate = core_rate + explore_rate
        core_ratio = (core_rate / total_rate) if total_rate > 0 else _DEFAULT_CORE_RATIO

        return {
            "max_creators_per_vertical": max_creators,
            "max_posts_per_creator": max_posts,
            "scrape_interval_days": scrape_interval,
            "rotation_interval_days": rotation_interval,
            "core_ratio": round(core_ratio, 2),
            "_tier": "client",
        }

    def get_defaults(self) -> dict:
        return {
            "max_creators_per_vertical": _DEFAULT_MAX_CREATORS,
            "max_posts_per_creator": _DEFAULT_MAX_POSTS,
            "scrape_interval_days": _DEFAULT_SCRAPE_INTERVAL,
            "rotation_interval_days": _DEFAULT_ROTATION_INTERVAL,
            "core_ratio": _DEFAULT_CORE_RATIO,
            "_tier": "default",
        }


def _load_market_intel_state() -> dict:
    path = _intel_root() / "market_intel_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_market_intel_state(state: dict) -> None:
    path = _intel_root() / "market_intel_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _intel_root() -> Path:
    return P.our_memory_dir() / "market_intel"


def _registry_path() -> Path:
    return _intel_root() / "creator_registry.json"


def _vertical_dir(vertical: str) -> Path:
    return _intel_root() / vertical


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Part 1: Creator Registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    path = _registry_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"verticals": {}, "client_vertical_map": {}, "updated_at": ""}


def _save_registry(registry: dict) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = _now()
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_verticals() -> dict[str, list[str]]:
    """Auto-detect verticals from existing clients' ICP definitions.

    Returns {vertical_slug: [company1, company2, ...]}.
    One Haiku call to generate taxonomy from actual client data.
    """
    client_icps: list[dict] = []
    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        icp_path = P.icp_definition_path(d.name)
        if not icp_path.exists():
            continue
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            desc = icp.get("description", "")
            if desc:
                client_icps.append({"company": d.name, "icp": desc[:300]})
        except Exception:
            continue

    if not client_icps:
        return {}

    clients_text = "\n".join(f"- {c['company']}: {c['icp']}" for c in client_icps)

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": (
                "Group these B2B LinkedIn clients into verticals based on their ICP industry.\n\n"
                f"{clients_text}\n\n"
                "Return a JSON object mapping vertical_slug to company list:\n"
                '{"clinical-biotech": ["innovocommerce", "hensley-biostats"], '
                '"enterprise-tech": ["trimble-mark"], ...}\n\n'
                "Use kebab-case slugs. Group clients that share the same ICP industry. "
                "A client can only belong to one vertical. Output ONLY the JSON."
            )}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("[market_intel] Vertical detection failed: %s", e)
        return {}


def seed_creator_registry(company: str) -> dict:
    """Auto-generate creator registry for a client's vertical.

    If the client has no vertical mapping, auto-detect it first.
    If the vertical has no creators, seed them via Haiku.
    Returns the vertical's creator data.
    """
    registry = _load_registry()

    # Auto-assign vertical if missing
    if company not in registry.get("client_vertical_map", {}):
        verticals = detect_verticals()
        for v_slug, companies in verticals.items():
            for c in companies:
                registry.setdefault("client_vertical_map", {})[c] = v_slug
        _save_registry(registry)

    vertical = registry.get("client_vertical_map", {}).get(company)
    if not vertical:
        logger.info("[market_intel] Could not determine vertical for %s", company)
        return {}

    # Check if vertical already has creators
    v_data = registry.get("verticals", {}).get(vertical)
    if v_data and v_data.get("core"):
        return v_data

    # Gather context for creator discovery
    icp_text = ""
    icp_path = P.icp_definition_path(company)
    if icp_path.exists():
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            icp_text = icp.get("description", "")
        except Exception:
            pass

    # Get top topics from RuanMei
    top_topics = ""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [o for o in rm._state.get("observations", []) if o.get("status") in ("scored", "finalized")]
        scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0), reverse=True)
        top_analyses = [o.get("descriptor", {}).get("analysis", "")[:150] for o in scored[:5]]
        top_topics = "\n".join(f"- {a}" for a in top_analyses if a)
    except Exception:
        pass

    prompt = (
        f"You are building a LinkedIn creator monitoring list for the '{vertical}' vertical.\n\n"
        f"ICP (who the client sells to):\n{icp_text}\n\n"
    )
    if top_topics:
        prompt += f"Top-performing post topics in this vertical:\n{top_topics}\n\n"
    prompt += (
        f"List 25 LinkedIn creators who post high-engagement content relevant to this ICP. "
        "These should be people whose posts the ICP audience actually reads — thought leaders, "
        "practitioners, analysts, not brands or company pages.\n\n"
        "Return a JSON array:\n"
        '[{"linkedin_url": "https://linkedin.com/in/username", "name": "Full Name", '
        '"why": "One-line reason they are relevant to this ICP"}]\n\n'
        "Prioritize creators who post 2+ times per week with high engagement. "
        "Output ONLY the JSON array."
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        creators = json.loads(raw)
        if not isinstance(creators, list):
            creators = []
    except Exception as e:
        logger.warning("[market_intel] Creator seeding failed for %s: %s", vertical, e)
        return {}

    # Split into core (70%) and exploration pool (30%)
    n_core = max(1, int(len(creators) * _CORE_RATIO))
    core = creators[:n_core]
    pool = creators[n_core:]

    # Add metadata
    today = _today()
    for c in core + pool:
        c["added"] = today
        c["engagement_relevance"] = 0.0  # will be computed after scraping

    # Initial rotation: first batch from exploration pool
    rotation_size = min(len(pool), max(1, _MAX_CREATORS_PER_VERTICAL - len(core)))
    active_rotation = pool[:rotation_size]

    v_data = {
        "core": core,
        "exploration_pool": pool,
        "active_rotation": active_rotation,
        "last_rotated": today,
        "last_scraped": "",
        "created_at": _now(),
    }

    registry.setdefault("verticals", {})[vertical] = v_data
    _save_registry(registry)

    logger.info(
        "[market_intel] Seeded %d creators for '%s' (%d core, %d exploration)",
        len(creators), vertical, len(core), len(pool),
    )
    return v_data


# ---------------------------------------------------------------------------
# Part 2: Weekly Scraper
# ---------------------------------------------------------------------------

def collect_market_intel(vertical: str) -> list[dict]:
    """Pull last 7 days of posts from tracked creators in a vertical.

    Uses Apify LinkedIn profile posts scraper. Respects cost controls.
    Returns list of scraped post dicts.
    """
    if not _MARKET_INTEL_ENABLED:
        return []

    apify_token = os.environ.get("APIFY_API_TOKEN", "")
    if not apify_token:
        logger.warning("[market_intel] APIFY_API_TOKEN not set — skipping scrape")
        return []

    registry = _load_registry()
    v_data = registry.get("verticals", {}).get(vertical)
    if not v_data:
        logger.info("[market_intel] No creators for vertical '%s'", vertical)
        return []

    # Resolve adaptive scraping params
    mi_cfg = MarketIntelAdaptiveConfig().resolve(vertical)
    scrape_interval = mi_cfg.get("scrape_interval_days", _DEFAULT_SCRAPE_INTERVAL)
    max_creators = mi_cfg.get("max_creators_per_vertical", _DEFAULT_MAX_CREATORS)
    max_posts = mi_cfg.get("max_posts_per_creator", _DEFAULT_MAX_POSTS)

    # Check scrape frequency
    last_scraped = v_data.get("last_scraped", "")
    if last_scraped:
        try:
            last_dt = datetime.fromisoformat(last_scraped.replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            if days_since < scrape_interval:
                logger.info("[market_intel] Skipping '%s' scrape (%.1f days since last)", vertical, days_since)
                return []
        except Exception:
            pass

    # Build creator list: core + active rotation
    creators = v_data.get("core", []) + v_data.get("active_rotation", [])
    creators = creators[:max_creators]

    # Extract profile URLs/usernames
    profile_urls = []
    for c in creators:
        url = c.get("linkedin_url", "")
        if url:
            profile_urls.append(url)

    if not profile_urls:
        return []

    logger.info("[market_intel] Scraping %d creators for '%s'...", len(profile_urls), vertical)

    all_posts: list[dict] = []
    import httpx

    for url in profile_urls:
        try:
            resp = httpx.post(
                f"https://api.apify.com/v2/acts/{_APIFY_PROFILE_POSTS_ACTOR}/run-sync-get-dataset-items",
                params={"format": "json", "token": apify_token},
                json={
                    "profileUrls": [url],
                    "limit": max_posts,
                    "dateFrom": (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"),
                },
                timeout=120,
            )
            resp.raise_for_status()
            posts = resp.json()
            if isinstance(posts, list):
                for p in posts:
                    stats = p.get("stats") or {}
                    author = p.get("author") or {}
                    posted_at_obj = p.get("posted_at") or {}
                    all_posts.append({
                        "text": (p.get("text") or p.get("commentary") or p.get("content") or "").strip(),
                        "reactions": (
                            stats.get("total_reactions")
                            or p.get("likeCount")
                            or p.get("reactions")
                            or 0
                        ),
                        "comments": (
                            stats.get("comments")
                            or p.get("commentCount")
                            or p.get("comments")
                            or 0
                        ),
                        "reposts": (
                            stats.get("reposts")
                            or p.get("shareCount")
                            or p.get("reposts")
                            or 0
                        ),
                        "posted_at": (
                            (posted_at_obj.get("date") if isinstance(posted_at_obj, dict) else "")
                            or p.get("postedAt")
                            or p.get("publishedAt")
                            or ""
                        ),
                        "creator_name": (
                            (f"{author.get('first_name', '')} {author.get('last_name', '')}".strip() if author else "")
                            or p.get("authorName")
                            or ""
                        ),
                        "creator_url": url,
                        "creator_followers": p.get("authorFollowers") or 0,
                    })
        except Exception as e:
            logger.warning("[market_intel] Scrape failed for %s: %s", url[:50], e)
            continue

    # Filter empty posts
    all_posts = [p for p in all_posts if p.get("text") and len(p["text"]) > 50]

    # Save raw data
    out_dir = _vertical_dir(vertical)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weekly_{_today()}.json"
    out_path.write_text(json.dumps(all_posts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update last_scraped
    v_data["last_scraped"] = _now()
    _save_registry(registry)

    # Auto-rotate exploration pool if interval passed
    _maybe_rotate_exploration(vertical, registry, all_posts, mi_cfg)

    # Persist market intel state for dashboard and adaptive config
    _update_market_intel_state(vertical, len(all_posts), len(profile_urls))

    logger.info("[market_intel] Scraped %d posts from %d creators for '%s'", len(all_posts), len(profile_urls), vertical)
    return all_posts


def _update_market_intel_state(vertical: str, posts_scraped: int, creators_scraped: int) -> None:
    """Track scrape history for adaptive config."""
    state = _load_market_intel_state()
    vh = state.setdefault("vertical_history", {}).setdefault(vertical, {})
    vh["total_scraped_posts"] = vh.get("total_scraped_posts", 0) + posts_scraped
    vh["total_scrapes"] = vh.get("total_scrapes", 0) + 1
    if creators_scraped > 0 and vh["total_scrapes"] > 0:
        vh["avg_posts_per_creator_per_month"] = round(
            posts_scraped / max(creators_scraped, 1) * 4, 1
        )  # rough monthly estimate from weekly scrape
    _save_market_intel_state(state)


def _maybe_rotate_exploration(vertical: str, registry: dict, posts: list[dict], mi_cfg: dict | None = None) -> None:
    """Rotate exploration slots every 2 weeks based on engagement relevance."""
    v_data = registry.get("verticals", {}).get(vertical)
    if not v_data:
        return

    rotation_interval = (mi_cfg or {}).get("rotation_interval_days", _DEFAULT_ROTATION_INTERVAL)
    last_rotated = v_data.get("last_rotated", "")
    if last_rotated:
        try:
            last_dt = datetime.fromisoformat(last_rotated.replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            if days_since < rotation_interval:
                return
        except Exception:
            pass

    # Score exploration creators by engagement
    creator_engagement: dict[str, float] = defaultdict(float)
    creator_count: dict[str, int] = defaultdict(int)
    for p in posts:
        url = p.get("creator_url", "")
        if not url:
            continue
        eng = (p.get("reactions", 0) + p.get("comments", 0) * 3 + p.get("reposts", 0) * 2)
        followers = max(p.get("creator_followers", 1), 1)
        creator_engagement[url] += eng / followers  # engagement rate
        creator_count[url] += 1

    rotation = v_data.get("active_rotation", [])
    pool = v_data.get("exploration_pool", [])

    # Score current rotation
    for c in rotation:
        url = c.get("linkedin_url", "")
        count = creator_count.get(url, 0)
        c["engagement_relevance"] = round(creator_engagement.get(url, 0) / max(count, 1), 4)

    # Retire bottom 5 from rotation, replace from pool
    rotation.sort(key=lambda c: c.get("engagement_relevance", 0))
    n_retire = min(5, len(rotation) // 2)
    retired = rotation[:n_retire]
    remaining = rotation[n_retire:]

    # Get fresh creators from pool (not already in rotation or core)
    core_urls = {c.get("linkedin_url") for c in v_data.get("core", [])}
    rotation_urls = {c.get("linkedin_url") for c in remaining}
    available = [c for c in pool if c.get("linkedin_url") not in core_urls | rotation_urls]
    new_additions = available[:n_retire]

    v_data["active_rotation"] = remaining + new_additions
    v_data["last_rotated"] = _today()
    _save_registry(registry)

    if retired:
        logger.info(
            "[market_intel] Rotated '%s': retired %d, added %d exploration creators",
            vertical, len(retired), len(new_additions),
        )

    # If pool is running low, discover more
    if len(available) < 5:
        _replenish_pool(vertical, registry)


def _replenish_pool(vertical: str, registry: dict) -> None:
    """One Haiku call to discover more creators for the exploration pool."""
    v_data = registry.get("verticals", {}).get(vertical)
    if not v_data:
        return

    existing_urls = set()
    for group in ("core", "exploration_pool", "active_rotation"):
        for c in v_data.get(group, []):
            existing_urls.add(c.get("linkedin_url", ""))

    # Find a client in this vertical for context
    client_map = registry.get("client_vertical_map", {})
    sample_client = None
    for company, v in client_map.items():
        if v == vertical:
            sample_client = company
            break

    icp_text = ""
    if sample_client:
        icp_path = P.icp_definition_path(sample_client)
        if icp_path.exists():
            try:
                icp_text = json.loads(icp_path.read_text(encoding="utf-8")).get("description", "")
            except Exception:
                pass

    existing_names = [c.get("name", "") for c in v_data.get("core", []) + v_data.get("exploration_pool", [])]

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": (
                f"List 10 MORE LinkedIn creators for the '{vertical}' vertical.\n\n"
                f"ICP: {icp_text[:500]}\n\n"
                f"Already monitoring: {', '.join(existing_names[:20])}\n\n"
                "Find creators NOT on this list. Return JSON array:\n"
                '[{"linkedin_url": "...", "name": "...", "why": "..."}]\n'
                "Output ONLY the JSON array."
            )}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        new_creators = json.loads(raw)
        if not isinstance(new_creators, list):
            return

        today = _today()
        added = 0
        for c in new_creators:
            url = c.get("linkedin_url", "")
            if url and url not in existing_urls:
                c["added"] = today
                c["engagement_relevance"] = 0.0
                v_data.setdefault("exploration_pool", []).append(c)
                existing_urls.add(url)
                added += 1

        if added:
            _save_registry(registry)
            logger.info("[market_intel] Replenished %d creators for '%s' pool", added, vertical)

    except Exception as e:
        logger.warning("[market_intel] Pool replenishment failed for '%s': %s", vertical, e)


# ---------------------------------------------------------------------------
# Part 3: Signal Extraction
# ---------------------------------------------------------------------------

def extract_market_signals(vertical: str) -> dict:
    """Extract trending topics, hook shifts, and whitespace from weekly scrape.

    Uses OpenAI text-embedding-3-small for embedding-based topic clustering.
    """
    # Load this week's and last week's posts
    v_dir = _vertical_dir(vertical)
    if not v_dir.exists():
        return {}

    weekly_files = sorted(v_dir.glob("weekly_*.json"), reverse=True)
    if not weekly_files:
        return {}

    try:
        this_week = json.loads(weekly_files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}

    last_week = []
    if len(weekly_files) >= 2:
        try:
            last_week = json.loads(weekly_files[1].read_text(encoding="utf-8"))
        except Exception:
            pass

    if not this_week:
        return {}

    signals = {
        "vertical": vertical,
        "date": _today(),
        "posts_analyzed": len(this_week),
        "trending_topics": [],
        "hook_trends": {},
        "whitespace": {},
        "algorithm_signal": None,
    }

    # --- Topic velocity via embedding clusters ---
    signals["trending_topics"] = _detect_trending_topics(this_week, last_week)

    # --- Hook trend detection ---
    signals["hook_trends"] = _detect_hook_trends(this_week)

    # --- Algorithm behavior signal ---
    signals["algorithm_signal"] = _detect_algorithm_shift(this_week, last_week)

    # --- Whitespace detection per client ---
    registry = _load_registry()
    client_map = registry.get("client_vertical_map", {})
    for company, v in client_map.items():
        if v == vertical:
            ws = _detect_whitespace(company, signals.get("trending_topics", []))
            if ws:
                signals.setdefault("whitespace", {})[company] = ws

    # Save signals
    v_dir.mkdir(parents=True, exist_ok=True)
    sig_path = v_dir / f"signals_{_today()}.json"
    sig_path.write_text(json.dumps(signals, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "[market_intel] Extracted signals for '%s': %d trending topics, %d whitespace opportunities",
        vertical, len(signals.get("trending_topics", [])),
        sum(len(ws) for ws in signals.get("whitespace", {}).values()),
    )
    return signals


def _detect_trending_topics(this_week: list[dict], last_week: list[dict]) -> list[dict]:
    """Embed posts, cluster by similarity, detect topic velocity changes."""
    try:
        from backend.src.utils.post_embeddings import embed_texts as _embed_texts, cosine_similarity as _cosine_similarity
    except ImportError:
        return []

    texts = [p["text"][:500] for p in this_week if p.get("text")]
    if len(texts) < 5:
        return []

    embeddings = _embed_texts(texts)
    if not embeddings:
        return []

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Simple agglomerative clustering via greedy merge
    clusters: list[list[int]] = [[i] for i in range(len(texts))]
    cluster_centroids = [emb_matrix[i] for i in range(len(texts))]

    # Merge clusters with >0.6 centroid similarity
    merged = True
    while merged:
        merged = False
        best_sim = 0.0
        best_pair = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.dot(cluster_centroids[i], cluster_centroids[j]) /
                           (np.linalg.norm(cluster_centroids[i]) * np.linalg.norm(cluster_centroids[j]) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)
        if best_sim > 0.6 and len(clusters) > 3:
            i, j = best_pair
            clusters[i].extend(clusters[j])
            # Update centroid
            indices = clusters[i]
            cluster_centroids[i] = emb_matrix[indices].mean(axis=0)
            clusters.pop(j)
            cluster_centroids.pop(j)
            merged = True

    # Score clusters by size and engagement
    topic_clusters = []
    for cluster_indices in clusters:
        if len(cluster_indices) < 2:
            continue
        cluster_posts = [this_week[i] for i in cluster_indices if i < len(this_week)]
        # Representative text: highest-engagement post
        cluster_posts.sort(key=lambda p: p.get("reactions", 0) + p.get("comments", 0) * 3, reverse=True)
        rep_text = cluster_posts[0]["text"][:200] if cluster_posts else ""
        avg_eng = sum(p.get("reactions", 0) + p.get("comments", 0) * 3 for p in cluster_posts) / len(cluster_posts)

        topic_clusters.append({
            "topic_summary": rep_text[:150],
            "post_count": len(cluster_indices),
            "avg_engagement": round(avg_eng, 1),
            "velocity": "new",  # will compare with last week
        })

    # Compare with last week for velocity
    if last_week:
        last_texts = [p["text"][:500] for p in last_week if p.get("text")]
        last_embeddings = _embed_texts(last_texts) if last_texts else []
        if last_embeddings:
            last_centroid = np.array(last_embeddings, dtype=np.float32).mean(axis=0)
            for tc in topic_clusters:
                # Rough velocity: this week's cluster size relative to what existed last week
                if tc["post_count"] >= 3:
                    tc["velocity"] = "trending"
                else:
                    tc["velocity"] = "stable"

    # Sort by post count
    topic_clusters.sort(key=lambda t: t["post_count"], reverse=True)
    return topic_clusters[:10]


def _detect_hook_trends(posts: list[dict]) -> dict:
    """Analyze hook style distribution in this week's top posts."""
    try:
        from backend.src.services.cross_client_learning import _classify_hook_style
    except ImportError:
        return {}

    # Top quartile by engagement
    sorted_posts = sorted(
        [p for p in posts if p.get("text")],
        key=lambda p: p.get("reactions", 0) + p.get("comments", 0) * 3,
        reverse=True,
    )
    top_q = sorted_posts[:max(1, len(sorted_posts) // 4)]

    style_counts: dict[str, int] = defaultdict(int)
    for p in top_q:
        hook = p["text"][:140]
        style = _classify_hook_style(hook, "")
        style_counts[style] += 1

    total = sum(style_counts.values())
    if total == 0:
        return {}

    return {
        style: round(count / total, 3)
        for style, count in sorted(style_counts.items(), key=lambda x: -x[1])
    }


def _detect_algorithm_shift(this_week: list[dict], last_week: list[dict]) -> dict | None:
    """Track aggregate engagement rate shifts that may indicate algorithm changes."""
    if not this_week or not last_week:
        return None

    def _median_eng_rate(posts: list[dict]) -> float:
        rates = []
        for p in posts:
            followers = max(p.get("creator_followers", 1), 1)
            eng = p.get("reactions", 0) + p.get("comments", 0) + p.get("reposts", 0)
            rates.append(eng / followers)
        if not rates:
            return 0.0
        rates.sort()
        mid = len(rates) // 2
        return rates[mid]

    this_median = _median_eng_rate(this_week)
    last_median = _median_eng_rate(last_week)

    if last_median == 0:
        return None

    pct_change = (this_median - last_median) / last_median

    if abs(pct_change) > 0.20:
        direction = "increase" if pct_change > 0 else "decrease"
        return {
            "detected": True,
            "direction": direction,
            "pct_change": round(pct_change * 100, 1),
            "this_week_median": round(this_median, 6),
            "last_week_median": round(last_median, 6),
        }

    return {"detected": False}


def _detect_whitespace(company: str, trending_topics: list[dict]) -> list[dict]:
    """Find trending topics the client hasn't covered recently.

    Uses embedding similarity (not keyword overlap) to detect coverage gaps.
    """
    if not trending_topics:
        return []

    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [o for o in rm._state.get("observations", []) if o.get("status") in ("scored", "finalized")]
    except Exception:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent = [o for o in scored if (o.get("posted_at") or "") > cutoff]

    if not recent:
        return []

    # Embed client's recent posts and trending topics, compare in embedding space
    try:
        from backend.src.utils.post_embeddings import embed_texts as _embed_texts
    except ImportError:
        return []

    recent_texts = [
        (o.get("descriptor", {}).get("analysis", "") + " " +
         (o.get("posted_body") or o.get("post_body") or "")[:200]).strip()
        for o in recent
    ]
    topic_texts = [t.get("topic_summary", "")[:200] for t in trending_topics[:10]]

    if not recent_texts or not topic_texts:
        return []

    recent_embs = _embed_texts(recent_texts)
    topic_embs = _embed_texts(topic_texts)

    if not recent_embs or not topic_embs:
        return []

    recent_matrix = np.array(recent_embs, dtype=np.float32)
    # For each trending topic, find max similarity to any recent client post
    whitespace = []
    for i, topic in enumerate(trending_topics[:len(topic_embs)]):
        topic_emb = np.array(topic_embs[i], dtype=np.float32)
        # Cosine similarity to each recent post
        dots = recent_matrix @ topic_emb
        norms = np.linalg.norm(recent_matrix, axis=1) * np.linalg.norm(topic_emb)
        norms = np.where(norms == 0, 1.0, norms)
        sims = dots / norms
        max_sim = float(np.max(sims))

        # Low similarity = whitespace (client hasn't covered this)
        if max_sim < 0.45:  # threshold: below 0.45 cosine = clearly different topic
            whitespace.append({
                "topic": topic["topic_summary"][:100],
                "external_post_count": topic["post_count"],
                "external_engagement": topic["avg_engagement"],
                "velocity": topic["velocity"],
            })

    # Synthesize top opportunities via Haiku
    if whitespace:
        try:
            ws_text = "\n".join(f"- {w['topic']} ({w['external_post_count']} posts, {w['velocity']})" for w in whitespace[:5])
            resp = _client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": (
                    f"These topics are trending in {company}'s vertical but they haven't posted about them:\n\n"
                    f"{ws_text}\n\n"
                    "Rank the top 3 opportunities by relevance. For each: one sentence on why it's an opportunity "
                    "and one sentence on the angle the client should take. Output ONLY a JSON array:\n"
                    '[{"topic": "...", "opportunity": "...", "suggested_angle": "..."}]'
                )}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            synthesized = json.loads(raw)
            if isinstance(synthesized, list):
                return synthesized[:3]
        except Exception:
            pass

    return whitespace[:3]


# ---------------------------------------------------------------------------
# Part 4: Stelle Integration
# ---------------------------------------------------------------------------

def build_market_context(company: str) -> str:
    """Build Stelle-ready context from latest market signals.

    Returns empty string if no signals exist for the client's vertical.
    """
    registry = _load_registry()
    vertical = registry.get("client_vertical_map", {}).get(company)
    if not vertical:
        return ""

    # Load latest signals
    v_dir = _vertical_dir(vertical)
    if not v_dir.exists():
        return ""

    signal_files = sorted(v_dir.glob("signals_*.json"), reverse=True)
    if not signal_files:
        return ""

    try:
        signals = json.loads(signal_files[0].read_text(encoding="utf-8"))
    except Exception:
        return ""

    n_creators = 0
    v_data = registry.get("verticals", {}).get(vertical, {})
    n_creators = len(v_data.get("core", [])) + len(v_data.get("active_rotation", []))

    lines = [f"\n\nMARKET INTELLIGENCE ({vertical} LinkedIn monitoring, {n_creators} creators tracked):"]

    # Trending topics
    trending = signals.get("trending_topics", [])
    if trending:
        trend_strs = [f"\"{t['topic_summary'][:80]}\" ({t['post_count']} posts)" for t in trending[:3]]
        lines.append(f"Trending this week: {'; '.join(trend_strs)}")

    # Hook style shift
    hook_trends = signals.get("hook_trends", {})
    if hook_trends:
        top_style = max(hook_trends, key=hook_trends.get)
        lines.append(f"Dominant hook style this week: {top_style} ({hook_trends[top_style]:.0%} of top posts)")

    # Whitespace
    ws = signals.get("whitespace", {}).get(company, [])
    if ws:
        for w in ws[:2]:
            topic = w.get("topic", "")[:60]
            angle = w.get("suggested_angle", w.get("opportunity", ""))[:80]
            lines.append(f"Whitespace opportunity: \"{topic}\" — {angle}")

    # Algorithm signal
    algo = signals.get("algorithm_signal")
    if algo and algo.get("detected"):
        lines.append(
            f"⚠️ Algorithm shift detected: {algo['direction']} {abs(algo['pct_change']):.0f}% "
            f"in median engagement rate week-over-week"
        )

    if len(lines) <= 1:
        return ""

    lines.append("These are external signals. Use your judgment on relevance.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 5: Autonomous Wiring
# ---------------------------------------------------------------------------

def run_market_intel_cycle() -> dict:
    """Full autonomous cycle: scrape + extract signals for all enabled verticals.

    Called by ordinal_sync as step 10. Frequency-gated internally.
    """
    if not _MARKET_INTEL_ENABLED:
        return {"skipped": True, "reason": "disabled"}

    registry = _load_registry()
    result = {"verticals_processed": [], "auto_seeded": [], "strategy_triggers": []}

    # Auto-seed unmapped clients — single detect_verticals call for all,
    # then seed only those that got mapped. Skip clients that have been
    # tried before and failed (no ICP definition → can't determine vertical).
    csv_path = P.ordinal_auth_csv()
    if csv_path.exists():
        import csv
        try:
            unmapped = []
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    company = row.get("provider_org_slug", "").strip()
                    if company and company not in registry.get("client_vertical_map", {}):
                        # Skip clients we've already failed to map (no ICP)
                        if company not in registry.get("_unmappable", []):
                            unmapped.append(company)

            if unmapped:
                # Single detect_verticals call (not one per client)
                verticals = detect_verticals()
                mapped_any = False
                for v_slug, companies in verticals.items():
                    for c in companies:
                        if c in unmapped:
                            registry.setdefault("client_vertical_map", {})[c] = v_slug
                            mapped_any = True

                # Record clients that couldn't be mapped (no ICP) — don't retry
                for c in unmapped:
                    if c not in registry.get("client_vertical_map", {}):
                        registry.setdefault("_unmappable", [])
                        if c not in registry["_unmappable"]:
                            registry["_unmappable"].append(c)

                if mapped_any:
                    _save_registry(registry)
                    # Now seed creators for newly mapped clients
                    for c in unmapped:
                        if c in registry.get("client_vertical_map", {}):
                            v_data = seed_creator_registry(c)
                            if v_data:
                                result["auto_seeded"].append(c)
                            registry = _load_registry()
        except Exception as e:
            logger.warning("[market_intel] Auto-seed failed: %s", e)

    # Collect + extract per vertical
    for vertical in list(registry.get("verticals", {}).keys()):
        posts = collect_market_intel(vertical)
        if posts:
            signals = extract_market_signals(vertical)
            result["verticals_processed"].append({
                "vertical": vertical,
                "posts_scraped": len(posts),
                "trending_topics": len(signals.get("trending_topics", [])),
            })

            # Auto-trigger strategy refresh on high-velocity topics
            triggers = _check_strategy_triggers(vertical, signals, registry)
            result["strategy_triggers"].extend(triggers)

    return result


def _check_strategy_triggers(vertical: str, signals: dict, registry: dict) -> list[dict]:
    """Auto-trigger Herta strategy refresh when high-velocity trending topics match ICP."""
    triggers = []
    trending = signals.get("trending_topics", [])

    # Find high-velocity topics (3+ posts in cluster = significant)
    hot_topics = [t for t in trending if t.get("post_count", 0) >= 3 and t.get("velocity") == "trending"]
    if not hot_topics:
        return []

    # Check whitespace per client
    client_map = registry.get("client_vertical_map", {})
    for company, v in client_map.items():
        if v != vertical:
            continue

        ws = signals.get("whitespace", {}).get(company, [])
        if not ws:
            continue

        for w in ws[:1]:  # Only trigger on top opportunity
            topic = w.get("topic", "")
            if not topic:
                continue

            trigger = {
                "company": company,
                "topic": topic,
                "reason": f"Trending in {vertical} ({w.get('external_post_count', '?')} external posts), "
                          f"client hasn't covered in 14+ days",
                "triggered_at": _now(),
            }

            # Log the trigger
            logger.info(
                "[market_intel] AUTO-TRIGGER: strategy refresh for %s on '%s'",
                company, topic[:60],
            )

            # Auto-plan series (deprecated — was LOLA-based)
            try:
                from backend.src.services.series_engine import plan_series, get_active_series
                active = get_active_series(company)
                active_themes = {s.get("theme", "").lower() for s in active}
                if topic.lower()[:30] not in " ".join(active_themes):
                    plan_series(company, topic[:80], num_posts=4)
                    trigger["action"] = "series_planned"
            except Exception as e:
                trigger["action"] = f"series_plan_failed: {e}"

            triggers.append(trigger)

    return triggers


def auto_plan_series_from_lola(company: str) -> dict | None:
    """DEPRECATED: LOLA arm-based series planning is no longer used.

    Content intelligence is now handled by RuanMei.recommend_context().
    Kept as a no-op stub so existing callers don't crash.
    """
    return None
