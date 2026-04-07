"""Cross-Client Learning Network — transfer learning across all Amphoreus clients.

Three components:
1. **Universal pattern extraction** — structured patterns from top performers
   across 2+ clients, stored in memory/our_memory/universal_patterns.json.
2. **Cross-client hook library** — top-quartile hooks with metadata, stored
   in memory/our_memory/hook_library.json.
3. **Cold-start LOLA seeding** — auto-generate seed arms for new clients
   from universal patterns + client ICP.

Wired into ordinal_sync as step 9 (after series health check).

Usage:
    from backend.src.services.cross_client_learning import (
        refresh_universal_patterns,
        refresh_hook_library,
        auto_seed_lola,
        load_hook_library_for_stelle,
    )

    # In ordinal_sync (runs hourly):
    refresh_universal_patterns()
    refresh_hook_library()

    # When LOLA initializes with 0 arms:
    auto_seed_lola("new-client")

    # Stelle context:
    hooks = load_hook_library_for_stelle(industry="biotech", limit=10)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

def _our_memory() -> Path:
    return P.our_memory_dir()


def _patterns_path() -> Path:
    return _our_memory() / "universal_patterns.json"


def _hook_library_path() -> Path:
    return _our_memory() / "hook_library.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# 1. Universal pattern extraction
# ------------------------------------------------------------------

def _load_all_scored() -> dict[str, list[dict]]:
    """Load scored observations for every client. Returns {company: [obs]}."""
    result: dict[str, list[dict]] = {}
    if not P.MEMORY_ROOT.exists():
        return result
    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        state_path = d / "ruan_mei_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            scored = [o for o in state.get("observations", []) if o.get("status") == "scored"]
            if len(scored) >= 5:
                result[d.name] = scored
        except Exception:
            continue
    return result


def _top_quartile(scored: list[dict]) -> list[dict]:
    """Return top 25% by immediate reward."""
    scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
    cutoff = int(len(scored) * 0.75)
    return scored[cutoff:]


def refresh_universal_patterns() -> list[dict]:
    """Scan all clients, extract universal patterns, persist to JSON.

    A pattern is "universal" if it appears in top-25% performers across 2+ clients.
    Uses Claude Haiku to cluster and deduplicate patterns from post analyses.

    Returns the list of patterns written.
    """
    all_scored = _load_all_scored()
    if len(all_scored) < 2:
        return []

    # Collect top analyses with client tag (anonymized as client_1, client_2, etc)
    top_entries: list[dict] = []
    client_map: dict[str, str] = {}
    for i, (company, scored) in enumerate(sorted(all_scored.items())):
        client_map[company] = f"client_{i+1}"
        top = _top_quartile(scored)
        for o in top:
            analysis = o.get("descriptor", {}).get("analysis", "")
            if not analysis:
                continue
            reward = o.get("reward", {}).get("immediate", 0)
            impressions = o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0)
            icp_rate = o.get("icp_match_rate")
            top_entries.append({
                "client": client_map[company],
                "analysis": analysis[:300],
                "reward": round(reward, 3),
                "impressions": impressions,
                "icp_match_rate": round(icp_rate, 2) if icp_rate is not None else None,
            })

    if len(top_entries) < 10:
        return []

    # Sample to keep prompt manageable
    top_entries.sort(key=lambda e: e["reward"], reverse=True)
    sample_size = min(60, max(20, len(top_entries)))  # scale with data, cap at 60
    sample = top_entries[:sample_size]

    parts = []
    for e in sample:
        header = f"[{e['client']} | reward={e['reward']:.3f} | impressions={e['impressions']}"
        if e.get("icp_match_rate") is not None:
            header += f" | icp_rate={e['icp_match_rate']}"
        header += "]"
        parts.append(f"{header}\n{e['analysis']}")
    entries_text = "\n\n".join(parts)

    prompt = (
        f"You are analyzing {len(top_entries)} top-performing LinkedIn posts across "
        f"{len(all_scored)} anonymous B2B clients.\n\n"
        "Extract UNIVERSAL PATTERNS — writing mechanics that appear in top performers "
        "across MULTIPLE clients (not client-specific topics).\n\n"
        f"TOP PERFORMERS:\n{entries_text}\n\n"
        "Return a JSON array of patterns. Each pattern:\n"
        "{\n"
        '  "pattern": "Posts that open with a specific number or metric consistently outperform",\n'
        '  "evidence_clients": 4,\n'
        '  "avg_reward_lift": 0.35,\n'
        '  "confidence": 0.85,\n'
        '  "category": "hook|structure|storytelling|specificity|format|engagement_driver"\n'
        "}\n\n"
        "Extract 5-10 patterns. Only include patterns supported by 2+ clients. "
        "Be specific about the writing mechanic, not vague ('good hooks' is useless; "
        "'opening with a concrete number before the thesis' is useful). "
        "Output ONLY the JSON array."
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        patterns = json.loads(raw)
        if not isinstance(patterns, list):
            patterns = []
    except Exception as e:
        logger.warning("[cross_client] Pattern extraction failed: %s", e)
        return []

    # Validate and normalize
    valid: list[dict] = []
    for p in patterns:
        if not p.get("pattern"):
            continue
        valid.append({
            "pattern": p["pattern"],
            "evidence_clients": max(2, int(p.get("evidence_clients", 2))),
            "avg_reward_lift": round(float(p.get("avg_reward_lift", 0)), 3),
            "confidence": round(max(0, min(1, float(p.get("confidence", 0.5)))), 2),
            "category": p.get("category", "general"),
            "updated_at": _now(),
            "source_observations": len(top_entries),
            "source_clients": len(all_scored),
        })

    # Persist
    path = _patterns_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "[cross_client] Extracted %d universal patterns from %d clients (%d observations)",
        len(valid), len(all_scored), len(top_entries),
    )
    return valid


def load_universal_patterns() -> list[dict]:
    """Load persisted universal patterns."""
    path = _patterns_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# ------------------------------------------------------------------
# 2. Cross-client hook library
# ------------------------------------------------------------------

def refresh_hook_library() -> list[dict]:
    """Extract hooks from top-quartile posts, persist with metadata.

    Hook = first 140 chars of posted_body or post_body (the "above the fold" text).
    """
    all_scored = _load_all_scored()
    if not all_scored:
        return []

    hooks: list[dict] = []
    seen_hooks: set[str] = set()

    for company, scored in all_scored.items():
        top = _top_quartile(scored)
        for o in top:
            body = (o.get("posted_body") or o.get("post_body") or "").strip()
            if not body or len(body) < 50:
                continue

            # Extract hook: first line or first 140 chars
            first_line = body.split("\n")[0].strip()
            hook = first_line[:140] if len(first_line) > 10 else body[:140]

            # Deduplicate
            hook_key = hook[:80].lower()
            if hook_key in seen_hooks:
                continue
            seen_hooks.add(hook_key)

            analysis = o.get("descriptor", {}).get("analysis", "")
            reward = o.get("reward", {}).get("immediate", 0)
            impressions = o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0)
            icp_rate = o.get("icp_match_rate")

            # Infer hook style from analysis
            hook_style = _classify_hook_style(hook, analysis)

            hooks.append({
                "hook": hook,
                "hook_style": hook_style,
                "engagement_score": round(reward, 3),
                "impressions": impressions,
                "icp_match_rate": round(icp_rate, 2) if icp_rate is not None else None,
                "char_count": len(body),
                "company_anonymized": False,  # we keep company for internal use
                "company": company,
            })

    # Sort by engagement
    hooks.sort(key=lambda h: h["engagement_score"], reverse=True)

    # Scale hook library with data, cap at 400
    hook_cap = min(400, max(100, len(hooks)))
    hooks = hooks[:hook_cap]

    path = _hook_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hooks, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("[cross_client] Hook library: %d hooks from %d clients", len(hooks), len(all_scored))
    return hooks


def _classify_hook_style(hook: str, analysis: str) -> str:
    """Classify hook style from text heuristics. No LLM call — must be fast."""
    h = hook.lower()

    # Number-led
    for char in h:
        if char.isdigit():
            return "number_led"
        if char.isalpha():
            break

    if any(h.startswith(w) for w in ("i ", "i'm ", "i've ", "my ", "when i ")):
        return "personal_story"

    if "?" in hook[:80]:
        return "question"

    if any(w in h[:60] for w in ("most ", "everyone ", "nobody ", "the biggest ", "the real ")):
        return "contrarian"

    if any(w in h[:60] for w in ("ceo", "cto", "vp ", "director", "founder", "engineer")):
        return "icp_callout"

    # Check analysis for clues
    a = analysis.lower()
    if "narrative" in a or "story" in a or "anecdot" in a:
        return "story_climax"
    if "specifi" in a and "number" in a:
        return "number_led"

    return "declarative"


def load_hook_library(limit: int = 50, hook_style: str | None = None) -> list[dict]:
    """Load hook library, optionally filtered by style."""
    path = _hook_library_path()
    if not path.exists():
        return []
    try:
        hooks = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if hook_style:
        hooks = [h for h in hooks if h.get("hook_style") == hook_style]

    return hooks[:limit]


def load_hook_library_for_stelle(company: str = "", limit: int = 15) -> str:
    """Build a Stelle-ready context string from the hook library.

    Two modes:
    - If LOLA has a content direction for the client: retrieve hooks by
      embedding similarity to that direction (relevant exemplars).
    - Fallback: return top hooks by engagement score (globally best).

    Excludes same-client hooks to avoid self-plagiarism.
    """
    hooks = load_hook_library(limit=limit * 4)
    if not hooks:
        return ""

    # Exclude same-client hooks
    if company:
        hooks = [h for h in hooks if h.get("company") != company]

    if not hooks:
        return ""

    # Try embedding-based retrieval: find hooks similar to client's content direction
    selected = _retrieve_hooks_by_embedding(company, hooks, limit)
    if not selected:
        # Fallback: top by engagement
        selected = hooks[:limit]

    lines = ["\n\nHOOK REFERENCE LIBRARY (top-performing hooks, relevance-ranked):"]
    for h in selected[:10]:
        score = h.get("engagement_score", 0)
        lines.append(f'  (score {score:.2f}) "{h["hook"]}"')

    lines.append(
        "\nStudy these hooks for scroll-stop patterns. Do NOT copy them — "
        "apply the structural patterns to this client's material."
    )
    return "\n".join(lines)


def _retrieve_hooks_by_embedding(company: str, hooks: list[dict], limit: int) -> list[dict]:
    """Retrieve hooks most relevant to the client's content direction.

    Uses LOLA's continuous reward field centroid as the query vector.
    Falls back to empty list if embeddings unavailable.
    """
    try:
        from backend.src.agents.lola import LOLA, _embed_texts
        import numpy as np
    except ImportError:
        return []

    lola = LOLA(company)
    points = lola._get_points()
    if len(points) < 5:
        return []

    # Compute content direction: reward-weighted centroid
    embeddings = [p.embedding for p in points if p.embedding]
    rewards = [max(p.reward, 0) for p in points if p.embedding]
    if not embeddings:
        return []

    emb_matrix = np.array(embeddings, dtype=np.float32)
    reward_weights = np.array(rewards, dtype=np.float32) + 1e-8
    centroid = np.average(emb_matrix, axis=0, weights=reward_weights)

    # Embed hooks
    hook_texts = [h["hook"] for h in hooks]
    hook_embs = _embed_texts(hook_texts)
    if not hook_embs or len(hook_embs) != len(hooks):
        return []

    # Score by cosine similarity to content direction
    hook_matrix = np.array(hook_embs, dtype=np.float32)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    hook_norms = hook_matrix / (np.linalg.norm(hook_matrix, axis=1, keepdims=True) + 1e-8)
    similarities = hook_norms @ centroid_norm

    # Blend similarity with engagement score for final ranking
    eng_scores = np.array([h.get("engagement_score", 0) for h in hooks], dtype=np.float32)
    eng_norm = eng_scores / (np.max(np.abs(eng_scores)) + 1e-8)
    final_scores = 0.6 * similarities + 0.4 * eng_norm

    # Sort and return top-k
    ranked_indices = np.argsort(final_scores)[::-1][:limit]
    return [hooks[i] for i in ranked_indices]


# ------------------------------------------------------------------
# 3. Cold-start LOLA seeding
# ------------------------------------------------------------------

def auto_seed_lola(company: str) -> int:
    """Auto-generate LOLA seed arms for a new client with no arms.

    Priority order:
    1. Manual override (topic_arms.json)
    2. Cross-client cold-start seeds (proven arms from similar client)
    3. LLM-generated from universal patterns + client ICP

    Returns number of arms seeded.
    """
    from backend.src.agents.lola import LOLA

    lola = LOLA(company)
    if lola._state.arms:
        return 0  # Already has arms

    # Check for topic_arms.json first (manual override)
    arms_path = P.memory_dir(company) / "topic_arms.json"
    if arms_path.exists():
        try:
            manual_arms = json.loads(arms_path.read_text(encoding="utf-8"))
            if manual_arms:
                return lola.seed_arms(manual_arms)
        except Exception:
            pass

    # Cross-client cold-start: seed from similar client's proven arms
    try:
        from backend.src.utils.cross_client import get_cold_start_seeds
        seeds = get_cold_start_seeds(company)
        if seeds and seeds.get("lola_arms"):
            seed_arms = [
                {
                    "label": a["label"],
                    "arm_type": a.get("arm_type", "topic"),
                    "description": a.get("description", ""),
                }
                for a in seeds["lola_arms"]
                if a.get("label") and a.get("description")
            ]
            if seed_arms:
                seeded = lola.seed_arms(seed_arms)
                if seeded:
                    logger.info(
                        "[cross_client] Cold-start seeded %d arms for %s from %s",
                        seeded, company, seeds.get("source_client", "?"),
                    )
                    return seeded
    except Exception as e:
        logger.debug("[cross_client] Cold-start seed failed for %s: %s", company, e)

    patterns = load_universal_patterns()
    if not patterns:
        logger.info("[cross_client] No universal patterns for LOLA seeding of %s", company)
        return 0

    # Load client context
    icp_text = ""
    icp_path = P.icp_definition_path(company)
    if icp_path.exists():
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            icp_text = icp.get("description", "")
        except Exception:
            pass

    strategy_text = ""
    cs_dir = P.content_strategy_dir(company)
    if cs_dir.exists():
        for f in sorted(cs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
            if f.suffix in (".md", ".txt"):
                try:
                    strategy_text = f.read_text(encoding="utf-8", errors="replace")[:1500]
                except Exception:
                    pass

    transcript_snippet = ""
    t_dir = P.transcripts_dir(company)
    if t_dir.exists():
        for f in sorted(t_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
            if f.suffix in (".txt", ".md"):
                try:
                    transcript_snippet = f.read_text(encoding="utf-8", errors="replace")[:1000]
                except Exception:
                    pass

    patterns_text = "\n".join(
        f"- [{p['category']}] {p['pattern']} (confidence {p['confidence']}, "
        f"reward lift +{p['avg_reward_lift']:.2f})"
        for p in patterns[:10]
    )

    prompt = (
        "You are initializing a content exploration system for a new LinkedIn client.\n\n"
        f"UNIVERSAL PATTERNS (proven across multiple B2B clients):\n{patterns_text}\n\n"
    )
    if icp_text:
        prompt += f"CLIENT ICP:\n{icp_text}\n\n"
    if strategy_text:
        prompt += f"CLIENT STRATEGY (excerpt):\n{strategy_text[:800]}\n\n"
    if transcript_snippet:
        prompt += f"CLIENT TRANSCRIPT (excerpt):\n{transcript_snippet[:600]}\n\n"

    prompt += (
        "Generate 6-8 topic arms and 3-4 format arms for this client. "
        "Each arm should combine a universal pattern with this client's specific domain.\n\n"
        "Return a JSON array:\n"
        "[\n"
        '  {"label": "protocol_error_storytelling", "arm_type": "topic", '
        '"description": "Personal stories about protocol errors and what they taught"},\n'
        '  {"label": "carousel_data", "arm_type": "format", '
        '"description": "Document/carousel format with data visualization"},\n'
        "  ...\n"
        "]\n\n"
        "Labels should be snake_case, 2-4 words. Descriptions should be specific "
        "to this client's domain, not generic. Output ONLY the JSON array."
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        arms = json.loads(raw)
        if not isinstance(arms, list):
            return 0
    except Exception as e:
        logger.warning("[cross_client] LOLA auto-seed failed for %s: %s", company, e)
        return 0

    # Validate and seed
    valid_arms = []
    for a in arms:
        label = a.get("label", "").strip()
        arm_type = a.get("arm_type", "topic").strip()
        description = a.get("description", "").strip()
        if label and description and arm_type in ("topic", "format"):
            valid_arms.append({
                "label": label,
                "arm_type": arm_type,
                "description": description,
            })

    if not valid_arms:
        return 0

    seeded = lola.seed_arms(valid_arms)
    if seeded:
        logger.info("[cross_client] Auto-seeded %d LOLA arms for %s", seeded, company)

    return seeded


# ------------------------------------------------------------------
# ordinal_sync integration — single entry point for step 9
# ------------------------------------------------------------------

def run_cross_client_sync() -> dict:
    """Run all cross-client learning tasks. Called by ordinal_sync as step 9.

    Returns summary dict.
    """
    result = {
        "patterns": 0,
        "hooks": 0,
        "seeded_clients": [],
    }

    # 1. Refresh universal patterns
    try:
        patterns = refresh_universal_patterns()
        result["patterns"] = len(patterns)
    except Exception as e:
        logger.warning("[cross_client] Pattern refresh failed: %s", e)

    # 2. Refresh hook library
    try:
        hooks = refresh_hook_library()
        result["hooks"] = len(hooks)
    except Exception as e:
        logger.warning("[cross_client] Hook library refresh failed: %s", e)

    # 3. Auto-seed LOLA for any client with 0 arms
    if not P.MEMORY_ROOT.exists():
        return result

    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        company = d.name
        try:
            from backend.src.agents.lola import LOLA
            lola = LOLA(company)
            if not lola._state.arms:
                seeded = auto_seed_lola(company)
                if seeded:
                    result["seeded_clients"].append({"company": company, "arms": seeded})
        except Exception:
            continue

    return result
