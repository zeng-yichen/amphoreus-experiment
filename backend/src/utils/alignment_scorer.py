"""360Brew alignment scorer — pre-publish semantic consistency check.

LinkedIn's 360Brew ranking model (arXiv 2501.16450) cross-references every
post against the author's established topical authority via its LLaMA 3
dual-encoder retrieval system (arXiv 2510.14223). Posts that drift from the
author's pillars get suppressed before entering the ranking pipeline.

Two scoring modes:
1. **Embedding-based** (primary) — cosine similarity between draft/source
   embedding and a cached client identity fingerprint built from accepted
   posts, content strategy, LinkedIn profile, and voice examples.
2. **LLM-based** (fallback) — Claude Haiku semantic evaluation when
   embeddings are unavailable.

Score is in [0.0, 1.0]:
  > 0.75  strong alignment  — high reach expected
  0.60-0.75 moderate        — some suppression risk
  < 0.60  drift risk        — significant reach penalty

The fingerprint embedding is cached for 24 hours in SQLite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_TTL = 86400  # 24 hours
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536

# Default thresholds (cold-start). Replaced by learned values when data exists.
_DEFAULT_STRONG_THRESHOLD = 0.75
_DEFAULT_DRIFT_THRESHOLD = 0.60
_MIN_OBS_FOR_ADAPTIVE_ALIGNMENT = 15


# ------------------------------------------------------------------
# Adaptive thresholds — learned from (alignment_score, engagement) pairs
# ------------------------------------------------------------------

class AlignmentAdaptiveConfig:
    """Learn alignment score thresholds from engagement data.

    For each candidate threshold, compute mean engagement of posts above vs
    below. Set strong_threshold where above-threshold posts significantly
    outperform; set drift_threshold where below-threshold posts significantly
    underperform.

    Three-tier cascade: client → aggregate → defaults.
    """

    MODULE_NAME = "alignment"

    def get_defaults(self) -> dict:
        return {
            "strong_threshold": _DEFAULT_STRONG_THRESHOLD,
            "drift_threshold": _DEFAULT_DRIFT_THRESHOLD,
            "_tier": "default",
        }

    def resolve(self, company: str) -> dict:
        """Three-tier cascade with cache check."""
        from backend.src.utils.adaptive_config import AdaptiveConfig as _AC

        # Check cache
        try:
            from backend.src.db import vortex as P
            cache_path = P.memory_dir(company) / "adaptive_config.json"
            if cache_path.exists():
                all_configs = json.loads(cache_path.read_text(encoding="utf-8"))
                entry = all_configs.get(self.MODULE_NAME)
                if entry:
                    from datetime import datetime, timezone
                    computed = entry.get("_computed_at", "")
                    if computed:
                        dt = datetime.fromisoformat(computed.replace("Z", "+00:00"))
                        age_s = (datetime.now(timezone.utc) - dt).total_seconds()
                        if age_s < 3600:
                            return entry
        except Exception:
            pass

        obs = self._get_obs(company)
        if len(obs) >= _MIN_OBS_FOR_ADAPTIVE_ALIGNMENT:
            try:
                config = self._compute(obs)
                config["_tier"] = "client"
                self._cache(company, config)
                return config
            except Exception as e:
                logger.debug("[AlignmentConfig] Client compute failed: %s", e)

        # Aggregate
        all_obs = self._get_all_obs()
        if len(all_obs) >= _MIN_OBS_FOR_ADAPTIVE_ALIGNMENT:
            try:
                config = self._compute(all_obs)
                config["_tier"] = "aggregate"
                self._cache(company, config)
                return config
            except Exception as e:
                logger.debug("[AlignmentConfig] Aggregate compute failed: %s", e)

        return self.get_defaults()

    def recompute(self, company: str) -> dict:
        """Force recompute, ignoring cache."""
        try:
            from backend.src.db import vortex as P
            cache_path = P.memory_dir(company) / "adaptive_config.json"
            if cache_path.exists():
                all_configs = json.loads(cache_path.read_text(encoding="utf-8"))
                all_configs.pop(self.MODULE_NAME, None)
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(all_configs, indent=2), encoding="utf-8")
                tmp.rename(cache_path)
        except Exception:
            pass
        return self.resolve(company)

    def _get_obs(self, company: str) -> list[dict]:
        try:
            from backend.src.agents.ruan_mei import RuanMei
            rm = RuanMei(company)
            return [
                o for o in rm._state.get("observations", [])
                if o.get("status") == "scored"
                and o.get("alignment_score") is not None
                and o.get("reward", {}).get("immediate") is not None
            ]
        except Exception:
            return []

    def _get_all_obs(self) -> list[dict]:
        from backend.src.db import vortex as P
        all_obs = []
        if P.MEMORY_ROOT.exists():
            for d in P.MEMORY_ROOT.iterdir():
                if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory":
                    all_obs.extend(self._get_obs(d.name))
        return all_obs

    def _compute(self, observations: list[dict]) -> dict:
        """Find optimal thresholds by engagement split analysis.

        No hard-coded bounds on where thresholds can land. The search
        covers the full score range present in the data. soft_bound()
        logs anomalies without clipping.
        """
        from backend.src.utils.adaptive_config import soft_bound

        pairs = [
            (o["alignment_score"], o["reward"]["immediate"])
            for o in observations
        ]
        pairs.sort(key=lambda p: p[0])

        scores = [p[0] for p in pairs]
        rewards = [p[1] for p in pairs]

        # Search the full range of observed scores in 0.05 steps
        min_score = scores[0]
        max_score = scores[-1]
        step = 0.05
        candidates = []
        cutoff = min_score + step
        while cutoff < max_score - step:
            candidates.append(round(cutoff, 3))
            cutoff += step

        if not candidates:
            from datetime import datetime, timezone
            return {
                "strong_threshold": _DEFAULT_STRONG_THRESHOLD,
                "drift_threshold": _DEFAULT_DRIFT_THRESHOLD,
                "observation_count": len(observations),
                "_computed_at": datetime.now(timezone.utc).isoformat(),
            }

        # For each candidate, compute the engagement gap between
        # above-cutoff and below-cutoff posts.
        gaps = []
        for cutoff in candidates:
            above = [r for s, r in pairs if s >= cutoff]
            below = [r for s, r in pairs if s < cutoff]

            if len(above) < 3 or len(below) < 3:
                gaps.append(0.0)
                continue

            mean_above = sum(above) / len(above)
            mean_below = sum(below) / len(below)
            gaps.append(mean_above - mean_below)

        # Strong threshold: candidate with highest gap in the upper half
        # of the score range (where "strong" lives).
        midpoint = (min_score + max_score) / 2
        best_strong_idx = None
        best_strong_gap = 0.0
        for i, (c, g) in enumerate(zip(candidates, gaps)):
            if c >= midpoint and g > best_strong_gap:
                best_strong_gap = g
                best_strong_idx = i

        # Drift threshold: candidate with highest gap in the lower half.
        best_drift_idx = None
        best_drift_gap = 0.0
        for i, (c, g) in enumerate(zip(candidates, gaps)):
            if c < midpoint and g > best_drift_gap:
                best_drift_gap = g
                best_drift_idx = i

        best_strong = candidates[best_strong_idx] if best_strong_idx is not None else _DEFAULT_STRONG_THRESHOLD
        best_drift = candidates[best_drift_idx] if best_drift_idx is not None else _DEFAULT_DRIFT_THRESHOLD

        # If drift >= strong, set drift to the next candidate below strong
        if best_drift >= best_strong:
            below_strong = [c for c in candidates if c < best_strong]
            best_drift = below_strong[-1] if below_strong else best_strong - step

        # soft_bound: log anomalies without clipping
        history_strong = [_DEFAULT_STRONG_THRESHOLD]
        history_drift = [_DEFAULT_DRIFT_THRESHOLD]
        best_strong = soft_bound(best_strong, history_strong, _DEFAULT_STRONG_THRESHOLD)
        best_drift = soft_bound(best_drift, history_drift, _DEFAULT_DRIFT_THRESHOLD)

        from datetime import datetime, timezone
        return {
            "strong_threshold": round(best_strong, 2),
            "drift_threshold": round(best_drift, 2),
            "observation_count": len(observations),
            "_computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _cache(self, company: str, config: dict) -> None:
        try:
            from backend.src.db import vortex as P
            cache_path = P.memory_dir(company) / "adaptive_config.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            all_configs = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
            all_configs[self.MODULE_NAME] = config
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(all_configs, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.rename(cache_path)
        except Exception:
            pass


# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------

def _get_embedding(text: str, model: str = _EMBEDDING_MODEL) -> list[float] | None:
    """Get an embedding vector from OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI()
        # Truncate to ~8000 tokens worth of text (~32000 chars)
        truncated = text[:32000]
        resp = client.embeddings.create(
            input=truncated,
            model=model,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.warning("[alignment_scorer] Embedding failed: %s", e)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ------------------------------------------------------------------
# Client fingerprint construction
# ------------------------------------------------------------------

def _load_fingerprint_texts(company: str) -> str:
    """Load all identity-signal texts for the client's fingerprint.

    Priority order (most to least authoritative):
    1. Accepted posts — the gold standard of what this client sounds like
    2. Voice examples (top posts by engagement) — what resonates
    3. Content strategy — explicit topic pillars
    4. LinkedIn profile (headline, about) — what 360Brew sees
    5. Published posts — broader pattern
    """
    from backend.src.db import vortex as P

    parts: list[str] = []

    # 1. Accepted posts (strongest signal)
    accepted = P.accepted_dir(company)
    if accepted.exists():
        for f in sorted(accepted.iterdir()):
            if f.suffix in (".txt", ".md") and f.stat().st_size < 10_000:
                try:
                    parts.append(f.read_text(encoding="utf-8").strip())
                except Exception:
                    pass

    # 2. Content strategy (explicit pillars)
    cs = P.content_strategy_dir(company)
    if cs.exists():
        for f in sorted(cs.iterdir()):
            if f.suffix in (".txt", ".md") and f.stat().st_size < 20_000:
                try:
                    parts.append(f.read_text(encoding="utf-8").strip())
                except Exception:
                    pass

    # 3. LinkedIn profile
    profile_path = P.memory_dir(company) / "profile.json"
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            headline = profile.get("headline", "")
            about = profile.get("about", "")
            if headline:
                parts.append(f"LinkedIn Headline: {headline}")
            if about:
                parts.append(f"LinkedIn About: {about}")
        except Exception:
            pass

    # 4. ICP definition (what audience we're targeting)
    icp_path = P.icp_definition_path(company)
    if icp_path.exists():
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            desc = icp.get("description", "")
            if desc:
                parts.append(f"Target ICP: {desc}")
        except Exception:
            pass

    return "\n\n---\n\n".join(p for p in parts if p)


def _fingerprint_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _get_or_build_fingerprint(company: str) -> list[float] | None:
    """Get cached fingerprint embedding or build a new one."""
    from backend.src.db.local import cache_get, cache_set

    fingerprint_text = _load_fingerprint_texts(company)
    if not fingerprint_text:
        return None

    text_hash = _fingerprint_hash(fingerprint_text)
    cache_key = f"fingerprint_embedding:{company}:{text_hash}"

    # Check cache
    cached = cache_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    # Build new embedding
    embedding = _get_embedding(fingerprint_text)
    if embedding:
        cache_set(cache_key, json.dumps(embedding), ttl_seconds=_CACHE_TTL)
        logger.info("[alignment_scorer] Built fingerprint embedding for %s (%d chars)", company, len(fingerprint_text))

    return embedding


# ------------------------------------------------------------------
# Scoring — embedding-based (primary)
# ------------------------------------------------------------------

def _classify_alignment(similarity: float, company: str = "") -> str:
    """Classify alignment score using learned or default thresholds."""
    config = AlignmentAdaptiveConfig().resolve(company) if company else AlignmentAdaptiveConfig().get_defaults()
    strong = config.get("strong_threshold", _DEFAULT_STRONG_THRESHOLD)
    drift = config.get("drift_threshold", _DEFAULT_DRIFT_THRESHOLD)

    if similarity >= strong:
        return "strong"
    elif similarity >= drift:
        return "moderate"
    else:
        return "drift"


def score_draft_alignment(
    company: str,
    draft_text: str,
) -> dict:
    """Score a draft post against the client's identity fingerprint.

    Returns:
        {
            "score": float,          # [0, 1] cosine similarity
            "label": str,            # "strong" | "moderate" | "drift"
            "method": str,           # "embedding" | "llm" | "skip"
            "summary": str,
            "aligned_topics": list,
            "drift_topics": list,
        }
    """
    if not draft_text or not draft_text.strip():
        return _neutral_result("No draft text provided.", "skip")

    fingerprint = _get_or_build_fingerprint(company)
    if fingerprint is None:
        # Fall back to LLM-based scoring
        return _score_llm_fallback(company, draft_text)

    draft_embedding = _get_embedding(draft_text)
    if draft_embedding is None:
        return _score_llm_fallback(company, draft_text)

    similarity = _cosine_similarity(fingerprint, draft_embedding)
    label = _classify_alignment(similarity, company)

    return {
        "score": round(similarity, 4),
        "label": label,
        "method": "embedding",
        "summary": _label_summary(label, similarity),
        "aligned_topics": [],  # Embedding method doesn't decompose topics
        "drift_topics": [],
    }


def score_batch(company: str, drafts: list[str]) -> list[dict]:
    """Score multiple drafts efficiently (reuses fingerprint)."""
    fingerprint = _get_or_build_fingerprint(company)
    if fingerprint is None:
        return [_score_llm_fallback(company, d) for d in drafts]

    results = []
    for draft in drafts:
        if not draft.strip():
            results.append(_neutral_result("Empty draft.", "skip"))
            continue
        draft_emb = _get_embedding(draft)
        if draft_emb is None:
            results.append(_score_llm_fallback(company, draft))
            continue
        sim = _cosine_similarity(fingerprint, draft_emb)
        label = _classify_alignment(sim, company)
        results.append({
            "score": round(sim, 4),
            "label": label,
            "method": "embedding",
            "summary": _label_summary(label, sim),
            "aligned_topics": [],
            "drift_topics": [],
        })
    return results


def _label_summary(label: str, score: float) -> str:
    if label == "strong":
        return f"Strong profile-content alignment ({score:.2f}). 360Brew reach expected to be high."
    elif label == "moderate":
        return (
            f"Moderate alignment ({score:.2f}). Some reach suppression risk. "
            "Consider anchoring more firmly to established topic pillars."
        )
    else:
        return (
            f"Topic drift detected ({score:.2f}). LinkedIn's retrieval system may suppress "
            "this post before it enters the ranking pipeline. Recommend pivoting to "
            "established pillars."
        )


# ------------------------------------------------------------------
# Scoring — LLM fallback
# ------------------------------------------------------------------

def _score_llm_fallback(company: str, text: str) -> dict:
    """LLM-based alignment scoring when embeddings are unavailable."""
    client_texts = _load_fingerprint_texts(company)
    if not client_texts:
        return _neutral_result("No client fingerprint available.", "skip")

    fp_snippet = client_texts[:3000]
    text_snippet = text[:2000]

    prompt = (
        f"You are analyzing content alignment for a LinkedIn ghostwriting client.\n\n"
        f"CLIENT'S ESTABLISHED CONTENT (accepted posts + strategy):\n{fp_snippet}\n\n"
        f"DRAFT POST OR SOURCE MATERIAL:\n{text_snippet}\n\n"
        "Evaluate how well this content aligns with the client's established "
        "topical pillars, voice, and expertise areas.\n\n"
        "LinkedIn's 360Brew algorithm rewards consistent topical authority. Posts that drift "
        "from established pillars get suppressed even with strong engagement.\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "score": 0.0-1.0,\n'
        '  "label": "strong" | "moderate" | "drift",\n'
        '  "summary": "1-2 sentences explaining alignment or drift",\n'
        '  "aligned_topics": ["topic1", "topic2"],\n'
        '  "drift_topics": ["topic3"]\n'
        "}\n"
        "No markdown, no explanation, just the JSON."
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        result["score"] = max(0.0, min(1.0, float(result.get("score", 0.5))))
        result["method"] = "llm"
        return result
    except Exception as e:
        logger.warning("[alignment_scorer] LLM fallback failed for %s: %s", company, e)
        return _neutral_result(f"Alignment scoring unavailable: {e}", "skip")


# ------------------------------------------------------------------
# Scoring — source material (pre-generation check)
# ------------------------------------------------------------------

def score_source_material(
    company: str,
    source_material: str,
    client_name: str = "",
) -> dict:
    """Score source material (transcripts) against client fingerprint.

    Same as score_draft_alignment but named distinctly for pre-generation use.
    """
    return score_draft_alignment(company, source_material)


# ------------------------------------------------------------------
# Stelle integration helpers
# ------------------------------------------------------------------

def build_stelle_context(company: str, source_material: str, client_name: str = "") -> str:
    """Generate a Stelle-ready context string from the alignment score.

    Returns empty string when score is strong (no noise needed).
    Returns a warning when score is moderate/drift.
    """
    result = score_source_material(company, source_material, client_name)
    score = result.get("score", 0.5)
    label = result.get("label", "moderate")
    summary = result.get("summary", "")
    drift_topics = result.get("drift_topics", [])
    aligned_topics = result.get("aligned_topics", [])

    if label == "strong":
        logger.info("[alignment_scorer] %s: strong alignment (%.2f)", company, score)
        return ""

    lines = ["\n\n360BREW ALIGNMENT CHECK:"]

    if label == "drift":
        lines.append(
            f"WARNING — source material drifts from {client_name or company}'s established "
            f"topic pillars (score {score:.2f}/1.0). LinkedIn's algorithm suppresses off-pillar "
            f"posts regardless of engagement quality."
        )
    else:
        lines.append(
            f"NOTE — moderate topic alignment (score {score:.2f}/1.0). "
            f"Consider anchoring posts to established pillars where possible."
        )

    if summary:
        lines.append(summary)
    if aligned_topics:
        lines.append(f"Aligned with: {', '.join(aligned_topics)}")
    if drift_topics:
        lines.append(f"Drifting toward: {', '.join(drift_topics)} (less established for this client)")

    return "\n".join(lines)


def build_critic_context(company: str, draft_text: str) -> dict:
    """Return alignment score dict for use by the SELF-REFINE critic."""
    return score_draft_alignment(company, draft_text)


def _neutral_result(reason: str, method: str = "skip") -> dict:
    return {
        "score": 0.5,
        "label": "moderate",
        "method": method,
        "summary": reason,
        "aligned_topics": [],
        "drift_topics": [],
    }
