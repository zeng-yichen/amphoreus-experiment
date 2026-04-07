"""LLM-native ICP scorer for post engagers (bitter-lesson aligned).

Semantically evaluates engager headlines against a free-text ICP
description and anti-description using Claude Haiku.

Scoring is continuous 0.0-1.0 per engager (preserves full gradient):
  1.0 = perfect ICP match
  0.5 = adjacent/plausible
  0.0 = anti-ICP or completely off-target

The scorer returns both a scalar mean and a segment breakdown dict
(derived from the continuous scores for backward compatibility).

Per-client ICP definition lives at ``memory/{company}/icp_definition.json``::

    {
        "description": "Free-text description of ideal engager profiles...",
        "anti_description": "Free-text description of off-target profiles..."
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_icp(company: str) -> Optional[dict]:
    from backend.src.db.vortex import icp_definition_path
    p = icp_definition_path(company)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[icp_scorer] Failed to load ICP for %s: %s", company, e)
        return None


_DEFAULT_ICP_SEGMENT_THRESHOLDS = {"exact": 0.75, "adjacent": 0.40, "neutral": 0.20}
_MIN_OBS_FOR_ADAPTIVE_ICP_SEGMENTS = 15


def _get_segment_thresholds(company: str) -> dict:
    """Return ICP segment boundaries — percentile-based when enough data exists.

    With ≥15 observations that have icp_match_rate, uses the 25th/50th/75th
    percentiles of historical per-post match rates as boundaries.
    Falls back to fixed 0.75/0.40/0.20 when insufficient data.
    """
    # Check cache
    try:
        from backend.src.db import vortex as P
        cache_path = P.memory_dir(company) / "icp_thresholds.json"
        if cache_path.exists():
            import json as _json
            cached = _json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("observation_count", 0) >= _MIN_OBS_FOR_ADAPTIVE_ICP_SEGMENTS:
                return cached
    except Exception:
        pass

    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        icp_scores = [
            o.get("icp_match_rate", o.get("reward", {}).get("icp_reward"))
            for o in rm._state.get("observations", [])
            if o.get("status") == "scored"
            and (o.get("icp_match_rate") is not None or o.get("reward", {}).get("icp_reward") is not None)
        ]
        icp_scores = [s for s in icp_scores if s is not None and 0.0 <= s <= 1.0]
    except Exception:
        icp_scores = []

    if len(icp_scores) < _MIN_OBS_FOR_ADAPTIVE_ICP_SEGMENTS:
        return dict(_DEFAULT_ICP_SEGMENT_THRESHOLDS)

    icp_scores.sort()
    n = len(icp_scores)
    p25 = icp_scores[int(n * 0.25)]
    p50 = icp_scores[int(n * 0.50)]
    p75 = icp_scores[int(n * 0.75)]

    result = {
        "exact": round(p75, 3),
        "adjacent": round(p50, 3),
        "neutral": round(p25, 3),
        "observation_count": n,
    }

    # Cache
    try:
        import json as _json
        from backend.src.db import vortex as P
        cache_path = P.memory_dir(company) / "icp_thresholds.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(result, indent=2), encoding="utf-8")
        tmp.rename(cache_path)
        logger.info("[icp_scorer] Learned segment thresholds for %s: exact=%.2f adj=%.2f neutral=%.2f (n=%d)",
                     company, p75, p50, p25, n)
    except Exception:
        pass

    return result


def score_engagers_segmented(company: str, headlines: list[str]) -> dict:
    """Score engagers with continuous 0-1 relevance scores and return breakdown.

    Segment boundaries are learned from the client's historical ICP score
    distribution (percentile-based) when ≥15 observations exist. Falls back
    to fixed 0.75/0.40/0.20 boundaries for new clients.

    Returns:
        {
            "score": float,          # mean relevance in [0, 1]
            "scores": list[float],   # per-engager continuous scores
            "segments": {            # derived from continuous scores
                "exact_icp": int,
                "adjacent": int,
                "neutral": int,
                "anti_icp": int,
                "total": int,
            },
            "icp_match_rate": float, # continuous mean, in [0, 1]
        }
    """
    empty = {
        "score": 0.0,
        "scores": [],
        "segments": {"exact_icp": 0, "adjacent": 0, "neutral": 0, "anti_icp": 0, "total": 0},
        "icp_match_rate": 0.0,
    }

    if not headlines:
        return empty

    icp = _load_icp(company)
    if not icp:
        return empty

    description = icp.get("description", "")
    anti_description = icp.get("anti_description", "")

    if not description and not anti_description:
        return empty

    # Build engager block — supports both string headlines and enriched dicts
    engager_lines = []
    for i, h in enumerate(headlines[:50]):
        if isinstance(h, dict):
            parts = []
            if h.get("headline"):
                parts.append(f"headline: \"{h['headline']}\"")
            if h.get("current_company"):
                parts.append(f"company: {h['current_company']}")
            if h.get("title"):
                parts.append(f"title: {h['title']}")
            if h.get("name"):
                parts.append(f"name: {h['name']}")
            engager_lines.append(f"{i+1}. {' | '.join(parts)}")
        else:
            engager_lines.append(f"{i+1}. {h}")
    hl_block = "\n".join(engager_lines)

    prompt = (
        "You are scoring LinkedIn post engagers against an Ideal Customer Profile (ICP).\n\n"
        f"ICP DESCRIPTION (people we WANT engaging):\n{description}\n\n"
    )
    if anti_description:
        prompt += f"ANTI-ICP (people we do NOT want):\n{anti_description}\n\n"
    prompt += (
        f"ENGAGER PROFILES:\n{hl_block}\n\n"
        "For each numbered engager, output a relevance score from 0.0 to 1.0.\n"
        "Use ALL available fields (headline, company, title) to make your assessment.\n"
        "When only a vague headline is available, score conservatively (0.3-0.5).\n\n"
        "Score range:\n"
        "  1.0 = perfect ICP match (title, industry, seniority all fit)\n"
        "  0.7 = strong match (most ICP criteria fit)\n"
        "  0.5 = adjacent (related role/industry, plausible buyer)\n"
        "  0.3 = weak/neutral (can't determine, or tangentially related)\n"
        "  0.0 = anti-ICP (job seeker, recruiter, bot, content marketer)\n\n"
        "Output ONLY a comma-separated list of scores.\n"
        "Example for 4 headlines: 0.9,0.5,0.3,0.0\n"
        "Output ONLY the scores, nothing else."
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        scores = _parse_continuous_scores(raw)
        if not scores:
            return empty
        return _compute_result(scores, company)
    except Exception as e:
        logger.warning("[icp_scorer] LLM scoring failed for %s: %s", company, e)
        return empty


def score_engagers(company: str, headlines: list[str]) -> float:
    """Backward-compatible: return a single scalar ICP score in [-1, 1].

    Maps the continuous [0, 1] mean to [-1, 1] for RuanMei compatibility:
    0.0 → -1.0, 0.5 → 0.0, 1.0 → 1.0
    """
    result = score_engagers_segmented(company, headlines)
    raw = result["score"]
    # Linear map: 0→-1, 0.5→0, 1.0→1.0
    return round(raw * 2 - 1, 4)


def _parse_continuous_scores(raw: str) -> list[float]:
    """Parse comma-separated 0.0-1.0 scores from LLM response."""
    parts = raw.replace(" ", "").split(",")
    scores: list[float] = []
    for p in parts:
        p = p.strip()
        try:
            v = float(p)
            scores.append(max(0.0, min(1.0, v)))
        except ValueError:
            # Handle legacy E/A/N/X labels for backward compat
            if p.upper() == "E" or p in ("+1", "1"):
                scores.append(1.0)
            elif p.upper() == "A":
                scores.append(0.5)
            elif p.upper() == "N" or p == "0":
                scores.append(0.3)
            elif p.upper() == "X" or p == "-1":
                scores.append(0.0)
    return scores


def _compute_result(scores: list[float], company: str = "") -> dict:
    """Compute aggregate result from per-engager continuous scores.

    Segment boundaries are learned from client history when available.
    """
    total = len(scores)
    if total == 0:
        return {
            "score": 0.0,
            "scores": [],
            "segments": {"exact_icp": 0, "adjacent": 0, "neutral": 0, "anti_icp": 0, "total": 0},
            "icp_match_rate": 0.0,
        }

    mean_score = sum(scores) / total

    # Segment boundaries: learned from client history or defaults
    thresholds = _get_segment_thresholds(company) if company else dict(_DEFAULT_ICP_SEGMENT_THRESHOLDS)
    exact_t = thresholds.get("exact", 0.75)
    adjacent_t = thresholds.get("adjacent", 0.40)
    neutral_t = thresholds.get("neutral", 0.20)

    segments = {"exact_icp": 0, "adjacent": 0, "neutral": 0, "anti_icp": 0}
    for s in scores:
        if s >= exact_t:
            segments["exact_icp"] += 1
        elif s >= adjacent_t:
            segments["adjacent"] += 1
        elif s >= neutral_t:
            segments["neutral"] += 1
        else:
            segments["anti_icp"] += 1

    # icp_match_rate: raw continuous mean (not segment-based).
    # The mean score IS the match rate — no categorical boundaries involved.
    # Segments are derived for display only.
    return {
        "score": round(mean_score, 4),
        "scores": [round(s, 3) for s in scores],
        "segments": {**segments, "total": total},
        "icp_match_rate": round(mean_score, 4),
    }


def icp_match_rate(company: str, headlines: list[str]) -> dict:
    """Diagnostic breakdown of ICP scoring for a post's engagers."""
    result = score_engagers_segmented(company, headlines)
    return {
        "total": result["segments"]["total"],
        "score": result["score"],
        "icp_match_rate": result["icp_match_rate"],
        "segments": result["segments"],
        "method": "llm-continuous",
    }
