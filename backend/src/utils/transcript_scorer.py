"""Transcript segment scorer — rank interview segments by predicted post quality.

When a new transcript comes in, this module splits it into segments, extracts
content features from each, and ranks segments by their predicted likelihood
of producing a high-performing post for this specific client.

Design:
1. Split transcript into paragraph-level segments (~200 words).
2. Extract 5 content features per segment via a single Haiku call:
   - specificity_score: names specific tools, documents, numbers, companies
   - narrative_structure: before/after, problem/solution, surprise
   - domain_depth: domain-specific terminology
   - emotional_charge: frustration, surprise, contradiction, passion
   - uniqueness: perspective the audience hasn't heard before
3. Score each segment by dot product with learned per-client feature weights.
4. Weights are learned by extracting the same 5 features from scored posts
   in the client's history and computing Spearman correlation with reward.
5. For clients without enough data, fall back to uniform weights (0.2 each)
   with a cross-client prior flag.

Per-client weights are cached in ``segment_feature_weights.json`` with a
7-day TTL. Per-post extracted features are cached in ``segment_features_cache.json``
to avoid re-extracting features for scored posts on every weight recompute.

Usage:
    from backend.src.utils.transcript_scorer import score_transcript

    ranked = score_transcript("innovocommerce", transcript_text)
    # → [ScoredSegment(text, features, score, rank), ...]
"""

import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_EXTRACTOR_MODEL = "claude-haiku-4-5"
_MIN_OBS_FOR_LEARNED_WEIGHTS = 10
_WEIGHT_CACHE_TTL_DAYS = 14
# Minimum absolute Spearman correlation for the learned signal to be trusted.
# Below this threshold, all 5 features are effectively noise and we fall back
# to uniform weights rather than amplifying a ~0 correlation into a hard 1.0
# weight via normalization.
_MIN_CORRELATION_FOR_LEARNED = 0.15
_MIN_SEGMENT_WORDS = 40      # below this, too thin to score meaningfully
_TARGET_SEGMENT_WORDS = 200  # target segment size for splitting
_MAX_SEGMENT_CHARS = 2000    # hard cap to keep LLM calls bounded

# Uniform cross-client prior — used when the client has no learned weights yet.
# Equal weight on all 5 features; they're all a-priori positively correlated
# with post quality per the bitter-lesson principle of expanding the observation space.
_UNIFORM_WEIGHTS = {
    "specificity_score": 0.2,
    "narrative_structure": 0.2,
    "domain_depth": 0.2,
    "emotional_charge": 0.2,
    "uniqueness": 0.2,
}

_FEATURE_KEYS = tuple(_UNIFORM_WEIGHTS.keys())


@dataclass
class ScoredSegment:
    """A single scored transcript segment."""
    text: str
    features: dict
    score: float
    rank: int = 0
    feature_breakdown: dict = field(default_factory=dict)  # weighted contributions


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

_EXTRACTOR_PROMPT = """\
Score this text on 5 dimensions from 0.0 (absent) to 1.0 (strongly present).

TEXT:
{text}

Dimensions:
- specificity_score: Does the text name specific tools, documents, numbers, dollar amounts, companies, roles, or precise timestamps? 1.0 = multiple concrete specifics. 0.0 = entirely abstract.
- narrative_structure: Is there a before/after, problem/solution, surprise reveal, or cause/effect structure? 1.0 = clear story arc. 0.0 = flat exposition.
- domain_depth: Does it use domain-specific terminology (regulatory, technical, industry jargon used correctly)? 1.0 = deep insider language. 0.0 = generic.
- emotional_charge: Is there frustration, surprise, contradiction, passion, or stakes? 1.0 = strong emotion or tension. 0.0 = neutral.
- uniqueness: Is this a perspective an industry audience would NOT already have heard? 1.0 = genuinely contrarian or rare insight. 0.0 = conventional wisdom.

Return ONLY a JSON object with the 5 keys and float values. No markdown, no explanation.
"""


def _extract_features(text: str) -> Optional[dict]:
    """Extract the 5 content features from a text snippet via Haiku.

    Returns a dict with the 5 feature keys mapping to floats in [0, 1],
    or None on failure.
    """
    if not text or len(text.strip()) < 30:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_EXTRACTOR_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": _EXTRACTOR_PROMPT.format(text=text[:_MAX_SEGMENT_CHARS]),
            }],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        features = {}
        for k in _FEATURE_KEYS:
            v = data.get(k)
            if v is None:
                return None
            features[k] = max(0.0, min(1.0, float(v)))
        return features
    except Exception as e:
        logger.debug("[transcript_scorer] Feature extraction failed: %s", e)
        return None


# ------------------------------------------------------------------
# Per-client weight learning
# ------------------------------------------------------------------

def _get_learned_weights(company: str) -> tuple[dict, str]:
    """Return (weights_dict, source_label) for a client.

    Three-tier cascade:
    1. Client has >= _MIN_OBS_FOR_LEARNED_WEIGHTS scored posts → learn from correlations
    2. Stale or insufficient data → fall back to uniform prior
    3. Returns source label ("learned" | "uniform") for downstream reporting
    """
    cache_path = vortex.memory_dir(company) / "segment_feature_weights.json"

    # Check cache
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            computed_at = cached.get("computed_at", "")
            if computed_at:
                dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
                if age_days < _WEIGHT_CACHE_TTL_DAYS:
                    return cached.get("weights", dict(_UNIFORM_WEIGHTS)), cached.get("source", "uniform")
        except Exception:
            pass

    # Load client's scored observations
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None

    if state is None:
        return dict(_UNIFORM_WEIGHTS), "uniform"

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("reward", {}).get("immediate") is not None
    ]

    if len(scored) < _MIN_OBS_FOR_LEARNED_WEIGHTS:
        return dict(_UNIFORM_WEIGHTS), "uniform"

    # Extract features per post (using a feature cache to avoid re-extraction)
    post_features_cache = _load_post_feature_cache(company)
    pairs = []  # (features_dict, reward)
    newly_extracted = 0

    for obs in scored:
        body = (obs.get("posted_body") or obs.get("post_body") or "").strip()
        if not body:
            continue
        post_hash = obs.get("post_hash", "") or hashlib.sha1(body.encode()).hexdigest()[:16]
        reward = obs.get("reward", {}).get("immediate", 0)

        features = post_features_cache.get(post_hash)
        if features is None:
            features = _extract_features(body)
            if features is not None:
                post_features_cache[post_hash] = features
                newly_extracted += 1
        if features is not None:
            pairs.append((features, reward))

    if newly_extracted:
        _save_post_feature_cache(company, post_features_cache)
        logger.info(
            "[transcript_scorer] Extracted features for %d new posts (%s, total cached: %d)",
            newly_extracted, company, len(post_features_cache),
        )

    if len(pairs) < _MIN_OBS_FOR_LEARNED_WEIGHTS:
        return dict(_UNIFORM_WEIGHTS), "uniform"

    # Compute Spearman correlation for each feature vs reward
    try:
        from backend.src.utils.correlation_analyzer import _spearman_correlation
    except Exception:
        return dict(_UNIFORM_WEIGHTS), "uniform"

    rewards = [p[1] for p in pairs]
    raw_correlations = {}
    for key in _FEATURE_KEYS:
        values = [p[0][key] for p in pairs]
        raw_correlations[key] = _spearman_correlation(values, rewards)

    # Signal strength guard: since we clamp negatives to 0, only the maximum
    # POSITIVE correlation actually contributes to weights. If no feature has
    # a positive correlation above the threshold, normalization would amplify
    # a near-zero signal into a hard 1.0 on a single feature. Fall back to uniform.
    max_positive = max((v for v in raw_correlations.values() if v > 0), default=0.0)
    if max_positive < _MIN_CORRELATION_FOR_LEARNED:
        weights = dict(_UNIFORM_WEIGHTS)
        source = "uniform_insufficient_signal"
        logger.info(
            "[transcript_scorer] %s: max positive correlation = %.3f < %.2f threshold. "
            "Falling back to uniform weights (n=%d, raw=%s).",
            company, max_positive, _MIN_CORRELATION_FOR_LEARNED, len(pairs),
            {k: round(v, 3) for k, v in raw_correlations.items()},
        )
    else:
        # Normalize: clamp negatives to 0, normalize to sum=1.
        # Features with negative correlation get zero weight (don't predict success).
        clamped = {k: max(0.0, v) for k, v in raw_correlations.items()}
        total = sum(clamped.values())
        if total < 1e-6:
            weights = dict(_UNIFORM_WEIGHTS)
            source = "uniform_fallback"
        else:
            weights = {k: round(v / total, 4) for k, v in clamped.items()}
            source = "learned"

    # Persist
    cache_data = {
        "weights": weights,
        "raw_correlations": {k: round(v, 4) for k, v in raw_correlations.items()},
        "observation_count": len(pairs),
        "source": source,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cache_path)

    logger.info(
        "[transcript_scorer] Learned weights for %s (n=%d): %s",
        company, len(pairs), weights,
    )

    return weights, source


def _load_post_feature_cache(company: str) -> dict:
    """Load cached per-post feature extractions."""
    path = vortex.memory_dir(company) / "segment_features_cache.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_post_feature_cache(company: str, cache: dict) -> None:
    path = vortex.memory_dir(company) / "segment_features_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------

def _split_into_segments(text: str) -> list[str]:
    """Split a transcript into paragraph-level segments of ~200 words.

    Respects paragraph boundaries where possible. Splits long paragraphs
    on sentence boundaries. Merges tiny paragraphs into neighbors.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # Huge paragraph — split on sentences
        if para_words > _TARGET_SEGMENT_WORDS * 2:
            # Flush current buffer first
            if current:
                segments.append("\n\n".join(current))
                current = []
                current_words = 0

            # Split into sentence-level chunks
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            buf_words = 0
            for sent in sentences:
                sw = len(sent.split())
                if buf_words + sw > _TARGET_SEGMENT_WORDS and buf:
                    segments.append(" ".join(buf))
                    buf = [sent]
                    buf_words = sw
                else:
                    buf.append(sent)
                    buf_words += sw
            if buf:
                segments.append(" ".join(buf))
            continue

        # Normal-sized paragraph
        if current_words + para_words > _TARGET_SEGMENT_WORDS and current:
            segments.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        segments.append("\n\n".join(current))

    # Filter tiny segments
    return [s for s in segments if len(s.split()) >= _MIN_SEGMENT_WORDS]


# ------------------------------------------------------------------
# Main scoring entry point
# ------------------------------------------------------------------

def score_transcript(company: str, transcript_text: str, top_k: int = 5) -> list[ScoredSegment]:
    """Score and rank transcript segments by predicted post quality.

    Returns the top ``top_k`` segments ranked by score, with feature breakdowns.
    If the transcript has fewer than ``top_k`` scorable segments, returns all.

    This is the public entry point used by the strategy brief generator.
    """
    segments = _split_into_segments(transcript_text)
    if not segments:
        return []

    weights, weight_source = _get_learned_weights(company)

    scored_segments: list[ScoredSegment] = []
    for seg in segments:
        features = _extract_features(seg)
        if features is None:
            continue

        # Weighted sum → score in [0, 1]
        feature_breakdown = {k: round(features[k] * weights.get(k, 0), 4) for k in _FEATURE_KEYS}
        score = sum(feature_breakdown.values())

        scored_segments.append(ScoredSegment(
            text=seg,
            features=features,
            score=round(score, 4),
            feature_breakdown=feature_breakdown,
        ))

    # Sort by score, assign ranks
    scored_segments.sort(key=lambda s: s.score, reverse=True)
    for i, seg in enumerate(scored_segments):
        seg.rank = i + 1

    logger.info(
        "[transcript_scorer] Scored %d segments for %s (weight source: %s, top score: %.3f)",
        len(scored_segments), company, weight_source,
        scored_segments[0].score if scored_segments else 0,
    )

    return scored_segments[:top_k]


def score_transcript_file(company: str, transcript_path, top_k: int = 5) -> list[ScoredSegment]:
    """Convenience wrapper: read a transcript file and score it."""
    from pathlib import Path
    p = Path(transcript_path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return score_transcript(company, text, top_k=top_k)
