"""Transcript segment scorer — embedding-based, learned from client engagement.

When a new transcript comes in, this module splits it into segments, describes
each via a single Haiku call ("what is this segment about, what moment does it
capture"), embeds the descriptions, and projects the embeddings to predicted
post quality via a linear model learned from the client's own engagement data.

## Design (bitter lesson compliance)

The scorer does NOT use a fixed feature taxonomy. There is no "specificity",
"narrative_structure", "domain_depth", "emotional_charge", or "uniqueness"
score. Those were hand-designed quality axes — the same class of violation
as Cyrene's original 7 fixed dimensions. Hand-designed features produce
uniform weights when none of them happen to correlate with engagement,
which means the scorer learns nothing.

Instead:

1. **Describe, don't score.** The Haiku call produces a 2-3 sentence free-text
   description of what the segment is about. No forced taxonomy. Same
   observe-don't-prescribe pattern as LOLA's strategy descriptors and the
   observation tagger's free-text tags.

2. **Embed the descriptions.** Uses ``lola._embed_texts`` (sentence-transformers,
   384-dim, local, cheap). Same embedding infrastructure as Cyrene's quality
   projection and LOLA's continuous reward field.

3. **Learn from data.** For clients with >= 15 scored+tagged observations,
   fits a ridge regression from (description_embedding, engagement_reward)
   pairs, using the client's own post bodies as training data. Past posts
   are described with the exact same Haiku prompt as new segments, so training
   and inference live in the same embedding space.

4. **Cross-client transfer.** When a client has no learned model yet,
   ``get_cold_start_seeds`` from cross_client.py surfaces the most-similar
   client's segment model. The new client inherits a warm-start projection
   until its own data accumulates.

5. **No hand-designed fallback.** If neither a client model nor a cross-client
   seed exists, ``score_transcript`` returns segments in document order with
   ``predicted_reward=None`` and a note. Scoring requires data. That's honest.

## Storage

- ``memory/{company}/segment_model.json`` — learned projection (weights, bias,
  LOO R², metadata)
- ``memory/{company}/segment_descriptions_cache.json`` — per-post description
  cache keyed by post_hash, avoids re-Haiku-ing on every recompute

## Usage

    from backend.src.utils.transcript_scorer import (
        build_segment_model,
        score_transcript,
    )

    # Called from ordinal_sync after the tagger runs:
    build_segment_model("innovocommerce")

    # Ad-hoc (e.g., from strategy brief):
    ranked = score_transcript("innovocommerce", transcript_text)
"""

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from backend.src.db import vortex

logger = logging.getLogger(__name__)

# Sonnet for segment descriptions. The description quality directly determines
# the embedding, which directly determines the learned projection. Haiku's
# descriptions were generic; Sonnet produces richer, more distinctive phrasing
# that embeds into a more separable space — exactly what the ridge regression
# needs to find structure at small n.
_DESCRIBER_MODEL = "claude-sonnet-4-6"
_EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI; 1536 dims; used by alignment_scorer
_EMBEDDING_DIM = 1536
_MIN_OBS_FOR_SEGMENT_MODEL = 15
_MODEL_CACHE_TTL_DAYS = 7
_MIN_SEGMENT_WORDS = 40      # below this, too thin to describe meaningfully
_TARGET_SEGMENT_WORDS = 200
_MAX_SEGMENT_CHARS = 2000    # hard cap per Haiku call
# Ridge penalty for the segment projection. Embeddings are 1536-dim and the
# training set is typically 15-50 observations — heavily underdetermined,
# so the regularization has to do real work. alpha scales roughly with
# trace(X^T X)/effective_rank ≈ (n*d)/n = d, so alpha on the order of the
# embedding dimension is a reasonable starting point. We use 50.0 to stay
# conservative without fully washing out the signal.
_RIDGE_ALPHA = 50.0


@dataclass
class ScoredSegment:
    """A single transcript segment with description, embedding, and predicted reward.

    ``predicted_reward`` is None when no learned model and no cross-client seed
    exist for this client. In that case, segments are returned in document
    order and the consumer should show the descriptions without a numeric score.
    """
    text: str
    description: str
    predicted_reward: Optional[float] = None
    rank: int = 0
    # The embedding is included for downstream components that want to compare
    # segments against each other (e.g., cluster by topic).
    embedding: Optional[list] = None


# ------------------------------------------------------------------
# Haiku description + embedding
# ------------------------------------------------------------------

_DESCRIBER_PROMPT = """\
Describe this text in 2-3 sentences. Cover:
- What it is about (the subject matter, the topic, the situation)
- What kind of moment or material it captures (a specific incident, a decision, \
an observation, a comparison, a failure, a realization, an opinion)
- What makes it distinctive (specific details, tension, contradiction, stakes, \
unusual perspective)

Write in plain prose. No bullets, no labels, no JSON. Just 2-3 descriptive sentences.

TEXT:
{text}
"""


def _describe_text(text: str) -> Optional[str]:
    """Produce a 2-3 sentence free-text description via Haiku.

    Returns the description string, or None on failure / insufficient content.
    Used for both past post bodies (training) and new transcript segments
    (inference) — same prompt, same embedding space.
    """
    if not text or len(text.strip()) < 30:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_DESCRIBER_MODEL,
            max_tokens=250,
            messages=[{
                "role": "user",
                "content": _DESCRIBER_PROMPT.format(text=text[:_MAX_SEGMENT_CHARS]),
            }],
        )
        desc = resp.content[0].text.strip()
        if not desc or len(desc) < 20:
            return None
        return desc
    except Exception as e:
        logger.debug("[transcript_scorer] Description failed: %s", e)
        return None


def _embed_descriptions(descriptions: list[str]) -> list[list[float]]:
    """Embed a batch of descriptions via OpenAI's text-embedding-3-small.

    This is the same embedding model used by ``alignment_scorer.py`` for
    client identity fingerprinting. Using OpenAI rather than
    sentence-transformers gives us:
      - deterministic availability (no optional dependency)
      - stable dimensions (1536) across all environments
      - a single canonical embedder so train-time and inference-time
        vectors are guaranteed to live in the same space

    The saved segment model records ``embedding_model`` so future schema
    changes (if we ever migrate embedders) can be detected by dim mismatch.

    Returns [] on failure. Caller treats empty as "embedding unavailable".
    """
    if not descriptions:
        return []
    try:
        from openai import OpenAI
        client = OpenAI()
        # OpenAI accepts up to 2048 inputs per call; our batches are small
        # (30-50 posts or ~25 segments) so a single request is fine.
        resp = client.embeddings.create(
            input=[d[:8000] for d in descriptions],  # token-safe truncation
            model=_EMBEDDING_MODEL,
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.warning("[transcript_scorer] OpenAI embedding failed: %s", e)
        return []


# ------------------------------------------------------------------
# Per-post description cache (training data preparation)
# ------------------------------------------------------------------

def _load_descriptions_cache(company: str) -> dict:
    """Load cached per-post descriptions keyed by post_hash."""
    path = vortex.memory_dir(company) / "segment_descriptions_cache.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_descriptions_cache(company: str, cache: dict) -> None:
    path = vortex.memory_dir(company) / "segment_descriptions_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


# ------------------------------------------------------------------
# Ridge projection — local helper, numpy-backed
# ------------------------------------------------------------------

def _ridge_fit(X: list[list[float]], y: list[float], alpha: float) -> tuple[list[float], float]:
    """Fit ``w, b`` minimizing ``||X w + b - y||^2 + alpha * ||w||^2``.

    Returns ``(weights_list, bias)`` where weights is a 384-length list of floats.
    Uses numpy for the linear solve (same pattern as Cyrene's LinearProjection).
    """
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    n, d = X_arr.shape
    # Centered design — predict mean + residual projection (standard bias trick)
    x_mean = X_arr.mean(axis=0)
    y_mean = float(y_arr.mean())
    X_c = X_arr - x_mean
    y_c = y_arr - y_mean

    XtX = X_c.T @ X_c + alpha * np.eye(d, dtype=np.float32)
    Xty = X_c.T @ y_c

    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return [], 0.0

    # b such that predict(x) = w . (x - x_mean) + y_mean = w . x + (y_mean - w . x_mean)
    b = y_mean - float(w @ x_mean)
    return w.tolist(), b


def _predict(embedding: list[float], weights: list[float], bias: float) -> float:
    """Apply the learned projection to a single embedding."""
    if not embedding or len(embedding) != len(weights):
        return bias
    return float(sum(e * w for e, w in zip(embedding, weights)) + bias)


# ------------------------------------------------------------------
# Segment model persistence
# ------------------------------------------------------------------

def _model_path(company: str):
    return vortex.memory_dir(company) / "segment_model.json"


def _load_segment_model(company: str) -> Optional[dict]:
    """Load the learned segment model for a client.

    Returns None if the model doesn't exist OR is stale beyond the TTL.
    The caller should then try cross-client cold-start seeds, or return
    unscored segments as the honest answer.
    """
    path = _model_path(company)
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # TTL check — stale models should be rebuilt on the next sync cycle,
    # but in the meantime they're still usable for scoring (a 14-day-old
    # projection is better than no scoring at all, and it gives a stable
    # baseline for the impact tracker).
    computed_at = model.get("computed_at", "")
    if computed_at:
        try:
            dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            model["_age_days"] = round(age_days, 2)
        except Exception:
            pass
    return model


def _save_segment_model(company: str, model: dict) -> None:
    path = _model_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


# ------------------------------------------------------------------
# Training: build_segment_model
# ------------------------------------------------------------------

def build_segment_model(company: str, force: bool = False) -> Optional[dict]:
    """Train a segment quality projection from the client's tagged observations.

    For each scored observation with a ``source_segment_type`` tag, describes
    the post body via the same Haiku prompt used for transcript segments at
    inference time, embeds the description, and fits a ridge regression
    against the engagement reward. Caches descriptions per post_hash so only
    new observations incur LLM cost on each recompute.

    Requires >= 15 such observations. Below that, returns None (the scorer
    then falls through to cross-client seeds or unscored output).

    Args:
        company: Client slug.
        force: If True, bypass the TTL check and rebuild even if the cached
            model is fresh.

    Returns the model dict on success, None otherwise.
    """
    # Fast exit: if a fresh cached model exists and force is False, return it.
    if not force:
        existing = _load_segment_model(company)
        if existing and existing.get("_age_days", 999) < _MODEL_CACHE_TTL_DAYS:
            cached_count = existing.get("observation_count", 0)
            # Recompute if new observations have come in since last build.
            current_count = _count_eligible_observations(company)
            if cached_count >= current_count:
                return existing

    # Load eligible observations
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None
    if state is None:
        return None

    eligible = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("source_segment_type")
        and o.get("reward", {}).get("immediate") is not None
        and (o.get("posted_body") or o.get("post_body"))
    ]

    if len(eligible) < _MIN_OBS_FOR_SEGMENT_MODEL:
        logger.debug(
            "[transcript_scorer] %s has %d eligible obs (need %d) — skipping segment model",
            company, len(eligible), _MIN_OBS_FOR_SEGMENT_MODEL,
        )
        return None

    # Step 1: gather or compute descriptions for each observation, cached.
    descriptions_cache = _load_descriptions_cache(company)
    newly_described = 0
    pairs: list[tuple[str, float]] = []  # (description, reward)

    for obs in eligible:
        body = (obs.get("posted_body") or obs.get("post_body", "")).strip()
        if not body:
            continue
        post_hash = obs.get("post_hash") or hashlib.sha1(body.encode()).hexdigest()[:16]
        reward = float(obs.get("reward", {}).get("immediate", 0))

        desc = descriptions_cache.get(post_hash)
        if desc is None:
            desc = _describe_text(body)
            if desc:
                descriptions_cache[post_hash] = desc
                newly_described += 1
        if desc:
            pairs.append((desc, reward))

    if newly_described:
        _save_descriptions_cache(company, descriptions_cache)
        logger.info(
            "[transcript_scorer] Described %d new posts for %s (cache total: %d)",
            newly_described, company, len(descriptions_cache),
        )

    if len(pairs) < _MIN_OBS_FOR_SEGMENT_MODEL:
        return None

    # Step 2: embed the descriptions
    descriptions = [p[0] for p in pairs]
    rewards = [p[1] for p in pairs]
    embeddings = _embed_descriptions(descriptions)
    if not embeddings or len(embeddings) != len(pairs):
        logger.warning("[transcript_scorer] Embedding failed for %s", company)
        return None

    # Step 3: ridge fit
    weights, bias = _ridge_fit(embeddings, rewards, alpha=_RIDGE_ALPHA)
    if not weights:
        logger.warning("[transcript_scorer] Ridge solve failed for %s", company)
        return None

    # Step 4: leave-one-out R² for honesty. At d=384, n≈30 the model will fit
    # training data well but LOO is the only signal that generalization is
    # real. We report it so the consumer can decide how much to trust the scores.
    loo_r2 = _leave_one_out_r_squared(embeddings, rewards, alpha=_RIDGE_ALPHA)

    # Also compute the in-sample mean + std of predicted rewards for calibration.
    train_preds = [_predict(e, weights, bias) for e in embeddings]
    pred_mean = sum(train_preds) / len(train_preds)
    pred_std = math.sqrt(
        sum((p - pred_mean) ** 2 for p in train_preds) / max(len(train_preds) - 1, 1)
    )

    model = {
        "company": company,
        "weights": weights,
        "bias": round(bias, 6),
        "embedding_dim": len(weights),
        "embedding_model": _EMBEDDING_MODEL,
        "ridge_alpha": _RIDGE_ALPHA,
        "observation_count": len(pairs),
        "loo_r_squared": round(loo_r2, 4),
        "train_pred_mean": round(pred_mean, 4),
        "train_pred_std": round(pred_std, 4),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_segment_model(company, model)

    logger.info(
        "[transcript_scorer] Built segment model for %s: n=%d, LOO R²=%.4f, "
        "pred_mean=%.3f, pred_std=%.3f",
        company, len(pairs), loo_r2, pred_mean, pred_std,
    )
    return model


def _count_eligible_observations(company: str) -> int:
    """Count observations eligible for the segment model (for TTL decisions)."""
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return 0
    if state is None:
        return 0
    return sum(
        1 for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("source_segment_type")
        and o.get("reward", {}).get("immediate") is not None
        and (o.get("posted_body") or o.get("post_body"))
    )


def _leave_one_out_r_squared(
    embeddings: list[list[float]],
    rewards: list[float],
    alpha: float,
) -> float:
    """Compute LOO R² for the ridge regression.

    For each (E_i, r_i), refit the model on the other n-1 points and predict r_i.
    Returns 1 - SS_res / SS_tot. Can be negative when the model generalizes worse
    than predicting the mean — which is a useful diagnostic, not a bug.
    """
    n = len(rewards)
    if n < 3:
        return 0.0
    y_mean = sum(rewards) / n
    ss_tot = sum((r - y_mean) ** 2 for r in rewards)
    if ss_tot < 1e-12:
        return 0.0

    ss_res = 0.0
    for i in range(n):
        train_X = [embeddings[j] for j in range(n) if j != i]
        train_y = [rewards[j] for j in range(n) if j != i]
        w, b = _ridge_fit(train_X, train_y, alpha=alpha)
        if not w:
            continue
        pred = _predict(embeddings[i], w, b)
        ss_res += (rewards[i] - pred) ** 2

    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------------
# Cross-client cold-start seed lookup
# ------------------------------------------------------------------

def _get_cold_start_segment_model(company: str) -> Optional[dict]:
    """Fetch a segment model from the most similar client.

    Uses ``cross_client.get_segment_model_seed`` which is dedicated to model
    transfer and does NOT require the target client to be "new" (< 5 obs).
    This matters because a client can have 10-14 scored observations (enough
    to not be considered new) but still lack its own segment model (threshold
    is 15) — transfer learning should still apply in that regime.

    As a secondary path, also checks the general ``get_cold_start_seeds``
    (which does have the new-client gate) so any LOLA/Cyrene/segment bundle
    a truly-new client gets is consistent.
    """
    try:
        from backend.src.utils.cross_client import get_segment_model_seed
        seed = get_segment_model_seed(company)
        if seed:
            return seed
    except Exception:
        pass

    # Fall through to the bundled cold-start seed (for genuinely new clients).
    try:
        from backend.src.utils.cross_client import get_cold_start_seeds
        seeds = get_cold_start_seeds(company)
    except Exception:
        return None
    if not seeds:
        return None
    return seeds.get("segment_model")


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------

def _split_into_segments(text: str) -> list[str]:
    """Split a transcript into paragraph-level segments of ~200 words.

    Respects paragraph boundaries where possible. Splits long paragraphs
    on sentence boundaries. Filters tiny segments.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > _TARGET_SEGMENT_WORDS * 2:
            if current:
                segments.append("\n\n".join(current))
                current = []
                current_words = 0

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

        if current_words + para_words > _TARGET_SEGMENT_WORDS and current:
            segments.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        segments.append("\n\n".join(current))

    return [s for s in segments if len(s.split()) >= _MIN_SEGMENT_WORDS]


# ------------------------------------------------------------------
# Main scoring entry point
# ------------------------------------------------------------------

def score_transcript(
    company: str,
    transcript_text: str,
    top_k: int = 5,
) -> list[ScoredSegment]:
    """Score and rank transcript segments for a client.

    Pipeline:
    1. Split transcript into ~200-word segments.
    2. Describe each via Haiku (2-3 sentences, free-text).
    3. Embed descriptions via sentence-transformers.
    4. Score via the client's learned segment model, or a cross-client seed,
       or return unscored if neither exists.
    5. Return top ``top_k`` segments ranked by predicted reward (or in
       document order when unscored).

    When scoring is unavailable, each returned ScoredSegment has
    ``predicted_reward=None`` — callers should display the descriptions
    without a numeric score in that case.
    """
    segments = _split_into_segments(transcript_text)
    if not segments:
        return []

    # Describe + embed
    descriptions: list[str] = []
    valid_segments: list[str] = []
    for seg in segments:
        desc = _describe_text(seg)
        if desc:
            descriptions.append(desc)
            valid_segments.append(seg)

    if not descriptions:
        return []

    embeddings = _embed_descriptions(descriptions)
    if not embeddings or len(embeddings) != len(descriptions):
        # Embedding failed — return descriptions without scores
        unscored = [
            ScoredSegment(
                text=s,
                description=d,
                predicted_reward=None,
                embedding=None,
            )
            for s, d in zip(valid_segments, descriptions)
        ]
        for i, seg in enumerate(unscored):
            seg.rank = i + 1
        return unscored[:top_k]

    # Find a model: own → cross-client seed → none
    model = _load_segment_model(company)
    model_source = "learned"
    if model is None:
        model = _get_cold_start_segment_model(company)
        model_source = "cross_client_seed" if model else "none"

    results: list[ScoredSegment] = []
    if model and model.get("weights"):
        weights = model["weights"]
        bias = float(model.get("bias", 0))
        if len(weights) != len(embeddings[0]):
            logger.warning(
                "[transcript_scorer] Model dim mismatch for %s: weights=%d, embedding=%d",
                company, len(weights), len(embeddings[0]),
            )
            model = None

    if model and model.get("weights"):
        weights = model["weights"]
        bias = float(model.get("bias", 0))
        for seg_text, desc, emb in zip(valid_segments, descriptions, embeddings):
            pred = _predict(emb, weights, bias)
            results.append(ScoredSegment(
                text=seg_text,
                description=desc,
                predicted_reward=round(pred, 4),
                embedding=emb,
            ))
        results.sort(key=lambda s: s.predicted_reward, reverse=True)
        for i, s in enumerate(results):
            s.rank = i + 1
        logger.info(
            "[transcript_scorer] Scored %d segments for %s via %s "
            "(LOO R²=%s, top predicted=%.3f)",
            len(results), company, model_source,
            model.get("loo_r_squared", "n/a"),
            results[0].predicted_reward if results else 0,
        )
    else:
        # No model available — return descriptions in document order, unscored.
        for seg_text, desc, emb in zip(valid_segments, descriptions, embeddings):
            results.append(ScoredSegment(
                text=seg_text,
                description=desc,
                predicted_reward=None,
                embedding=emb,
            ))
        for i, s in enumerate(results):
            s.rank = i + 1
        logger.info(
            "[transcript_scorer] No segment model for %s (needs >= %d tagged "
            "observations or a cross-client seed) — returning %d unscored segments",
            company, _MIN_OBS_FOR_SEGMENT_MODEL, len(results),
        )

    return results[:top_k]


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
