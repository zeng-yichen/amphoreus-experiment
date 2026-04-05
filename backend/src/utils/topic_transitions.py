"""Topic & format transition model — first-order Markov chain on content sequences.

Content strategy is sequential. Two posts on the same topic back-to-back
have diminishing returns. Alternating between topics keeps the audience
engaged. This module learns the transition structure from data per client.

Requires observations to be tagged (see observation_tagger.py) with
``topic_tag`` and ``format_tag`` set. With 15+ tagged observations,
computes a first-order Markov transition matrix and reward-conditioned
transitions for both topic and format sequences.

The model is intentionally simple — first-order, no smoothing, no embedding-
space interpolation. At n<50 observations, anything fancier would overfit.

Usage:
    from backend.src.utils.topic_transitions import (
        build_transition_model,
        recommend_next_topic,
        recommend_next_format,
    )

    # During ordinal_sync (after tagging):
    build_transition_model("innovocommerce")

    # When planning the next post:
    topic_recs = recommend_next_topic("innovocommerce")
    format_recs = recommend_next_format("innovocommerce")
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_MIN_OBS_FOR_TRANSITIONS = 15
_LAST_N_CONTEXT = 3  # look at last N posts for recommendation context
_MIN_TRANSITION_COUNT_FOR_CONFIDENCE = 2  # below this, flag as exploration opportunity


def build_transition_model(company: str) -> Optional[dict]:
    """Build first-order transition models for both topic and format sequences.

    Returns the model dict or None if insufficient tagged data.
    Caches to ``memory/{company}/topic_transitions.json`` containing both
    topic and format transition matrices.
    """
    state = _load_state(company)
    if state is None:
        return None

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("topic_tag")
        and o.get("reward", {}).get("immediate") is not None
    ]

    if len(scored) < _MIN_OBS_FOR_TRANSITIONS:
        logger.debug(
            "[topic_transitions] %s has %d tagged obs (need %d)",
            company, len(scored), _MIN_OBS_FOR_TRANSITIONS,
        )
        return None

    # Chronological order
    def _ts(obs):
        return obs.get("posted_at") or obs.get("recorded_at") or ""
    scored.sort(key=_ts)

    topic_matrix, topic_stats = _build_matrix(scored, "topic_tag")
    format_matrix, format_stats = _build_matrix(scored, "format_tag")

    # Recent sequence for recommendation anchoring
    recent_topics = [o.get("topic_tag") for o in scored[-_LAST_N_CONTEXT:]]
    recent_formats = [o.get("format_tag") for o in scored[-_LAST_N_CONTEXT:]]

    model = {
        "company": company,
        "topic": {
            "matrix": topic_matrix,
            "stats": topic_stats,
            "recent": recent_topics,
            "unique_count": len(topic_stats),
        },
        "format": {
            "matrix": format_matrix,
            "stats": format_stats,
            "recent": recent_formats,
            "unique_count": len(format_stats),
        },
        "observation_count": len(scored),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    path = vortex.memory_dir(company) / "topic_transitions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)

    logger.info(
        "[topic_transitions] Built model for %s: %d obs, %d topics, %d formats, "
        "%d topic transitions, %d format transitions",
        company, len(scored), len(topic_stats), len(format_stats),
        sum(len(v) for v in topic_matrix.values()),
        sum(len(v) for v in format_matrix.values()),
    )

    return model


def _build_matrix(scored_chronological: list[dict], tag_field: str) -> tuple[dict, dict]:
    """Build a transition matrix and marginal stats for one tag field.

    Returns ``(matrix, stats)`` where:
    - ``matrix[prev][next] = {count, probability, mean_reward}``
    - ``stats[tag] = {count, mean_reward}`` (marginal)
    """
    transitions: dict = defaultdict(lambda: defaultdict(lambda: {"count": 0, "rewards": []}))

    for i in range(1, len(scored_chronological)):
        prev = scored_chronological[i - 1].get(tag_field)
        nxt = scored_chronological[i].get(tag_field)
        if not prev or not nxt:
            continue
        reward = scored_chronological[i].get("reward", {}).get("immediate", 0)
        transitions[prev][nxt]["count"] += 1
        transitions[prev][nxt]["rewards"].append(reward)

    matrix: dict = {}
    for prev, nexts in transitions.items():
        total = sum(v["count"] for v in nexts.values())
        matrix[prev] = {}
        for nxt, data in nexts.items():
            rewards = data["rewards"]
            matrix[prev][nxt] = {
                "count": data["count"],
                "probability": round(data["count"] / total, 4) if total else 0.0,
                "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            }

    # Marginal stats — used for cold-start and unseen-transition fallback.
    stats: dict = {}
    buckets: dict = defaultdict(list)
    for obs in scored_chronological:
        tag = obs.get(tag_field)
        if tag:
            buckets[tag].append(obs.get("reward", {}).get("immediate", 0))
    for tag, rs in buckets.items():
        stats[tag] = {
            "count": len(rs),
            "mean_reward": round(sum(rs) / len(rs), 4),
        }

    return matrix, stats


def recommend_next_topic(company: str, top_k: int = 5) -> Optional[list[dict]]:
    """Recommend the next post's topic given recent history.

    Returns a ranked list, or None if no model exists.
    Each item: ``{"topic", "expected_reward", "confidence", "source", "rationale"}``

    Ranking prioritizes:
    1. Direct transitions with ≥2 observations (reliable signal)
    2. Marginal topic performance for unseen transitions (exploration)
    3. Avoids recommending a direct repeat of the last topic
    """
    return _recommend_next("topic", company, top_k=top_k)


def recommend_next_format(company: str, top_k: int = 5) -> Optional[list[dict]]:
    """Recommend the next post's format given recent history.

    Same structure as recommend_next_topic, operating on format_tag.
    """
    return _recommend_next("format", company, top_k=top_k)


def _recommend_next(dimension: str, company: str, top_k: int = 5) -> Optional[list[dict]]:
    path = vortex.memory_dir(company) / "topic_transitions.json"
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    sub = model.get(dimension)
    if not sub:
        return None

    matrix = sub.get("matrix", {})
    stats = sub.get("stats", {})
    recent = sub.get("recent", [])

    if not recent:
        return None

    last_tag = recent[-1]

    candidates: dict = {}

    # 1. Direct transitions observed from the last tag
    if last_tag in matrix:
        for nxt, data in matrix[last_tag].items():
            if data["count"] >= _MIN_TRANSITION_COUNT_FOR_CONFIDENCE:
                source = "direct_transition"
                rationale = (
                    f"After '{last_tag}', this {dimension} has historically "
                    f"averaged reward {data['mean_reward']:+.3f} "
                    f"({data['count']} observations)"
                )
            else:
                source = "sparse_transition"
                rationale = (
                    f"Only {data['count']} observation of '{last_tag}' → '{nxt}'. "
                    f"Exploration opportunity."
                )
            candidates[nxt] = {
                dimension: nxt,
                "expected_reward": data["mean_reward"],
                "confidence": data["count"],
                "source": source,
                "rationale": rationale,
            }

    # 2. Unexplored targets — marginal performance only
    for tag, tag_stats in stats.items():
        if tag == last_tag:
            continue  # don't recommend a direct repeat
        if tag in candidates:
            continue
        candidates[tag] = {
            dimension: tag,
            "expected_reward": tag_stats["mean_reward"],
            "confidence": 0,
            "source": "unexplored_transition",
            "rationale": (
                f"Never followed '{last_tag}' in the history. "
                f"Marginal {dimension} performance: {tag_stats['mean_reward']:+.3f} "
                f"({tag_stats['count']} posts)."
            ),
        }

    # Rank: (has reliable signal, expected reward, confidence)
    def _sort_key(c: dict) -> tuple:
        has_signal = 1 if c["confidence"] >= _MIN_TRANSITION_COUNT_FOR_CONFIDENCE else 0
        return (has_signal, c["expected_reward"], c["confidence"])

    ranked = sorted(candidates.values(), key=_sort_key, reverse=True)
    return ranked[:top_k]


def _load_state(company: str) -> Optional[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
        if state is not None:
            return state
    except Exception:
        pass

    path = vortex.memory_dir(company) / "ruan_mei_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
