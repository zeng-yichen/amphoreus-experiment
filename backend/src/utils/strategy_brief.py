"""Strategy brief generator — data-driven weekly content strategy per client.

Culminates workstream A (observation tagging → transition model → transcript
scorer → this). Produces a one-page markdown brief for the human operator
(Yichen) that makes content strategy decisions explicit and data-backed.

This is NOT a directive for Stelle. It's a planning tool for the human.
Stelle's system prompt and generation logic are untouched.

Sections produced:
1. Performance summary — last 2 weeks trend vs client baseline, notable wins/drops
2. Recommended topics — top 3-5 from topic transitions + LOLA arms
3. Recommended format sequence — based on format transitions
4. Top transcript segments — from transcript_scorer over unscored transcripts
5. Recommended ABM targets — companies in abm_profiles not recently mentioned
6. Causal drivers (when causal_filter is built — wired in by B2)
7. Cross-client insights — unexploited universal patterns relevant to this client

Usage:
    from backend.src.utils.strategy_brief import generate_strategy_brief

    brief = generate_strategy_brief("innovocommerce")
    # Written to memory/{company}/strategy_brief.md
"""

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_BRIEF_CACHE_TTL_DAYS = 7  # regenerate weekly by default
_RECENT_WINDOW_DAYS = 14
_BASELINE_WINDOW_DAYS = 90
_TOP_SEGMENTS_PER_TRANSCRIPT = 3
_MAX_TRANSCRIPTS_SCORED = 2  # cap LLM cost per brief

# Max chars for the compact Stelle-injection version of the brief.
# Full brief is 5-9k chars; compact is ~1000-1500 to fit user_prompt budget
# without crowding out transcripts and other context.
_STELLE_COMPACT_MAX_CHARS = 1800


def generate_strategy_brief(company: str, force: bool = False) -> Optional[str]:
    """Generate a markdown strategy brief for a client.

    Returns the markdown string, or None if insufficient data to say anything
    useful. Caches to ``memory/{company}/strategy_brief.md``.

    Args:
        company: Client slug.
        force: If True, regenerate even if the cached brief is recent.
    """
    brief_path = vortex.memory_dir(company) / "strategy_brief.md"

    # Cache check
    if not force and brief_path.exists():
        try:
            age_seconds = (
                datetime.now(timezone.utc).timestamp()
                - brief_path.stat().st_mtime
            )
            if age_seconds < _BRIEF_CACHE_TTL_DAYS * 86400:
                return brief_path.read_text(encoding="utf-8")
        except Exception:
            pass

    sections: list[str] = []

    # Header
    sections.append(f"# Strategy Brief — {company}")
    sections.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    sections.append("")
    sections.append(
        "This brief is a data-driven planning tool for the human operator. "
        "It does not prescribe content to Stelle. All recommendations are "
        "grounded in the client's observation history and cross-client learning."
    )
    sections.append("")

    # 1. Performance summary
    perf = _build_performance_summary(company)
    if perf:
        sections.append("## Performance Summary")
        sections.append(perf)
        sections.append("")

    # 2. Recommended topics (from transition model + LOLA arms)
    topic_recs = _build_topic_recommendations(company)
    if topic_recs:
        sections.append("## Recommended Topics")
        sections.append(topic_recs)
        sections.append("")

    # 3. Format sequence recommendation
    format_recs = _build_format_recommendations(company)
    if format_recs:
        sections.append("## Format Sequence")
        sections.append(format_recs)
        sections.append("")

    # 4. Top transcript segments
    transcript_recs = _build_transcript_segment_recommendations(company)
    if transcript_recs:
        sections.append("## Top Transcript Segments")
        sections.append(transcript_recs)
        sections.append("")

    # 5. ABM targets
    abm_recs = _build_abm_recommendations(company)
    if abm_recs:
        sections.append("## ABM Targets")
        sections.append(abm_recs)
        sections.append("")

    # 6. Causal drivers (populated by B2 when causal_filter is built)
    causal = _build_causal_section(company)
    if causal:
        sections.append("## Causal Drivers")
        sections.append(causal)
        sections.append("")

    # 7. Cross-client insights
    xclient = _build_cross_client_section(company)
    if xclient:
        sections.append("## Cross-Client Insights")
        sections.append(xclient)
        sections.append("")

    if len(sections) <= 4:  # only header, no substantive sections
        logger.debug("[strategy_brief] %s has insufficient data for a brief", company)
        return None

    brief_md = "\n".join(sections)

    # Persist
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief_md, encoding="utf-8")

    logger.info(
        "[strategy_brief] Generated brief for %s (%d chars, %d sections)",
        company, len(brief_md), sum(1 for s in sections if s.startswith("## ")),
    )

    return brief_md


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _build_performance_summary(company: str) -> str:
    """Last 2 weeks vs client baseline, with notable wins/drops."""
    state = _load_ruan_mei_state(company)
    if state is None:
        return ""

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("reward", {}).get("immediate") is not None
    ]
    if len(scored) < 5:
        return ""

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=_RECENT_WINDOW_DAYS)

    def _parse_ts(obs: dict) -> Optional[datetime]:
        ts = obs.get("posted_at") or obs.get("recorded_at", "")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    recent = []
    baseline = []
    for obs in scored:
        dt = _parse_ts(obs)
        if dt is None:
            continue
        reward = obs.get("reward", {}).get("immediate", 0)
        if dt >= recent_cutoff:
            recent.append((obs, reward))
        else:
            baseline.append((obs, reward))

    if not recent:
        return ""

    recent_mean = sum(r for _, r in recent) / len(recent)
    baseline_mean = (sum(r for _, r in baseline) / len(baseline)) if baseline else 0.0
    delta = recent_mean - baseline_mean

    arrow = "→"
    if delta > 0.3:
        arrow = "↑"
    elif delta < -0.3:
        arrow = "↓"

    lines = [
        f"- Last {_RECENT_WINDOW_DAYS} days: **{len(recent)} posts**, "
        f"avg reward **{recent_mean:+.3f}** ({arrow} {delta:+.3f} vs baseline)",
        f"- Baseline (prior {_BASELINE_WINDOW_DAYS}d): {len(baseline)} posts, "
        f"avg reward {baseline_mean:+.3f}",
        f"- Total scored posts in history: {len(scored)}",
    ]

    # Notable wins and drops in the recent window
    recent_sorted = sorted(recent, key=lambda x: x[1], reverse=True)
    if recent_sorted and recent_sorted[0][1] > 0.5:
        top_obs, top_reward = recent_sorted[0]
        body = (top_obs.get("posted_body") or top_obs.get("post_body") or "")[:120]
        topic = top_obs.get("topic_tag", "untagged")
        lines.append(
            f"- **Notable win:** reward {top_reward:+.3f} — *{topic}*: "
            f"\"{body.strip()}...\""
        )
    if len(recent_sorted) >= 2 and recent_sorted[-1][1] < -0.5:
        low_obs, low_reward = recent_sorted[-1]
        body = (low_obs.get("posted_body") or low_obs.get("post_body") or "")[:120]
        topic = low_obs.get("topic_tag", "untagged")
        lines.append(
            f"- **Notable drop:** reward {low_reward:+.3f} — *{topic}*: "
            f"\"{body.strip()}...\""
        )

    return "\n".join(lines)


def _load_causal_classifications(company: str) -> dict:
    """Load causal_dimensions.json and return {dimension_name: classification}.

    Returns an empty dict if the file doesn't exist (B1 hasn't run yet or
    insufficient data). Used by topic/format sections to caveat recommendations.
    """
    path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {d["dimension"]: d for d in data.get("dimensions", [])}
    except Exception:
        return {}


def _build_topic_recommendations(company: str) -> str:
    """Top topics from transition model, annotated with LOLA performance if available."""
    try:
        from backend.src.utils.topic_transitions import recommend_next_topic
    except Exception:
        return ""

    recs = recommend_next_topic(company, top_k=5)
    if not recs:
        return ""

    # Load LOLA arm data for cross-reference (optional)
    lola_arms: dict = {}
    try:
        lola_path = vortex.memory_dir(company) / "lola_state.json"
        if lola_path.exists():
            lola_data = json.loads(lola_path.read_text(encoding="utf-8"))
            for arm in lola_data.get("arms", []):
                if arm.get("n_pulls", 0) > 0:
                    lola_arms[arm.get("label", "")] = {
                        "mean_reward": arm.get("sum_reward", 0) / max(arm.get("n_pulls", 1), 1),
                        "n_pulls": arm.get("n_pulls", 0),
                    }
    except Exception:
        pass

    # Show the transition context
    path = vortex.memory_dir(company) / "topic_transitions.json"
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        recent = model["topic"]["recent"]
        context_line = f"Recent topic sequence: {' → '.join(recent) if recent else '(none)'}"
    except Exception:
        context_line = ""

    lines = []
    if context_line:
        lines.append(f"*{context_line}*")
        lines.append("")

    # Causal caveat — if topic_tag is confounded or uncertain, flag it.
    # If it's causal, boost confidence.
    causal_map = _load_causal_classifications(company)
    topic_causal = causal_map.get("topic_tag")
    if topic_causal:
        cls = topic_causal.get("classification", "")
        partial = topic_causal.get("partial_correlation", 0)
        if cls == "causal":
            lines.append(
                f"> ✅ **Topic is a causal driver** for this client "
                f"(partial r={partial:+.3f}). Topic recommendations have stronger evidence "
                f"than format or timing for this audience."
            )
            lines.append("")
        elif cls == "confounded":
            lines.append(
                f"> ⚠️ **Topic appears confounded** by other factors for this client "
                f"(marginal {topic_causal.get('marginal_correlation', 0):+.3f} vs "
                f"partial {partial:+.3f}). Treat these recommendations as contextual; "
                f"the actual driver is elsewhere."
            )
            lines.append("")
        elif cls == "uncertain" and abs(partial) >= 0.10:
            lines.append(
                f"> ℹ️ Topic has a weak partial effect ({partial:+.3f}) after controlling "
                f"for other factors. Recommendations are directional, not definitive."
            )
            lines.append("")

    for i, rec in enumerate(recs, 1):
        topic = rec["topic"]
        expected = rec["expected_reward"]
        confidence = rec["confidence"]
        source = rec["source"]

        marker = "🟢" if confidence >= 2 and expected > 0 else "🔵" if source == "unexplored_transition" else "⚪"
        lines.append(
            f"{i}. {marker} **{topic}** — expected reward `{expected:+.3f}` "
            f"({source}, n={confidence})"
        )
        lines.append(f"   > {rec['rationale']}")

    lines.append("")
    lines.append(
        "*Legend: 🟢 reliable direct transition · 🔵 exploration opportunity · "
        "⚪ low-confidence direct transition*"
    )

    return "\n".join(lines)


def _build_format_recommendations(company: str) -> str:
    """Format sequence recommendation from format transitions."""
    try:
        from backend.src.utils.topic_transitions import recommend_next_format
    except Exception:
        return ""

    recs = recommend_next_format(company, top_k=4)
    if not recs:
        return ""

    path = vortex.memory_dir(company) / "topic_transitions.json"
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        recent = model["format"]["recent"]
        context_line = f"Recent format sequence: {' → '.join(recent) if recent else '(none)'}"
    except Exception:
        context_line = ""

    lines = []
    if context_line:
        lines.append(f"*{context_line}*")
        lines.append("")

    # Causal caveat for format
    causal_map = _load_causal_classifications(company)
    format_causal = causal_map.get("format_tag")
    if format_causal:
        cls = format_causal.get("classification", "")
        partial = format_causal.get("partial_correlation", 0)
        marginal = format_causal.get("marginal_correlation", 0)
        if cls == "confounded":
            lines.append(
                f"> ⚠️ **Format is confounded** in this client's data "
                f"(marginal {marginal:+.3f} → partial {partial:+.3f}). "
                f"The apparent format effect is explained by other variables "
                f"(likely topic selection). Treat format as sequence hygiene, "
                f"not as an independent lever."
            )
            lines.append("")
        elif cls == "causal":
            lines.append(
                f"> ✅ **Format is a causal driver** (partial r={partial:+.3f}) — "
                f"these recommendations represent real independent leverage."
            )
            lines.append("")

    for i, rec in enumerate(recs, 1):
        fmt = rec["format"]
        expected = rec["expected_reward"]
        confidence = rec["confidence"]
        source = rec["source"]
        marker = "🟢" if confidence >= 2 and expected > 0 else "🔵" if source == "unexplored_transition" else "⚪"
        lines.append(
            f"{i}. {marker} **{fmt}** — expected reward `{expected:+.3f}` (n={confidence})"
        )

    return "\n".join(lines)


def _build_transcript_segment_recommendations(company: str) -> str:
    """Top segments from unscored transcripts.

    Scores the most recent transcripts (bounded by _MAX_TRANSCRIPTS_SCORED)
    and surfaces top segments per transcript. Uses a file-level cache keyed
    by mtime to skip re-scoring unchanged transcripts.
    """
    try:
        from backend.src.utils.transcript_scorer import score_transcript_file
    except Exception:
        return ""

    transcripts_dir = vortex.transcripts_dir(company)
    if not transcripts_dir.exists():
        return ""

    # Pick text-format transcripts, most recent first
    candidates: list[Path] = []
    for f in transcripts_dir.iterdir():
        if f.is_file() and f.suffix in (".txt", ".md") and f.stat().st_size > 2000:
            candidates.append(f)
    if not candidates:
        return ""

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = candidates[:_MAX_TRANSCRIPTS_SCORED]

    # Use a brief-specific cache keyed by (path, mtime) to skip re-scoring
    cache_path = vortex.memory_dir(company) / "strategy_brief_segment_cache.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    lines: list[str] = []
    cache_updated = False

    for transcript in candidates:
        cache_key = f"{transcript.name}::{int(transcript.stat().st_mtime)}"
        cached_segments = cache.get(cache_key)

        if cached_segments is None:
            segments = score_transcript_file(company, transcript, top_k=_TOP_SEGMENTS_PER_TRANSCRIPT)
            cached_segments = [
                {
                    "rank": s.rank,
                    "predicted_reward": s.predicted_reward,
                    "description": s.description,
                    "text": s.text[:800],
                }
                for s in segments
            ]
            cache[cache_key] = cached_segments
            cache_updated = True

        if not cached_segments:
            continue

        lines.append(f"### From `{transcript.name}`")
        lines.append("")
        # Detect whether scores are available. Either all segments have a
        # predicted_reward (scored via learned or cross-client model) or none
        # do (insufficient data to score — honest fallback).
        has_scores = any(s.get("predicted_reward") is not None for s in cached_segments)
        if not has_scores:
            lines.append(
                "_Insufficient data to score segments for this client. "
                "Needs ≥15 scored observations with source tags to train the "
                "segment model, or a similar client with one for cross-client transfer. "
                "Descriptions shown in document order._"
            )
            lines.append("")

        for s in cached_segments:
            pred = s.get("predicted_reward")
            if pred is not None:
                header = f"**Rank {s['rank']}** | predicted reward `{pred:+.3f}`"
            else:
                header = f"**Segment {s['rank']}** (unscored)"
            lines.append(header)
            desc = s.get("description", "").strip()
            if desc:
                lines.append(f"*{desc}*")
            preview = s["text"].strip()[:500]
            if len(s["text"]) > 500:
                preview += "…"
            lines.append(f"> {preview}")
            lines.append("")

    if cache_updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(cache_path)

    return "\n".join(lines).strip()


def _build_abm_recommendations(company: str) -> str:
    """ABM target recommendations — companies in abm_profiles not recently mentioned."""
    abm_dir = vortex.abm_dir(company)
    if not abm_dir.exists():
        return ""

    # Collect ABM target names from filenames and file content
    abm_targets: list[dict] = []
    for f in sorted(abm_dir.iterdir()):
        if not f.is_file() or f.suffix not in (".txt", ".md", ".json"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Best-effort name extraction: filename stem, or "Company:" line
        name = f.stem.replace("_", " ").replace("-", " ").strip().title()
        for line in text.split("\n")[:10]:
            m = re.match(r"^\s*(?:company|target|account)\s*[:=]\s*(.+)$", line, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                break

        abm_targets.append({
            "name": name,
            "file": f.name,
            "content_preview": text[:200].strip(),
        })

    if not abm_targets:
        return ""

    # Check which ABM targets have been mentioned in recent posts
    state = _load_ruan_mei_state(company)
    recent_bodies = ""
    if state:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=30)
        for obs in state.get("observations", []):
            ts = obs.get("posted_at") or obs.get("recorded_at", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt >= cutoff:
                    recent_bodies += (obs.get("posted_body") or obs.get("post_body") or "") + "\n"
            except Exception:
                continue
    recent_lower = recent_bodies.lower()

    untapped = []
    mentioned = []
    for target in abm_targets:
        name_lower = target["name"].lower()
        first_word = name_lower.split()[0] if name_lower else ""
        if first_word and first_word in recent_lower:
            mentioned.append(target)
        else:
            untapped.append(target)

    lines: list[str] = []
    if untapped:
        lines.append("**Untapped targets** (not mentioned in posts from the last 30 days):")
        for t in untapped[:5]:
            preview = t["content_preview"].replace("\n", " ")[:140]
            lines.append(f"- **{t['name']}** — {preview}…")
        lines.append("")
    if mentioned:
        lines.append(f"**Recently mentioned** ({len(mentioned)} target{'s' if len(mentioned) != 1 else ''}):")
        for t in mentioned[:3]:
            lines.append(f"- {t['name']}")

    return "\n".join(lines).strip()


def _build_causal_section(company: str) -> str:
    """Causal drivers section — populated by B2 when causal_filter is built.

    Reads ``memory/{company}/causal_dimensions.json`` if it exists and
    formats the top causal vs confounded vs inert dimensions.
    """
    path = vortex.memory_dir(company) / "causal_dimensions.json"
    if not path.exists():
        return ""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    causal = [d for d in data.get("dimensions", []) if d.get("classification") == "causal"]
    confounded = [d for d in data.get("dimensions", []) if d.get("classification") == "confounded"]

    lines = []
    lines.append(
        f"*Partial-correlation analysis over {data.get('observation_count', 0)} observations. "
        "Controls for observed confounders; does not identify true causality.*"
    )
    lines.append("")

    if causal:
        lines.append("**Dimensions that predict reward (controlling for other measured factors):**")
        for d in sorted(causal, key=lambda x: abs(x.get("partial_correlation", 0)), reverse=True):
            lines.append(
                f"- `{d['dimension']}` — marginal `{d.get('marginal_correlation', 0):+.3f}`, "
                f"partial `{d.get('partial_correlation', 0):+.3f}` (p={d.get('p_value', 1):.3f})"
            )
        lines.append("")

    if confounded:
        lines.append("**Appears correlated but is confounded** (likely driven by other factors):")
        for d in confounded[:5]:
            lines.append(
                f"- `{d['dimension']}` — marginal `{d.get('marginal_correlation', 0):+.3f}`, "
                f"partial `{d.get('partial_correlation', 0):+.3f}`"
            )

    return "\n".join(lines).strip()


def _build_cross_client_section(company: str) -> str:
    """Surface universal patterns that might apply to this client."""
    patterns_path = vortex.our_memory_dir() / "universal_patterns.json"
    if not patterns_path.exists():
        return ""

    try:
        patterns = json.loads(patterns_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not isinstance(patterns, list) or not patterns:
        return ""

    # Filter to LLM-generated patterns (they have 'pattern' and 'confidence')
    llm_patterns = [
        p for p in patterns
        if p.get("pattern") and p.get("confidence", 0) >= 0.8
    ]
    if not llm_patterns:
        return ""

    # Sort by confidence × evidence
    llm_patterns.sort(
        key=lambda p: p.get("confidence", 0) * p.get("evidence_clients", 1),
        reverse=True,
    )

    lines = [
        "High-confidence patterns observed across multiple B2B clients. "
        "Not all apply to every client — consider which match this client's voice and audience.",
        "",
    ]
    for p in llm_patterns[:5]:
        conf = p.get("confidence", 0)
        clients = p.get("evidence_clients", 0)
        lift = p.get("avg_reward_lift", 0)
        lines.append(f"- **{p.get('category', 'general')}** (confidence {conf:.0%}, "
                     f"{clients} clients, avg lift +{lift:.2f}):")
        lines.append(f"  > {p['pattern'][:300]}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_ruan_mei_state(company: str) -> Optional[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
        if state is not None:
            return state
    except Exception:
        pass

    path = vortex.ruan_mei_state_path(company)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


# ------------------------------------------------------------------
# Compact Stelle-injection version
# ------------------------------------------------------------------

def get_brief_version(company: str) -> Optional[str]:
    """Return a stable version string for the current strategy brief, or None.

    Uses the file's mtime as the version. This changes whenever the brief is
    regenerated, so observations tagged with a version can be correlated with
    the specific brief that was active at generation time.
    """
    path = vortex.memory_dir(company) / "strategy_brief.md"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def build_aglaea_strategy_context(company: str) -> str:
    """Build a strategy brief context block for Aglaea's interview prep agent.

    Returns the FULL strategy brief (markdown) wrapped with an Aglaea-specific
    framing note. Aglaea's workspace context is already large (transcripts,
    references, published posts) — adding the full brief is a small token
    addition relative to the existing context budget, and Aglaea benefits
    from seeing the complete causal caveats and transcript segment scores.

    Returns empty string if no brief exists.
    """
    path = vortex.memory_dir(company) / "strategy_brief.md"
    if not path.exists():
        return ""
    try:
        brief = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not brief:
        return ""

    framing = (
        "## Data-Driven Strategy Context (auto-generated from engagement data)\n\n"
        "The following strategy brief was generated from this client's engagement "
        "history. It contains topic recommendations, format analysis, causal drivers, "
        "and transcript segment scores. Use this data to inform (not dictate) which "
        "interview questions you prioritize. The data is directional, not definitive — "
        "respect the caveats about confidence levels and confounded dimensions.\n\n"
    )
    return framing + brief + "\n"


def build_herta_strategy_context(company: str) -> str:
    """Build a strategy brief context block for Herta's content strategy agent.

    Returns the full strategy brief PLUS the raw ``topic_transitions.json`` and
    ``causal_dimensions.json`` as structured appendices, wrapped with a
    Herta-specific framing note. Herta reasons over content strategy decisions
    and benefits from access to the raw machine-readable data, not just the
    human-formatted markdown.

    Returns empty string if no brief exists.
    """
    brief_path = vortex.memory_dir(company) / "strategy_brief.md"
    if not brief_path.exists():
        return ""
    try:
        brief = brief_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not brief:
        return ""

    parts = [
        "## Learned Strategy Data (auto-generated from engagement history)\n",
        "This client has enough engagement history to produce data-driven strategy "
        "signals. The strategy brief below is the human-facing summary; the raw JSON "
        "appendices contain machine-readable topic transitions and causal dimension "
        "analyses. Use all three to inform (not dictate) your strategy recommendations. "
        "The data is directional, not definitive — respect confidence levels, causal "
        "caveats, and observation counts.\n",
        "### Strategy Brief",
        brief,
    ]

    # Raw topic transitions appendix
    trans_path = vortex.memory_dir(company) / "topic_transitions.json"
    if trans_path.exists():
        try:
            trans = json.loads(trans_path.read_text(encoding="utf-8"))
            parts.append("\n### Topic Transitions (raw JSON)")
            parts.append("```json")
            parts.append(json.dumps(trans, indent=2, ensure_ascii=False))
            parts.append("```")
        except Exception:
            pass

    # Raw causal dimensions appendix
    causal_path = vortex.memory_dir(company) / "causal_dimensions.json"
    if causal_path.exists():
        try:
            causal = json.loads(causal_path.read_text(encoding="utf-8"))
            parts.append("\n### Causal Dimensions (raw JSON)")
            parts.append("```json")
            parts.append(json.dumps(causal, indent=2, ensure_ascii=False))
            parts.append("```")
        except Exception:
            pass

    return "\n\n".join(parts) + "\n"


def build_stelle_strategy_context(company: str) -> str:
    """Build a compact strategy recommendation block for Stelle's user prompt.

    This is the *only* component in the pipeline that accounts for sequence
    (topic transitions, format repetition decay). Injecting a condensed version
    of the brief into Stelle's user_prompt closes the loop between strategy
    analysis and generation — otherwise the brief only reaches the human
    operator and Stelle keeps picking topics/formats blindly.

    Distinct from ``generate_strategy_brief``: the full brief is 5-9k chars
    of markdown for human consumption; this returns a ~1000-1500 char block
    optimized for a generation prompt.

    Returns empty string when there's insufficient data to make a recommendation.
    """
    try:
        from backend.src.utils.topic_transitions import (
            recommend_next_topic, recommend_next_format,
        )
    except Exception:
        return ""

    topic_recs = recommend_next_topic(company, top_k=3) or []
    format_recs = recommend_next_format(company, top_k=3) or []

    if not topic_recs and not format_recs:
        return ""

    causal_map = _load_causal_classifications(company)

    lines = [
        "",
        "",
        "STRATEGY RECOMMENDATION (derived from this client's engagement history):",
        "These are data-driven directions based on what has historically worked. "
        "They are context, not directives. Override when the source material "
        "strongly points elsewhere.",
    ]

    # Topic sequence context
    try:
        model_path = vortex.memory_dir(company) / "topic_transitions.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        recent_topics = model["topic"]["recent"]
        recent_formats = model["format"]["recent"]
    except Exception:
        recent_topics = []
        recent_formats = []

    if recent_topics:
        lines.append(f"\nRecent topic sequence: {' → '.join(recent_topics)}")
    if recent_formats:
        lines.append(f"Recent format sequence: {' → '.join(recent_formats)}")

    # Topic recommendations
    if topic_recs:
        lines.append("\nSuggested next topics (ranked by expected engagement):")
        topic_causal = causal_map.get("topic_tag", {}).get("classification")
        for rec in topic_recs:
            confidence_note = ""
            if rec["confidence"] >= 2:
                confidence_note = f"reliable direct transition (n={rec['confidence']})"
            elif rec["source"] == "unexplored_transition":
                confidence_note = "exploration opportunity (untested after current topic)"
            else:
                confidence_note = f"weak direct transition (n={rec['confidence']})"
            lines.append(
                f"  • {rec['topic']} — expected reward {rec['expected_reward']:+.2f}, "
                f"{confidence_note}"
            )

        if topic_causal == "causal":
            lines.append(
                "  (topic is a validated causal driver for this client — "
                "prioritize these recommendations)"
            )
        elif topic_causal == "confounded":
            lines.append(
                "  (topic appears confounded by other factors — treat as loose context)"
            )

    # Format recommendations
    if format_recs:
        lines.append("\nSuggested next formats:")
        format_causal = causal_map.get("format_tag", {}).get("classification")
        for rec in format_recs:
            confidence_note = (
                f"n={rec['confidence']} history"
                if rec["confidence"] > 0
                else "untested after current format"
            )
            lines.append(
                f"  • {rec['format']} — expected reward {rec['expected_reward']:+.2f}, "
                f"{confidence_note}"
            )

        if format_causal == "confounded":
            lines.append(
                "  (format is confounded by topic for this client — "
                "don't optimize format as an independent lever)"
            )
        elif format_causal == "causal":
            lines.append("  (format is a validated causal driver — meaningful signal)")

    result = "\n".join(lines)
    # Hard cap to prevent runaway context
    if len(result) > _STELLE_COMPACT_MAX_CHARS:
        result = result[:_STELLE_COMPACT_MAX_CHARS] + "\n  [truncated]"

    return result
