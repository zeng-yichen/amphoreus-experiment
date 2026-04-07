"""Feedback Distiller — close the learning→generation loop.

Three signal sources, in decreasing reliability:
1. **Inline editorial feedback** — [FEEDBACK:] comments in feedback/ files
   where the client directly says what to change.
2. **Accepted vs. generated diffs** — accepted/ posts (gold standard) vs.
   Stelle's typical output patterns from RuanMei observations.
3. **Engagement-grounded diagnostics** — what distinguishes the client's
   top-quartile posts from their bottom-quartile, stated as specific
   writing directives (not analytics).

Output: memory/{company}/learned_directives.json — a list of specific,
evidence-grounded rules injected into Stelle's dynamic directives section
at system-prompt authority level.

Also provides `build_engagement_diagnostic(company, draft_text)` which
compares a specific draft against the client's empirical success profile
and returns natural-language coaching.

Usage:
    from backend.src.utils.feedback_distiller import (
        distill_directives,
        build_engagement_diagnostic,
        build_stelle_directives_section,
    )

    # During ordinal_sync:
    distill_directives("innovocommerce")

    # In Stelle's _build_dynamic_directives:
    section = build_stelle_directives_section("innovocommerce")

    # For Cyrene critic context:
    diagnostic = build_engagement_diagnostic("innovocommerce", draft_text)
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_MIN_FEEDBACK_ITEMS = 1      # even one feedback file is worth extracting from
_MIN_OBS_FOR_ENGAGEMENT = 10  # need top/bottom split to be meaningful
_MAX_DIRECTIVES = 8           # cap to avoid prompt bloat
_DIRECTIVE_CACHE_TTL_DAYS = 7
# Engagement-only directives need at least this many observations to earn
# "high" (must-follow) priority. Below this threshold, they're downgraded to
# "medium" (strong preference). At small n, overconfident highs at system-prompt
# authority level are a bigger risk than missed signal. Editorial and accepted-
# source directives bypass this floor — they're direct signal from the client.
_PRIORITY_FLOOR_OBS = 50

# Directive efficacy attribution thresholds.
# A directive needs at least this many posts in each arm (active / not-active)
# before we're willing to classify it. Below this n, classification stays
# "untested" and the directive is kept as-is.
_MIN_OBS_PER_ARM_FOR_EFFICACY = 5
# Effect size thresholds (Cohen's d on z-scored reward).
_VALIDATED_EFFECT_SIZE = 0.2       # d >= +0.2 → validated
_COUNTERPRODUCTIVE_EFFECT_SIZE = -0.2  # d <= -0.2 → counterproductive


# ------------------------------------------------------------------
# Directive identity
# ------------------------------------------------------------------

def _directive_id(directive_text: str) -> str:
    """Stable hash-based ID for a directive.

    Uses SHA1 of the directive text; first 12 hex chars. Stable across
    distiller re-runs as long as the LLM produces the exact same string.
    When the LLM rewords a directive on re-run, it gets a new ID and
    efficacy tracking resets — this is the right behavior: a reworded
    directive is effectively a different directive.
    """
    normalized = directive_text.strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------
# Signal source 1: Inline editorial feedback
# ------------------------------------------------------------------

def _extract_feedback_signals(company: str) -> list[dict]:
    """Extract editorial feedback from feedback/ files.

    Looks for:
    - [FEEDBACK:] inline comments
    - COMMENTS: sections
    - Before/After revision pairs in feedback/edits/
    """
    fb_dir = vortex.feedback_dir(company)
    if not fb_dir.exists():
        return []

    raw_feedback: list[str] = []

    for f in sorted(fb_dir.rglob("*")):
        if not f.is_file() or f.suffix not in (".txt", ".md", ".json"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Extract [FEEDBACK:] inline comments
        for line in text.split("\n"):
            line_stripped = line.strip()
            if "[FEEDBACK:" in line_stripped.upper() or "FEEDBACK:" in line_stripped.upper():
                raw_feedback.append(line_stripped)
            elif "COMMENTS:" in line_stripped.upper() and len(line_stripped) > 20:
                raw_feedback.append(line_stripped)

        # If the file itself is a before/after, include the whole thing
        if "## Before" in text and "## After" in text:
            raw_feedback.append(f"[REVISION DIFF from {f.name}]\n{text[:2000]}")

    return [{"type": "editorial", "text": fb} for fb in raw_feedback if len(fb) > 15]


# ------------------------------------------------------------------
# Signal source 2: Accepted posts (gold standard style)
# ------------------------------------------------------------------

def _extract_accepted_signals(company: str) -> list[dict]:
    """Extract style patterns from accepted/ posts.

    These are the client's gold standard — what they actually want
    their posts to look and sound like.
    """
    accepted_dir = vortex.accepted_dir(company)
    if not accepted_dir.exists():
        return []

    accepted_texts: list[str] = []
    for f in sorted(accepted_dir.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".md") and f.stat().st_size < 10_000:
            try:
                accepted_texts.append(f.read_text(encoding="utf-8", errors="replace").strip())
            except Exception:
                pass

    if not accepted_texts:
        return []

    return [{"type": "accepted", "text": t[:2000]} for t in accepted_texts[:6]]


# ------------------------------------------------------------------
# Signal source 3: Engagement-grounded patterns
# ------------------------------------------------------------------

def _extract_engagement_signals(company: str) -> dict:
    """Extract concrete patterns from top vs. bottom performers.

    Returns a dict with top/bottom posts and their characteristics.
    """
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None

    if state is None:
        state_path = vortex.memory_dir(company) / "ruan_mei_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        else:
            return {}

    scored = [o for o in state.get("observations", []) if o.get("status") == "scored"]
    if len(scored) < _MIN_OBS_FOR_ENGAGEMENT:
        return {}

    scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
    n = len(scored)
    bottom = scored[:max(2, n // 4)]
    top = scored[max(0, n - n // 4):]

    def _summarize(posts: list[dict]) -> dict:
        lengths = [len(o.get("posted_body", o.get("post_body", ""))) for o in posts]
        rewards = [o.get("reward", {}).get("immediate", 0) for o in posts]
        bodies = [(o.get("posted_body", "") or o.get("post_body", ""))[:500] for o in posts]
        analyses = [o.get("descriptor", {}).get("analysis", "")[:300] for o in posts]
        return {
            "count": len(posts),
            "avg_length": int(sum(lengths) / len(lengths)) if lengths else 0,
            "avg_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0,
            "bodies": bodies[:4],
            "analyses": [a for a in analyses if a][:4],
        }

    return {
        "top": _summarize(top),
        "bottom": _summarize(bottom),
        "total_scored": n,
    }


# ------------------------------------------------------------------
# Distillation: signals → directives
# ------------------------------------------------------------------

def distill_directives(company: str, force: bool = False) -> Optional[list[dict]]:
    """Distill all feedback signals into actionable directives.

    Returns list of directive dicts, or None if insufficient signal.
    Caches to memory/{company}/learned_directives.json.

    Args:
        company: Client slug.
        force: If True, bypass cache and recompute from current data. Used
            after signal-altering changes like edit_similarity backfill.
    """
    # Check cache freshness (unless forced)
    cache_path = vortex.memory_dir(company) / "learned_directives.json"
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_at = cached.get("computed_at", "")
            if cached_at:
                dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
                if age_days < _DIRECTIVE_CACHE_TTL_DAYS:
                    return cached.get("directives")
        except Exception:
            pass

    # Gather all signals
    editorial = _extract_feedback_signals(company)
    accepted = _extract_accepted_signals(company)
    engagement = _extract_engagement_signals(company)

    if not editorial and not accepted and not engagement:
        logger.debug("[feedback_distiller] No signals for %s", company)
        return None

    # Build the LLM prompt
    prompt_parts = [
        "You are analyzing editorial feedback, approved posts, and engagement data "
        "for a specific LinkedIn ghostwriting client. Your job is to extract SPECIFIC, "
        "ACTIONABLE writing rules that the ghostwriter must follow.\n\n"
        "Rules must be:\n"
        "- Specific enough that a writer can follow them without interpretation\n"
        "- Grounded in evidence from the data below (cite which signal)\n"
        "- About writing mechanics the writer controls (not audience or timing)\n"
        "- Stated as DO/DON'T directives, not observations\n\n"
    ]

    if editorial:
        prompt_parts.append("EDITORIAL FEEDBACK (direct notes from the client/editor):\n")
        for i, fb in enumerate(editorial[:10], 1):
            prompt_parts.append(f"  [{i}] {fb['text'][:500]}\n")
        prompt_parts.append("\n")

    if accepted:
        prompt_parts.append("APPROVED POSTS (the client's gold standard — this is what they want):\n")
        for i, acc in enumerate(accepted[:4], 1):
            prompt_parts.append(f"  [Approved {i}] {acc['text'][:600]}\n\n")
        prompt_parts.append("\n")

    if engagement:
        top = engagement.get("top", {})
        bottom = engagement.get("bottom", {})
        prompt_parts.append(
            f"ENGAGEMENT DATA ({engagement.get('total_scored', 0)} posts analyzed):\n"
            f"Top performers ({top.get('count', 0)} posts, avg {top.get('avg_length', 0)} chars, "
            f"avg reward {top.get('avg_reward', 0):.3f}):\n"
        )
        for analysis in top.get("analyses", [])[:3]:
            prompt_parts.append(f"  - {analysis[:300]}\n")
        for body in top.get("bodies", [])[:2]:
            prompt_parts.append(f"  [TOP POST EXCERPT] {body[:400]}\n\n")

        prompt_parts.append(
            f"Bottom performers ({bottom.get('count', 0)} posts, avg {bottom.get('avg_length', 0)} chars, "
            f"avg reward {bottom.get('avg_reward', 0):.3f}):\n"
        )
        for analysis in bottom.get("analyses", [])[:3]:
            prompt_parts.append(f"  - {analysis[:300]}\n")
        for body in bottom.get("bodies", [])[:2]:
            prompt_parts.append(f"  [BOTTOM POST EXCERPT] {body[:400]}\n\n")

    scored_obs_count = engagement.get("total_scored", 0) if engagement else 0
    if scored_obs_count < _PRIORITY_FLOOR_OBS:
        floor_note = (
            f"**This client has {scored_obs_count} observations, below the "
            f"{_PRIORITY_FLOOR_OBS}-observation floor. Engagement-source directives "
            f"WILL be mechanically downgraded to 'medium' after your response. "
            f"Reserve 'high' for editorial/accepted-source directives only.**"
        )
    else:
        floor_note = (
            f"This client has {scored_obs_count} observations — above the "
            f"{_PRIORITY_FLOOR_OBS}-observation floor. Engagement-source directives "
            f"may be 'high' if strongly supported by the data."
        )

    prompt_parts.append(
        f"Extract {_MAX_DIRECTIVES} or fewer specific writing directives. "
        "Each directive should be something the ghostwriter can immediately act on.\n\n"
        "PRIORITY RULES:\n"
        '- "high" (must follow): directives backed by editorial feedback or '
        "approved-post patterns (direct signal from the client), OR engagement "
        f"patterns from at least {_PRIORITY_FLOOR_OBS} observations.\n"
        '- "medium" (strong preference): engagement patterns below that threshold, '
        "softer signals, stylistic nudges.\n"
        f"{floor_note}\n\n"
        "Return a JSON array:\n"
        "[\n"
        '  {"directive": "Always open with a specific story from a named interaction '
        '(a conversation, a meeting, a demo) rather than an abstract industry observation", '
        '"evidence": "Top 4 posts all open with \'I spoke with...\' or \'During a demo...\'; '
        'bottom 3 open with generic industry claims", '
        '"source": "engagement", "priority": "medium"},\n'
        '  {"directive": "Use the full term \'clinical operations\' not abbreviations like '
        '\'ClinOps\'", "evidence": "Editor adds the full term in 3 of 4 feedback notes", '
        '"source": "editorial", "priority": "high"},\n'
        "  ...\n"
        "]\n\n"
        'priority is "high" (must follow) or "medium" (strong preference).\n'
        "source is \"editorial\" | \"accepted\" | \"engagement\".\n"
        "Output ONLY the JSON array."
    )

    prompt = "".join(prompt_parts)

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        directives = json.loads(raw)
        if not isinstance(directives, list):
            return None
    except Exception as e:
        logger.warning("[feedback_distiller] Distillation failed for %s: %s", company, e)
        return None

    # Validate and assign stable IDs
    valid = []
    for d in directives[:_MAX_DIRECTIVES]:
        if d.get("directive") and len(d["directive"]) > 10:
            valid.append({
                "id": _directive_id(d["directive"]),
                "directive": d["directive"],
                "evidence": d.get("evidence", ""),
                "source": d.get("source", "engagement"),
                "priority": d.get("priority", "medium"),
                # Efficacy fields — populated by compute_directive_efficacy
                # after enough observations accumulate. "untested" is the
                # default state; the directive is still injected into Stelle
                # at its assigned priority.
                "efficacy_classification": "untested",
                "efficacy_effect_size": None,
                "efficacy_n_active": 0,
                "efficacy_n_inactive": 0,
            })

    if not valid:
        return None

    # Preserve efficacy data from the previous cache when the directive ID
    # matches (i.e., the distiller produced the exact same directive text).
    # This prevents efficacy tracking from resetting on every re-run.
    if cache_path.exists():
        try:
            old = json.loads(cache_path.read_text(encoding="utf-8"))
            old_by_id = {
                od.get("id"): od for od in old.get("directives", [])
                if od.get("id")
            }
            for d in valid:
                prior = old_by_id.get(d["id"])
                if prior:
                    for k in ("efficacy_classification", "efficacy_effect_size",
                              "efficacy_n_active", "efficacy_n_inactive"):
                        if k in prior:
                            d[k] = prior[k]
        except Exception:
            pass

    # Priority floor: enforce mechanically regardless of what the LLM assigned.
    # Engagement-source "high" priority requires >= _PRIORITY_FLOOR_OBS observations.
    # Editorial and accepted-source directives bypass the floor (direct client signal).
    downgrades = 0
    for d in valid:
        if d.get("priority") == "high" and d.get("source") == "engagement":
            if scored_obs_count < _PRIORITY_FLOOR_OBS:
                d["priority"] = "medium"
                d["_floor_downgrade"] = (
                    f"engagement-source at n={scored_obs_count} < {_PRIORITY_FLOOR_OBS}"
                )
                downgrades += 1

    if downgrades:
        logger.info(
            "[feedback_distiller] Priority floor: downgraded %d engagement-only "
            "high-priority directives to medium for %s (n=%d < %d)",
            downgrades, company, scored_obs_count, _PRIORITY_FLOOR_OBS,
        )

    # Cache
    cache_data = {
        "directives": valid,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "signal_counts": {
            "editorial": len(editorial),
            "accepted": len(accepted),
            "engagement_obs": engagement.get("total_scored", 0),
        },
        "priority_floor": {
            "threshold_obs": _PRIORITY_FLOOR_OBS,
            "downgrades_applied": downgrades,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cache_path)

    logger.info(
        "[feedback_distiller] Distilled %d directives for %s (editorial=%d, accepted=%d, engagement=%d)",
        len(valid), company, len(editorial), len(accepted),
        engagement.get("total_scored", 0),
    )

    return valid


# ------------------------------------------------------------------
# Stelle integration: build the directives section
# ------------------------------------------------------------------

def backfill_active_directives(company: str, rm_state: dict, directives: list[dict]) -> int:
    """Backfill active_directives on historical observations.

    For each observation missing active_directives, checks if the directive
    cache existed at the time the observation was scored. If yes, assigns
    the directive IDs to the observation (those directives COULD have been
    active when the post was generated).

    Uses the cache-level computed_at as the earliest timestamp directives
    could have existed (individual directives don't have created_at).
    Idempotent: skips observations that already have active_directives set.

    Returns the count of observations updated.
    """
    if not directives:
        return 0

    # Get the earliest directive timestamp from the cache
    cache_path = vortex.memory_dir(company) / "learned_directives.json"
    cache_computed_at = None
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_computed_at = cache_data.get("computed_at")
        except Exception:
            pass

    if not cache_computed_at:
        return 0  # conservatively skip — we don't know when directives were created

    directive_ids = [d.get("id") for d in directives if d.get("id")]
    if not directive_ids:
        return 0

    updated = 0
    for obs in rm_state.get("observations", []):
        # Skip if already has active_directives populated
        if isinstance(obs.get("active_directives"), list) and obs["active_directives"]:
            continue
        if obs.get("status") != "scored":
            continue

        # Check if the observation was scored after the directives existed
        scored_at = obs.get("scored_at") or obs.get("recorded_at", "")
        if not scored_at:
            continue

        try:
            from datetime import datetime as _dt_bf, timezone as _tz_bf
            obs_ts = _dt_bf.fromisoformat(scored_at.replace("Z", "+00:00"))
            cache_ts = _dt_bf.fromisoformat(cache_computed_at.replace("Z", "+00:00"))
            if obs_ts >= cache_ts:
                obs["active_directives"] = list(directive_ids)
                updated += 1
        except Exception:
            continue

    return updated


def get_active_directive_ids(company: str) -> list[str]:
    """Return the IDs of directives currently active for a client.

    "Active" means present in the distilled cache AND not classified as
    counterproductive by the efficacy attribution. Used by Stelle's
    observation recording code to stamp which directives were in effect
    at generation time, enabling retrospective efficacy analysis.
    """
    cache_path = vortex.memory_dir(company) / "learned_directives.json"
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    ids = []
    for d in data.get("directives", []):
        if d.get("efficacy_classification") == "counterproductive":
            continue  # not actually active; filtered out before injection
        did = d.get("id")
        if did:
            ids.append(did)
    return ids


def build_stelle_directives_section(company: str) -> str:
    """Build a Stelle-ready directives section from learned rules.

    Returns a formatted string for injection into _build_dynamic_directives,
    or empty string if no directives exist.

    Directives classified as "counterproductive" by the efficacy attribution
    (compute_directive_efficacy) are filtered out — they historically produced
    posts that scored worse than posts without them.
    Directives classified as "validated" get a ✓ marker so the model sees
    which rules have proven efficacy.
    """
    cache_path = vortex.memory_dir(company) / "learned_directives.json"
    if not cache_path.exists():
        return ""

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        directives = data.get("directives", [])
    except Exception:
        return ""

    # Strip counterproductive directives entirely
    directives = [
        d for d in directives
        if d.get("efficacy_classification") != "counterproductive"
    ]

    if not directives:
        return ""

    def _marker(d: dict) -> str:
        cls = d.get("efficacy_classification", "untested")
        if cls == "validated":
            return "✓ "  # empirically validated against engagement
        return ""

    lines = [
        "## Learned Writing Rules\n",
        "These rules were extracted from editorial feedback, approved posts, and "
        "engagement data for this specific client. Follow them. "
        "Rules marked with ✓ have been empirically validated by post-publication "
        "engagement attribution — posts generated with those rules active scored "
        "higher than posts without them.\n",
    ]

    high = [d for d in directives if d.get("priority") == "high"]
    medium = [d for d in directives if d.get("priority") != "high"]

    if high:
        lines.append("**Must follow:**")
        for d in high:
            lines.append(f"- {_marker(d)}{d['directive']}")
            if d.get("evidence"):
                lines.append(f"  (Evidence: {d['evidence'][:150]})")
        lines.append("")

    if medium:
        lines.append("**Strong preference:**")
        for d in medium:
            lines.append(f"- {_marker(d)}{d['directive']}")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Efficacy attribution
# ------------------------------------------------------------------

def compute_directive_efficacy(company: str) -> Optional[dict]:
    """Classify each directive as validated / neutral / counterproductive / untested.

    For each directive in the current cache:
    1. Partition scored observations into (directive was active, directive was not)
       using the ``active_directives`` field populated at generation time.
    2. Compute Cohen's d on reward.immediate between the two arms.
    3. Classify:
       - n_active < _MIN_OBS_PER_ARM_FOR_EFFICACY OR n_inactive < same → untested
       - d >= _VALIDATED_EFFECT_SIZE → validated
       - d <= _COUNTERPRODUCTIVE_EFFECT_SIZE → counterproductive
       - otherwise → neutral
    4. Write the updated classifications back into learned_directives.json.

    Returns a summary dict, or None if no directives exist.

    This is the 'feedback loop on the feedback loop' — it prevents wrong
    directives at system-prompt authority from degrading generation quality
    indefinitely.
    """
    cache_path = vortex.memory_dir(company) / "learned_directives.json"
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    directives = data.get("directives", [])
    if not directives:
        return None

    # Load scored observations with both a reward and an active_directives field.
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None
    if state is None:
        return None

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("reward", {}).get("immediate") is not None
    ]

    # An observation contributes to attribution only if active_directives was
    # recorded (a list, even if empty). Observations from before this feature
    # shipped don't have the field and are excluded.
    attributable = [
        o for o in scored
        if isinstance(o.get("active_directives"), list)
    ]

    if not attributable:
        # No data yet — leave all directives as "untested". This is the
        # expected state immediately after shipping directive tracking; the
        # attribution activates as new observations accumulate.
        return {
            "company": company,
            "directives_total": len(directives),
            "attributable_observations": 0,
            "classifications": {"untested": len(directives)},
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    classifications_count = {
        "validated": 0, "neutral": 0, "counterproductive": 0, "untested": 0,
    }

    for d in directives:
        did = d.get("id")
        if not did:
            classifications_count["untested"] += 1
            continue

        active_rewards = []
        inactive_rewards = []
        for obs in attributable:
            reward = obs.get("reward", {}).get("immediate", 0)
            if did in obs.get("active_directives", []):
                active_rewards.append(reward)
            else:
                inactive_rewards.append(reward)

        n_active = len(active_rewards)
        n_inactive = len(inactive_rewards)

        if n_active < _MIN_OBS_PER_ARM_FOR_EFFICACY or n_inactive < _MIN_OBS_PER_ARM_FOR_EFFICACY:
            d["efficacy_classification"] = "untested"
            d["efficacy_effect_size"] = None
            d["efficacy_n_active"] = n_active
            d["efficacy_n_inactive"] = n_inactive
            classifications_count["untested"] += 1
            continue

        # Cohen's d with pooled standard deviation
        mean_active = sum(active_rewards) / n_active
        mean_inactive = sum(inactive_rewards) / n_inactive
        var_active = sum((r - mean_active) ** 2 for r in active_rewards) / max(n_active - 1, 1)
        var_inactive = sum((r - mean_inactive) ** 2 for r in inactive_rewards) / max(n_inactive - 1, 1)
        pooled_sd = math.sqrt(
            ((n_active - 1) * var_active + (n_inactive - 1) * var_inactive)
            / max(n_active + n_inactive - 2, 1)
        )
        effect_size = (mean_active - mean_inactive) / pooled_sd if pooled_sd > 0 else 0.0

        if effect_size >= _VALIDATED_EFFECT_SIZE:
            cls = "validated"
        elif effect_size <= _COUNTERPRODUCTIVE_EFFECT_SIZE:
            cls = "counterproductive"
        else:
            cls = "neutral"

        d["efficacy_classification"] = cls
        d["efficacy_effect_size"] = round(effect_size, 4)
        d["efficacy_n_active"] = n_active
        d["efficacy_n_inactive"] = n_inactive
        classifications_count[cls] += 1

    # Persist updated cache with new classifications
    data["directives"] = directives
    data["efficacy"] = {
        "attributable_observations": len(attributable),
        "classifications": classifications_count,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cache_path)

    logger.info(
        "[feedback_distiller] Directive efficacy for %s: "
        "%d validated, %d counterproductive, %d neutral, %d untested "
        "(attributable n=%d)",
        company,
        classifications_count["validated"],
        classifications_count["counterproductive"],
        classifications_count["neutral"],
        classifications_count["untested"],
        len(attributable),
    )

    return {
        "company": company,
        "directives_total": len(directives),
        "attributable_observations": len(attributable),
        "classifications": classifications_count,
        "computed_at": data["efficacy"]["computed_at"],
    }


# ------------------------------------------------------------------
# Engagement diagnostic for a specific draft
# ------------------------------------------------------------------

def build_engagement_diagnostic(company: str, draft_text: str) -> str:
    """Compare a specific draft against the client's success profile.

    Returns natural-language coaching grounded in empirical data.
    For injection into Cyrene's critic prompt or standalone use.
    """
    engagement = _extract_engagement_signals(company)
    if not engagement:
        return ""

    top = engagement.get("top", {})
    bottom = engagement.get("bottom", {})
    if not top.get("count") or not bottom.get("count"):
        return ""

    draft_len = len(draft_text)

    # Build comparison context
    parts = []

    # Length comparison
    top_avg_len = top.get("avg_length", 0)
    bottom_avg_len = bottom.get("avg_length", 0)
    if top_avg_len > 0 and bottom_avg_len > 0:
        if abs(top_avg_len - bottom_avg_len) > 200:
            if draft_len > max(top_avg_len, bottom_avg_len) + 300:
                parts.append(
                    f"This draft is {draft_len} chars. Top performers average "
                    f"{top_avg_len} chars; bottom performers average {bottom_avg_len} chars. "
                    f"Consider cutting."
                )
            elif draft_len < min(top_avg_len, bottom_avg_len) - 300:
                parts.append(
                    f"This draft is {draft_len} chars. Top performers average "
                    f"{top_avg_len} chars. It may need more substance."
                )

    # Opening pattern comparison
    draft_opening = draft_text[:200].strip()
    top_openings = [b[:200] for b in top.get("bodies", []) if b]
    bottom_openings = [b[:200] for b in bottom.get("bodies", []) if b]

    # Use LLM for the actual comparison — this is the key differentiator
    # from the numeric engagement predictor
    prompt = (
        "You are coaching a LinkedIn ghostwriter on a specific draft. "
        "Compare this draft against the client's empirical success patterns.\n\n"
        f"DRAFT OPENING ({draft_len} chars total):\n{draft_text[:500]}\n\n"
    )

    if top_openings:
        prompt += "TOP PERFORMING POSTS (highest engagement) — openings:\n"
        for i, o in enumerate(top_openings[:3], 1):
            prompt += f"  [{i}] {o}\n"
        if top.get("analyses"):
            prompt += "\nWhat made them work:\n"
            for a in top["analyses"][:2]:
                prompt += f"  - {a[:250]}\n"
        prompt += f"\n  Average length: {top_avg_len} chars, avg reward: {top['avg_reward']:.3f}\n\n"

    if bottom_openings:
        prompt += "BOTTOM PERFORMING POSTS — openings:\n"
        for i, o in enumerate(bottom_openings[:3], 1):
            prompt += f"  [{i}] {o}\n"
        if bottom.get("analyses"):
            prompt += "\nWhat went wrong:\n"
            for a in bottom["analyses"][:2]:
                prompt += f"  - {a[:250]}\n"
        prompt += f"\n  Average length: {bottom_avg_len} chars, avg reward: {bottom['avg_reward']:.3f}\n\n"

    prompt += (
        "Write 2-3 sentences of specific coaching for this draft. "
        "Is it more like the top performers or the bottom performers? "
        "What specific changes would push it toward the top pattern? "
        "Be concrete — reference the draft's actual opening and structure. "
        "No generic advice. No bullet points."
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        diagnostic = resp.content[0].text.strip()
    except Exception as e:
        logger.warning("[feedback_distiller] Diagnostic generation failed: %s", e)
        # Fall back to pure numeric comparison
        diagnostic = "\n".join(parts) if parts else ""

    if parts and diagnostic:
        return "\n".join(parts) + "\n\n" + diagnostic
    return diagnostic or "\n".join(parts)
