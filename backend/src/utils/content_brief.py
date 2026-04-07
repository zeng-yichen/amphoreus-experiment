"""Content brief — translates analyst findings into interview objectives.

The analyst discovers what predicts engagement. The interview is where
the raw material for that content gets extracted. This module connects
them: it reads the analyst's model, findings, and topic transitions,
and produces a concrete brief telling the interviewer exactly what
kinds of stories to extract.

The brief answers: "We need source material for N posts matching the
analyst's winning pattern. Here's what that pattern looks like and
what kinds of questions would surface it."

Usage:
    from backend.src.utils.content_brief import build_interview_brief

    brief = build_interview_brief("hensley-biostats", n_posts=6)
    # → structured dict with content targets, source material gaps,
    #   and specific question guidance
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)


def build_interview_brief(company: str, n_posts: int = 6) -> Optional[dict]:
    """Build an interview brief from whatever data exists for this client.

    Three tiers of data, used in combination:
    1. Client's own analyst findings (if analyst has run)
    2. Client's own scored observations (even without analyst)
    3. Cross-client patterns and LinkedIn-wide benchmarks (always available)

    A brief is ALWAYS produced — even for a brand-new client with zero
    observations. The brief evolves continuously as data accumulates:
    - 0 obs: cross-client patterns + ICP-based recommendations
    - 5 obs: early client-specific signals from observation tags
    - 15+ obs: full analyst model + validated findings

    Returns a structured dict. Returns None only if the client directory
    doesn't exist at all.
    """
    if not vortex.memory_dir(company).exists():
        return None

    # --- Load whatever client data exists ---

    # Analyst findings (may not exist for new clients)
    latest_findings = []
    analyst_runs = []
    analyst_path = vortex.memory_dir(company) / "analyst_findings.json"
    if analyst_path.exists():
        try:
            af = json.loads(analyst_path.read_text(encoding="utf-8"))
            findings = af.get("findings", [])
            analyst_runs = af.get("runs", [])
            if analyst_runs:
                latest_rid = analyst_runs[-1].get("run_id")
                if latest_rid:
                    latest_findings = [f for f in findings if f.get("run_id") == latest_rid]
                else:
                    latest_findings = [f for f in findings if f.get("run_id") is None] or findings[-10:]
            else:
                latest_findings = findings[-10:]
        except Exception:
            pass

    # Scored observations (may be empty for new clients)
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None

    scored = []
    if state:
        scored = [o for o in state.get("observations", []) if o.get("status") == "scored"]

    # --- All findings for context (unsplit) ---
    all_analyst_findings = [
        {
            "finding": f.get("claim", "")[:250],
            "confidence": f.get("confidence", "suggestive"),
        }
        for f in latest_findings
        if f.get("type") != "model"
    ]

    # --- Cross-client patterns (always available, even for 0 observations) ---
    cross_client_patterns = []
    try:
        patterns_path = vortex.our_memory_dir() / "universal_patterns.json"
        if patterns_path.exists():
            patterns = json.loads(patterns_path.read_text(encoding="utf-8"))
            cross_client_patterns = [
                {
                    "pattern": p.get("pattern", "")[:200],
                    "confidence": p.get("confidence", 0),
                    "clients": p.get("evidence_clients", 0),
                    "lift": p.get("avg_reward_lift", 0),
                }
                for p in sorted(
                    patterns,
                    key=lambda x: x.get("confidence", 0) * x.get("evidence_clients", 1),
                    reverse=True,
                )[:5]
                if p.get("confidence", 0) >= 0.8
            ]
    except Exception:
        pass

    # --- ICP definition (for new clients without observation data) ---
    icp_context = None
    try:
        icp_path = vortex.icp_definition_path(company)
        if icp_path.exists():
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            icp_context = icp.get("description", "")[:300]
    except Exception:
        pass

    # --- Build topic mix recommendation ---
    topic_dist = Counter(o.get("topic_tag") for o in scored if o.get("topic_tag"))
    format_dist = Counter(o.get("format_tag") for o in scored if o.get("format_tag"))

    # Find top-performing topics from observations
    topic_rewards = {}
    for o in scored:
        t = o.get("topic_tag")
        r = o.get("reward", {}).get("immediate")
        if t and r is not None:
            topic_rewards.setdefault(t, []).append(r)
    topic_performance = {
        t: sum(rs) / len(rs)
        for t, rs in topic_rewards.items()
    }
    best_topics = sorted(topic_performance.items(), key=lambda x: x[1], reverse=True)

    # Same for formats
    format_rewards = {}
    for o in scored:
        f = o.get("format_tag")
        r = o.get("reward", {}).get("immediate")
        if f and r is not None:
            format_rewards.setdefault(f, []).append(r)
    format_performance = {
        f: sum(rs) / len(rs)
        for f, rs in format_rewards.items()
    }
    best_formats = sorted(format_performance.items(), key=lambda x: x[1], reverse=True)

    # --- Build the recommended content plan ---
    # Three tiers: client-specific (15+ obs) → early signal (5+ obs) → cross-client (0 obs)
    content_plan = []
    data_tier = "cross_client"  # default for new clients

    if best_topics and len(scored) >= 5:
        # Client has enough data for topic/format performance rankings
        data_tier = "client_specific" if len(scored) >= 15 else "early_signal"
        top_topic = best_topics[0][0]
        top_format = best_formats[0][0] if best_formats else "storytelling"
        second_topic = best_topics[1][0] if len(best_topics) > 1 else top_topic

        n_top = max(1, round(n_posts * 0.6))
        n_second = max(1, round(n_posts * 0.25))
        n_explore = max(0, n_posts - n_top - n_second)

        confidence_note = (
            f"(avg reward {topic_performance.get(top_topic, 0):+.2f}, "
            f"from {len(scored)} scored posts)"
        )
        content_plan.append({
            "topic": top_topic,
            "format": top_format,
            "count": n_top,
            "rationale": f"Top-performing topic {confidence_note}",
        })
        if second_topic != top_topic:
            content_plan.append({
                "topic": second_topic,
                "format": best_formats[1][0] if len(best_formats) > 1 else top_format,
                "count": n_second,
                "rationale": f"Second topic (avg reward {topic_performance.get(second_topic, 0):+.2f}) for variety",
            })
        if n_explore > 0:
            content_plan.append({
                "topic": "exploration",
                "format": "any",
                "count": n_explore,
                "rationale": "Untested territory — to expand the model's coverage",
            })
    else:
        # New client or very few observations — use cross-client patterns
        content_plan.append({
            "topic": "client's core domain",
            "format": "storytelling",
            "count": max(1, round(n_posts * 0.5)),
            "rationale": "Storytelling is the top format across 22 clients (avg lift +0.38). Focus on the client's primary expertise area.",
        })
        content_plan.append({
            "topic": "client's core domain",
            "format": "contrarian",
            "count": max(1, round(n_posts * 0.25)),
            "rationale": "Contrarian/hot take format tests audience appetite for opinion-led content (mixed results across clients — worth testing early).",
        })
        content_plan.append({
            "topic": "exploration",
            "format": "any",
            "count": max(1, n_posts - round(n_posts * 0.5) - round(n_posts * 0.25)),
            "rationale": "Early-stage exploration — each post teaches the system what works for this specific client.",
        })

    # --- Load the analyst's model for the hook guidance ---
    model_entry = None
    for f in reversed(latest_findings):
        if f.get("type") == "model":
            model_entry = f
            break

    hook_guidance = None
    if model_entry:
        spec = model_entry.get("model_spec", {})
        heuristic = spec.get("heuristic_layer", {})
        if heuristic.get("scoring_guidance", {}).get("hook_bonus"):
            hook_guidance = (
                "The analyst's model gives a bonus for posts that open with a "
                "verbatim customer/prospect quote in the first few lines. "
                "Interview questions should extract specific conversations: "
                "'Walk me through the exact conversation you had with that CEO.' "
                "'What did they say, word for word, when they saw the results?'"
            )

    brief = {
        "company": company,
        "n_posts_target": n_posts,
        "data_tier": data_tier,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_plan": content_plan,
        "analyst_findings": all_analyst_findings[:8],
        "cross_client_patterns": cross_client_patterns[:5],
        "hook_guidance": hook_guidance,
        "icp_context": icp_context,
        "topic_distribution": dict(topic_dist.most_common()) if topic_dist else {},
        "format_distribution": dict(format_dist.most_common()) if format_dist else {},
        "best_topics": [{"topic": t, "avg_reward": round(r, 3)} for t, r in best_topics[:5]],
        "best_formats": [{"format": f, "avg_reward": round(r, 3)} for f, r in best_formats[:5]],
        "observation_count": len(scored),
        "analyst_runs": len(analyst_runs),
    }

    return brief


def build_aglaea_interview_objectives(company: str, n_posts: int = 6) -> str:
    """Build a markdown interview objectives section for Aglaea's user prompt.

    This is the connection between the analyst's findings and the interview.
    Instead of "find interesting untold stories," it says "we need source
    material for THESE specific content types because the data says they'll
    perform."

    Returns empty string if no analyst data exists.
    """
    brief = build_interview_brief(company, n_posts=n_posts)
    if not brief:
        return ""

    lines = [
        "## Interview Objectives (from engagement data)\n",
        "The analyst has identified what content performs best for this client. "
        "Design your interview questions to extract source material for these "
        "specific content types. Every post we generate should trace back to "
        "a moment you surface in this interview.\n",
    ]

    # Content plan
    plan = brief.get("content_plan", [])
    if plan:
        lines.append(f"### Content targets for the next {brief['n_posts_target']} posts\n")
        for p in plan:
            if p["topic"] == "exploration":
                lines.append(
                    f"- **{p['count']} exploration post(s)**: ask about topics the client "
                    f"hasn't discussed before — new projects, recent surprises, industry "
                    f"shifts. This teaches the system about untested content territory."
                )
            else:
                lines.append(
                    f"- **{p['count']} post(s) on {p['topic']}** in **{p['format']}** format "
                    f"— {p['rationale']}"
                )
        lines.append("")

    # Hook/source material guidance
    hook = brief.get("hook_guidance")
    if hook:
        lines.append("### How to extract the best material\n")
        lines.append(f"{hook}\n")

    # Analyst findings (client-specific, when available)
    findings = brief.get("analyst_findings", [])
    if findings:
        lines.append("### What the data says about this client's engagement\n")
        lines.append(
            "Use these findings to weight your questions. Dig deeper into "
            "topics and styles that drive engagement. Redirect away from "
            "patterns that consistently underperform.\n"
        )
        for f in findings:
            lines.append(f"- **[{f['confidence']}]** {f['finding']}")
        lines.append("")

    # Cross-client patterns (always available, especially important for new clients)
    cross_client = brief.get("cross_client_patterns", [])
    if cross_client:
        tier = brief.get("data_tier", "cross_client")
        if tier == "cross_client":
            lines.append("### What works across similar clients (no client-specific data yet)\n")
            lines.append(
                "These patterns were discovered across 22+ B2B clients and hold "
                "with 80-93% confidence. Use them as a starting framework until "
                "this client's own data accumulates.\n"
            )
        else:
            lines.append("### Cross-client patterns (supplementary)\n")
        for p in cross_client:
            lines.append(
                f"- (confidence {p['confidence']:.0%}, {p['clients']} clients, "
                f"avg lift +{p['lift']:.2f}) {p['pattern']}"
            )
        lines.append("")

    return "\n".join(lines)
