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
    """Build an interview brief from the analyst's findings.

    Translates statistical findings into concrete interview objectives:
    what topics to target, what formats to aim for, what kinds of stories
    to extract, and what to avoid.

    Returns a structured dict, or None if insufficient analyst data.
    """
    # Load analyst findings
    analyst_path = vortex.memory_dir(company) / "analyst_findings.json"
    if not analyst_path.exists():
        return None

    try:
        af = json.loads(analyst_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    findings = af.get("findings", [])
    runs = af.get("runs", [])
    if not findings:
        return None

    # Get latest run's findings
    if runs:
        latest_rid = runs[-1].get("run_id")
        if latest_rid:
            latest_findings = [f for f in findings if f.get("run_id") == latest_rid]
        else:
            latest_findings = [f for f in findings if f.get("run_id") is None] or findings[-10:]
    else:
        latest_findings = findings[-10:]

    if not latest_findings:
        return None

    # Load observation data for topic/format distribution
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
    # Don't try to classify findings into target/avoid buckets via keywords.
    # That produces double-counting (a finding mentioning both "highest" and
    # "worst" ends up in both). Instead, show all findings and let the LLM
    # interpret them. The content plan provides the structured targeting.
    all_analyst_findings = [
        {
            "finding": f.get("claim", "")[:250],
            "confidence": f.get("confidence", "suggestive"),
        }
        for f in latest_findings
        if f.get("type") != "model"  # skip model specs, show discoveries only
    ]

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
    # Allocate n_posts across the best-performing topic/format combinations
    content_plan = []
    if best_topics and best_formats:
        top_topic = best_topics[0][0] if best_topics else "general"
        top_format = best_formats[0][0] if best_formats else "storytelling"
        second_topic = best_topics[1][0] if len(best_topics) > 1 else top_topic

        # Weighted allocation: 60% top topic, 25% second topic, 15% exploration
        n_top = max(1, round(n_posts * 0.6))
        n_second = max(1, round(n_posts * 0.25))
        n_explore = max(0, n_posts - n_top - n_second)

        content_plan.append({
            "topic": top_topic,
            "format": top_format,
            "count": n_top,
            "rationale": f"Top-performing topic (avg reward {topic_performance.get(top_topic, 0):+.2f}) in top format",
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

    # --- Load the analyst's model for the hook guidance ---
    model_entry = None
    for f in reversed(findings):
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_plan": content_plan,
        "analyst_findings": all_analyst_findings[:8],
        "hook_guidance": hook_guidance,
        "topic_distribution": dict(topic_dist.most_common()),
        "format_distribution": dict(format_dist.most_common()),
        "best_topics": [{"topic": t, "avg_reward": round(r, 3)} for t, r in best_topics[:5]],
        "best_formats": [{"format": f, "avg_reward": round(r, 3)} for f, r in best_formats[:5]],
        "observation_count": len(scored),
        "analyst_runs": len(runs),
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

    # All analyst findings — the LLM interprets these to design questions.
    # Not pre-classified into target/avoid because findings are nuanced
    # (a single finding can describe both what works and what doesn't).
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

    return "\n".join(lines)
