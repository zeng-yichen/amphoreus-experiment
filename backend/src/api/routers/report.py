"""Client Progress Report API — generates presentation-ready reports for client video calls.

Covers:
  1. Account progress in the last N weeks (posts published, engagement metrics)
  2. Content strategy applied (topics, formats, LOLA performance, learned directives)
  3. Strategy shifts for upcoming weeks (analyst findings, data-driven recommendations)

Endpoints:
  GET /api/report/{company}          — JSON report data
  GET /api/report/{company}/html     — HTML as JSON: {"html": "..."}
  GET /api/report/{company}/view     — HTML directly for browser rendering
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/report", tags=["report"])

# ---------------------------------------------------------------------------
# Ordinal API helpers
# ---------------------------------------------------------------------------

_API_BASE = "https://app.tryordinal.com/api/v1"


def _load_ordinal_keys() -> dict[str, tuple[str, str]]:
    """Returns {slug: (company_id, api_key)} from ordinal_auth_rows.csv."""
    from backend.src.db import vortex
    csv_path = vortex.MEMORY_ROOT / "ordinal_auth_rows.csv"
    if not csv_path.exists():
        # Fallback: try openclaw workspace secrets
        csv_path = Path.home() / ".openclaw" / "workspace" / "secrets" / "ordinal-keys.csv"
    keys: dict[str, tuple[str, str]] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                slug = row.get("provider_org_slug") or row.get("slug", "")
                keys[slug] = (row.get("company_id", ""), row.get("api_key", ""))
    return keys


def _clean_ordinal_markup(text: str) -> str:
    """Strip Ordinal's LinkedIn mention markup: @[Name](urn:li:...) → Name."""
    import re
    return re.sub(r"@\[([^\]]+)\]\([^)]+\)", r"\1", text)


def _reward_label(reward: float) -> str:
    """Translate z-scored reward into client-friendly performance label."""
    if reward > 1.0:
        return "well above average"
    elif reward > 0.3:
        return "above average"
    elif reward > -0.3:
        return "near average"
    elif reward > -1.0:
        return "below average"
    else:
        return "well below average"


def _ordinal_get(endpoint: str, api_key: str) -> dict | list | None:
    url = f"{_API_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read(), strict=False)
    except Exception as e:
        logger.warning("Ordinal API error for %s: %s", endpoint, e)
        return None


def _fetch_all_posts(api_key: str) -> list[dict]:
    posts: list[dict] = []
    cursor = None
    while True:
        ep = "/posts?limit=100"
        if cursor:
            ep += f"&cursor={cursor}"
        data = _ordinal_get(ep, api_key)
        if not data or isinstance(data, list):
            break
        batch = data.get("posts", [])
        posts.extend(batch)
        if data.get("hasMore") and data.get("nextCursor"):
            cursor = data["nextCursor"]
        else:
            break
    return posts


def _extract_profile_id(posts: list[dict]) -> str | None:
    for p in posts:
        profile = (p.get("linkedIn") or {}).get("profile") or {}
        pid = profile.get("id")
        if pid:
            return pid
    return None


def _extract_profile_name(posts: list[dict]) -> str:
    for p in posts:
        profile = (p.get("linkedIn") or {}).get("profile") or {}
        name = profile.get("name")
        if name:
            return name
    return "Unknown"


def _fetch_analytics(api_key: str, profile_id: str, start: str, end: str) -> list[dict]:
    data = _ordinal_get(
        f"/analytics/linkedin/{profile_id}/posts?startDate={start}&endDate={end}",
        api_key,
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("posts", [])
    return []


# ---------------------------------------------------------------------------
# Amphoreus data helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_observations(company: str) -> list[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        from backend.src.db import vortex
        state = _load_json(vortex.memory_dir(company) / "ruan_mei_state.json")
    return (state or {}).get("observations", [])


def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report data assembly
# ---------------------------------------------------------------------------

def _build_report(company: str, weeks: int = 2) -> dict:
    """Assemble the full report data structure."""
    from backend.src.db import vortex

    keys = _load_ordinal_keys()
    api_key = keys.get(company, ("", ""))[1]

    today = date.today()
    cutoff = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(weeks=weeks)
    start_date = (today - timedelta(days=90)).isoformat()
    end_date = today.isoformat()

    # Ordinal data
    posts: list[dict] = []
    analytics: list[dict] = []
    profile_name = company
    if api_key:
        posts = _fetch_all_posts(api_key)
        profile_id = _extract_profile_id(posts)
        profile_name = _extract_profile_name(posts)
        if profile_id:
            analytics = _fetch_analytics(api_key, profile_id, start_date, end_date)

    # Analytics indexed by ordinal post id
    analytics_by_id: dict[str, dict] = {}
    for a in analytics:
        oid = (a.get("ordinalPost") or {}).get("id")
        if oid:
            analytics_by_id[oid] = a

    # Amphoreus data
    observations = _load_observations(company)
    lola = _load_json(vortex.memory_dir(company) / "lola_state.json")
    analyst = _load_json(vortex.memory_dir(company) / "analyst_findings.json")
    directives = _load_json(vortex.memory_dir(company) / "learned_directives.json")

    # --- Posts published in window ---
    published = []
    for p in posts:
        if p.get("status") != "Posted":
            continue
        pub = p.get("publishedAt") or p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and ts >= cutoff:
            pid = p.get("id", "")
            a = analytics_by_id.get(pid, {})
            # Skip posts without analytics (just published, data not in yet)
            if not a.get("impressionCount"):
                continue
            copy_text = (p.get("linkedIn") or {}).get("copy") or ""
            hook = copy_text.split("\n")[0][:120] if copy_text else ""
            eng_rate = a.get("engagement")
            published.append({
                "date": ts.isoformat(),
                "date_display": ts.strftime("%a %b %d"),
                "title": _clean_ordinal_markup(p.get("title") or "(untitled)"),
                "hook": _clean_ordinal_markup(hook),
                "impressions": a.get("impressionCount", 0),
                "likes": a.get("likeCount", 0),
                "comments": a.get("commentCount", 0),
                "reposts": a.get("shareCount", 0),
                "engagement_rate_pct": f"{eng_rate * 100:.1f}%" if eng_rate else "—",
            })
    published.sort(key=lambda x: x["date"])

    total_imp = sum(p["impressions"] for p in published)
    total_likes = sum(p["likes"] for p in published)
    total_comments = sum(p["comments"] for p in published)
    n_published = len(published)

    # --- Period-over-period comparison ---
    prior_cutoff = cutoff - timedelta(weeks=weeks)
    prior_published = []
    for p in posts:
        if p.get("status") != "Posted":
            continue
        pub = p.get("publishedAt") or p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and prior_cutoff <= ts < cutoff:
            a = analytics_by_id.get(p.get("id", ""), {})
            if a.get("impressionCount"):
                prior_published.append({
                    "impressions": a.get("impressionCount", 0),
                    "likes": a.get("likeCount", 0),
                    "comments": a.get("commentCount", 0),
                })
    prior_imp = sum(p["impressions"] for p in prior_published)
    prior_likes = sum(p["likes"] for p in prior_published)
    prior_comments = sum(p["comments"] for p in prior_published)
    prior_count = len(prior_published)

    period_comparison = None
    if prior_count > 0 and n_published > 0:
        imp_change = ((total_imp - prior_imp) / prior_imp * 100) if prior_imp > 0 else 0
        period_comparison = {
            "prior_period": f"{prior_cutoff.strftime('%b %d')} – {cutoff.strftime('%b %d')}",
            "prior_posts": prior_count,
            "prior_impressions": prior_imp,
            "prior_avg_impressions": prior_imp // max(prior_count, 1),
            "prior_reactions": prior_likes,
            "prior_comments": prior_comments,
            "impressions_change_pct": f"{imp_change:+.0f}%",
            "posts_change": n_published - prior_count,
        }

    # --- Upcoming posts ---
    future_cutoff = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    upcoming = []
    for p in posts:
        if p.get("status") == "Posted":
            continue
        pub = p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and ts >= future_cutoff:
            upcoming.append({
                "date": ts.isoformat(),
                "date_display": ts.strftime("%a %b %d"),
                "title": _clean_ordinal_markup(p.get("title") or "(untitled)"),
                "status": p.get("status", "?"),
            })
    upcoming.sort(key=lambda x: x["date"])

    # --- Recent observations (window) ---
    recent_obs = [o for o in observations if (_parse_ts(o.get("posted_at", "")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    scored_recent = [o for o in recent_obs if o.get("status") == "scored"]

    # Topics and formats in window
    topics: dict[str, int] = {}
    formats: dict[str, int] = {}
    for o in recent_obs:
        t = o.get("topic_tag")
        f = o.get("format_tag")
        if t:
            topics[t] = topics.get(t, 0) + 1
        if f:
            formats[f] = formats.get(f, 0) + 1

    # Window performance — translated to client-friendly language
    window_performance = None
    if scored_recent:
        rewards = [o.get("reward", {}).get("immediate", 0) for o in scored_recent]
        avg_r = sum(rewards) / len(rewards)
        best = max(scored_recent, key=lambda o: o.get("reward", {}).get("immediate", 0))
        worst = min(scored_recent, key=lambda o: o.get("reward", {}).get("immediate", 0))
        window_performance = {
            "summary": (
                f"This period's posts performed {'above' if avg_r > 0.1 else 'near' if avg_r > -0.1 else 'below'} "
                f"your historical average."
            ),
            "best": {
                "performance": _reward_label(best.get("reward", {}).get("immediate", 0)),
                "impressions": (best.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0),
                "hook": (best.get("post_body") or best.get("posted_body") or "")[:120].split("\n")[0],
            },
            "worst": {
                "performance": _reward_label(worst.get("reward", {}).get("immediate", 0)),
                "impressions": (worst.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0),
                "hook": (worst.get("post_body") or worst.get("posted_body") or "")[:120].split("\n")[0],
            },
        }

    # --- Content strategy testing (LOLA arms, translated for client) ---
    strategy_testing = []
    if lola:
        for a in lola.get("arms", []):
            pulls = a.get("n_pulls", 0)
            if pulls > 0:
                avg = a.get("sum_reward", 0) / max(pulls, 1)
                # Use description (client-friendly) instead of label (internal)
                desc = a.get("description") or a.get("label", "?").replace("_", " ").title()
                strategy_testing.append({
                    "strategy": desc,
                    "times_tested": pulls,
                    "performance": _reward_label(avg),
                })
        strategy_testing.sort(key=lambda x: x["performance"], reverse=True)

    # --- Learned writing rules (translated for client) ---
    # Show what the system learned, not internal directives. Strip "DO"/"DON'T"
    # prefixes and technical framing.
    writing_rules = []
    if directives and directives.get("directives"):
        for d in directives["directives"]:
            if d.get("priority") in ("high", "medium"):
                rule = d.get("directive", "")
                # Strip internal prefixes
                for prefix in ("DO ", "DON'T ", "DONT ", "Always ", "Never "):
                    if rule.startswith(prefix):
                        rule = rule[len(prefix):]
                        break
                source = d.get("source", "")
                source_label = {
                    "editorial": "from your feedback",
                    "accepted": "from your approved posts",
                    "engagement": "from engagement data",
                }.get(source, "")
                writing_rules.append({
                    "rule": rule[0].upper() + rule[1:] if rule else "",
                    "source": source_label,
                })

    # --- Analyst findings (latest run only) ---
    findings_strong = []
    findings_moderate = []
    if analyst and analyst.get("findings"):
        _runs = analyst.get("runs", [])
        _all_findings = analyst["findings"]

        if _runs:
            _latest_rid = _runs[-1].get("run_id")  # None for pre-migration runs
            if _latest_rid:
                # New-style run with run_id tag: exact match
                _filtered = [f for f in _all_findings if f.get("run_id") == _latest_rid]
            else:
                # Pre-migration run (no run_id): show findings that also lack run_id
                # (they're from the same era). Fall back to last N if that's empty.
                _filtered = [f for f in _all_findings if f.get("run_id") is None]
                if not _filtered:
                    _filtered = _all_findings[-10:]
        else:
            _filtered = _all_findings[-10:]

        for f in _filtered:
            conf = f.get("confidence", "")
            # Client-friendly: use the claim but strip technical jargon from evidence
            claim = f.get("claim", "")
            # Remove Cohen's d, Spearman, partial correlation references from the claim
            # These are valuable for the operator but confusing for the client
            import re
            clean_claim = re.sub(r"\s*[\(\[—–-]\s*Cohen's d\s*=\s*[^)\]]*[\)\]]?", "", claim)
            clean_claim = re.sub(r"\s*[\(\[—–-]\s*partial correlation[^)\]]*[\)\]]?", "", clean_claim)
            clean_claim = re.sub(r"\s*[\(\[—–-]\s*Spearman[^)\]]*[\)\]]?", "", clean_claim)
            entry = {"claim": clean_claim.strip(), "evidence": f.get("evidence", "")[:200]}
            if conf in ("strong", "high"):
                findings_strong.append(entry)
            elif conf in ("moderate", "medium", "suggestive"):
                findings_moderate.append(entry)

    # --- Recommendations (client-friendly) ---
    recommendations_up = []
    recommendations_down = []
    if strategy_testing:
        for a in strategy_testing:
            if "above average" in a["performance"] or "well above" in a["performance"]:
                recommendations_up.append(a["strategy"])
            elif "below average" in a["performance"] or "well below" in a["performance"]:
                recommendations_down.append(a["strategy"])

    # --- Engagement trend ---
    all_scored = [o for o in observations if o.get("status") == "scored"]
    all_scored.sort(key=lambda o: o.get("posted_at", ""))
    trend = None
    if len(all_scored) >= 5:
        recent_5 = all_scored[-5:]
        prior_5 = all_scored[-10:-5] if len(all_scored) >= 10 else all_scored[:len(all_scored)//2]
        if prior_5:
            recent_avg = sum(o.get("reward", {}).get("immediate", 0) for o in recent_5) / len(recent_5)
            prior_avg = sum(o.get("reward", {}).get("immediate", 0) for o in prior_5) / len(prior_5)
            delta = recent_avg - prior_avg
            trend = {
                "direction": "improving" if delta > 0.1 else "declining" if delta < -0.1 else "stable",
                "recent_avg": round(recent_avg, 2),
                "prior_avg": round(prior_avg, 2),
                "delta": round(delta, 2),
            }

    # --- All-time context ---
    all_time = None
    if all_scored:
        n = len(all_scored)
        rewards = [o.get("reward", {}).get("immediate", 0) for o in all_scored]
        avg_imp = sum((o.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0) for o in all_scored) / n
        avg_react = sum((o.get("reward", {}).get("raw_metrics") or {}).get("reactions", 0) for o in all_scored) / n
        all_time = {
            "total_tracked": len(observations),
            "total_scored": n,
            "avg_impressions": round(avg_imp),
            "avg_reactions": round(avg_react),
            "reward_min": round(min(rewards), 2),
            "reward_max": round(max(rewards), 2),
            "reward_mean": round(sum(rewards) / n, 2),
        }

    # --- System learning status ---
    # Proves the system gets smarter over time. This is the core thesis:
    # "Virio ships posts. We ship a system that gets smarter with every post."
    learning_status = {}

    # Prediction accuracy
    pred_acc = _load_json(vortex.memory_dir(company) / "prediction_accuracy.json")
    if pred_acc and pred_acc.get("n_predictions", 0) > 0:
        learning_status["prediction_accuracy"] = {
            "n_predictions": pred_acc.get("n_predictions", 0),
            "spearman": pred_acc.get("spearman", 0),
            "mean_abs_error": pred_acc.get("mean_abs_error", 0),
            "trend": pred_acc.get("trend", "insufficient_data"),
            "interpretation": pred_acc.get("interpretation", ""),
        }

    # Model readiness (from analyst)
    if analyst:
        _all_findings_lr = analyst.get("findings", [])
        model_entry = None
        for _f in reversed(_all_findings_lr):
            if _f.get("type") == "model":
                model_entry = _f
                break
        if model_entry:
            spec = model_entry.get("model_spec", {})
            learning_status["model"] = {
                "ready": model_entry.get("model_ready", False),
                "loo_r2": spec.get("loo_r2"),
                "features": spec.get("regression_features", []),
                "explanation": model_entry.get("model_readiness_explanation", "")[:300],
            }

        # Learning trajectory across analyst runs
        _runs = analyst.get("runs", [])
        if _runs:
            learning_status["analyst_runs"] = {
                "total_runs": len(_runs),
                "first_run": _runs[0].get("timestamp", "")[:10] if _runs else None,
                "latest_run": _runs[-1].get("timestamp", "")[:10] if _runs else None,
                "latest_tool_calls": _runs[-1].get("tool_calls", 0) if _runs else 0,
                "latest_findings": _runs[-1].get("findings_stored", 0) if _runs else 0,
            }

    # Observation coverage
    if all_scored:
        tagged_count = sum(1 for o in all_scored if o.get("topic_tag"))
        posts_with_predictions = sum(1 for o in all_scored if o.get("predicted_engagement") is not None)
        learning_status["coverage"] = {
            "total_scored": len(all_scored),
            "tagged": tagged_count,
            "with_predictions": posts_with_predictions,
        }

    return {
        "company": company,
        "profile_name": profile_name,
        "weeks": weeks,
        "period_start": (today - timedelta(weeks=weeks)).isoformat(),
        "period_end": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts_published": published,
        "period_totals": {
            "count": n_published,
            "impressions": total_imp,
            "avg_impressions": total_imp // max(n_published, 1),
            "reactions": total_likes,
            "comments": total_comments,
        },
        "upcoming": upcoming,
        "period_comparison": period_comparison,
        "strategy_applied": {
            "topics": topics,
            "formats": formats,
            "window_performance": window_performance,
            "strategy_testing": strategy_testing[:8],
            "writing_rules": writing_rules[:5],
        },
        "strategy_shifts": {
            "findings_strong": findings_strong[:5],
            "findings_moderate": findings_moderate[:3],
            "recommendations_increase": recommendations_up[:3],
            "recommendations_decrease": recommendations_down[:3],
            "trend": trend,
        },
        "learning_status": learning_status,
        "all_time": all_time,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_REPORT_HTML_SYSTEM = """\
You are a designer creating a clean, presentation-ready client progress report for a video call.

Your job is to convert structured report data into a minimal, visually clear HTML page that can
be shared on screen during a client meeting. It must look polished and be easy to absorb.

CONTENT RULES
  Include:
    1. Masthead: client name, "Progress Report", date range, generated date
    2. Posts Published: table or card list with date, title, impressions, likes, comments, engagement rate.
       Period totals below: total posts, total impressions, avg impressions/post, total reactions.
    3. Upcoming Posts: compact list with date, title, status badge (color-coded)
    4. Period Comparison (if available):
       - Compare this period vs prior period: posts published, total impressions,
         impressions change %, reactions, comments. Show as a compact before/after
         with green/red arrows for increases/decreases.
    5. Content Strategy Applied:
       - Window performance summary (plain-English: "above/near/below your average")
       - Strategy testing table (strategy name, times tested, performance label)
       - Writing rules learned from feedback (bullet list with source labels)
    6. Strategy Shifts:
       - High-confidence insights (bullet list with claim — already cleaned of
         statistical jargon, ready for client display)
       - Data-driven recommendations: "Increase" (green) and "Decrease" (red)
       - Engagement trend indicator (improving/stable/declining with numbers)
    6. System Learning Status (NEW — the core value prop):
       - Prediction accuracy: if available, show Spearman correlation between
         our predictions and actual outcomes, with a plain-English interpretation
         ("our model predicted 12 posts and ranked them with 0.25 correlation
         to actual outcomes — the system is learning what works for you")
       - Model readiness: show whether the system can reliably rank draft ideas.
         If not ready, show what's needed ("15 more posts to reach reliable prediction")
       - Observation coverage: how many posts the system has learned from, how diverse
         the content is. Frame as "the more we publish, the smarter the system gets"
       - Learning trajectory: is accuracy improving? "When we started 6 weeks ago,
         predictions were random. Now they rank posts at 0.25 correlation."
       - If no prediction data exists yet, show: "Prediction tracking activates with
         the next batch of posts. Each post teaches the system what works for your audience."
    7. All-Time Context: small stat cards (total posts, avg impressions, avg reactions, reward range)

  Exclude:
    - No raw JSON, no technical jargon about z-scores or bandits for the client
    - Translate rewards into plain language: positive = above average, negative = below average
    - No mentions of "LOLA", "bandit", "z-scored" — use "content strategy testing" or "performance score"

DESIGN RULES
  - Single self-contained HTML file — all CSS in one <style> block, zero external dependencies
  - Font stack: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
  - Max-width 900px, centred, white background, generous padding
  - Body text: 15px / 1.6 line-height; all text #1a1a2a
  - Color palette:
      Primary sections: white cards with subtle border #e5e7eb, border-radius 8px
      Accent emerald: #0d6e4f (positive/improving)
      Accent red: #dc2626 (negative/declining)
      Accent navy: #1a5a8a (neutral/informational)
      Muted text: #6b7280
      Light backgrounds: #f9fafb, #f0faf5 (green tint), #fef2f2 (red tint)
  - Section headers: 18px bold, with a thin colored left border (4px)
  - Tables: clean, no heavy borders, alternating row shading #f9fafb, header in #f3f4f6
  - Status badges: small rounded pills — Scheduled=#dbeafe/#1e40af, ForReview=#fef3c7/#92400e,
    Finalized=#d1fae5/#065f46, InProgress=#e0e7ff/#3730a3, Posted=#d1fae5/#065f46
  - Stat cards: grid of 3-4, subtle shadow (0 1px 3px rgba(0,0,0,0.08))
  - Trend arrows: ↑ green for improving, → grey for stable, ↓ red for declining
  - Separators: 1px solid #f0f0f0, 24px vertical margin
  - @media print: page-break-inside avoid on cards and tables, reduce padding
  - NO JavaScript. NO animations. NO images. NO external fonts.

Output ONLY the complete HTML document — nothing before <!DOCTYPE html>, nothing after </html>.
"""

_REPORT_HTML_USER = """\
Convert the following client progress report data into the presentation HTML described in your instructions.

Client name: {profile_name}
Period: {period_start} to {period_end}
Generated: {generated_at}

---

{report_json}
"""


def _generate_report_html(report_data: dict) -> str | None:
    """Use Claude to convert the report data into presentation HTML."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic not installed, cannot generate HTML report")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        user_msg = _REPORT_HTML_USER.format(
            profile_name=report_data["profile_name"],
            period_start=report_data["period_start"],
            period_end=report_data["period_end"],
            generated_at=report_data["generated_at"],
            report_json=json.dumps(report_data, indent=2, default=str),
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16384,
            system=_REPORT_HTML_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        html = response.content[0].text.strip()
        if "<!DOCTYPE" not in html[:200] and "<html" not in html[:200]:
            logger.warning("Report HTML looks malformed")
        return html
    except Exception as e:
        logger.exception("Failed to generate report HTML: %s", e)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{company}")
async def get_report(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Get the full progress report as JSON."""
    try:
        return _build_report(company, weeks)
    except Exception as e:
        logger.exception("Report generation failed for %s", company)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{company}/html")
async def get_report_html(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Get the progress report as HTML (JSON-wrapped)."""
    report = _build_report(company, weeks)
    html = _generate_report_html(report)
    if not html:
        raise HTTPException(status_code=500, detail="HTML generation failed")
    return {"html": html}


@router.get("/{company}/view", response_class=HTMLResponse)
async def view_report_html(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Serve the progress report HTML directly for browser rendering."""
    report = _build_report(company, weeks)
    html = _generate_report_html(report)
    if not html:
        raise HTTPException(status_code=500, detail="HTML generation failed")
    return HTMLResponse(content=html)
