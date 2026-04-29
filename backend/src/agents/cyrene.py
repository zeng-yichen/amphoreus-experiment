"""Cyrene — strategic growth agent.

The outermost loop in the Amphoreus content pipeline. Every other agent
operates within a single generation run (Stelle drafts posts, Irontomb
simulates audience reactions). Cyrene operates ACROSS runs, across
interviews, across months — studying what happened, identifying what to
do next, and producing a strategic brief that shapes the entire
operation.

## Objective

Maximize the client's ICP exposure and pipeline generation on LinkedIn
over time. Three layers, in order of importance:

  1. Pipeline: engagement from ICP prospects → conversations → deals
  2. ICP exposure: each successive batch attracts more of the right people
  3. Engagement: posts perform well (the base layer that enables 1 and 2)

## Architecture

Turn-based tool-calling agent, same skeleton as Irontomb and Stelle.
Runs on demand (the operator triggers it when they want a strategy
review), produces a JSON strategic brief, self-schedules the next run.

Uses Opus for deep strategic reasoning. Runs infrequently ($5-10 per
run, maybe twice a month per client) so cost is not the bottleneck.

## Output

A JSON strategic brief with: interview questions, asset requests,
content priorities, content avoidance, ABM targets, DM-ready warm
prospects, Stelle scheduling recommendation, ICP exposure assessment,
and a self-schedule trigger for the next Cyrene run.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CYRENE_MODEL = "claude-opus-4-6"
_CYRENE_MAX_TOKENS = 4096
_CYRENE_MAX_TURNS = 40

# Opus 4.6 pricing per million tokens
_INPUT_COST_PER_MTOK = 15.0
_OUTPUT_COST_PER_MTOK = 75.0
_CACHE_READ_COST_PER_MTOK = 1.50
_CACHE_WRITE_COST_PER_MTOK = 18.75

_BRIEF_FILENAME = "cyrene_brief.json"
_BRIEF_HISTORY_FILENAME = "cyrene_brief_history.jsonl"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Fields produced by observation_tagger.py for DASHBOARD DISPLAY only.
# Cyrene must not condition strategic recommendations on these because
# they're a hand-designed taxonomy — a Bitter Lesson trap. She reasons
# from post text + engagement + reactor identity, not from human buckets.
_DISPLAY_ONLY_FIELDS = ("topic_tag", "source_segment_type", "format_tag")


def _strip_display_tags(obs_list: list[dict]) -> list[dict]:
    out = []
    for o in obs_list:
        o2 = {k: v for k, v in o.items() if k not in _DISPLAY_ONLY_FIELDS}
        out.append(o2)
    return out


def _load_cyrene_observations(
    company: str,
    user_id: Optional[str] = None,
) -> list[dict]:
    """Build the observation ledger for a company on-demand from the
    canonical Amphoreus Supabase tables.

    Schema returned: each observation is a dict with
        {
          "provider_urn":      LinkedIn urn (identity)
          "posted_at":         ISO timestamp of the published post
          "post_body":         Stelle's original draft, if this was a
                               Stelle-generated post (local_posts.content
                               JOIN linkedin_posts via matched_provider_urn).
                               None for organic posts the creator wrote
                               independently.
          "posted_body":       What the creator actually published on
                               LinkedIn (linkedin_posts.post_text).
          "status":            "scored" when engagement data is present,
                               "pending" otherwise. (Matches the legacy
                               shape that downstream tools expect.)
          "reward":            {"raw_metrics": {...}, "immediate": z-score}
          "icp_match_rate":    None for now (previous pipeline depended on
                               ordinal_sync scoring; left here as a schema
                               anchor so tools that destructure it don't
                               KeyError. Rebuild live via icp_scorer if
                               ever needed.)
          "matched_provider_urn": same as provider_urn
          "match_method":      "manual_date" | "semantic" | "organic"
        }

    Replaces the old ``ruan_mei_load(company)`` read, which returned
    empty for every client after the ``ruan_mei_state`` SQLite table
    stopped receiving scored-status promotions (ordinal_sync went off
    2026-04-18). The new function pulls from ``linkedin_posts`` (always
    fresh via the mirror + Amphoreus scrape) joined with local_posts
    on ``matched_provider_urn``, so every client with scraped LinkedIn
    content produces a real observation ledger.

    Scoped to the FOC: caller can pass ``user_id`` explicitly (in-
    process invocations like ``run_strategic_review``), or it falls
    back to ``DATABASE_USER_UUID`` env (the MCP-subprocess path). At
    multi-FOC clients (Trimble, Commenda, Hyperspell, Koah) one of
    the two MUST be set or handle resolution fails and the function
    returns empty.
    """
    import os as _os
    import statistics as _stats
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    try:
        from backend.src.db.amphoreus_supabase import _get_client
        from backend.src.agents.stelle import _resolve_linkedin_username
    except Exception as exc:
        logger.warning("[Cyrene] observation rebuild: import failed: %s", exc)
        return []

    sb = _get_client()
    if sb is None:
        return []

    # Explicit kwarg wins; env fallback covers MCP subprocess context
    # where the parent wrote DATABASE_USER_UUID into the subprocess env.
    if not user_id:
        user_id = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
    username = _resolve_linkedin_username(company, user_id=user_id)
    if not username:
        logger.info(
            "[Cyrene] observation rebuild: could not resolve handle for "
            "company=%s user_id=%s — returning empty", company, user_id,
        )
        return []

    # Pull this creator's published history. For prototype clients we
    # know the agency-start date; the lookback narrows to that boundary
    # so pre-Virio content (different voice era, more promotional)
    # doesn't pollute Cyrene's diagnosis. For non-prototype clients,
    # default 180d is a wide-enough window for strategic trend reading.
    # Same agency-start helper Stelle's post_bundle uses, just with a
    # larger default — Cyrene wants months, Stelle wants 3 weeks.
    from backend.src.services.post_bundle import _effective_lookback_days
    lookback_days = _effective_lookback_days(user_id, default_days=180)
    since_iso = (_dt.now(_tz.utc) - _td(days=lookback_days)).isoformat()
    if lookback_days != 180:
        logger.info(
            "[Cyrene] %s: agency-start lookback active — %dd window "
            "(vs 180d default) for observation rebuild",
            username, lookback_days,
        )
    try:
        lp_rows = (
            sb.table("linkedin_posts")
              .select(
                  "provider_urn, post_text, posted_at, total_reactions, "
                  "total_comments, total_reposts, engagement_score, "
                  "is_outlier"
              )
              .eq("creator_username", username)
              .or_("is_company_post.is.null,is_company_post.eq.false")
              .is_("reshared_post_urn", "null")
              .gte("posted_at", since_iso)
              .order("posted_at", desc=True)
              .limit(500)
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.warning(
            "[Cyrene] observation rebuild: linkedin_posts fetch for %s failed: %s",
            username, exc,
        )
        return []

    if not lp_rows:
        return []

    # Pull matched Stelle drafts for this creator via local_posts.
    # Restrict to the FOC's drafts via user_id when available; a
    # multi-FOC company without user_id threads sees every FOC's
    # drafts (matches the legacy ruan_mei_load company-wide behavior).
    local_by_urn: dict[str, dict] = {}
    try:
        lp_q = (
            sb.table("local_posts")
              .select(
                  "id, content, matched_provider_urn, match_method, "
                  "pre_revision_content, scheduled_date, created_at"
              )
              .not_.is_("matched_provider_urn", "null")
        )
        if user_id:
            lp_q = lp_q.eq("user_id", user_id)
        else:
            lp_q = lp_q.eq("company", company)
        matched_drafts = lp_q.execute().data or []
        for d in matched_drafts:
            urn = (d.get("matched_provider_urn") or "").strip()
            if urn:
                local_by_urn[urn] = d
    except Exception as exc:
        logger.debug(
            "[Cyrene] observation rebuild: local_posts join skipped: %s", exc,
        )

    # Compute per-client reaction distribution for the z-scored
    # ``reward.immediate`` composite. Cheap (O(n) over ~60 posts).
    reactions_series = [int(r.get("total_reactions") or 0) for r in lp_rows]
    mean_rx = _stats.mean(reactions_series) if reactions_series else 0.0
    std_rx = _stats.pstdev(reactions_series) if len(reactions_series) > 1 else 0.0

    obs: list[dict] = []
    for r in lp_rows:
        urn = (r.get("provider_urn") or "").strip()
        if not urn:
            continue
        draft = local_by_urn.get(urn)
        rx = int(r.get("total_reactions") or 0)
        z = (rx - mean_rx) / std_rx if std_rx > 0 else 0.0
        raw_metrics = {
            # ``impressions`` unavailable since Ordinal churned; callers
            # that depend on it (analyst filter helpers) treat 0 as
            # "unknown". All other fields are real.
            "impressions": 0,
            "reactions":   rx,
            "comments":    int(r.get("total_comments") or 0),
            "reposts":     int(r.get("total_reposts") or 0),
        }
        obs.append({
            "provider_urn":        urn,
            "ordinal_post_id":     "",  # retained for legacy field compat
            "posted_at":           r.get("posted_at") or "",
            "post_body":           (draft or {}).get("content") or None,
            "posted_body":         r.get("post_text") or "",
            "status":              "scored",
            "reward":              {"raw_metrics": raw_metrics, "immediate": round(z, 3)},
            "icp_match_rate":      None,
            "matched_provider_urn": urn,
            "match_method":        (draft or {}).get("match_method") or "organic",
        })

    logger.info(
        "[Cyrene] rebuilt %d observation(s) for @%s (matched_drafts=%d, "
        "organic=%d)",
        len(obs), username,
        sum(1 for o in obs if o.get("post_body")),
        sum(1 for o in obs if not o.get("post_body")),
    )
    return _strip_display_tags(obs)


def _query_observations(company: str, args: dict) -> str:
    """Reuse the shared observation query tool from the analyst module."""
    try:
        from backend.src.agents.analyst import _tool_query_observations
        # Also forbid filtering BY the tags Cyrene can't see.
        args = {
            k: v for k, v in (args or {}).items()
            if k not in ("topic_filter", "format_filter")
        }
        return _tool_query_observations(args, _load_cyrene_observations(company))
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _query_top_engagers(company: str, args: dict) -> str:
    """Rank LinkedIn reactors by frequency × recency across this creator's
    recent posts.

    Reads directly from the Amphoreus Supabase mirror
    (``linkedin_reactions`` joined with ``linkedin_profiles`` and scoped
    to this creator's posts), replacing the legacy ``get_top_icp_engagers``
    SQLite read which returned empty after ordinal_sync went off. ICP
    ordering is a byproduct: since ``linkedin_profiles`` has headline +
    company for each reactor, the caller (Cyrene herself, via
    ``execute_python`` if she wants a finer cut) can filter/rank
    without us pre-imposing an ICP taxonomy here.

    Returns ``engagers`` sorted by engagement_count desc.
    """
    import os as _os
    from collections import Counter as _Counter
    try:
        from backend.src.db.amphoreus_supabase import _get_client
        from backend.src.agents.stelle import _resolve_linkedin_username
    except Exception as e:
        return json.dumps({"error": f"import failed: {str(e)[:200]}"})

    sb = _get_client()
    if sb is None:
        return json.dumps({"count": 0, "engagers": [], "error": "supabase not configured"})

    limit = min(int(args.get("limit", 30)), 50)
    user_id = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
    username = _resolve_linkedin_username(company, user_id=user_id)
    if not username:
        return json.dumps({"count": 0, "engagers": [], "error": "could not resolve creator handle"})

    try:
        # Pull this creator's post URNs (same scoping as observations).
        post_rows = (
            sb.table("linkedin_posts")
              .select("provider_urn")
              .eq("creator_username", username)
              .or_("is_company_post.is.null,is_company_post.eq.false")
              .is_("reshared_post_urn", "null")
              .limit(500)
              .execute().data
            or []
        )
        urns = [r["provider_urn"] for r in post_rows if r.get("provider_urn")]
        if not urns:
            return json.dumps({"count": 0, "engagers": [], "note": "no posts for creator"})

        # Reactions across those URNs (chunk the IN-list to stay under
        # PostgREST URL length cap).
        reactor_counter: _Counter = _Counter()
        reactor_urn_to_profile: dict[str, dict] = {}
        _CHUNK = 80
        for i in range(0, len(urns), _CHUNK):
            rs = (
                sb.table("linkedin_reactions")
                  .select("provider_profile_urn, reactor_name, reactor_headline")
                  .in_("provider_post_urn", urns[i:i+_CHUNK])
                  .limit(5000)
                  .execute().data
                or []
            )
            for r in rs:
                purn = (r.get("provider_profile_urn") or "").strip()
                if not purn:
                    continue
                reactor_counter[purn] += 1
                reactor_urn_to_profile.setdefault(purn, {
                    "name":     (r.get("reactor_name") or "").strip(),
                    "headline": (r.get("reactor_headline") or "").strip(),
                })

        # Enrich top N with linkedin_profiles (current_company, etc.).
        top_urns = [urn for urn, _ in reactor_counter.most_common(limit * 2)]
        if top_urns:
            for i in range(0, len(top_urns), _CHUNK):
                prof_rows = (
                    sb.table("linkedin_profiles")
                      .select("provider_urn, first_name, last_name, headline, positions_text, location_full")
                      .in_("provider_urn", top_urns[i:i+_CHUNK])
                      .execute().data
                    or []
                )
                for p in prof_rows:
                    u = (p.get("provider_urn") or "").strip()
                    if u in reactor_urn_to_profile:
                        reactor_urn_to_profile[u].update({
                            "headline":  p.get("headline") or reactor_urn_to_profile[u].get("headline", ""),
                            "positions": p.get("positions_text") or "",
                            "location":  p.get("location_full") or "",
                        })

        engagers = []
        for purn, count in reactor_counter.most_common(limit):
            p = reactor_urn_to_profile.get(purn, {})
            engagers.append({
                "provider_profile_urn": purn,
                "name":             p.get("name", ""),
                "headline":         p.get("headline", ""),
                "positions":        p.get("positions", ""),
                "location":         p.get("location", ""),
                "engagement_count": count,
            })

        return json.dumps({"count": len(engagers), "engagers": engagers}, default=str)
    except Exception as e:
        logger.exception("[Cyrene] top_engagers failed")
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"})


# _database_sourced_transcripts was retired 2026-04-24 along with the
# jacquard_direct-based transcript path it fed. Cyrene's
# query_transcript_inventory now uses the canonical
# ``amphoreus_supabase.get_client_transcripts`` helper — same one
# Tribbie + the Mirror Transcripts tab use — which resolves multi-FOC
# scoping correctly and covers both jacquard-sourced + amphoreus-
# uploaded meetings in a single query.


def _query_transcript_inventory(company: str, args: dict) -> str:
    """List transcripts or read one.

    Reads from the Amphoreus Supabase ``meetings`` table via
    ``get_client_transcripts`` — the same helper Tribbie + the Mirror
    Transcripts tab use. Covers both client-interview transcripts
    (Jacquard-sourced via meeting_participants join) AND operator-
    uploaded content-interview transcripts (Amphoreus-native rows).

    2026-04-24 rewrite: previous implementation split on ``is_database_mode``
    and hit ``jacquard_direct`` helpers, which for multi-FOC bare-UUID
    companies returned empty — a Mark/Trimble run would see zero
    transcripts and Cyrene would conclude cold-start. The new path
    uses the canonical Supabase query that knows how to resolve
    slugs/UUIDs + per-FOC scope, matching the rest of the Amphoreus
    pipeline's read patterns.
    """
    read_filename = (args.get("read_filename") or "").strip()

    try:
        from backend.src.db.amphoreus_supabase import get_client_transcripts
    except Exception as exc:
        return json.dumps({"error": f"import failed: {str(exc)[:200]}"})

    transcripts = get_client_transcripts(company, limit=200)

    if read_filename:
        safe_name = Path(read_filename).name
        match = next(
            (t for t in transcripts if t.get("filename") == safe_name),
            None,
        )
        if not match:
            return json.dumps({"error": f"transcript not found: {safe_name}"})
        text = match.get("text") or ""
        return json.dumps({
            "filename":  safe_name,
            "n_chars":   len(text),
            "text":      text[:20000],
            "truncated": len(text) > 20000,
        })

    # List-only view — return filename / size / posted_at without bodies.
    out_rows = []
    for t in transcripts:
        body = t.get("text") or ""
        size_kb = round(len(body.encode("utf-8")) / 1024, 1) if body else 0
        posted_at = str(t.get("posted_at") or "")
        out_rows.append({
            "filename":         t.get("filename") or "(untitled)",
            "size_kb":          size_kb,
            "modified":         posted_at[:19],
            "duration_seconds": t.get("duration_seconds"),
            "meeting_subtype":  t.get("meeting_subtype"),
            "source":           t.get("source"),
        })

    return json.dumps({
        "transcripts":    out_rows,
        "n_transcripts":  len(out_rows),
    }, default=str)


def _query_icp_exposure_trend(company: str, args: dict) -> str:
    """Compute per-week engagement trend from the rebuilt observation
    ledger.

    2026-04-24 rewrite: reads from ``_load_cyrene_observations`` (now
    backed by linkedin_posts, not the dead ruan_mei_state). Each
    observation carries engagement counts; per-post ``icp_match_rate``
    is currently None across the board (the previous pipeline's ICP-
    scoring step depended on ordinal_sync writes), so we aggregate
    what's reliably present — mean reactions/comments per week —
    and leave ``mean_icp_match_rate`` NULL for now. When per-post ICP
    scoring is re-plumbed (via a future icp_scorer pass over reactor
    headlines), this field will populate without further code change.
    """
    obs = [o for o in _load_cyrene_observations(company) if o.get("posted_at")]
    if not obs:
        return json.dumps({"error": "no observations with posted_at"})

    # Parse posted_at into weeks
    weekly: dict[str, list[dict]] = {}
    for o in obs:
        try:
            dt = datetime.fromisoformat(
                o["posted_at"].replace("Z", "+00:00")
            )
            # ISO week key: YYYY-WNN
            week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        except Exception:
            continue
        weekly.setdefault(week_key, []).append(o)

    trend: list[dict] = []
    for week in sorted(weekly):
        posts = weekly[week]
        icp_rates = [
            o["icp_match_rate"] for o in posts
            if isinstance(o.get("icp_match_rate"), (int, float))
        ]
        raw_metrics = [
            (o.get("reward") or {}).get("raw_metrics") or {}
            for o in posts
        ]
        impressions = [m.get("impressions", 0) for m in raw_metrics]
        reactions = [m.get("reactions", 0) for m in raw_metrics]

        trend.append({
            "week": week,
            "n_posts": len(posts),
            "mean_icp_match_rate": round(
                sum(icp_rates) / len(icp_rates), 4
            ) if icp_rates else None,
            "mean_impressions": round(
                sum(impressions) / len(impressions), 1
            ) if impressions else 0,
            "mean_reactions": round(
                sum(reactions) / len(reactions), 1
            ) if reactions else 0,
        })

    return json.dumps({
        "company": company,
        "n_weeks": len(trend),
        "trend": trend,
    }, default=str)


def _query_warm_prospects(company: str, args: dict) -> str:
    """Return reactors who engaged with ``>= min_engagements`` posts.

    Thin wrapper around ``_query_top_engagers`` — same data, filtered
    by engagement_count. ``linkedin_reactions`` + ``linkedin_profiles``
    via the Amphoreus mirror; no fly-local ABM file dependency.
    """
    min_eng = int(args.get("min_engagements", 2))
    inner = _query_top_engagers(company, {"limit": 200})
    try:
        payload = json.loads(inner)
    except Exception:
        return inner
    engagers = payload.get("engagers") or []
    warm = [e for e in engagers if (e.get("engagement_count") or 0) >= min_eng]
    return json.dumps({
        "min_engagements":  min_eng,
        "n_warm_prospects": len(warm),
        "prospects":        warm[:50],
    }, default=str)


def _query_engagement_trajectories(company: str, args: dict) -> str:
    """Return this creator's recent posts ranked by kinetic trajectory.

    Reads directly from the ``linkedin_post_engagement_snapshots``
    time-series table (populated on every scrape pass — both Jacquard
    mirror sync and the Amphoreus Apify scrape write rows here).
    Computes per-post shape metrics on the fly:

      * ``velocity_first_6h`` — reactions/hour in the first 6 hours
        after posting
      * ``peak_velocity``     — highest reactions/hour seen between
        adjacent snapshots
      * ``longevity_ratio``   — (reactions at latest snapshot) /
        (reactions at first snapshot)
      * ``time_to_plateau_hours`` — hours until snapshot-over-snapshot
        velocity drops below 10% of peak

    Excludes posts with fewer than 2 snapshots (can't compute a shape
    from a single point).

    2026-04-24 rewrite: replaced the legacy read from
    ``ruan_mei_load.observations[i].trajectory`` which always returned
    empty after the ruan_mei_state wipe. Goes straight to the snapshot
    table now.
    """
    import os as _os
    from collections import defaultdict as _dd
    sort_by = args.get("sort_by", "velocity_first_6h")
    limit = min(int(args.get("limit", 10)), 20)

    try:
        from backend.src.db.amphoreus_supabase import _get_client
        from backend.src.agents.stelle import _resolve_linkedin_username
    except Exception as e:
        return json.dumps({"error": f"import failed: {str(e)[:200]}"})

    sb = _get_client()
    if sb is None:
        return json.dumps({"error": "supabase not configured"})

    user_id = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
    username = _resolve_linkedin_username(company, user_id=user_id)
    if not username:
        return json.dumps({"error": "could not resolve creator handle"})

    # Fetch creator's recent posts (last 30 days — outside this window
    # trajectory has stabilized and a shape reading is noise).
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    since_iso = (_dt.now(_tz.utc) - _td(days=30)).isoformat()
    try:
        post_rows = (
            sb.table("linkedin_posts")
              .select("provider_urn, post_text, posted_at, total_reactions")
              .eq("creator_username", username)
              .or_("is_company_post.is.null,is_company_post.eq.false")
              .is_("reshared_post_urn", "null")
              .gte("posted_at", since_iso)
              .limit(200).execute().data
            or []
        )
    except Exception as e:
        return json.dumps({"error": f"linkedin_posts query failed: {str(e)[:200]}"})

    if not post_rows:
        return json.dumps({"error": "no recent posts for this creator"})

    urns = [r["provider_urn"] for r in post_rows if r.get("provider_urn")]
    # Pull snapshots for those URNs (chunked IN-list).
    snaps_by_urn: dict[str, list[dict]] = _dd(list)
    _CHUNK = 80
    try:
        for i in range(0, len(urns), _CHUNK):
            rows = (
                sb.table("linkedin_post_engagement_snapshots")
                  .select("provider_urn, scraped_at, total_reactions, total_comments, total_reposts")
                  .in_("provider_urn", urns[i:i+_CHUNK])
                  .order("scraped_at", desc=False)
                  .limit(5000).execute().data
                or []
            )
            for s in rows:
                u = s.get("provider_urn")
                if u:
                    snaps_by_urn[u].append(s)
    except Exception as e:
        return json.dumps({"error": f"snapshots query failed: {str(e)[:200]}"})

    def _parse_iso(s: str):
        try:
            return _dt.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return None

    def _compute_shape(posted_at: str, snaps: list[dict]) -> dict | None:
        if len(snaps) < 2:
            return None
        t0 = _parse_iso(posted_at)
        if not t0:
            return None
        # Sort snapshots, keep only those after posted_at.
        pts = []
        for s in snaps:
            ts = _parse_iso(s.get("scraped_at"))
            if ts is None or ts < t0:
                continue
            pts.append((ts, int(s.get("total_reactions") or 0)))
        pts.sort(key=lambda p: p[0])
        if len(pts) < 2:
            return None

        first_ts, first_rx = pts[0]
        last_ts,  last_rx  = pts[-1]
        # Velocity in first 6h: rx/hour between posted_at and the latest
        # snapshot that's within 6h of posted_at (otherwise use first snap).
        six_h = t0 + _td(hours=6)
        in6 = [p for p in pts if p[0] <= six_h]
        if in6:
            dt6 = (in6[-1][0] - t0).total_seconds() / 3600.0
            velocity_first_6h = round(in6[-1][1] / dt6, 2) if dt6 > 0 else 0.0
        else:
            velocity_first_6h = 0.0

        # Peak velocity across adjacent deltas.
        peak_velocity = 0.0
        for (a_ts, a_rx), (b_ts, b_rx) in zip(pts, pts[1:]):
            dh = (b_ts - a_ts).total_seconds() / 3600.0
            if dh > 0 and (b_rx - a_rx) > 0:
                v = (b_rx - a_rx) / dh
                if v > peak_velocity:
                    peak_velocity = v

        longevity_ratio = (
            round(last_rx / first_rx, 2) if first_rx > 0 else None
        )

        # time_to_plateau: first adjacent window where velocity drops
        # below 10% of peak.
        time_to_plateau = None
        if peak_velocity > 0:
            plateau_threshold = peak_velocity * 0.1
            for (a_ts, a_rx), (b_ts, b_rx) in zip(pts, pts[1:]):
                dh = (b_ts - a_ts).total_seconds() / 3600.0
                v = (b_rx - a_rx) / dh if dh > 0 else 0
                if v < plateau_threshold:
                    time_to_plateau = round((b_ts - t0).total_seconds() / 3600.0, 1)
                    break

        return {
            "velocity_first_6h":    velocity_first_6h,
            "peak_velocity":        round(peak_velocity, 2),
            "longevity_ratio":      longevity_ratio,
            "time_to_plateau_hours": time_to_plateau,
            "n_snapshots":          len(pts),
        }

    results = []
    for r in post_rows:
        urn = r.get("provider_urn")
        traj = _compute_shape(r.get("posted_at") or "", snaps_by_urn.get(urn, []))
        if not traj:
            continue
        body = (r.get("post_text") or "").strip()
        results.append({
            "provider_urn": urn,
            "posted_at":    r.get("posted_at"),
            "hook":         body.split("\n")[0][:120] if body else "",
            "reactions":    int(r.get("total_reactions") or 0),
            "trajectory":   traj,
        })

    if not results:
        return json.dumps({
            "note": "no trajectory data yet (need ≥2 engagement snapshots per post)",
            "hint": "snapshot writes happen on every mirror/scrape pass; "
                    "posts <72h old accumulate fastest",
        })

    def _sort_key(r: dict) -> float:
        val = (r.get("trajectory") or {}).get(sort_by)
        return float(val) if isinstance(val, (int, float)) else 0.0

    ranked = sorted(results, key=_sort_key, reverse=True)[:limit]

    return json.dumps({
        "sorted_by": sort_by,
        "returned":  len(ranked),
        "posts":     ranked,
    }, default=str)


def _execute_python(company: str, args: dict) -> str:
    """Reuse the shared Python execution tool from the analyst module."""
    try:
        from backend.src.agents.analyst import _tool_execute_python
        # Pipe embeddings into the subprocess preamble so Cyrene can
        # operate on raw continuous vectors (cosine sim, PCA, clustering)
        # instead of reaching for hand-engineered feature buckets.
        try:
            from backend.src.utils.post_embeddings import get_post_embeddings
            emb = get_post_embeddings(company)
        except Exception:
            emb = None
        return _tool_execute_python(args, _load_cyrene_observations(company), embeddings=emb)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _search_linkedin_corpus(company: str, args: dict) -> str:
    """Reuse the shared LinkedIn bank search."""
    try:
        from backend.src.agents.analyst import _tool_search_linkedin_bank
        return _tool_search_linkedin_bank(args)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _fetch_url(company: str, args: dict) -> str:
    """Resolve a URL to readable plain text.

    For when a transcript mentions a link (e.g. "Sachil sent this article:
    https://...") and Cyrene needs to know what the article actually says
    rather than hallucinating from URL tokens.
    """
    try:
        from backend.src.utils.fetch_url import fetch_url as _fetch
        url = (args.get("url") or "").strip()
        if not url:
            return json.dumps({"error": "url is required"})
        max_chars = int(args.get("max_chars", 12000))
        result = _fetch(url, max_chars=min(max_chars, 20000))
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"fetch_url failed: {str(e)[:200]}"})


def _web_search(company: str, args: dict) -> str:
    """Web search via Parallel API."""
    query = (args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})
    try:
        import httpx
        api_key = __import__("os").environ.get("PARALLEL_API_KEY", "")
        if not api_key:
            return json.dumps({"error": "PARALLEL_API_KEY not set"})
        resp = httpx.post(
            "https://api.parallel.ai/v1/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": 5},
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), default=str)[:8000]
    except Exception as e:
        return json.dumps({"error": f"web search failed: {str(e)[:200]}"})


# _query_ordinal_posts was removed 2026-04-24. It pulled the client's
# full LinkedIn history via the retired Ordinal API (fetched from
# app.tryordinal.com). With Ordinal churn complete, the same data is
# now in Amphoreus's own ``linkedin_posts`` mirror (populated by the
# LinkedIn scrape legs), and ``_load_cyrene_observations`` already
# rebuilds the full observation ledger from that table — including
# organic posts the creator wrote independently, labeled with
# ``match_method='organic'``. No functionality lost.


def _query_brief_history(company: str, args: dict) -> str:
    """Return the trajectory of all previous Cyrene briefs for this client.

    Each entry includes the timestamp, strategic_themes, topics_to_probe,
    topics_exhausted, prose, and cost. This lets Cyrene see how its own
    strategic direction has evolved over time and correlate with outcomes.
    Legacy briefs (pre-2026-04-22) are surfaced via their old schema
    fields (content_priorities/content_avoid) for backward-compatible
    trend reading.

    Source precedence: Amphoreus Supabase ``cyrene_briefs`` (authoritative);
    fly-local JSONL is only consulted when the Supabase layer returns
    nothing — keeps backfill history readable during the migration window.
    """
    limit = min(int(args.get("limit", 20)), 50)

    # --- primary: Amphoreus Supabase ---------------------------------
    briefs: list[dict] = []
    try:
        from backend.src.db.amphoreus_supabase import list_cyrene_briefs
        briefs = list_cyrene_briefs(company, limit=limit)
    except Exception as exc:
        logger.warning(
            "[Cyrene] list_cyrene_briefs(%s) failed: %s", company, exc
        )
        briefs = []

    # --- fallback: fly-local JSONL -----------------------------------
    if not briefs:
        history_path = P.memory_dir(company) / _BRIEF_HISTORY_FILENAME
        if not history_path.exists():
            return json.dumps({
                "n_briefs": 0,
                "briefs": [],
                "note": "No brief history yet. This is the first Cyrene run for this client.",
            })
        try:
            for line in history_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    briefs.append(json.loads(line))
        except Exception as e:
            return json.dumps({"error": f"failed to read brief history: {str(e)[:200]}"})
        # Local JSONL is oldest-first; slice + reverse to match Supabase ordering.
        briefs = briefs[-limit:][::-1]

    # Slim down to strategically relevant fields. New briefs (2026-04-22
    # onward) carry ``prose`` as the primary strategic payload. Legacy
    # briefs still carry the old rigid-schema fields; we pass through
    # whichever exists so the agent sees history in whatever form each
    # entry was written.
    slim: list[dict] = []
    for b in briefs:
        entry: dict = {
            "computed_at": b.get("_computed_at", ""),
            "cost_usd": b.get("_cost_usd", 0),
            "dm_targets_count": len(b.get("dm_targets", [])),
        }
        # Current-schema fields (2026-04-22 onward): strategic_themes +
        # topics_to_probe + topics_exhausted + prose. Surface whatever
        # the brief actually carries so Cyrene can track how its own
        # direction has drifted across cycles.
        if b.get("strategic_themes") or b.get("topics_to_probe"):
            entry["strategic_themes"] = b.get("strategic_themes", [])
            entry["topics_to_probe"] = b.get("topics_to_probe", [])
            entry["topics_exhausted"] = b.get("topics_exhausted", [])

        prose = (b.get("prose") or "").strip()
        if prose:
            entry["prose"] = prose[:4000]  # cap for trend-history call

        # Legacy-schema fallback — only if the brief has neither the
        # new structured fields nor prose. Pre-2026-04-22 briefs used
        # content_priorities/content_avoid which framed Cyrene as
        # Stelle-instructor; we still surface them for trend continuity.
        if not (b.get("strategic_themes") or b.get("topics_to_probe") or prose):
            entry["content_priorities"] = b.get("content_priorities", [])
            entry["content_avoid"] = b.get("content_avoid", [])
            entry["icp_exposure_assessment"] = b.get("icp_exposure_assessment", "")
            entry["stelle_timing"] = b.get("stelle_timing", "")

        slim.append(entry)

    return json.dumps({
        "n_briefs": len(slim),
        "briefs": slim,
    }, default=str)


def _note(company: str, args: dict) -> str:
    """Working memory — append to per-run notes list."""
    # Notes are managed by the agent loop, not stored persistently.
    # The agent loop tracks them in a list and passes them back as context.
    return json.dumps({"ok": True})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "query_observations",
        "description": (
            "Query this client's full scored post history. Each observation: "
            "stelle_draft (post_body), client-published version (posted_body), "
            "engagement metrics (impressions/reactions/comments/reposts), "
            "icp_match_rate, per-post reactor list with individual icp_scores. "
            "Filter by min_reward, max_reward. Sort by any metric. "
            "Pass summary_only=true for aggregate stats without full texts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {"type": "string", "default": "posted_at"},
                "limit": {"type": "integer", "default": 10},
                "min_reward": {"type": "number"},
                "max_reward": {"type": "number"},
                "summary_only": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "query_top_engagers",
        "description": (
            "Aggregated top engagers across all scored posts, ranked by "
            "ICP fit × engagement frequency. Shows who repeatedly engages "
            "with this client's content and their profile details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 30},
            },
        },
    },
    {
        "name": "query_transcript_inventory",
        "description": (
            "Without arguments: list all interview transcripts (filename, "
            "size, modified date) plus the story inventory showing which "
            "stories have been turned into posts vs are still untapped. "
            "With ``read_filename``: return the raw text of one transcript. "
            "Transcripts are the SINGLE source of truth for anything the "
            "client has said — mine them for both (a) untapped story "
            "material and (b) offline pipeline signals the client "
            "mentioned in passing (DMs from ICP, meetings booked, deals "
            "sourced). Those signals will not appear anywhere else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "read_filename": {
                    "type": "string",
                    "description": "Name of a specific transcript to read in full.",
                },
            },
        },
    },
    {
        "name": "query_icp_exposure_trend",
        "description": (
            "ICP match rate averaged per week over the client's history. "
            "Shows whether successive batches of posts are attracting more "
            "of the right people. THE STRATEGIC KPI — if this is flat or "
            "declining, the content strategy needs adjustment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "query_warm_prospects",
        "description": (
            "Reactor pool cross-referenced against ABM targets. Returns "
            "people who have engaged with multiple posts, their ICP score, "
            "which posts they engaged with, and whether they're from a named "
            "ABM target account. These are DM-ready warm leads — the client "
            "should reach out to them directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_engagements": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum posts engaged with to qualify as warm.",
                },
            },
        },
    },
    {
        "name": "query_engagement_trajectories",
        "description": (
            "Posts ranked by trajectory metrics: velocity_first_6h (how "
            "fast engagement grew initially), longevity_ratio (how much "
            "engagement continued past 24h), peak_velocity_imp_per_h, "
            "time_to_plateau_hours. Shows which content shapes have legs "
            "vs which peak and die. Only available for posts with enough "
            "engagement snapshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "default": "velocity_first_6h",
                    "description": "Trajectory metric to rank by.",
                },
                "limit": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "execute_python",
        "description": (
            "Run arbitrary Python code with scored observations and raw "
            "1536-dim OpenAI embeddings pre-loaded, plus numpy/scipy/"
            "sklearn/pandas. Pre-loaded globals:\n"
            "  • obs — list of scored observation dicts\n"
            "  • embeddings — {post_hash: [1536 floats]}\n"
            "  • emb_matrix — np.array shape (N, 1536), rows aligned to emb_hashes\n"
            "  • emb_hashes — list[str] of post_hash keys\n"
            "  • emb_by_obs — embeddings aligned to obs order (None for missing)\n"
            "Use the vectors directly for cosine similarity, PCA, "
            "clustering, nearest-neighbor lookups — anything continuous. "
            "Print results to stdout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "search_linkedin_corpus",
        "description": (
            "Search the 200K+ LinkedIn post corpus by keyword or semantic "
            "similarity. Use for competitive intelligence — what's working "
            "in adjacent niches that this client could adapt?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic"],
                    "default": "keyword",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for industry news, regulatory updates, "
            "competitive moves — anything timely that could become a hook "
            "for the next batch of posts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Resolve a specific URL to readable plain text. Use when a "
            "transcript mentions a link the client shared (article, report, "
            "tweet) and you need to know what it actually says rather than "
            "inferring from the URL alone. Returns title + body text, "
            "nav/script/footer stripped. Fails gracefully on paywalls or "
            "4xx/5xx with a clean error message — don't fabricate content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "default": 12000,
                    "description": "Maximum characters to return (capped at 20000).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "query_brief_history",
        "description": (
            "Return the trajectory of all previous Cyrene briefs for this "
            "client. Shows how your strategic_themes, topics_to_probe, and "
            "topics_exhausted have evolved over time. Use this to see what "
            "direction you've steered before and correlate with outcomes "
            "from query_observations and query_icp_exposure_trend. Were "
            "the themes you pushed last cycle borne out by the posts and "
            "engagement that followed? Legacy briefs (pre-2026-04-22) "
            "surface via content_priorities/content_avoid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max briefs to return (default 20, max 50).",
                },
            },
        },
    },
    {
        "name": "note",
        "description": "Record an observation to your working memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "submit_brief",
        "description": (
            "Terminal tool. Submit the strategic brief for this FOC.\n\n"
            "The brief is a strategic artifact consumed by **two** readers:\n"
            "  (a) the operator — for long-term planning of this client's\n"
            "      LinkedIn journey (arc over weeks/months, not next post).\n"
            "  (b) Tribbie (interview agent) — for the next client call,\n"
            "      where she'll probe specific threads the brief flags.\n"
            "\n"
            "Stelle does NOT consume this brief. She extracts content from\n"
            "raw transcripts at generation time. Don't write this as\n"
            "instructions to Stelle.\n"
            "\n"
            "Six structured fields + a prose narrative:\n"
            "  - ``current_strategy_diagnosis``: THE SPINE. Three buckets —\n"
            "    what_is_working, what_is_broken, blind_spots — each\n"
            "    cited with concrete evidence. This is the answer to the\n"
            "    core question. Themes / probes below are downstream.\n"
            "  - ``strategic_themes``: the 3-5 directions this client's\n"
            "    public voice should develop over the next 4-8 weeks. Each\n"
            "    cites which diagnosis bucket it addresses (`addresses`\n"
            "    field) and is tied to engagement / ICP-exposure evidence.\n"
            "  - ``topics_to_probe``: threads Tribbie should pull on\n"
            "    in the next client interview to advance the strategy\n"
            "    diagnosed above. Forward-looking: 'the diagnosis says\n"
            "    X is broken/missing — we need the client to articulate\n"
            "    Y on the next call to fix it.' Each entry cites which\n"
            "    diagnosis bucket it addresses (``addresses`` field).\n"
            "    NOT backward-looking follow-ups on already-discussed\n"
            "    transcript material (those recycle content instead of\n"
            "    advancing the strategy). 5-12 entries.\n"
            "  - ``topics_exhausted``: patterns / angles the client has\n"
            "    already posted to diminishing returns. Cite which posts.\n"
            "    0-5 entries. Do not invent.\n"
            "  - ``dm_targets``: warm prospects worth a direct outreach now\n"
            "    (3-8 strongest).\n"
            "  - ``next_run_trigger``: when Cyrene should run again, why.\n"
            "  - ``prose``: strategic narrative. Where is this client's\n"
            "    journey going? What's shifting in the ICP reaction? What's\n"
            "    the arc over the past month? 400-1200 words.\n"
            "\n"
            "Hard rule: every ``strategic_themes``, ``topics_exhausted``,\n"
            "and ``topics_to_probe`` entry cites concrete evidence from\n"
            "your tool calls — a specific observation, engager, trend\n"
            "number, or transcript quote. No invented priorities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_strategy_diagnosis": {
                    "type": "object",
                    "description": (
                        "THE SPINE OF THE BRIEF. Explicit articulation of "
                        "what's working, what's broken, and what's missing "
                        "in this client's current LinkedIn strategy. The "
                        "answer to the core question. Themes / probes / "
                        "exhausted topics below are downstream of this. "
                        "Every entry cites concrete evidence (observation, "
                        "engager, trend number, transcript quote). For "
                        "first-run-for-this-client cases (no prior "
                        "strategy yet), what_is_broken can be empty; "
                        "what_is_working and blind_spots should still "
                        "be populated from voice / audience / ICP signals."
                    ),
                    "properties": {
                        "what_is_working": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "pattern":  {
                                        "type": "string",
                                        "description": "The pattern / angle / structure / cadence that's landing for this client.",
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "description": "Concrete data: post IDs, engagement numbers, ICP-scored reactor counts, comment quotes. NOT vibes.",
                                    },
                                },
                                "required": ["pattern", "evidence"],
                            },
                            "description": "1-3 patterns that are working. Concrete, evidence-grounded.",
                        },
                        "what_is_broken": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "pattern": {
                                        "type": "string",
                                        "description": "The pattern / approach / habit that's underperforming.",
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "description": "Concrete data showing the failure: post IDs, declining trend numbers, off-ICP engagement, transcript moments the strategy isn't reflecting.",
                                    },
                                    "consequence": {
                                        "type": "string",
                                        "description": "What this is costing — missed pipeline, off-ICP engagement, audience fatigue, voice drift, etc.",
                                    },
                                },
                                "required": ["pattern", "evidence"],
                            },
                            "description": "0-3 patterns that are broken. Empty only on first-run-for-this-client when there's no prior strategy yet.",
                        },
                        "blind_spots": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "observation": {
                                        "type": "string",
                                        "description": "What the strategy is missing entirely — a topic, an audience segment, a structural element, a story area the transcripts surface but no post has touched.",
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "description": "What in the data points to this gap — comments asking about X, ICP segment with high engagement on competitor posts, transcript quotes never converted into posts, etc.",
                                    },
                                },
                                "required": ["observation", "evidence"],
                            },
                            "description": "0-2 things the current strategy is missing entirely. Forward-looking gaps the diagnosis surfaces.",
                        },
                    },
                    "required": ["what_is_working", "what_is_broken", "blind_spots"],
                },
                "strategic_themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "theme": {
                                "type": "string",
                                "description": "Short label — 4-10 words for the direction (e.g. 'from product-features to industry-scarring anecdotes').",
                            },
                            "addresses": {
                                "type": "string",
                                "description": (
                                    "Which diagnosis bucket this theme addresses. "
                                    "Format: 'what_is_broken[0]' / 'what_is_working[1]' / "
                                    "'blind_spots[0]'. Makes the critique → fix link "
                                    "explicit so the operator can read the brief as "
                                    "diagnosis → prescription, not as freestanding "
                                    "creative re-direction."
                                ),
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Why this theme now: which observations / engagers / trend numbers support steering toward it. Cite concretely.",
                            },
                            "arc": {
                                "type": "string",
                                "description": "How this theme should develop over the next 4-8 weeks. Where does it start, where does it go?",
                            },
                        },
                        "required": ["theme", "addresses", "evidence"],
                    },
                    "description": (
                        "The 3-5 directions this FOC's public voice should "
                        "develop toward over the coming weeks. Each MUST "
                        "cite which diagnosis bucket it addresses. Not "
                        "tactics, not next-post instructions — strategic "
                        "direction. Each grounded in real data, not vibes."
                    ),
                },
                "topics_to_probe": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "thread": {
                                "type": "string",
                                "description": (
                                    "The topic / angle / story area Tribbie "
                                    "should pull on in the next interview. "
                                    "Forward-looking: what does this client "
                                    "need to talk about NEXT to advance the "
                                    "strategy diagnosed above?"
                                ),
                            },
                            "addresses": {
                                "type": "string",
                                "description": (
                                    "Which diagnosis bucket this probe "
                                    "advances. Format: "
                                    "``what_is_broken[N]`` / "
                                    "``blind_spots[N]`` / "
                                    "``what_is_working[N]`` (where N is "
                                    "the index in that bucket's array). "
                                    "Each probe MUST be tied to a specific "
                                    "diagnosis entry — if you can't say "
                                    "which bucket it advances, it doesn't "
                                    "belong here. Drill-deeper-on-already-"
                                    "discussed-detail probes are only "
                                    "valid when the diagnosis explicitly "
                                    "flags that drill as the strategic "
                                    "move (e.g., ``what_is_broken`` says "
                                    "the client only mentioned topic X in "
                                    "passing and needs to develop it)."
                                ),
                            },
                            "why": {
                                "type": "string",
                                "description": (
                                    "Forward-looking rationale: what's "
                                    "MISSING from the current content arc "
                                    "that this probe would unlock? Cite "
                                    "the specific gap (engagement signal, "
                                    "comment thread, transcript silence, "
                                    "ICP-segment exposure miss). NOT 'in "
                                    "transcript X they mentioned Y, let's "
                                    "ask more about Y' — that's "
                                    "backward-looking. The right shape is "
                                    "'the diagnosis says Z is broken, and "
                                    "we need this client to articulate W "
                                    "to fix it.'"
                                ),
                            },
                            "suggested_entry_point": {
                                "type": "string",
                                "description": "Optional — a specific opening question or reference point Tribbie could use to get the client into the thread.",
                            },
                        },
                        "required": ["thread", "addresses", "why"],
                    },
                    "description": (
                        "Threads Tribbie should pull on in the next client "
                        "interview to advance the strategy diagnosed above. "
                        "5-12 entries.\n\n"
                        "FORWARD-LOOKING, NOT FOLLOW-UP. The default failure "
                        "mode is 'in transcript X the client mentioned Y, "
                        "let's drill deeper into Y.' That produces probes "
                        "that recycle existing material instead of "
                        "developing the strategy.\n\n"
                        "Right shape: 'The diagnosis says Z is broken / "
                        "missing / under-developed. To fix Z, we need this "
                        "client to articulate W on the next call.' The "
                        "``addresses`` field forces this — every probe "
                        "must point at a specific diagnosis bucket entry. "
                        "If the probe doesn't advance the diagnosis, it "
                        "doesn't belong in the brief.\n\n"
                        "Drill-deeper probes ARE valid when the diagnosis "
                        "itself names the drill as the strategic move "
                        "(e.g., what_is_broken says 'client only "
                        "name-dropped topic X once, never developed the "
                        "actual point' — then a probe to develop X is "
                        "advancing the diagnosis, not avoiding it). "
                        "Default mode is forward-looking; drill-deeper "
                        "is the named exception."
                    ),
                },
                "topics_exhausted": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The topic / angle / framing that's been worked over.",
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Which posts show diminishing returns. Cite post IDs or dates + engagement.",
                            },
                        },
                        "required": ["pattern", "evidence"],
                    },
                    "description": (
                        "Topics the client has already posted to fatigue. "
                        "Operator uses this to steer interviews away from "
                        "rehashes. 0-5 entries — leave empty unless you "
                        "have specific post evidence."
                    ),
                },
                "dm_targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":            {"type": "string"},
                            "headline":        {"type": "string"},
                            "company":         {"type": "string"},
                            "icp_score":       {"type": "number"},
                            "posts_engaged":   {"type": "integer"},
                            "suggested_angle": {"type": "string"},
                        },
                    },
                    "description": (
                        "Warm prospects the client should DM on LinkedIn "
                        "now. 3-8 strongest. Include which posts they "
                        "reacted to and a suggested opener."
                    ),
                },
                "next_run_trigger": {
                    "type": "object",
                    "properties": {
                        "condition":     {"type": "string"},
                        "or_after_days": {"type": "integer"},
                        "rationale":     {"type": "string"},
                    },
                    "description": "When Cyrene should run again.",
                },
                "prose": {
                    "type": "string",
                    "description": (
                        "Strategic narrative, 400-1200 words. Where is "
                        "this client's LinkedIn journey going? What's "
                        "shifting in who engages? What's the arc over "
                        "the past month? Cite specific posts / engagers "
                        "/ numbers. This is context + reasoning — the "
                        "structured fields above are the actionable output."
                    ),
                },
            },
            "required": [
                "current_strategy_diagnosis",
                "strategic_themes",
                "topics_to_probe",
                "next_run_trigger",
                "prose",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Cyrene, the strategic growth agent for a LinkedIn ghostwriting \
operation. You operate ABOVE the post generation pipeline — your job is \
to make every successive cycle of content creation more effective than \
the last by studying what happened, identifying what to do next, and \
producing a strategic brief that informs the entire operation.

## Core question

Every Cyrene run answers exactly one question:

**What is wrong with this client's current content strategy, and what \
specifically should change?**

Every tool you call, every observation you read, every transcript you \
query, every brief you cite from history — they all serve answering \
this question. Your output is the answer. The brief's structured \
fields are the *shape* of the answer; this question is its *spine*.

If you find yourself proposing themes without first articulating \
what's broken, you've drifted. Re-anchor: diagnose, then prescribe.

Hard rule: every claim about what's wrong, and every claim about \
what should change, must trace to a specific observation, a specific \
engager, a specific trend number, or a specific transcript quote. If \
you can't ground a critique in concrete evidence, leave it out — \
don't fabricate to fill a field.

First-run-for-this-client edge case: when no current strategy exists \
yet (this is the first Cyrene run for this client), the question \
reframes to *"given what the data shows about voice, audience, and \
ICP, what should the starting strategy be — and what should we \
explicitly NOT do?"* The "should not" half preserves the diagnostic \
posture even on day zero. ``what_is_broken`` may be empty in this \
case; ``what_is_working`` and ``blind_spots`` should not be.

## Your objective

Maximize this client's ICP exposure and pipeline generation on LinkedIn \
over time. Not post by post — across the full arc of their presence. \
Three layers, in order of importance:

  1. Pipeline: engagement from ICP prospects should convert to \
     conversations and deals. This is what the client pays for.
  2. ICP exposure: the RIGHT people should be engaging, and each batch \
     should attract MORE of them than the last.
  3. Engagement: posts should perform well. Base layer enabling 1 and 2.

## How the content pipeline works (what you are optimizing)

The operator (a ghostwriter at a content agency) runs a repeating cycle:

  1. INTERVIEW: the operator interviews the client on a video call, \
     asking questions designed to surface personal stories worth posting.
  2. TRANSCRIPTS: the interview is transcribed and stored in the \
     Amphoreus Supabase meetings table; use query_transcript_inventory.
  3. STELLE GENERATES: Stelle (the post-writing agent) reads the \
     transcripts, mines them for angles, and writes a batch of LinkedIn \
     posts. Stelle has access to:
       - Raw transcripts as source material
       - Client voice examples (top past posts by engagement)
       - Published post history (for topic dedup)
       - Scored observation history via query_observations (draft vs \
         published diffs, engagement metrics, reactor identities)
       - 200K LinkedIn post corpus for reference
       - Irontomb (audience simulator) runs post-hoc on each final \
         post, producing engagement predictions grounded in client + \
         cross-client data. These predictions are saved to \
         `irontomb_posthoc_latest.json` in client memory. \
     Stelle writes authentically from specific transcript moments \
     without mid-loop scoring pressure.
  4. PUBLISH: posts are pushed to Ordinal, scheduled, go live on LinkedIn.
  5. ENGAGEMENT: real engagement data collected via Ordinal analytics \
     every hour (every 15 min for posts <72h old). Reactor identities \
     scraped via Apify. Per-reactor ICP scores computed. Engagement \
     trajectories (velocity, plateau, longevity) tracked.
  6. LEARNING: RuanMei scores each post. Irontomb's post-hoc \
     predictions are compared against real T+7d outcomes. The \
     prediction-vs-reality delta is your gradient signal — which \
     kinds of posts does Irontomb systematically misjudge? That \
     tells you something about what conventional engagement \
     wisdom gets wrong for this specific client.
  7. REPEAT from step 1.

You sit between step 6 and step 1. After the system learns from the \
latest batch, you study everything and produce a brief that shapes the \
NEXT cycle.

## Your brief history

Every brief you've ever produced is saved. Use `query_brief_history` \
to see the trajectory of your own recommendations over time. Compare \
what you recommended in previous cycles against what actually happened \
(via `query_observations` and `query_icp_exposure_trend`). This is \
your gradient signal: which of your past recommendations led to posts \
that outperformed? Which led to underperformers? Which recommendations \
did you repeat cycle after cycle without improvement? Adjust accordingly.

## Client context

{client_context}

This client has {n_scored} scored observations in their history.

## What the data actually measures

`icp_match_rate` is computed from people who REACTED to a post. \
Anything about readers who didn't react — a scroll-past, a \
memorized-for-later, an offline DM a week later — is not in that \
number. Offline pipeline outcomes (DMs, calls, deals) show up only \
when the client mentions them on interview calls, which are \
transcribed into `transcripts/`. `query_transcript_inventory` with \
`read_filename` returns a transcript's full text.

## Tools to answer the core question

Study the data across multiple tools before forming the answer. A \
good Cyrene run takes 15-30 turns. You are answering the core \
question from evidence, not generating a quick summary.

No hand-engineered strategy frameworks. No "post 3x/week." No \
"rotate between TOFU/MOFU/BOFU." No off-the-shelf failure-mode \
taxonomies. Read the data, see what's working, see what's broken, \
see what's missing, form your own view.

## Structured shape of the answer

Two downstream consumers read your brief: the **operator** (long-term \
planning of this client's LinkedIn journey) and **Tribbie** (next \
interview prep). **Stelle does NOT consume it** — she extracts \
content from raw transcripts at generation time. Don't write the \
brief as instructions to Stelle.

`submit_brief` takes structured fields documented in the tool schema. \
The four most important:

  - ``current_strategy_diagnosis`` — the answer to the core question, \
    in three buckets: ``what_is_working``, ``what_is_broken``, \
    ``blind_spots``. Every entry cites concrete evidence. **This is \
    the spine of the brief.** Themes / probes / exhausted topics \
    below are downstream of it.
  - ``strategic_themes`` (3-5) — directions to develop the public \
    voice. Each theme cites which diagnosis bucket it addresses \
    (`addresses` field: e.g. "what_is_broken[0]"). Critique → fix \
    link is explicit. Not tactics, not next-post instructions — \
    strategic direction over 4-8 weeks.
  - ``topics_to_probe`` (5-12) — threads for Tribbie's next \
    interview, derived FROM THE DIAGNOSIS YOU JUST WROTE. Forward-\
    looking: "the diagnosis says X is broken/missing — we need the \
    client to articulate Y on the next call to fix it." Each probe \
    cites which diagnosis bucket entry it advances (``addresses`` \
    field, same shape as ``strategic_themes``). \
    \
    Standard failure mode: backward-looking probes that drill into \
    transcript material already covered ("in interview 3 they \
    mentioned topic Z, let's ask more about Z"). That recycles \
    existing content instead of moving the strategy. Drill-deeper \
    probes are valid ONLY when the diagnosis itself names the drill \
    as the strategic move — otherwise default to forward-looking \
    new-territory probes that fill diagnosis gaps.
  - ``topics_exhausted`` (0-5), ``dm_targets`` (3-8), \
    ``next_run_trigger``, ``prose`` (400-1200 words) — supporting \
    structure. ``prose`` ties the diagnosis to the prescription \
    in narrative form.

When ready, call submit_brief.

## Previous brief

{previous_brief}
"""


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

# (Removed _query_irontomb_predictions — it read from a posthoc file
# Irontomb never actually writes, so every call returned "not available".
# Tracked as future work once Irontomb's post-hoc evaluation loop lands.)


_TOOL_DISPATCH: dict[str, Any] = {
    "query_observations": _query_observations,
    "query_top_engagers": _query_top_engagers,
    "query_transcript_inventory": _query_transcript_inventory,
    "query_icp_exposure_trend": _query_icp_exposure_trend,
    "query_warm_prospects": _query_warm_prospects,
    "query_engagement_trajectories": _query_engagement_trajectories,
    "execute_python": _execute_python,
    "search_linkedin_corpus": _search_linkedin_corpus,
    "web_search": _web_search,
    "fetch_url": _fetch_url,
    "query_brief_history": _query_brief_history,
    "note": _note,
}


def _dispatch_tool(company: str, name: str, args: dict, notes: list[str]) -> str:
    """Route a tool call to its implementation."""
    if name == "note":
        text = (args.get("text") or "").strip()
        if text:
            notes.append(text)
        return json.dumps({"ok": True, "note_count": len(notes)})

    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        return handler(company, args)
    except Exception as e:
        logger.exception("[Cyrene] tool %s failed", name)
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_strategic_review(
    company: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run a full Cyrene strategic review for one client.

    Turn-based agent loop. Returns the strategic brief dict on success,
    or a dict with `_error` on failure. Persists the brief to
    memory/{company}/cyrene_brief.json.

    ``user_id`` (optional) scopes the brief to a specific FOC user at a
    multi-FOC company. When provided:
      * The prompt is prefixed with that user's name / role context.
      * The saved brief is keyed with ``user_id`` so subsequent
        ``get_latest_cyrene_brief(company, user_id=...)`` reads it
        back for that specific user.
      * Previous-brief lookup preferentially reads the per-user brief.
    When omitted the behavior is unchanged (company-wide brief) —
    correct for single-FOC clients like Hume/Innovo/Flora.
    """
    # CLI short-circuit: route the entire run through Claude CLI when the
    # feature flag is on. No API spend. Hard-fail on CLI error — no silent
    # fallback to API.
    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli
    if _use_cli():
        logger.info("[Cyrene] CLI mode enabled — delegating to run_cyrene_cli()")
        from backend.src.mcp_bridge.claude_cli import run_cyrene_cli
        return run_cyrene_cli(company, user_id=user_id)

    # Load audience context from transcripts
    from backend.src.agents.irontomb import _load_icp_context
    client_context = _load_icp_context(company)

    # Count scored observations — uses the rebuilt ledger (pulls from
    # linkedin_posts + matched local_posts). Replaces the old
    # ruan_mei_load read which always returned 0 after the
    # ruan_mei_state wipe. 2026-04-24.
    try:
        n_scored = len(_load_cyrene_observations(company, user_id=user_id))
    except Exception:
        n_scored = 0

    # Load previous brief so Cyrene knows what she already recommended
    # and can build on it rather than repeat herself. Primary source is
    # the Amphoreus Supabase (cyrene_briefs, newest row); fall back to
    # the fly-local file if Supabase has nothing / is unreachable.
    previous_brief = "No previous brief exists. This is the first Cyrene run for this client."
    prev_data: Optional[dict] = None
    try:
        from backend.src.db.amphoreus_supabase import get_latest_cyrene_brief
        # strict_user_only=True when user_id is set: a multi-FOC client
        # must not bootstrap one FOC's review with another FOC's prior
        # brief as "previous context" — that contaminates voice +
        # strategic direction across roles. If no prior brief for THIS
        # FOC exists, prev_data stays None and Cyrene runs from
        # scratch (correct behavior for first-run-per-FOC).
        prev_data = get_latest_cyrene_brief(
            company, user_id=user_id, strict_user_only=bool(user_id),
        )
    except Exception as exc:
        logger.warning(
            "[Cyrene] get_latest_cyrene_brief(%s) failed: %s", company, exc
        )
    if not prev_data:
        try:
            prev_path = P.memory_dir(company) / _BRIEF_FILENAME
            if prev_path.exists():
                prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            prev_data = None
    if prev_data:
        # Pass the full previous brief through — the model decides what's
        # relevant. No lossy summary.
        previous_brief = json.dumps(prev_data, indent=2, ensure_ascii=False, default=str)

    system_prompt = _SYSTEM_PROMPT.format(
        client_context=client_context,
        n_scored=n_scored,
        company=company,
        previous_brief=previous_brief,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Run a strategic review for {company}. Study the data "
                f"across your tools, form your strategy from evidence, "
                f"and produce a comprehensive brief via submit_brief."
            ),
        }
    ]

    client = anthropic.Anthropic()
    notes: list[str] = []
    total_cost = 0.0
    turns_used = 0
    brief: Optional[dict] = None

    for turn in range(1, _CYRENE_MAX_TURNS + 1):
        turns_used = turn
        try:
            resp = client.messages.create(
                model=_CYRENE_MODEL,
                max_tokens=_CYRENE_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=_TOOLS,
                messages=messages,
            )
        except Exception as e:
            logger.warning("[Cyrene] API call failed turn=%d: %s", turn, e)
            return {
                "_error": f"API call failed on turn {turn}: {str(e)[:200]}",
                "_turns_used": turns_used,
            }

        # Cost tracking
        try:
            usage = resp.usage
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
            total_cost += (
                (in_tok / 1e6) * _INPUT_COST_PER_MTOK
                + (out_tok / 1e6) * _OUTPUT_COST_PER_MTOK
                + (cache_r / 1e6) * _CACHE_READ_COST_PER_MTOK
                + (cache_w / 1e6) * _CACHE_WRITE_COST_PER_MTOK
            )
        except Exception:
            pass

        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            logger.warning("[Cyrene] %s: no tool call on turn %d", company, turn)
            break

        tool_results: list[dict] = []
        for tu in tool_uses:
            if tu.name == "submit_brief":
                if isinstance(tu.input, dict):
                    brief = dict(tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "Brief submitted. Review complete.",
                })
            else:
                result = _dispatch_tool(company, tu.name, tu.input or {}, notes)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result[:12000],
                })

        messages.append({"role": "user", "content": tool_results})

        if brief is not None:
            break

        if resp.stop_reason == "end_turn":
            break

    if brief is None:
        return {
            "_error": f"no submit_brief within {_CYRENE_MAX_TURNS} turns",
            "_turns_used": turns_used,
            "_cost_usd": round(total_cost, 2),
            "_notes": notes,
        }

    # Stamp metadata
    brief["_company"] = company
    brief["_computed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    brief["_turns_used"] = turns_used
    brief["_cost_usd"] = round(total_cost, 2)
    brief["_notes"] = notes

    # Persist the brief.
    #
    # Primary store: Amphoreus Supabase (cyrene_briefs). Cyrene-authored
    # artifact, Amphoreus-owned, read by Tribbie to bootstrap interviews.
    # The history table is the same rows — no separate JSONL needed since
    # every insert retains its timestamp and full payload.
    #
    # Fallback: fly-local memory dir (legacy single-tenant path). We
    # still write there when Supabase persistence fails so the brief
    # isn't lost if PostgREST is transiently down — but the local copy
    # is NOT the source of truth once Supabase is back.
    try:
        from backend.src.db.amphoreus_supabase import save_cyrene_brief
        saved = save_cyrene_brief(
            company=company,
            brief=brief,
            created_by=None,  # TODO: thread operator email through the caller
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning("[Cyrene] Amphoreus-Supabase save failed for %s: %s", company, exc)
        saved = None

    if saved:
        logger.info(
            "[Cyrene] %s: brief saved to Amphoreus Supabase (id=%s, %d turns, $%.2f).",
            company, saved.get("id"), turns_used, total_cost,
        )
    else:
        # Supabase layer returned None — write a fly-local fallback so
        # we don't lose the brief. When Supabase is healthy again, a
        # future Cyrene run will overwrite the primary store.
        try:
            mem_dir = P.memory_dir(company)
            mem_dir.mkdir(parents=True, exist_ok=True)
            brief_path = mem_dir / _BRIEF_FILENAME
            tmp = brief_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(brief, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.rename(brief_path)
            history_path = mem_dir / _BRIEF_HISTORY_FILENAME
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(brief, ensure_ascii=False, default=str) + "\n")
            logger.warning(
                "[Cyrene] %s: Amphoreus Supabase unavailable — brief saved to "
                "fly-local fallback at %s (re-run Cyrene when Supabase is "
                "back to replace primary).",
                company, brief_path,
            )
        except Exception as exc:
            logger.error("[Cyrene] %s: brief save failed everywhere: %s", company, exc)

    return brief


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 cyrene.py <company>")
        sys.exit(1)

    company_arg = sys.argv[1]
    result = run_strategic_review(company_arg)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
