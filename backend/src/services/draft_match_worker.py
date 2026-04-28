"""Semantic match-back: pair Jacquard-synced LinkedIn posts with Stelle drafts.

Context
-------

Pre-Ordinal-retirement, every Stelle draft got an ``ordinal_post_id``
assigned at push time and the (draft, published) pair was resolved by
exact-id lookup in ``memory/<company>/draft_map.json``. Ordinal is being
retired, so that exact-id chain is breaking. Going forward, operators
will take Stelle drafts, possibly edit them, and post directly to
LinkedIn — no ordinal id to match on.

This worker bridges the gap. It runs inside the 8-hour
``jacquard_mirror_cron`` (right after the mirror sync finishes, so
``linkedin_posts`` is fresh). For each tracked creator, it:

  1. Pulls recent ``linkedin_posts`` rows from the Amphoreus mirror that
     aren't already paired with a Stelle draft.
  2. Pulls unmatched ``local_posts`` rows for the corresponding company,
     within a 14-day window before the LinkedIn post's ``posted_at``.
  3. Embeds the LinkedIn post body with OpenAI text-embedding-3-small.
  4. Cosine-similarity-scores against each candidate draft's stored
     ``embedding`` column.
  5. On a confident match (top-1 ≥ 0.82 AND margin to 2nd ≥ 0.04),
     updates ``local_posts`` with ``matched_provider_urn``, ``matched_at``,
     ``match_similarity``, ``match_method='semantic'``.

Thresholds rationale:
  * 0.82 is strict enough that incidental topical overlap (shared jargon,
    client voice) won't trigger a match, but permissive enough that real
    drafts with heavy human edits still pair.
  * 0.04 margin prevents ambiguity — if two drafts are near-ties for the
    same LinkedIn post, we can't be confident which one it was; leaving
    both unmatched is better than a 50/50 guess.

Not (yet) doing:
  * Writing matched pairs into RuanMei's observation ledger. That's a
    separate decision — the learning loop is currently partly dormant
    anyway (Ordinal syncs are off). Track as follow-up if we decide to
    keep RuanMei.
  * Companies that share an Ordinal workspace (Commenda/Trimble
    sub-slugs) are handled correctly because ``local_posts.company``
    carries the full sub-slug and ``creator_username`` is filtered per
    user, so the draft candidates never cross FOC-user boundaries.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lookback windows. These bound both the Jacquard-side scan (how far
# back do we look for unpaired LinkedIn posts?) and the draft-candidate
# window (how old can the draft be relative to the published post?).
_LINKEDIN_LOOKBACK_DAYS = 30
_DRAFT_WINDOW_DAYS      = 14

# Strict thresholds — bias toward "no match" over "wrong match" because
# we have no cheap way to unwind a wrong pairing later.
_MATCH_SIM_MIN        = 0.82
_MATCH_MARGIN_MIN     = 0.04


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_match_back() -> dict[str, Any]:
    """Run one pass of draft↔published pairing. Safe to call from any cron.

    Two passes happen in order:

      1. **Date pairing (deterministic)** — for every draft with
         ``scheduled_date`` set but ``matched_provider_urn`` still
         NULL, look up the creator's LinkedIn post with a
         ``posted_at`` on that PT calendar day. Exactly-one match
         stamps the pair with ``match_method='manual_date_deferred'``.
         This is the "auto-pair-on-scrape" flow the operator's
         ``POST /api/posts/{id}/set-publish-date`` endpoint depends
         on when the LinkedIn post isn't yet in the mirror at the
         moment the date is recorded.
      2. **Semantic pairing (cosine fallback)** — for drafts without
         a ``scheduled_date`` (or where date pairing returned
         zero/ambiguous), run the embedding-based nearest-neighbor
         match from the client's recent LinkedIn posts.

    Returns a summary dict:
        {
          "date_matched":      int,    # drafts paired via scheduled_date
          "date_skipped":      int,    # scheduled_date set but no unique match
          "semantic_matched":  int,    # semantic cosine matches
          "matched":           int,    # date_matched + semantic_matched
          "checked_posts":     int,    # LinkedIn posts inspected for semantic
          "candidate_drafts":  int,    # drafts considered for semantic
          "skipped_unsure":    int,    # semantic top-1 below threshold or tie
          "creators":          int,    # tracked creators processed (semantic)
          "duration_seconds":  float,
          "errors":            list[str],
        }

    Failures are swallowed at sub-pass level — a broken creator or a
    broken date-pair can't poison the whole run. Callers
    (``jacquard_mirror_sync``, ``amphoreus_linkedin_scrape``) also
    wrap this in try/except as a belt-and-braces.
    """
    t0 = time.time()
    summary: dict[str, Any] = {
        "date_matched":     0,
        "date_skipped":     0,
        "semantic_matched": 0,
        "matched":          0,
        "checked_posts":    0,
        "candidate_drafts": 0,
        "skipped_unsure":   0,
        "creators":         0,
        "duration_seconds": 0.0,
        "errors":           [],
    }

    # Pass 1 (date-based pairing) was removed 2026-04-28 along with the
    # operator-facing ``POST /api/posts/{id}/set-publish-date`` endpoint.
    # Operators no longer set scheduled_date manually, so the date pass
    # has nothing to do — drafts pair via semantic match only.
    # ``_run_date_pass`` is preserved below in case a backfill script
    # needs to re-process old drafts that still have scheduled_date set.

    # --- Pass 1: semantic pairing ---
    try:
        creators = _load_creator_to_company()
    except Exception as exc:
        logger.exception("[draft_match_worker] failed to load creators")
        summary["errors"].append(f"load_creators: {exc}")
        summary["duration_seconds"] = round(time.time() - t0, 2)
        summary["matched"] = summary["semantic_matched"]
        return summary

    if not creators:
        logger.info("[draft_match_worker] no tracked creators — nothing to semantically match")
        summary["duration_seconds"] = round(time.time() - t0, 2)
        summary["matched"] = summary["semantic_matched"]
        return summary

    for username, company_slugs in creators.items():
        summary["creators"] += 1
        try:
            per = _process_creator(username, company_slugs)
        except Exception as exc:
            logger.exception("[draft_match_worker] creator %s failed", username)
            summary["errors"].append(f"{username}: {str(exc)[:200]}")
            continue
        summary["semantic_matched"] += per["matched"]
        summary["checked_posts"]    += per["checked_posts"]
        summary["candidate_drafts"] += per["candidate_drafts"]
        summary["skipped_unsure"]   += per["skipped_unsure"]

    summary["matched"] = summary["semantic_matched"]
    summary["duration_seconds"] = round(time.time() - t0, 2)
    logger.info(
        "[draft_match_worker] pass done: semantic=%d / "
        "%d posts checked / %d drafts considered / %d unsure / %.1fs",
        summary["semantic_matched"],
        summary["checked_posts"], summary["candidate_drafts"],
        summary["skipped_unsure"], summary["duration_seconds"],
    )
    return summary


# ---------------------------------------------------------------------------
# Pass 1 — deterministic date pairing
# ---------------------------------------------------------------------------

def _run_date_pass() -> dict[str, int]:
    """Pair any draft with ``scheduled_date`` but no match yet.

    Query shape:
      SELECT id, company, user_id, scheduled_date
      FROM local_posts
      WHERE scheduled_date IS NOT NULL
        AND matched_provider_urn IS NULL;

    For each row:
      1. Resolve the creator's LinkedIn handle via
         ``_resolve_linkedin_username(company, user_id=user_id)`` —
         multi-FOC-aware.
      2. Convert ``scheduled_date`` (YYYY-MM-DD, PT) to a UTC
         instant-range covering the whole PT day. DST handled via
         zoneinfo.
      3. Query ``linkedin_posts`` for creator's posts in that window,
         excluding reshares + company-account posts.
      4. If exactly one match: stamp
         ``matched_provider_urn`` / ``matched_at`` /
         ``match_method='manual_date_deferred'``. If zero or
         multiple, leave the draft unmatched — it stays eligible for
         the next scrape cycle (zero case) or operator
         disambiguation (multiple case).

    Date-pairing is idempotent: once a draft gets
    ``matched_provider_urn``, the query at step 1 excludes it from
    future passes.
    """
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        logger.warning("[draft_match_worker] zoneinfo unavailable; date pass aborting")
        return {"matched": 0, "skipped": 0}

    from backend.src.db.amphoreus_supabase import _get_client
    from backend.src.agents.stelle import _resolve_linkedin_username
    sb = _get_client()
    if sb is None:
        return {"matched": 0, "skipped": 0}

    try:
        candidates = (
            sb.table("local_posts")
              .select("id, company, user_id, scheduled_date")
              .not_.is_("scheduled_date", "null")
              .is_("matched_provider_urn", "null")
              .limit(500)
              .execute().data
            or []
        )
    except Exception as exc:
        logger.warning("[draft_match_worker] date-pass candidate query failed: %s", exc)
        return {"matched": 0, "skipped": 0}

    matched = 0
    skipped = 0
    tz = ZoneInfo("America/Los_Angeles")

    for draft in candidates:
        post_id       = (draft.get("id") or "").strip()
        company       = (draft.get("company") or "").strip()
        user_id       = (draft.get("user_id") or "").strip() or None
        scheduled_str = (draft.get("scheduled_date") or "").strip()
        if not post_id or not company or not scheduled_str:
            skipped += 1
            continue

        # Parse the scheduled_date (YYYY-MM-DD) as a PT calendar day.
        # Some rows may have time components if older writes stamped
        # an ISO datetime — tolerate both.
        try:
            date_part = scheduled_str.split("T", 1)[0].split(" ", 1)[0]
            y, m, d = (int(x) for x in date_part.split("-"))
            day_start_pt = datetime(y, m, d, 0, 0, 0, tzinfo=tz)
            day_end_pt   = datetime(y, m, d, 23, 59, 59, 999999, tzinfo=tz)
            day_start_utc = day_start_pt.astimezone(timezone.utc)
            day_end_utc   = day_end_pt.astimezone(timezone.utc)
        except Exception:
            logger.debug(
                "[draft_match_worker] unparseable scheduled_date %r for %s, skipping",
                scheduled_str, post_id[:12],
            )
            skipped += 1
            continue

        username = _resolve_linkedin_username(company, user_id=user_id)
        if not username:
            # Can't resolve handle — defer until the upstream fix lands
            # (multi-FOC handle resolution is now fixed; legacy rows
            # without user_id at multi-FOC companies still fail).
            skipped += 1
            continue

        try:
            rows = (
                sb.table("linkedin_posts")
                  .select("provider_urn, posted_at")
                  .eq("creator_username", username)
                  .or_("is_company_post.is.null,is_company_post.eq.false")
                  .is_("reshared_post_urn", "null")
                  .gte("posted_at", day_start_utc.isoformat())
                  .lte("posted_at", day_end_utc.isoformat())
                  .execute().data
                or []
            )
        except Exception as exc:
            logger.debug(
                "[draft_match_worker] date-pass query for %s/%s failed: %s",
                username, date_part, exc,
            )
            skipped += 1
            continue

        if len(rows) != 1:
            # Zero matches → mirror hasn't seen it yet; keep eligible
            # for next scrape cycle. Multiple → needs operator disamb.
            continue

        matched_urn = (rows[0].get("provider_urn") or "").strip()
        if not matched_urn:
            skipped += 1
            continue

        try:
            sb.table("local_posts").update({
                "matched_provider_urn": matched_urn,
                "matched_at":           datetime.now(timezone.utc).isoformat(),
                "match_similarity":     None,
                "match_method":         "manual_date_deferred",
            }).eq("id", post_id).execute()
            matched += 1
            logger.info(
                "[draft_match_worker] date-paired draft %s → %s "
                "(@%s, %s PT)",
                post_id[:12], matched_urn[:32], username, date_part,
            )
        except Exception as exc:
            logger.warning(
                "[draft_match_worker] date-pair write failed for %s: %s",
                post_id[:12], exc,
            )
            skipped += 1

    return {"matched": matched, "skipped": skipped}


# ---------------------------------------------------------------------------
# Creator / company resolution
# ---------------------------------------------------------------------------

def _load_creator_to_company() -> dict[str, list[str]]:
    """Build ``{linkedin_username: [company_uuid, ...]}``.

    ``local_posts.company`` is stamped with the Jacquard
    ``user_companies.id`` UUID (post-``resolve_to_company_and_user``),
    not a slug, so we index candidate drafts by UUID to match. A single
    LinkedIn username can in principle map to multiple company UUIDs
    (shouldn't happen under our model) — we keep the list shape so
    nothing breaks if it ever does.

    Universe matches ``jacquard_mirror_sync._load_tracked_creators``:
    Jacquard ``users`` WHERE ``posts_content=true AND is_internal=false``.
    """
    import re
    from backend.src.services.jacquard_mirror_sync import _jacquard_client

    jcq = _jacquard_client()
    try:
        users = (
            jcq.table("users")
               .select("linkedin_url, company_id")
               .eq("posts_content", True)
               .eq("is_internal", False)
               .limit(500)
               .execute()
               .data
            or []
        )
    except Exception as exc:
        logger.warning("[draft_match_worker] users fetch failed: %s", exc)
        return {}

    out: dict[str, list[str]] = {}
    for u in users:
        url = (u.get("linkedin_url") or "").strip()
        cid = u.get("company_id")
        m = re.search(r"linkedin\.com/in/([^/?#]+)", url)
        if not (m and cid):
            continue
        username = m.group(1).strip().lower().rstrip("/")
        out.setdefault(username, []).append(cid)
    return out


# ---------------------------------------------------------------------------
# Per-creator match loop
# ---------------------------------------------------------------------------

def _process_creator(username: str, company_slugs: list[str]) -> dict[str, int]:
    """Run match-back for one creator across all their company slugs.

    A single Amphoreus deployment may have multiple sub-slug companies
    that share a LinkedIn username (e.g. trimble-mark vs trimble-heather
    if they ever collapsed onto one profile). We collect drafts across
    every slug we know about for this username.
    """
    from backend.src.db.amphoreus_supabase import (
        _get_client,
        embed_text_for_local_post,
        load_local_posts_for_match_back,
        record_local_post_match,
    )

    out = {"matched": 0, "checked_posts": 0, "candidate_drafts": 0, "skipped_unsure": 0}

    sb = _get_client()
    if sb is None:
        return out

    since_iso = (
        datetime.now(timezone.utc) - timedelta(days=_LINKEDIN_LOOKBACK_DAYS)
    ).isoformat()

    # Already-matched URNs for this creator (so we don't recompute on every cron)
    try:
        already_matched_rows = (
            sb.table("local_posts")
              .select("matched_provider_urn")
              .in_("company", company_slugs)
              .not_.is_("matched_provider_urn", "null")
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.warning(
            "[draft_match_worker] matched-urn lookup failed for %s: %s",
            username, exc,
        )
        already_matched_rows = []
    already_matched = {r.get("matched_provider_urn") for r in already_matched_rows if r.get("matched_provider_urn")}

    # LinkedIn posts for this creator we haven't paired yet.
    try:
        linkedin_posts = (
            sb.table("linkedin_posts")
              .select("provider_urn, post_text, posted_at, creator_username")
              .eq("creator_username", username)
              .gte("posted_at", since_iso)
              .not_.is_("post_text", "null")
              .order("posted_at", desc=True)
              .limit(200)
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.warning(
            "[draft_match_worker] linkedin_posts fetch failed for %s: %s",
            username, exc,
        )
        return out

    unmatched = [p for p in linkedin_posts if p.get("provider_urn") not in already_matched]
    out["checked_posts"] = len(unmatched)
    if not unmatched:
        return out

    # Candidate drafts for this creator's companies, in the lookback window.
    draft_since_iso = (
        datetime.now(timezone.utc) - timedelta(days=_LINKEDIN_LOOKBACK_DAYS + _DRAFT_WINDOW_DAYS)
    ).isoformat()
    candidates: list[dict[str, Any]] = []
    for slug in company_slugs:
        candidates.extend(load_local_posts_for_match_back(slug, generated_after_iso=draft_since_iso))
    out["candidate_drafts"] = len(candidates)
    if not candidates:
        return out

    # Build the full bipartite graph of (post, draft, sim) edges.
    # Then pair greedily max-edge-first with a both-side margin check.
    #
    # Why not "for each post, take best draft" (the previous algorithm)?
    # That greedy-by-post-iteration-order locks in early matches even
    # when a later post would be a much better fit for the same draft.
    # Worked example: best edge in graph is (P2, D1, 0.91), but loop
    # processed P1 first and gave it D1 at 0.85 because P1 came first
    # in date-descending order. P2 then had to settle for D2 at 0.84,
    # and the actual best pairing (P2↔D1) was unreachable.
    #
    # Max-edge-first (sort all edges by sim desc, take greedily) is
    # provably within 1/2 of optimal for max-weight bipartite matching
    # and almost always within a few percent in practice. The both-
    # side margin check (sim - second-best-for-this-post ≥ margin AND
    # sim - second-best-for-this-draft ≥ margin) preserves the
    # "no ambiguous pairings" property the previous one-sided check
    # was trying to enforce.
    all_edges: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    # Cache embeddings per post so we don't re-embed for the second pass.
    post_embs: dict[str, list[float]] = {}
    for post in unmatched:
        post_text = (post.get("post_text") or "").strip()
        provider_urn = (post.get("provider_urn") or "").strip()
        if not post_text or not provider_urn:
            continue
        posted_at_iso = post.get("posted_at")
        window_candidates = _filter_by_draft_window(
            candidates, posted_at_iso, _DRAFT_WINDOW_DAYS,
        )
        if not window_candidates:
            continue
        post_emb = embed_text_for_local_post(post_text)
        if post_emb is None:
            continue
        post_embs[provider_urn] = post_emb
        for draft in window_candidates:
            emb = draft.get("embedding")
            if not isinstance(emb, list) or len(emb) != len(post_emb):
                continue
            sim = _cosine(post_emb, emb)
            if sim < _MATCH_SIM_MIN:
                # Cheap floor — skip edges that can't possibly clear
                # the threshold to keep all_edges small on big graphs.
                continue
            all_edges.append((sim, post, draft))

    if not all_edges:
        return out

    # Precompute per-post and per-draft second-best similarities so the
    # margin check is consistent regardless of pairing order.
    by_post: dict[str, list[float]] = {}
    by_draft: dict[str, list[float]] = {}
    for sim, post, draft in all_edges:
        by_post.setdefault(post["provider_urn"], []).append(sim)
        by_draft.setdefault(draft["id"], []).append(sim)
    for k in by_post:
        by_post[k].sort(reverse=True)
    for k in by_draft:
        by_draft[k].sort(reverse=True)

    def _margin_ok(sim: float, post_urn: str, draft_id: str) -> bool:
        """sim must beat the second-best sim from BOTH sides by ≥ margin."""
        post_sims = by_post.get(post_urn, [])
        draft_sims = by_draft.get(draft_id, [])
        post_second = post_sims[1] if len(post_sims) > 1 else 0.0
        draft_second = draft_sims[1] if len(draft_sims) > 1 else 0.0
        return (
            (sim - post_second) >= _MATCH_MARGIN_MIN
            and (sim - draft_second) >= _MATCH_MARGIN_MIN
        )

    # Sort all edges descending by similarity. Greedy iteration takes
    # the global best available edge at each step.
    all_edges.sort(key=lambda e: e[0], reverse=True)

    paired_post_urns: set[str] = set()
    paired_draft_ids: set[str] = set()

    for sim, post, draft in all_edges:
        post_urn = post["provider_urn"]
        draft_id = draft["id"]
        if post_urn in paired_post_urns or draft_id in paired_draft_ids:
            continue
        if not _margin_ok(sim, post_urn, draft_id):
            out["skipped_unsure"] += 1
            logger.debug(
                "[draft_match_worker] unsure (max-edge-first): "
                "%s ↔ %s sim=%.3f post_2nd=%.3f draft_2nd=%.3f",
                post_urn[:24], draft_id[:12], sim,
                (by_post.get(post_urn, [0])[1] if len(by_post.get(post_urn, [])) > 1 else 0.0),
                (by_draft.get(draft_id, [0])[1] if len(by_draft.get(draft_id, [])) > 1 else 0.0),
            )
            continue
        ok = record_local_post_match(
            post_id=draft_id,
            matched_provider_urn=post_urn,
            similarity=sim,
            method="semantic",
        )
        if ok:
            out["matched"] += 1
            paired_post_urns.add(post_urn)
            paired_draft_ids.add(draft_id)
            logger.info(
                "[draft_match_worker] paired draft %s ↔ %s (sim=%.3f, max-edge)",
                draft_id[:12], post_urn[:32], sim,
            )

    return out


def _filter_by_draft_window(
    candidates: list[dict[str, Any]],
    posted_at_iso: Optional[str],
    days: int,
) -> list[dict[str, Any]]:
    """Keep drafts whose ``created_at`` is within ``[posted_at - days, posted_at]``."""
    if not posted_at_iso:
        return candidates
    try:
        pub = datetime.fromisoformat(posted_at_iso.replace("Z", "+00:00"))
    except Exception:
        return candidates
    floor = pub - timedelta(days=days)
    out = []
    for c in candidates:
        created_iso = c.get("created_at")
        if not created_iso:
            continue
        try:
            created = datetime.fromisoformat(str(created_iso).replace("Z", "+00:00"))
        except Exception:
            continue
        if floor <= created <= pub:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Similarity math
# ---------------------------------------------------------------------------

def _two_closest_drafts(
    target_emb: list[float],
    candidates: list[dict[str, Any]],
):
    """Return ``((sim1, draft1), (sim2, draft2)|None)`` — top-1 and top-2 by cosine.

    candidates each carry an ``embedding`` column (list[float]) pulled
    by load_local_posts_for_match_back.
    """
    if not candidates:
        return (None, None)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        emb = c.get("embedding")
        if not isinstance(emb, list) or len(emb) != len(target_emb):
            continue
        sim = _cosine(target_emb, emb)
        scored.append((sim, c))
    if not scored:
        return (None, None)
    scored.sort(key=lambda t: t[0], reverse=True)
    top1 = scored[0]
    top2 = scored[1] if len(scored) > 1 else None
    return (top1, top2)


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity. OpenAI text-embedding-3-small outputs
    are already L2-normalized, so the dot product is the cosine. We do
    the full cosine anyway to stay correct if that assumption ever
    breaks (e.g. switch to a different model)."""
    dot = 0.0
    a2 = 0.0
    b2 = 0.0
    for x, y in zip(a, b):
        dot += x * y
        a2 += x * x
        b2 += y * y
    if a2 == 0.0 or b2 == 0.0:
        return 0.0
    return dot / (math.sqrt(a2) * math.sqrt(b2))


# ---------------------------------------------------------------------------
# CLI — handy for local smoke tests: ``python -m backend.src.services.draft_match_worker``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Load repo-root .env so SUPABASE_URL etc. are set when run as a
    # standalone script. In production this module is imported by
    # ``jacquard_mirror_sync.run_sync`` which already has env from the
    # Fly machine config; this block is CLI-only.
    try:
        from dotenv import load_dotenv
        from pathlib import Path as _Path
        candidate = _Path(__file__).resolve().parents[3] / ".env"
        if candidate.exists():
            load_dotenv(candidate)
    except Exception:
        pass
    summary = run_match_back()
    import json
    print(json.dumps(summary, indent=2))
