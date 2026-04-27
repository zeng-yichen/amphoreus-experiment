"""Unified per-post context bundle for Stelle and Aglaea.

Motivation
----------
Prior to this module the same post appeared in up to six disjoint
prompt blocks (dedup hooks, voice examples, feedback list, edit
deltas, engagement numbers, etc.) — one facet per block, never joined.
The model had to cross-reference "which comment was about which
draft" across the prompt, and rarely did.

This module joins the four signals that matter for generation-time
context — body, engagement, comments, delta — on their natural keys,
grouped into three classes:

* ``Published``  — posts we have engagement data for. Either shipped
  via the legacy Ordinal path (``Posted`` status + Jacquard-mirrored
  URN) or picked up by the Amphoreus LinkedIn scrape directly.
* ``Draft``      — anything in the Posts tab that hasn't shipped yet.
  Collapsed 2026-04-23 from the former ``InFlight`` / ``Unpushed``
  split — the Ordinal-side wipe was retired during the Ordinal churn,
  so every draft in the Posts tab simply persists. The model infers
  ship-state from the ENGAGEMENT line (real numbers = shipped, "—
  (not yet published)" = draft), no class taxonomy needed.
  Paired with any comments the operator wrote on the draft — inline
  comments are rendered with their text anchor, post-wide comments
  as a flat list — so follow-up runs read the operator's guidance
  inline with the draft it refers to.
* ``Rejected``   — ``local_posts`` rows where ``status='rejected'``.
  Operator flipped the state via ``POST /api/posts/{id}/reject``;
  row + feedback preserved as a negative learning signal. NOT a
  dedup source — the client said no to *this execution*, not to the
  topic. Stays a distinct class because the paired comments flip
  meaning (learn-from, don't-regenerate).

Bitter Lesson note
------------------
This module only joins raw tables on their natural keys. No
heuristic weighting, no LLM curation, no "Stelle should read this
twice" annotations. Structural assembly only — the kind of
pre-chewing the filter forbids is explicitly absent.

Layer 1 addendum (2026-04-23): each rendered post block now also
shows its top-2 semantic neighbors from the same creator's own
shipped history (cosine over text-embedding-3-small embeddings).
That's structural — similarity is a raw measurement, not a human-
assigned topic label. Stelle infers clusters + trajectories from
the rendered grid. We don't name clusters, don't set thresholds
that filter posts out, don't enforce topic-diversity rules.

Neighbors are attached to **every** rendered block that has body
text, including drafts and rejected drafts — not just published
ones. For drafts, we embed the draft body on-the-fly and look up
its nearest published cousins. That's the highest-leverage case:
the operator is about to ship this, and showing "last time you
wrote in this neighborhood it hit 23 rx and declined" is the
calibration moment.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

_WINDOW_DAYS                = 90     # comment/delta lookback window
# LinkedIn post lookback windows. 21 days is the primary filter — tight
# enough to keep the creator's CURRENT voice-era (post-Virio-onboarding)
# in the bundle without dragging in pre-Virio promotional shorts or
# older content-eras that bias voice calibration. If 21d leaves <5
# originals (under the neighbor-compute minimum), we widen to 60d as a
# backoff so low-frequency creators still get semantic neighbors.
# 2026-04-23: was a single 180-day window, which pulled each FOC's
# pre-Virio content into every bundle build and bimodal-averaged the
# voice target — e.g. Mark Schwartz's Nov 2025 was 75% promotional
# shorts <500 chars, directly biasing Stelle toward shorter drafts.
_LINKEDIN_WINDOW_DAYS          = 21
_LINKEDIN_WINDOW_FALLBACK_DAYS = 60
_MIN_POSTS_BEFORE_FALLBACK     = 5   # matches _NEIGHBOR_MIN_POSTS
_MAX_COMMENT_BODY           = 600    # char cap per comment
_MAX_SELECTED_TEXT          = 200    # char cap per inline anchor
_MAX_POST_BODY_CHARS        = 4000   # char cap per post body in bundle

# Engagement-trajectory rendering. See module docstring "Layer 1
# addendum" + backend/scripts/amphoreus_supabase_engagement_snapshots_schema.sql.
# Only posts younger than this are worth showing kinetics for; beyond
# that window engagement has stabilized and a trajectory line is token
# bloat.
_TRAJECTORY_MAX_AGE_DAYS    = 14
_TRAJECTORY_MIN_POINTS      = 2      # need ≥2 snapshots to plot a shape
_TRAJECTORY_MAX_POINTS      = 6      # render at most this many points

# NO bundle-level block or char cap.
#
# An earlier iteration of this file applied a 60-block / 500 KB cap
# after the Virio ARG_MAX incident (2026-04-23). Reverting: the cap
# masked the real bug. Bundle size exploding to 1+ MB is a signal
# that per-FOC scoping is broken upstream (bundle was called without
# ``user_id`` at a multi-FOC company, leaking every FOC's local_posts
# into the output). If that regresses, we WANT the Claude CLI exec
# to fail with ``OSError: [Errno 7] Argument list too long`` — it's
# the load-bearing canary that tells us scoping broke. A silent
# truncate-to-60 hides the bug and serves a cross-FOC-leaked bundle
# (at a company with, say, 500 of FOC A's posts + 50 of FOC B's, the
# cap picks mostly A with some B mixed in — worst possible outcome).
#
# The correct fix to keep bundles small is strict per-FOC filtering
# at read-time (``build_post_bundle(company, user_id=X)`` → threads
# into ``list_local_posts`` which strict-filters). That lives below
# and must stay. If it regresses, let ARG_MAX surface.

# Amphoreus-generated ``draft_feedback`` rows that aren't real feedback —
# they are OUR OWN commentary that Hyacinthia posts to Ordinal at push
# time. Surfacing them as "client feedback" pollutes the signal.
_FEEDBACK_SKIP_PREFIXES: tuple[str, ...] = (
    "why we're posting this (internal):",
    "## castorice fact-check",
    # Castorice citation_comments auto-posted by Hyacinthia — format
    # is ``Claim: "<sentence>"\nSource: <ref>`` per castorice.py:384.
    # These are our fact-check receipts, not reviewer feedback.
    'claim: "',
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_post_bundle(
    company: str,
    *,
    window_days: int = _WINDOW_DAYS,
    user_id: Optional[str] = None,
) -> str:
    """Return the full markdown bundle block for a company.

    ``user_id`` (2026-04-23) scopes ``local_posts`` reads to a single
    FOC at shared-company clients (Virio, Trimble, Commenda). Without
    it the bundle returned every FOC's drafts — for Virio (19 FOCs)
    that meant ~750 posts per bundle, pushing the Claude CLI argv past
    Linux's ~2 MB ARG_MAX and exec-failing with ``OSError: [Errno 7]``.
    Callers should pass the ``DATABASE_USER_UUID`` env value (set by
    ``stelle_runner``) so each run sees only its target FOC's drafts.
    When None, bundle is company-wide (single-FOC clients still work).

    Returns an empty string on any failure so callers can inline it
    without guarding every code path. Never raises.

    Callers that want neighbor-rendering stats for observability
    (``neighbors_rendered``, ``posts_with_neighbors``, etc.) should
    call :func:`build_post_bundle_with_stats` instead — same core
    logic, tuple return.
    """
    bundle, _stats = build_post_bundle_with_stats(
        company, window_days=window_days, user_id=user_id,
    )
    return bundle


def build_post_bundle_with_stats(
    company: str,
    *,
    window_days: int = _WINDOW_DAYS,
    user_id: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Same as :func:`build_post_bundle` but also returns a stats dict.

    The stats are the neighbor-rendering observability surface:

      * ``embedded_posts``          — creator's shipped posts embedded
      * ``neighbor_map_size``       — keys in the creator top-k map
      * ``blocks_total``            — rendered blocks in the bundle
      * ``blocks_with_neighbors``   — blocks whose render got neighbors
      * ``draft_query_embeds``      — on-the-fly draft-body embed calls
      * ``draft_query_embed_fails`` — draft embeds that bailed
      * ``skip_reason``             — why neighbors were globally skipped
                                       (``None`` when they were attempted);
                                       one of ``no_shipped_posts`` |
                                       ``below_min_posts`` |
                                       ``embed_batch_failed``.

    A downstream consumer (Stelle's save path, a dashboard, a
    back-test script) can use these to correlate neighbor-signal
    presence with output behaviour.
    """
    company = (company or "").strip()
    stats: dict[str, Any] = {
        "embedded_posts":           0,
        "neighbor_map_size":        0,
        "blocks_total":             0,
        "blocks_with_neighbors":    0,
        "draft_query_embeds":       0,
        "draft_query_embed_fails":  0,
        "skip_reason":              None,
    }
    if not company:
        return "", stats

    try:
        ordinal_posts        = _fetch_ordinal_posts(company)
        # Per-FOC filter on Ordinal posts. Shared-Ordinal-workspace
        # clients (Virio has one Ordinal workspace for 8+ LinkedIn
        # profiles) return every teammate's posts from /posts, keyed
        # by the shared api_key. Without this filter a bundle for
        # ``virio-jon`` would include Eric's, Daniel's, Emmett's, etc.
        # Ordinal posts — hundreds of rows, cross-FOC leak, blows
        # past ARG_MAX. Filter by matching ``linkedIn.profile.name``
        # against the target FOC's display name + ``profile.detail``
        # against the linkedin_username, whichever lands first.
        if user_id:
            ordinal_posts = _filter_ordinal_posts_to_foc(
                ordinal_posts, user_id, company=company,
            )
        local_posts_by_oid, local_posts_by_id, rejected_local_posts = \
            _fetch_local_posts(company, user_id=user_id)
        # user_id threaded through so multi-FOC companies (Trimble/
        # Commenda/Virio) resolve their LinkedIn handle correctly.
        # Without it, _resolve_linkedin_username bails when a company
        # has >1 FOC, returning empty engagement → no neighbors → no
        # NEAREST CREATOR POSTS block on any rendered post. 2026-04-23.
        linkedin_posts_by_urn = _fetch_linkedin_engagement(company, user_id=user_id)
        feedback_by_draft     = _fetch_feedback(local_posts_by_id.keys())
        trajectories_by_urn   = _fetch_recent_trajectories(linkedin_posts_by_urn)
    except Exception:
        logger.exception("[post_bundle] build failed for %s", company)
        return "", stats

    # Layer 1 semantic neighbors (2026-04-23). Embed the creator's
    # shipped posts once and compute a top-k neighbor map; each rendered
    # block then shows its neighbors inline. Guarded to noop when the
    # creator has too few posts, OpenAI isn't configured, or the embed
    # batch fails — the bundle stays structurally identical, just
    # without the NEAREST CREATOR POSTS section.
    #
    # ``creator_embeddings`` is retained past the top-k compute because
    # we also use it to resolve on-the-fly embeddings of draft bodies
    # against the same creator pool (see ``_neighbors_for_draft_body``).
    neighbors_by_urn: dict[str, list[tuple[str, float]]] = {}
    creator_embeddings: dict[str, list[float]] = {}
    if not linkedin_posts_by_urn:
        stats["skip_reason"] = "no_shipped_posts"
    elif len(linkedin_posts_by_urn) < _NEIGHBOR_MIN_POSTS:
        stats["skip_reason"] = "below_min_posts"
    else:
        try:
            creator_embeddings = _embed_creator_posts(linkedin_posts_by_urn)
            if creator_embeddings:
                neighbors_by_urn = _compute_top_k_neighbors(
                    creator_embeddings, k=_NEIGHBOR_K,
                )
                stats["embedded_posts"]    = len(creator_embeddings)
                stats["neighbor_map_size"] = len(neighbors_by_urn)
                logger.info(
                    "[post_bundle] computed neighbors for %d posts "
                    "(creator, company=%s)",
                    len(neighbors_by_urn), company,
                )
            else:
                stats["skip_reason"] = "embed_batch_failed"
        except Exception as exc:
            stats["skip_reason"] = "embed_batch_failed"
            logger.warning("[post_bundle] neighbor computation failed: %s", exc)

    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    # Draft-body → nearest-shipped-post lookup. Used for every
    # non-published block (drafts, rejected) so Stelle sees
    # "last time you wrote in this cluster, it hit N rx" alongside
    # the draft it's about to iterate on. On-the-fly; skipped if
    # the creator pool wasn't embedded above.
    def _draft_neighbors(body: str) -> list[dict]:
        if not creator_embeddings or not body.strip():
            return []
        try:
            emb = _embed_single_text(body)
        except Exception as exc:
            logger.debug("[post_bundle] draft-body embed failed: %s", exc)
            stats["draft_query_embed_fails"] += 1
            return []
        if not emb:
            stats["draft_query_embed_fails"] += 1
            return []
        stats["draft_query_embeds"] += 1
        return _neighbors_for_query_embedding(
            emb, creator_embeddings, linkedin_posts_by_urn, k=_NEIGHBOR_K,
        )

    # (2026-04-23) Flattened three classes — Published, InFlight,
    # Unpushed — into a single POSTS section. Stelle can infer
    # publication status from each block's ENGAGEMENT line (real
    # numbers = shipped; "— (not yet published)" = draft). Separating
    # them was operator-facing taxonomy leaking into the prompt; the
    # model doesn't need a human's "queue stage" concept, it needs the
    # actual signal (body + engagement + comments), and the fewer
    # imposed buckets the less prompt bloat.
    #
    # REJECTED stays separate because it carries a DIFFERENT semantic:
    # learning signal (don't re-write), not dedup signal (don't
    # re-topic). That's a real distinction Stelle needs spelled out.
    draft_blocks:    list[str] = []
    rejected_blocks: list[str] = []

    def _append_block(
        block_str: str, got_neighbors: bool, is_rejected: bool,
    ) -> None:
        stats["blocks_total"] += 1
        if got_neighbors:
            stats["blocks_with_neighbors"] += 1
        (rejected_blocks if is_rejected else draft_blocks).append(block_str)

    # -- Pass 1: Ordinal posts (legacy — Ordinal churning but historical
    #            rows still show here until they roll out of window)
    for op in ordinal_posts:
        ordinal_id = (op.get("id") or "").strip()
        status = (op.get("status") or "").strip()
        li = op.get("linkedIn") or op.get("linkedin") or {}
        body = (li.get("copy") or li.get("text") or "").strip()
        title = (op.get("title") or "").strip()
        if not body and not title:
            continue
        urn = (
            li.get("urn")
            or li.get("postUrn")
            or li.get("providerUrn")
            or li.get("provider_urn")
            or ""
        ).strip()
        date_str = _extract_date(op)

        local_post = local_posts_by_oid.get(ordinal_id) if ordinal_id else None
        local_id = (local_post or {}).get("id")
        delta = _build_delta(local_post)
        # Draft ↔ published delta — only renders when this draft is
        # paired to a linkedin_posts URN (manual date-match or the
        # semantic matcher) AND that URN's text is in our mirror.
        published_delta = _build_published_delta(local_post, linkedin_posts_by_urn)
        comments = feedback_by_draft.get(local_id, []) if local_id else []
        engagement = linkedin_posts_by_urn.get(urn) if urn else None
        # Neighbor lookup: published posts with a URN look up in the
        # precomputed top-k map; in-queue posts (no URN yet) fall back
        # to on-the-fly draft-body embedding so they still get signal.
        if urn:
            neighbors = _neighbors_for_render(
                urn, neighbors_by_urn, linkedin_posts_by_urn,
            )
        else:
            neighbors = _draft_neighbors(body)

        trajectory = (
            trajectories_by_urn.get(urn) if (urn and engagement) else None
        )
        block = _render_block(
            status=status or "InFlight",
            title=title,
            body=body,
            date_str=date_str,
            engagement=engagement,
            delta=delta,
            published_delta=published_delta,
            comments=comments,
            neighbors=neighbors,
            trajectory=trajectory,
        )
        _append_block(block, bool(neighbors), is_rejected=False)

    # -- Pass 2: rejected drafts (Amphoreus-only)
    for lp in rejected_local_posts:
        body = (lp.get("content") or "").strip()
        title = (lp.get("title") or "").strip()
        if not body and not title:
            continue
        date_str = _local_post_date(lp)
        comments = feedback_by_draft.get(lp.get("id"), [])
        delta = _build_delta(lp)
        published_delta = _build_published_delta(lp, linkedin_posts_by_urn)
        neighbors = _draft_neighbors(body)
        block = _render_block(
            status="Rejected",
            title=title,
            body=body,
            date_str=date_str,
            engagement=None,          # never shipped
            delta=delta,
            published_delta=published_delta,
            comments=comments,
            neighbors=neighbors,
        )
        _append_block(block, bool(neighbors), is_rejected=True)

    # -- Pass 2.5: drafts in the Posts tab (status='draft', no Ordinal
    # coverage). The Ordinal-wipe-at-run-start was retired during the
    # Ordinal churn (2026-04-23), so these simply persist across runs:
    #   (a) operator comments on a draft survive across re-generations;
    #   (b) Stelle's dedup counts in-review drafts the same as any
    #       other queue entry — prevents re-generating a topic already
    #       in the operator's review pile.
    # Post-Ordinal-churn this is the dominant class; the Pass 1 loop
    # above will be empty for any FOC that never pushed to Ordinal.
    for lp in local_posts_by_id.values():
        status = (lp.get("status") or "").strip().lower()
        if status != "draft":
            continue
        if (lp.get("ordinal_post_id") or "").strip():
            continue  # covered by Pass 1 (Ordinal-side status governs)
        body = (lp.get("content") or "").strip()
        title = (lp.get("title") or "").strip()
        if not body and not title:
            continue
        date_str = _local_post_date(lp)
        comments = feedback_by_draft.get(lp.get("id"), [])
        delta = _build_delta(lp)
        published_delta = _build_published_delta(lp, linkedin_posts_by_urn)
        neighbors = _draft_neighbors(body)
        block = _render_block(
            status="InFlight",
            title=title,
            body=body,
            date_str=date_str,
            engagement=None,          # never shipped
            delta=delta,
            published_delta=published_delta,
            comments=comments,
            neighbors=neighbors,
        )
        _append_block(block, bool(neighbors), is_rejected=False)

    # -- Pass 3: Jacquard LinkedIn posts not already covered via Ordinal URN
    for urn, lp_row in linkedin_posts_by_urn.items():
        # Skip if this URN was already rendered via an Ordinal post.
        if urn in _ordinal_urns(ordinal_posts):
            continue
        posted_at = lp_row.get("posted_at") or ""
        try:
            if posted_at and datetime.fromisoformat(
                posted_at.replace("Z", "+00:00")
            ) < since - timedelta(days=_LINKEDIN_WINDOW_DAYS - window_days):
                # Jacquard window is wider; still accept.
                pass
        except Exception:
            pass
        body = (lp_row.get("post_text") or "").strip()
        if not body:
            continue
        neighbors = _neighbors_for_render(
            urn, neighbors_by_urn, linkedin_posts_by_urn,
        )
        trajectory = trajectories_by_urn.get(urn)
        block = _render_block(
            status="Posted",
            title=(lp_row.get("hook") or "").strip(),
            body=body,
            date_str=(posted_at or "")[:10],
            engagement=lp_row,
            delta=None,                # no Amphoreus draft to diff against
            comments=[],               # no Amphoreus draft → no draft_feedback
            neighbors=neighbors,
            trajectory=trajectory,
        )
        _append_block(block, bool(neighbors), is_rejected=False)

    if not (draft_blocks or rejected_blocks):
        return "", stats
    # No count cap. If this produces a bundle large enough to trip
    # Claude CLI's ARG_MAX on exec, scoping is broken — see the
    # deleted-cap comment block at the top of this file.

    parts = [
        "=== POSTS (body · engagement · comments · edits, last 90d) ===",
        "",
        "Each block below contains one post with every signal we have",
        "about it bundled together. Posts with real engagement numbers",
        "(reactions, comments, reposts) shipped to LinkedIn — ingest",
        "for voice calibration + performance reference. Posts with",
        "`ENGAGEMENT: — (not yet published)` are sitting in the",
        "operator's Amphoreus review pile; treat as topic dedup",
        "(don't re-write the same angle) and pay close attention to any",
        "inline/post-wide comments below each one — those are the",
        "operator's guidance.",
        "",
        "Each block may also carry a NEAREST CREATOR POSTS section —",
        "the most semantically-similar posts from this creator's own",
        "shipped history, each with its reaction count. Use these to",
        "read your own content trajectory: posts that cluster together",
        "show how a topic / angle / framing has performed over time.",
        "",
        "Rejected = client said no to THIS execution; DO NOT treat as",
        "topic dedup, but learn from the paired comments below each.",
    ]
    if draft_blocks:
        parts.append("")
        parts.append(f"--- POSTS ({len(draft_blocks)}) ---")
        parts.extend(draft_blocks)
    if rejected_blocks:
        parts.append("")
        parts.append(f"--- REJECTED ({len(rejected_blocks)}) ---")
        parts.extend(rejected_blocks)
    parts.append("")
    parts.append("=== END POSTS ===")

    # Structured stats log — grep-friendly so a downstream audit
    # script (``neighbor_signal_audit``) can aggregate bundle builds
    # without parsing free-text log lines. Keys match the returned
    # dict exactly.
    logger.info(
        "[post_bundle] stats company=%s user_id=%s "
        "blocks_total=%d blocks_with_neighbors=%d "
        "embedded_posts=%d neighbor_map_size=%d "
        "draft_query_embeds=%d draft_query_embed_fails=%d "
        "skip_reason=%s",
        company, user_id or "-",
        stats["blocks_total"], stats["blocks_with_neighbors"],
        stats["embedded_posts"], stats["neighbor_map_size"],
        stats["draft_query_embeds"], stats["draft_query_embed_fails"],
        stats["skip_reason"] or "none",
    )
    # No char-budget truncation. Same reasoning as the removed block
    # cap above: a bundle large enough to trip Claude CLI's ARG_MAX
    # is a regression signal, not something to silently hide.
    return "\n".join(parts), stats


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_ordinal_posts(company: str) -> list[dict[str, Any]]:
    """Paginate Ordinal ``/api/v1/posts`` and return every row. No status
    filter — see module docstring.
    """
    from backend.src.agents.stelle import _get_ordinal_api_key
    api_key = _get_ordinal_api_key(company)
    if not api_key:
        return []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    out: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: dict[str, Any] = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = httpx.get(
                "https://app.tryordinal.com/api/v1/posts",
                params=params, headers=headers, timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[post_bundle] Ordinal fetch failed for %s: %s", company, exc)
            break
        out.extend(data.get("posts", []) or [])
        if not data.get("hasMore") or not data.get("nextCursor"):
            break
        cursor = data["nextCursor"]
    return out


def _fetch_local_posts(
    company: str,
    *,
    user_id: Optional[str] = None,
) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    """Return three views of Amphoreus ``local_posts``:
      * by ``ordinal_post_id`` — lets Ordinal posts pick up delta/comments.
      * by ``id``             — lets us look up draft_feedback.draft_id.
      * rejected list         — rejected-class bundle entries.

    Queries both the slug identifier AND the canonical UUID. Callers
    (Stelle, Aglaea, claude_cli) pass the slug ("hume"), but during the
    company-slug-to-UUID migration window many rows are stored under
    the UUID (``9abcb96e-…`` for Hume). Without this dual lookup the
    bundler silently returns zero rows for every UUID-keyed company.

    ``user_id`` (2026-04-23): when supplied, strict-filters on
    ``local_posts.user_id`` so a per-FOC call to the bundler sees only
    that FOC's drafts. Critical at shared-company clients (Virio 19
    FOCs, Trimble 2 FOCs, Commenda 2 FOCs) where an unfiltered read
    returned every teammate's drafts, blowing up both prompt size and
    cross-FOC topic bleed.
    """
    by_oid: dict[str, dict] = {}
    by_id:  dict[str, dict] = {}
    rejected: list[dict] = []

    # Resolve slug → UUID; query whichever keys exist.
    candidates: list[str] = []
    seen: set[str] = set()
    for key in (company, _resolve_company_uuid(company)):
        if key and key not in seen:
            seen.add(key)
            candidates.append(key)

    try:
        from backend.src.db.local import list_local_posts as _list_lp
        for key in candidates:
            # Per-FOC scoping: when user_id given, list_local_posts
            # strict-filters (see db/local.py list_local_posts docstring).
            # When None, company-wide view (single-FOC clients + admin).
            rows = (
                _list_lp(company=key, user_id=user_id, limit=500)
                if user_id
                else _list_lp(company=key, limit=500)
            )
            for row in rows:
                rid = (row.get("id") or "").strip()
                if not rid or rid in by_id:
                    continue
                by_id[rid] = row
                oid = (row.get("ordinal_post_id") or "").strip()
                if oid:
                    by_oid[oid] = row
                if (row.get("status") or "").strip().lower() == "rejected":
                    rejected.append(row)
    except Exception as exc:
        logger.warning("[post_bundle] local_posts fetch failed for %s: %s", company, exc)
    return by_oid, by_id, rejected


def _resolve_company_uuid(identifier: str) -> Optional[str]:
    """Best-effort slug → UUID resolution. Returns None if already a
    UUID or unresolvable — callers handle that case.
    """
    try:
        from backend.src.lib.company_resolver import resolve_to_uuid
        return resolve_to_uuid(identifier)
    except Exception:
        return None


def _filter_ordinal_posts_to_foc(
    posts: list[dict[str, Any]],
    user_id: str,
    *,
    company: str,
) -> list[dict[str, Any]]:
    """Return only the Ordinal posts targeting this FOC's LinkedIn profile.

    Ordinal workspaces at shared-api-key clients (Virio, Trimble,
    Commenda) contain posts for every FOC under the workspace. A
    GET /posts returns them all — we need to pick the subset where
    ``linkedIn.profile`` points at the target user.

    Matching is liberal-but-precise:
      1. ``profile.detail`` matches the linkedin_username as a
         dash-prefixed slug component ("jongillon" inside
         "jongillon-virio"). This is Ordinal's canonical per-profile
         slug; strongest signal when present.
      2. ``profile.name`` matches the user's "First Last" display
         name (case-insensitive). Fallback when the detail slug
         isn't structured as expected.

    Returns an empty list if neither signal resolves — i.e. the user
    has no Ordinal profile in this workspace, which is the correct
    answer for a FOC who's never had a draft pushed. Never raises.
    """
    try:
        from backend.src.db.amphoreus_supabase import _get_client
        sb = _get_client()
        display_name: Optional[str] = None
        linkedin_username: Optional[str] = None
        if sb is not None:
            row = (
                sb.table("users")
                  .select("first_name, last_name, linkedin_url")
                  .eq("id", user_id)
                  .limit(1)
                  .execute()
                  .data
                or [None]
            )[0]
            if row:
                first = (row.get("first_name") or "").strip()
                last = (row.get("last_name") or "").strip()
                if first or last:
                    display_name = f"{first} {last}".strip()
                url = (row.get("linkedin_url") or "").strip()
                # Extract "jongillon" from "https://www.linkedin.com/in/jongillon/"
                if url:
                    import re as _re
                    m = _re.search(r"/in/([^/?#]+)", url)
                    if m:
                        linkedin_username = m.group(1).strip().lower()
    except Exception as exc:
        logger.debug("[post_bundle] ordinal FOC-filter lookup failed: %s", exc)
        return posts  # fail-open rather than drop everything

    if not display_name and not linkedin_username:
        logger.warning(
            "[post_bundle] could not resolve display_name / linkedin_username "
            "for user_id=%s (company=%s) — Ordinal posts cannot be per-FOC "
            "filtered. Returning unfiltered list.",
            user_id, company,
        )
        return posts

    target_name_lower = (display_name or "").lower()
    target_un = (linkedin_username or "").lower()

    out: list[dict[str, Any]] = []
    for p in posts:
        prof = (p.get("linkedIn") or p.get("linkedin") or {}).get("profile") or {}
        detail = (prof.get("detail") or "").lower()
        name   = (prof.get("name") or "").lower()
        # Detail match: username appears as a dash-separated component
        # (e.g. "jongillon-virio" or just "jongillon"). Prefix +
        # component-boundary check to avoid false positives on
        # e.g. "jongillondoppel-virio".
        matched = False
        if target_un and detail:
            if detail == target_un or detail.startswith(f"{target_un}-"):
                matched = True
        if not matched and target_name_lower and name:
            if name == target_name_lower:
                matched = True
        if matched:
            out.append(p)
    logger.info(
        "[post_bundle] Ordinal posts filtered %d → %d for user_id=%s "
        "(display_name=%r, linkedin_username=%r)",
        len(posts), len(out), user_id, display_name, linkedin_username,
    )
    return out


def _fetch_linkedin_engagement(
    company: str, user_id: Optional[str] = None,
) -> dict[str, dict]:
    """Return ``{provider_urn: linkedin_post_row}`` for this creator's
    recent posts. Provides engagement numbers + fallback body text for
    posts that went live via a non-Ordinal path.

    ``user_id`` is threaded through to ``_resolve_linkedin_username``.
    **Load-bearing for multi-FOC companies** (Trimble/Commenda/Virio):
    without it, the resolver bails on bare-UUID company inputs with
    >1 FOC and returns None, so the bundle renders with empty
    engagement data and zero semantic neighbors. See the 2026-04-23
    fix in ``stelle.py::_resolve_linkedin_username``.

    **Source cutover (2026-04-23):** reads from the Amphoreus mirror
    (``AMPHOREUS_SUPABASE_URL``), not Jacquard upstream
    (``SUPABASE_URL``). This lets the Amphoreus-owned Apify scrape
    (``amphoreus_linkedin_scrape.py``) land fresh engagement counts
    directly into Stelle's bundle without waiting on Jacquard's
    scrape cadence.

    The mirror is populated by BOTH legs — the 8-hour Jacquard mirror
    sync writes rows with ``_source='jacquard'``, and the weekday-
    midnight Amphoreus scrape writes/updates rows with
    ``_source='amphoreus'``. Stelle reads whatever's freshest in the
    union. When only one leg has seen a post, we still get it; when
    both have, the more-recently-refreshed counts win (Jacquard writes
    through its mirror cadence, Amphoreus writes through its scrape
    cadence, both update engagement columns without clobbering each
    other's identity fields).
    """
    from backend.src.agents.stelle import _resolve_linkedin_username
    username = _resolve_linkedin_username(company, user_id=user_id)
    if not username:
        return {}
    import os
    url = os.environ.get("AMPHOREUS_SUPABASE_URL", "").strip()
    key = os.environ.get("AMPHOREUS_SUPABASE_KEY", "").strip()
    if not url or not key:
        return {}

    def _fetch(days: int) -> list[dict]:
        since_iso = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        try:
            resp = httpx.get(
                f"{url}/rest/v1/linkedin_posts",
                params={
                    "select": (
                        "provider_urn,post_text,hook,posted_at,total_reactions,"
                        "total_comments,total_reposts,engagement_score,is_outlier"
                    ),
                    "creator_username":  f"eq.{username}",
                    # Exclude explicit company-post rows but keep NULLs.
                    # Jacquard sets ``is_company_post`` reliably, the
                    # Amphoreus Apify scrape leaves it unset. PostgREST's
                    # ``eq.false`` and ``not.eq.true`` BOTH reject NULL
                    # (SQL tri-valued logic), so an Amphoreus-scraped row
                    # vanishes from the bundle if we use either. The
                    # explicit ``or=`` clause is the only way to say
                    # "false or null" in PostgREST query params.
                    "or":                "(is_company_post.is.null,is_company_post.eq.false)",
                    # Exclude reshares — they're someone else's content
                    # the creator chose to repost, voice-neutral signal.
                    # 2026-04-23: previously slipped through and
                    # polluted the bundle + neighbor compute.
                    "reshared_post_urn": "is.null",
                    "post_text":         "not.is.null",
                    "posted_at":         f"gte.{since_iso}",
                    "order":             "posted_at.desc",
                    "limit":             "200",
                },
                headers={
                    "apikey":        key,
                    "Authorization": f"Bearer {key}",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json() or []
        except Exception as exc:
            logger.warning(
                "[post_bundle] linkedin_posts fetch failed for %s (window=%dd): %s",
                company, days, exc,
            )
            return []

    # Primary: tight 21-day window → clean current-voice-era calibration.
    # Rationale: a creator's voice shifts over time, and for a newly-
    # onboarded client the pre-Virio content is usually much shorter /
    # more promotional. Tight window keeps Stelle's voice target
    # aligned with the posting style of the last ~3 weeks.
    rows = _fetch(_LINKEDIN_WINDOW_DAYS)
    if len(rows) < _MIN_POSTS_BEFORE_FALLBACK:
        # Fallback: widen to 60d so low-frequency creators still get
        # enough posts to compute semantic neighbors (which require
        # _NEIGHBOR_MIN_POSTS=5). Logged so we can spot the FOCs
        # that never reach the 21d threshold.
        logger.info(
            "[post_bundle] only %d post(s) in last %dd for %s — "
            "widening to %dd fallback window",
            len(rows), _LINKEDIN_WINDOW_DAYS, username,
            _LINKEDIN_WINDOW_FALLBACK_DAYS,
        )
        rows = _fetch(_LINKEDIN_WINDOW_FALLBACK_DAYS)

    out: dict[str, dict] = {}
    for r in rows:
        urn = (r.get("provider_urn") or "").strip()
        if urn and urn not in out:
            out[urn] = r
    return out


# ---------------------------------------------------------------------------
# Semantic neighbors (Layer 1 — 2026-04-23)
# ---------------------------------------------------------------------------
#
# For each Jacquard-mirrored post in the bundle, compute its top-2 most
# semantically-similar priors from the SAME creator's history. Rendered
# inline with each block so Stelle sees "this post clusters with these
# others, which landed at these reaction counts." Intended effect: the
# model infers topic clusters and their engagement trajectories from
# the raw grid, without us naming clusters or encoding a taxonomy.
#
# Why on-the-fly vs using ``post_embeddings`` (pgvector): the mirror
# corpus has a known coverage gap — rows with ``linkedin_posts.id =
# null`` in Jacquard (≈278k of 390k) have never been embedded. For
# shared-Ordinal-account clusters (Virio, Trimble, Commenda) that gap
# hits almost every recent post. Embedding on-demand at bundle-build
# time sidesteps the gap without blocking on the queued PK migration
# (see ``delegated-tumbling-quasar`` plan file). Cost per build is
# ~$0.0002 (30 posts × 300 words × $0.02/M tok).

_NEIGHBOR_MIN_POSTS     = 5      # skip neighbor rendering below this
_NEIGHBOR_MAX_POSTS     = 30     # embed cap per bundle build
_NEIGHBOR_EMBED_MODEL   = "text-embedding-3-small"  # matches post_embeddings model
_NEIGHBOR_K             = 2      # top-k per post
_NEIGHBOR_BODY_CAP      = 4000   # char cap on text sent to OpenAI


def _embed_creator_posts(
    linkedin_posts_by_urn: dict[str, dict],
) -> dict[str, list[float]]:
    """Return ``{urn: embedding_vector}`` for the creator's posts.

    ``text-embedding-3-small`` outputs are L2-normalized, so downstream
    cosine similarity is equivalent to a plain dot product. We rely on
    that in :func:`_compute_top_k_neighbors`.
    """
    if not linkedin_posts_by_urn:
        return {}
    try:
        from openai import OpenAI
    except Exception as exc:
        logger.debug("[post_bundle] openai SDK unavailable; skipping neighbors: %s", exc)
        return {}
    import os as _os
    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.debug("[post_bundle] OPENAI_API_KEY not set; skipping neighbors")
        return {}

    # Order by posted_at desc + cap so a creator with 100+ posts in
    # window doesn't balloon the OpenAI batch. Most recent posts are
    # the ones Stelle needs trajectory visibility on.
    items = sorted(
        linkedin_posts_by_urn.items(),
        key=lambda kv: kv[1].get("posted_at") or "",
        reverse=True,
    )[:_NEIGHBOR_MAX_POSTS]

    pairs = [(u, (r.get("post_text") or "")[:_NEIGHBOR_BODY_CAP]) for u, r in items]
    pairs = [(u, t) for u, t in pairs if t.strip()]
    if len(pairs) < _NEIGHBOR_MIN_POSTS:
        return {}

    urns = [u for u, _ in pairs]
    texts = [t for _, t in pairs]
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=_NEIGHBOR_EMBED_MODEL, input=texts)
    except Exception as exc:
        logger.warning("[post_bundle] creator-post embedding batch failed: %s", exc)
        return {}

    return {u: list(d.embedding) for u, d in zip(urns, resp.data)}


def _compute_top_k_neighbors(
    embeddings: dict[str, list[float]],
    k: int = _NEIGHBOR_K,
) -> dict[str, list[tuple[str, float]]]:
    """For each urn, return its ``k`` nearest neighbors by cosine.

    Self-excluded. O(n²) matrix; fine for ≤30 posts per creator.
    Relies on text-embedding-3-small's L2-normalized outputs so the
    dot product equals cosine similarity.
    """
    if len(embeddings) < 2:
        return {}
    urns = list(embeddings.keys())
    out: dict[str, list[tuple[str, float]]] = {}
    for i, a_urn in enumerate(urns):
        a = embeddings[a_urn]
        sims: list[tuple[str, float]] = []
        for j, b_urn in enumerate(urns):
            if i == j:
                continue
            b = embeddings[b_urn]
            # Pure dot product — vectors are already L2-normalized.
            dot = 0.0
            for x, y in zip(a, b):
                dot += x * y
            sims.append((b_urn, dot))
        sims.sort(key=lambda t: -t[1])
        out[a_urn] = sims[:k]
    return out


def _fetch_recent_trajectories(
    linkedin_posts_by_urn: dict[str, dict],
) -> dict[str, list[dict]]:
    """Return ``{provider_urn: [snapshot_rows...]}`` for posts still
    inside the trajectory-rendering window (<14d old).

    Older posts are skipped entirely — their engagement has stabilized
    and the trajectory line is token bloat. One batched Supabase read
    covers every in-window URN.
    """
    if not linkedin_posts_by_urn:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=_TRAJECTORY_MAX_AGE_DAYS)
    recent_urns: list[str] = []
    for urn, row in linkedin_posts_by_urn.items():
        posted_at_iso = (row.get("posted_at") or "").strip()
        if not posted_at_iso:
            continue
        try:
            dt = datetime.fromisoformat(posted_at_iso.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff:
            recent_urns.append(urn)
    if not recent_urns:
        return {}
    try:
        from backend.src.db.amphoreus_supabase import fetch_engagement_trajectories
        return fetch_engagement_trajectories(
            recent_urns,
            # Pull history slightly wider than the render window so a
            # post at day 13.9 still has its full history.
            since_iso=(datetime.now(timezone.utc) - timedelta(
                days=_TRAJECTORY_MAX_AGE_DAYS + 2,
            )).isoformat(),
            per_urn_limit=20,
        )
    except Exception as exc:
        logger.debug("[post_bundle] trajectory fetch failed: %s", exc)
        return {}


def _neighbors_for_render(
    urn: str,
    neighbors_by_urn: dict[str, list[tuple[str, float]]],
    linkedin_posts_by_urn: dict[str, dict],
) -> list[dict]:
    """Resolve neighbor URNs to minimal rendering payloads.

    Returns an empty list when the post has no neighbors (creator with
    < 5 posts in window, embedding fetch failed, etc.) so the render
    helper can conditionally suppress the section. Similarity score
    is intentionally *not* in the returned dict — see the comment in
    :func:`_render_block`'s neighbor block.
    """
    raw = neighbors_by_urn.get(urn) or []
    out: list[dict] = []
    for n_urn, _sim in raw:
        r = linkedin_posts_by_urn.get(n_urn)
        if not r:
            continue
        hook = (r.get("hook") or r.get("post_text") or "").split("\n", 1)[0].strip()[:100]
        out.append({
            "posted_at": (r.get("posted_at") or "")[:10],
            "reactions": r.get("total_reactions") or 0,
            "hook": hook,
        })
    return out


def _embed_single_text(text: str) -> Optional[list[float]]:
    """Embed one arbitrary piece of text with text-embedding-3-small.

    Used for draft-body → nearest-shipped-post lookup: we embed the
    in-review draft's content and find its nearest creator-history
    neighbors on the fly. Same model + L2-normalization contract as
    :func:`_embed_creator_posts` so dot product equals cosine.

    Returns ``None`` on any failure (no API key, OpenAI 5xx, empty
    text). Caller should treat None as "no neighbors" and render
    the block without the NEAREST CREATOR POSTS section.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    import os as _os
    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(
            model=_NEIGHBOR_EMBED_MODEL,
            input=text[:_NEIGHBOR_BODY_CAP],
        )
    except Exception as exc:
        logger.warning("[post_bundle] single-text embedding failed: %s", exc)
        return None
    return list(resp.data[0].embedding) if resp.data else None


def _neighbors_for_query_embedding(
    query_emb: list[float],
    creator_embeddings: dict[str, list[float]],
    linkedin_posts_by_urn: dict[str, dict],
    k: int = _NEIGHBOR_K,
) -> list[dict]:
    """Return the top-``k`` nearest creator-history posts to an ad-hoc
    query embedding (e.g. a draft body).

    Only considers URNs present in ``linkedin_posts_by_urn`` — posts
    we couldn't resolve back to a row get dropped so the render never
    ends up with a dangling neighbor reference.
    """
    if not creator_embeddings or not query_emb:
        return []
    scored: list[tuple[str, float]] = []
    for u, v in creator_embeddings.items():
        if len(v) != len(query_emb):
            continue
        dot = 0.0
        for x, y in zip(query_emb, v):
            dot += x * y
        scored.append((u, dot))
    scored.sort(key=lambda t: -t[1])
    out: list[dict] = []
    for n_urn, _sim in scored[:k]:
        r = linkedin_posts_by_urn.get(n_urn)
        if not r:
            continue
        hook = (r.get("hook") or r.get("post_text") or "").split("\n", 1)[0].strip()[:100]
        out.append({
            "posted_at": (r.get("posted_at") or "")[:10],
            "reactions": r.get("total_reactions") or 0,
            "hook": hook,
        })
    return out


def _fetch_feedback(draft_ids) -> dict[str, list[dict]]:
    """Return ``{draft_id: [comment_rows...]}`` for the given drafts.
    Filters out Amphoreus self-rationale prefixes so the learning signal
    is only real reviewer input.
    """
    ids = [i for i in draft_ids if i]
    if not ids:
        return {}
    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
        if not is_configured():
            return {}
        sb = _get_client()
        if sb is None:
            return {}
    except Exception:
        return {}
    try:
        rows = (
            sb.table("draft_feedback")
              .select(
                  "draft_id, source, body, selected_text, selection_start, "
                  "selection_end, author_email, author_name, created_at"
              )
              .in_("draft_id", list(ids)[:500])
              .order("created_at", desc=True)
              .limit(2000)
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.warning("[post_bundle] draft_feedback fetch failed: %s", exc)
        return {}
    out: dict[str, list[dict]] = {}
    for r in rows:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        body_lc = body.lower()
        if any(body_lc.startswith(p) for p in _FEEDBACK_SKIP_PREFIXES):
            continue
        out.setdefault(r.get("draft_id"), []).append(r)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_date(ordinal_post: dict) -> str:
    for key in ("publishDate", "publishAt", "createdAt"):
        val = ordinal_post.get(key)
        if val:
            return str(val)[:10]
    return ""


def _local_post_date(row: dict) -> str:
    ca = row.get("created_at")
    if isinstance(ca, (int, float)):
        try:
            return datetime.fromtimestamp(ca, tz=timezone.utc).date().isoformat()
        except Exception:
            return ""
    if isinstance(ca, str):
        return ca[:10]
    return ""


def _ordinal_urns(ordinal_posts: list[dict]) -> set[str]:
    urns: set[str] = set()
    for op in ordinal_posts:
        li = op.get("linkedIn") or op.get("linkedin") or {}
        urn = (
            li.get("urn")
            or li.get("postUrn")
            or li.get("providerUrn")
            or li.get("provider_urn")
            or ""
        ).strip()
        if urn:
            urns.add(urn)
    return urns


def _build_delta(local_post: Optional[dict]) -> Optional[dict]:
    """Return ``{pre, post}`` if the operator/Castorice edited this
    draft between Stelle's first write and the current content, else
    None. Both strings are trimmed to ``_MAX_POST_BODY_CHARS``.
    """
    if not local_post:
        return None
    pre = (local_post.get("pre_revision_content") or "").strip()
    post = (local_post.get("content") or "").strip()
    if not pre or not post or pre == post:
        return None
    return {
        "pre":  pre[:_MAX_POST_BODY_CHARS],
        "post": post[:_MAX_POST_BODY_CHARS],
    }


def _build_published_delta(
    local_post: Optional[dict],
    linkedin_posts_by_urn: dict[str, dict],
) -> Optional[dict]:
    """Return ``{draft, published}`` when this draft is paired to a
    published LinkedIn post AND the two texts actually differ.

    Sourced from:
      * ``local_post.matched_provider_urn`` — set either by the
        semantic match-back worker (``match_method='semantic'``) or
        the operator-driven ``POST /api/posts/{id}/set-publish-date``
        flow (``match_method='manual_date'``).
      * ``linkedin_posts_by_urn[matched_provider_urn].post_text`` —
        the scraped published body. Only available in the bundle
        when the creator's mirror fetch covered this URN.

    Returns None if no pairing, no resolvable published text, or the
    draft/published bodies are effectively identical (no signal).
    """
    if not local_post:
        return None
    urn = (local_post.get("matched_provider_urn") or "").strip()
    if not urn:
        return None
    matched_row = linkedin_posts_by_urn.get(urn)
    if not matched_row:
        return None
    draft_text = (local_post.get("content") or "").strip()
    published_text = (matched_row.get("post_text") or "").strip()
    if not draft_text or not published_text:
        return None
    if draft_text == published_text:
        return None
    method = (local_post.get("match_method") or "").strip() or "unknown"
    return {
        "draft":          draft_text[:_MAX_POST_BODY_CHARS],
        "published":      published_text[:_MAX_POST_BODY_CHARS],
        "match_method":   method,
        "published_date": (matched_row.get("posted_at") or "")[:10],
    }


def _render_block(
    *,
    status:          str,
    title:           str,
    body:            str,
    date_str:        str,
    engagement:      Optional[dict],
    delta:           Optional[dict],
    comments:        list[dict],
    neighbors:       Optional[list[dict]] = None,
    trajectory:      Optional[list[dict]] = None,
    published_delta: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    # (2026-04-23) Header is just the date + a REJECTED marker when
    # applicable. Previously we stamped [Posted] / [InFlight] /
    # [Unpushed] on every block — stripped because Stelle now reads
    # the ENGAGEMENT line as the publication signal, and the
    # inflight/unpushed distinction doesn't exist in the prompt
    # anymore. REJECTED stays explicit because it flips the
    # interpretation of the paired comments (learning signal vs dedup).
    header_bits: list[str] = []
    if status == "Rejected":
        header_bits.append("[REJECTED]")
    if date_str:
        header_bits.append(date_str)
    header = " · ".join(header_bits) if header_bits else "—"
    lines.append("")
    lines.append(header)
    if title:
        lines.append(f'"{title}"')
    lines.append("—")
    lines.append("BODY:")
    lines.append(body[:_MAX_POST_BODY_CHARS])

    if engagement is not None:
        react  = engagement.get("total_reactions") or 0
        cmts   = engagement.get("total_comments")  or 0
        reps   = engagement.get("total_reposts")   or 0
        score  = engagement.get("engagement_score") or 0
        disp   = (score / 100) if isinstance(score, (int, float)) and score else 0
        out    = (
            f"ENGAGEMENT: {react} reactions · {cmts} comments · "
            f"{reps} reposts · {disp:.2f} score"
        )
        if engagement.get("is_outlier"):
            out += " · OUTLIER (top 20%)"
        lines.append("")
        lines.append(out)
        # Kinetic trajectory — renders only when the post is <14d old
        # AND we have ≥2 snapshots spanning real time. Older posts have
        # stabilized; showing their history is token bloat. Format:
        # ``TRAJECTORY: d0: 8/2  d1: 18/5  d3: 23/6`` (dN = days since
        # posted_at, values = reactions/comments). See
        # ``_format_trajectory`` for the shape logic.
        traj_line = _format_trajectory(trajectory, engagement.get("posted_at"))
        if traj_line:
            lines.append(f"  {traj_line}")
    elif status == "Rejected":
        lines.append("")
        lines.append("ENGAGEMENT: — (rejected, never shipped)")
    else:
        lines.append("")
        lines.append("ENGAGEMENT: — (not yet published)")

    if delta is not None:
        lines.append("")
        lines.append("DELTA (pre-revision → current):")
        lines.append("  pre:")
        lines.append(_indent(delta["pre"],  "    "))
        lines.append("  current:")
        lines.append(_indent(delta["post"], "    "))

    if published_delta is not None:
        # What the client actually shipped to LinkedIn vs. what we
        # handed them. This is the delta that was dormant for months
        # after Ordinal churned — reinstated via the manual-date
        # pairing flow (POST /api/posts/{id}/set-publish-date) + the
        # semantic match-back worker. A non-empty block here means
        # the operator either edited our draft or the matcher paired
        # a post that diverged during review.
        pub_date = published_delta.get("published_date") or "?"
        method   = published_delta.get("match_method") or "?"
        lines.append("")
        lines.append(
            f"DELTA (draft → published, {pub_date}, match={method}):"
        )
        lines.append("  draft (ours):")
        lines.append(_indent(published_delta["draft"],     "    "))
        lines.append("  published (client shipped):")
        lines.append(_indent(published_delta["published"], "    "))

    if comments:
        lines.append("")
        lines.append("COMMENTS:")
        for c in comments:
            src     = (c.get("source") or "").strip()
            author  = (c.get("author_email") or c.get("author_name") or "unknown").strip()
            sel     = (c.get("selected_text") or "").strip()
            body_c  = (c.get("body") or "").strip()[:_MAX_COMMENT_BODY]
            if src == "operator_inline" or sel:
                anchor = f'inline, line: "{sel[:_MAX_SELECTED_TEXT]}"' if sel else "inline"
                lines.append(f"  [{anchor}]")
            else:
                lines.append(f"  [{src or 'post-wide'}]")
            lines.append(f"    {author} — {body_c}")

    if neighbors:
        # Top-k semantically-similar priors from the same creator.
        # Rendered as a compact grid so the model can infer
        # topic-cluster membership + engagement trajectory per cluster.
        # See module docstring "Layer 1 addendum" for rationale.
        #
        # Similarity score is intentionally omitted from the rendered
        # line — the float invites the model to reason about the
        # number (is 0.82 high? low?) instead of reading the cluster
        # pattern. Raw similarity stays in the debug log for ops;
        # the model sees just date + reactions + hook, which is
        # enough to recognise the cluster.
        lines.append("")
        lines.append("NEAREST CREATOR POSTS (semantic):")
        for n in neighbors:
            lines.append(
                f"  [{n.get('posted_at', '?'):10s}] "
                f"{n.get('reactions', 0):>4} rx  "
                f"\"{n.get('hook', '')[:100]}\""
            )

    lines.append("")
    lines.append("---")
    return "\n".join(lines)


def _format_trajectory(
    trajectory: Optional[list[dict]],
    posted_at_iso: Optional[str],
) -> Optional[str]:
    """Render a kinetic trajectory line, or None if it shouldn't render.

    Guards:
      * trajectory must have ``_TRAJECTORY_MIN_POINTS`` or more rows
      * posted_at must parse; without it we can't compute ``dN``
      * each snapshot is labelled ``dN`` where N = days since posted_at
        (floor); collisions on same N keep the latest snapshot
      * at most ``_TRAJECTORY_MAX_POINTS`` bucket labels are rendered;
        oldest points are dropped when over the cap (final datapoint
        is always kept — it's the endpoint the model can correlate
        against the main ENGAGEMENT line)

    Format: ``TRAJECTORY: d0: 8/2  d1: 18/5  d3: 23/6`` where the
    values are ``<reactions>/<comments>``. Reposts are omitted; they
    are usually small and cluttering.
    """
    if not trajectory or len(trajectory) < _TRAJECTORY_MIN_POINTS:
        return None
    if not posted_at_iso:
        return None
    try:
        posted_at = datetime.fromisoformat(
            str(posted_at_iso).replace("Z", "+00:00")
        )
    except Exception:
        return None

    # Bucket by "days since post." Keep the LATEST snapshot within
    # each day-bucket; model doesn't need two datapoints on d0.
    buckets: dict[int, dict] = {}
    for snap in trajectory:
        ts_iso = snap.get("scraped_at")
        if not ts_iso:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        except Exception:
            continue
        dt = ts - posted_at
        day = max(0, int(dt.total_seconds() // 86400))
        buckets[day] = snap  # latest wins (input is chronologically asc)

    if len(buckets) < _TRAJECTORY_MIN_POINTS:
        return None

    # Drop oldest points first if over the cap, preserving the latest
    # datapoint (it's the most actionable signal).
    days_sorted = sorted(buckets.keys())
    if len(days_sorted) > _TRAJECTORY_MAX_POINTS:
        # Keep: first point + (_MAX - 1) most recent
        days_sorted = [days_sorted[0]] + days_sorted[-(_TRAJECTORY_MAX_POINTS - 1):]
        days_sorted = sorted(set(days_sorted))

    parts: list[str] = []
    for d in days_sorted:
        snap = buckets[d]
        rx = snap.get("total_reactions") or 0
        cm = snap.get("total_comments") or 0
        parts.append(f"d{d}: {rx}/{cm}")
    return "TRAJECTORY: " + "  ".join(parts)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())
