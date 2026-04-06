"""Background Ordinal sync — RuanMei engagement data via Ordinal analytics only.

Does not write to Supabase; other modules may read from Supabase where needed.
"""

import csv
import logging
import threading
import time

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_SYNC_INTERVAL = 3600  # 1 hour default
_sync_thread: threading.Thread | None = None
_stop_event = threading.Event()

# Set to None to process all clients (original behavior).
# When set, only listed clients run through per-client steps 1-8.
# Steps 9-10 (cross-client learning, market intel) still read ALL client data from disk.
_ACTIVE_CLIENT_ALLOWLIST: set[str] | None = {
    "innovocommerce",
    "hensley-biostats",
    "trimble-mark",
    "trimble-heather",
    "hume-andrew",
}


def start_sync_loop(interval: int = _SYNC_INTERVAL) -> None:
    """Start the background Ordinal sync loop."""
    global _sync_thread
    if _sync_thread and _sync_thread.is_alive():
        logger.info("Ordinal sync already running")
        return

    _stop_event.clear()

    def _loop():
        while not _stop_event.is_set():
            try:
                sync_all_companies()
            except Exception:
                logger.exception("Ordinal sync cycle failed")
            _stop_event.wait(interval)

    _sync_thread = threading.Thread(target=_loop, daemon=True, name="ordinal-sync")
    _sync_thread.start()
    logger.info("Ordinal sync started (interval=%ds)", interval)


def stop_sync_loop() -> None:
    _stop_event.set()


def sync_all_companies() -> None:
    """Iterate ordinal_auth CSV and sync each company."""
    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.debug("No ordinal_auth_rows.csv found — skipping sync")
        return

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_key = row.get("api_key", "").strip()
                company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
                if api_key and company:
                    # Gate: skip per-client processing for inactive clients
                    if _ACTIVE_CLIENT_ALLOWLIST is not None and company not in _ACTIVE_CLIENT_ALLOWLIST:
                        logger.info("[sync] Skipping %s (not in active allowlist)", company)
                        continue

                    # RuanMei: ingest all posts + update pending observations.
                    profile_id = row.get("profile_id", "").strip()
                    if not profile_id:
                        profile_id = vortex.resolve_profile_id(company)
                    if profile_id:
                        analytics_posts = _fetch_ordinal_analytics(profile_id, api_key, company)
                        if analytics_posts:
                            try:
                                import asyncio
                                from backend.src.agents.ruan_mei import RuanMei
                                rm = RuanMei(company)

                                # 1. Update any pending Stelle-generated observations.
                                _update_ruan_mei_from_posts(rm, analytics_posts)

                                # 2. Ingest all Ordinal posts as scored observations.
                                #    Posts already in the history are skipped (deduped by hash).
                                try:
                                    _loop = asyncio.get_running_loop()
                                except RuntimeError:
                                    _loop = None
                                if _loop and _loop.is_running():
                                    import concurrent.futures
                                    ingested = concurrent.futures.ThreadPoolExecutor().submit(
                                        lambda: asyncio.run(rm.ingest_from_ordinal(analytics_posts))
                                    ).result(timeout=120)
                                else:
                                    ingested = asyncio.run(rm.ingest_from_ordinal(analytics_posts))
                                if ingested:
                                    logger.info("RuanMei ingested %d new posts for %s", ingested, company)

                                # 2a. Edit similarity backfill: fix observations where
                                #     post_body was set to live text by the old
                                #     ingest_from_ordinal path. Populates the draft-vs-live
                                #     diff signal the feedback distiller depends on.
                                #     Idempotent — skips observations already correct.
                                try:
                                    _backfilled = rm.backfill_edit_similarity_from_draft_map()
                                    if _backfilled:
                                        logger.info(
                                            "[ordinal_sync] Edit similarity backfilled on %d observations for %s",
                                            _backfilled, company,
                                        )
                                except Exception:
                                    logger.debug("Edit similarity backfill skipped for %s", company, exc_info=True)

                                # 2b. Quality embedding persistence: store (embedding, reward)
                                #     pairs for Cyrene's linear projection training.
                                #     Only embeds observations not already in the cache.
                                try:
                                    from backend.src.agents.cyrene import _persist_quality_embedding
                                    from backend.src.agents.lola import _embed_texts
                                    import json as _json2b
                                    _qe_path = vortex.memory_dir(company) / "quality_embeddings.json"
                                    _existing_hashes = set()
                                    if _qe_path.exists():
                                        try:
                                            _qe_data = _json2b.loads(_qe_path.read_text(encoding="utf-8"))
                                            _existing_hashes = {p.get("post_hash") for p in _qe_data.get("pairs", [])}
                                        except Exception:
                                            pass
                                    _new_obs = [
                                        obs for obs in rm._state.get("observations", [])
                                        if (obs.get("status") == "scored"
                                            and obs.get("descriptor", {}).get("analysis")
                                            and obs.get("reward", {}).get("immediate") is not None
                                            and obs.get("post_hash", "") not in _existing_hashes
                                            and obs.get("post_hash", ""))
                                    ]
                                    if _new_obs:
                                        _analyses = [o["descriptor"]["analysis"] for o in _new_obs]
                                        _embs = _embed_texts(_analyses)
                                        for obs, emb in zip(_new_obs, _embs or []):
                                            _persist_quality_embedding(
                                                company, emb,
                                                obs["reward"]["immediate"],
                                                obs["post_hash"],
                                            )
                                except Exception:
                                    logger.debug("Quality embedding persistence skipped for %s", company, exc_info=True)

                                # 2c. Cyrene adaptive config: recompute dimension weights
                                #     from newly scored observations with cyrene_dimensions.
                                try:
                                    from backend.src.agents.cyrene import CyreneAdaptiveConfig
                                    CyreneAdaptiveConfig().recompute(company)
                                except Exception:
                                    logger.debug("Cyrene adaptive config skipped for %s", company, exc_info=True)

                                # 2d. Depth weights: recompute learned depth component
                                #     weights from engagement correlations.
                                try:
                                    rm.recompute_depth_weights()
                                except Exception:
                                    logger.debug("Depth weight recompute skipped for %s", company, exc_info=True)

                                # 2e. Alignment thresholds: recompute learned strong/drift
                                #     thresholds from (alignment_score, reward) pairs.
                                try:
                                    from backend.src.utils.alignment_scorer import AlignmentAdaptiveConfig
                                    AlignmentAdaptiveConfig().recompute(company)
                                except Exception:
                                    logger.debug("Alignment threshold recompute skipped for %s", company, exc_info=True)

                                # 2f. Constitutional adaptive config: recompute principle
                                #     weights from (constitutional_results, reward) pairs.
                                try:
                                    from backend.src.utils.constitutional_verifier import ConstitutionalAdaptiveConfig
                                    ConstitutionalAdaptiveConfig().recompute(company)
                                except Exception:
                                    logger.debug("Constitutional config recompute skipped for %s", company, exc_info=True)

                                # 2g. Reward component weights: recompute from lagged
                                #     engagement correlations.
                                try:
                                    rm._get_reward_weights()  # forces recompute if obs grew
                                except Exception:
                                    logger.debug("Reward weights recompute skipped for %s", company, exc_info=True)

                                # 2h. ICP segment thresholds: recompute from engager score
                                #     distribution percentiles.
                                try:
                                    from backend.src.utils.icp_scorer import _get_segment_thresholds
                                    _get_segment_thresholds(company)  # recomputes + caches
                                except Exception:
                                    logger.debug("ICP threshold recompute skipped for %s", company, exc_info=True)

                                # 2i. CV thresholds: recompute constitutional verification
                                #     gating thresholds from cyrene_composite vs engagement.
                                try:
                                    from backend.src.agents.stelle import _compute_cv_thresholds
                                    _compute_cv_thresholds(company)
                                except Exception:
                                    logger.debug("CV threshold recompute skipped for %s", company, exc_info=True)

                                # 2j. Observation tagger (A1): extract topic_tag, source_segment_type,
                                #     format_tag from post bodies via Haiku. Foundational for
                                #     topic transitions, strategy brief, and causal filter.
                                try:
                                    from backend.src.utils.observation_tagger import backfill_client_tags
                                    _tagged = backfill_client_tags(company)
                                    if _tagged:
                                        logger.info(
                                            "[ordinal_sync] Tagged %d observations for %s",
                                            _tagged, company,
                                        )
                                except Exception:
                                    logger.debug("Observation tagging skipped for %s", company, exc_info=True)

                                # 2j2. Segment model: train embedding→predicted-reward projection
                                #      from tagged observations. Replaces the old 5-feature scorer
                                #      with a per-client learned model. Internally cache-gated on
                                #      observation count, so this is a no-op when no new data.
                                try:
                                    from backend.src.utils.transcript_scorer import build_segment_model
                                    _seg_model = build_segment_model(company)
                                    if _seg_model:
                                        logger.info(
                                            "[ordinal_sync] Segment model for %s: n=%d, LOO R²=%.4f",
                                            company,
                                            _seg_model.get("observation_count", 0),
                                            _seg_model.get("loo_r_squared", 0),
                                        )
                                except Exception:
                                    logger.debug("Segment model build skipped for %s", company, exc_info=True)

                                # 2k. Client profile: extract cross-client learning profile
                                #     vector for similarity-based cold-start seeding.
                                try:
                                    from backend.src.utils.cross_client import build_client_profile
                                    _profile = build_client_profile(company)
                                    if _profile:
                                        logger.info(
                                            "[ordinal_sync] Client profile built for %s: %d obs",
                                            company,
                                            _profile.get("observation_count", 0),
                                        )
                                except Exception:
                                    logger.debug("Client profile build skipped for %s", company, exc_info=True)

                                # 2k2. Feedback distiller: extract writing directives from
                                #      editorial feedback, accepted posts, and engagement
                                #      patterns. Kept because the analyst produces analytical
                                #      findings, not prescriptive writing rules. The distiller
                                #      converts "client always adds ClinOps terms" into a rule
                                #      Stelle must follow. The analyst says "customer voice
                                #      hooks outperform" — descriptive, not directive. Both
                                #      are needed.
                                try:
                                    from backend.src.utils.feedback_distiller import distill_directives
                                    _directives = distill_directives(company)
                                    if _directives:
                                        logger.info(
                                            "[ordinal_sync] Distilled %d writing directives for %s",
                                            len(_directives), company,
                                        )
                                except Exception:
                                    logger.debug("Feedback distillation skipped for %s", company, exc_info=True)

                                # 2l. Analyst agent: hypothesis-driven engagement analysis.
                                #     Replaces the fixed analysis pipeline (steps 2k-2p in the
                                #     old architecture: topic transitions, causal filter,
                                #     engagement predictor, feedback distiller, directive
                                #     efficacy, strategy brief). The analyst has access to the
                                #     same statistical primitives as tools and decides what to
                                #     run based on what it finds.
                                #
                                #     Weekly-gated: only runs if no findings file exists or the
                                #     existing one is >7 days old. At ~$2-3 per client per run,
                                #     daily would cost ~$50/day for 22 clients.
                                try:
                                    _analyst_path = vortex.memory_dir(company) / "analyst_findings.json"
                                    _run_analyst = True
                                    if _analyst_path.exists():
                                        try:
                                            import json as _json_analyst
                                            _af = _json_analyst.loads(_analyst_path.read_text(encoding="utf-8"))
                                            _last_run = _af.get("runs", [{}])[-1] if _af.get("runs") else {}
                                            _last_ts = _last_run.get("timestamp", "")
                                            if _last_ts:
                                                from datetime import datetime as _dt_analyst, timezone as _tz_analyst
                                                _age = (_dt_analyst.now(_tz_analyst.utc) - _dt_analyst.fromisoformat(
                                                    _last_ts.replace("Z", "+00:00")
                                                )).total_seconds() / 86400
                                                if _age < 7:
                                                    _run_analyst = False
                                                    logger.debug(
                                                        "[ordinal_sync] Analyst skipped for %s (last run %.1f days ago)",
                                                        company, _age,
                                                    )
                                        except Exception:
                                            pass
                                    if _run_analyst:
                                        from backend.src.agents.analyst import run_analysis
                                        _analyst_result = run_analysis(company)
                                        if _analyst_result and not _analyst_result.get("error"):
                                            logger.info(
                                                "[ordinal_sync] Analyst for %s: %d tool calls, "
                                                "%d findings, %d turns, %.1fs",
                                                company,
                                                _analyst_result.get("tool_calls", 0),
                                                _analyst_result.get("findings_stored", 0),
                                                _analyst_result.get("turns", 0),
                                                _analyst_result.get("elapsed_seconds", 0),
                                            )
                                        elif _analyst_result and _analyst_result.get("error"):
                                            logger.debug(
                                                "[ordinal_sync] Analyst skipped for %s: %s",
                                                company, _analyst_result["error"],
                                            )
                                except Exception:
                                    logger.debug("Analyst skipped for %s", company, exc_info=True)

                                # 3. ICP auto-generation — create definition if missing.
                                try:
                                    from backend.src.services.icp_generator import generate_icp_definition
                                    generate_icp_definition(company)
                                except Exception:
                                    logger.debug("ICP generation skipped for %s", company)

                                # 4. ICP scoring — fetch engager profiles for posts that
                                #    don't have an ICP reward yet.
                                _run_icp_scoring(rm, company)

                                # 5. Pinecone embedding — keep semantic index fresh.
                                try:
                                    from backend.src.services.linkedin_bank import embed_posts_to_pinecone
                                    embed_posts_to_pinecone(analytics_posts, company=company)
                                except Exception:
                                    logger.debug("Pinecone embedding skipped for %s", company)

                                # 6. Topic velocity — refresh industry signal.
                                try:
                                    from backend.src.services.topic_velocity import refresh_topic_velocity
                                    refresh_topic_velocity(company)
                                except Exception:
                                    logger.debug("Topic velocity refresh skipped for %s", company)

                                # 7. LOLA: update bandit arm rewards from newly scored
                                #    RuanMei observations.  Must run after step 1 so the
                                #    rewards are already z-scored and available.
                                try:
                                    from backend.src.agents.lola import LOLA
                                    _lola = LOLA(company)
                                    _lola_updated = _lola.update_from_ruan_mei()
                                    if _lola_updated:
                                        logger.info(
                                            "[ordinal_sync] LOLA updated %d arm rewards for %s",
                                            _lola_updated, company,
                                        )
                                except Exception:
                                    logger.debug("LOLA update skipped for %s", company, exc_info=True)

                                # 8. Series Engine: update series post scores from RuanMei,
                                #    then check series health for wrap/extend signals.
                                try:
                                    from backend.src.services.series_engine import (
                                        update_series_from_ruan_mei as _series_update,
                                        check_series_health as _series_health,
                                    )
                                    _series_scored = _series_update(company)
                                    if _series_scored:
                                        logger.info(
                                            "[ordinal_sync] Series engine updated %d posts for %s",
                                            _series_scored, company,
                                        )
                                    _series_changes = _series_health(company)
                                    for sc in _series_changes:
                                        logger.info(
                                            "[ordinal_sync] Series '%s' %s → %s (trend: %s) for %s",
                                            sc["theme"], sc["old_status"], sc["new_status"],
                                            sc["trend"], company,
                                        )
                                except Exception:
                                    logger.debug("Series engine update skipped for %s", company, exc_info=True)
                            except Exception:
                                logger.exception("RuanMei sync failed for %s", company)
    except Exception:
        logger.exception("Failed to read ordinal auth CSV")

    # 9. Cross-client learning: refresh universal patterns and hook library.
    #    Runs once per sync cycle (not per-company) since it aggregates all clients.
    try:
        from backend.src.services.cross_client_learning import run_cross_client_sync
        ccl_result = run_cross_client_sync()
        if ccl_result.get("patterns") or ccl_result.get("hooks") or ccl_result.get("seeded_clients"):
            logger.info(
                "[ordinal_sync] Cross-client learning: %d patterns, %d hooks, %d clients seeded",
                ccl_result.get("patterns", 0),
                ccl_result.get("hooks", 0),
                len(ccl_result.get("seeded_clients", [])),
            )
    except Exception:
        logger.debug("Cross-client learning skipped", exc_info=True)

    # 9a. Cross-client profile aggregation: update universal structural
    #     patterns from all client profiles (complements step 9's LLM patterns).
    try:
        from backend.src.utils.cross_client import update_universal_patterns
        _structural = update_universal_patterns()
        if _structural:
            logger.info(
                "[ordinal_sync] Cross-client structural patterns: %d patterns updated",
                len(_structural),
            )
    except Exception:
        logger.debug("Cross-client structural patterns skipped", exc_info=True)

    # 9b. Feedback learning happens through RuanMei observations
    # (draft → final → engagement). No separate feedback file processing needed.

    # 9c. Constitutional principle discovery (weekly).
    try:
        from backend.src.utils.constitutional_verifier import _discover_principles
        discovered = _discover_principles()
        if discovered:
            logger.info("[ordinal_sync] Constitutional: discovered %d principles", len(discovered))
    except Exception:
        logger.debug("Constitutional principle discovery skipped", exc_info=True)

    # 10. Market intelligence: weekly creator scraping + signal extraction.
    #     Frequency-gated internally (skips if last collection <6 days ago).
    #     Also auto-seeds vertical mappings for unmapped clients and
    #     auto-triggers strategy refreshes on high-velocity trending topics.
    try:
        from backend.src.services.market_intelligence import run_market_intel_cycle
        mi_result = run_market_intel_cycle()
        if mi_result.get("verticals_processed"):
            logger.info(
                "[ordinal_sync] Market intel: %d verticals processed, %d strategy triggers",
                len(mi_result["verticals_processed"]),
                len(mi_result.get("strategy_triggers", [])),
            )
        if mi_result.get("auto_seeded"):
            logger.info(
                "[ordinal_sync] Market intel auto-seeded: %s",
                ", ".join(mi_result["auto_seeded"]),
            )
    except Exception:
        logger.debug("Market intelligence skipped", exc_info=True)


def _fetch_ordinal_analytics(profile_id: str, api_key: str, company: str) -> list[dict] | None:
    """Fetch post analytics from Ordinal once. Returns parsed post list, or None on failure."""
    import httpx
    from datetime import datetime, timedelta, timezone

    base = "https://app.tryordinal.com/api/v1"
    start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = httpx.get(
            f"{base}/analytics/linkedin/{profile_id}/posts",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            params={"startDate": start, "endDate": end},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Ordinal analytics fetch failed for %s: %s", company, e)
        return None

    posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
    if not posts:
        logger.info("No analytics posts returned from Ordinal for %s", company)
        return []

    logger.info("Fetched %d analytics posts from Ordinal for %s", len(posts), company)
    return posts


def _extract_ordinal_post_id(post: dict) -> str:
    """Extract the Ordinal workspace post id from a LinkedIn analytics payload.

    The documented field is ``ordinalPost.id`` — present when the post was
    published through Ordinal, null otherwise.
    """
    op = post.get("ordinalPost")
    if isinstance(op, dict):
        v = op.get("id")
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _extract_linkedin_url(post: dict) -> str:
    """Extract the live LinkedIn post URL from an Ordinal analytics payload."""
    return (
        post.get("url")
        or post.get("linkedInUrl")
        or post.get("linkedin_url")
        or post.get("postUrl")
        or ""
    )


def _update_ruan_mei_from_posts(rm, analytics_posts: list[dict]) -> None:
    """Push pre-fetched Ordinal analytics into RuanMei for pending observations.

    Prefer matching by Ordinal workspace post id (set at push time) so copy can change on LinkedIn.
    Fall back to hashing live post text for older observations without a linked id.
    """
    updated = 0

    for post in analytics_posts:
        text = (
            post.get("commentary") or post.get("text") or post.get("copy")
            or post.get("content") or post.get("post_text") or ""
        ).strip()
        if not text:
            continue

        impressions = post.get("impressionCount") or post.get("impressions") or 0
        reactions = post.get("likeCount") or post.get("reactions") or post.get("total_reactions") or 0
        comments = post.get("commentCount") or post.get("comments") or post.get("total_comments") or 0
        reposts = post.get("shareCount") or post.get("repostCount") or post.get("reposts") or 0

        if impressions == 0:
            continue

        posted_at = post.get("publishedAt") or post.get("postedAt") or post.get("published_at") or ""
        linkedin_url = _extract_linkedin_url(post)

        metrics = {
            "impressions": impressions,
            "reactions": reactions,
            "comments": comments,
            "reposts": reposts,
            "posted_at": posted_at,
        }

        oid = _extract_ordinal_post_id(post)
        matched = False
        if oid:
            matched = rm.update_by_ordinal_post_id(oid, metrics, posted_body=text, linkedin_post_url=linkedin_url)
        if not matched:
            matched = rm.update_by_text(
                text, metrics,
                ordinal_post_id=oid,
                linkedin_post_url=linkedin_url,
            )
        if matched:
            updated += 1

    if updated:
        logger.info("[RuanMei] Updated %d observations for %s", updated, rm.company)

    # Backfill generation metadata from draft_map onto observations
    _backfill_generation_metadata(rm)


def _backfill_generation_metadata(rm) -> int:
    """Attach generation-time quality data from draft_map.json to observations.

    This closes the persistence gap: Stelle stores cyrene_dimensions etc.
    in draft_map at push time, ordinal_sync picks them up here when scoring.
    """
    import json as _json
    dm_path = vortex.draft_map_path(rm.company)
    if not dm_path.exists():
        return 0

    try:
        draft_map = _json.loads(dm_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    backfilled = 0
    fields_to_copy = (
        "cyrene_composite", "cyrene_dimensions", "cyrene_dimension_set",
        "cyrene_iterations", "cyrene_weights_tier",
        "constitutional_results", "alignment_score",
    )

    for obs in rm._state.get("observations", []):
        # Skip if already has Cyrene quality data
        if obs.get("cyrene_composite") is not None:
            continue

        oid = obs.get("ordinal_post_id", "")
        if not oid or oid not in draft_map:
            continue

        entry = draft_map[oid]
        changed = False
        for field in fields_to_copy:
            val = entry.get(field)
            if val is not None and obs.get(field) is None:
                obs[field] = val
                changed = True

        if changed:
            backfilled += 1

    if backfilled:
        rm._save()
        logger.info("[ordinal_sync] Backfilled generation metadata on %d observations for %s",
                     backfilled, rm.company)

    return backfilled


# Minimum reactions for ICP scoring to be worthwhile.
# Posts with fewer engagers produce noisy scores (sample too small).
# 47 of 262 posts (18%) have <10 reactions — skipping saves ~18% of Apify cost
# with zero quality loss (3-8 headlines can't reliably represent an audience).
_MIN_REACTIONS_FOR_ICP = 10

# ICP scoring is now gated by _ACTIVE_CLIENT_ALLOWLIST (per-client loop gate).
# No separate allowlist needed.


def _run_icp_scoring(rm, company: str) -> None:
    """For every scored observation that has a LinkedIn URL but no ICP reward, fetch engagers and score.

    This runs after the main sync so engagement data is always current first.
    Only processes posts where engagers haven't been fetched yet (idempotent).

    Cost optimisations (total: ~87% reduction vs naive approach):
      - Skip posts with <10 reactions (noisy sample, not worth fetching)
      - Reactions only, no comments actor (reactors are 83% of engagers)
      - Capped at 30 results per post (diminishing returns past ~20)
    """
    try:
        from backend.src.services.engager_fetcher import fetch_and_persist
        from backend.src.utils.icp_scorer import score_engagers, score_engagers_segmented
    except ImportError as e:
        logger.warning("[ordinal_sync] ICP scoring imports failed: %s", e)
        return

    scored_obs = [
        o for o in rm._state.get("observations", [])
        if o.get("status") == "scored"
        and o.get("linkedin_post_url")
        and o.get("ordinal_post_id")
        and o.get("reward", {}).get("icp_reward") is None
    ]

    if not scored_obs:
        return

    # Filter out low-engagement posts — not enough engagers for meaningful ICP signal.
    scoreable = []
    skipped_low = 0
    for obs in scored_obs:
        raw = obs.get("reward", {}).get("raw_metrics", {})
        reactions = raw.get("reactions", 0)
        if reactions < _MIN_REACTIONS_FOR_ICP:
            skipped_low += 1
            continue
        scoreable.append(obs)

    if skipped_low:
        logger.info(
            "[ordinal_sync] Skipped %d posts with <%d reactions for ICP scoring (%s)",
            skipped_low, _MIN_REACTIONS_FOR_ICP, company,
        )

    if not scoreable:
        return

    logger.info("[ordinal_sync] Running ICP scoring for %d posts (%s)", len(scoreable), company)

    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 3

    for obs in scoreable:
        oid = obs["ordinal_post_id"]
        url = obs["linkedin_post_url"]
        try:
            engager_profiles = fetch_and_persist(company, oid, url)
            if engager_profiles is None:
                continue
            if not engager_profiles:
                consecutive_empty += 1
                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                    logger.warning(
                        "[ordinal_sync] %d consecutive empty engager fetches for %s — "
                        "likely auth issue, skipping remaining ICP scoring",
                        consecutive_empty, company,
                    )
                    break
                continue
            consecutive_empty = 0
            segmented = score_engagers_segmented(company, engager_profiles)
            icp_score = segmented["score"]
            # Map to legacy [-1, 1] range for RuanMei composite
            legacy_score = icp_score if icp_score >= 0 else icp_score * 2
            rm.update_icp_reward(
                oid, legacy_score,
                linkedin_post_url=url,
                icp_segments=segmented.get("segments"),
                icp_match_rate=segmented.get("icp_match_rate"),
            )
        except Exception:
            logger.exception("[ordinal_sync] ICP scoring failed for post %s (%s)", oid[:12], company)
