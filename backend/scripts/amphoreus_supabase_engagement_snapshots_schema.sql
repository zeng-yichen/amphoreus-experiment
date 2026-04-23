-- linkedin_post_engagement_snapshots
--
-- Time-series of engagement counts (reactions / comments / reposts)
-- observed on each scrape of a LinkedIn post. Written by BOTH scrape
-- legs so a downstream consumer can answer kinetic questions without
-- needing to re-query LinkedIn:
--
--   * How did this post's reactions evolve — frontloaded vs built slow?
--   * Has this post stabilized (counts flat over multiple days)?
--   * Do posts in a semantic cluster share a kinetic pattern?
--
-- The existing ``linkedin_posts`` table stores only the LATEST
-- snapshot — engagement updates are destructive overwrites. This
-- table captures the history that gets overwritten.
--
-- Writers
-- -------
--   * ``amphoreus_linkedin_scrape`` — on every insert + update (weekday
--     00:00 UTC)
--   * ``jacquard_mirror_sync``      — on every linkedin_posts upsert
--     (8h cadence)
--
-- Readers
-- -------
--   * ``post_bundle._fetch_engagement_trajectories`` — batch-fetches
--     last 14d of snapshots for posts currently being rendered, then
--     stamps a TRAJECTORY line on the <14d-old block's ENGAGEMENT
--     section so Stelle sees kinetics, not just endpoint counts.
--
-- Storage envelope
-- ----------------
-- ~30 FOCs × ~30 tracked posts × ~5 writes/post/week (Amphoreus
-- midnights + opportunistic Jacquard upserts) ≈ 4.5k rows/week ≈
-- 230k rows/year. Trivial at Postgres scale.
--
-- Duplicate-write protection
-- --------------------------
-- PK is (provider_urn, scraped_at). Two writers landing at the EXACT
-- same microsecond collapse to one row, which is the right behavior —
-- distinct wall-clock samples are the unit of interest, not distinct
-- writer identity.
--
-- Safe to re-run. No data in this table is load-bearing for anything
-- other than diagnostics + the trajectory render; losing it doesn't
-- break generation.

create table if not exists linkedin_post_engagement_snapshots (
    provider_urn      text        not null,
    scraped_at        timestamptz not null,
    total_reactions   int,
    total_comments    int,
    total_reposts     int,
    -- Which scrape leg wrote this row. Useful for diagnosing "did
    -- Jacquard see counts stabilize before Amphoreus did" questions.
    -- Nullable so old rows from a future change in writer set don't
    -- break the PK.
    scraped_by        text,
    primary key (provider_urn, scraped_at)
);

-- Hot-path read: "give me the last N snapshots for this URN (or set
-- of URNs)". DESC on scraped_at so ``order by scraped_at desc limit
-- 20`` is an index scan.
create index if not exists eng_snaps_urn_time_idx
    on linkedin_post_engagement_snapshots (provider_urn, scraped_at desc);
