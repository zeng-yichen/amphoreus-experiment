-- Addendum to amphoreus_supabase_mirror_schema.sql — tables the agent
-- pipeline ingests but which weren't in the initial schema pass.
--
-- Kept as a separate file so the main schema can evolve independently;
-- both files are idempotent (every table/index uses IF NOT EXISTS).
--
-- Tables added here are pulled by jacquard_direct.py:
--   * linkedin_reactions  — Cyrene's reactor analysis + Irontomb grounding
--   * linkedin_comments   — Cyrene's comment analysis
--   * context_files       — Stelle's account context (PDFs etc. uploaded
--                           to Jacquard)
--
-- Same conventions as the main schema: _source / _synced_at bookkeeping,
-- PKs match Jacquard's PKs, no cross-table FKs, no enum types mirrored.

create table if not exists linkedin_reactions (
    id                      bigint primary key,
    reacted_at              timestamptz,
    provider_profile_urn    text,
    provider_post_urn       text,
    type                    text,
    created_at              timestamptz,
    reactor_name            text,
    reactor_headline        text,
    _source                 text default 'jacquard',
    _synced_at              timestamptz default now()
);
create index if not exists linkedin_reactions_post_urn_idx
    on linkedin_reactions (provider_post_urn);
create index if not exists linkedin_reactions_profile_urn_idx
    on linkedin_reactions (provider_profile_urn);


create table if not exists linkedin_comments (
    id                      bigint primary key,
    provider_profile_urn    text,
    provider_post_urn       text,
    text                    text,
    comment_url             text,
    commented_at            timestamptz,
    created_at              timestamptz,
    parent_comment_url      text,
    search_vector           tsvector,
    commenter_name          text,
    commenter_headline      text,
    _source                 text default 'jacquard',
    _synced_at              timestamptz default now()
);
create index if not exists linkedin_comments_post_urn_idx
    on linkedin_comments (provider_post_urn);
create index if not exists linkedin_comments_profile_urn_idx
    on linkedin_comments (provider_profile_urn);


-- Account-level context (PDFs, decks, etc. that operators upload to
-- attach to a company). Body lives in GCS; the mirror stores
-- extracted_text inline for Stelle to read without hitting GCS.
create table if not exists context_files (
    id                  uuid primary key,
    company_id          uuid,
    uploaded_by         uuid,
    filename            text,
    gcs_url             text,
    extracted_text      text,
    content_type        text,
    size_bytes          bigint,
    created_at          timestamptz,
    _source             text default 'jacquard',
    _synced_at          timestamptz default now()
);
create index if not exists context_files_company_idx
    on context_files (company_id);


-- ---------------------------------------------------------------- sync state

-- Per-(table, scope) watermark so incremental syncs only pull the
-- delta since the last successful run. Populated by
-- backend/src/services/jacquard_mirror_sync.py.
--
-- ``scope`` is either 'global' (sync pulled the whole table) or a
-- company UUID string (on-demand sync scoped to one client). We keep
-- separate watermarks per scope so a global sync doesn't clobber
-- an on-demand sync's progress and vice versa.
create table if not exists mirror_sync_state (
    table_name   text not null,
    scope        text not null default 'global',
    cursor_value text,
    updated_at   timestamptz default now(),
    primary key (table_name, scope)
);


-- ---------------------------------------------------------------- storage bookkeeping

-- ``_storage_path`` on meetings is the Supabase Storage path where
-- the mirror placed a copy of the transcript body (original lives in
-- Jacquard's GCS). Populated by the transcript-copy step of
-- jacquard_mirror_sync. ``NULL`` means we haven't copied it yet —
-- the sync worker uses that as the work queue.
--
-- Safe to re-run: column add is idempotent. Requires the main
-- schema to have already created the `meetings` table.
alter table if exists meetings
    add column if not exists _storage_path text;


-- ---------------------------------------------------------------- per-FOC context scoping

-- ``context_files`` historically had only ``company_id`` — so every FOC
-- at a shared-slug company (e.g. Trimble's heather + mark, Commenda's
-- logan + sam) saw the same context blob pool. Adding ``user_id`` lets
-- Amphoreus-uploaded context (``_source='amphoreus'``) be tied to a
-- specific FOC. Jacquard-mirrored rows stay NULL (Jacquard has no
-- per-FOC column upstream either) and are treated as company-wide in
-- read helpers for backward compatibility.
--
-- Read contract: ``get_client_context_files(company, user_id=X)``
-- returns rows where ``company_id=company AND (user_id=X OR user_id IS NULL)``.
-- Company-wide rows still show; user-specific rows only show to that user.
--
-- Safe to re-run: ``add column if not exists`` + ``create index if not exists``.
alter table if exists context_files
    add column if not exists user_id uuid;

create index if not exists context_files_company_user_idx
    on context_files (company_id, user_id)
    where user_id is not null;


-- ---------------------------------------------------------------- ordinal_auth (Jacquard mirror)

-- Jacquard's ``ordinal_auth`` table is the canonical source of Ordinal
-- API keys, keyed by ``user_companies.id`` (one key per client).
-- Historically Amphoreus read these from a flat CSV on the Fly volume
-- at ``/data/memory/ordinal_auth_rows.csv`` — a one-time export that
-- drifted out of date as Jacquard added new clients. The 2026-04-22
-- investigation found Jacquard had 36 rows, the CSV had 19 → 17
-- clients couldn't push drafts to Ordinal despite having valid keys
-- upstream.
--
-- This table mirrors Jacquard's ``ordinal_auth`` (upsert-on-sync, keyed
-- by ``company_id``). Columns matching Jacquard's schema carry
-- ``_source='jacquard'``. The ``profile_id`` column is an Amphoreus-
-- side enrichment (Ordinal's LinkedIn scheduling profile UUID, resolved
-- on-demand by ``vortex.resolve_profile_id``) — kept out of the sync
-- payload so subsequent sync runs don't clobber it.
--
-- No ``updated_at`` / ``created_at`` columns upstream, so the sync does
-- a full fetch + upsert every run. Cheap: 36 rows × 3 columns × 1 KB.
create table if not exists ordinal_auth (
    company_id         uuid primary key,
    api_key            text,
    provider_org_slug  text,
    profile_id         text,            -- Amphoreus enrichment, NOT in Jacquard
    _source            text default 'jacquard',
    _synced_at         timestamptz default now()
);
create index if not exists ordinal_auth_slug_idx on ordinal_auth (provider_org_slug);


-- ---------------------------------------------------------------- local_posts: draft ↔ published semantic match-back
--
-- Columns used by backend/src/services/draft_match_worker.py (the v2
-- match-back path that runs inside the 8-hour jacquard_mirror_cron,
-- right after the mirror sync finishes). With Ordinal being retired,
-- the exact-id (ordinal_post_id → provider_urn) chain is breaking;
-- this column set lets us pair a Stelle draft to the LinkedIn post it
-- eventually became via cosine similarity between their embeddings.
--
-- Relationship to the older draft_publish_matches table (see
-- amphoreus_supabase_draft_match_schema.sql): that was the v1 design
-- driven by draft_publish_matcher.py, which stores embeddings on a
-- separate ``draft_embedding`` column and writes match rows into
-- ``draft_publish_matches``. v2 folds all of that onto local_posts
-- itself — less join, fewer moving parts, and a single row carries
-- both the draft content and its pairing outcome. The two systems
-- currently coexist during cutover; remove v1 once v2 has a few weeks
-- of clean runs.
--
-- text-embedding-3-small is 1536 dimensions and matches the model used
-- on the Jacquard-side ``post_embeddings`` table so that future
-- cross-retrieval (Stelle draft → nearest peer post) is cosine-
-- comparable without re-embedding.
--
-- Safe to re-run: every add-column + index uses IF NOT EXISTS.

-- Requires pgvector — already enabled for post_embeddings, but the
-- extension create is idempotent so stating it here keeps this file
-- self-contained in case someone runs it standalone.
create extension if not exists vector;

alter table if exists local_posts
    add column if not exists embedding             vector(1536);
alter table if exists local_posts
    add column if not exists matched_provider_urn  text;
alter table if exists local_posts
    add column if not exists matched_at            timestamptz;
alter table if exists local_posts
    add column if not exists match_similarity      real;
alter table if exists local_posts
    add column if not exists match_method          text
    check (match_method is null or match_method in ('semantic','ordinal_id'));

-- Hot query for the match-back worker: "which posts in this company
-- have been paired already?" (so we don't recompute each cron tick).
create index if not exists local_posts_matched_urn_idx
    on local_posts (matched_provider_urn)
    where matched_provider_urn is not null;

-- Inverse partial index: the backfill + match-back worker both need a
-- cheap "rows still needing embedding" scan.
create index if not exists local_posts_embedding_null_idx
    on local_posts (id)
    where embedding is null;

-- IVFFlat cosine index for cross-retrieval queries (Stelle draft →
-- nearest peer LinkedIn post). Lists=50 is the same build parameter
-- used on post_embeddings_embedding_ivfflat; it's tuned for a corpus
-- in the low-tens-of-thousands, which is local_posts' headroom. The
-- build can need ``maintenance_work_mem`` bumped on first run —
-- re-run with ``SET maintenance_work_mem = '512MB';`` in the same
-- psql session if it complains.
create index if not exists local_posts_embedding_ivfflat
    on local_posts using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);


-- ---------------------------------------------------------------- linkedin_posts: Amphoreus-owned scrape marker
--
-- Added by backend/src/services/amphoreus_linkedin_scrape.py, the
-- weekday-midnight cron that pulls engagement counts for our clients'
-- FOCs via Apify directly (independent of Jacquard's mirror).
--
-- ``_last_amphoreus_scraped_at`` is the timestamp of the most recent
-- Amphoreus-side scrape for this row, distinct from Jacquard's
-- ``_synced_at`` (last time the mirror worker touched it). Having both
-- lets us tell at a glance which leg of the data pipeline most recently
-- observed a post:
--
--   SELECT provider_urn,
--          _source,
--          _synced_at,
--          _last_amphoreus_scraped_at,
--          greatest(_synced_at, _last_amphoreus_scraped_at) AS fresh_at
--   FROM linkedin_posts WHERE ...
--
-- Rows never touched by the Amphoreus scrape (i.e. all historical rows,
-- plus any URN Amphoreus's profile-posts actor hasn't seen) carry
-- NULL here, which is what we want.
--
-- Safe to re-run.
alter table if exists linkedin_posts
    add column if not exists _last_amphoreus_scraped_at timestamptz;

create index if not exists linkedin_posts_amph_scraped_idx
    on linkedin_posts (_last_amphoreus_scraped_at desc)
    where _last_amphoreus_scraped_at is not null;


-- ---------------------------------------------------------------- draft_convergence_log
--
-- Per-critic-call telemetry. Every Irontomb get_reader_reaction and
-- Aglaea check_client_comfort run during Stelle generation appends one
-- row here. Written by backend/src/services/convergence_log.py;
-- backfilled with the eventual ``local_posts.id`` by
-- ``backfill_local_post_id`` when submit_draft lands.
--
-- This table was originally created in the Supabase console without a
-- repo-side definition. Declaring it here (idempotent) so the schema
-- is in source control and the indexes needed for fast reads exist.
--
-- The BL-adjacent guardrail: this table records what two LLM critics
-- SAID per iteration — it is process telemetry, not predictor-training
-- data. Analyses driven from this should describe the pipeline's
-- behaviour (iteration counts, time-to-convergence, critic
-- disagreement patterns) rather than train a scalar that feeds back
-- into generation. See convergence_log.py's module docstring.
--
-- Safe to re-run: ``IF NOT EXISTS`` on every object.

create table if not exists draft_convergence_log (
    id              uuid primary key,
    logged_at       timestamptz not null default now(),
    -- Null until backfill_local_post_id runs (at submit_draft time).
    local_post_id   text,
    -- 16-char prefix of sha256(draft_text). Correlation key across
    -- critic calls on the same draft body.
    draft_hash      text not null,
    critic          text not null check (critic in ('irontomb', 'aglaea')),
    -- Iteration index within a trajectory. Irontomb sets this from
    -- len(_prior_reactions)+1. Aglaea leaves it NULL — temporal
    -- ordering by logged_at reconstructs the sequence.
    iteration       int,
    -- Free-text critic output. Bounded at 1000/500 chars at the
    -- write site so rows stay small.
    reaction        text,
    anchor          text,
    elapsed_s       real,
    -- Aglaea-only: numeric score + structured spans.
    score           int,
    flagged_spans   jsonb
);

-- Fast backfill of whole trajectories once a draft ships.
create index if not exists draft_convergence_log_draft_hash_idx
    on draft_convergence_log (draft_hash, critic, logged_at);

-- "Show me the convergence trajectory for this published post" —
-- the primary read pattern once local_post_id is backfilled.
create index if not exists draft_convergence_log_local_post_idx
    on draft_convergence_log (local_post_id, critic, iteration)
    where local_post_id is not null;

-- Temporal sweeps for drift analysis + un-backfilled rows.
create index if not exists draft_convergence_log_logged_at_idx
    on draft_convergence_log (logged_at desc);

-- FK cascade migration. The table was originally created in the
-- Supabase console WITH a FK on ``local_post_id`` referencing
-- ``local_posts(id)`` but WITHOUT ``ON DELETE CASCADE``. That blocked
-- every Posts-tab delete for any post Stelle had generated (which is
-- all of them — every critic call writes a convergence row), producing
-- "delete button does nothing" symptoms on 2026-04-23. This block
-- drops the old constraint if present and re-adds it with cascade.
-- Idempotent: the DO-block checks pg_constraint before touching.
-- Safe to run multiple times.
do $$
begin
    if exists (
        select 1
        from pg_constraint
        where conname = 'draft_convergence_log_local_post_id_fkey'
          and conrelid = 'draft_convergence_log'::regclass
    ) then
        alter table draft_convergence_log
            drop constraint draft_convergence_log_local_post_id_fkey;
    end if;
    -- Re-add with cascade. Only adds if the column is actually a FK
    -- target — skips cleanly on fresh installs where the column is a
    -- plain text reference (the canonical create-table block above
    -- declares it without FK since we key on hash-matching rather
    -- than strict referential integrity).
    if exists (
        select 1
        from information_schema.columns
        where table_name = 'draft_convergence_log'
          and column_name = 'local_post_id'
    ) and exists (
        select 1
        from information_schema.columns
        where table_name = 'local_posts'
          and column_name = 'id'
    ) then
        alter table draft_convergence_log
            add constraint draft_convergence_log_local_post_id_fkey
            foreign key (local_post_id)
            references local_posts(id)
            on delete cascade;
    end if;
end $$;
