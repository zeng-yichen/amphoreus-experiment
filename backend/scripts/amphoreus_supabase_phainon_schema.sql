-- creator_exemplars — Phainon's persistent memory of generated draft candidates
--
-- Phainon (named after the Deliverer in the Amphoreus arc) explores draft-
-- space for each tracked FOC and stores its top candidates here. Stelle's
-- post_bundle reads the latest batch when generating drafts so she sees
-- "directions Phainon explored that scored well" alongside the creator's
-- shipped history.
--
-- Three operating modes per FOC:
--   * 'full'        — N candidates generated, all scored by V2a reward,
--                     top-K kept. Use when reward Spearman ≥ 0.4 on
--                     ground-truth-era calibration.
--   * 'diverse'     — N candidates generated, NO scoring (reward is
--                     anti-predictive or near-zero for this FOC). All
--                     stored as "candidates Phainon explored." Stelle
--                     doesn't treat them as ranked exemplars.
--   * 'warm_start'  — Like diverse, but generated using the FOC's full
--                     pre+post history because their post-Virio era is
--                     too short for voice-only context. Carries a
--                     "voice may include pre-Virio tone" caveat.
--
-- Mode signals to the reader (Stelle, the operator) what level of
-- confidence to attach. Without this discriminator, the prototype would
-- silently surface bad rankings for some FOCs.
--
-- Rows are append-only — every weekly Phainon run inserts a fresh batch
-- per creator. Reads pull "the latest batch" via posted_at desc + LIMIT.
-- No deletion logic; old batches are kept as audit trail.
--
-- Storage envelope
-- ----------------
-- 6 prototype FOCs × ~20 exemplars/run × 1 run/week ≈ 120 rows/week.
-- Trivial.
--
-- Safe to re-run.

create table if not exists creator_exemplars (
    id                          uuid primary key default gen_random_uuid(),
    creator_handle              text not null,
    company_id                  uuid,
    company_label               text,
    onboarding_date             date,           -- ground-truth Virio onboarding
    mode                        text not null   -- 'full' | 'diverse' | 'warm_start'
                                check (mode in ('full', 'diverse', 'warm_start')),
    -- Generation provenance
    generated_at                timestamptz not null default now(),
    batch_id                    uuid not null,  -- one batch_id per phainon run per FOC
    angle                       text,           -- the angle prompt the candidate was generated under
    -- The candidate itself
    exemplar_text               text not null,
    -- Optional: scoring (only set when mode='full')
    predicted_band              int check (predicted_band is null or predicted_band between 1 and 5),
    reasoning                   text,
    rank_within_batch           int,            -- 1 = best by predicted_band, NULL when not ranked
    creator_post_count_at_gen   int,
    -- Cost provenance for budget tracking
    cost_usd                    numeric
);

create index if not exists creator_exemplars_handle_time_idx
    on creator_exemplars (creator_handle, generated_at desc);

create index if not exists creator_exemplars_batch_idx
    on creator_exemplars (batch_id);
