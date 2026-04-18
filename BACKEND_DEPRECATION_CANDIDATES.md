# Backend Deprecation Candidates

Status: mix of **completed removals** (see changelog at bottom) and **proposed only**.

The frontend has been pruned (Cyrene rebrand, three-box home page: Ghostwriter /
Interview Prep / Transcripts). Everything listed here is now **frontend-orphaned**:
no page in `frontend/src/app/**` references it, and no API client in
`frontend/src/lib/api.ts` calls into it.

The routers remain **registered** in `backend/src/main.py` so nothing breaks at
import / startup, and so we can revert the frontend trivially if we change our
minds. This is deliberate: per the user's directive, "don't delete anything in
the backend, just disconnect the frontend."

Before deleting anything here, confirm:
1. Stelle does not import from it. (Stelle pulls from `agents/analyst.py` and
   `agents/ruan_mei.py` — **DO NOT TOUCH THOSE**.)
2. No background task / cron / sync loop imports from it.
3. Audit log / Supabase writers do not call into it.

## Routers (safe to unregister from `main.py` first, then delete)

| File | What it does | Why it's orphaned |
| --- | --- | --- |
| `backend/src/api/routers/learning.py` | "Learning Intelligence" dashboard endpoints | Frontend page deleted (`/learning/**`). No other caller. |
| `backend/src/api/routers/desktop.py` | Launches the legacy Tkinter desktop GUI via subprocess | Frontend button removed. No other caller. |
| `backend/src/api/routers/strategy.py` | Generates content strategy via Herta | Frontend page deleted (`/strategy/**`). No other caller. |

The `briefings.py` router is **still used** — it's the Aglaea flow that now runs
inline inside the Interview Prep page (`/interview/{company}`). Do not delete.

## Agents (only delete after confirming Stelle does not import them)

| File | Role | Notes |
| --- | --- | --- |
| `backend/src/agents/herta.py` | Content strategy generator | Only called from `strategy.py` router + `services/market_intelligence.py`, `services/series_engine.py`, `evals/harness/runner.py`. **Stelle references "herta" only in comments at lines 1875/1881/1885/4176 of `stelle.py` — no imports.** Verify callers are also orphaned before deleting. |
| `backend/src/agents/herta_adapter.py` | Thin wrapper around Herta for router consumption | Dies with Herta. |

**DO NOT DELETE** (Stelle-critical, verified by reading `stelle.py` directly):
- `agents/analyst.py` — Stelle imports `_tool_query_observations`,
  `_tool_search_linkedin_bank`, `_tool_execute_python` at line 3419.
- `agents/ruan_mei.py` — Stelle imports `RuanMei as _RM` at line 4072 and calls
  `_rm_inst.analyze_post()` + `_rm_inst.record()` after every generated post
  (the observation write-back pipeline).

## Suggested deletion sequence (when ready)

1. Remove `learning`, `desktop`, `strategy` from the `from backend.src.api.routers
   import (...)` tuple in `backend/src/main.py`.
2. Remove the matching `app.include_router(...)` calls.
3. Deploy + smoke test the frontend. If nothing 500s or 404s, proceed.
4. Delete the three router files.
5. Confirm with `rg` that nothing imports `agents.herta` or `agents.herta_adapter`
   anywhere in the repo.
6. Delete `agents/herta.py` and `agents/herta_adapter.py`.
7. Drop any leftover prompt files under `backend/src/agents/herta_*` if present.

## Not in this list (intentionally)

- `cyrene.py` agent — name collision with `demiurge.py` (which exports a class
  also named `Cyrene` — the old SELF-REFINE critic). Resolve the naming
  separately; both agents are actively used.
- `aglaea.py` / `aglaea_adapter.py` — still powers briefing generation.
- `tribbie.py` / `tribbie_adapter.py` — still powers live interview companion.
- `phainon*` — actively used for image assembly via `imagesApi` in the
  ghostwriter frontend page.
- `castorice.py` — actively used; runs post-Stelle as the fact-checker +
  source annotator, citations published as Ordinal thread comments by
  Hyacinthia.
- `demiurge.py` — actively imported by `stelle_adapter.py`, `posts.py`,
  and `evals/harness/runner.py` as the SELF-REFINE critic-revise loop.
- `stelle*.py`, `analyst.py` (tombstone-but-still-exports-tool-primitives),
  `ruan_mei.py`, `hyacinthia.py` — Stelle-critical.
- `icp_simulator` — still referenced; audit separately.

## Changelog — completed removals

### 2026-04-17 — Orphaned research agents + Lola

Removed four agents whose backend routes were registered but no frontend
component ever called them, and whose removal was confirmed safe by a
full-repo grep sweep + `ast.parse` + OpenAPI spec diff.

| Removed | Replacement / notes |
| --- | --- |
| `backend/src/agents/lola.py` | No replacement. Was unimported anywhere in the repo. Its 24 `memory/{company}/lola_state.json` artifacts (one per client, ~2MB each) were also deleted. `test_workspace_ingestion.py::test_no_lola_state` was already asserting these files should not exist, so the deletion makes that test correct. |
| `backend/src/agents/anaxa.py` | No replacement. Was a 48-line wrapper around Gemini + Google Search, reached only via `POST /api/research/web` — no caller in any frontend component. For ad-hoc web research, use Claude Code's `WebSearch` or Gemini directly. |
| `backend/src/agents/cerydra.py` | No replacement. Was a document-grounded Q&A helper reached only via `POST /api/research/documents` — no caller. Cyrene and Stelle both read client documents directly without needing a Q&A router. |
| `backend/src/agents/hysilens.py` | No replacement. Snippet-source-identification helper reached only via `POST /api/research/source` — no caller. Castorice performs source annotation as part of its fact-check pass. |
| `backend/src/agents/mydei.py` | **Agent removed, artifacts preserved.** Mydei generated `memory/{company}/abm_profiles/mydei_briefing.md` files which Cyrene and Stelle still read. The existing briefings for innovocommerce, hume-andrew, hensley-biostats, terrafort stay on disk as static artifacts. No automated regeneration path now exists — if we need new briefings, re-introduce Mydei or port the logic elsewhere. |

Downstream removals in the same pass:

- `backend/src/api/routers/research.py` — entire router file removed; all four routes (`/web`, `/documents`, `/source`, `/abm`) gone.
- `backend/src/main.py` — `research` removed from the router import tuple and `app.include_router(research.router)` line deleted.
- `frontend/src/lib/api.ts` — `researchApi` export removed entirely (had 4 methods, all orphaned).
- `backend/src/skills/research/SKILL.md` — deleted along with the empty `skills/research/` directory.

Verification performed after removal:

- `ast.parse()` on every `.py` under `backend/src/` → 0 errors
- `grep` for dangling imports of `lola|anaxa|cerydra|hysilens|mydei` → 0 hits
- `grep` for `researchApi` or `/api/research/` across `frontend/src` + `backend/src` → 0 hits
- Live `GET /openapi.json` confirmed no `/api/research/*` routes registered
- Uvicorn `--reload` picked up changes without restart

Not yet removed (deliberate): `BACKEND_DEPRECATION_CANDIDATES.md`'s earlier
list of `herta`, `learning`, `desktop`, `strategy` — those are the next
pruning candidates but were out of scope for this pass.

### 2026-04-18 — Sync loops disabled + dead service modules removed + Lineage ingestion directives expanded

Stelle is now primarily invoked by Jacquard (Lineage platform), which
serves transcripts / engagement / context / reports via its own workspace
HTTP endpoints. The local Ordinal sync that fed `memory/{company}/` is no
longer the primary data path — it's demoted to opt-in legacy.

**Sync loops:** `backend/src/main.py` now defaults the hourly + 15-min
Ordinal sync loops to DISABLED. Set `ENABLE_SYNC_LOOPS=true` (was
previously `DISABLE_SYNC_LOOPS=true` to skip) to re-enable for legacy
local-mode runs. Flipped the default because Lineage-mode Stelle never
reads the resulting artifacts — the local `memory/` mirror was growing
stale with nobody consuming it. The sync code itself is untouched, so
manual invocations still work.

**Deleted (2 modules, 0 callers):**

| Removed | Why safe |
| --- | --- |
| `backend/src/services/topic_velocity.py` | Produced `memory/{company}/topic_velocity.md` via Perplexity, consumed by `services/temporal_orchestrator.py::should_accelerate_topic`. The consumer gracefully returns `False` when the file is missing — so deleting the writer just means acceleration never triggers. No live code path depended on the signal. |
| `backend/src/services/writer_productivity.py` | Writer-productivity metrics logger that wrote to a Supabase `writer_productivity_logs` table. Zero callers in backend. Table either doesn't exist anymore or is orphan. |

**Preserved (deliberate) — Stelle + Cyrene still import these during
local-mode runs:**

- `services/temporal_orchestrator.py` (5 callers: stelle, hyacinthia, posts, claude_cli, series_engine)
- `services/series_engine.py` (3 callers)
- `services/workspace_manager.py` (4 callers)
- `services/cross_client_learning.py`, `services/engager_fetcher.py`, `services/linkedin_bank.py` (1 caller each — likely dead but not verified)

**Stelle Lineage ingestion directives expanded** in
`backend/src/agents/stelle.py`:

- Added `reports/` (ICP report JSON + Typst template) to the read-path
  list in both USER_TARGETED and COMPANY_WIDE directive blocks
- Listed the specific engagement JSON files available (`posts.json`,
  `reactions.json`, `comments.json`, `profiles.json`,
  `work_experiences.json`, `client_info.json`) so Stelle knows what to
  expect
- Clarified `conversations/` purpose: `trigger-log.jsonl` is the replay
  of every prior interview / CE feedback diff / manual run for this
  company, chronological. Stelle now scans it at session start.
- Added explicit "IGNORE" note for `.pi/` (Jacquard-agent skill files,
  not ours)
- Clarified `posts/drafts/` is do-not-write-directly — use `submit_draft`
- Replaced the old "write_file into posts/drafts/content.md" workflow
  with a proper ingestion-order checklist (1: list root, 2: read trigger
  log, 3: per-slug strategy/edits/transcripts/engagement/reports reads,
  4: update strategy.md at end)
- Documented that `submit_draft` runs Castorice fact-check automatically
  before POST to Lineage, so reviewers see the fact-check report
  attached as a draft comment

**Also in this pass:**

- `backend/src/agents/stelle.py` — registered `get_reader_reaction`
  handler into the per-session `run_handlers` dict (was advertised to the
  LLM but missing from dispatch → `KeyError` on first flame-chase call)
- `backend/src/agents/stelle.py` — wrapped `submit_draft` dispatch with
  Castorice fact-check: every submitted post is fact-checked, corrected
  content replaces the original, and the fact-check report is appended
  to `why_post` so Lineage reviewers see it in the draft_comment thread

Verification performed after this pass:

- `ast.parse()` on every `.py` under `backend/src/` → 0 errors
- `grep` for dangling imports of `topic_velocity|writer_productivity`
  → 0 hits outside of `ordinal_sync.py` comments
- Stelle's advertised tools now all map to either `run_handlers` entries
  or the loop's special-cased `write_result` intercept — zero dispatch
  gaps
- Uvicorn `--reload` picked up changes without restart
