# Backend Deprecation Candidates

Status: **proposed only — nothing in this file has been deleted**.

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

- `cyrene.py` agent — name collision with the new brand, but may still be
  referenced. Leave alone and audit separately.
- `aglaea.py` / `aglaea_adapter.py` — still powers briefing generation.
- `tribbie.py` / `tribbie_adapter.py` — still powers live interview companion.
- All `stelle*.py`, `analyst.py`, `ruan_mei.py`, `phainon*`, `mydei`, `hysilens`,
  `castorice`, `cerydra`, `hyacinthia`, `anaxa`, `lola`, `icp_simulator` —
  assume Stelle-dependent until proven otherwise.
