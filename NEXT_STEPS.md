# Amphoreus: Next Steps

Written during a long architectural session on 2026-04-10. This document
captures everything we discussed, in priority order, so we have a shared
reference for what needs to happen next.

## Status (as of 2026-04-10 evening)

- ✅ **Strip complete.** Prescriptive injection layer removed from Stelle.
  Tools `web_research` and `subagent` removed. See "Priority #1" below.
- ✅ **4 new tools added to Stelle:** `query_observations`,
  `query_top_engagers`, `search_linkedin_corpus`, `execute_python`. All
  wired through `_run_agent_loop` with per-run dispatch binding the
  client context. See "Priority #2" below.
- ✅ **ICP Simulator Phase 1 complete.** New module at
  `backend/src/agents/icp_simulator.py`. Tool `simulate_icp_reaction` is
  live as Stelle's 14th tool. Uses Opus 4.6, 5 persona variants,
  forced tool_use for structured output. Live-tested on the mediocre
  "Ten vials of blood" post — simulator independently caught every
  failure mode we diagnosed by hand (hearsay framing, AI "it's not X,
  it's Y" constructions, generic vendor pitch pivot, vague closing
  epigram). Cost: ~$0.085 per call. See "Priority #3" below.
- ✅ **ICP Simulator is now a turn-based learning agent (2026-04-11).**
  Replaced the one-shot persona call with `ICPSimulatorAgent`, a stateful
  agent that spawns in a background thread when Stelle's run starts and
  runs in tandem with her. Phase A: uses its own tool set
  (`get_client_posts`, `get_reactor_profiles`, `note_pattern`, `mark_ready`)
  to study historical posts, reactor identities, and (draft, published)
  editorial deltas over up to 20 turns — distinguishing general audience
  from ICP audience via per-reactor `icp_score`, distinguishing
  high-engagement-but-wrong-audience posts from high-ICP-resonance posts.
  Phase B: reacts to Stelle's drafts using the accumulated persona model
  (summary + notes). Stelle's `simulate_icp_reaction` tool call blocks on
  Phase A the first time, then dispatches straight to Phase B for subsequent
  calls. Clean teardown via try/finally around Stelle's run loop. See
  `backend/src/agents/icp_simulator.py::ICPSimulatorAgent`.
- ✅ **edit_similarity dropped as a learning signal (2026-04-11).** Per
  user instruction: reducing a (draft, published) pair to a single scalar
  destroys information. The stored field is still there for legacy
  consumers but nothing the model reads (query_observations, the simulator
  calibration block, the agent's Phase A tool output) surfaces it anymore.
  Stelle and the simulator read both texts directly and see the exact diff.
- ✅ **Every call to the simulator still logs to simulator_predictions.jsonl**
  via `_log_prediction` — the agent's Phase B reactions append the same
  audit trail as the legacy one-shot path.
- ✅ **Stelle tool description tightened.** Three places carry "MANDATORY
  before every submission" framing: the tool list summary, the Process
  section (simulation is explicit step 4 between draft revision and save),
  and the tool schema description itself. The schema description now also
  explains what the simulator actually is — a turn-based agent that spawned
  in parallel and has been learning this specific client's audience.
- ⏳ **Cyrene not yet built.** Long-trajectory agent. Still theoretical.
- ⏳ **Edit capture bug not yet investigated.** `edit_similarity = 0` for
  most clients, meaning the draft→published delta isn't being tracked.

## Known pre-existing bugs surfaced during the strip

These were not caused by the strip — they were already present. Noted
here for visibility:

1. **Pinecone dependency mismatch.** `pinecone-client` package is
   installed in the ruanmei conda env with a shim `pinecone/__init__.py`
   that raises `Exception("The official Pinecone python package has
   been renamed from pinecone-client to pinecone")` on import. Effect:
   any fresh semantic search call (no cache) fails. RuanMei's
   `_query_linkedin_bank` works only because the 7-day cache is
   populated. Stelle's new `search_linkedin_corpus` tool in semantic
   mode will fail on cache miss. **Fix:** `pip uninstall pinecone-client
   && pip install pinecone` in the ruanmei env. Single command, blocks
   whenever you want to spend 30 seconds on it.

2. **Supabase keyword search 500.** The analyst's `_search_keyword`
   helper (used by Stelle's `search_linkedin_corpus` keyword mode) issues
   a Supabase PostgREST query with `or=(hook.ilike.*X*,post_text.ilike.*X*)`
   syntax that Supabase is rejecting with HTTP 500. Keyword mode is
   broken; semantic mode is a fine substitute. **Fix:** rewrite the
   `or=` clause or URL-encode the wildcards differently. Low priority
   since semantic mode works (pending #1).

## The core insight

After running the OG comparison on innovocommerce and reading Sachil's
posts side-by-side with and without RuanMei:

- **Stelle (Opus 4.6) is already strong enough to generate locally-optimal
  content given good source material.** The OG posts were markedly better
  than the RuanMei-guided ones.
- **Prescriptive pattern injection is net-negative at current model capability.**
  Forcing Stelle to follow structural patterns extracted from small datasets
  makes her fake specificity and produces AI-shaped content.
- **The real optimization problem is a long-trajectory one**: maximizing
  client pipeline outcomes (ICP exposure, meetings, thought leadership)
  over months, not per-post engagement in the current moment.
- **The per-post quality ceiling exists because Stelle has no fast feedback
  signal from her actual audience at draft time.** Turn-based gave her
  iteration; she needs something to iterate against.

## Architectural moves, in order of priority

### 1. Strip the prescriptive injection layer from Stelle ✅ DONE (2026-04-10)

The content landscape, content brief, RuanMei insights, hook library,
analyst context, and RuanMei-derived char/cadence overrides all get
deleted from Stelle's prompt. The OG comparison already showed that OG
mode (which has all of this stripped) produces better posts.

**Scope:**
- Delete `analyst_context`, `ruan_mei_insight_context`, `content_intel_context`,
  `hook_library_context`, `trajectory_context` blocks from `generate_one_shot`
- Delete `memory/strategy.md` write in `_setup_workspace`
- Delete `_build_strategy_slot_context` + `_build_content_brief_context` calls
  in `_build_dynamic_directives` (keeps the ABM/feedback/revisions static
  sections and the Herta strategy doc fallback)
- Delete `_build_exemplar_section` injection (from RuanMei state)
- Stop calling `generate_content_strategy()` from `ordinal_sync`
- Remove `topic_velocity.md` and `market_intelligence.md` pre-injection
  (Stelle can web_search for this when she wants)
- Remove the 2 redundant tools: `web_research` (duplicates web_search),
  `subagent` (redundant with Stelle's own reasoning)

**NOT touched:**
- Stelle's system prompt (identity, process, standards)
- Stelle's tools list (except the 2 removals above)
- Her workspace structure (transcripts, voice examples, ICP, profile)
- Castorice fact-check pipeline

### 2. Ensure Stelle has all the tools she needs ✅ DONE (2026-04-10)

**Tools to add (as new native tools, not pre-injection):**

- `query_observations` — fetch scored post history with full text, rewards,
  impressions, edit similarity. Currently locked in RuanMei state.
  Gives Stelle direct access to her own historical engagement data without
  a curation layer.
- `query_top_engagers` — fetch the ICP profiles of who actually engages
  with this client's posts (from `get_top_icp_engagers`). Currently not
  shown to Stelle at all. High-signal context for writing to real readers.
- `search_linkedin_corpus` — full access to the 200K+ LinkedIn bank via
  Pinecone + Supabase. Keyword and semantic modes. Already exists as
  `tools/query_posts.py` and `tools/semantic_search_posts.py` bash scripts;
  should be promoted to native tools for discoverability.
- `execute_python` — the same tool the analyst has. Lets Stelle run her
  own statistical analysis on the LinkedIn corpus or on her historical
  posts. Lower risk than the analyst because Stelle is querying, not
  committing findings. Useful for sanity-checking her own intuitions
  ("am I the only ghostwriter using 'It's not X. It's Y.' patterns?").

**Why these specifically:**
- All are genuinely non-curatable (Stelle couldn't get them via web_search or
  workspace files)
- None impose patterns on Stelle — they're raw data access
- They scale with model capability (smarter models extract more signal)

### 3. **ICP Persona Simulation** (the breakthrough) — ⏳ PHASE 1 DONE, 2+3 PENDING

**This is the core structural innovation.** Give Stelle a tool that
simulates how her target audience would react to a draft, at inference
time, before she commits to it.

**How it works:**
- Tool: `simulate_icp_reaction(draft_text, persona_variant)`
- Implementation: spawns a Sonnet call with a persona prompt derived from
  the client's ICP definition
- The persona "reads" the draft and responds authentically as an ICP member
  would: what's compelling, what's generic, would they scroll past, would
  they react, would they comment
- Stelle reads the simulated reaction, edits the draft, re-simulates, iterates
- She can call it with different persona variants (`default_icp`, `skeptical_icp`,
  `busy_icp`, `practitioner_icp`) to get multi-angle feedback

**Why it's the breakthrough:**
- Turn-based gave Stelle iteration. ICP simulation gives her **a target to
  iterate against**.
- It closes the inference-time feedback loop between draft and audience
  reaction, using base model capability on both sides.
- It's AlphaGo self-play applied to content: instead of waiting for real
  engagement to learn, simulate the audience and iterate now.
- Bitter Lesson compliant: no hand-engineered rules, just a persona prompt
  + model capability on both sides.
- Compounds with model improvements: every Claude release makes the
  simulator sharper without changing the architecture.
- Hard for competitors to think of and hard to replicate well because it
  depends on high-quality ICP definitions and persona calibration.
- Something a human would come up with (it's exactly what a good ghostwriter
  does internally: "would my reader care?").
- Works on the first run, not after months of data accumulation.

**Calibration plan:**
- For each client, backtest the simulator on their historical posts where
  real engagement is known. Measure whether the simulator's predicted
  reaction matches actual engagement direction.
- Tune persona prompts based on calibration drift.
- Over months, each client's simulator gets sharper. Competitors would
  need the same historical data + iteration cycles to match.

**Cost:** ~$0.02 per simulation call. Typical run: 6 posts × 2-3 simulations
each = ~$0.36. Negligible.

### 4. Rebuild Cyrene as a turn-based trajectory agent

**Cyrene's role under the new framing:** she's the active learning policy
for preference acquisition across the long client relationship. She does
NOT produce per-post advice. She produces **structured requests for what
to extract from the client next**.

**Architecture:**
- Turn-based agent, same template as the new analyst and the turn-based
  RuanMei we built today
- Tools: `query_observations`, `read_transcripts`, `read_published_posts`,
  `read_analyst_findings`, `read_previous_acquisition_briefs`,
  `query_cross_client`, `search_linkedin_corpus`, `web_search`,
  `submit_acquisition_brief`
- Output: a structured acquisition brief with prioritized requests:
  - `what_i_need`: the specific story/opinion/position to extract
  - `why_it_matters`: strategic rationale tied to trajectory
  - `expected_use`: how this would change the content
  - `urgency`: high/medium/low
  - `interview_question`: exact phrasing to use with the client
- Also outputs: `trajectory_assessment` (where the client is in their
  long-term journey) and `stelle_directives` (strategic direction that
  isn't acquisition-dependent)

**How the loop closes:**
1. Cyrene runs (weekly or before each interview)
2. Cyrene outputs acquisition brief
3. Human executes the brief — runs interview, asks the questions, captures
   responses
4. New transcripts/material enter the workspace
5. Stelle generates posts from updated raw material
6. Posts published, engagement scored
7. Cyrene runs again, sees new state, computes next gradient step

The human is the I/O layer, not the strategist. Cyrene does the strategic
thinking. Stelle does the writing.

## What each agent sees today (data access asymmetry)

After the strip + new tools + ICP Simulator Phase 1, here's the current
state of data access per agent:

| Data source | Stelle | ICP Simulator (Phase 1) |
|---|---|---|
| Transcripts | ✅ via `read_file` | ❌ |
| Voice examples (top accepted posts) | ✅ via `read_file` | ❌ |
| ICP definition | ✅ via `read_file` | ✅ injected into persona prompt |
| LinkedIn profile | ✅ via `read_file` | ❌ |
| Scored observations (draft/published/engagement) | ✅ via `query_observations` | ❌ |
| Top engagers (ICP audience composition) | ✅ via `query_top_engagers` | ❌ |
| LinkedIn bank (200K+ posts) | ✅ via `search_linkedin_corpus` | ❌ |
| Python runtime on scored obs | ✅ via `execute_python` | ❌ |

**The simulator is currently blind** — it runs on the persona prompt +
ICP + draft only. No historical context, no calibration examples, no
awareness of what has or hasn't worked for this specific client.

This is Phase 1 by design — the simulator is valuable immediately using
Sonnet's base priors about how busy professionals react to LinkedIn
content. But Phase 2 + 3 fix this asymmetry by feeding the simulator
historical (draft, published, engagement) triples as few-shot calibration
examples on every call.

## Simulator model choice

**Decided 2026-04-10: Opus 4.6.** Rationale:

- The simulator is the critical feedback loop — any marginal quality
  improvement compounds across every Stelle iteration against it
- Opus may catch subtler failures Sonnet misses
- Cost delta (~$3-4/run at 54 calls) is inside the noise relative to
  the $10+ Stelle generation cost
- We'd rather pay for sharper audience reactions than save on a
  subsystem whose whole job is to steer the main agent

Trade-offs accepted:
- Higher latency per call (5-7s vs 2-3s for Sonnet)
- Uses Opus capacity alongside the main Stelle generation

If this turns out to be wrong, flip `_SIMULATOR_MODEL` in
`backend/src/agents/icp_simulator.py:35` back to Sonnet — no other
changes required beyond updating the pricing constants.

### 5. Fix the edit capture bug

`edit_similarity = 0` for most clients across all scored observations.
This means the system is not tracking the draft → published delta, which
is the strongest preference signal a ghostwriting system can have. This
is an existing bug that needs investigation.

**Investigation:**
- Where in the pipeline should the draft → published mapping be captured?
- Is the issue upstream (drafts aren't being stored with the right ID) or
  downstream (the mapping function isn't running)?
- Which clients DO have `edit_similarity` > 0, and what's different about
  their pipeline?

**Once fixed:** every edit becomes high-resolution preference data for
the client's `query_observations` tool to surface.

### 6. (Optional, later) The preference substrate

After the above is working, consider building a per-client preference
substrate: a continuously updated corpus of every signal the client has
ever given about their taste. Edits, approvals, rejections, conversation
transcripts, reaction patterns, implicit signals.

**Important caveat (raised during discussion):** the preference substrate
makes the *client* happy with drafts (reduces edit cycles, increases
approval rate), but does not directly optimize per-post audience
engagement or ICP exposure. These are different loss functions. The
substrate is necessary (you can't publish what the client rejects) but
not sufficient (client-approved ≠ audience-engaging).

Build this only after the ICP simulation loop is proving value, because
ICP simulation directly optimizes the audience-engagement loss function
while the substrate optimizes the client-approval one.

## Deprecated / retired

These components are no longer the primary job of the system under the
new framing:

- **RuanMei synthesis layer** — the prescriptive landscape generation is
  retired. The turn-based RuanMei agent we built today becomes dormant
  unless we find a new job for it. The underlying observation storage
  and reward computation infrastructure remains (it's system-of-record).
- **Analyst positive pattern prescriptions** — the analyst still runs
  (it finds real failure modes like "curated content underperforms" and
  "hot takes flop" which are useful as human-facing artifacts), but its
  positive pattern findings stop being injected into Stelle's prompt
  because they conflate descriptive correlations with prescriptive rules.
- **Content brief as Stelle input** — the content_brief.json file stops
  being written into Stelle's workspace. It may still be generated for
  human review and for Aglaea's interview prep.
- **Content landscape as Stelle input** — same as above.

## Open questions

1. **Does the analyst still need to run at all?** If its findings stop
   being injected into Stelle, the main consumer is the human strategist.
   Is that worth $2-3 per client per week? Probably yes while Cyrene
   doesn't exist, as a human-facing signal. Probably no once Cyrene is
   doing trajectory work.
2. **Do we need RuanMei as a class at all?** It's still the system-of-record
   for scored observations, reward computation, embedding management. The
   synthesis layer can go but the data layer stays. Consider renaming the
   class to make its reduced role clear.
3. **How does ICP simulation calibrate across clients?** The first version
   uses a single persona template. Per-client calibration will reveal
   whether each ICP needs a different persona voice. We'll learn this by
   running backtests after the tool exists.
4. **What's the right N for persona variants?** Too few = narrow signal.
   Too many = cost bloat and Stelle can't synthesize. Start with 1, expand
   if useful.
5. **Should Stelle's own reasoning be extended with a `think` tool?**
   Anthropic documents a "think" tool pattern that gives agents explicit
   scratchpad space. Might help for complex content planning. Low priority
   but worth trying after (3).

## Not doing

Things explicitly rejected during discussion:

- **Fine-tuning a LinkedIn-specific model** — too expensive, base models
  pass it within a generation, violates the Bitter Lesson.
- **Hand-coded pattern libraries** — we just deleted these, they hurt more
  than they help.
- **Threshold-based gating of RuanMei** (e.g., "use RuanMei when analyst
  LOO R² > 0.3") — the threshold is arbitrary and the whole point of
  stripping is that prescriptive injection hurts regardless of confidence.
- **Converting RuanMei to deeper turn-based synthesis** — we built this
  today but the OG comparison data makes it unclear whether synthesis at
  any depth helps. Keeping the infrastructure but not actively calling it.
- **Adding 10+ new tools** — tool bloat has real attention cost. New tools
  are limited to the high-leverage 4 listed above.

## Priority order for implementation

1. ✅ **Strip the prescriptive injection layer** — DONE 2026-04-10
2. ✅ **Add the 4 new tools** (query_observations, query_top_engagers,
   search_linkedin_corpus, execute_python) — DONE 2026-04-10
3. ✅ **ICP Persona Simulation — turn-based learning agent**
   - ✅ Phase 1: minimal viable simulator (one-shot persona + ICP + draft) — DONE 2026-04-10
   - ✅ Phase 2: prediction logging to simulator_predictions.jsonl — DONE 2026-04-10
   - ✅ Phase 3: turn-based learning agent replaces the one-shot path — DONE 2026-04-11.
     Agent studies historical posts, reactor identities, and (draft, published) deltas
     over multiple turns; builds internal audience model; transitions to Phase B for
     reaction. Runs in parallel with Stelle in a background thread.
   - ⏳ Phase 5: accuracy tracking + meta-calibration (optional) — include the simulator's
     own past predictions alongside actual engagement outcomes as self-correcting signal.
     Requires joining simulator_predictions.jsonl with engagement data by draft_hash.
4. ⏳ **Fix pre-existing Pinecone dependency bug** — `pip uninstall pinecone-client && pip install pinecone` in the ruanmei env (30 seconds). Unblocks Stelle's semantic search on cache miss.
5. ⏳ **Fix pre-existing Supabase keyword search 500** — rewrite the `or=` clause in `analyst.py::_search_keyword`. Low priority since semantic mode is the usable fallback.
6. ⏳ **Fix edit capture bug** — `edit_similarity = 0` for most clients. Investigation first. See the edit capture section above.
7. ⏳ **Build Cyrene** as turn-based trajectory agent — 1-2 days
8. ⏳ **Run a full comparison** of new architecture (stripped Stelle + all 4 new tools + ICP simulator) vs old pipeline on all 6 clients. Gives empirical validation of the architectural pivot. ~$60-120.

## Cost budget

Rough estimates:
- Stripping prescriptive injection: $0 (code-only change)
- Adding 4 tools: $0 (code-only change)
- ICP Persona Simulation implementation: $0 code, ~$0.02/simulation in prod
- Edit capture bug fix: $0 (code-only)
- Building Cyrene: $0 code, ~$1-2/run in prod
- Final comparison run: ~$60-120 (6 clients × 1-2 runs each)

Total new infrastructure cost: negligible. The expensive part is the
comparison runs, which give empirical data on whether the new architecture
actually improves outcomes.
