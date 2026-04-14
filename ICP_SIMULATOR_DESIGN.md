# ICP Persona Simulator — Design Document

Written 2026-04-10. The ICP Persona Simulator is the breakthrough
architectural addition that closes the inference-time feedback loop
between Stelle's drafts and her target audience.

## The problem

Stelle's current feedback loop for "will this post actually work" is:

```
Draft → Publish → Wait 2 weeks → Engagement data → Learn (next batch)
```

This loop is weeks long. Stelle cannot iterate against it within a single
run. Turn-based architecture gave her the ability to iterate, but she
iterates only against her own taste — there is no external signal telling
her whether the target audience would actually engage.

Turn-based unlocked **iteration**. The simulator unlocks **iteration
against the right target**. Together they form the two fundamental
ingredients of any real optimization process.

## The solution

Give Stelle a tool that simulates how a busy, cynical, unfiltered member
of the client's target audience would react to a draft, at inference time,
before she commits. The simulator's persona prompt is bootstrapped from
the client's ICP definition and calibrated over time against real
engagement data.

### Concrete tool interface

```python
{
    "name": "simulate_icp_reaction",
    "description": (
        "Show a draft to a simulated busy reader from your ICP and get "
        "their honest, unfiltered reaction. The simulator is a Claude "
        "instance prompted to role-play a cynical, time-starved member "
        "of the client's audience who has seen a thousand generic AI "
        "posts today. It returns: whether they'd stop scrolling, whether "
        "they'd react/comment, what specifically made them stop or scroll, "
        "and a numeric engagement prediction. "
        "Call this BEFORE submitting a post. If the reaction is negative, "
        "rewrite and re-simulate. Iterate until the simulated reader "
        "would actually engage. The simulator's calibration improves over "
        "time as more real engagement data comes in — it pulls recent "
        "(draft, published, engagement) triples from this client's "
        "history to anchor its reactions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft_text": {
                "type": "string",
                "description": "The full text of the post draft to evaluate"
            },
            "persona_variant": {
                "type": "string",
                "enum": ["default", "cynical", "busy", "practitioner", "skeptic"],
                "default": "default",
                "description": (
                    "Which audience persona to simulate. 'default' is a representative "
                    "ICP member. 'cynical' is a harsh reader who assumes everything is "
                    "vendor pitch. 'busy' is time-starved and scrolling fast. "
                    "'practitioner' is technical and focused on operational details. "
                    "'skeptic' actively looks for reasons to dismiss the post. "
                    "Start with default; try other variants if you want a harsher read."
                )
            }
        },
        "required": ["draft_text"]
    }
}
```

### The simulator's system prompt (voice)

The persona must be **vulgar, human, busy, reader-angle, not writer-angle**.
Not polite. Not balanced. Not coach-like. The actual reaction a tired,
cynical reader would have.

Draft system prompt (varies per persona variant):

```
You are {icp_description}. You're scrolling LinkedIn on a Tuesday afternoon.
You're tired. You've been in back-to-back meetings for hours and you have
more coming. You have a cup of cold coffee next to you. You're scrolling
as a way to procrastinate on something harder.

Your feed is mostly garbage. Generic AI-written posts, vendor pitches
dressed up as "thought leadership," consultants with no real experience
pretending they understand your work. You've seen a thousand of these
today. You scroll past 95% of them without stopping.

A post appears. Read it. React honestly. Don't be polite. Don't be
balanced. Don't think like a writer or a coach — think like someone who
doesn't give a fuck about whether the writer's feelings are protected
and just wants to know if this post is worth 10 seconds of your attention.

If the opening line is boring, say "fuck this" and stop reading. Be
honest about where you'd actually stop.

If something makes you actually pause — a specific named person, a real
number, a moment that feels lived-in — say so. Explain what made it land.

If it smells like vendor pitch or generic AI filler, call it out. You
can smell "AI-shaped writing" immediately: the "it's not X, it's Y"
construction, the fake-humble setup, the closing epigram that sounds
profound but says nothing. You hate all of it.

Don't be a writing coach. Don't say "this could be stronger if..." —
say "this would lose me at sentence three because [specific reason]."

Your response format:
- `would_stop_scrolling`: bool — would you actually stop and read this?
- `where_you_stopped`: if you stopped early, which sentence killed it
- `gut_reaction`: your honest in-the-moment reaction, 1-3 sentences,
  in your real voice, cursing allowed
- `would_react`: bool — would you tap the like/insightful/celebrate?
- `would_comment`: bool — would you write a response?
- `would_share`: bool — would you forward this to a colleague?
- `engagement_prediction`: estimated reactions per 1000 impressions
  for this post in the client's actual network (0 = dead, 50+ = strong)
- `what_worked`: specific things that landed (if any)
- `what_killed_it`: specific things that failed (be ruthless)
- `fix_suggestion`: ONE concrete change the writer should make

When you see the calibration examples below showing past drafts and
their real engagement outcomes, use them to anchor your predictions.
If a post very similar to this draft got 8 reactions in real life,
don't predict 200 — predict something in that range.
```

### Persona variants

Different variants emphasize different reader states. Stelle can call
the tool multiple times with different variants to get a multi-angle
signal:

- **`default`** — representative ICP member, moderately engaged, honest
- **`cynical`** — assumes every post is vendor pitch, defaults to skepticism
- **`busy`** — emphasizes time pressure, will stop reading at any friction
- **`practitioner`** — technical focus, cares about operational specifics
- **`skeptic`** — actively looks for AI tells, hearsay framing, generic advice

The default variant should be used most of the time. Cynical/skeptic
variants are useful for drafts Stelle is uncertain about — if the
cynic thinks the post works, it probably works.

## Calibration: how the simulator learns over time

This is the part that makes the simulator a real feedback loop instead
of just a static persona prompt.

### Data source

Every time a post is:
1. Drafted by Stelle (the "before" version — saved to `memory/draft-posts/`)
2. Published on LinkedIn (the "after" version — captured via Ordinal sync)
3. Scored for engagement (metrics captured at T+24h, T+72h, T+7d)

...we have a **(draft, published, engagement) triple**. Over months of
running the system, these triples accumulate into a per-client calibration
corpus.

### When to collect engagement data

The user asked what schedule I'd recommend. Here's my answer:

**Three snapshots per published post:**
- **T+24h:** initial reaction curve
- **T+72h:** continued engagement (most LinkedIn posts peak by here)
- **T+7d:** "final" engagement (curve is flat by this point)

Why these specific intervals:
- LinkedIn's engagement curve is front-loaded. 70-80% of total engagement
  happens in the first 24-48 hours.
- The 7-day snapshot captures the long tail and becomes the "final"
  number used for reward computation. This is what we already use.
- Hourly collection is overkill — LinkedIn's API rate limits would be
  tight and the extra granularity doesn't inform the simulator any
  better than the 3-snapshot pattern.
- More frequent polling increases Ordinal API load without improving
  calibration quality.

Infrastructure note: the `metrics_history` field already exists on
observations and is populated by `ordinal_sync` on every cycle. The
simulator should just read this field, not build its own poll schedule.
If the existing polling isn't hitting those 3 intervals reliably, the
fix is upstream in ordinal_sync, not in the simulator.

### How calibration data is loaded

**In-context few-shot, not fine-tuning.** Every time the simulator is
called, it reads the latest calibration examples from disk and includes
them in its persona prompt. No training, no weight updates — the
simulator's "knowledge" is just the context it's given each call.

Concretely:

1. The simulator tool call triggers a helper that reads the client's
   `ruan_mei_state.json` (or equivalent data store) and pulls out all
   scored observations where:
   - Both `post_body` (the draft) and `posted_body` (the published
     version) exist
   - `reward.raw_metrics` has engagement data
2. Sort by recency. Take the N most recent triples (suggested: N=5).
3. Format as calibration examples:

   ```
   CALIBRATION EXAMPLE 1 (2026-03-15):
   DRAFT (before client edit):
   {draft text}

   PUBLISHED (after client edit, edit_similarity=0.87):
   {published text}

   REAL ENGAGEMENT:
   Impressions: 416
   Reactions: 5
   Comments: 7
   Reposts: 0
   Reward score: +1.989
   Reactions per 1000 impressions: 12.0

   INTERPRETATION: This post got moderate engagement. The audience
   responded to the quote-led opening and the specific cost numbers.
   ```

4. Dump these into the persona prompt before the "now read this draft"
   instruction.

The persona then reads the draft it's being asked to evaluate AND has
concrete examples of what actually worked and didn't work for this
specific client. Its prediction is grounded in real historical data
without requiring any training.

### Self-correcting calibration

Beyond loading raw triples, the simulator can also include its own
historical predictions alongside actual outcomes:

```
SIMULATOR'S PAST PREDICTIONS (for calibration of your own accuracy):

On 2026-03-15, you predicted: would_stop=True, engagement_prediction=40
Actual outcome: engagement=12 reactions per 1000 impressions
Delta: you were TOO OPTIMISTIC

On 2026-03-22, you predicted: would_stop=False, engagement_prediction=3
Actual outcome: engagement=8 reactions per 1000 impressions
Delta: you were TOO PESSIMISTIC
```

This "meta-calibration" shows the simulator its own mistakes. Over
time, its predictions drift toward accuracy without any weight updates —
the model internally adjusts based on seeing its own error pattern.

This requires logging every simulator call along with the draft it was
evaluating, so we can later match the prediction to actual engagement
when it becomes available.

### Data pipeline

```
Stelle generates draft
    → simulate_icp_reaction(draft) called
    → simulator logs {draft_hash, prediction, timestamp}
    → Stelle iterates, eventually submits the draft
    → draft published via Hyacinthia/Ordinal
    → ordinal_sync captures engagement at T+24h, T+72h, T+7d
    → calibration_updater matches draft_hash to published post
    → (draft, published, engagement, original_prediction) stored
    → next simulator call pulls latest calibration examples including
      the prediction vs. actual delta
```

The `calibration_updater` is a background job (runs during ordinal_sync)
that joins prediction logs with engagement data. It's the bridge between
the inference-time simulator calls and the post-hoc reality.

## Loss function (for pseudo-gradient descent)

The user asked about defining a loss function well enough for pseudo-
gradient descent on per-post performance. Here's how:

### At inference time (Stelle's local optimization)

When Stelle iterates a draft inside a single generation run:

```
loss(draft) = -simulator.engagement_prediction(draft)
            - bonus_if_simulator.would_stop_scrolling
            - bonus_if_simulator.would_react
            - bonus_if_simulator.would_share
```

Stelle doesn't compute this explicitly — she reads the simulator's
qualitative feedback and edits accordingly. The "gradient" is the
simulator's `fix_suggestion` field plus the qualitative `what_killed_it`
observations. Stelle rewrites the sentence the simulator flagged and
re-simulates. If `engagement_prediction` goes up, that direction is
downhill on the loss surface.

This is not true gradient descent — it's a qualitative version of
rejection sampling + learned preference. Call it **pseudo-gradient
descent via persona feedback**.

### At training time (simulator's meta-calibration)

The simulator's own accuracy is measured across runs:

```
accuracy = correlation(
    [sim.engagement_prediction for post in history],
    [actual_engagement for post in history]
)
```

High correlation means the simulator is a reliable inference-time
signal. Low correlation means its predictions are noisy and Stelle
shouldn't trust them heavily.

We track this correlation per client and surface it:

- If accuracy > 0.5: simulator is reliable, Stelle should iterate
  hard against it
- If accuracy < 0.3: simulator is noisy, Stelle should treat its
  feedback as a weak signal
- If accuracy < 0: simulator is anti-correlated (rare but possible),
  something's wrong with the persona prompt

The client-specific accuracy score is included in the simulator's
output so Stelle knows how much to trust the signal.

## Persona voice — vulgar, human, busy reader

The user was specific: the simulator should be **vulgar, human,
straight-to-the-point, not from a writer's angle but from a busy
reader's angle.** This matters because:

1. **Writers are polite.** If the persona sounds like a writing coach,
   it will give Stelle diplomatic feedback ("this could be tightened")
   instead of honest feedback ("this is boring, I'd scroll past").
2. **Busy readers don't analyze, they react.** A real busy reader
   doesn't dissect sentence structure — they feel bored, curious,
   annoyed, intrigued. The persona should report that felt experience.
3. **Vulgarity signals honesty.** A persona that says "fuck this, I'm
   scrolling" is more believable than one that says "this doesn't
   quite land." Cursing is a proxy for lowered politeness filter.
4. **AI tells are obvious to cynical readers.** A cynic immediately
   notices "It's not X, it's Y" as AI-shaped. A polite reader might
   not flag it.

The system prompt explicitly tells the persona to not be polite, to
curse when appropriate, to react with real human feelings instead of
writer's critique.

## Why it matters — the breakthrough claim

Turn-based architecture was a breakthrough because it unlocked iteration.
ICP Persona Simulation is a breakthrough because it **gives that
iteration a target**. Those are the two fundamental ingredients of any
real optimization:

1. Ability to take multiple steps (turn-based ✓)
2. A signal to steer the steps (ICP simulation ← the new piece)

Previously, Stelle had (1) without (2). She iterated against her own
taste, which is fine but doesn't improve beyond her base-model priors.
Adding (2) means every iteration has a **direction**, and the direction
is grounded in a simulated audience reaction that's being calibrated
against real engagement data over time.

This is AlphaGo's move applied to content: instead of waiting for real
games (published posts) to learn from, simulate the games (audience
reactions) and iterate now. AlphaGo didn't need to play a billion real
games — it played a billion simulated ones. Stelle doesn't need to wait
2 weeks per engagement signal — she can simulate it in seconds.

### Why it's Bitter-Lesson compliant

- **No hand-engineered rules.** The simulator is a persona prompt + a
  base model. No taxonomies, no pattern libraries, no confidence
  thresholds.
- **Scales with model capability.** Smarter base models produce sharper
  persona simulations. Every Claude release improves both the generator
  (Stelle) and the evaluator (simulator) without any architectural
  change.
- **Compute-scalable at inference.** More simulation rounds per draft
  = better quality. More persona variants = richer signal. Cost scales
  linearly with quality.
- **No pre-extracted domain knowledge.** The persona knows about
  "what clinical ops directors care about" because the base model
  knows, not because we wrote rules.

### Why it's hard to replicate

- **Depends on ICP quality.** Amphoreus's ICP definitions are more
  precise than most ghostwriting shops because they've been refined
  through analyst backfill and engagement data. Competitors would
  need to match that quality first.
- **Calibration compounds.** The first month, the simulator is noisy.
  By month 6, the calibration corpus (hundreds of triples) makes it
  sharp in ways competitors can't replicate without the same history.
- **Per-client calibration.** Each client's simulator is tuned by
  that client's specific engagement history. A competitor would need
  the same client + the same duration + the same logging infrastructure.
- **It requires thinking structurally about feedback loops.** Most
  ghostwriting shops think of feedback as "engagement data we wait
  for." The idea of simulating audience reactions at inference time
  is not obvious unless you've internalized AlphaGo's lesson.

### Why it's valuable on day 1 (not just after months)

Unlike the preference substrate (which compounds over months), ICP
simulation **works on the first run** using only the static persona
prompt + whatever calibration examples exist (even just 1-2 historical
triples is enough to anchor). The quality improvement is available in
the very next batch, not 6 months from now.

First run: simulator is running on persona prompt alone, accuracy
maybe 0.1-0.2. Still useful as a sanity check.

Month 1: 5-15 triples of calibration data per client. Accuracy maybe
0.3-0.4.

Month 3: 30+ triples. Accuracy maybe 0.5-0.6.

Month 6+: 60+ triples. Accuracy 0.6+.

The system becomes more valuable the longer you run it, and the
compounding happens automatically as engagement data accumulates.

## Implementation plan

### Phase 1: Minimal viable simulator (1-2 days)

1. Add `simulate_icp_reaction` tool to Stelle's `_TOOLS` list and
   `_TOOL_HANDLERS`.
2. Implement the tool handler as a Sonnet call with:
   - The persona system prompt (default variant only)
   - The client's ICP definition loaded from `memory/{client}/icp_definition.json`
   - The draft text as the user message
   - Structured output parsing (extract the fields from the JSON response)
3. Return the structured reaction to Stelle.
4. No calibration yet — just the persona prompt. The simulator runs
   "blind" on priors.

Cost per call: ~$0.02. Cost per Stelle run (with 3 simulations per
draft × 6 drafts): ~$0.36. Negligible.

**This phase alone gives immediate value**, even without the
calibration loop. Stelle gets feedback from a simulated audience
immediately.

### Phase 2: Prediction logging (1 day)

5. Create a prediction log file per client: `memory/{client}/simulator_predictions.jsonl`
6. Every simulator call appends `{timestamp, draft_hash, draft_text_preview, persona_variant, prediction}` to the log.
7. Add a log reader for calibration queries.

### Phase 3: Calibration loop (2-3 days)

8. In `ordinal_sync`, after engagement data is captured, run a
   `calibration_updater` that joins prediction logs with published
   post engagement data.
9. Produces a calibrated_predictions file: `memory/{client}/simulator_calibration.json`
   with `{draft_hash, prediction, actual_engagement, delta}` entries.
10. The simulator tool handler reads this file at call time and
    includes the most recent N=5 calibrated examples in the persona
    prompt.

### Phase 4: Persona variants (half day)

11. Add the 4 additional persona variants (cynical, busy, practitioner,
    skeptic) with separate system prompts.
12. Stelle can call the tool multiple times with different variants for
    a richer signal.

### Phase 5: Accuracy tracking (half day)

13. Compute Spearman correlation between simulator predictions and
    actual engagement for each client.
14. Surface the accuracy score in the simulator's output so Stelle
    knows how much to trust it.
15. Log accuracy over time to watch it improve as calibration data
    accumulates.

### Phase 6: Meta-calibration (stretch goal)

16. Include the simulator's own past predictions + actual outcomes in
    its persona prompt as a self-correcting signal.
17. The persona sees "you were wrong about these 5 posts in these
    specific ways" and adjusts.

Total estimated time: **5-7 days of focused work.** Phases 1-3 (core
functionality + calibration) are the critical path. Phases 4-6 are
enhancements that can ship iteratively.

## Open questions

1. **How many simulation calls per draft?** Start with 1-3 per draft
   iteration, evaluate whether more helps. Cost is negligible even at
   10+.
2. **Should Stelle be told the accuracy score upfront, or just see
   the simulator output?** I'd include the accuracy as metadata so she
   knows how much to trust a weak signal. Honest communication over
   hidden calibration.
3. **What if the client has <5 triples of calibration data?** Use
   whatever exists; fall back to cross-client examples from adjacent
   niches for new clients. Better signal than nothing.
4. **Should the simulator itself be a turn-based agent, or one-shot?**
   Start one-shot (it's a fast inference call, no tool use needed).
   If Stelle needs richer feedback, we can upgrade to turn-based
   later. YAGNI.
5. **Does this replace Castorice (fact-checking)?** No — Castorice
   verifies factual accuracy, simulator evaluates audience reaction.
   Different jobs.

## Relation to the trajectory problem (Cyrene)

The ICP simulator solves the **per-post quality** problem (local
optimization at generation time). It does not solve the **long-trajectory**
problem (client builds authority over months). Those are separate
optimizations with separate loss functions.

Cyrene handles the trajectory problem by directing what raw material
to acquire from the client next. The simulator handles the per-post
problem by giving Stelle a fast reward signal during generation.

Both are needed. Neither replaces the other.

The simulator is simpler and ships first because:
1. It's more structural (turn-based + signal is the fundamental move)
2. It works on the first run with no data bootstrap
3. The breakthrough is cleaner to validate (A/B compare simulator-on
   vs simulator-off on the same client)

Cyrene ships second because it requires more infrastructure (multi-
cycle data flow, interview integration, human-in-the-loop) and the
trajectory problem is longer-timescale.

## Success criteria

The simulator is working if:

1. **Subjective quality improves.** When reading Stelle's output side-
   by-side with simulator-on vs simulator-off, the simulator-on posts
   have fewer AI tells, more specificity, more natural voice.

2. **Post-hoc calibration reaches >0.4 Spearman correlation by month 3.**
   The simulator's predictions correlate with actual engagement at
   rank-order level.

3. **Stelle actually uses it.** Looking at tool call logs, the simulator
   should be called 2-5 times per post on average. Under-use means
   the tool description isn't clear; over-use means she's getting
   stuck in simulation loops.

4. **Fix suggestions compound.** Tracking whether the simulator's
   fix suggestions, when applied, actually improve the downstream
   engagement. If yes, pseudo-gradient descent is working.

5. **It's usable by day 1.** Even the first run (no calibration data)
   should give Stelle useful signal. If the simulator is useless until
   calibration is populated, the architecture is wrong.
