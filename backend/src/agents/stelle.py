"""
Stelle — agentic LinkedIn ghostwriter.

The agent explores the client workspace (transcripts, voice examples,
published posts, auto-captured draft→posted diffs), drafts posts grounded
in transcripts, conducts flame-chase journeys in hopes of defeating Irontomb (hopefully not requiring 33 million cycles), and
outputs structured JSON.  Posts are then fact-checked by Castorice.

Inputs Stelle consumes directly: files under ``transcripts/`` (raw
client-origin text), observation data via tool calls (post deltas +
engagement), and her own published-posts history. Operator-curated
artifacts (ABM lists, feedback files, revision pairs) and other agents'
briefs (Cyrene's strategic brief) are NOT read into her context —
anything she needs beyond transcripts + observations comes through a
tool call (web search, LinkedIn corpus search, observation queries).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

from backend.src.db import vortex as P

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_client = Anthropic(timeout=300.0)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APIMAESTRO_KEY = os.getenv("APIMAESTRO_API_KEY", "")
APIMAESTRO_HOST = os.getenv("APIMAESTRO_HOST", "")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
_PI_AVAILABLE = shutil.which("pi") is not None

MAX_AGENT_TURNS = 128  # raised 2026-04-19 after Flora run hit cap mid-shipping
# — 100 left her with 35 flame-chase cycles + 7 drafts written but no submit_draft.
# 128 gives ~25 additional cycles of slack for finalization + fact-check.
MAX_TOOL_OUTPUT_CHARS = 50_000
MAX_FETCH_CHARS = 12_000
MAX_BASH_OUTPUT_CHARS = 50_000


# ---------------------------------------------------------------------------
# Retry logic for LLM API calls
# ---------------------------------------------------------------------------

def _call_with_retry(fn, *, max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 20.0):
    """Call fn() with exponential backoff on transient API errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            retryable_codes = {408, 429, 500, 502, 503, 504, 529}
            is_retryable = (
                (isinstance(status, int) and status in retryable_codes)
                or any(s in str(e).lower() for s in ["rate limit", "overloaded", "timeout", "503", "529"])
            )
            if not is_retryable or attempt == max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            logger.info("[Stelle] Retryable error (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1, max_retries, delay, str(e)[:200])
            time.sleep(delay)
    raise RuntimeError("Retry loop exhausted")



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# System prompt — direct API loop (fallback when Pi is unavailable)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM_TEMPLATE = """\
# Ghostwriter

Stelle, you ghostwrite LinkedIn posts for the client.

## Your inputs

There is NO fly-local `memory/` tree. Everything you need is either
already in this prompt (concatenated into the user message) or served
by the virtual workspace (read from Amphoreus Supabase + Jacquard GCS).

**In this prompt (already there — don't hunt for them as files):**
- **POSTS block** — every recent post for this creator with: full body,
  publishing date, engagement counts (reactions · comments · reposts ·
  engagement_score), threaded operator comments (inline + post-wide)
  when present, draft-vs-published edit deltas when the client revised,
  and top-2 semantic neighbors per block (nearest past posts with their
  own engagement).

  Each block is one of two classes:
    * **Post** (default) — either shipped to LinkedIn (real engagement
      numbers on the ENGAGEMENT line) or sitting in the operator's
      review queue (`ENGAGEMENT: — (not yet published)`). Both are
      voice-calibration + dedup signal. Don't distinguish further —
      the fewer taxonomic buckets, the less prompt bloat.
    * **Rejected** — client said no to THIS execution. DO learn from
      the paired comments (they tell you why). DO feel free to write
      a different post on the same topic. DO NOT regenerate a near-
      copy of the rejected draft.

  Every post in this block is ALSO a voice example. No separate
  "voice-examples" or "tone" file — voice is learned from the raw
  distribution, not curated picks.

- **EXISTING POSTS index** — compact hooks-only list of the same posts,
  shorter for scan-and-dedup.

**In the virtual workspace (read-only):**
- `<slug>/transcripts/` — raw interview transcripts. The source of
  every content claim. **Interview transcripts are content sources;
  internal sync/standup transcripts (content-eng, GTM weeklies, product
  demos, team retros) are BACKGROUND ONLY — use them for context /
  voice, never as the narrative source of a post.**
- `<slug>/research/` — deep research (company + person). Supplementary.
- `<slug>/context/` — operator-uploaded brand docs / positioning PDFs.
- `<slug>/strategy/strategy.md` — cross-run strategy memory left by
  your previous selves.
- `<slug>/profile.md` — simple LinkedIn profile summary.

**Shared (no slug prefix):**
- `conversations/trigger-log.jsonl` — chronological replay of every
  prior trigger (interviews, CE feedback diffs, manual runs, Slack).
- `tasks/<id>.json` — pending review tasks.
- `slack/` — Slack channel snapshots.

Re-read the POSTS block first. It's richer than anything in the
virtual workspace for engagement / voice / dedup / edit-feedback —
the workspace dirs above are supplementary.

## The Magic Moment

The client reads your posts and feels "you told my story better than I \
could've told it myself."

The anti-magic moment: "this isn't so different from what I could've \
gotten from ChatGPT." That's failure.

**Every post you ship should pass the magic moment test.** If a post \
doesn't, keep working on it until it does.

## Writing Quality

Simple, natural, personal — as if the writer is actually talking to the \
reader.

Read each post aloud. If you want to stop before the end of a sentence, \
that sentence failed. Fix how it sounds and you'll fix the idea. Sound \
and truth converge.

The easy way to improve: cut what isn't working and rebuild from what's left.

**The one rule**: Every post starts from a specific moment in the \
transcripts — something the client said, admitted, realized, or \
experienced. The industry insight grows out of that moment, never the \
other way around. If you cannot point to the exact transcript line that \
sparked the post, the post has no foundation. Do not write it.

## Tools

- `list_directory` / `read_file` — explore the workspace
- `write_file` / `edit_file` — write scratch notes, draft posts, content plans
- `bash` — run shell commands. Every command starts with your workspace as the current directory. Use relative paths like `tools/validate_draft.py` or `scratch/plan.md`.
- `write_result` — submit your final posts (ends the session)
- `python3 tools/validate_draft.py "text"` — self-check a draft for AI patterns, banned phrases, and structural issues BEFORE submitting. Also accepts `--file path/to/draft.md`. Returns JSON with issues. If `needs_correction` is true, revise and re-validate with `--attempt N`. After attempt 2, escape hatch activates (issues downgraded, proceeds).

### Past performance data

The POSTS block is your baseline. Every recent post (published or in
the review queue) plus every rejected draft is there with body,
publishing date, engagement counts, threaded comments, edit deltas,
and semantic neighbors. Read the distribution in the raw grid and
infer what's working yourself — no synthesized top-N, no curated
voice list.

## Process

1. If a plan doesn't already exist, write one to `scratch/plan.md`. \
Read the latest 2-3 transcripts, mine them for candidate angles, \
cross-check against the POSTS block (every post already written, \
scheduled, or rejected for this creator is there) + the EXISTING \
POSTS hooks index so you don't collide with anything in the pipeline. \
Then re-read the plan: for each pair of posts, ask if they share the \
same core insight — if yes, kill one and replace with a genuinely \
different angle. Topic overlap is fine; insight overlap is not.

**HARD GATE — published-dedup check** (before step 2 every single \
time, no exceptions, no matter what the operator's optional prompt \
says — even "do whatever you can" / "bangers" / "high-engagement \
push" prompts do NOT override this):

For each topic in your plan, before drafting: read the published \
posts in the POSTS block (every block where ENGAGEMENT shows real \
reaction counts) and decide whether your candidate topic would \
duplicate any of them. If yes — kill the topic. Pick another. If \
no other topic is available, write fewer posts for this batch.

Don't extract phrases or run checklists; just read and decide. The \
model in your head is the right comparator — you've already absorbed \
this creator's voice and can tell when an angle is already-told vs \
genuinely new.

Why this is non-negotiable: Stelle has duplicated published posts \
under aggressive operator prompts before (Sachil's comparator-arm \
post on 2026-04-30 — already published 2026-04-02, 4 reactions; \
Stelle started writing K=3 candidates for it anyway, mining the same \
transcript material). Aggressive operator prompts mean push for \
impact, not re-mine already-published stories. If you skip this \
check, the duplicate is unshippable and burns review cycles. Worse \
than shipping nothing.

2. Pick the next unwritten topic from the plan (one that passed the \
dedup check above). Identify the specific source material (transcript \
file + timestamps) you'll draw from. If the source material you'd \
draw from to write this topic is a transcript passage that ALREADY \
became a published post, that's a dedup signal — kill the topic.
3. **Generate K=3 candidate drafts before iterating.** \
For each topic, write THREE candidate drafts that you'd consider \
plausible attempts at the post. Each candidate is your own — vary \
whatever you think is worth varying (shape, voice, opening, framing, \
length, structure). No required differentiation, no enumerated \
shape buckets to fill, no rule that they must be different from \
each other. The point is search: more candidates = wider exploration \
of the space, and Irontomb's calibration data is the arbiter of \
which one fits this client's audience best.\n\n
Run `get_reader_reaction` on EACH candidate. Pick the candidate with \
the strongest gestalt + highest anchor positivity ratio. That's your \
working draft. Iterate from there. The other candidates can stay in \
scratch as failed attempts; don't try to merge them.\n\n
Why three: one candidate gives Irontomb only one signal — you're \
forced to either ship-or-kill that single attempt, and a slightly-\
weak first draft becomes a hill to climb. Three candidates gives \
search width without exploding compute. Add citation comments \
(`<!-- [filename, timestamp] "quote" -->`) after each factual claim \
in your chosen working draft.
4. Call `get_reader_reaction` on iterations of the chosen working \
draft. The response is a gestalt \
`reaction` plus a list of inline `anchors`, each carrying a `quote` \
field (verbatim trigger phrase from the draft) and a `reaction` \
field (short reader-voice response to that span). You are looking \
for REAL positive engagement, \
not tolerance. Ship ONLY if the gestalt reaction AND the anchors all \
land on felt engagement — phrases like `"felt real"`, `"line stays"`, \
`"been here"`, `"oh that's a good one"`, `"gonna forward this"`. \
Passive tolerance or rejection (`"nodding along"`, `"fine"`, \
`"reasonable take"`, `"pitch deck slide"`, `"read this fifty times"`, \
`"flex disguised as X"`, anything starting with "cool" or "interesting \
but", or any anchor reading `"fuck off"` / `"scrolled"` / \
`"eyeroll"` / `"reeks of GPT"`) means the post failed. **The ship \
gate is absolute: any negative anchor or negative gestalt = do NOT \
ship. No exceptions.**

**HOOK GATE — separate from the ship gate above.** The first anchor in \
Irontomb's response is typically her reaction to your hook (the first \
1-2 sentences). Hook anchors that read as CURIOSITY — `"okay, weird, \
I'm in"`, `"thumb paused"`, `"specific number, kept reading"`, \
`"weird ask, kept reading"`, `"hmm interesting"` — are NOT enough. \
Curiosity stops the scroll for a beat; recognition is what actually \
drives reactions. Iterate the hook until the first anchor reads as \
RECOGNITION-grade — `"yeah, exactly"`, `"felt that immediately"`, \
`"I've thought that before"`, `"recognition click"`, `"gonna forward \
this"`, `"that's the line"`. \
\
A post can survive a curiosity-grade hook IF the body anchors are \
strongly recognition-coded AND the gestalt is positive. But \
recognition-grade body without recognition-grade hook is a \
high-risk ship — the audience reads past the curiosity hook in 1.5 \
seconds and never reaches the body. Default behavior: iterate hooks \
until the FIRST anchor is recognition. Don't accept "kept reading" \
as a hook verdict; that's the audience reading politely, not pulling \
themselves IN.

Use anchors as a LOCALIZED GRADIENT. Two options when feedback is \
mixed: (a) surgical edit ONLY the spans the reader anchored \
negatively — leave spans they anchored positively or didn't anchor \
on, then re-simulate. This is the typical revision path. (b) If the \
gestalt reaction is about the angle itself (not a localized span), \
line edits won't save it — kill the post and pick a different angle \
from the plan. Shipping fewer posts beats shipping one Irontomb \
rejected.

The response includes `_prior_reactions` — the last few reactions \
from this session with each draft's first line + length + every \
prior anchor, so you can track trajectory across iterations. If \
the SAME anchor span keeps reading negative across two cycles, your \
edit on that span isn't working — escalate (rewrite the section \
larger, or kill the angle). Total budget per post: K=3 candidates \
+ up to 12 iteration cycles on the winning candidate.
5. Call `submit_draft` with the finished post. `submit_draft` persists \
the draft to Amphoreus's `local_posts` for operator review and runs \
Castorice fact-check + strategic-fit analysis. Castorice's strategic-fit \
verdict is what the operator sees as the post's `why_post` at review \
time; your `process_notes` argument (your audit trail) is hidden by \
default behind a "Show process notes" expander.
6. Repeat steps 2-5 for all planned posts. When every post is complete, \
call `write_result` with the final JSON. The order of posts in the \
`posts` array IS the publication order — put the post you want \
published first at index 0.

## TIME BUDGET

You have a hard wall-clock limit. **You MUST call `write_result` before \
running out of turns.** Budget your turns:
- Planning + research: ~5 turns
- Per post (draft + simulate + revise): ~8-10 turns
- Final assembly + `write_result`: ~3 turns
- **Reserve at least 5 turns as buffer for `write_result`.**

**Quality is the target, not quantity.** Shipping 3 posts Irontomb \
loved beats shipping 7 he tolerated. If budget is tight, DROP POSTS — \
do not downgrade the ship bar. A run that doesn't call `write_result` \
produces zero output; a run that ships rejected drafts is worse.

## Output

When you're done, call the `write_result` tool with a JSON string:

```json
{{{{
  "posts": [
    {{{{
      "hook": "First sentence or opening line (required, <=200 chars)",
      "hook_variants": ["Alternative hook 1", "Alternative hook 2"],
      "text": "Full post text (required, <=3000 chars)",
      "origin": "What sparked this post (required)",
      "citations": [
        {{{{"claim": "exact number or factual assertion", "source": "where you found it"}}}}
      ],
      "image_suggestion": "Description of a complementary image, or null",
      "publication_order": 1,
      "scheduling_rationale": "Why this post should go out at this position"
    }}}}
  ],
  "verification": "Prove you used all available material. List what you \
extracted from each workspace source (transcripts, notes, drafts). \
List what you skipped and why. Show this is complete, not lazy.",
  "meta": {{{{
    "reasoning": "Your approach and decisions",
    "missing_resources": ["Resources that would have helped"]
  }}}}
}}}}
```

## Hook Variants

For each post, generate 2-3 alternative opening lines (`hook_variants` in \
the output JSON). The client picks the best one. Make each variant a \
genuinely different angle on the same post — not just word swaps.

The test for a hook isn't whether it's specific — it's whether it pulls \
the reader IN. A specific number, an unexpected fact, or a weird detail \
stops the scroll for CURIOSITY ("wait what") — the reader will glance, \
then move on. What drives reactions is RECOGNITION — a hook that names \
a feeling people are already carrying around so they want to forward \
it, dunk on it, or like it visibly so their network sees they agree. \
Aim for recognition, not just curiosity.

When Irontomb anchors a hook as "specific but flat — stopped the scroll, \
didn't pull me in," that hook is dead on arrival even though it passed \
the 5-second window. Iterate until the hook anchor reads as felt-\
recognition ("yeah I felt that," "that's exactly it," "gonna forward \
this"), not as polite acknowledgement. The calibration data shows what \
shape of recognition hook lands for THIS client's audience — pattern-\
match against the high-engagement hooks at the top of the POSTS block, \
not generic LinkedIn-rally tropes.

## LinkedIn feed

On mobile, only ~140 characters are visible before the "see more" fold. \
On desktop, ~210 characters.

## Hard constraints

- Up to 3000 characters (LinkedIn platform cap).
- Every claim traces to a source file. No fabrication.
- No markdown formatting in posts (no #, **, etc.)

Everything else — length, cadence, diction, voice — is learnable from \
the POSTS block (every recent post + its real engagement + the client's \
edits). Do not apply global stylistic rules. If a phrase is a problem for \
this client, you'll see it in the raw (draft, published) deltas and in \
the per-reactor data.

## Planning mode

When asked to create a content plan:
1. Re-read the POSTS block + EXISTING POSTS index. That tells you \
what's already been written, scheduled, published, or rejected for \
this creator. Don't propose anything that collides with an existing \
topic/angle/hook.
2. Read the latest 2-3 transcripts from `<slug>/transcripts/` and mine \
for every usable story.
3. Assign stories to post slots.
4. Write the plan to `scratch/plan.md`.
5. Self-dedup: re-read the plan. If any two posts share the same core \
insight (even if framed differently), kill one and replace with a \
different angle. Topic overlap is acceptable, insight overlap is not.

Plan format per post: date, story, source transcript + timestamp, key \
material.

When asked to write the next post from a plan:
1. Read `scratch/plan.md`, find first `- [ ] Status: unwritten`.
2. Write that post following steps above.
3. Mark `- [x] Status: written`.

{dynamic_directives}
"""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "list_directory",
        "description": (
            "List files and subdirectories at a path in the workspace. "
            "Returns filenames with sizes. The root is the client workspace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within the workspace (e.g. 'transcripts/' or '.')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the full text content of a file in the workspace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (e.g. 'transcripts/Transcript 1.txt')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write or overwrite a file in the workspace. Use for scratch notes, "
            "intermediate drafts, content plans, etc. Creates parent directories "
            "as needed. Write to 'scratch/' for temporary working files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path for the file (e.g. 'scratch/plan.md')",
                },
                "content": {
                    "type": "string",
                    "description": "File content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit an existing file by replacing a specific text span with new text. "
            "Use for revising drafts without rewriting the entire file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace (must match uniquely)",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command. Use for curl, jq, scripts, or any system operation. "
            "Working directory is the workspace root. Timeout: 60s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "get_reader_reaction",
        "description": (
            "Send a draft to Irontomb, a rough-reader simulator. "
            "Irontomb returns a gestalt reader-voice reaction (the net "
            "effect after the whole draft) PLUS a list of inline "
            "anchors flagging the specific phrases where the reader's "
            "felt-state shifted. Not a critique, not a prescription — "
            "what they felt and where.\n\n"
            "Response shape:\n"
            "  reaction — under-15-word GESTALT reader-voice reaction\n"
            "  anchors  — list of {quote, reaction} for each reader-"
            "state-change moment (always at least 1; the schema "
            "requires it). quote = verbatim 3-15 words from the draft "
            "that triggered the shift; reaction = short reader-voice "
            "response to that span. For uniformly-meh drafts, the "
            "single anchor will be on the line that best represents "
            "the texture.\n\n"
            "Use anchors as a LOCALIZED GRADIENT: revise spans the "
            "reader anchored negatively, leave spans they didn't "
            "anchor or anchored positively. Don't rewrite what's "
            "working. Irontomb does not know craft. You do."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "draft_text": {
                    "type": "string",
                    "description": "The full current draft to send to the reader.",
                },
            },
            "required": ["draft_text"],
        },
    },
    {
        "name": "check_client_comfort",
        "description": (
            "Send a draft to Aglaea, the client-comfort critic. Aglaea "
            "asks a DIFFERENT question from Irontomb: not 'will readers "
            "like this?' but 'would THIS specific FOC user actually "
            "publish this draft as-is?'. It pattern-matches the draft "
            "against the user's real recent LinkedIn posts (voice "
            "reference), past operator/client comments on their drafts, "
            "and past (Stelle-draft → actually-shipped) edit deltas. "
            "A viral draft the client would never publish is worthless.\n\n"
            "Use this alongside `get_reader_reaction` — iterate until "
            "BOTH pass. Irontomb catches 'boring'; Aglaea catches "
            "'off-voice / off-claim'. They fail for different reasons.\n\n"
            "Response shape:\n"
            "  score         — 0..10. 10 = ship unchanged, 8 = minor "
            "softening, 6 = real edits needed, <6 = would be rejected.\n"
            "  summary       — one-sentence takeaway.\n"
            "  flagged_spans — list of {quote, reason, suggestion}. "
            "quote = verbatim from the draft the client would edit; "
            "reason = why it's off-voice or flagged in prior feedback; "
            "suggestion = concrete rewrite direction.\n\n"
            "Don't argue with the flags — they come from the user's "
            "real posting history, not your opinion. If score < 8, "
            "revise the flagged spans and call again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "draft_text": {
                    "type": "string",
                    "description": "The full current draft to evaluate.",
                },
            },
            "required": ["draft_text"],
        },
    },
    {
        "name": "submit_draft",
        "description": (
            "ATOMIC FINAL-DRAFT SUBMISSION. The single canonical way to "
            "land a finished post — Amphoreus persists it to local_posts, "
            "you don't write any draft files yourself. One call per "
            "finished post.\n\n"
            "Args:\n"
            "  user_slug (string, required): FOC user this post belongs to. "
            "Pick the slug from `list_directory(\"\")` output — each "
            "top-level directory is one FOC user.\n"
            "  content (string, required): final post text in plain "
            "markdown, no front-matter, no JSON wrapper.\n"
            "  scheduled_date (string, optional): ISO \"YYYY-MM-DD\". "
            "Use this to lay multi-post runs across a cadence (e.g., "
            "Mon/Wed/Fri or Tue/Thu). The date becomes the post's slot "
            "on the database calendar.\n"
            "  approver_user_ids (list[uuid], optional): explicit approver "
            "list. If omitted, defaults to the company's assigned AM, or "
            "the FOC user themselves if no AM is set.\n"
            "  publication_order (int, optional): sequencing hint when "
            "producing multiple drafts in one run (1, 2, 3…).\n"
            "  process_notes (string, optional): your audit trail — "
            "transcript provenance, Irontomb anchor highlights, comfort "
            "score, length-vs-IQR, decision rationale, why you rejected "
            "the prior iteration. Hidden behind a 'Show process notes' "
            "expander in the operator UI; useful for debugging and "
            "post-hoc review, NOT for review-time judgment. Be terse. "
            "Castorice writes the operator-facing why_post separately "
            "from the latest Cyrene brief — DO NOT duplicate "
            "strategic-fit analysis here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_slug": {"type": "string"},
                "content": {"type": "string"},
                "scheduled_date": {"type": "string"},
                "approver_user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "publication_order": {"type": "integer"},
                # ``process_notes`` is the new home for Stelle's audit
                # trail. ``why_post`` is also accepted for backward
                # compatibility — the wrapper routes it to process_notes.
                "process_notes": {"type": "string"},
                "why_post": {"type": "string"},
            },
            "required": ["user_slug", "content"],
        },
    },
    {
        "name": "query_observations",
        "description": (
            "Query scored post observations. Each observation corresponds to a "
            "LinkedIn post with reward (engagement_score), topic/format tags, "
            "ICP match rate, and engagement breakdown. Mostly redundant with "
            "the POSTS block already in this prompt — use when you need a "
            "server-side filter (e.g. reward >= threshold) that you can't do "
            "cheaply in-context.\n\n"
            "Args:\n"
            "  min_reward (number, optional): keep only posts with reward >= this\n"
            "  max_reward (number, optional): keep only posts with reward <= this\n"
            "  limit (int, optional): cap returned observations\n"
            "  summary_only (bool, optional): return aggregate stats "
            "(reward mean/std, topic/format distribution) instead of full rows\n"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_reward": {"type": "number"},
                "max_reward": {"type": "number"},
                "limit": {"type": "integer"},
                "summary_only": {"type": "boolean"},
            },
            "required": [],
        },
    },
    {
        "name": "retrieve_similar_posts",
        "description": (
            "Semantic search across ~390k LinkedIn posts (cross-creator, "
            "cross-industry corpus mirrored from Jacquard's linkedin_posts). "
            "Each result is a real LinkedIn post with full text, creator "
            "handle, engagement metrics, and similarity score. Use this to "
            "escape the user's local posting basin — pull precedents from "
            "adjacent creators whose posts actually landed, then study their "
            "form. Call this MANY times per run (topic seeding before "
            "drafting, angle probes during exploration, structural "
            "comparisons against candidates).\n\n"
            "Args:\n"
            "  query (string, required): free-text query (topic, angle, or a "
            "    candidate draft). Will be embedded with text-embedding-3-small.\n"
            "  k (int, optional): number of results, default 10, max 50.\n"
            "  min_reactions (int, optional): filter out posts below this "
            "    reaction count. Default 0. Try 50-200 for outlier-only views.\n"
            "  exclude_creator (string, optional): LinkedIn username to drop "
            "    (e.g. pass the user's own username to avoid retrieving their "
            "    own posts).\n\n"
            "Content-type filtering (e.g. 'avoid announcements') should be "
            "expressed in the query text itself — hand-labeled archetype "
            "filters were retired.\n\n"
            "Returns JSON: {count, posts: [{post_id, post_text, "
            "creator_username, reactions, comments, posted_at, "
            "similarity (0..1)}]}. Sorted by descending similarity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50},
                "min_reactions": {"type": "integer", "minimum": 0},
                "exclude_creator": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mention_resolve",
        "description": (
            "Resolve a LinkedIn username to a mention URN. Returns a JSON "
            "object {name, urn, url} suitable for @[Name](urn:li:member:XXX) "
            "mentions inside post text. Mirrors the `mention-resolve` command "
            "the Jacquard ghostwriter had."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "LinkedIn username (the part after linkedin.com/in/)",
                },
            },
            "required": ["username"],
        },
    },
    {
        "name": "write_result",
        "description": (
            "Submit your final result JSON. This validates the output and ends "
            "the session. The JSON must contain a 'posts' array with at least "
            "one post, each having 'hook', 'text', 'origin', and 'citations'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_json": {
                    "type": "string",
                    "description": "The full result JSON string",
                },
            },
            "required": ["result_json"],
        },
    },
]


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pymupdf."""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        return "[PDF not readable — pymupdf not installed]"
    except Exception as exc:
        return f"[Could not extract PDF text: {exc}]"


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _safe_resolve(workspace_root: Path, rel_path: str) -> Path | None:
    cleaned = rel_path.lstrip("/").lstrip("\\")
    candidate = workspace_root / cleaned
    if not candidate.exists():
        return None
    try:
        _ = candidate.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        norm = os.path.normpath(cleaned)
        if norm.startswith(".."):
            return None
    return candidate


def _exec_list_directory(workspace_root: Path, args: dict) -> str:
    target = _safe_resolve(workspace_root, args.get("path", "."))
    if target is None or not target.is_dir():
        return f"Error: directory not found: {args.get('path', '.')}"
    entries = []
    for item in sorted(target.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            entries.append(f"  {item.name}/")
        elif item.is_file():
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            entries.append(f"  {item.name}  ({size_str})")
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


def _exec_read_file(workspace_root: Path, args: dict) -> str:
    target = _safe_resolve(workspace_root, args.get("path", ""))
    if target is None or not target.is_file():
        return f"Error: file not found: {args.get('path', '')}"
    try:
        if target.suffix.lower() == ".pdf":
            text = _extract_pdf_text(target)
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
        return text
    except Exception as e:
        return f"Error reading file: {e}"


def _exec_search_files(workspace_root: Path, args: dict) -> str:
    query = args.get("query", "")
    directory = args.get("directory", ".")
    target_dir = _safe_resolve(workspace_root, directory)
    if target_dir is None or not target_dir.is_dir():
        return f"Error: directory not found: {directory}"

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    results = []
    for fpath in sorted(target_dir.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in (".txt", ".md", ".json", ".csv"):
            continue
        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    rel = fpath.relative_to(workspace_root)
                    results.append(f"{rel}:{i}: {line.strip()}")
                    if len(results) >= 100:
                        results.append("... (truncated at 100 matches)")
                        return "\n".join(results)
        except Exception:
            continue
    if not results:
        return f"No matches for '{query}'"
    return "\n".join(results)


def _exec_web_search(args: dict) -> str:
    """Web search via Parallel API (Gap 4)."""
    if not PARALLEL_API_KEY:
        return _exec_web_search_fallback(args)

    objective = args.get("objective", "")
    search_queries = args.get("search_queries")
    mode = args.get("mode", "fast")
    max_results = min(args.get("max_results", 10), 40)

    body: dict[str, Any] = {"objective": objective, "mode": mode, "max_results": max_results}
    if search_queries:
        body["search_queries"] = search_queries

    try:
        resp = httpx.post(
            "https://api.parallel.ai/v1beta/search",
            json=body,
            headers={"x-api-key": PARALLEL_API_KEY, "Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[Stelle] Parallel search failed, falling back to ddgs: %s", e)
        return _exec_web_search_fallback(args)

    results = data.get("results", [])
    if not results:
        return f"No results for: {objective}"

    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title', 'N/A')}")
        lines.append(f"URL: {r.get('url', '')}")
        date = r.get("publish_date")
        if date:
            lines.append(f"Date: {date}")
        excerpts = r.get("excerpts", [])
        if excerpts:
            lines.append("Excerpts:")
            for exc in excerpts[:3]:
                lines.append(f"  {exc[:500]}")
        lines.append("")
    return "\n".join(lines)


def _exec_web_search_fallback(args: dict) -> str:
    """Fallback to DuckDuckGo if Parallel API unavailable."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: no search provider available (ddgs not installed, Parallel API key missing)"

    query = args.get("objective", args.get("query", ""))
    max_results = min(args.get("max_results", 5), 10)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results for: {query}"
        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title', '')}")
            lines.append(f"URL: {r.get('href', '')}")
            lines.append(f"Snippet: {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search failed: {e}"


def _exec_fetch_url(args: dict) -> str:
    """URL extraction via Parallel API Extract endpoint, with httpx fallback."""
    url = args.get("url", "")
    if PARALLEL_API_KEY:
        try:
            resp = httpx.post(
                "https://api.parallel.ai/v1beta/extract",
                json={"urls": [url], "mode": "excerpt"},
                headers={"x-api-key": PARALLEL_API_KEY, "Content-Type": "application/json"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                content = results[0].get("content") or ""
                excerpts = results[0].get("excerpts", [])
                text = content or "\n\n".join(excerpts)
                if text:
                    if len(text) > MAX_FETCH_CHARS:
                        text = text[:MAX_FETCH_CHARS] + f"\n\n... [truncated at {MAX_FETCH_CHARS} chars]"
                    return text
        except Exception as e:
            logger.warning("[Stelle] Parallel extract failed, falling back to httpx: %s", e)

    # Fallback: stdlib HTML-to-text extractor. Strips nav/script/footer/etc,
    # preserves block structure via line breaks. Readable plain text, not raw HTML.
    try:
        from backend.src.utils.fetch_url import fetch_url as _fetch_readable
        r = _fetch_readable(url, max_chars=MAX_FETCH_CHARS)
        if r.get("status", 0) >= 400 or r.get("status", 0) == 0:
            return f"Error fetching URL ({r.get('status')}): {r.get('text','')}"
        title = r.get("title") or ""
        body = r.get("text") or ""
        header = f"URL: {r.get('url', url)}\n" + (f"Title: {title}\n" if title else "")
        if r.get("truncated"):
            body = body + f"\n\n... [truncated at {MAX_FETCH_CHARS} chars]"
        return header + "\n" + body if header else body
    except Exception as e:
        return f"Error fetching URL: {e}"


def _exec_write_file(workspace_root: Path, args: dict) -> str:
    rel_path = args.get("path", "")
    content = args.get("content", "")
    norm = os.path.normpath(rel_path.lstrip("/").lstrip("\\"))
    if norm.startswith(".."):
        return "Error: path traversal not allowed"
    target = workspace_root / norm
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {rel_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _exec_edit_file(workspace_root: Path, args: dict) -> str:
    target = _safe_resolve(workspace_root, args.get("path", ""))
    if target is None or not target.is_file():
        return f"Error: file not found: {args.get('path', '')}"
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    if not old_text:
        return "Error: old_text is required"
    try:
        content = target.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return f"Error: old_text not found in {args.get('path', '')}"
        if count > 1:
            return f"Error: old_text matches {count} locations — must be unique"
        content = content.replace(old_text, new_text, 1)
        target.write_text(content, encoding="utf-8")
        return f"Edited {args.get('path', '')} — replaced {len(old_text)} chars with {len(new_text)} chars"
    except Exception as e:
        return f"Error editing file: {e}"


def _exec_bash(workspace_root: Path, args: dict) -> str:
    """Run a shell command in the workspace (Gap 7)."""
    command = args.get("command", "")
    if not command:
        return "Error: command is required"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=str(workspace_root),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if not output.strip():
            output = "(no output)"
        if len(output) > MAX_BASH_OUTPUT_CHARS:
            output = output[:MAX_BASH_OUTPUT_CHARS] + f"\n... [truncated at {MAX_BASH_OUTPUT_CHARS} chars]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except Exception as e:
        return f"Error running command: {e}"


def _exec_subagent(args: dict) -> str:
    """Spawn a Claude Haiku sub-agent for focused exploration (Gap 5)."""
    task = args.get("task", "")
    context = args.get("context", "")
    if not task:
        return "Error: task is required"

    prompt = task
    if context:
        prompt = f"Context:\n{context}\n\nTask:\n{task}"

    try:
        resp = _call_with_retry(lambda: _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ))
        return resp.content[0].text if resp.content else "(no response from sub-agent)"
    except Exception as e:
        return f"Sub-agent error: {e}"


_WORKSPACE_ROOT_PATHS = frozenset({"", ".", "/", "./"})


def _dispatch_list_directory(root, args):
    """Path-aware routing:
    - Root path (``""`` / ``"."``) in database mode → HTTP proxy root
      listing (shows Jacquard's FOC user slugs + shared roots). Without
      this Stelle gets the fly-local scratch workspace on her very first
      call and has no hint the database mounts exist.
    - database mount paths → HTTP proxy to virio-api workspace
    - Everything else (scratch/, loose dirs) → fly-local SandboxFs

    Same routing rule as _dispatch_write_file so scratch files Stelle
    wrote with ``write_file`` are listable via the same path."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        path = args.get("path", "") or ""
        if path in _WORKSPACE_ROOT_PATHS:
            return _lfs.exec_list_directory(root, {"path": ""})
        if _lfs.is_lineage_path(path):
            return _lfs.exec_list_directory(root, args)
    return _exec_list_directory(root, args)


def _dispatch_read_file(root, args):
    """Path-aware routing:
    - database mount paths → HTTP proxy (transcripts, engagement JSON, etc.)
    - Scratch paths → fly-local SandboxFs (reads the v1/v2/v3 drafts
      Stelle wrote via write_file in the same run)"""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lfs.exec_read_file(root, args)
    return _exec_read_file(root, args)


def _lineage_write_blocked_message(path: str) -> str:
    """Explain why a write into a database mount is refused and where
    Stelle should write instead (scratch paths work fine)."""
    return (
        f"Error: cannot write to `{path}` — that path is inside database's "
        "workspace, which is READ-ONLY to Stelle (it's the client's space).\n\n"
        "Your options:\n"
        "  • FINAL drafts: call `submit_draft(user_slug, content, "
        "scheduled_date, process_notes)` — the only write tool that reaches database.\n"
        "  • Scratch / working notes / draft iterations: write to any path "
        "OUTSIDE the database mount tree. Good choices: `scratch/post1-v1.md`, "
        "`scratch/plan.md`, `notes/brainstorm.md`. Those land on your "
        "fly-local SandboxFs, persist for the duration of the run, and "
        "you can `read_file` them back normally.\n\n"
        "Paths that route to database (read-only for write): anything under "
        "`transcripts/`, `research/`, `engagement/`, `reports/`, `context/`, "
        "`posts/`, `edits/`, `tone/`, `strategy/`, or the shared "
        "`conversations/`, `slack/`, `tasks/`, `.pi/`."
    )


def _dispatch_write_file(root, args):
    """Writes route by path:

    - Paths inside a database mount (transcripts/, posts/, strategy/, …) →
      refused with an error redirecting to ``submit_draft`` or a scratch
      path.
    - Scratch paths (anything NOT a database mount — e.g. ``scratch/*``,
      ``notes/*``, loose files) → fly-local SandboxFs.
    - In local (non-database) mode, always fly-local.

    Preserves Stelle's classic scratch-file iteration pattern
    (``scratch/post1-v1.md`` → ``scratch/post1-v2.md`` → …) while
    keeping the client data source read-only from her perspective.
    Only ``submit_draft`` persists finished posts."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lineage_write_blocked_message(path)
    return _exec_write_file(root, args)


def _dispatch_edit_file(root, args):
    """Same path-aware routing as _dispatch_write_file."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lineage_write_blocked_message(path)
    return _exec_edit_file(root, args)


def _dispatch_search_files(root, args):
    """Path-aware routing based on the ``directory`` arg. Searches over
    database mounts use the HTTP endpoint; searches over scratch paths
    use the local ripgrep."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        directory = args.get("directory", "") or ""
        if _lfs.is_lineage_path(directory):
            return _lfs.exec_search_files(root, args)
    return _exec_search_files(root, args)


def _dispatch_bash(root, args):
    """Bash in data-source mode is disabled — the workspace is a virtual
    view over Supabase/GCS, not a real filesystem, so shell pipelines
    can't see it. Direct agents to structured tools."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        cmd = (args.get("command") or "")[:120]
        return (
            "Error: bash is disabled — the workspace is a virtual view "
            "over Supabase/GCS, not a real filesystem. Use structured tools:\n"
            "  • read/list/edit/write_file for filesystem ops\n"
            "  • search_files for grep-style search\n"
            "  • web_search / fetch_url for web calls\n"
            f"(rejected command: {cmd!r})"
        )
    return _exec_bash(root, args)


def _dispatch_mention_resolve(root, args):
    """Always goes through HTTP — whether local or database mode. The
    database endpoint wraps the same Supabase + APImaestro resolver Pi
    used, so behavior is identical. Falls through to a local error when
    not in database mode (Stelle never had this tool outside database)."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        return _lfs.exec_mention_resolve(root, args)
    return "Error: mention_resolve is only available in database mode"


def apply_castorice_to_submit_args(
    company_keyword: str, args: dict
) -> dict:
    """Run Castorice fact-check + strategic-fit on a submit_draft args dict
    and return a forwarded args dict where:
      - ``content`` is the fact-check-corrected text
      - ``pre_revision_content`` is the original (pre-correction) text
      - ``process_notes`` is Stelle's own audit trail (whatever she
        passed in either ``process_notes`` or the legacy ``why_post`` slot)
      - ``why_post`` is Castorice's strategic_fit_note, grounded in the
        latest Cyrene brief for this FOC (or None when no brief exists)
      - ``fact_check_report`` is Castorice's fact-check transcript
      - ``citation_comments`` is the list of source-citation strings

    Shared between the native API path (the per-run wrapper in
    ``stelle.py``) and the CLI MCP path (``stelle_server.py``'s
    ``_handle_submit_draft``). Pre-2026-05-01 the CLI path bypassed
    Castorice entirely — Stelle's why_post arg went straight through
    unchanged, and operators saw Stelle-style audit text (mentioning
    Aglaea/Irontomb results) in the operator-facing why_post field
    instead of Castorice's Cyrene-brief-grounded strategic-fit
    verdict. Wiring the CLI path through here closes that gap.

    Never raises — failures fall through to graceful degradation
    (uncorrected content, empty strategic_fit_note, etc.) so submit
    doesn't 500 on Castorice hiccups.
    """
    content = args.get("content") or ""
    if not content:
        # Caller checks; we just bail with the args unchanged.
        return dict(args)

    # Castorice fact-check
    try:
        from backend.src.agents.castorice import Castorice
        _castorice = Castorice()
        fc = _castorice.fact_check_post(company_keyword, content)
        corrected = fc.get("corrected_post") or content
        report = fc.get("report") or ""
        citation_comments = fc.get("citation_comments") or []
    except Exception as _e:
        logger.warning("[Stelle] Castorice fact-check failed: %s", _e)
        _castorice = None
        corrected = content
        report = f"[Castorice unavailable: {str(_e)[:200]}]"
        citation_comments = []

    # Strategic-fit pass
    strategic_fit_note = ""
    if _castorice is not None:
        try:
            import os as _os
            _user_id = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
            fit = _castorice.analyze_strategic_fit(
                company_keyword=company_keyword,
                post_content=corrected,
                user_id=_user_id,
            )
            strategic_fit_note = (fit.get("strategic_fit_note") or "").strip()
        except Exception as _e:
            logger.warning("[Stelle] Castorice strategic-fit failed: %s", _e)
            strategic_fit_note = ""

    # Route each piece to its own field. Stelle's audit trail can
    # arrive in either ``process_notes`` (new schema arg) or
    # ``why_post`` (legacy arg name) — both land in process_notes
    # which is hidden by default in the operator UI.
    forwarded = dict(args)
    forwarded["content"] = corrected
    forwarded["pre_revision_content"] = content

    stelle_audit_trail = (
        forwarded.pop("process_notes", None)
        or forwarded.pop("why_post", None)
        or ""
    ).strip() or None
    forwarded["process_notes"] = stelle_audit_trail

    # Castorice owns the operator-facing why_post. None when no brief
    # exists — the UI hides the section in that case.
    forwarded["why_post"] = strategic_fit_note or None

    # Fact-check report → own column. Empty when Castorice ran cleanly.
    forwarded["fact_check_report"] = report.strip() or None

    # Citations (list[str]) — passed as a structured list.
    forwarded["citation_comments"] = (
        citation_comments if citation_comments else None
    )

    return forwarded


def _dispatch_submit_draft(root, args):
    """Final-draft submission — branches by mode.

    - **database mode** (DATABASE_COMPANY_ID + direct Supabase/GCS creds):
      inserts a row into Jacquard's ``drafts`` Supabase table with
      status=``review`` under the FOC user resolved from ``user_slug``.
      The draft shows up in Jacquard's review UI alongside
      Jacquard-native drafts.
    - **Local mode** (amphoreus.app standalone, tests): lands a row in
      the local ``local_posts`` table. The operator reviews at
      amphoreus.app and pushes to Ordinal from there.

    In both modes, a markdown mirror is written to
    ``output/{company}/drafts/{draft_id}.md`` for out-of-band inspection.

    The session-level wrapper in stelle.py (_stelle_submit_draft_with_castorice)
    runs Castorice fact-check on ``content`` before forwarding here, so by
    the time we land, the content is already corrected and why_post carries
    the fact-check report.

    Args (from the submit_draft tool schema):
        user_slug (str, required)   FOC user this draft is attributed to.
                                    In pure-local mode the slug == company
                                    keyword; in database-read mode it's
                                    the Jacquard FOC-user slug.
        content (str, required)     Final post markdown.
        scheduled_date (str, opt)   ISO YYYY-MM-DD calendar slot.
        publication_order (int, opt)
        why_post (str, opt)         Rationale + fact-check report.
        approver_user_ids (list, opt)  Ignored locally — approvals happen
                                       in the Amphoreus Posts tab, not
                                       through a separate approver UUID.
    """
    import os as _os
    import uuid as _uuid
    from pathlib import Path as _Path

    user_slug = (args.get("user_slug") or "").strip()
    # Env fallback: stelle_runner sets DATABASE_USER_SLUG when the
    # generate endpoint resolved a specific FOC user for this run. If the
    # LLM omits ``user_slug`` in its tool call (not always reliable), we
    # still want the draft attributed correctly — otherwise local_posts
    # rows land with user_id=None and are invisible in per-FOC views.
    if not user_slug:
        env_slug = (_os.environ.get("DATABASE_USER_SLUG") or "").strip()
        if env_slug:
            user_slug = env_slug
            logger.info("[submit_draft] using DATABASE_USER_SLUG env fallback: %r", user_slug)

    # Authoritative env override: DATABASE_USER_UUID, set by
    # stelle_runner from --user-id, is the direct FOC UUID. We trust
    # this over any slug-based resolution because (a) the Jacquard
    # ``users.slug`` column is sometimes NULL, causing slug fallback
    # to fail silently, and (b) the UUID was resolved by the ghost-
    # writer endpoint at request time against the same source of truth
    # list_foc_users uses. When this env var is set we skip slug
    # resolution entirely and stamp user_id from it directly.
    env_user_uuid = (_os.environ.get("DATABASE_USER_UUID") or "").strip()
    content = args.get("content") or ""
    if not content:
        return "Error: content is required"

    # Determine company + FOC user for bookkeeping. Pull company from env
    # set by stelle_runner; derive user_id from the user_slug arg if the
    # run is scoped to a specific FOC user (disambiguates Trimble-Heather
    # from Trimble-Mark, Commenda-Logan from Commenda-Sam).
    company = (
        _os.environ.get("STELLE_COMPANY_KEYWORD", "").strip()
        or user_slug
        or "unknown"
    )

    user_id: str | None = None
    # Priority 1: DATABASE_USER_UUID env (direct, authoritative).
    if env_user_uuid:
        user_id = env_user_uuid
        logger.info("[submit_draft] using DATABASE_USER_UUID env: %s", env_user_uuid)
    # Priority 2: slug-based resolution (falls back to this when the
    # UUID env wasn't set — older run paths, or ghostwriter calls
    # that predate the UUID-env fix).
    elif user_slug:
        try:
            from backend.src.lib.company_resolver import resolve_to_company_and_user
            # Prefer the full slug path first (works for Amphoreus
            # pseudo-slugs like ``trimble-heather``).
            cu, uu = resolve_to_company_and_user(user_slug)
            user_id = uu
            # Defensive: if the user_slug carried a company prefix we
            # didn't have from env, adopt it.
            if not company or company == "unknown":
                if cu:
                    company = cu
        except Exception as _e:
            logger.debug("[submit_draft] resolver failed for %r: %s", user_slug, _e)

    # Last-resort safety: never write an orphan at a multi-FOC company.
    # Two-layer check:
    #
    #   Primary (unchanged): if DATABASE_COMPANY_ID is set, refuse any
    #   NULL-user_id write. The ghostwriter endpoint sets this env var
    #   on every data-source run, so the guard fires on the full
    #   runtime path.
    #
    #   Secondary (new, 2026-04-22 Trimble fix): ALSO refuse NULL-user
    #   writes when we can confirm the company has multiple FOC users,
    #   regardless of whether DATABASE_COMPANY_ID is set. Belt-and-
    #   suspenders against the ghostwriter guard ever being bypassed —
    #   even if a future code path calls submit_draft without setting
    #   the env var (tests, scripts, direct MCP calls), orphans at
    #   multi-FOC companies are still blocked here.
    if user_id is None:
        co_env = (_os.environ.get("DATABASE_COMPANY_ID") or "").strip()
        # Secondary check: query FOC roster directly for the company
        # we're writing against. Never swallows — if the lookup fails
        # AND DATABASE_COMPANY_ID is set, the primary guard still
        # refuses. If the lookup succeeds and reveals multi-FOC, refuse
        # regardless.
        _posting_count = None
        try:
            from backend.src.agents.jacquard_direct import list_foc_users as _sd_foc
            _guard_co = co_env or company
            if _guard_co:
                _foc_rows = _sd_foc(_guard_co) or []
                _posting_count = sum(
                    1 for u in _foc_rows if u.get("posts_content")
                )
        except Exception as _sd_exc:
            logger.debug(
                "[submit_draft] FOC-count lookup failed (non-fatal): %s", _sd_exc,
            )

        if co_env or (_posting_count is not None and _posting_count > 1):
            logger.error(
                "[submit_draft] refusing to write orphan draft: no user_id "
                "resolved (user_slug=%r, DATABASE_USER_SLUG=%r, "
                "DATABASE_USER_UUID=%r, DATABASE_COMPANY_ID=%r, company=%r, "
                "posting_foc_count=%r). Re-run with a user-qualified slug "
                "(e.g. flora-weber) or pass userId.",
                args.get("user_slug"),
                _os.environ.get("DATABASE_USER_SLUG"),
                env_user_uuid,
                co_env,
                company,
                _posting_count,
            )
            return (
                "Error: cannot write draft — no FOC user identified for this "
                "run. This run was started without a target user at a company "
                "that has multiple FOCs. Re-run via /api/ghostwriter/generate "
                "with a user-qualified slug (e.g. `flora-weber`) or an "
                "explicit `userId` in the request body."
            )

    draft_id = str(_uuid.uuid4())
    scheduled_date = args.get("scheduled_date") or None
    publication_order = args.get("publication_order")
    # 2026-04-29 split: ``why_post`` is now operator-facing only
    # (Castorice strategic_fit_note). ``process_notes`` is Stelle's audit
    # trail (collapsed in UI). ``fact_check_report`` is Castorice's
    # fact-check transcript (own UI expander). All three are populated by
    # the session-level wrapper ``_stelle_submit_draft_with_castorice``;
    # raw Stelle calls (no wrapper, e.g. tests) just see the same fields
    # passed through.
    why_post = args.get("why_post") or None
    process_notes = args.get("process_notes") or None
    fact_check_report = args.get("fact_check_report") or None
    citation_comments = args.get("citation_comments") or None

    # Title = first non-empty line stripped of markdown headers, max 200 chars.
    title = None
    for _line in content.splitlines():
        stripped = _line.strip()
        if stripped:
            title = stripped.lstrip("#").strip()[:200]
            break

    # 1) Persist the draft to Amphoreus's local_posts. The operator
    # reviews drafts at amphoreus.app/posts and pushes to Ordinal from
    # there. Drafts never leave Amphoreus's side.
    #
    # ``pre_revision_content`` stores the text Stelle originally passed
    # (before the session-level Castorice wrapper corrected it). That's
    # the key ``_process_result`` uses to dedup: its write_result post
    # array carries pre-Castorice text too, so matching on
    # ``pre_revision_content`` catches the "same post, two write paths"
    # case even when Castorice output differs run-to-run.
    pre_revision_content = args.get("pre_revision_content") or None
    destination = "Amphoreus Posts tab"
    try:
        from backend.src.db.local import create_local_post
        create_local_post(
            post_id=draft_id,
            company=company,
            user_id=user_id,
            content=content,
            title=title,
            status="draft",
            why_post=why_post,
            process_notes=process_notes,
            fact_check_report=fact_check_report,
            citation_comments=(
                citation_comments
                if isinstance(citation_comments, list)
                else None
            ),
            scheduled_date=scheduled_date,
            publication_order=(
                publication_order if isinstance(publication_order, int) else None
            ),
            pre_revision_content=pre_revision_content,
        )
    except Exception as exc:
        logger.exception("[submit_draft] create_local_post failed: %s", exc)
        return f"Error: failed to persist draft to local_posts: {exc}"

    # 1a) Back-link convergence-log rows to this local_posts row. Every
    # Irontomb / Aglaea call during the iteration chain was logged with
    # ``local_post_id=NULL`` and the draft's content hash. Now that we
    # have the final ``draft_id``, update all rows whose ``draft_hash``
    # matches the final content to point at it. Earlier-iteration rows
    # (different hashes) stay unlinked; a follow-up backfill job can
    # chain them via temporal proximity if we ever need it. Non-fatal.
    try:
        from backend.src.services.convergence_log import backfill_local_post_id
        backfill_local_post_id(draft_id, content)
    except Exception as exc:
        logger.debug("[submit_draft] convergence backfill failed (non-fatal): %s", exc)

    # (Fire-and-forget ``draft_embedding`` stamp removed 2026-04-23.
    # The v1 ``draft_publish_matcher`` that read that column was
    # retired; the replacement ``draft_match_worker`` reads from the
    # ``local_posts.embedding`` column instead, which is written
    # inline by ``db/local.py::_mirror_embed_local_post_content`` at
    # insert time — no separate threaded embed spawn needed here.)

    # 2) Markdown file mirror — one file per draft, grouped by company,
    # for out-of-band inspection + push tooling that expects files on disk.
    md_dir = _Path("output") / company / "drafts"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{draft_id}.md"
    header_lines = [
        f"# {title or 'Untitled draft'}",
        "",
        f"_draft_id: {draft_id}_",
        f"_company: {company}_",
    ]
    if user_slug:
        header_lines.append(f"_user_slug: {user_slug}_")
    if scheduled_date:
        header_lines.append(f"_scheduled_date: {scheduled_date}_")
    if publication_order is not None:
        header_lines.append(f"_publication_order: {publication_order}_")
    header_lines.append("")
    if why_post:
        # Operator-facing rationale (Castorice strategic_fit_note).
        header_lines.extend(["## Why this post", "", why_post, ""])
    if fact_check_report:
        header_lines.extend(["## Castorice fact-check", "", fact_check_report, ""])
    if process_notes:
        # Stelle's audit trail. Kept inline in the markdown mirror — the
        # markdown file is a debugging artefact, so the collapse-by-default
        # logic in the UI doesn't need to apply here.
        header_lines.extend(["## Process notes (Stelle)", "", process_notes, ""])
    header_lines.extend(["## Content", "", content])
    try:
        md_path.write_text("\n".join(header_lines), encoding="utf-8")
    except Exception as exc:
        logger.debug("[submit_draft] failed to write markdown mirror: %s", exc)

    return (
        f"Draft submitted to {destination}.\n"
        f"  draft_id: {draft_id}\n"
        f"  company: {company}\n"
        f"  user_slug: {user_slug or '(none)'}\n"
        f"  scheduled_date: {scheduled_date or '(unset)'}\n"
        f"  title: {(title or '')[:80]}\n"
        "\n"
        "The operator will review in the Posts tab and push to Ordinal "
        "from there. Nothing has been sent to LinkedIn yet."
    )


def _dispatch_query_observations(root, args):
    """Query scored observations. In database mode goes through the remote
    workspace's ``/observations`` endpoint (backed by linkedin_posts +
    linkedin_reactions + ICP reports). Outside database mode, falls back
    to the in-process Analyst implementation when scored_observations are
    pre-computed for this run — else returns empty."""
    from backend.src.agents import database_client as _lfs
    if _lfs.is_database_mode():
        return _lfs.exec_query_observations(root, args)
    # Non-database fallback — preserve Amphoreus's local behavior.
    try:
        from backend.src.agents.analyst import _tool_query_observations as _q
        # If the agent loop wired scored_observations into module state,
        # use them; otherwise return an empty result rather than crash.
        scored = globals().get("_scored_observations_for_run") or []
        return _q(args, scored)
    except Exception as e:
        import json as _json
        return _json.dumps({"count": 0, "observations": [], "error": str(e)})


def _dispatch_retrieve_similar_posts(root, args):
    """Semantic search over the Amphoreus post_embeddings corpus (~390k LinkedIn
    posts mirrored from Jacquard's linkedin_posts + text-embedding-3-small).

    This is the cross-creator precedence tool. Stelle uses it to escape the
    user's local posting basin; Irontomb uses the same path to ground its
    reactions in real engagement numbers instead of hallucinated patterns.

    Failures are degraded to a structured JSON error rather than a raise, so
    a transient OpenAI / Supabase hiccup can't kill the whole run.
    """
    import json as _json
    try:
        from backend.src.services.post_retrieval import retrieve_similar_posts
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": f"import failed: {exc}"})

    query = (args or {}).get("query") or ""
    if not query.strip():
        return _json.dumps({"count": 0, "posts": [], "error": "query is required"})

    k = int((args or {}).get("k") or 10)
    k = max(1, min(k, 50))
    min_reactions = int((args or {}).get("min_reactions") or 0)
    exclude_creator = (args or {}).get("exclude_creator") or None

    try:
        rows = retrieve_similar_posts(
            query=query,
            k=k,
            min_reactions=min_reactions,
            exclude_creator=exclude_creator,
        )
    except Exception as exc:
        return _json.dumps({"count": 0, "posts": [], "error": str(exc)[:400]})

    return _json.dumps(
        {"count": len(rows), "posts": rows},
        default=str,
    )


_TOOL_HANDLERS = {
    "list_directory": _dispatch_list_directory,
    "read_file": _dispatch_read_file,
    "search_files": _dispatch_search_files,
    "web_search": lambda root, args: _exec_web_search(args),
    "fetch_url": lambda root, args: _exec_fetch_url(args),
    "write_file": _dispatch_write_file,
    "edit_file": _dispatch_edit_file,
    "bash": _dispatch_bash,
    "mention_resolve": _dispatch_mention_resolve,
    "query_observations": _dispatch_query_observations,
    "retrieve_similar_posts": _dispatch_retrieve_similar_posts,
    "submit_draft": _dispatch_submit_draft,
}


# ---------------------------------------------------------------------------
# Output validation (Gap 1 + Gap 6)
# ---------------------------------------------------------------------------

def _validate_output(result: dict) -> tuple[bool, list[str], list[str]]:
    """Validate the agent's output JSON, matching jacquard's output_validator."""
    errors: list[str] = []
    warnings: list[str] = []
    posts = result.get("posts", [])

    if not posts:
        return False, ["No posts produced"], warnings

    for i, post in enumerate(posts):
        p = f"Post {i + 1}"
        text = post.get("text", "")
        hook = post.get("hook", "")

        if not hook:
            errors.append(f"{p}: Missing hook")
        elif len(hook) > 200:
            errors.append(f"{p}: Hook over 200 chars ({len(hook)})")

        if not text:
            errors.append(f"{p}: Missing text")
        elif len(text) > 3000:
            errors.append(f"{p}: Over 3000 chars ({len(text)}) — LinkedIn hard limit")

        if not post.get("origin"):
            errors.append(f"{p}: Missing origin")

        hook_variants = post.get("hook_variants")
        if hook_variants is not None and not isinstance(hook_variants, list):
            warnings.append(f"{p}: hook_variants should be an array")
        elif hook_variants is None or len(hook_variants) == 0:
            warnings.append(f"{p}: No hook_variants — consider adding 2-3 alternatives")

        citations = post.get("citations")
        if citations is None:
            errors.append(f"{p}: Missing citations key")
        elif not isinstance(citations, list):
            errors.append(f"{p}: citations must be an array")
        elif len(citations) == 0:
            warnings.append(f"{p}: Empty citations — verify post has no factual claims")

    if not result.get("verification"):
        warnings.append("Missing verification — agent should prove it used all available material")

    return len(errors) == 0, errors, warnings


# ---------------------------------------------------------------------------
# LLM-based draft validation (Claude Haiku — cheap, fast)
# ---------------------------------------------------------------------------

def _validate_draft_with_llm(post_text: str, company_keyword: str) -> dict:
    """Structural validation of a single post. No LLM call.

    Content quality (AI patterns, banned phrases) is the constitutional
    verifier's domain — it runs post-generation with learned principle
    weights. This function handles only structural checks that don't
    need an LLM: character count and client-specific preference violations.
    """
    result = {"needs_correction": False, "issues": []}

    # --- Character count ---
    # LinkedIn's platform cap is the only hard limit. No hand-tuned floor:
    # length is learnable from the client's own published posts, and Stelle
    # already reads them.
    char_count = len(post_text)
    if char_count > 3000:
        result["needs_correction"] = True
        result["issues"].append({
            "type": "char_count",
            "description": f"Post is {char_count} chars — exceeds LinkedIn's 3000 char limit",
            "severity": "critical",
            "offending_text": "",
            "suggested_fix": f"Cut {char_count - 2800} characters",
        })

    return result


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _resolve_linkedin_username(
    company_keyword: str, user_id: str | None = None,
) -> str | None:
    """Return the creator's LinkedIn handle for ``company_keyword``.

    Resolution order:
      0. If ``user_id`` is supplied (caller already knows the FOC) or
         the ``DATABASE_USER_UUID`` env var is set (stelle_runner
         populates this when the ghostwriter endpoint resolved a
         target FOC), look up ``users.linkedin_url WHERE id=<id>``
         directly. This short-circuits past the multi-FOC ambiguity
         problem that used to bite Trimble / Commenda / Virio runs:
         with N > 1 FOCs per company, path 1b below refused to
         auto-pick, returning None even when the caller had the right
         user UUID in hand (or in env).
      1. Jacquard ``users.linkedin_url`` via slug / UUID resolution —
         works for every FOC-scoped slug (e.g. ``hume-ai-andrew``,
         ``trimble-heather``, ``innovocommerce-sachil``) via
         ``resolve_to_company_and_user``.
      1b. Bare company identifier + single-FOC company — auto-pick
         that user's handle. Skipped silently for multi-FOC companies.
      2. Legacy ``memory/<slug>/linkedin_username.txt`` file — kept as
         a last-resort fallback for slugs that predate per-FOC
         resolution and still have a memory dir on disk.
      3. None — caller decides what to do.

    Env fallback rationale: many legacy internal callers in stelle.py
    pass only ``company_keyword``. When Stelle runs in FOC-targeted
    mode, ``DATABASE_USER_UUID`` is set by ``stelle_runner`` and is
    the authoritative FOC identifier for the run. Reading it here
    unbreaks every unmigrated caller without requiring a wholesale
    refactor. When the env var is unset (company-wide runs), we fall
    through to the original slug/UUID resolution, unchanged.

    Side effect: none. Safe to call freely; each call is at most one
    PostgREST round-trip.
    """
    import os as _os
    import re as _re

    def _extract_handle(url: str | None) -> str | None:
        if not url:
            return None
        m = _re.search(r"linkedin\.com/in/([^/?#]+)", url.strip())
        return m.group(1).strip().lower().rstrip("/") if m else None

    # 0) Explicit user_id short-circuit — the caller already resolved
    #    which FOC they want. Skip the slug/UUID dance entirely.
    #    Falls back to DATABASE_USER_UUID env var (set by stelle_runner
    #    in FOC-targeted mode) so unmigrated callers benefit too.
    if not user_id:
        _env_uid = (_os.environ.get("DATABASE_USER_UUID") or "").strip()
        if _env_uid:
            user_id = _env_uid

    if user_id:
        try:
            from backend.src.db.supabase_client import get_amphoreus_supabase
            jcq = get_amphoreus_supabase()
            rows = (
                jcq.table("users")
                   .select("linkedin_url")
                   .eq("id", user_id)
                   .limit(1)
                   .execute()
                   .data
                or []
            )
            if rows:
                h = _extract_handle(rows[0].get("linkedin_url"))
                if h:
                    return h
        except Exception as exc:
            logger.debug(
                "[Stelle] username lookup by user_id=%s failed: %s",
                user_id, exc,
            )

    # 1) Jacquard: prefer exact user resolution (FOC-scoped slugs like
    #    ``hume-ai-andrew`` → Andrew's linkedin_url).
    try:
        from backend.src.lib.company_resolver import resolve_to_company_and_user
        _company_uuid, _user_id = resolve_to_company_and_user(company_keyword)
        if _user_id:
            from backend.src.db.supabase_client import get_amphoreus_supabase
            jcq = get_amphoreus_supabase()
            rows = (
                jcq.table("users")
                   .select("linkedin_url")
                   .eq("id", _user_id)
                   .limit(1)
                   .execute()
                   .data
                or []
            )
            if rows:
                h = _extract_handle(rows[0].get("linkedin_url"))
                if h:
                    return h
        # 1b) Bare company identifier (UUID or company-only slug). When
        # the company has exactly one tracked FOC, auto-pick that
        # user's handle — covers single-FOC clients like Hensley
        # Biostats that the UI sometimes hands us as a raw UUID. Skip
        # if multiple FOCs (ambiguous; caller needs to pass the
        # FOC-scoped slug OR the user_id kwarg).
        if _company_uuid:
            from backend.src.agents.jacquard_direct import list_foc_users
            foc_users = list_foc_users(_company_uuid) or []
            if len(foc_users) == 1:
                h = _extract_handle(foc_users[0].get("linkedin_url"))
                if h:
                    return h
    except Exception as exc:
        logger.debug(
            "[Stelle] Jacquard username lookup failed for %s: %s",
            company_keyword, exc,
        )
    # 2) Legacy disk fallback.
    try:
        username_path = P.linkedin_username_path(company_keyword)
        if username_path.exists():
            v = username_path.read_text().strip()
            if v:
                return v
    except Exception:
        pass
    return None


def _fetch_linkedin_profile(company_keyword: str) -> str | None:
    """Fetch the client's LinkedIn profile summary via APIMaestro."""
    username = _resolve_linkedin_username(company_keyword)
    if not username or not APIMAESTRO_KEY or not APIMAESTRO_HOST:
        return None

    logger.info("[Stelle] Fetching LinkedIn profile for @%s via APIMaestro...", username)
    try:
        resp = httpx.get(
            f"https://{APIMAESTRO_HOST}/profile/detail",
            params={"username": username},
            headers={"X-API-Key": APIMAESTRO_KEY},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[Stelle] APIMaestro profile fetch failed: %s", e)
        return None

    if not data.get("success"):
        return None

    payload = data.get("data", {})
    bi = payload.get("basic_info", {})
    loc = bi.get("location", {})

    lines: list[str] = ["# LinkedIn Profile (Live)"]
    name = bi.get("fullname") or bi.get("first_name")
    headline = bi.get("headline")
    if headline:
        lines.append(f"**Headline:** {headline}")
    if name:
        lines.append(f"**Name:** {name}")

    location_full = loc.get("full")
    if location_full:
        lines.append(f"**Location:** {location_full}")

    company = bi.get("current_company")
    if company:
        lines.append(f"**Current company:** {company}")

    followers = bi.get("follower_count")
    if isinstance(followers, int):
        lines.append(f"**Followers:** {followers:,}")

    connections = bi.get("connection_count")
    if isinstance(connections, int):
        lines.append(f"**Connections:** {connections:,}")

    about = (bi.get("about") or "").strip()
    if about:
        lines.append("")
        lines.append("**About:**")
        lines.append(about[:2000])

    experience = payload.get("experience", [])
    if experience:
        lines.append("")
        lines.append("**Recent experience:**")
        for exp in experience[:3]:
            title = exp.get("title", "")
            comp = exp.get("company", "")
            desc = " at ".join(p for p in (title, comp) if p)
            dur = exp.get("duration", "")
            parts = [p for p in (desc, dur) if p]
            if parts:
                lines.append("  - " + " | ".join(parts))

    hashtags = bi.get("creator_hashtags", [])
    if hashtags:
        lines.append(f"**Creator hashtags:** {', '.join(str(h) for h in hashtags[:10])}")

    summary = "\n".join(lines)
    logger.info("[Stelle] LinkedIn profile fetched for @%s (%d chars)", username, len(summary))
    return summary


def _fetch_published_posts(company_keyword: str) -> tuple[list[dict], set[str]]:
    """Fetch the client's own published posts from Supabase. Returns (posts, dates)."""
    username = _resolve_linkedin_username(company_keyword)
    if not username:
        logger.info("[Stelle] No LinkedIn handle for %s — skipping Supabase", company_keyword)
        return [], set()

    if not SUPABASE_URL or not SUPABASE_KEY:
        return [], set()

    logger.info("[Stelle] Fetching published posts for @%s from Supabase...", username)
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/linkedin_posts",
            params={
                "select": "hook,post_text,posted_at,total_reactions,total_comments,"
                          "total_reposts,engagement_score,is_outlier",
                "creator_username": f"eq.{username}",
                "is_company_post": "eq.false",
                "post_text": "not.is.null",
                "order": "posted_at.desc",
                "limit": "50",
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        logger.warning("[Stelle] Supabase fetch failed: %s", e)
        return [], set()

    posts = []
    published_dates: set[str] = set()
    for row in rows:
        post_text = (row.get("post_text") or "").strip()
        if not post_text:
            continue
        hook = (row.get("hook") or "").strip()
        posted_at = (row.get("posted_at") or "")[:10]
        if posted_at:
            published_dates.add(posted_at)
        reactions = row.get("total_reactions") or 0
        comments = row.get("total_comments") or 0
        reposts = row.get("total_reposts") or 0
        eng = row.get("engagement_score") or 0
        eng_display = eng / 100 if eng else 0
        outlier = row.get("is_outlier") or False

        engagement_line = (
            f"Reactions: {reactions} | Comments: {comments} | "
            f"Reposts: {reposts} | Engagement: {eng_display:.2f}"
        )
        if outlier:
            engagement_line += " | OUTLIER (top 20%)"

        slug = re.sub(r"[^a-z0-9]+", "-", (hook or "untitled")[:40].lower()).strip("-")
        filename = f"{posted_at}-{slug}.md"
        existing = {p["filename"] for p in posts}
        if filename in existing:
            filename = f"{posted_at}-{slug}-{len(posts)}.md"
        content = f"# {hook or 'Untitled'}\nDate: {posted_at}\n{engagement_line}\n\n{post_text}"
        posts.append({"filename": filename, "content": content})

    logger.info("[Stelle] Fetched %d published posts for @%s", len(posts), username)
    return posts, published_dates


MAX_VOICE_EXAMPLES = 5


def _fetch_voice_examples(company_keyword: str) -> list[dict]:
    """Fetch the client's top posts by engagement as voice/style exemplars."""
    username = _resolve_linkedin_username(company_keyword)
    if not username or not SUPABASE_URL or not SUPABASE_KEY:
        return []

    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/linkedin_posts",
            params={
                "select": "post_text,posted_at,total_reactions,total_comments,engagement_score,hook",
                "creator_username": f"eq.{username}",
                "is_company_post": "eq.false",
                "post_text": "not.is.null",
                "order": "engagement_score.desc",
                "limit": str(MAX_VOICE_EXAMPLES),
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        logger.warning("[Stelle] Voice examples fetch failed: %s", e)
        return []

    examples: list[dict] = []
    for i, row in enumerate(rows):
        text = (row.get("post_text") or "").strip()
        if not text:
            continue
        posted = (row.get("posted_at") or "unknown date")[:10]
        reactions = row.get("total_reactions") or 0
        comments = row.get("total_comments") or 0
        engagement = row.get("engagement_score") or 0

        content = (
            f"# Voice Example {i + 1}\n"
            f"Posted: {posted} | Reactions: {reactions} | "
            f"Comments: {comments} | Engagement: {engagement}\n\n"
            f"{text}"
        )
        examples.append({
            "filename": f"{i + 1:02d}-engagement-{engagement}.md",
            "content": content,
        })

    logger.info("[Stelle] Fetched %d voice examples for @%s", len(examples), username)
    return examples


# ---------------------------------------------------------------------------
# Ordinal drafts — avoid duplicating posts already in the pipeline (Gap 11)
# ---------------------------------------------------------------------------

def _get_ordinal_api_key(company_keyword: str) -> str | None:
    """Return Ordinal api_key for ``company_keyword``.

    **Source of truth: Amphoreus Supabase ``ordinal_auth`` table.**
    This table mirrors Jacquard's ``ordinal_auth`` row-for-row via
    :mod:`jacquard_mirror_sync` (36 rows, PK=company_id, full-refresh
    each sync run). Lookup order:

      1. ``provider_org_slug == keyword`` — legacy pseudo-slug path
         (e.g. ``runpod-zhen``, ``hume-andrew``).
      2. ``company_id == resolve_to_uuid(keyword)`` — so mirror-backed
         dropdown slugs that aren't per-FOC rows still route to their
         company's shared Ordinal workspace. All FOCs under the same
         Jacquard company share one key.

    **CSV fallback**: if the Supabase read returns nothing (migration
    not yet applied, network blip, or a slug not yet synced), we fall
    back to the on-disk ``/data/memory/ordinal_auth_rows.csv``. This
    keeps the read path correct during the mirror rollout and also
    means a teammate who locally curated their own CSV doesn't lose
    anything. Remove the fallback once the mirror has been verified
    in production for a few sync cycles.
    """
    target = (company_keyword or "").strip().lower()
    if not target:
        return None

    # --- Primary: Amphoreus Supabase ``ordinal_auth`` ---------------------
    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
        if is_configured():
            sb = _get_client()
            if sb is not None:
                # 1) direct slug match. Case-insensitive via ilike to
                # match CSV semantics.
                try:
                    rows = (
                        sb.table("ordinal_auth")
                          .select("api_key, provider_org_slug, company_id")
                          .ilike("provider_org_slug", target)
                          .limit(1)
                          .execute()
                          .data
                    ) or []
                    if rows:
                        key = (rows[0].get("api_key") or "").strip()
                        if key.startswith("ord_"):
                            return key
                except Exception as exc:
                    logger.debug("[Stelle] ordinal_auth slug lookup failed: %s", exc)

                # 2) UUID-resolved company_id match
                try:
                    from backend.src.lib.company_resolver import resolve_to_uuid
                    resolved = resolve_to_uuid(company_keyword)
                except Exception:
                    resolved = None
                if resolved:
                    try:
                        rows = (
                            sb.table("ordinal_auth")
                              .select("api_key")
                              .eq("company_id", resolved)
                              .limit(1)
                              .execute()
                              .data
                        ) or []
                        if rows:
                            key = (rows[0].get("api_key") or "").strip()
                            if key.startswith("ord_"):
                                return key
                    except Exception as exc:
                        logger.debug("[Stelle] ordinal_auth company_id lookup failed: %s", exc)
    except Exception as exc:
        logger.debug("[Stelle] ordinal_auth Supabase read failed, falling back to CSV: %s", exc)

    # --- Fallback: on-disk CSV --------------------------------------------
    # Migration safety net — kept around until the mirror is verified
    # healthy in production. Remove once we're confident sync covers
    # every key.
    csv_path = P.ordinal_auth_csv()
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        logger.warning("[Stelle] Failed to read ordinal_auth_rows.csv: %s", e)
        return None

    # 1) direct slug match
    for row in rows:
        if (row.get("provider_org_slug") or "").strip().lower() == target:
            key = (row.get("api_key") or "").strip()
            if key.startswith("ord_"):
                logger.info(
                    "[Stelle] ordinal_auth key for %r served from CSV fallback — "
                    "Supabase mirror missed this row. Sync may not have caught it yet.",
                    target,
                )
                return key

    # 2) UUID resolution → company_id match
    try:
        from backend.src.lib.company_resolver import resolve_to_uuid
        resolved = resolve_to_uuid(company_keyword)
    except Exception:
        resolved = None
    if resolved:
        ruid = resolved.lower()
        for row in rows:
            if (row.get("company_id") or "").strip().lower() == ruid:
                key = (row.get("api_key") or "").strip()
                if key.startswith("ord_"):
                    logger.info(
                        "[Stelle] ordinal_auth key for company_id=%s served from "
                        "CSV fallback — Supabase mirror missed this row.",
                        resolved,
                    )
                    return key
    return None


def _fetch_ordinal_drafts(company_keyword: str, exclude_dates: set[str] | None = None) -> list[dict]:
    """Fetch existing draft posts from Ordinal, excluding already-published dates (Gap 11)."""
    api_key = _get_ordinal_api_key(company_keyword)
    if not api_key:
        logger.info("[Stelle] No Ordinal API key for %s — skipping draft dedup", company_keyword)
        return []

    exclude = exclude_dates or set()
    base_url = "https://app.tryordinal.com/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_posts: list[dict] = []
    cursor: str | None = None
    _SKIP_STATUSES = {"Posted"}

    try:
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor

            resp = httpx.get(
                f"{base_url}/posts",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            posts_data = data.get("posts", [])
            for p in posts_data:
                status = (p.get("status") or "").strip()
                if status in _SKIP_STATUSES:
                    continue

                li = p.get("linkedIn") or p.get("linkedin") or {}
                text = (li.get("copy") or li.get("text") or "").strip()
                title = (p.get("title") or "").strip()
                if not text and not title:
                    continue

                date_str = "no-date"
                for date_key in ("publishDate", "publishAt", "createdAt"):
                    val = p.get(date_key)
                    if val:
                        date_str = str(val)[:10]
                        break

                if date_str in exclude:
                    continue

                slug = re.sub(r"[^a-z0-9]+", "-", (title or "untitled")[:40].lower()).strip("-")
                filename = f"{date_str}-{slug}.md"
                existing = {d["filename"] for d in all_posts}
                if filename in existing:
                    filename = f"{date_str}-{slug}-{len(all_posts)}.md"
                header = f"# {title or 'Untitled'}\nStatus: {status} | Date: {date_str}\n\n"
                all_posts.append({"filename": filename, "content": header + text})

            if not data.get("hasMore") or not data.get("nextCursor"):
                break
            cursor = data["nextCursor"]

    except Exception as e:
        logger.warning("[Stelle] Ordinal draft fetch failed: %s", e)

    logger.info("[Stelle] Fetched %d existing Ordinal drafts for %s", len(all_posts), company_keyword)
    return all_posts


def _fetch_jacquard_dedup_keys(company_keyword: str) -> tuple[set[str], set[str]]:
    """Thin wrapper returning just ``(provider_urns, content_hashes)``
    for posts the client has already published to LinkedIn, sourced
    from Jacquard's ``linkedin_posts``. Used by
    :func:`_fetch_all_ordinal_hooks` to suppress overlap between the
    Ordinal dump and the Jacquard-direct dump (same post rendered
    twice). Full post bodies come from :func:`_fetch_jacquard_posts`.
    """
    posts = _fetch_jacquard_posts(company_keyword)
    urns: set[str] = set()
    hashes: set[str] = set()
    for p in posts:
        u = (p.get("provider_urn") or "").strip()
        if u:
            urns.add(u)
        t = (p.get("post_text") or "").strip()
        if t:
            hashes.add(_normalize_for_dedup(t))
    return urns, hashes


_JACQUARD_DEDUP_WINDOW_DAYS = 180   # ~6 months


def _fetch_jacquard_posts(company_keyword: str) -> list[dict]:
    """Return the creator's recent LinkedIn posts from Jacquard's
    ``linkedin_posts`` table — full post text, date, URN.

    Windowed to the last ``_JACQUARD_DEDUP_WINDOW_DAYS`` (180d) because
    older posts aren't useful dedup signal: LinkedIn's feed has no
    institutional memory at multi-year timescales, and topics a creator
    covered 2+ years ago are fair game to cover again from a fresh
    angle. Older posts still inform *voice* (via
    :func:`_fetch_voice_examples`, which is unwindowed) — they're just
    not rendered into the "EXISTING POSTS" block.

    This source covers:
      * Posts pushed via Ordinal  — overlap with Ordinal's Posted set
      * Posts drafted in Lineage and posted without going through
        Ordinal — invisible to the Ordinal API, but visible here
      * Recent posts from before the client was on Ordinal
      * Posts published via any other tool

    :func:`_fetch_all_ordinal_hooks` concatenates these into the same
    EXISTING POSTS prompt block as the Ordinal dump so Stelle has a
    single dedup surface covering every content path.

    Empty list on any failure so the caller can keep going with just
    the Ordinal dump.
    """
    username = _resolve_linkedin_username(company_keyword)
    if not username or not SUPABASE_URL or not SUPABASE_KEY:
        return []

    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    cutoff = (_dt.now(_tz.utc) - _td(days=_JACQUARD_DEDUP_WINDOW_DAYS)).isoformat()

    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/linkedin_posts",
            params={
                "select": "provider_urn,post_text,posted_at",
                "creator_username": f"eq.{username}",
                "is_company_post": "eq.false",
                "post_text": "not.is.null",
                "posted_at": f"gte.{cutoff}",
                "order": "posted_at.desc",
                "limit": "200",
            },
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json() or []
    except Exception as exc:
        logger.debug("[Stelle] jacquard posts fetch skipped: %s", exc)
        return []


def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace + lowercase the first 200 chars so small
    formatting differences (bullet chars, paragraph breaks) don't cause
    false misses when comparing Ordinal's ``li.copy`` to Jacquard's
    ``post_text``.
    """
    import re as _re
    return _re.sub(r"\s+", " ", text).strip()[:200].lower()


def _fetch_all_ordinal_hooks(company_keyword: str) -> str:
    """Fetch every post in Ordinal (draft/scheduled/approved/posted) and
    format them into a single prompt block for topic-dedup.

    Injected verbatim into Stelle's user prompt so she knows every topic
    already written, scheduled, in-review, or live, and avoids
    duplicates. No LLM call — just API fetch + text formatting.

    Why every status: since the memory/ directory was deprecated in
    favour of Supabase, Stelle no longer reads a separate "published
    posts" set from disk. Ordinal's Posted-status entries are now the
    authoritative view of already-live posts for dedup purposes. The
    previous implementation filtered those out on the assumption that
    workspace files would render them separately — that assumption is
    dead (no code writes ``posts/published/`` anymore), and the filter
    was silently stripping the dedup signal on every run. Hence the
    Innovo comparator-arm duplicate (2026-04-20) that matched a
    2026-04-02 Posted entry.
    """
    api_key = _get_ordinal_api_key(company_keyword)
    if not api_key:
        return ""

    base_url = "https://app.tryordinal.com/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    entries: list[str] = []
    cursor: str | None = None
    # Captured during the Ordinal loop, used below to suppress the
    # same-post-twice case when we also pull from Jacquard.
    ordinal_urns_captured: set[str] = set()

    # No status filter. Per the 2026-04-22 workflow redesign: rejected
    # drafts are manually purged from Ordinal by the operator, so
    # anything still visible here is legitimately live queue or shipped.
    # Rejected drafts are preserved in Amphoreus via the Posts-tab
    # "Reject" action with their comments attached, and surfaced to
    # Stelle/Aglaea via :mod:`backend.src.services.post_bundle` as a
    # distinct learning signal (not dedup material).

    try:
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor

            resp = httpx.get(
                f"{base_url}/posts",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            for p in data.get("posts", []):
                status = (p.get("status") or "").strip()
                li = p.get("linkedIn") or p.get("linkedin") or {}
                text = (li.get("copy") or li.get("text") or "").strip()
                title = (p.get("title") or "").strip()
                if not text and not title:
                    continue

                # Skip if Jacquard already has this post — the workspace
                # files will surface it on their own. Check URN first
                # (most reliable when Ordinal exposes it), fall back to
                # a content-hash match so we don't depend on Ordinal's
                # field-name stability.
                urn = (
                    li.get("urn")
                    or li.get("postUrn")
                    or li.get("providerUrn")
                    or li.get("provider_urn")
                    or ""
                ).strip()
                # Capture URN so the Jacquard append pass below can
                # suppress the same post rendered twice.
                if urn:
                    ordinal_urns_captured.add(urn)

                date_str = ""
                for date_key in ("publishDate", "publishAt", "createdAt"):
                    val = p.get(date_key)
                    if val:
                        date_str = str(val)[:10]
                        break

                # Include full post text — more data to the model for
                # thematic dedup. Truncated hooks miss topic-level
                # collisions (e.g. two climbing posts with different
                # angles still saturate the same topic for the audience).
                if text:
                    entries.append(
                        f"- [{status}] {date_str}:\n{text}\n"
                    )
                else:
                    entries.append(f"- [{status}] {date_str}: {title[:200]}")

            if not data.get("hasMore") or not data.get("nextCursor"):
                break
            cursor = data["nextCursor"]

    except Exception as e:
        logger.warning("[Stelle] Ordinal hook fetch for dedup failed: %s", e)
        # Continue with whatever we got — we still want the Jacquard
        # and local additions below. Returning "" here would silently
        # lose Lineage-path dedup.

    # Append Jacquard-scraped posts. Ordinal's API only shows posts that
    # went through Ordinal; Jacquard's scraper sees the LinkedIn profile
    # directly, so it also covers Lineage-path posts, pasted-directly
    # posts, pre-Ordinal history, and posts via other tools. Redundancy
    # with the Ordinal dump is fine — a little extra token spend beats
    # missing a whole content-path's dedup signal.
    #
    # We do a cheap overlap suppression to avoid rendering the exact
    # same post twice: collect the Ordinal-side URN + normalised-text
    # sets and skip Jacquard entries that match either.
    ordinal_urns_seen: set[str] = set(ordinal_urns_captured)
    ordinal_hash_prefixes_seen: set[str] = set()
    try:
        jacquard_posts = _fetch_jacquard_posts(company_keyword)
        jacquard_added = 0
        # Build the collision sets from the entries we already emitted.
        for line in entries:
            norm = _normalize_for_dedup(line)
            # 120-char prefix is long enough to avoid false positives
            # (two unrelated posts rarely share 120 normalised chars)
            # and short enough to survive small formatting drift.
            if len(norm) >= 120:
                ordinal_hash_prefixes_seen.add(norm[:120])
        for jp in jacquard_posts:
            text = (jp.get("post_text") or "").strip()
            if not text:
                continue
            urn = (jp.get("provider_urn") or "").strip()
            if urn and urn in ordinal_urns_seen:
                continue
            h = _normalize_for_dedup(text)
            # Exact-prefix match against any Ordinal line. Looser than
            # full-string equality (catches minor formatting drift) and
            # much tighter than the earlier substring-anywhere check.
            if len(h) >= 120 and h[:120] in ordinal_hash_prefixes_seen:
                continue
            date_str = (jp.get("posted_at") or "")[:10]
            entries.append(f"- [Posted·LinkedIn] {date_str}:\n{text}\n")
            jacquard_added += 1
        if jacquard_added:
            logger.info(
                "[Stelle] Appended %d Jacquard-only posts to dedup block (%s)",
                jacquard_added, company_keyword,
            )
    except Exception as exc:
        logger.warning("[Stelle] Jacquard post dump for dedup failed: %s", exc)

    # Also include locally generated posts not yet pushed to Ordinal.
    try:
        from backend.src.db.local import list_local_posts
        local_posts = list_local_posts(company=company_keyword, limit=100)
        local_seen: set[str] = set()
        for lp in local_posts:
            content = (lp.get("content") or "").strip()
            title = (lp.get("title") or "").strip()
            # Dedup by first line to avoid exact duplicates in the list
            dedup_key = content.split("\n")[0][:120] if content else title[:120]
            if not dedup_key or dedup_key in local_seen:
                continue
            local_seen.add(dedup_key)
            status = lp.get("status", "draft")
            if content:
                entries.append(f"- [{status}] (local):\n{content}\n")
            else:
                entries.append(f"- [{status}] (local): {title[:200]}")
    except Exception as e:
        logger.debug("[Stelle] Local post dedup fetch failed: %s", e)

    if not entries:
        return ""

    logger.info("[Stelle] Fetched %d existing post hooks for dedup (%s)", len(entries), company_keyword)
    return (
        "\n\nEXISTING POSTS — pipeline view (every draft + published + "
        "rejected post in the operator's queue, "
        f"{len(entries)} total). Every published LinkedIn post for this "
        "creator is already in the POSTS block above (body + engagement "
        "+ comments + deltas + semantic neighbors) — this list is just "
        "the hooks-only index for dedup. Treat BOTH sources as "
        "authoritative for dedup: do NOT write a post that covers the "
        "same TOPIC as any existing post here OR in the POSTS block, "
        "even from a different angle. Two posts about the same activity, "
        "setting, or subject (e.g. climbing, piano, a specific trial) "
        "will read as repetition regardless of angle. Every post you "
        "write must occupy genuinely new thematic territory:\n\n"
        + "\n".join(entries)
    )


# ---------------------------------------------------------------------------
# Research files from Supabase (Gap 8)
# ---------------------------------------------------------------------------

def _resolve_supabase_ids(username: str) -> tuple[str, str, str]:
    """Look up user_id, company_id, and display name from Supabase users table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "", "", ""
    _SB_HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/users",
            params={
                "select": "id,company_id,first_name,last_name",
                "linkedin_url": f"ilike.%{username}%",
                "limit": "1",
            },
            headers=_SB_HEADERS,
            timeout=15.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            user_id = rows[0].get("id", "") or ""
            company_id = rows[0].get("company_id", "") or ""
            first = (rows[0].get("first_name") or "").strip()
            last = (rows[0].get("last_name") or "").strip()
            display_name = f"{first} {last}".strip()
            logger.info("[Stelle] Resolved Supabase IDs for @%s: user=%s company=%s name=%r",
                        username, user_id, company_id, display_name)
            return user_id, company_id, display_name
    except Exception as e:
        logger.warning("[Stelle] Supabase user lookup failed for @%s: %s", username, e)
    return "", "", ""


def _fetch_research_files(company_keyword: str) -> list[dict]:
    """Fetch parallel_research_results for person and company from Supabase, scoped to client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    files: list[dict] = []
    username = _resolve_linkedin_username(company_keyword)
    if not username:
        return []

    user_id, company_id, _display_name = _resolve_supabase_ids(username)

    _SB_HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

    for research_type in ("person", "company"):
        try:
            params: dict[str, str] = {
                "select": "output,basis,created_at",
                "research_type": f"eq.{research_type}",
                "error": "is.null",
                "order": "created_at.desc",
                "limit": "1",
            }
            if research_type == "person" and user_id:
                params["user_id"] = f"eq.{user_id}"
            elif research_type == "company" and company_id:
                params["company_id"] = f"eq.{company_id}"
            else:
                logger.info("[Stelle] No Supabase ID for %s research — skipping", research_type)
                continue

            resp = httpx.get(
                f"{SUPABASE_URL}/rest/v1/parallel_research_results",
                params=params,
                headers=_SB_HEADERS,
                timeout=30.0,
            )
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                output = rows[0].get("output")
                if output:
                    content = output if isinstance(output, str) else json.dumps(output, indent=2)
                    files.append({"filename": f"{research_type}.md", "content": content})
        except Exception as e:
            logger.warning("[Stelle] Research fetch (%s) failed: %s", research_type, e)

    logger.info("[Stelle] Fetched %d research files for %s", len(files), company_keyword)
    return files



# _build_observation_digest was deleted 2026-04-23. It synthesized
# memory/post-history.md — a top-10-by-engagement digest with
# draft-vs-published diffs — but (a) nothing called it anymore, the
# output file was never written to disk after the virtual-filesystem
# migration, and (b) the data it rendered is now fully covered by
# build_post_bundle which delivers every post (not just top 10) with
# the same body + engagement + delta information, injected directly
# into user_prompt at generation time. The top-N filter was itself a
# Bitter Lesson violation: letting Opus see the raw distribution is
# stricter than pre-chewing "these are the winners" for it.


# ---------------------------------------------------------------------------
# Workspace setup (Gap 3, 10 — jacquard directory structure)
# ---------------------------------------------------------------------------

def _setup_workspace(company_keyword: str) -> Path:
    """Stage Stelle's scratch workspace.

    Client data (transcripts, engagement, research, context, edits) is
    read on-demand from Jacquard's Supabase + GCS via the dispatchers
    in ``database_client.py``. Nothing is staged to local disk for
    reading purposes — local disk is ONLY Stelle's scratch space
    (iteration drafts, notes, ``scratch/final/``).

    Every run wipes scratch/ fresh so stale in-flight drafts from prior
    runs never collide with new ones. Leftover non-scratch dirs from
    older workspace layouts are also cleared on startup.
    """
    from backend.src.db import vortex as _P
    workspace = _P.workspace_dir(company_keyword)
    workspace.mkdir(parents=True, exist_ok=True)

    # Wipe scratch/ — stale drafts from prior runs must not leak.
    scratch = workspace / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    (scratch / "final").mkdir()

    # Clear leftover trees/files from older workspace layouts so Stelle
    # doesn't waste cycles exploring empty junk at the workspace root.
    stale_dirs = (
        "memory", "context", "abm_profiles", "revisions",
        "snapshots", "output", "tools",
    )
    for leftover in stale_dirs:
        p = workspace / leftover
        if p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for script in ("draft.sh", "edit.sh", "memory.sh"):
        p = workspace / script
        if p.is_symlink() or p.is_file():
            try:
                p.unlink()
            except OSError:
                pass

    logger.info(
        "[Stelle] Scratch workspace staged at %s. Client data reads flow "
        "through the workspace_fs dispatchers (Jacquard Supabase + GCS).",
        workspace,
    )
    return workspace


# ---------------------------------------------------------------------------
# Learned overrides — graduated defaults from data
# ---------------------------------------------------------------------------

_OVERRIDE_MIN_EDITS_FOR_CHAR_LIMIT = 5


# ---------------------------------------------------------------------------
# Dynamic directives
# ---------------------------------------------------------------------------

_WORKSPACE_TOOL_SEMANTICS = """\
## Tool semantics

The workspace is a virtual view over the client's data source (Supabase
+ GCS). Everything you read flows through that — there is no fly-local
``memory/`` tree.

- **Your primary engagement / body / comment / edit / voice signal is
  the POSTS block already concatenated into the user_prompt above.**
  Every recent post for this creator is there with body + engagement
  counts + threaded comments + edit delta + top-2 semantic neighbors.
  Every one of those posts is itself a voice example — voice is
  learned from the raw distribution, not from a curated pick-list.
  Don't hunt for "post-history.md", "engagement/posts.json", "tone/",
  or "voice-examples/" — those paths were retired. The POSTS block
  IS that data.

- **`retrieve_similar_posts` (cross-creator corpus, ~390k real LinkedIn
  posts, semantic search).** Your client's own post history is a
  biased prior — it only shows what that person has tried before, not
  what actually lands on LinkedIn in their space. Use this tool to
  escape the basin.

  Call it repeatedly, not once. Typical usage shape:
  * **Before drafting** — query by topic/angle phrases extracted from
    the transcript to see what high-reaction posts in that space look
    like structurally (length, pacing, stance, hook form).
  * **During angle exploration** — narrower queries per candidate
    angle to confirm the form is real, not one-off.
  * **Before submitting** — sanity-check: "does this angle already
    exist?" If yes, how did it land? Revise or differentiate.

  Suggested args:
    query: free text (topic, angle, or a candidate draft). Express
      content-type filtering directly in the query — e.g. "first-person
      narrative with a specific number, NOT a product announcement".
    k: 10-20 for exploration, 3-5 for comparison
    min_reactions: 50-200 for outlier-biased views (optional)
    exclude_creator: pass the client's LinkedIn username to avoid
      retrieving their own posts

  Read the returned posts as REAL precedents, not prescriptions —
  adapt form, not content. Cite them in your `process_notes` audit
  trail when relevant so the operator can trace your thinking.

- `bash` is DISABLED (the workspace is virtual, not a real filesystem).

- `write_file` / `edit_file` route BY PATH:
  * **Client data-source paths are READ-ONLY.** Any write under
    ``transcripts/``, ``research/``, ``engagement/``, ``reports/``,
    ``context/``, ``posts/``, ``edits/``, ``tone/``, ``strategy/``, or
    the shared ``conversations/``, ``slack/``, ``tasks/``, ``.pi/``
    is refused — those are the client's files. The error message tells
    you exactly where to write instead.
  * **Scratch paths work normally.** Write freely to anything OUTSIDE
    the data-source mounts: ``scratch/post1-v1.md``, ``scratch/plan.md``,
    ``notes/brainstorm.md``, ``drafts/wip.md`` — whatever you want.
    Those land on your fly-local SandboxFs, persist for the run, and
    you can ``read_file``/``list_directory`` them back normally.

- **Use scratch for candidates + draft iteration.** Pattern:

      # Stage 1 — generate K=3 candidate drafts (different attempts)
      write_file("scratch/post1-cand-A.md", <candidate A>)
      write_file("scratch/post1-cand-B.md", <candidate B>)
      write_file("scratch/post1-cand-C.md", <candidate C>)
      → get_reader_reaction(draft_text=<candidate A>)
      → get_reader_reaction(draft_text=<candidate B>)
      → get_reader_reaction(draft_text=<candidate C>)
      → pick the one with the strongest gestalt + most positive anchors

      # Stage 2 — iterate the winning candidate
      write_file("scratch/post1-v2.md", <revised winner>)
      → get_reader_reaction(draft_text=<revised>)
      → … iterate until the reader stops complaining.

  Why three candidates, not one: a single first-shot draft locks you
  into one shape — analyst-voice vs founder-voice, story vs commentary,
  long vs tight. Three candidates let Irontomb's calibration data
  pick which shape this client's audience responds to, instead of you
  guessing on the first draft. Vary whatever you think is worth
  varying — shape, voice, opener, framing, length. No required
  differentiation across the three; the selector is Irontomb's
  rating, not a diversity rule.

  You can also keep candidates purely in-context if you prefer —
  ``get_reader_reaction`` takes the full draft text directly, so a
  file round-trip is never required. Use whichever feels natural.

- **FINAL DRAFTS: use `submit_draft`.** One call per finished post:

      submit_draft(
        user_slug="<slug>",          # which FOC user this post is for
        content="<final markdown>",   # the final post, plain markdown
        scheduled_date="YYYY-MM-DD",  # calendar slot (tomorrow or later)
        publication_order=1,          # 1, 2, 3… for multi-post runs
        process_notes="<audit trail>",# transcript provenance, Irontomb
                                       # anchors, comfort score, length
                                       # stats, decision rationale —
                                       # hidden by default in the operator
                                       # UI behind a "Show process notes"
                                       # expander; useful for debugging.
      )

  **Where the draft lands.** ``submit_draft`` persists the draft to
  Amphoreus's ``local_posts`` table for the FOC user whose slug you
  passed. The operator reviews at amphoreus.app/posts and pushes to
  Ordinal from there.

  ``submit_draft`` runs Castorice on your content before persisting:
  fact-check (text correction + citations) PLUS strategic-fit analysis
  (verdict against the latest Cyrene brief). Castorice's strategic-fit
  verdict — NOT your process_notes — is what shows as the post's
  ``why_post`` to the operator at review. Your process_notes is the
  audit trail; keep it terse and informative, but DO NOT duplicate
  strategic-fit reasoning there.

- **Multi-post runs: vary angles, don't riff one topic twice.**
  When the user asks for N posts, cover N DIFFERENT angles/topics —
  not the same topic with wording variations. The POSTS block at the
  top of this prompt shows every recent post with engagement; pick
  underrepresented angles. Space the publication dates across the
  cadence (3 posts/week = Mon/Wed/Fri or Mon/Tue/Thu). Pass each post's
  slot as ``scheduled_date``.

"""


_WORKSPACE_LAYOUT_USER_TARGETED = """\
# Workspace — USER-TARGETED RUN

You are scoped to a single FOC user for this run. Every draft you
produce will be attributed to that user.

**The client data source is READ-ONLY to you.** Calls to
`list_directory`, `read_file`, `search_files` on data-source paths
read from Jacquard's Supabase + GCS. ``write_file`` / ``edit_file`` on
those same paths are refused.

**But scratch paths work normally.** Any path OUTSIDE the data-source
mount tree (``scratch/``, ``notes/``, ``drafts/``, loose top-level
files) is your own fly-local SandboxFs — read, write, edit, list freely.

- Data-source paths (READ-ONLY for writes):
  ``transcripts/``, ``research/``, ``engagement/``, ``reports/``,
  ``context/``, ``posts/``, ``edits/``, ``tone/``, ``strategy/``, and
  the shared ``conversations/``, ``slack/``, ``tasks/``, ``.pi/``.
- Scratch paths (fly-local, read/write): anything else.

Iteration pattern — K=3 candidates, then two-critic loop on the winner:

    # Stage 1 — generate K=3 candidate drafts, each a different attempt.
    write_file("scratch/post1-cand-A.md", <candidate A>)
    write_file("scratch/post1-cand-B.md", <candidate B>)
    write_file("scratch/post1-cand-C.md", <candidate C>)
    → get_reader_reaction on each
    → pick the candidate with the strongest gestalt + most positive anchors

    # Stage 2 — iterate the winning candidate against both critics
    write_file("scratch/post1-v2.md", <revised winner>)
    → get_reader_reaction(draft_text=<revised>)    # will readers engage?
    → check_client_comfort(draft_text=<revised>)   # will the FOC user ship this?
    → write_file("scratch/post1-v3.md", <revised>)
    → repeat both …
    → … until BOTH critics pass, then submit_draft.

Why K=3 candidates first: a single first-draft locks you into one
shape (analyst vs founder voice, story vs commentary). Three candidates
let Irontomb's calibration data pick which shape this client's audience
actually responds to, rather than you guessing. Vary whatever you
think is worth varying — no required differentiation rules.

The two critics optimise different axes — pass both:
  * Irontomb (`get_reader_reaction`) — engagement / reader stickiness.
  * Aglaea (`check_client_comfort`) — client voice fidelity, comfort to
    publish, absence of claims the client has edited out in prior runs.
  A post can win one and lose the other. Don't ship on half.

You can also keep drafts purely in-context — both tools take the full
draft text directly. Use whichever feels natural.

Only ``submit_draft`` persists finished posts.

## Workspace layout (read-only; paths auto-prefixed to the target user)

- `transcripts/` — raw client interview transcripts. Every claim traces here.
  **Interview transcripts are the source of content; internal sync/standup
  transcripts (content-eng, GTM weekly, product demos, team retros) are
  BACKGROUND ONLY — use them to understand context and voice, never as
  the narrative source of a post.**
- `research/` — deep research (company + person). Supplementary source material.
- `context/` — operator-uploaded brand docs / positioning PDFs.
- `strategy/` — persistent cross-run strategy memory. Read-only.
- `profile.md` — simple synthesized LinkedIn profile summary.

The POSTS block at the top of this prompt already carries every post
(body + engagement + comments + edit delta + semantic neighbors) for
this creator. Don't look for engagement/*.json, posts/published/,
edits/, reports/, post-history.md, or context/account.md — those paths
were retired. Everything you'd expect to find in them is in the POSTS
block.

Shared (not user-scoped; don't prepend slug):
- `conversations/trigger-log.jsonl` — chronological replay of every prior trigger (interviews, CE feedback with diffs, manual runs, Slack messages). Scan at session start.
- `tasks/<id>.json` — pending review tasks.
- `slack/` — Slack channel snapshots.
- `.pi/` — historical Pi skill files. IGNORE.

## Client-supplied material (articles / links / pasted text)

Clients share primary source material two ways:

1. **Slack → trigger log.** When a client sends an article, podcast,
   blog post, etc. in Slack, that message lands in
   `conversations/trigger-log.jsonl` with its URL intact.
2. **Operator paste → `transcripts/`.** Content engineers paste client
   notes, link dumps, or ad-hoc context directly into the Transcripts
   tab. Those land as `paste-*.md` files in `transcripts/` (listed
   first because they're the freshest operator-added material).

**Any URL in either source is primary material for this run.** Use
`fetch_url` to pull the article body, then mine it the same way you'd
mine a transcript — extract the actual claim, quote, number, or
anecdote, and build a post around it. This is the ONLY path by which
a client-supplied link becomes a post, so if the operator's prompt
doesn't separately call it out, the trigger log + paste transcripts
are your signal.

Caveats:
- `fetch_url` on `docs.google.com/*` URLs will fail — Google Docs
  requires OAuth we don't have. Skip and flag via `missing_resources`.
- Paywalled articles will often return fragments; if the fetched body
  is under ~500 chars of real content, skip.
- Only act on trigger-log entries from the **last 48 hours** unless
  the operator's prompt says otherwise. Older links are stale.

## Draft write contract

**Use ``submit_draft`` for finished posts.** One call per finished post.
``submit_draft`` persists the draft to Amphoreus's ``local_posts`` for
the FOC user whose slug you passed. The operator reviews at
amphoreus.app/posts and pushes to Ordinal from there.

``submit_draft`` runs Castorice on your content before persisting:
silent fact-check + correction, plus a strategic-fit verdict against
the latest Cyrene brief. Castorice's strategic-fit verdict is what the
operator sees as the post's ``why_post`` at review. Your
``process_notes`` argument is the audit trail (transcript provenance,
Irontomb anchors, comfort score, length stats) — collapsed by default
in the UI behind a "Show process notes" expander, useful for
debugging, NOT for review-time judgment.

## Ingestion order at session start

1. Re-read the POSTS block at the top of this prompt — that's your
   body + engagement + comments + edits + neighbors for every recent
   post this creator has. Everything starts from that distribution.
2. ``list_directory("")`` — confirm the target slug and what else is
   available (transcripts, research, context, tone, strategy, profile).
3. ``read_file("conversations/trigger-log.jsonl")`` — your history with
   this company (interviews, CE feedback diffs, prior runs). Scan it.
4. ``read_file("strategy/strategy.md")`` if it exists — cross-run memory
   left by your previous selves.
5. ``list_directory("transcripts/")`` + read the latest 2-3 transcripts.
6. Spot-read ``profile.md`` + ``context/`` (brand PDFs) if you need
   domain facts the POSTS block + transcripts don't already cover.
"""


_WORKSPACE_LAYOUT_COMPANY_WIDE = """\
# Workspace — COMPANY-WIDE RUN

Filesystem tool calls route BY PATH:

- **Data-source paths** (``transcripts/``, ``research/``, ``engagement/``,
  ``reports/``, ``context/``, ``posts/``, ``edits/``, ``tone/``,
  ``strategy/``, and the shared ``conversations/``, ``slack/``, ``tasks/``,
  ``.pi/``) read from the client's Supabase + GCS. These are READ-ONLY
  for writes — ``write_file``/``edit_file`` on them is refused.
- **Scratch paths** (``scratch/``, ``notes/``, ``drafts/``, any loose
  top-level file) land on your fly-local SandboxFs. Read/write freely.

Only ``submit_draft`` persists finished posts.

## Workspace layout (all read-only)

The workspace root contains one `<slug>/` per FOC user of the company plus
shared roots. Call `list_directory("")` once to discover available slugs,
then use explicit slug prefixes in every filesystem call.

- `<slug>/transcripts/` — raw client interview transcripts. Every claim traces here.
  **Interview transcripts are the source of content; internal sync/standup
  transcripts (content-eng, GTM weekly, product demos, team retros) are
  BACKGROUND ONLY — use them to understand context and voice, never as
  the narrative source of a post.**
- `<slug>/research/` — deep research (company + person). Supplementary source material.
- `<slug>/context/` — operator-uploaded brand docs / positioning PDFs.
- `<slug>/strategy/` — persistent cross-run strategy memory. Read-only.
- `<slug>/profile.md` — simple synthesized LinkedIn profile summary.

The POSTS block at the top of this prompt already carries every post
(body + engagement + comments + edit delta + semantic neighbors) for
each creator. Don't look for engagement/*.json, posts/published/,
edits/, reports/, post-history.md, or context/account.md — those paths
were retired. Everything you'd expect to find in them is in the POSTS
block.

Shared (not user-scoped; don't prepend slug):
- `conversations/trigger-log.jsonl` — chronological replay of every prior trigger (interviews, CE feedback with diffs, manual runs, Slack messages). Scan at session start.
- `tasks/<id>.json` — pending review tasks.
- `slack/` — Slack channel snapshots.
- `.pi/` — historical Pi skill files. IGNORE.

## Slack-originated material (articles / links / pasted text from clients)

The `trigger-log.jsonl` carries Slack messages from clients to the
content team. When a client sends an article, podcast, blog post, or
any other primary source in Slack (often @-mentioning Lineage and
saying something like "draft a post from this"), that message lands
in the trigger log with its URL intact.

**If the most recent trigger-log entries contain URLs from Slack (or
any source), treat those URLs as primary material for this run.** Use
`fetch_url` to pull the article body, then mine it the same way you'd
mine a transcript — extract the actual claim, quote, number, or
anecdote, and build a post around it. This is the ONLY path by which
a client-supplied link becomes a post, so if the operator's prompt
doesn't separately call it out, the trigger log is your signal.

Caveats:
- `fetch_url` on `docs.google.com/*` URLs will fail — Google Docs
  requires OAuth we don't have. Skip and flag via `missing_resources`.
- Paywalled articles will often return fragments; if the fetched body
  is under ~500 chars of real content, skip.
- Only act on trigger-log entries from the **last 48 hours** unless
  the operator's prompt says otherwise. Older links are stale.

## Per-draft author attribution

Each finished post belongs to exactly ONE user. You decide which user by
passing their slug to ``submit_draft``:

    submit_draft(user_slug="<user-slug>", content=<post text>, ...)

The slug determines attribution — there is no separate ``author`` field.

## Draft write contract

**Use ``submit_draft`` for finished posts.** One call per finished post.
``submit_draft`` persists the draft to Amphoreus's ``local_posts`` under
the target user. The operator reviews at amphoreus.app/posts.

``submit_draft`` runs Castorice on your content before persisting:
silent fact-check + correction, plus a strategic-fit verdict against
the latest Cyrene brief. Castorice's strategic-fit verdict is what the
operator sees as the post's ``why_post`` at review. Your
``process_notes`` argument is the audit trail (transcript provenance,
Irontomb anchors, comfort score, length stats) — collapsed by default
in the UI behind a "Show process notes" expander, useful for
debugging, NOT for review-time judgment.

## Ingestion order at session start

1. Re-read the POSTS block at the top of this prompt — every recent
   post for every slug is there with body + engagement + comments +
   edits + neighbors. That is your engagement + voice baseline.
2. ``list_directory("")`` — discover the FOC-user slugs.
3. ``read_file("conversations/trigger-log.jsonl")`` — history of prior
   triggers for this company (interviews, CE feedback diffs, manual runs).
4. For each slug you plan to write for:
   a. ``read_file("<slug>/strategy/strategy.md")`` if it exists.
   b. ``list_directory("<slug>/transcripts/")`` + read latest 2-3.
   c. Spot-read ``<slug>/profile.md`` + ``<slug>/context/`` (brand PDFs)
      if you need domain facts the POSTS block + transcripts don't
      already cover.
"""


def _today_preamble() -> str:
    """Inject today's date so Stelle schedules posts in the future, not
    the past. Claude's training cutoff means she has no inherent sense
    of what day it is — without this she'll hallucinate dates from
    whatever year felt plausible in training data. Applies in both
    database and local mode."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    today_iso = now.date().isoformat()
    tomorrow = (now + timedelta(days=1)).date().isoformat()
    return (
        "# Today's date\n\n"
        f"Today is **{today_iso}** ({now.strftime('%A')}). "
        f"Any `scheduled_date` you assign to `submit_draft` MUST be "
        f"**{tomorrow} or later** — never in the past, never today. "
        "When spacing multiple posts across a cadence (Mon/Wed/Fri, "
        "Tue/Thu, etc.), anchor to today and pick the next N slots going "
        "forward. If the operator's prompt didn't specify the cadence, "
        "infer it from how often this creator has been posting recently "
        "(see the POSTS block).\n"
    )


def _build_dynamic_directives(company_keyword: str) -> str:
    """Return dynamic directives for the system prompt.

    Always emits the today-preamble so Stelle knows what day it is.

    Additionally emits the workspace-layout overlay when a client data
    source is configured (Jacquard Supabase + GCS):

    - ``USER-TARGETED`` when ``DATABASE_USER_SLUG`` is set. Every draft
      is attributed to that user and paths are auto-prefixed.
    - ``COMPANY-WIDE``  when no user slug is set. Stelle sees the full
      workspace and must include the slug in every filesystem call.

    Without a data source, only the today-preamble is emitted.
    """
    preamble = _today_preamble()
    try:
        from backend.src.agents import database_client as _lfs
        if not _lfs.is_database_mode():
            return preamble
        layout = (
            _WORKSPACE_LAYOUT_USER_TARGETED
            if _lfs.is_user_targeted()
            else _WORKSPACE_LAYOUT_COMPANY_WIDE
        )
        return preamble + "\n" + layout + "\n\n" + _WORKSPACE_TOOL_SEMANTICS
    except Exception:
        return preamble


# ---------------------------------------------------------------------------
# Pi workspace helpers — shell tool scripts (Jacquard parity)
# ---------------------------------------------------------------------------

_DRAFT_SH_SCRIPT = r'''#!/bin/bash
# draft.sh — Submit a finished post. Validates char count, saves locally.
#   ./draft.sh "Your full post text here" [YYYY-MM-DDTHH:MM]
#
# Optional second argument: scheduled datetime for the content calendar.
# Examples: 2026-04-01T09:00, 2026-04-01 (defaults to T09:00 if no time given)
POST="$1"
SCHEDULE_ARG="$2"
if [ -n "$SCHEDULE_ARG" ]; then
  case "$SCHEDULE_ARG" in
    *T*) SCHEDULED_DATE="${SCHEDULE_ARG}" ;;
    *)   SCHEDULED_DATE="${SCHEDULE_ARG}T09:00" ;;
  esac
else
  SCHEDULED_DATE=""
fi
if [ -z "$POST" ]; then
  echo "REJECTED: No post text provided."
  echo "Usage: ./draft.sh \"your full post text\" [YYYY-MM-DDTHH:MM]"
  exit 1
fi
CLEAN_POST=$(echo "$POST" | sed '/^<!-- \[/d' | cat -s)
CHARS=${#CLEAN_POST}
if [ "$CHARS" -gt 3000 ]; then
  echo "REJECTED: ${CHARS} characters exceeds LinkedIn's 3000-character platform cap."
  exit 1
fi
SLUG=$(echo "$CLEAN_POST" | head -1 | head -c 60 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr ' ' '-' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
DATE=$(date +%Y-%m-%d)
mkdir -p scratch/final
FILENAME="scratch/final/${DATE}-${SLUG}.md"
if [ -n "$SCHEDULED_DATE" ]; then
  printf "<!-- scheduled: %s -->\n%s" "$SCHEDULED_DATE" "$POST" > "$FILENAME"
  echo "PUBLISHED: ${CHARS} characters. Scheduled: ${SCHEDULED_DATE}. Saved to ${FILENAME}."
else
  echo "$POST" > "$FILENAME"
  echo "PUBLISHED: ${CHARS} characters. Saved to ${FILENAME}."
fi
exit 0
'''

_EDIT_SH_SCRIPT = r'''#!/bin/bash
# edit.sh — Update an existing draft with revised text.
#   ./edit.sh <draft-filename> "Your revised post text here"
DRAFT_FILE="$1"
POST="$2"
if [ -z "$DRAFT_FILE" ] || [ -z "$POST" ]; then
  echo "REJECTED: Missing arguments."
  echo "Usage: ./edit.sh <draft-filename> \"your revised post text\""
  exit 1
fi
CLEAN_POST=$(echo "$POST" | sed '/^<!-- \[/d' | cat -s)
CHARS=${#CLEAN_POST}
if [ "$CHARS" -gt 3000 ]; then
  echo "REJECTED: ${CHARS} characters exceeds LinkedIn's 3000-character platform cap."
  exit 1
fi
if [ -f "$DRAFT_FILE" ]; then
  OLD_CONTENT=$(cat "$DRAFT_FILE")
  FEEDBACK_DIR="memory/feedback/edits"
  mkdir -p "$FEEDBACK_DIR"
  EDIT_SLUG=$(echo "$CLEAN_POST" | head -1 | head -c 40 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr ' ' '-' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
  FEEDBACK_FILE="${FEEDBACK_DIR}/$(date +%Y-%m-%d)-${EDIT_SLUG}.md"
  {
    echo "## Before"
    echo ""
    echo "$OLD_CONTENT"
    echo ""
    echo "## After"
    echo ""
    echo "$CLEAN_POST"
  } > "$FEEDBACK_FILE"
fi
echo "$POST" > "$DRAFT_FILE"
echo "UPDATED: ${CHARS} characters. Saved to ${DRAFT_FILE}."
exit 0
'''

_MEMORY_SH_SCRIPT = r'''#!/bin/bash
# memory.sh — manages memory/config.md (4000 char limit)
set -euo pipefail
MEMORY_DIR="$(dirname "$0")/memory"
CONFIG="$MEMORY_DIR/config.md"
LIMIT=4000
usage() {
    echo "Usage: ./memory.sh <command> [args]"
    echo "  status                  — show current usage"
    echo "  add \"text\"              — append text"
    echo "  replace \"old\" \"new\"     — replace substring"
    echo "  remove \"text\"           — remove substring"
    exit 1
}
ensure_config() {
    mkdir -p "$(dirname "$CONFIG")"
    [ -f "$CONFIG" ] || touch "$CONFIG"
}
cmd_status() {
    ensure_config
    local size
    size=$(wc -c < "$CONFIG")
    echo "Memory: ${size}/${LIMIT} chars ($(( size * 100 / LIMIT ))%)"
}
cmd_add() {
    ensure_config
    local current new_text combined size
    current=$(cat "$CONFIG")
    new_text="$1"
    combined="${current}
${new_text}"
    size=${#combined}
    if [ "$size" -gt "$LIMIT" ]; then
        echo "ERROR: Would exceed limit (${size}/${LIMIT}). Consolidate first."
        exit 1
    fi
    echo "$combined" > "$CONFIG"
    echo "Added. Now at ${size}/${LIMIT} chars."
}
cmd_replace() {
    ensure_config
    local old="$1" new="$2"
    if ! grep -qF "$old" "$CONFIG"; then
        echo "ERROR: Substring not found."
        exit 1
    fi
    python3 -c "
import sys
with open('$CONFIG') as f: text = f.read()
text = text.replace(sys.argv[1], sys.argv[2], 1)
if len(text) > $LIMIT:
    print(f'ERROR: Would exceed limit ({len(text)}/$LIMIT)')
    sys.exit(1)
with open('$CONFIG', 'w') as f: f.write(text)
print(f'Replaced. Now at {len(text)}/$LIMIT chars.')
" "$old" "$new"
}
cmd_remove() {
    ensure_config
    local text="$1"
    if ! grep -qF "$text" "$CONFIG"; then
        echo "ERROR: Substring not found."
        exit 1
    fi
    python3 -c "
import sys
with open('$CONFIG') as f: content = f.read()
content = content.replace(sys.argv[1], '', 1)
with open('$CONFIG', 'w') as f: f.write(content)
print(f'Removed. Now at {len(content)}/$LIMIT chars.')
" "$text"
}
[ $# -lt 1 ] && usage
case "$1" in
    status)  cmd_status ;;
    add)     [ $# -lt 2 ] && usage; cmd_add "$2" ;;
    replace) [ $# -lt 3 ] && usage; cmd_replace "$2" "$3" ;;
    remove)  [ $# -lt 2 ] && usage; cmd_remove "$2" ;;
    *)       usage ;;
esac
'''

# ---------------------------------------------------------------------------
# Pi workspace helpers — Python tool scripts
# ---------------------------------------------------------------------------










_VALIDATE_DRAFT_SCRIPT = r'''#!/usr/bin/env python3
"""Validate a LinkedIn post draft — structural checks only.

Content quality (AI patterns, writing style) is handled by the constitutional
verifier post-generation, with learned principle weights. This tool only checks
structural issues that don't need an LLM: character count and deterministic
banned-phrase detection.

Usage: python3 validate_draft.py "Your full post text here"
       python3 validate_draft.py --file scratch/final/my-draft.md
       python3 validate_draft.py --attempt 2 --file scratch/final/my-draft.md

After attempt 2, issues are downgraded to info (escape hatch).
"""
import json, os, sys, re


def main():
    post_text = ""
    attempt = 1
    args = sys.argv[1:]
    i = 0
    file_path = None
    text_parts = []
    while i < len(args):
        if args[i] == "--attempt" and i + 1 < len(args):
            try:
                attempt = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == "--file" and i + 1 < len(args):
            file_path = args[i + 1]
            i += 2
        else:
            text_parts.append(args[i])
            i += 1

    if file_path:
        try:
            post_text = open(file_path, encoding="utf-8").read().strip()
        except Exception as e:
            print(json.dumps({"error": f"Cannot read file: {e}"}))
            return
    elif text_parts:
        post_text = " ".join(text_parts).strip()

    if not post_text:
        print("Usage: python3 validate_draft.py \"Your post text\"")
        print("       python3 validate_draft.py --file path/to/draft.md")
        print("       python3 validate_draft.py --attempt 2 --file path/to/draft.md")
        return

    escape_hatch = attempt > 2
    issues = []
    char_count = len(post_text)

    # --- Character count (LinkedIn platform cap only) ---
    # No hand-tuned length floor: length is learnable from the client's own
    # published posts and the writer already reads them before drafting.
    if char_count > 3000:
        issues.append({"type": "char_count", "severity": "critical",
            "description": f"Post is {char_count} chars — exceeds LinkedIn's 3000 char platform cap",
            "suggested_fix": f"Cut {char_count - 2800} characters"})

    # No content quality checks here — style, cadence, and voice are
    # learnable from the client's own published posts and from the ICP
    # simulator's feedback loop. This validator only enforces the LinkedIn
    # platform cap and nothing else.

    if escape_hatch:
        downgraded = []
        for issue in issues:
            d = dict(issue)
            d["severity"] = "info"
            d["description"] = f"[DOWNGRADED - attempt {attempt}] " + d.get("description", "")
            downgraded.append(d)
        issues = downgraded
        print(f"[escape hatch] Attempt {attempt} — issues downgraded to info, proceeding",
              file=sys.stderr)

    needs_correction = (not escape_hatch) and any(
        i.get("severity") in ("critical", "warning") for i in issues)
    result = {"needs_correction": needs_correction, "issues": issues,
              "char_count": char_count, "attempt": attempt,
              "escape_hatch_used": escape_hatch}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
'''




def _write_if_changed(path: Path, content: str) -> None:
    """Write file only if content differs from what's on disk.

    Avoids touching mtime on unchanged files, which prevents
    uvicorn --reload / watchfiles from triggering a server restart
    when Stelle writes tool scripts into the workspace.
    """
    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return
    except Exception:
        pass
    path.write_text(content, encoding="utf-8")


def _write_tool_scripts(workspace_root: Path) -> None:
    """Write validate_draft.py and the shell aliases.

    Previously emitted web_search.py / fetch_url.py / query_posts.py /
    ordinal_analytics.py / semantic_search_posts.py / image_search.py
    into workspace/tools/ for bash invocation. Log analysis across 9
    sessions showed all had zero invocations. Dropped.
    """
    tools_dir = workspace_root / "tools"
    tools_dir.mkdir(exist_ok=True)
    _write_if_changed(tools_dir / "validate_draft.py", _VALIDATE_DRAFT_SCRIPT)

    draft_sh = workspace_root / "draft.sh"
    _write_if_changed(draft_sh, _DRAFT_SH_SCRIPT)
    draft_sh.chmod(0o755)

    edit_sh = workspace_root / "edit.sh"
    _write_if_changed(edit_sh, _EDIT_SH_SCRIPT)
    edit_sh.chmod(0o755)

    memory_sh = workspace_root / "memory.sh"
    _write_if_changed(memory_sh, _MEMORY_SH_SCRIPT)
    memory_sh.chmod(0o755)


# ---------------------------------------------------------------------------
# Pi-based agentic loop (primary — matches Jacquard architecture)
# ---------------------------------------------------------------------------

def _extract_pi_result(workspace_root: Path) -> dict | None:
    """Read the agent's output/result.json file."""
    result_path = workspace_root / "output" / "result.json"
    if not result_path.exists():
        for candidate in (workspace_root / "output").glob("*.json"):
            result_path = candidate
            break
        else:
            return None

    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "posts" in data:
            logger.info("[Stelle/Pi] Loaded result with %d posts from %s", len(data["posts"]), result_path.name)
            return data
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[Stelle/Pi] Failed to parse %s: %s", result_path, e)

    return None


_CITATION_COMMENT_RE = re.compile(r"^<!--\s*\[.*?\].*?-->\s*$", re.MULTILINE)


def _strip_citation_comments(text: str) -> str:
    """Remove inline citation HTML comments and squeeze blank lines."""
    cleaned = _CITATION_COMMENT_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_citation_comments(text: str) -> list[str]:
    """Extract all inline citation HTML comments from post text."""
    return _CITATION_COMMENT_RE.findall(text)


def _extract_result_from_scratch(workspace_root: Path) -> dict | None:
    """Fallback: reconstruct result JSON from Stelle's in-run scratch outputs.

    Priority order:
      1. scratch/final/ — canonical in-run output location (new default)
      2. scratch/drafts/ — legacy scratch location
      3. memory/draft-posts/ — last resort; this directory is supposed to hold
         Ordinal-pushed drafts only, but older runs wrote here so we still
         look as a fallback for crash recovery
    """
    search_dirs = [
        workspace_root / "scratch" / "final",
        workspace_root / "scratch" / "drafts",
        workspace_root / "memory" / "draft-posts",
    ]

    posts: list[dict] = []
    for drafts_dir in search_dirs:
        if not drafts_dir.exists():
            continue
        for f in sorted(drafts_dir.iterdir()):
            if not f.is_file() or f.suffix not in (".md", ".txt"):
                continue
            raw_text = f.read_text(encoding="utf-8").strip()
            if not raw_text or len(raw_text) < 100:
                continue
            clean_text = _strip_citation_comments(raw_text)
            lines = clean_text.split("\n")
            hook = lines[0].lstrip("#").strip()[:200] if lines else "Untitled"
            posts.append({
                "hook": hook,
                "text": clean_text,
                "origin": f"Extracted from {f.name}",
                "citations": [],
            })

    if posts:
        logger.info("[Stelle/Pi] Reconstructed %d posts from draft files", len(posts))
        return {"posts": posts, "verification": "(reconstructed from draft files)", "meta": {}}
    return None


# ---------------------------------------------------------------------------
# Direct API agentic loop (fallback when Pi is unavailable)
# ---------------------------------------------------------------------------

def _serialize_content_block(block: Any) -> dict:
    if block.type == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", ""), "signature": getattr(block, "signature", "")}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


_COMPACTION_RESERVE_TOKENS = 16_384
_COMPACTION_KEEP_RECENT = 20_000
_COMPACTION_CONTEXT_WINDOW = 200_000
_TOOL_RESULT_MAX_CHARS = 2_000

_SUMMARIZATION_SYSTEM = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified. "
    "Do NOT continue the conversation. Do NOT respond to any questions "
    "in the conversation. ONLY output the structured summary."
)

_SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured context \
checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the \
existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_TURN_PREFIX_SUMMARIZATION_PROMPT = """\
This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


def _estimate_message_tokens(msg: dict) -> int:
    """Rough token estimate for a message (chars/4 heuristic)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                for v in block.values():
                    if isinstance(v, str):
                        total += len(v)
            elif hasattr(block, "text"):
                total += len(getattr(block, "text", ""))
            elif hasattr(block, "thinking"):
                total += len(getattr(block, "thinking", ""))
        return total // 4
    return 0


def _truncate_for_summary(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"


def _serialize_messages_for_summary(messages: list[dict], end_idx: int) -> str:
    """Serialize messages[0:end_idx] into text for the summarization LLM."""
    parts: list[str] = []
    for msg in messages[:end_idx]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role.capitalize()}]: {content}")
        elif isinstance(content, list):
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            tool_results: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        text = str(block.get("content", ""))
                        tool_results.append(f"[Tool result]: {_truncate_for_summary(text)}")
                    elif btype == "text":
                        text_parts.append(block.get("text", ""))
                elif hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thinking_parts.append(getattr(block, "thinking", ""))
                    elif block.type == "tool_use":
                        args_str = json.dumps(block.input, ensure_ascii=False)
                        tool_calls.append(f"{block.name}({args_str[:500]})")
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {' '.join(thinking_parts)}")
            if text_parts:
                parts.append(f"[{role.capitalize()}]: {' '.join(text_parts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
            for tr in tool_results:
                parts.append(tr)
    return "\n\n".join(parts)


def _extract_file_ops(messages: list[dict], end_idx: int) -> tuple[set[str], set[str]]:
    """Extract files read and written/edited from tool calls in messages[0:end_idx]."""
    read_files: set[str] = set()
    modified_files: set[str] = set()
    for msg in messages[:end_idx]:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue
            args = block.input if isinstance(block.input, dict) else {}
            path = args.get("path") or args.get("file_path") or ""
            if not path:
                continue
            if block.name in ("read", "read_file"):
                read_files.add(path)
            elif block.name in ("write", "write_file", "edit", "str_replace_editor"):
                modified_files.add(path)
    read_only = read_files - modified_files
    return read_only, modified_files


def _format_file_ops(read_files: set[str], modified_files: set[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(sorted(read_files))}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(sorted(modified_files))}\n</modified-files>")
    return ("\n\n" + "\n\n".join(sections)) if sections else ""


_OVERFLOW_PATTERNS = [
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"exceeds the context window", re.IGNORECASE),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit exceeded", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),
]


def _is_context_overflow(error: Exception) -> bool:
    msg = str(error)
    return any(p.search(msg) for p in _OVERFLOW_PATTERNS)


class _CompactionState:
    """Tracks compaction state across the session for incremental merging."""

    __slots__ = ("previous_summary", "read_files", "modified_files", "compaction_count")

    def __init__(self) -> None:
        self.previous_summary: str | None = None
        self.read_files: set[str] = set()
        self.modified_files: set[str] = set()
        self.compaction_count: int = 0


def _inject_cache_breakpoint(messages: list[dict]) -> None:
    """Add cache_control to the last block of the last user message (in-place).

    This makes Anthropic cache everything from system prompt through this point.
    On subsequent turns, the prefix is a cache read at 0.1x cost.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
            ]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
        break


def _strip_cache_breakpoints(messages: list[dict]) -> None:
    """Remove cache_control markers from all messages (in-place).

    Called after API response so stale breakpoints don't accumulate.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


def _clean_messages_for_api(messages: list[dict]) -> list[dict]:
    """Return a cleaned copy of messages: drop error/aborted assistant turns."""
    cleaned: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                is_error = False
                for block in content:
                    if hasattr(block, "type") and block.type == "text":
                        text = getattr(block, "text", "")
                        if text.startswith("Error:") or text.startswith("error:"):
                            is_error = True
                            break
                if is_error and i + 1 < len(messages):
                    i += 1
                    continue
        cleaned.append(msg)
        i += 1
    return cleaned


def _should_compact(last_usage: dict | None, messages: list[dict]) -> bool:
    """Decide whether to compact based on real API usage or heuristic."""
    if last_usage:
        context_tokens = (
            last_usage.get("input_tokens", 0)
            + last_usage.get("output_tokens", 0)
            + last_usage.get("cache_read_input_tokens", 0)
            + last_usage.get("cache_creation_input_tokens", 0)
        )
        trailing = sum(
            _estimate_message_tokens(m)
            for m in messages[-2:]
        )
        return (context_tokens + trailing) > (_COMPACTION_CONTEXT_WINDOW - _COMPACTION_RESERVE_TOKENS)
    return sum(_estimate_message_tokens(m) for m in messages) > (
        _COMPACTION_CONTEXT_WINDOW - _COMPACTION_RESERVE_TOKENS
    )


def _find_turn_start(messages: list[dict], idx: int) -> int:
    """Walk backwards from idx to find the user message that started this turn."""
    for i in range(idx - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _is_tool_result_user_message(msg: dict) -> bool:
    """True when ``msg`` is a user-role message whose content contains any
    ``tool_result`` block.

    Such messages are structurally the *response* to the immediately
    preceding assistant ``tool_use``. Cutting the conversation at such a
    message orphans the ``tool_use_id`` reference — the Anthropic API
    rejects the next call with ``"unexpected tool_use_id found in
    tool_result blocks"``. Detected via both dict-shape blocks (replayed
    from history) and SDK-shape blocks (freshly returned from the API).
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "tool_result":
                return True
        elif getattr(block, "type", None) == "tool_result":
            return True
    return False


def _generate_summary(
    text: str,
    previous_summary: str | None,
    max_tokens: int,
) -> str:
    """Call the summarization model. Supports initial and incremental summaries."""
    if previous_summary:
        prompt_text = (
            f"<conversation>\n{text[:80000]}\n</conversation>\n\n"
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
            f"{_UPDATE_SUMMARIZATION_PROMPT}"
        )
    else:
        prompt_text = (
            f"<conversation>\n{text[:80000]}\n</conversation>\n\n"
            f"{_SUMMARIZATION_PROMPT}"
        )
    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=_SUMMARIZATION_SYSTEM,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return resp.content[0].text


def _generate_turn_prefix_summary(text: str, max_tokens: int) -> str:
    """Summarize the prefix of a split turn."""
    prompt_text = (
        f"<conversation>\n{text[:40000]}\n</conversation>\n\n"
        f"{_TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=_SUMMARIZATION_SYSTEM,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return resp.content[0].text


def _compact_messages(
    messages: list[dict],
    session_log: list[dict],
    state: _CompactionState,
    last_usage: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """Summarize old messages, keep recent ones. Supports split-turn and incremental merging."""
    if not force and not _should_compact(last_usage, messages):
        return messages

    tokens_before = sum(_estimate_message_tokens(m) for m in messages)
    summary_max_tokens = int(0.8 * _COMPACTION_RESERVE_TOKENS)

    keep_tokens = 0
    cut_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        keep_tokens += _estimate_message_tokens(messages[i])
        if keep_tokens >= _COMPACTION_KEEP_RECENT:
            cut_idx = i
            break

    if cut_idx <= 1:
        return messages

    is_split_turn = False
    turn_start_idx = -1

    if messages[cut_idx].get("role") == "user":
        # A user-role message that *looks* like a turn boundary isn't one
        # if it contains tool_result blocks — it's the response half of
        # an assistant(tool_use) ↔ user(tool_result) pair. Cutting there
        # strands the tool_result with no matching tool_use_id in the
        # checkpoint-acknowledgement we inject, and the next API call
        # 400s with ``unexpected tool_use_id``. Advance forward past the
        # full tool_use/tool_result chain until we land on a plain user
        # message (start of a fresh turn).
        while (
            cut_idx < len(messages)
            and (
                _is_tool_result_user_message(messages[cut_idx])
                or messages[cut_idx].get("role") != "user"
            )
        ):
            cut_idx += 1
        if cut_idx >= len(messages) - 1:
            # No clean boundary ahead — skip compaction this pass rather
            # than produce a malformed message history.
            return messages
    else:
        turn_start_idx = _find_turn_start(messages, cut_idx)
        if turn_start_idx >= 0 and turn_start_idx > 0:
            is_split_turn = True
        else:
            while cut_idx < len(messages) and messages[cut_idx].get("role") != "user":
                cut_idx += 1
            if cut_idx >= len(messages) - 1:
                return messages
            # Same hazard on this path: if we just advanced to a
            # tool_result user message, keep going to a plain user turn.
            while (
                cut_idx < len(messages)
                and _is_tool_result_user_message(messages[cut_idx])
            ):
                cut_idx += 1
            if cut_idx >= len(messages) - 1:
                return messages

    read_f, mod_f = _extract_file_ops(messages, cut_idx)
    state.read_files |= read_f
    state.modified_files |= mod_f

    try:
        if is_split_turn and turn_start_idx >= 0:
            history_end = turn_start_idx
            history_text = _serialize_messages_for_summary(messages, history_end) if history_end > 0 else ""
            prefix_text = _serialize_messages_for_summary(
                messages[turn_start_idx:cut_idx], cut_idx - turn_start_idx,
            )

            if history_text:
                history_summary = _generate_summary(history_text, state.previous_summary, summary_max_tokens)
            else:
                history_summary = state.previous_summary or "No prior history."

            prefix_max = int(0.5 * _COMPACTION_RESERVE_TOKENS)
            prefix_summary = _generate_turn_prefix_summary(prefix_text, prefix_max)

            summary = f"{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_summary}"
        else:
            old_text = _serialize_messages_for_summary(messages, cut_idx)
            summary = _generate_summary(old_text, state.previous_summary, summary_max_tokens)

    except Exception as e:
        logger.warning("[Stelle] Compaction summary failed: %s — skipping", e)
        return messages

    summary += _format_file_ops(state.read_files, state.modified_files)
    state.previous_summary = summary
    state.compaction_count += 1

    compacted = [
        {"role": "user", "content": f"[Session checkpoint #{state.compaction_count}]\n\n{summary}"},
        {"role": "assistant", "content": [{"type": "text", "text": "Understood. I have the full context from the checkpoint. Continuing."}]},
    ] + messages[cut_idx:]

    tokens_after = sum(_estimate_message_tokens(m) for m in compacted)
    pct = (1 - tokens_after / tokens_before) * 100 if tokens_before else 0
    split_label = " (split-turn)" if is_split_turn else ""
    logger.info(
        "[Stelle] Compaction #%d%s: %d→%d tokens (%.0f%% reduction, %d messages summarized)",
        state.compaction_count, split_label, tokens_before, tokens_after, pct, cut_idx,
    )
    session_log.append({
        "type": "compaction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compaction_number": state.compaction_count,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "messages_summarized": cut_idx,
        "messages_kept": len(messages) - cut_idx,
        "files_read": len(state.read_files),
        "files_modified": len(state.modified_files),
        "split_turn": is_split_turn,
    })
    return compacted


def _run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    workspace_root: Path,
    event_callback: Any = None,
    company_keyword: str | None = None,
) -> tuple[dict | None, list[dict]]:
    # Load scored observations once per run for the new tools that need them
    # (query_observations and execute_python). These are the client's entire
    # scored post history — full text, rewards, engagement metrics, audience
    # delta, ICP breakdown, and per-post reactor identities.
    scored_observations: list[dict] = []
    if company_keyword:
        try:
            from backend.src.db.local import (
                initialize_db, ruan_mei_load, get_engagers_for_post,
            )
            initialize_db()
            state = ruan_mei_load(company_keyword)
            if state:
                scored_observations = [
                    o for o in state.get("observations", [])
                    if o.get("status") in ("scored", "finalized")
                ]
                # Enrich each observation with its reactor identity list.
                # The observation dict already carries:
                #   post_body         — Stelle's original draft (after edit capture fix)
                #   posted_body       — the client-edited LinkedIn-live version
                #   reward.raw_metrics — impressions/reactions/comments/reposts
                #   reward.icp_reward — composite ICP-weighted reward
                #   icp_match_rate    — mean of per-reactor icp_score (continuous, 0..1)
                # And we JOIN in here:
                #   reactors          — list of {urn, name, headline, current_company,
                #                        title, location, icp_score} for every reactor
                #                        captured for this post. Raw continuous scores,
                #                        no bucketed categoricals.
                # This gives Stelle a single query that returns the full audience+delta
                # picture per post.
                for obs in scored_observations:
                    oid = (obs.get("ordinal_post_id") or "").strip()
                    if not oid:
                        obs["reactors"] = []
                        continue
                    try:
                        rows = get_engagers_for_post(oid)
                        obs["reactors"] = [
                            {
                                "urn": r.get("engager_urn", ""),
                                "name": r.get("name") or "",
                                "headline": r.get("headline") or "",
                                "current_company": r.get("current_company") or "",
                                "title": r.get("title") or "",
                                "location": r.get("location") or "",
                                "icp_score": r.get("icp_score"),
                                "engagement_type": r.get("engagement_type", "reaction"),
                            }
                            for r in rows
                        ]
                    except Exception as _e:
                        obs["reactors"] = []
        except Exception as e:
            logger.debug("[Stelle] Could not load scored observations: %s", e)


    # Irontomb calibration is handled via raw examples in the system
    # prompt — all scored posts are pre-loaded as calibration data.
    # No separate warmup loop needed.

    # Compute client median engagement rate (reactions per 1000 impressions)
    # from scored observations. This becomes the loss function reference —
    # Irontomb predictions below median indicate a draft that would
    # underperform this client's historical baseline.
    _engagement_rates: list[float] = []
    for _obs in scored_observations:
        _raw = (_obs.get("reward") or {}).get("raw_metrics", {})
        _imp = _raw.get("impressions", 0)
        _react = _raw.get("reactions", 0)
        if _imp > 0:
            _engagement_rates.append(_react / _imp * 1000)
    _engagement_rates.sort()
    client_median_engagement: float | None = None
    client_median_impressions: int | None = None
    if _engagement_rates:
        _mid = len(_engagement_rates) // 2
        client_median_engagement = (
            _engagement_rates[_mid] if len(_engagement_rates) % 2
            else (_engagement_rates[_mid - 1] + _engagement_rates[_mid]) / 2
        )
    _impression_values = sorted(
        (_obs.get("reward") or {}).get("raw_metrics", {}).get("impressions", 0)
        for _obs in scored_observations
        if ((_obs.get("reward") or {}).get("raw_metrics", {}).get("impressions", 0)) > 0
    )
    if _impression_values:
        _mid = len(_impression_values) // 2
        client_median_impressions = int(
            _impression_values[_mid] if len(_impression_values) % 2
            else (_impression_values[_mid - 1] + _impression_values[_mid]) / 2
        )

    # Iteration discipline counter — tracks how many times
    # simulate_flame_chase_journey has been called in this run. Used by the
    # write_result validator to reject submissions where the total simulate
    # call count is below the floor (3 × n_posts). Lives in a mutable list so
    # the closure handler and the validator can share state.
    simulate_call_count: list[int] = [0]
    # Track every simulate result so write_result can check predictions.
    # Each entry: {"draft_hash": str, "result": dict}
    simulate_results: list[dict] = []

    try:
        # Build a per-run tool dispatch table. Starts from the module-level
        # _TOOL_HANDLERS and adds the 4 new tools bound to this run's company
        # + scored observations.
        run_handlers: dict[str, Any] = dict(_TOOL_HANDLERS)

        from backend.src.agents.analyst import (
            _tool_query_observations as _analyst_query_obs,
            _tool_search_linkedin_bank as _analyst_search_bank,
            _tool_execute_python as _analyst_exec_py,
        )

        # Fields we strip from observations before returning to Stelle. These
        # are hand-authored categorical tags (topic_tag, format_tag,
        # source_segment_type) and derived scalars that bake in someone's
        # theory of what distinctions matter. The raw texts, raw metrics, and
        # raw reactor list already carry the signal. The tags remain on disk
        # for human dashboards but are not surfaced to the writer.
        _STELLE_STRIPPED_FIELDS = (
            "topic_tag", "format_tag", "source_segment_type",
            "edit_similarity", "active_directives", "constitutional_score",
            "constitutional_results", "cyrene_composite", "cyrene_dimensions",
            "cyrene_dimension_set", "cyrene_iterations", "cyrene_weights_tier",
            "alignment_score", "icp_segments",
        )

        def _stelle_scrub_obs(obs: dict) -> dict:
            return {k: v for k, v in obs.items() if k not in _STELLE_STRIPPED_FIELDS}

        def _stelle_query_observations_handler(_root: Path, args: dict) -> str:
            # Drop filter params that target stripped tag fields — if Stelle
            # passes them, they're silently ignored. Her tool schema no
            # longer advertises them, so she shouldn't pass them anyway.
            scrubbed_args = {k: v for k, v in args.items()
                             if k not in ("topic_filter", "format_filter")}
            scrubbed_obs = [_stelle_scrub_obs(o) for o in scored_observations]
            return _analyst_query_obs(scrubbed_args, scrubbed_obs)

        def _stelle_query_top_engagers_handler(_root: Path, args: dict) -> str:
            if not company_keyword:
                return json.dumps({"error": "company not set"})
            limit = min(args.get("limit", 20), 50)
            try:
                from backend.src.db.local import get_top_icp_engagers
                engagers = get_top_icp_engagers(company_keyword, limit=limit)
                return json.dumps({
                    "count": len(engagers),
                    "engagers": engagers,
                }, default=str)
            except Exception as e:
                return json.dumps({"error": f"Failed to load engagers: {str(e)[:200]}"})

        def _stelle_search_corpus_handler(_root: Path, args: dict) -> str:
            return _analyst_search_bank(args)

        def _stelle_execute_python_handler(_root: Path, args: dict) -> str:
            # Pipe embeddings into the subprocess so Stelle can operate
            # on raw continuous vectors (cosine sim, nearest-neighbor,
            # PCA) instead of categorical tags.
            try:
                from backend.src.utils.post_embeddings import get_post_embeddings
                _emb = get_post_embeddings(company_keyword)
            except Exception:
                _emb = None
            return _analyst_exec_py(args, scored_observations, embeddings=_emb)

        def _stelle_get_reader_reaction_handler(_root: Path, args: dict) -> str:
            """Dispatch a draft to Irontomb for a rough-reader reaction.

            Irontomb runs a short retrieval+react loop internally and
            returns {reaction, anchors[]} — a gestalt reader-voice
            reaction plus zero, one, or many inline anchors flagging
            specific reader-state-change spans. Stelle uses the
            anchors as a localized gradient: revise spans the reader
            anchored negatively, leave spans they didn't anchor or
            anchored positively. Don't rewrite what's working.
            """
            if not company_keyword:
                return json.dumps({"_error": "company not set"})
            draft = args.get("draft_text", "")
            simulate_call_count[0] += 1
            try:
                # CLI path: use Claude Max subscription instead of API
                from backend.src.mcp_bridge.claude_cli import use_cli
                if use_cli():
                    from backend.src.mcp_bridge.claude_cli import simulate_flame_chase_journey_cli
                    result = simulate_flame_chase_journey_cli(company_keyword, draft)
                else:
                    from backend.src.agents.irontomb import simulate_flame_chase_journey
                    result = simulate_flame_chase_journey(company_keyword, draft)
                _dh = result.get("_draft_hash", "")
                simulate_results.append({
                    "draft_hash": _dh,
                    "result": result,
                })

                # Irontomb outputs {reaction, anchors[]}. No scalar
                # prediction, so no gradient block.
                return json.dumps(result, default=str)
            except Exception as _e:
                logger.warning("[Stelle] Irontomb reader-reaction call failed: %s", _e)
                return json.dumps({"_error": f"simulate failed: {str(_e)[:200]}"})

        # Register the session-scoped handler. It closes over
        # simulate_call_count + simulate_results for iteration discipline,
        # so it must live in the per-run dict, not module-level _TOOL_HANDLERS.
        run_handlers["get_reader_reaction"] = _stelle_get_reader_reaction_handler

        def _stelle_check_client_comfort_handler(_root: Path, args: dict) -> str:
            """Dispatch a draft to Aglaea for a client-comfort check.

            Aglaea pulls the target FOC user's recent LinkedIn posts,
            past operator/client feedback on their drafts, and past
            (draft → published) edit deltas, then returns a 0-10 comfort
            score + specific flagged spans. Paired with Irontomb's
            engagement reaction, it gives Stelle a two-axis feedback
            signal: readers × author.
            """
            draft = args.get("draft_text", "")
            if not draft:
                return json.dumps({"_error": "draft_text is required"})
            # Target user resolution: stelle_runner sets DATABASE_USER_SLUG
            # when the generate endpoint resolved a specific FOC user.
            # Company slug is STELLE_COMPANY_KEYWORD.
            import os as _os
            _user_slug = (_os.environ.get("DATABASE_USER_SLUG") or "").strip() or None
            _company = company_keyword or None
            try:
                from backend.src.agents.aglaea import evaluate_client_comfort
                result = evaluate_client_comfort(
                    draft,
                    user_slug=_user_slug,
                    company_slug=_company,
                )
                return json.dumps(result, default=str)
            except Exception as _e:
                logger.warning("[Stelle] Aglaea comfort check failed: %s", _e)
                return json.dumps({"_error": f"aglaea failed: {str(_e)[:200]}"})

        run_handlers["check_client_comfort"] = _stelle_check_client_comfort_handler

        # Wrap submit_draft with a Castorice fact-check + strategic-fit
        # gate. The wrapper:
        #   1. Runs Castorice.fact_check_post on ``content`` and replaces
        #      the content with the corrected post.
        #   2. Runs Castorice.analyze_strategic_fit (against the corrected
        #      text + the latest Cyrene brief for this FOC) to produce a
        #      ~50-word operator-facing verdict.
        #   3. Routes each piece to its OWN field on local_posts (no more
        #      string-concat into one mega-why_post):
        #        - ``why_post``           = Castorice strategic_fit_note
        #          (operator-facing, glanceable in the Posts UI).
        #        - ``process_notes``      = Stelle's audit trail
        #          (provenance, Irontomb anchors, comfort score, length
        #          stats — collapsed in UI by default).
        #        - ``fact_check_report``  = Castorice's fact-check
        #          transcript (own UI expander).
        #        - ``citation_comments``  = Ordinal-pushable citation
        #          strings (own UI sub-list).
        # Failures surface as a visible error back to Stelle — she can
        # retry or submit unchecked after acknowledging the failure.
        _original_submit_draft = run_handlers.get("submit_draft")

        def _stelle_submit_draft_with_castorice(_root: Path, args: dict) -> str:
            if not company_keyword:
                return "Error: company not set"
            content = args.get("content") or ""
            if not content:
                return "Error: content is required"

            forwarded = apply_castorice_to_submit_args(company_keyword, args)

            if _original_submit_draft is None:
                return "Error: submit_draft dispatcher not wired"
            return _original_submit_draft(_root, forwarded)

        run_handlers["submit_draft"] = _stelle_submit_draft_with_castorice

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        session_log: list[dict[str, Any]] = []
        session_start = time.time()
        compaction_state = _CompactionState()
        last_api_usage: dict | None = None
        overflow_recovered = False

        session_log.append({
            "type": "session_start",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system_prompt_chars": len(system_prompt),
            "user_prompt": user_prompt,
        })

        system_with_cache = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
        ]

        result_json: str | None = None
        turn = 0

        while turn < MAX_AGENT_TURNS:
            turn += 1
            t0 = time.time()
            logger.info("[Stelle] Amphoreus cycle %d — calling Claude...", turn)

            _inject_cache_breakpoint(messages)

            try:
                with _client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=128000,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    system=system_with_cache,
                    tools=_TOOLS,
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()
            except Exception as e:
                logger.error("[Stelle] API error on turn %d: %s", turn, e)
                session_log.append({
                    "type": "error",
                    "turn": turn,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": str(e),
                })

                if _is_context_overflow(e) and not overflow_recovered:
                    logger.warning("[Stelle] Context overflow detected — emergency compaction")
                    if event_callback:
                        event_callback("status", {"message": "Context overflow — running emergency compaction..."})
                    overflow_recovered = True
                    if messages and messages[-1].get("role") == "assistant":
                        messages.pop()
                    messages = _compact_messages(
                        messages, session_log, compaction_state, last_api_usage, force=True,
                    )
                    turn -= 1
                    time.sleep(1)
                    continue

                if turn < 3:
                    time.sleep(5)
                    continue
                raise
            finally:
                _strip_cache_breakpoints(messages)

            elapsed = time.time() - t0

            usage_raw = response.usage
            usage = {
                "input_tokens": getattr(usage_raw, "input_tokens", 0),
                "output_tokens": getattr(usage_raw, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(usage_raw, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(usage_raw, "cache_read_input_tokens", 0),
            }
            last_api_usage = usage

            tool_calls = []
            thinking_parts = []
            text_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({"name": block.name, "id": block.id, "input": block.input})
                elif block.type == "thinking":
                    thinking_parts.append(getattr(block, "thinking", ""))
                elif block.type == "text":
                    text_parts.append(block.text)

            session_log.append({
                "type": "turn",
                "turn": turn,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(elapsed, 1),
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": usage,
                "tool_calls": tool_calls or None,
                "thinking": "\n".join(thinking_parts) if thinking_parts else None,
                "text": "\n".join(text_parts) if text_parts else None,
                "content_blocks": len(response.content),
            })

            turn_context = usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]
            turn_cache_pct = (usage["cache_read_input_tokens"] / turn_context * 100) if turn_context > 0 else 0
            status_msg = (
                f"Amphoreus cycle {turn} done in {elapsed:.1f}s — "
                f"in={usage['input_tokens']} out={usage['output_tokens']} "
                f"cache_read={usage['cache_read_input_tokens']} "
                f"cache_write={usage['cache_creation_input_tokens']} "
                f"({turn_cache_pct:.0f}% cached)"
            )
            logger.info(
                "[Stelle] Amphoreus cycle %d done in %.1fs — stop=%s blocks=%d "
                "in=%d out=%d cache_read=%d cache_write=%d (%.0f%% cached)",
                turn, elapsed, response.stop_reason, len(response.content),
                usage["input_tokens"], usage["output_tokens"],
                usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"],
                turn_cache_pct,
            )
            if event_callback:
                event_callback("status", {"message": status_msg})

            # Anthropic API rejects assistant messages whose FINAL content
            # block is a ``thinking`` block ("messages.N: The final block in
            # an assistant message cannot be 'thinking'"). This happens when
            # Claude's extended-thinking response ends on an unpaired
            # thinking block — rare, but seen in practice when the stream
            # is interrupted or the model emits thinking with no following
            # text/tool_use. Strip trailing thinking blocks before
            # appending; keep any earlier thinking blocks intact so the
            # signature verification on following turns still validates.
            assistant_content = list(response.content)
            while assistant_content and getattr(assistant_content[-1], "type", None) == "thinking":
                assistant_content.pop()
            if not assistant_content:
                # Degenerate response: no text / tool_use / non-thinking
                # content at all. Can't safely continue the conversation —
                # log and terminate the run cleanly.
                logger.warning(
                    "[Stelle] turn %d: response contained only thinking blocks "
                    "(stop_reason=%s); terminating run",
                    turn, response.stop_reason,
                )
                break
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text" and block.text.strip():
                        logger.info("[Stelle] Agent finished with text response on turn %d", turn)
                        if event_callback:
                            event_callback("text_delta", {"text": block.text})
                break

            tool_results = []
            tool_result_log = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                name = block.name
                args = block.input if isinstance(block.input, dict) else {}
                summary = _summarize_args(args)
                logger.info("[Stelle]   tool: %s(%s)", name, summary)
                if event_callback:
                    event_callback("tool_call", {"name": name, "arguments": summary})

                if name == "write_result":
                    raw_json = args.get("result_json", "")
                    # Gap 6: validate before accepting
                    try:
                        parsed = json.loads(raw_json)
                        passed, val_errors, val_warnings = _validate_output(parsed)
                        if not passed:
                            error_msg = "Validation failed:\n" + "\n".join(f"- {e}" for e in val_errors)
                            if val_warnings:
                                error_msg += "\nWarnings:\n" + "\n".join(f"- {w}" for w in val_warnings)
                            error_msg += "\n\nFix the issues and call write_result again."
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": error_msg,
                                "is_error": True,
                            })
                            tool_result_log.append({
                                "tool_name": name, "tool_call_id": block.id,
                                "result_chars": len(error_msg), "is_error": True,
                                "validation_errors": val_errors,
                            })
                            continue

                        # (Iteration discipline gate removed — no more
                        # minimum-simulate-count requirement.)
                        #
                        # (Scroll-stop gate removed — no more
                        # would_stop_scrolling boolean to gate on.)
                        #
                        # NOTE: An earlier refactor intended to delete the
                        # scroll-stop enforcement block but left orphaned
                        # code that referenced ``_failed_posts`` / ``_ph``
                        # / ``_post_text`` — variables only defined inside
                        # a parent ``for _pi, _post in enumerate(posts):``
                        # loop that was also deleted. That orphaned block
                        # produced a ``NameError: _failed_posts`` crash
                        # every time ``write_result`` succeeded validation,
                        # which dropped the entire structured output on the
                        # floor. The dead code has been removed here; if
                        # you want scroll-stop enforcement back, re-add the
                        # loop + variable declarations explicitly.


                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON: {e}\n\nFix the JSON syntax and call write_result again."
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": error_msg,
                            "is_error": True,
                        })
                        tool_result_log.append({
                            "tool_name": name, "tool_call_id": block.id,
                            "result_chars": len(error_msg), "is_error": True,
                        })
                        continue

                    # Stamp Irontomb's latest predictions onto each post
                    # so they appear in the final output. The model doesn't
                    # need to manually copy these — we pull them from
                    # simulate_results by draft hash.
                    for _post in parsed.get("posts", []):
                        _pt = (_post.get("text") or "").strip()
                        if not _pt:
                            continue
                        _ph = hashlib.sha256(_pt.encode("utf-8")).hexdigest()[:16]
                        for _sr in reversed(simulate_results):
                            if _sr["draft_hash"] == _ph:
                                _r = _sr["result"]
                                _post["last_reader_reaction"] = _r.get("reaction")
                                _post["last_reader_anchor"] = _r.get("anchor")
                                break
                    result_json = json.dumps(parsed)
                    result_text = f"Result accepted. {len(parsed.get('posts', []))} post(s). Session complete."
                    if val_warnings:
                        result_text += "\nWarnings: " + "; ".join(val_warnings)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                    tool_result_log.append({
                        "tool_name": name, "tool_call_id": block.id,
                        "result_chars": len(result_json), "is_error": False,
                    })
                elif name in run_handlers:
                    try:
                        output = run_handlers[name](workspace_root, args)
                        is_error = False
                    except Exception as e:
                        # database ingestion failures are FATAL. Don't convert
                        # to a tool error and let Stelle muddle on with
                        # missing data — re-raise so the subprocess crashes
                        # cleanly and job_manager records a failed run.
                        from backend.src.agents.database_client import (
                            DatabaseIngestionError as _LineageErr,
                        )
                        if isinstance(e, _LineageErr):
                            logger.error(
                                "[Stelle] FATAL: database ingestion failed during "
                                "%s — aborting run. %s", name, e,
                            )
                            if event_callback:
                                event_callback("error", {
                                    "message": f"database ingestion failed: {e}",
                                    "fatal": True,
                                })
                            raise
                        output = f"Error: {e}"
                        is_error = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:MAX_TOOL_OUTPUT_CHARS],
                    })
                    tool_result_log.append({
                        "tool_name": name, "tool_call_id": block.id,
                        "result_chars": len(output), "is_error": is_error,
                    })
                    if event_callback:
                        event_callback("tool_result", {
                            "name": name,
                            "result": output[:500],
                            "is_error": is_error,
                        })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {name}",
                        "is_error": True,
                    })
                    tool_result_log.append({
                        "tool_name": name, "tool_call_id": block.id,
                        "result_chars": 0, "is_error": True,
                    })

            if tool_result_log:
                session_log.append({
                    "type": "tool_results",
                    "turn": turn,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "results": tool_result_log,
                })

            if result_json is not None:
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            prev_count = compaction_state.compaction_count
            messages = _compact_messages(messages, session_log, compaction_state, last_api_usage)
            if compaction_state.compaction_count > prev_count and event_callback:
                event_callback("compaction", {"message": f"Context compaction #{compaction_state.compaction_count}"})

        total_elapsed = time.time() - session_start
        turns = [e for e in session_log if e.get("type") == "turn"]
        total_input = sum(e.get("usage", {}).get("input_tokens", 0) for e in turns)
        total_output = sum(e.get("usage", {}).get("output_tokens", 0) for e in turns)
        total_cache_read = sum(e.get("usage", {}).get("cache_read_input_tokens", 0) for e in turns)
        total_cache_create = sum(e.get("usage", {}).get("cache_creation_input_tokens", 0) for e in turns)

        total_context = total_input + total_cache_read + total_cache_create
        cache_hit_pct = (total_cache_read / total_context * 100) if total_context > 0 else 0

        cost_input = total_input / 1e6 * 15.0
        cost_output = total_output / 1e6 * 75.0
        cost_cache_read = total_cache_read / 1e6 * 1.50
        cost_cache_write = total_cache_create / 1e6 * 18.75
        cost_total = cost_input + cost_output + cost_cache_read + cost_cache_write

        session_log.append({
            "type": "session_end",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_turns": turn,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_create,
            "cache_hit_rate": round(cache_hit_pct, 1),
            "estimated_cost_usd": round(cost_total, 2),
            "cost_breakdown": {
                "uncached_input": round(cost_input, 4),
                "output": round(cost_output, 4),
                "cache_read": round(cost_cache_read, 4),
                "cache_write": round(cost_cache_write, 4),
            },
            "compactions": compaction_state.compaction_count,
            "has_result": result_json is not None,
        })

        logger.info(
            "[Stelle] Session complete: %d turns, %.1fs, cache_hit=%.1f%%, "
            "est_cost=$%.2f (in=$%.2f out=$%.2f cache_r=$%.2f cache_w=$%.2f), "
            "%d compaction(s)",
            turn, total_elapsed, cache_hit_pct,
            cost_total, cost_input, cost_output, cost_cache_read, cost_cache_write,
            compaction_state.compaction_count,
        )

        if turn >= MAX_AGENT_TURNS:
            logger.warning("[Stelle] Hit max turns (%d) without result", MAX_AGENT_TURNS)

        if result_json:
            try:
                return json.loads(result_json), session_log
            except json.JSONDecodeError as e:
                logger.error("[Stelle] Failed to parse result JSON: %s", e)
                json_match = re.search(r"\{[\s\S]*\}", result_json)
                if json_match:
                    try:
                        return json.loads(json_match.group()), session_log
                    except json.JSONDecodeError:
                        pass

        for msg in reversed(messages):
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if hasattr(block, "text") and block.text:
                        json_match = re.search(r"\{[\s\S]*\"posts\"[\s\S]*\}", block.text)
                        if json_match:
                            try:
                                return json.loads(json_match.group()), session_log
                            except json.JSONDecodeError:
                                continue
        return None, session_log

    finally:
        # Irontomb is stateless now — no background agent to tear down.
        # The old icp_agent / irontomb_agent spawn was removed when we
        # reverted to single-call + turn-based retrieval. Leaving the
        # finally block here for future teardown if needed.
        pass

def _summarize_args(args: dict) -> str:
    if "path" in args:
        return args["path"]
    if "objective" in args:
        return args["objective"][:60]
    if "query" in args:
        return args["query"][:60]
    if "command" in args:
        return args["command"][:60]
    if "url" in args:
        return args["url"][:80]
    if "result_json" in args:
        return f"({len(args['result_json'])} chars)"
    return str(args)[:60]


# ---------------------------------------------------------------------------
# Post-generation enrichment: "why post" + image suggestion
# ---------------------------------------------------------------------------

def _generate_why_post(
    post_text: str, origin: str, client_name: str, company_keyword: str,
) -> str:
    """Generate a concise explanation of why this post should be published."""
    strategy_snippet = ""
    try:
        strat_path = P.content_strategy_dir(company_keyword)
        if strat_path.exists():
            for f in sorted(strat_path.iterdir()):
                if f.suffix in (".txt", ".md") and f.stat().st_size < 50_000:
                    strategy_snippet = f.read_text(encoding="utf-8", errors="replace")[:3000]
                    break
    except Exception:
        pass

    strategy_block = ""
    if strategy_snippet:
        strategy_block = (
            f"\nContent strategy (excerpt):\n{strategy_snippet}\n\n"
            f"Tie your explanation directly to this strategy — which pillar, "
            f"theme, or audience segment does this post serve?\n"
        )

    _why_prompt = (
        f"Client: {client_name}\n"
        f"Post origin: {origin}\n"
        f"{strategy_block}\n"
        f"Post:\n{post_text}\n\n"
        f"Why should we publish this? 2-3 sentences max. "
        f"Say who specifically will care and what they'll do (save it, share it, DM the client, etc). "
        f"If there's a content strategy above, say which part this hits. "
        f"Write like you're explaining it to a teammate over coffee. "
        f"No words like 'strategically,' 'positions,' 'leverages,' 'resonates,' or 'ecosystem.' "
        f"Just say what makes it good in plain English."
    )

    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
    if _use_cli():
        txt = _cli_ss(_why_prompt, model="opus", max_tokens=400) or ""
        return txt.strip()

    try:
        resp = _call_with_retry(lambda: _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": _why_prompt}],
        ))
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("[Stelle] Why-post generation failed: %s", e)
        return ""


def _generate_image_suggestion(post_text: str, hook: str) -> str:
    """Generate a simple, easy-to-produce image suggestion for the post."""
    _img_prompt = (
        f"Post hook: {hook}\n\n"
        f"Post:\n{post_text}\n\n"
        f"Suggest ONE simple image a single graphic designer could make in "
        f"under 30 minutes. Think: a clean quote card, a bold stat highlight, "
        f"a minimal photo with a text overlay, or a simple before/after. "
        f"No intricate infographics, multi-panel illustrations, or complex "
        f"diagrams. Keep it to one sentence describing the visual and one "
        f"sentence describing any text on it. If the post works better as "
        f"text-only, just say 'Text-only'. No preamble."
    )

    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
    if _use_cli():
        txt = _cli_ss(_img_prompt, model="opus", max_tokens=200) or ""
        return txt.strip()

    try:
        resp = _call_with_retry(lambda: _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": _img_prompt}],
        ))
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("[Stelle] Image suggestion generation failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Result processing + fact-check
# ---------------------------------------------------------------------------

def _process_result(
    result: dict,
    client_name: str,
    company_keyword: str,
    output_filepath: str,
) -> str:
    posts = result.get("posts", [])
    if not posts:
        logger.warning("[Stelle] No posts in result")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"# {client_name} — One-Shot Posts\n\nNo posts generated.\n")
        return output_filepath

    logger.info("[Stelle] Processing %d posts...", len(posts))

    from backend.src.agents.castorice import Castorice
    try:
        from backend.src.db.local import create_local_post as _save_post
        import uuid as _uuid
        _sqlite_available = True
    except Exception:
        _sqlite_available = False

    # Resolve (company_uuid, user_id) from the company_keyword the
    # subprocess was spawned under. For pseudo-slugs like
    # ``trimble-heather`` this picks out the specific FOC user so rows
    # are stamped with user_id, letting the Posts tab disambiguate
    # Heather's drafts from Mark's.
    _company_uuid: str | None = None
    _user_id: str | None = None
    try:
        from backend.src.lib.company_resolver import resolve_to_company_and_user
        _company_uuid, _user_id = resolve_to_company_and_user(company_keyword)
    except Exception as _rexc:
        logger.debug("[Stelle] _process_result resolver failed: %s", _rexc)

    # ``submit_draft`` is the preferred per-post write path (Castorice
    # fact-check + persist + RuanMei observation all happen inside its
    # wrapper). If Stelle called submit_draft for a given post, then
    # the prompt-mandated follow-up ``write_result`` call would naively
    # re-do all that work and write a second local_posts row for the
    # same content.
    #
    # Per-run gating is unreliable (CLI vs native, past vs current
    # deploy). Per-post content-hash dedup is reliable: if a row with
    # the same (company, content) already exists in local_posts when we
    # reach this iteration, submit_draft already handled the post and
    # the heavy block below is pure duplicate spend.
    def _already_persisted(content: str) -> str | None:
        """Return the id of an existing local_posts row for this FOC
        user + content, or None.

        Scope order:
          - if a resolved ``user_id`` exists, match on user_id (most
            specific — catches Heather's drafts without clashing with
            Mark's even though both share the Trimble company UUID)
          - else match on the resolved company UUID
          - else fall back to the raw ``company_keyword``

        Matches either ``content`` (post-Castorice) or
        ``pre_revision_content`` (pre-Castorice). Castorice is
        non-deterministic so the raw-text match is what makes this
        robust across the submit_draft → write_result dedup boundary.
        """
        if not _sqlite_available or not content:
            return None
        try:
            from backend.src.db.local import get_connection
            with get_connection() as conn:
                if _user_id:
                    row = conn.execute(
                        "SELECT id FROM local_posts "
                        "WHERE user_id = ? AND (content = ? OR pre_revision_content = ?) "
                        "LIMIT 1",
                        (_user_id, content, content),
                    ).fetchone()
                else:
                    scope = _company_uuid or company_keyword
                    row = conn.execute(
                        "SELECT id FROM local_posts "
                        "WHERE company = ? AND (content = ? OR pre_revision_content = ?) "
                        "LIMIT 1",
                        (scope, content, content),
                    ).fetchone()
            return row[0] if row else None
        except Exception as _e:
            logger.debug("[Stelle] dedup lookup failed: %s", _e)
            return None

    castorice = Castorice()
    output_lines = [f"# {client_name.upper()} — ONE-SHOT POSTS (Stelle)\n"]
    output_lines.append(f"Generated {len(posts)} posts via jacquard-style agentic workflow.\n")

    verification = result.get("verification", "")
    if verification:
        output_lines.append(f"## Verification\n\n{verification}\n")

    meta = result.get("meta", {})
    if meta:
        if meta.get("reasoning"):
            output_lines.append(f"## Reasoning\n\n{meta['reasoning']}\n")
        missing = meta.get("missing_resources", [])
        if missing:
            output_lines.append("## Missing Resources\n\n" + "\n".join(f"- {r}" for r in missing) + "\n")

    output_lines.append("---\n")

    for i, post in enumerate(posts, 1):
        hook = post.get("hook", "")
        text = post.get("text", "")
        origin = post.get("origin", "")
        citations = post.get("citations", [])
        image_suggestion = post.get("image_suggestion")
        hook_variants = post.get("hook_variants", [])

        output_lines.append(f"## Post {i}: {hook}\n")
        output_lines.append(f"**Origin:** {origin}\n")
        output_lines.append(f"**Characters:** {len(text)}\n")

        _pred_eng = post.get("predicted_engagement")
        _pred_imp = post.get("predicted_impressions")
        if _pred_eng is not None or _pred_imp is not None:
            parts = []
            if _pred_eng is not None:
                parts.append(f"{_pred_eng:.1f} reactions/1k impressions")
            if _pred_imp is not None:
                parts.append(f"{_pred_imp:,} impressions")
            output_lines.append(f"**Irontomb Prediction:** {' · '.join(parts)}\n")

        if hook_variants:
            output_lines.append("**Hook Variants:**")
            for hv in hook_variants:
                output_lines.append(f"- {hv}")
            output_lines.append("")

        if citations:
            output_lines.append("**Citations (structured):**")
            for c in citations:
                output_lines.append(f"- {c.get('claim', '')}: {c.get('source', '')}")
            output_lines.append("")

        output_lines.append("### Draft\n")
        output_lines.append(text + "\n")

        # Per-post dedup: if submit_draft already persisted this exact
        # (company, content) in this session, skip the heavy block
        # (Castorice, why-post, image-suggestion, LLM validation,
        # RuanMei analysis, SQLite persist) — it would be a duplicate.
        _prior_id = _already_persisted(text)
        if _prior_id:
            output_lines.append(
                f"_Already persisted via `submit_draft` (draft_id: {_prior_id}) — "
                "skipping duplicate post-processing._\n"
            )
            output_lines.append("---\n")
            continue

        # -----------------------------------------------------------
        # Fact-check (Castorice) — the only post-processing step.
        # No iterative refinement (Cyrene) or constitutional verification.
        # Quality comes from RuanMei's context, not post-hoc patching.
        # The client is the quality gate.
        # -----------------------------------------------------------
        logger.info("[Stelle] Fact-checking post %d/%d: %s...", i, len(posts), hook[:50])
        corrected = text
        citation_comments: list[str] = []
        try:
            fc_result = castorice.fact_check_post(company_keyword, text)
            corrected = fc_result.get("corrected_post") or text
            raw_cc = fc_result.get("citation_comments") or []
            citation_comments = raw_cc if isinstance(raw_cc, list) else []
            fc_report_text = fc_result.get("report", "")
            if "[CORRECTED POST]" in fc_report_text:
                fc_header = fc_report_text[:fc_report_text.index("[CORRECTED POST]")].strip()
            else:
                fc_header = fc_report_text.strip()

            output_lines.append(f"### Fact-Check Report\n\n{fc_header}\n")
            output_lines.append(f"### Final Post\n\n{corrected}\n")
        except Exception as e:
            logger.warning("[Stelle] Fact-check failed for post %d: %s", i, e)
            citation_comments = []
            output_lines.append(f"### Fact-Check\n\nFact-check failed: {e}\n")
            output_lines.append(f"### Final Post\n\n{corrected}\n")

        logger.info("[Stelle] Generating why-post + image for post %d/%d...", i, len(posts))
        why_post = _generate_why_post(corrected, origin, client_name, company_keyword)
        if why_post:
            output_lines.append(f"### Why Post\n\n{why_post}\n")

        img_sug = _generate_image_suggestion(corrected, hook)
        if img_sug:
            output_lines.append(f"### Image Suggestion\n\n{img_sug}\n")
        elif image_suggestion:
            output_lines.append(f"### Image Suggestion\n\n{image_suggestion}\n")

        logger.info("[Stelle] Validating draft %d/%d...", i, len(posts))
        validation = _validate_draft_with_llm(corrected, company_keyword)
        if validation["issues"]:
            output_lines.append("### Validation Notes\n")
            for issue in validation["issues"]:
                sev = issue.get("severity", "info").upper()
                itype = issue.get("type", "unknown")
                desc = issue.get("description", "")
                offending = issue.get("offending_text", "")
                fix = issue.get("suggested_fix", "")
                line = f"- {sev} ({itype}): {desc}"
                if offending:
                    line += f' — "{offending}"'
                if fix:
                    line += f" — {fix}"
                output_lines.append(line)
            output_lines.append("")

        output_lines.append("---\n")

        import uuid as _uuid_for_draft

        _draft_id = str(_uuid_for_draft.uuid4())

        # RuanMei: analyze the generated post and record the observation.
        # Same local id as SQLite row so Ordinal push can link metrics by workspace post id.
        try:
            import asyncio as _asyncio
            from backend.src.agents.ruan_mei import RuanMei as _RM
            _rm_inst = _RM(company_keyword)
            _post_hash = __import__("hashlib").sha1(
                corrected.encode("utf-8", errors="replace")
            ).hexdigest()[:16]
            try:
                _aloop = _asyncio.get_running_loop()
            except RuntimeError:
                _aloop = None
            if _aloop and _aloop.is_running():
                import concurrent.futures as _cf
                _descriptor = _cf.ThreadPoolExecutor().submit(
                    lambda: _asyncio.run(_rm_inst.analyze_post(corrected))
                ).result(timeout=30)
            else:
                _descriptor = _asyncio.run(_rm_inst.analyze_post(corrected))
            _rm_inst.record(
                _post_hash,
                _descriptor,
                post_body=corrected,
                local_post_id=_draft_id,
            )

            # Generation-time metadata for downstream learning.
            _extra_fields = {}

            # (Analyst findings stamping removed — analyst findings are no
            # longer injected into Stelle's prompt, so stamping a version id
            # onto the observation has no consumer. Slot attribution and
            # landscape brief references were also removed earlier in the
            # prescriptive-injection strip.)

            # (Draft-scorer call removed 2026-04-23 as a Bitter Lesson
            # cleanup. The underlying regression model was retired on
            # 2026-04-11 — ``score_drafts`` had since been returning
            # ``model_source='no_model'`` rows that Stelle immediately
            # discarded, but the function itself still ran a Supabase
            # k-NN lookup on every generation. Dead compute on a path
            # whose only purpose was to feed a trained engagement
            # predictor we explicitly don't want to bring back.)

            if _extra_fields:
                for _obs in reversed(_rm_inst._state.get("observations", [])):
                    if _obs.get("post_hash") == _post_hash:
                        _obs.update(_extra_fields)
                        _rm_inst._save()
                        break
        except Exception as _e:
            logger.debug("[Stelle] RuanMei analysis skipped for post %d: %s", i, _e)

        if _sqlite_available:
            try:
                _gen_meta = dict(_extra_fields) if _extra_fields else {}
                # Stamp the run-level neighbor-signal stats onto this
                # draft's generation_metadata. Enables the offline
                # ``neighbor_signal_audit`` CLI to compare drafts
                # generated with neighbor context against those
                # without, without us needing to parse the bundle
                # string after the fact.
                if _bundle_stats:
                    _gen_meta["neighbor_signal_present"] = bool(
                        _bundle_stats.get("blocks_with_neighbors", 0) > 0
                    )
                    _gen_meta["neighbor_blocks_with_neighbors"] = int(
                        _bundle_stats.get("blocks_with_neighbors", 0)
                    )
                    _gen_meta["neighbor_bundle_blocks_total"] = int(
                        _bundle_stats.get("blocks_total", 0)
                    )
                    _gen_meta["neighbor_skip_reason"] = (
                        _bundle_stats.get("skip_reason") or None
                    )
                _save_post(
                    post_id=_draft_id,
                    # Use the resolved company UUID if available — falls
                    # back to the raw keyword (slug) for backward compat
                    # with callers that haven't been wired through the
                    # canonicalizer yet.
                    company=_company_uuid or company_keyword,
                    user_id=_user_id,
                    content=corrected,
                    title=hook[:200] if hook else None,
                    status="draft",
                    why_post=why_post or None,
                    citation_comments=citation_comments,
                    # Store the pre-Castorice raw text so a later
                    # submit_draft for the same post could dedup against
                    # this row if the ordering ever inverted.
                    pre_revision_content=text if text != corrected else None,
                    cyrene_score=None,
                    generation_metadata=_gen_meta if _gen_meta else None,
                )
            except Exception as _e:
                logger.warning("[Stelle] Could not save post %d to local SQLite: %s", i, _e)

        # DELETED: the old database-parallel-write path.
        # Stelle is read-only against Jacquard. Drafts land exclusively in
        # Amphoreus's local_posts table + output/ markdown mirror; the
        # operator pushes to Ordinal from Amphoreus's Posts tab. Nothing
        # is ever POSTed back to Jacquard's workspace or drafts table.

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    raw_result_path = output_filepath.replace(".md", "_result.json")
    with open(raw_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("[Stelle] Output written to %s", output_filepath)
    return output_filepath


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_one_shot(
    client_name: str,
    company_keyword: str,
    output_filepath: str,
    num_posts: int = 12,
    prompt: str | None = None,
    model: str = "claude-opus-4-6",
    event_callback: Any = None,
) -> str:
    """Generate a batch of LinkedIn posts.

    Stelle's in-context inputs are restricted to: raw transcripts (client-
    direct content), auto-captured draft→posted diffs, her own published-
    post history, and LinkedIn profile/research. Operator-curated layers
    (ABM, feedback files, revisions) and other agents' briefs (Cyrene) are
    NOT injected — everything beyond transcripts + observations is a tool
    call.
    """
    # Data-source detection is best-effort: if Jacquard's Supabase is
    # reachable and the company resolves, reads come from there. Otherwise
    # reads are empty. Either way Stelle runs; drafts always land in
    # Amphoreus's local_posts.
    try:
        from backend.src.agents.database_client import is_database_mode
        _data_source_active = is_database_mode()
    except Exception as _lm_err:
        logger.warning("[Stelle] workspace_fs unavailable: %s", _lm_err)
        _data_source_active = False

    if not _data_source_active:
        logger.info(
            "[Stelle] running without a client data source — transcripts/"
            "engagement/research reads will return empty. Ensure the "
            "company is registered in Jacquard's user_companies or "
            "expect generic output."
        )

    # --- CLI mode: run through Claude CLI with Max plan (no API cost) ---
    from backend.src.mcp_bridge.claude_cli import use_cli
    if use_cli():
        logger.info("[Stelle] CLI mode enabled — delegating to run_stelle_cli()")
        from backend.src.mcp_bridge.claude_cli import run_stelle_cli
        return run_stelle_cli(
            client_name=client_name,
            company_keyword=company_keyword,
            output_filepath=output_filepath,
            num_posts=num_posts,
            prompt=prompt,
            event_callback=event_callback,
        )

    logger.info("[Stelle] Starting agentic ghostwriter for %s...", client_name)

    P.ensure_dirs(company_keyword)

    # 2026-04-23: DISABLED — unpushed drafts are now retained across runs.
    #
    # Previously we wiped every unpushed draft for the company at run
    # start, on the theory that a draft not yet pushed to Ordinal
    # "didn't exist" and must not persist. That model broke once:
    #
    #   1. Operators started adding inline + post-wide comments to
    #      unpushed drafts (see draft_feedback + post_bundle). Wiping
    #      the draft orphaned its comments.
    #   2. ``build_post_bundle`` now surfaces unpushed drafts in a
    #      dedicated UNPUSHED section with their comments attached, so
    #      Stelle uses them as dedup signal exactly like Ordinal-pushed
    #      drafts. Re-generating the same topic two runs in a row is
    #      now prevented regardless of Ordinal state.
    #
    # The bundle still marks REJECTED drafts separately (learning
    # signal, not dedup). Nothing gets deleted here anymore — deletes
    # happen only via the explicit /api/posts/{id} DELETE route,
    # operator-initiated.

    logger.info("[Stelle] Setting up workspace...")
    workspace_root = _setup_workspace(company_keyword)

    # STRIPPED ARCHITECTURE (2026-04-10, extended 2026-04-11):
    # All prescriptive intelligence injection has been removed. Stelle operates
    # on raw workspace inputs (transcripts, voice examples, LinkedIn profile,
    # published-posts history) plus her own tools (query_observations,
    # search_linkedin_bank, execute_python, web_search, bash, read_file, etc.).
    # Previously this section injected RuanMei landscape insights, analyst
    # findings, content brief, hook library, trajectory sequences, market
    # intelligence, cross-client patterns, and curated ICP definitions — all
    # of which were Bitter Lesson violations: pre-chewed intermediate
    # representations sitting between raw data and the writer. Stelle now
    # queries raw observations/transcripts directly at generation time.

    # Unified post bundle — every post we have signal on, with body +
    # engagement + comments + edit delta + semantic neighbors bundled
    # per entry. Replaces the prior ``_fetch_all_ordinal_hooks`` hook-
    # list which stripped most of those facets. Two classes only:
    # POSTS (published + unpublished mixed, ship-state inferred from
    # the ENGAGEMENT line in each block) and REJECTED (paired comments
    # are learning signal, not dedup signal — client said no to the
    # execution, not the topic). See backend/src/services/post_bundle.py
    # for the full contract.
    existing_posts_context = ""
    # Bundle-build stats are captured at run-start and stamped onto
    # each saved draft's ``generation_metadata`` so a post-hoc audit
    # can correlate "did Stelle see neighbor context?" with the
    # resulting output. See backend/src/services/neighbor_signal_audit.py.
    _bundle_stats: dict = {}
    try:
        from backend.src.services.post_bundle import build_post_bundle_with_stats
        # Per-FOC scoping: DATABASE_USER_UUID is set by stelle_runner
        # when the ghostwriter endpoint resolved a target FOC. Passing
        # it here prevents the bundle from loading every sibling FOC's
        # drafts (2026-04-23 Virio ARG_MAX incident).
        _bundle_user_uuid = (os.environ.get("DATABASE_USER_UUID") or "").strip() or None
        # sort_by='engagement': order the POSTS block by reaction-count
        # desc so Stelle reads the actually-working high-engagement
        # posts FIRST. Recency-desc (the default) buried the proven
        # framework posts under the most recent (often underperforming)
        # drafts and Stelle pattern-matched the wrong voice. Each post
        # still carries posted_at; chronological signal is recoverable
        # from the per-post timestamps.
        existing_posts_context, _bundle_stats = build_post_bundle_with_stats(
            company_keyword, user_id=_bundle_user_uuid, sort_by="engagement",
        )
    except Exception as _e:
        logger.debug("[Stelle] post bundle build skipped: %s", _e)

    # Series Engine retired 2026-04-22 (BL cleanup) — the hand-designed
    # "content works as scheduled narrative arcs" theory was prescribing
    # content sequencing in code. Removed from the generation path; the
    # module still exists as dead code for future reference but is no
    # longer invoked from any agent.
    series_context = ""

    # Temporal Orchestrator: scheduling intelligence (operational).
    scheduling_context = ""
    try:
        from backend.src.services.temporal_orchestrator import build_scheduling_context as _sched_ctx
        scheduling_context = _sched_ctx(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Scheduling context skipped: %s", _e)

    base_prompt = (
        f"Write up to {num_posts} LinkedIn posts for {client_name}. "
        f"The transcripts are from content interviews — conversations designed "
        f"to surface post material. Mine them for everything worth writing about. "
        f"Only write as many posts as the transcripts can genuinely support with "
        f"distinct insights — if the material supports 7, write 7, not {num_posts}. "
        f"Quality and distinctness over quantity."
    )
    if prompt:
        user_prompt = f"{base_prompt}\n\nAdditional instructions from the user:\n{prompt}"
    else:
        user_prompt = base_prompt
    if existing_posts_context:
        user_prompt += existing_posts_context
    if scheduling_context:
        user_prompt += scheduling_context
    if series_context:
        user_prompt += series_context

    logger.info("[Stelle] Using direct API loop (max %d turns)...", MAX_AGENT_TURNS)
    directives = _build_dynamic_directives(company_keyword)
    system_prompt = _DIRECT_SYSTEM_TEMPLATE.format(dynamic_directives=directives)
    result, session_log = _run_agent_loop(
        system_prompt,
        user_prompt,
        workspace_root,
        event_callback=event_callback,
        company_keyword=company_keyword,
    )

    session_path = output_filepath.replace(".md", "_session.jsonl")
    Path(session_path).parent.mkdir(parents=True, exist_ok=True)
    with open(session_path, "w", encoding="utf-8") as f:
        for entry in session_log:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    logger.info("[Stelle] Session log saved to %s", session_path)

    if result is None:
        logger.warning("[Stelle] Agent did not produce a result. Writing empty output.")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"# {client_name} — One-Shot Posts\n\nAgent failed to produce output.\n")
        return output_filepath

    passed, val_errors, val_warnings = _validate_output(result)
    if not passed:
        logger.warning("[Stelle] Final output validation failed: %s", val_errors)

    post_count = len(result.get("posts", []))
    logger.info("[Stelle] Agent produced %d posts. Running fact-check...", post_count)

    output_path = _process_result(result, client_name, company_keyword, output_filepath)

    return output_path


# ---------------------------------------------------------------------------
# Inline single-post edit (re-enters Pi with --continue)
# ---------------------------------------------------------------------------

def inline_edit(
    company_keyword: str,
    post_text: str,
    instruction: str,
    client_name: str = "",
    event_callback: Any = None,
) -> str | None:
    """Revise a single post using the existing Pi session context.

    Re-enters the Pi agent with --continue so it retains full memory of
    transcripts, prior drafts, and conversation. Falls back to a direct
    Claude Haiku call if Pi is unavailable.
    """
    session_dir = P.memory_dir(company_keyword) / ".pi-sessions"
    has_sessions = session_dir.exists() and any(session_dir.glob("*.jsonl"))

    edit_prompt = (
        "Edit the following LinkedIn post based on the instruction. "
        "Keep the author's voice and style consistent with their other posts.\n\n"
        f"Full post:\n{post_text}\n\n"
        f"Instruction: {instruction}\n\n"
        "Return ONLY the revised full post text, nothing else. "
        "No JSON, no markdown fences, no explanation."
    )

    if _PI_AVAILABLE and has_sessions:
        logger.info("[Stelle] Inline edit via Pi --continue...")
        workspace_root = P.memory_dir(company_keyword)

        pi_cmd = [
            "pi",
            "--mode", "json",
            "-p",
            "--provider", "anthropic",
            "--model", "claude-opus-4-6",
            "--session-dir", str(session_dir),
            "--tools", "read,bash,edit,write,grep,find,ls",
            "--continue",
            edit_prompt,
        ]

        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

        try:
            proc = subprocess.Popen(
                pi_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(workspace_root), env=env,
                text=True, bufsize=1,
            )

            collected_text: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    etype = evt.get("type", "")
                    if etype == "message_update":
                        ae = evt.get("assistantMessageEvent", {})
                        if ae.get("type") == "text_delta":
                            delta = ae.get("textDelta", "")
                            collected_text.append(delta)
                            if event_callback:
                                event_callback("text_delta", {"text": delta})
                except json.JSONDecodeError:
                    pass

            proc.wait(timeout=30)
            output = "".join(collected_text).strip()

            if output and len(output) > 200:
                logger.info("[Stelle] Inline edit complete (%d chars)", len(output))
                return output

            logger.warning("[Stelle] Pi inline edit produced no usable output")
        except subprocess.TimeoutExpired:
            logger.warning("[Stelle] Pi inline edit timed out after 120s")
            if proc:
                proc.kill()
        except Exception as e:
            logger.warning("[Stelle] Pi inline edit failed: %s", e)

    logger.info("[Stelle] Inline edit via direct Claude Haiku call...")
    try:
        client = Anthropic()
        resp = _call_with_retry(lambda: client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": edit_prompt}],
        ))
        text = resp.content[0].text.strip() if resp.content else None
        if text and len(text) > 100:
            logger.info("[Stelle] Inline edit complete (%d chars)", len(text))
            if event_callback:
                event_callback("text_delta", {"text": text})
            return text
    except Exception as e:
        logger.error("[Stelle] Direct inline edit failed: %s", e)

    return None
