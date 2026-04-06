"""
Stelle — Jacquard-style agentic LinkedIn ghostwriter.

Delegates the agentic loop to Pi (pi.dev CLI), which handles context
compaction, session persistence, and tool execution natively.  Falls back
to a direct Anthropic API loop if Pi is not installed.

The agent explores the client workspace (transcripts, published posts,
content strategy, ABM profiles, feedback, revisions), drafts posts grounded
in transcripts, self-reviews against a "magic moment" quality bar, and
outputs structured JSON.  Posts are then fact-checked by Cyrene Terrae.
"""

from __future__ import annotations

import csv
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
logger = logging.getLogger("stelle")

_client = Anthropic()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APIMAESTRO_KEY = os.getenv("APIMAESTRO_API_KEY", "")
APIMAESTRO_HOST = os.getenv("APIMAESTRO_HOST", "")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
_PI_AVAILABLE = shutil.which("pi") is not None

MAX_AGENT_TURNS = 60
MAX_TOOL_OUTPUT_CHARS = 80_000
MAX_FETCH_CHARS = 12_000
MAX_BASH_OUTPUT_CHARS = 30_000

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
# System prompt — Pi-native AGENTS.md (primary path)
# ---------------------------------------------------------------------------

_PI_AGENTS_TEMPLATE = """\
# Ghostwriter

You ghostwrite LinkedIn posts for the client. Your workspace:

- `memory/config.md` — what you know about the client. Bounded at 4000 \
chars. Manage with `./memory.sh`.
- `memory/story-inventory.md` — cross-session record of stories already \
told (published or drafted) and candidate stories not yet used. Read this \
before drafting to avoid repeating angles the client has already published.
- `memory/profile.md` — client's LinkedIn profile, company facts, ICP \
segments, active initiatives, recent context.
- `memory/strategy.md` — content strategy, angles, cadence, guardrails.
- `memory/constraints.md` — voice/tone rules, brand safety, approval \
requirements.
- `memory/source-material/` — raw interview transcripts. Every claim traces \
here.
- `memory/references/` — articles, URLs, and reference material the client \
considers relevant. Treat these as supplementary source material — mine \
them for insights, angles, and supporting evidence just like transcripts.
- `memory/published-posts/` — Client's published posts with engagement \
metrics. Quality assured and exhibits their true voice.
- `memory/voice-examples/` — The client's **top posts by engagement**. \
These are the gold standard for voice, tone, and structure. Study these \
first and match this level of quality.
- `memory/draft-posts/` — Draft posts of unknown or unfinished quality — \
could exhibit AI tendencies.
- `memory/feedback/edits/` — writer's corrections to your past drafts \
(before/after diffs). Study these BEFORE writing.
- `memory/plan.md` — content calendar (if it exists).
- `scratch/` — your working space.
- `context/research/` — deep research on client and company.
- `context/org/` — company context — industry, positioning, competitors.
- `abm_profiles/` — ABM target briefings.
- `revisions/` — before/after revision pairs.

Read all files. Study `memory/voice-examples/` first — that is the voice \
to match. Then read `memory/published-posts/` for broader patterns.

## The Magic Moment

The client reads your posts and feels "you told my story better than I \
could've told it myself."

The anti-magic moment: "this isn't so different from what I could've \
gotten from ChatGPT." That's failure.

**Every post you ship should pass the magic moment test.** If a post \
doesn't, keep working on it until it does.

You have time to iterate. Don't ship a post until you'd honestly show it \
to the client and say "this captures your story better than you could've \
told it."

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

## Process

1. If `memory/plan.md` doesn't exist, create it first. Read every \
transcript and published post, mine them for candidate angles, cross-check \
against `memory/draft-posts/` and `memory/published-posts/` to avoid \
collisions, and write the plan.
1a. Re-read the plan you just wrote. For each pair of posts, ask: do \
these share the same core insight? If yes, kill one and replace it with \
a genuinely different angle from the transcripts. Two posts can share a \
topic domain but must have different underlying insights. Repeat until \
every post in the plan is distinct.
1b. Read `memory/story-inventory.md`. If it is empty or missing, build \
it now: scan every transcript and published/draft post, and write a list \
of every story, anecdote, or specific moment you find — one bullet per \
story, with file + timestamp + one-sentence description, and whether it \
has been used. If the inventory already exists, consult it before picking \
angles: do not draft a post around a story already marked as used. \
Do not mark stories as used yourself — the publisher marks them after \
confirmed Ordinal push.
2. Pick the next unwritten topic from the plan. Identify the specific \
source material (file + timestamps) you'll draw from.
3. Draft in `scratch/`. Read it back. Check it against \
`memory/config.md` and `memory/constraints.md`. Revise until it's right.
4. Submit the final version: `./draft.sh "your full post text" \
YYYY-MM-DDTHH:MM` (date from the plan, if any).
5. Repeat steps 2-4 for all planned posts.
6. When every post is drafted, write the final result JSON to \
`output/result.json`.

## Memory tool

`./memory.sh` manages `memory/config.md` (4000 char limit):
- `./memory.sh status` — check current usage
- `./memory.sh add "new fact"` — append (fails if over limit)
- `./memory.sh replace "old substring" "new content"` — update existing
- `./memory.sh remove "substring"` — delete an entry

When memory is above 80%, consolidate entries before adding new ones.

## Banned phrases

Words: "game-changer", "leverage", "unlock", "empower", "navigate", \
"deep dive", "buckle up"

Patterns (any variation counts — don't paraphrase around these):
- "nobody is talking about" / "nobody tells you" / "what people don't say"
- "Here's the thing" / "Here's what I've learned" / "Let me be honest" / \
"The truth is" / "The reality is"
- "It's not X, it's Y" / "It's not about X, it's about Y" — don't use \
reductive framing as a thesis
- "In today's..." / "In the world of..." / "In an era of..."
- "There is a gap between X and Y" as the main point — instead, just say \
what's true
- "let that sink in"

## LinkedIn feed

On mobile, only ~140 characters are visible before the "see more" fold. \
On desktop, ~210 characters.

## Hard constraints

- 1300-3000 characters.
- Every claim traces to a source file. No fabrication.
- No em-dashes. No emojis unless the client's published posts use them.
- No markdown formatting in posts (no #, **, etc.)
- Censor profanity: "sh*t" not "shit".

## Tools

- `./draft.sh "post text" YYYY-MM-DDTHH:MM` — submit a new draft
- `./edit.sh <draft-filename> "revised text"` — update an existing draft
- `./memory.sh` — manage config.md (see Memory tool section)

## Web Research

To search the web (Parallel API):
```bash
python3 tools/web_search.py "your search query here"
```

To extract content from a URL:
```bash
python3 tools/fetch_url.py "https://example.com/article"
```

To search for images (Serper Images API):
```bash
python3 tools/image_search.py "professional office teamwork" 10
```

## LinkedIn Post Database

Search 200K+ real LinkedIn posts with engagement metrics to study what \
formats, hooks, and angles drive the highest engagement across the industry:
```bash
python3 tools/query_posts.py "topic or keyword"
```

Search a specific creator's posts by username to study their style, hooks, \
and top performers:
```bash
python3 tools/query_posts.py --creator "username"
```

Use this to benchmark your drafts against what actually performs. Study the \
hooks, structures, and storytelling patterns of top-engagement posts in the \
client's space.

## Semantic Post Search

Search the post database by meaning, not just keywords:
```bash
python3 tools/semantic_search_posts.py "building trust through vulnerability in leadership"
```

This finds conceptually similar posts even when they use different words. \
Use alongside keyword search for deeper research — especially for abstract \
topics like tone, narrative structure, or emotional resonance.

## Ordinal Analytics

Get real LinkedIn performance data — follower growth, post engagement, \
and posting cadence:
```bash
python3 tools/ordinal_analytics.py profiles                # list scheduling profiles
python3 tools/ordinal_analytics.py followers <profileId>   # follower count + growth
python3 tools/ordinal_analytics.py posts <profileId>       # post impressions + engagement
python3 tools/ordinal_analytics.py cadence <profileId>     # posting frequency + gap analysis
```

Check posting cadence before writing — if the client hasn't posted in a \
while, prioritize timely/newsy hooks. Use follower and post analytics to \
understand what actually drives growth for this specific client.

## Draft Validation

Self-check a draft BEFORE submitting via `draft.sh`:
```bash
python3 tools/validate_draft.py "Your full post text here"
python3 tools/validate_draft.py --file memory/draft-posts/my-draft.md
python3 tools/validate_draft.py --attempt 2 --file memory/draft-posts/my-draft.md
```

Returns JSON with `needs_correction` (bool), `issues` array, and `attempt` \
number. Checks AI slop patterns, banned phrases, character count, and \
structural problems. If `needs_correction` is true, revise and re-validate \
with `--attempt N` (increment each time). After attempt 2, all issues are \
downgraded to info and `needs_correction` becomes false (escape hatch) — \
this prevents infinite revision loops. Track your attempt number!

**Content strategy from data**: Analyze the client's published posts in \
`memory/published-posts/`. Each file includes engagement metrics (reactions, \
comments, reposts, engagement score, outlier flags). Identify which topics, \
formats, angles, and hooks drive the strongest engagement. Use these patterns \
as your de facto content strategy — write more of what resonates, less of \
what falls flat. If no `memory/strategy.md` exists, intuit one entirely from \
the engagement data — you have everything you need. If a strategy document \
does exist, treat it as an intent layer (e.g. pivots, ABM targets, \
compliance) that can override the data, but default to what the numbers show.

## Output

When you're done, also write a JSON file to `output/result.json`:

```json
{{
  "posts": [
    {{
      "hook": "First sentence or opening line (required, <=200 chars)",
      "hook_variants": ["Alternative hook 1", "Alternative hook 2"],
      "text": "Full post text (required, <=3000 chars)",
      "origin": "What sparked this post (required)",
      "citations": [
        {{"claim": "exact number or factual assertion", "source": "where you found it"}}
      ],
      "image_suggestion": "Description of a complementary image, or null"
    }}
  ],
  "verification": "Prove you used all available material. List what you \
extracted from each workspace source (transcripts, notes, drafts). \
List what you skipped and why. Show this is complete, not lazy.",
  "meta": {{
    "reasoning": "Your approach and decisions",
    "missing_resources": ["Resources that would have helped"]
  }}
}}
```

## Hook Variants

For each post, generate 2-3 alternative opening lines (`hook_variants` in \
the output JSON). The client picks the best one. Make each variant a \
genuinely different angle on the same post — not just word swaps.

## Planning mode

When asked to create a content plan:
1. Read all memory files including feedback.
2. Mine transcripts for every usable story.
3. Check `memory/published-posts/` and `memory/draft-posts/` — don't \
repeat what's already been written.
4. Assign stories to post slots.
5. Write plan to `memory/plan.md`.
6. Self-dedup: re-read the plan. If any two posts share the same core \
insight (even if framed differently), kill one and replace it with a \
different angle. The plan should have zero overlap.

Plan format per post: date, story, source transcript + timestamp, key \
material.

When asked to write the next post from a plan:
1. Read `memory/plan.md`, find first `- [ ] Status: unwritten`.
2. Write that post following steps above.
3. Mark `- [x] Status: written`.

{dynamic_directives}
"""

# ---------------------------------------------------------------------------
# System prompt — direct API loop (fallback when Pi is unavailable)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM_TEMPLATE = """\
# Ghostwriter

You ghostwrite LinkedIn posts for the client. Your workspace:

- `memory/config.md` — what you know about the client. Bounded at 4000 \
chars.
- `memory/story-inventory.md` — cross-session record of stories already \
told (published or drafted) and candidate stories not yet used. Read this \
before drafting to avoid repeating angles the client has already published.
- `memory/profile.md` — client's LinkedIn profile, company facts, ICP \
segments, active initiatives, recent context.
- `memory/strategy.md` — content strategy, angles, cadence, guardrails.
- `memory/constraints.md` — voice/tone rules, brand safety, approval \
requirements.
- `memory/source-material/` — raw interview transcripts. Every claim traces \
here.
- `memory/references/` — articles, URLs, and reference material the client \
considers relevant. Treat these as supplementary source material — mine \
them for insights, angles, and supporting evidence just like transcripts.
- `memory/published-posts/` — Client's published posts with engagement \
metrics. Quality assured and exhibits their true voice.
- `memory/voice-examples/` — The client's **top posts by engagement**. \
These are the gold standard for voice, tone, and structure. Study these \
first and match this level of quality.
- `memory/draft-posts/` — Draft posts of unknown or unfinished quality — \
could exhibit AI tendencies.
- `memory/feedback/edits/` — writer's corrections to your past drafts \
(before/after diffs). Study these BEFORE writing.
- `memory/plan.md` — content calendar (if it exists).
- `scratch/` — your working space.
- `context/research/` — deep research on client and company.
- `context/org/` — company context — industry, positioning, competitors.
- `abm_profiles/` — ABM target briefings.
- `revisions/` — before/after revision pairs.

Read all files. Study `memory/voice-examples/` first — that is the voice \
to match. Then read `memory/published-posts/` for broader patterns.

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

- `list_directory` / `read_file` / `search_files` — explore the workspace
- `write_file` / `edit_file` — write scratch notes, draft posts, content plans
- `web_search` — search the web (Parallel API — returns ranked excerpts)
- `web_research` — deep research on a topic (Parallel API — synthesized analysis)
- `fetch_url` — extract content from a URL
- `bash` — run shell commands (curl, jq, scripts, etc.)
- `subagent` — spawn a fast, cheap sub-agent for focused research tasks
- `write_result` — submit your final posts (ends the session)
- `query_posts` — search 200K+ real LinkedIn posts by keyword, ranked by \
  engagement. Use to study what formats, hooks, and angles perform best in \
  the client's space. Also supports `--creator "username"` to search a \
  specific creator's posts.
- `semantic_search_posts` — search the post database by meaning, not \
  keywords. Finds conceptually similar posts even when they use different \
  words. Use for abstract topics like tone, narrative structure, or emotional \
  resonance.
- `ordinal_analytics` — get real LinkedIn performance data. Subcommands: \
  `profiles` (list scheduling profiles), `followers <id>` (follower growth), \
  `posts <id>` (post engagement), `cadence <id>` (posting frequency + gaps). \
  Check cadence before writing — if the client hasn't posted recently, \
  prioritize timely/newsy hooks.
- `validate_draft` — self-check a draft for AI patterns, banned phrases, \
  and structural issues BEFORE submitting. Returns JSON with issues. If \
  `needs_correction` is true, revise and re-validate with `--attempt N`. \
  After attempt 2, escape hatch activates (issues downgraded, proceeds).

## Process

1. If `memory/plan.md` doesn't exist, create it first. Read every \
transcript and published post, mine them for candidate angles, cross-check \
against `memory/draft-posts/` and `memory/published-posts/` to avoid \
collisions, and write the plan.
1a. Re-read the plan you just wrote. For each pair of posts, ask: do \
these share the same core insight? If yes, kill one and replace it with \
a genuinely different angle from the transcripts. Two posts can share a \
topic domain but must have different underlying insights. Repeat until \
every post in the plan is distinct.
1b. Read `memory/story-inventory.md`. If it is empty or missing, build \
it now: scan every transcript and published/draft post, and write a list \
of every story, anecdote, or specific moment you find — one bullet per \
story, with file + timestamp + one-sentence description, and whether it \
has been used. If the inventory already exists, consult it before picking \
angles: do not draft a post around a story already marked as used. \
Do not mark stories as used yourself — the publisher marks them after \
confirmed Ordinal push.
2. Pick the next unwritten topic from the plan. Identify the specific \
source material (file + timestamps) you'll draw from.
3. Draft in `scratch/`. Read it back. Revise until it's right.
4. Save each final draft to `memory/draft-posts/`.
5. When every post is complete, call `write_result` with the final JSON.

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
      "image_suggestion": "Description of a complementary image, or null"
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

## Banned phrases

Words: "game-changer", "leverage", "unlock", "empower", "navigate", \
"deep dive", "buckle up"

Patterns (any variation counts — don't paraphrase around these):
- "nobody is talking about" / "nobody tells you" / "what people don't say"
- "Here's the thing" / "Here's what I've learned" / "Let me be honest" / \
"The truth is" / "The reality is"
- "It's not X, it's Y" / "It's not about X, it's about Y" — don't use \
reductive framing as a thesis
- "In today's..." / "In the world of..." / "In an era of..."
- "There is a gap between X and Y" as the main point — instead, just say \
what's true
- "let that sink in"

## LinkedIn feed

On mobile, only ~140 characters are visible before the "see more" fold. \
On desktop, ~210 characters.

## Hard constraints

- 1300-3000 characters.
- Every claim traces to a source file. No fabrication.
- No em-dashes. No emojis unless the client's published posts use them.
- No markdown formatting in posts (no #, **, etc.)
- Censor profanity: "sh*t" not "shit".

**Content strategy from data**: Analyze the client's published posts in \
`memory/published-posts/`. Each file includes engagement metrics (reactions, \
comments, reposts, engagement score, outlier flags). Identify which topics, \
formats, angles, and hooks drive the strongest engagement. Use these patterns \
as your de facto content strategy — write more of what resonates, less of \
what falls flat. If no `memory/strategy.md` exists, intuit one entirely from \
the engagement data — you have everything you need. If a strategy document \
does exist, treat it as an intent layer (e.g. pivots, ABM targets, \
compliance) that can override the data, but default to what the numbers show.

## Planning mode

When asked to create a content plan:
1. Read all memory files including feedback.
2. Mine transcripts for every usable story.
3. Check `memory/published-posts/` and `memory/draft-posts/` — don't \
repeat what's already been written.
4. Assign stories to post slots.
5. Write plan to `memory/plan.md`.
6. Self-dedup: re-read the plan. If any two posts share the same core \
insight (even if framed differently), kill one and replace it with a \
different angle. The plan should have zero overlap.

Plan format per post: date, story, source transcript + timestamp, key \
material.

When asked to write the next post from a plan:
1. Read `memory/plan.md`, find first `- [ ] Status: unwritten`.
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
        "name": "search_files",
        "description": (
            "Search for a text pattern across all files in a workspace directory. "
            "Returns matching lines with filenames."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (default: entire workspace)",
                    "default": ".",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web via Parallel API. Returns ranked URLs with extended "
            "excerpts suitable for LLM consumption. Use to verify claims or "
            "research topics. Mode 'fast' for quick lookups, 'one-shot' for "
            "comprehensive single-query results, 'agentic' for token-efficient results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Natural-language description of what to find",
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional keyword queries to guide the search",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "one-shot", "agentic"],
                    "description": "Search mode (default: fast)",
                    "default": "fast",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["objective"],
        },
    },
    {
        "name": "web_research",
        "description": (
            "Deep web research via Parallel API. Same as web_search but uses "
            "'one-shot' mode for more comprehensive, synthesized results. Use "
            "when you need thorough background on a topic rather than quick facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Natural-language description of what to research",
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional keyword queries to guide the search",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["objective"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Extract content from a URL via Parallel API. Returns markdown-formatted "
            "excerpts. Useful for reading articles found via web_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract content from",
                },
            },
            "required": ["url"],
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
        "name": "subagent",
        "description": (
            "Spawn a fast, cheap sub-agent (Claude Haiku) for focused exploration "
            "or research. The sub-agent has access to web_search and returns a "
            "text response. Use for tasks like: 'Research recent news about X', "
            "'Summarize this topic for background context', 'Find the latest "
            "stats on Y'. The sub-agent cannot access workspace files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the sub-agent to complete",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to provide the sub-agent",
                    "default": "",
                },
            },
            "required": ["task"],
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

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
        text = resp.text
        if len(text) > MAX_FETCH_CHARS:
            text = text[:MAX_FETCH_CHARS] + f"\n\n... [truncated at {MAX_FETCH_CHARS} chars]"
        return text
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


_TOOL_HANDLERS = {
    "list_directory": lambda root, args: _exec_list_directory(root, args),
    "read_file": lambda root, args: _exec_read_file(root, args),
    "search_files": lambda root, args: _exec_search_files(root, args),
    "web_search": lambda root, args: _exec_web_search(args),
    "web_research": lambda root, args: _exec_web_search({**args, "mode": "one-shot"}),
    "fetch_url": lambda root, args: _exec_fetch_url(args),
    "write_file": lambda root, args: _exec_write_file(root, args),
    "edit_file": lambda root, args: _exec_edit_file(root, args),
    "bash": lambda root, args: _exec_bash(root, args),
    "subagent": lambda root, args: _exec_subagent(args),
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
    char_count = len(post_text)

    # Use learned char limit if available (from _build_overrides)
    char_min, char_max = 1300, 3000
    try:
        overrides = _build_overrides(company_keyword)
        if overrides.get("char_limit_min"):
            char_min = overrides["char_limit_min"]
        if overrides.get("char_limit_max"):
            char_max = overrides["char_limit_max"]
    except Exception:
        pass

    if char_count > 3000:  # LinkedIn platform limit (not ours — theirs)
        result["needs_correction"] = True
        result["issues"].append({
            "type": "char_count",
            "description": f"Post is {char_count} chars — exceeds LinkedIn's 3000 char limit",
            "severity": "critical",
            "offending_text": "",
            "suggested_fix": f"Cut {char_count - 2800} characters",
        })
    elif char_count < char_min:
        result["issues"].append({
            "type": "char_count",
            "description": f"Post is {char_count} chars — below {'learned' if char_min != 1300 else 'default'} minimum {char_min} for this client",
            "severity": "info",
            "offending_text": "",
            "suggested_fix": f"This client's accepted posts are typically {char_min}-{char_max} chars",
        })

    return result


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _fetch_linkedin_profile(company_keyword: str) -> str | None:
    """Fetch the client's LinkedIn profile summary via APIMaestro."""
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        return None

    username = username_path.read_text().strip()
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
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        logger.info("[Stelle] No linkedin_username.txt for %s — skipping Supabase", company_keyword)
        return [], set()

    username = username_path.read_text().strip()
    if not username or not SUPABASE_URL or not SUPABASE_KEY:
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
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        return []

    username = username_path.read_text().strip()
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
    csv_path = P.ordinal_auth_csv()
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                slug = (row.get("provider_org_slug") or "").strip()
                if slug.lower() == company_keyword.lower():
                    key = (row.get("api_key") or "").strip()
                    return key if key.startswith("ord_") else None
    except Exception as e:
        logger.warning("[Stelle] Failed to read ordinal_auth_rows.csv: %s", e)
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


# ---------------------------------------------------------------------------
# Research files from Supabase (Gap 8)
# ---------------------------------------------------------------------------

def _fetch_research_files(company_keyword: str) -> list[dict]:
    """Fetch parallel_research_results for person and company from Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    files: list[dict] = []
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        return []

    _SB_HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

    for research_type in ("person", "company"):
        try:
            filter_key = "user_id" if research_type == "person" else "company_id"
            resp = httpx.get(
                f"{SUPABASE_URL}/rest/v1/parallel_research_results",
                params={
                    "select": "output,basis,created_at",
                    "research_type": f"eq.{research_type}",
                    "error": "is.null",
                    "order": "created_at.desc",
                    "limit": "1",
                },
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


# ---------------------------------------------------------------------------
# Workspace setup (Gap 3, 10 — jacquard directory structure)
# ---------------------------------------------------------------------------

def _setup_workspace(company_keyword: str) -> Path:
    """Stage workspace matching Jacquard's memory/ layout.

    Uses a persistent workspace under data/workspaces/ instead of a temp dir
    so that snapshots, rollback, and feedback all share the same root.
    Symlinks and Supabase-sourced files are rebuilt on each run.
    """
    from backend.src.db import vortex as _P
    workspace = _P.workspace_dir(company_keyword)
    workspace.mkdir(parents=True, exist_ok=True)

    mem = workspace / "memory"
    if mem.exists():
        shutil.rmtree(mem)
    ctx = workspace / "context"
    if ctx.exists():
        shutil.rmtree(ctx)
    for d in ("abm_profiles", "revisions"):
        p = workspace / d
        if p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    client_mem = P.memory_dir(company_keyword)

    # ----- memory/ hierarchy (Jacquard-style) -----
    mem = workspace / "memory"
    mem.mkdir()

    # memory/source-material/ — raw interview transcripts
    transcripts_src = client_mem / "transcripts"
    source_mat = mem / "source-material"
    if transcripts_src.exists():
        os.symlink(transcripts_src.resolve(), source_mat)
        for pdf in transcripts_src.glob("*.pdf"):
            txt_companion = source_mat / (pdf.name + ".txt")
            if not txt_companion.exists():
                txt_companion.write_text(
                    _extract_pdf_text(pdf), encoding="utf-8",
                )
    else:
        source_mat.mkdir()

    # memory/references/ — client-provided URLs, articles, reference material
    refs_src = client_mem / "references"
    refs_dst = mem / "references"
    if refs_src.exists() and any(refs_src.iterdir()):
        os.symlink(refs_src.resolve(), refs_dst)
    else:
        refs_dst.mkdir()

    # memory/published-posts/ — from Supabase
    pub_posts, _published_dates = _fetch_published_posts(company_keyword)
    pub_dir = mem / "published-posts"
    pub_dir.mkdir()
    for post in pub_posts:
        (pub_dir / post["filename"]).write_text(post["content"], encoding="utf-8")

    # memory/voice-examples/ — top posts by engagement as style exemplars
    voice_dir = mem / "voice-examples"
    voice_dir.mkdir()
    voice_examples = _fetch_voice_examples(company_keyword)
    for ve in voice_examples:
        (voice_dir / ve["filename"]).write_text(ve["content"], encoding="utf-8")

    # memory/draft-posts/ — draft.sh writes here; seed from past_posts
    draft_posts_dir = mem / "draft-posts"
    draft_posts_dir.mkdir()
    past_src = client_mem / "past_posts"
    if past_src.exists():
        for f in sorted(past_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                os.symlink(f.resolve(), draft_posts_dir / f.name)

    # memory/feedback/edits/ — client feedback + auto-saved edit diffs
    feedback_dir = mem / "feedback"
    feedback_dir.mkdir()
    edits_dir = feedback_dir / "edits"
    edits_dir.mkdir()
    fb_src = client_mem / "feedback"
    if fb_src.exists():
        for f in sorted(fb_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                os.symlink(f.resolve(), feedback_dir / f.name)

    # memory/profile.md — LinkedIn profile data
    profile_summary = _fetch_linkedin_profile(company_keyword)
    if profile_summary:
        (mem / "profile.md").write_text(profile_summary, encoding="utf-8")

    # memory/strategy.md — content strategy
    cs_src = client_mem / "content_strategy"
    strategy_parts: list[str] = []
    if cs_src.exists():
        for f in sorted(cs_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                strategy_parts.append(f.read_text(encoding="utf-8", errors="replace"))
    if strategy_parts:
        (mem / "strategy.md").write_text("\n\n---\n\n".join(strategy_parts), encoding="utf-8")

    # memory/constraints.md — voice/tone rules from accepted posts
    accepted_src = client_mem / "accepted"
    if accepted_src.exists() and any(accepted_src.iterdir()):
        constraint_lines = ["Voice and tone reference from accepted posts:\n"]
        for f in sorted(accepted_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                constraint_lines.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')}\n")
        (mem / "constraints.md").write_text("\n".join(constraint_lines), encoding="utf-8")

    # memory/config.md — bounded agent memory, persisted across sessions
    persistent_config = client_mem / "config.md"
    ws_config = mem / "config.md"
    if persistent_config.exists():
        os.symlink(persistent_config.resolve(), ws_config)
    else:
        persistent_config.parent.mkdir(parents=True, exist_ok=True)
        persistent_config.write_text("", encoding="utf-8")
        os.symlink(persistent_config.resolve(), ws_config)

    # memory/story-inventory.md — cross-session record of stories told/untold
    persistent_inventory = client_mem / "story_inventory.md"
    ws_inventory = mem / "story-inventory.md"
    if persistent_inventory.exists():
        os.symlink(persistent_inventory.resolve(), ws_inventory)
    else:
        persistent_inventory.parent.mkdir(parents=True, exist_ok=True)
        persistent_inventory.write_text("", encoding="utf-8")
        os.symlink(persistent_inventory.resolve(), ws_inventory)

    # ----- context/ hierarchy (our additions beyond Jacquard) -----
    ctx = workspace / "context"
    ctx.mkdir()

    # context/org/ — company context
    org_dir = ctx / "org"
    org_dir.mkdir()
    if cs_src.exists():
        for f in sorted(cs_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                os.symlink(f.resolve(), org_dir / f.name)

    # context/research/ — from Supabase parallel_research_results
    research_files = _fetch_research_files(company_keyword)
    research_dir = ctx / "research"
    research_dir.mkdir()
    for rf in research_files:
        (research_dir / rf["filename"]).write_text(rf["content"], encoding="utf-8")

    # context/topic-velocity.md — recent industry signal from Serper
    tv_src = client_mem / "topic_velocity.md"
    if tv_src.exists():
        os.symlink(tv_src.resolve(), ctx / "topic-velocity.md")

    # ----- Other top-level dirs -----
    for subdir in ("abm_profiles", "revisions"):
        src = client_mem / subdir
        dst = workspace / subdir
        if src.exists():
            os.symlink(src.resolve(), dst)

    # scratch/ and output/ — preserved across runs
    scratch_dir = workspace / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    (scratch_dir / "drafts").mkdir(exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)

    logger.info(
        "[Stelle] Workspace staged at %s (%d published posts, %d voice examples, %d research files)",
        workspace, len(pub_posts), len(voice_examples), len(research_files),
    )

    return workspace


# ---------------------------------------------------------------------------
# Learned overrides — graduated defaults from data
# ---------------------------------------------------------------------------

_OVERRIDE_MIN_EDITS_FOR_CHAR_LIMIT = 5


def _build_overrides(company_keyword: str) -> dict:
    """Check feedback history and RuanMei insights for learnable overrides.

    Returns dict with keys like "char_limit" that replace hard-coded defaults
    when enough data supports it. Logs when an override activates.
    """
    overrides: dict[str, Any] = {}

    # --- 1. Character limit override ---
    # If the feedback engine has 5+ edit deltas where the human editor
    # consistently makes posts longer/shorter, adjust the range.
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company_keyword)
        scored = [
            o for o in rm._state.get("observations", [])
            if o.get("status") == "scored"
            and (o.get("posted_body") or "").strip()
        ]
        # Use posted_body (final accepted text) char counts
        accepted_lengths = [len(o["posted_body"].strip()) for o in scored if o.get("posted_body", "").strip()]

        if len(accepted_lengths) >= _OVERRIDE_MIN_EDITS_FOR_CHAR_LIMIT:
            accepted_lengths.sort()
            median_len = accepted_lengths[len(accepted_lengths) // 2]
            learned_min = max(400, int(median_len * 0.7))
            learned_max = max(learned_min + 200, int(median_len * 1.3))
            # Only override if the learned range differs meaningfully from default
            if learned_min != 1300 or learned_max != 3000:
                overrides["char_limit"] = f"{learned_min}-{learned_max}"
                overrides["char_limit_min"] = learned_min
                overrides["char_limit_max"] = learned_max
                overrides["char_limit_median"] = median_len
                overrides["char_limit_sample_size"] = len(accepted_lengths)
                logger.info(
                    "[Stelle] Override active for %s: char_limit=%s (median=%d, n=%d)",
                    company_keyword, overrides["char_limit"], median_len, len(accepted_lengths),
                )
    except Exception as e:
        logger.debug("[Stelle] Char limit override check failed for %s: %s", company_keyword, e)

    # --- 2. Cadence override ---
    # If RuanMei insights show a specific cadence pattern, inject as system-level.
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company_keyword)
        scored = [o for o in rm._state.get("observations", []) if o.get("status") == "scored"]
        if len(scored) >= 10:
            # Compute actual posting cadence from data
            timestamps = []
            for o in scored:
                ts = o.get("posted_at") or o.get("recorded_at", "")
                if ts:
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt)
                    except Exception:
                        pass
            if len(timestamps) >= 5:
                timestamps.sort()
                gaps = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
                gaps = [g for g in gaps if g > 0]  # filter same-day
                if gaps:
                    median_gap = sorted(gaps)[len(gaps) // 2]
                    if median_gap <= 2:
                        cadence_note = f"This client posts frequently (median {median_gap} days between posts). Maintain this momentum."
                    elif median_gap <= 4:
                        cadence_note = f"This client posts ~{7 // median_gap}x per week (median gap: {median_gap} days). Match this rhythm."
                    elif median_gap <= 7:
                        cadence_note = f"This client posts weekly (median gap: {median_gap} days)."
                    else:
                        cadence_note = f"This client posts infrequently (median gap: {median_gap} days). Each post matters more — prioritize quality."
                    overrides["cadence_note"] = cadence_note
                    logger.info(
                        "[Stelle] Override active for %s: cadence (median_gap=%d days)",
                        company_keyword, median_gap,
                    )
    except Exception as e:
        logger.debug("[Stelle] Cadence override check failed for %s: %s", company_keyword, e)

    return overrides


def _apply_overrides_to_prompt(prompt: str, overrides: dict) -> str:
    """Replace hard-coded values in system prompt with learned overrides."""
    if not overrides:
        return prompt

    # Character limit override
    char_limit = overrides.get("char_limit")
    if char_limit:
        prompt = prompt.replace("- 1300-3000 characters.", f"- {char_limit} characters.")

    # Cadence override — inject as a system-level directive
    cadence_note = overrides.get("cadence_note")
    if cadence_note:
        # Insert cadence insight right after the hard constraints section
        cadence_block = f"\n## Posting Cadence (learned from data)\n\n{cadence_note}\n"
        # Insert before the Tools section
        if "## Tools" in prompt:
            prompt = prompt.replace("## Tools", cadence_block + "## Tools", 1)
        elif "## Process" in prompt:
            prompt = prompt.replace("## Process", cadence_block + "## Process", 1)

    return prompt


# ---------------------------------------------------------------------------
# Dynamic directives
# ---------------------------------------------------------------------------

def _build_dynamic_directives(company_keyword: str) -> str:
    """Assemble per-client context appended to Stelle's system prompt template.

    Section order is intentional. Learned writing rules are front-loaded so
    the LLM sees the empirically-distilled directives before the much longer
    content strategy document (~16k chars for innovocommerce). At 21k+ chars
    of combined context, attention dilutes toward the end — the small,
    evidence-backed rules deserve the top slot.

    Strategy brief context is NOT injected here. It's injected into the user
    prompt via ``build_stelle_strategy_context`` in ``generate_one_shot``. The
    split is deliberate: per-generation recommendations belong in the user
    message (ephemeral, specific to this invocation); persistent client
    identity belongs in the system prompt.
    """
    sections = []

    # 1. Learned writing rules (front-loaded) — small, distilled, high-signal.
    try:
        from backend.src.utils.feedback_distiller import build_stelle_directives_section
        _learned = build_stelle_directives_section(company_keyword)
        if _learned:
            sections.append(_learned)
    except Exception:
        pass

    # 2. Content strategy — the human strategist's intent layer. Large but
    #    authoritative; can override what the engagement data suggests.
    cs_dir = P.content_strategy_dir(company_keyword)
    if cs_dir.exists():
        for f in sorted(cs_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                sections.append(f"## Content Strategy\n\n{f.read_text(encoding='utf-8', errors='replace')}")

    # 3. ABM targets.
    abm = P.abm_dir(company_keyword)
    if abm.exists():
        for f in sorted(abm.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    sections.append(
                        f"## ABM Targets\n\n"
                        f"Some posts should strategically mention these companies/products. "
                        f"Weave mentions in naturally — never force them.\n\n{text}"
                    )

    # 4. Client feedback on previous drafts.
    fb_dir = P.feedback_dir(company_keyword)
    if fb_dir.exists():
        fb_texts = []
        for f in sorted(fb_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                fb_texts.append(f.read_text(encoding="utf-8", errors="replace"))
        if fb_texts:
            sections.append(
                f"## Client Feedback on Previous Drafts\n\n"
                f"Learn from this feedback — avoid the same mistakes.\n\n"
                + "\n---\n".join(fb_texts)
            )

    # 5. Before/after revision pairs.
    rev_dir = P.revisions_dir(company_keyword)
    if rev_dir.exists():
        rev_texts = []
        for f in sorted(rev_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                rev_texts.append(f.read_text(encoding="utf-8", errors="replace"))
        if rev_texts:
            sections.append(
                f"## Before/After Revisions\n\n"
                f"Study the delta between pipeline drafts and human-revised versions. "
                f"Internalize the patterns.\n\n"
                + "\n---\n".join(rev_texts)
            )

    return "\n\n".join(sections) if sections else ""


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
if [ "$CHARS" -lt 1300 ]; then
  echo "REJECTED: ${CHARS} characters (minimum 1300)."
  exit 1
fi
if [ "$CHARS" -gt 3000 ]; then
  echo "REJECTED: ${CHARS} characters (maximum 3000)."
  exit 1
fi
SLUG=$(echo "$CLEAN_POST" | head -1 | head -c 60 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr ' ' '-' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
DATE=$(date +%Y-%m-%d)
mkdir -p memory/draft-posts
FILENAME="memory/draft-posts/${DATE}-${SLUG}.md"
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
if [ "$CHARS" -lt 1300 ]; then
  echo "REJECTED: ${CHARS} characters (minimum 1300)."
  exit 1
fi
if [ "$CHARS" -gt 3000 ]; then
  echo "REJECTED: ${CHARS} characters (maximum 3000)."
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

_WEB_SEARCH_SCRIPT = '''\
#!/usr/bin/env python3
"""Search the web via Parallel API. Usage: python3 web_search.py "query" """
import hashlib, json, os, sys
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratch", ".cache")

def _cache_get(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"search-{hashlib.md5(key.encode()).hexdigest()}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return None

def _cache_set(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"search-{hashlib.md5(key.encode()).hexdigest()}.txt")
    open(path, "w", encoding="utf-8").write(value)

def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python3 web_search.py \\"your query\\""); return
    cached = _cache_get(query)
    if cached:
        print("[cached]"); print(cached); return
    key = os.environ.get("PARALLEL_API_KEY", "")
    if not key:
        print("PARALLEL_API_KEY not set — web search unavailable"); return
    try:
        resp = httpx.post(
            "https://api.parallel.ai/v1beta/search",
            json={"objective": query, "mode": "fast", "max_results": 8},
            headers={"x-api-key": key, "Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        lines = []
        for r in resp.json().get("results", []):
            lines.append(f"Title: {r.get('title', 'N/A')}")
            lines.append(f"URL: {r.get('url', '')}")
            d = r.get("publish_date")
            if d: lines.append(f"Date: {d}")
            for exc in r.get("excerpts", [])[:2]:
                lines.append(f"  {exc[:500]}")
            lines.append("")
        output = "\\n".join(lines)
        _cache_set(query, output)
        print(output)
    except Exception as e:
        print(f"Search error: {e}")

if __name__ == "__main__":
    main()
'''

_FETCH_URL_SCRIPT = '''\
#!/usr/bin/env python3
"""Extract content from a URL via Parallel API. Usage: python3 fetch_url.py "url" """
import hashlib, json, os, sys
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratch", ".cache")

def _cache_get(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"url-{hashlib.md5(key.encode()).hexdigest()}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return None

def _cache_set(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"url-{hashlib.md5(key.encode()).hexdigest()}.txt")
    open(path, "w", encoding="utf-8").write(value)

def main():
    url = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not url:
        print("Usage: python3 fetch_url.py \\"https://example.com\\""); return
    cached = _cache_get(url)
    if cached:
        print("[cached]"); print(cached); return
    key = os.environ.get("PARALLEL_API_KEY", "")
    if key:
        try:
            resp = httpx.post(
                "https://api.parallel.ai/v1beta/extract",
                json={"urls": [url], "mode": "excerpt"},
                headers={"x-api-key": key, "Content-Type": "application/json"},
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                content = results[0].get("content") or "\\n\\n".join(results[0].get("excerpts", []))
                content = content[:12000]
                _cache_set(url, content)
                print(content); return
        except Exception:
            pass
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15.0)
        content = resp.text[:12000]
        _cache_set(url, content)
        print(content)
    except Exception as e:
        print(f"Fetch error: {e}")

if __name__ == "__main__":
    main()
'''


_QUERY_POSTS_SCRIPT = '''\
#!/usr/bin/env python3
"""Search 200K+ LinkedIn posts by keyword or creator, ranked by engagement.
Usage:
  python3 query_posts.py "voice AI emotional intelligence"
  python3 query_posts.py --creator "username"
"""
import json, os, sys
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

SELECT_COLS = ("hook,post_text,posted_at,creator_username,"
               "total_reactions,total_comments,total_reposts,"
               "engagement_score,is_outlier")

def fetch_by_creator(url, key, username):
    resp = httpx.get(
        f"{url}/rest/v1/linkedin_posts",
        params={
            "select": SELECT_COLS,
            "creator_username": f"eq.{username}",
            "is_company_post": "eq.false",
            "post_text": "not.is.null",
            "order": "engagement_score.desc",
            "limit": "20",
        },
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()

def fetch_by_keyword(url, key, query):
    keywords = [w.strip() for w in query.split() if len(w.strip()) >= 3]
    if not keywords:
        print("Query too short — use 3+ character words"); return []
    keywords.sort(key=len, reverse=True)
    primary = keywords[0]
    resp = httpx.get(
        f"{url}/rest/v1/linkedin_posts",
        params={
            "select": SELECT_COLS,
            "post_text": f"ilike.*{primary}*",
            "is_company_post": "eq.false",
            "order": "engagement_score.desc",
            "limit": "80",
        },
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    rows = resp.json()
    results = []
    for row in rows:
        text = (row.get("post_text") or "").lower()
        if all(kw.lower() in text for kw in keywords):
            results.append(row)
        if len(results) >= 15:
            break
    return results

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 query_posts.py \\"topic or keyword\\"")
        print("       python3 query_posts.py --creator \\"username\\"")
        return
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("SUPABASE_URL / SUPABASE_KEY not set"); return

    try:
        if args[0] == "--creator" and len(args) >= 2:
            username = args[1].strip().strip("@")
            results = fetch_by_creator(url, key, username)
            label = f"@{username}"
        else:
            query = " ".join(args).strip()
            results = fetch_by_keyword(url, key, query)
            label = query
    except Exception as e:
        print(f"Supabase query error: {e}"); return

    if not results:
        print(f"No posts found for: {label}")
        return

    print(f"Found {len(results)} top-engagement posts for: {label}\\n")
    for row in results:
        hook = (row.get("hook") or "").strip()
        text = (row.get("post_text") or "").strip()
        user = row.get("creator_username") or "unknown"
        date = (row.get("posted_at") or "")[:10]
        reactions = row.get("total_reactions") or 0
        comments = row.get("total_comments") or 0
        reposts = row.get("total_reposts") or 0
        eng = row.get("engagement_score") or 0
        eng_display = eng / 100 if eng else 0
        outlier = row.get("is_outlier") or False

        print(f"--- Post by @{user} ({date}) ---")
        metrics = f"Reactions: {reactions} | Comments: {comments} | Reposts: {reposts} | Engagement: {eng_display:.2f}"
        if outlier:
            metrics += " | OUTLIER"
        print(metrics)
        if hook:
            print(f"Hook: {hook}")
        print(text[:2000])
        print()

if __name__ == "__main__":
    main()
'''


_ORDINAL_ANALYTICS_SCRIPT = '''\
#!/usr/bin/env python3
"""Ordinal LinkedIn analytics: profiles, followers, posts, cadence.
Usage:
  python3 ordinal_analytics.py profiles
  python3 ordinal_analytics.py followers <profileId>
  python3 ordinal_analytics.py posts <profileId>
  python3 ordinal_analytics.py cadence <profileId>
"""
import json, os, sys
from datetime import datetime, timedelta
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

BASE = "https://app.tryordinal.com/api/v1"

def headers():
    key = os.environ.get("ORDINAL_API_KEY", "")
    if not key:
        print("ORDINAL_API_KEY not set — Ordinal analytics unavailable"); sys.exit(1)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def cmd_profiles():
    resp = httpx.get(f"{BASE}/profiles/scheduling", headers=headers(), timeout=15)
    resp.raise_for_status()
    profiles = resp.json()
    if isinstance(profiles, list):
        for p in profiles:
            pid = p.get("id", "")
            name = p.get("name") or p.get("displayName") or "unnamed"
            platform = p.get("platform", "")
            print(f"  {name} ({platform}): {pid}")
    else:
        print(json.dumps(profiles, indent=2))

def cmd_followers(profile_id):
    start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    resp = httpx.get(
        f"{BASE}/analytics/linkedin/{profile_id}/followers",
        headers=headers(), params={"startDate": start, "endDate": end}, timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))

def cmd_posts(profile_id):
    start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    resp = httpx.get(
        f"{BASE}/analytics/linkedin/{profile_id}/posts",
        headers=headers(), params={"startDate": start, "endDate": end}, timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))

def cmd_cadence(profile_id):
    start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    resp = httpx.get(
        f"{BASE}/analytics/linkedin/{profile_id}/posts",
        headers=headers(), params={"startDate": start, "endDate": end}, timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    dates = []
    posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
    for p in posts:
        d = p.get("publishedAt") or p.get("postedAt") or p.get("date") or ""
        if d:
            try:
                dates.append(datetime.fromisoformat(d.replace("Z", "+00:00")).date())
            except Exception:
                pass
    dates.sort(reverse=True)

    if not dates:
        print("No post dates found in Ordinal analytics."); return

    print(f"Last {min(10, len(dates))} posts: {', '.join(str(d) for d in dates[:10])}")

    if len(dates) >= 2:
        gaps = [(dates[i-1] - dates[i]).days for i in range(1, len(dates))]
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        max_idx = gaps.index(max_gap)
        print(f"Average gap: {avg_gap:.1f} days")
        print(f"Longest gap: {max_gap} days ({dates[max_idx+1]} to {dates[max_idx]})")

    days_since = (datetime.today().date() - dates[0]).days
    print(f"Days since last post: {days_since}")
    if days_since > (avg_gap * 1.5 if len(dates) >= 2 else 7):
        print("Recommended: Post soon — gap is above average.")
    else:
        print("Cadence looks healthy.")

def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1].lower()
    try:
        if cmd == "profiles":
            cmd_profiles()
        elif cmd == "followers" and len(sys.argv) >= 3:
            cmd_followers(sys.argv[2])
        elif cmd == "posts" and len(sys.argv) >= 3:
            cmd_posts(sys.argv[2])
        elif cmd == "cadence" and len(sys.argv) >= 3:
            cmd_cadence(sys.argv[2])
        else:
            print(__doc__)
    except Exception as e:
        print(f"Ordinal error: {e}")

if __name__ == "__main__":
    main()
'''


_SEMANTIC_SEARCH_SCRIPT = '''\
#!/usr/bin/env python3
"""Semantic search over 200K+ LinkedIn posts using vector similarity.
Usage: python3 semantic_search_posts.py "emotional intelligence in voice agents"
"""
import json, os, sys

def ensure_deps():
    for pkg in ("pinecone", "openai", "httpx"):
        try:
            __import__(pkg)
        except ImportError:
            import subprocess as _sp
            _sp.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_deps()

from openai import OpenAI
from pinecone import Pinecone
import httpx

INDEX_NAME = "linkedin-posts"
NAMESPACE = "v2"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 15

def embed_query(client, text, dimensions):
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL, dimensions=dimensions)
    return resp.data[0].embedding

def hydrate_from_supabase(urns, sb_url, sb_key):
    if not urns:
        return {}
    urn_filter = ",".join(f"\\"{u}\\"" for u in urns[:TOP_K])
    try:
        resp = httpx.get(
            f"{sb_url}/rest/v1/linkedin_posts",
            params={
                "select": "provider_urn,hook,post_text,posted_at,creator_username,"
                          "total_reactions,total_comments,total_reposts,"
                          "engagement_score,is_outlier",
                "provider_urn": f"in.({urn_filter})",
            },
            headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        return {r["provider_urn"]: r for r in rows if r.get("provider_urn")}
    except Exception as e:
        print(f"Supabase hydration error: {e}")
        return {}

def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python3 semantic_search_posts.py \\"topic or concept\\""); return

    oai_key = os.environ.get("OPENAI_API_KEY", "")
    pc_key = os.environ.get("PINECONE_API_KEY", "")
    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")

    if not oai_key or not pc_key:
        print("OPENAI_API_KEY / PINECONE_API_KEY not set"); return

    try:
        pc = Pinecone(api_key=pc_key)
        idx = pc.Index(INDEX_NAME)
        stats = idx.describe_index_stats()
        dims = stats.get("dimension", 1536)
    except Exception as e:
        print(f"Pinecone connection error: {e}"); return

    oai = OpenAI(api_key=oai_key)
    try:
        vec = embed_query(oai, query, dims)
    except Exception as e:
        print(f"Embedding error: {e}"); return

    try:
        results = idx.query(vector=vec, top_k=TOP_K, namespace=NAMESPACE, include_metadata=True)
    except Exception as e:
        print(f"Pinecone query error: {e}"); return

    matches = results.get("matches", [])
    if not matches:
        print(f"No semantically similar posts found for: {query}"); return

    urns = [m["id"] for m in matches]
    scores = {m["id"]: m.get("score", 0) for m in matches}

    if sb_url and sb_key:
        posts = hydrate_from_supabase(urns, sb_url, sb_key)
    else:
        posts = {}

    print(f"Found {len(matches)} semantically similar posts for: {query}\\n")

    for urn in urns:
        score = scores.get(urn, 0)
        row = posts.get(urn)
        if row:
            hook = (row.get("hook") or "").strip()
            text = (row.get("post_text") or "").strip()
            user = row.get("creator_username") or "unknown"
            date = (row.get("posted_at") or "")[:10]
            reactions = row.get("total_reactions") or 0
            comments = row.get("total_comments") or 0
            reposts = row.get("total_reposts") or 0
            eng = row.get("engagement_score") or 0
            eng_display = eng / 100 if eng else 0
            outlier = row.get("is_outlier") or False

            print(f"--- Post by @{user} ({date}) [similarity: {score:.3f}] ---")
            metrics = f"Reactions: {reactions} | Comments: {comments} | Reposts: {reposts} | Engagement: {eng_display:.2f}"
            if outlier:
                metrics += " | OUTLIER"
            print(metrics)
            if hook:
                print(f"Hook: {hook}")
            print(text[:2000])
        else:
            meta = matches[[m["id"] for m in matches].index(urn)].get("metadata", {})
            text = meta.get("post_text") or meta.get("text") or "(no text in metadata)"
            print(f"--- Post {urn} [similarity: {score:.3f}] ---")
            print(str(text)[:2000])
        print()

if __name__ == "__main__":
    main()
'''


_VALIDATE_DRAFT_SCRIPT = r'''#!/usr/bin/env python3
"""Validate a LinkedIn post draft — structural checks only.

Content quality (AI patterns, writing style) is handled by the constitutional
verifier post-generation, with learned principle weights. This tool only checks
structural issues that don't need an LLM: character count and deterministic
banned-phrase detection.

Usage: python3 validate_draft.py "Your full post text here"
       python3 validate_draft.py --file memory/draft-posts/my-draft.md
       python3 validate_draft.py --attempt 2 --file memory/draft-posts/my-draft.md

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

    # --- Character count (structural, always checked) ---
    if char_count > 3000:
        issues.append({"type": "char_count", "severity": "critical",
            "description": f"Post is {char_count} chars — exceeds 3000 char limit",
            "suggested_fix": f"Cut {char_count - 2800} characters"})
    elif char_count < 400:
        issues.append({"type": "char_count", "severity": "critical",
            "description": f"Post is only {char_count} chars — too short for engagement",
            "suggested_fix": "Add more substance: examples, details, data, or a story. Target 1300+"})
    elif char_count < 800:
        issues.append({"type": "char_count", "severity": "warning",
            "description": f"Post is only {char_count} chars — unusually short",
            "suggested_fix": f"Consider adding ~{800 - char_count} chars of substance"})

    # No content quality checks here — AI patterns, banned phrases, and
    # writing style are the constitutional verifier's domain. It runs
    # post-generation with learned principle weights that adapt to what
    # actually affects engagement for each client.

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


_IMAGE_SEARCH_SCRIPT = '''\
#!/usr/bin/env python3
"""Search for images via Serper Images API.
Usage: python3 image_search.py "query" [num_results]
Returns JSON array of image results with url, title, width, height, source.
"""
import json, os, sys
try:
    import requests
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if not query:
        print(json.dumps({"error": "No query provided"})); return

    base_url = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev/search")
    api_url = base_url.replace("/search", "/images")
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        print(json.dumps({"error": "SERPER_API_KEY not set"})); return

    try:
        resp = requests.post(
            api_url,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for img in data.get("images", []):
            results.append({
                "url": img.get("imageUrl", ""),
                "title": img.get("title", ""),
                "width": img.get("imageWidth", 0),
                "height": img.get("imageHeight", 0),
                "source": img.get("link", ""),
            })
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
'''


def _write_tool_scripts(workspace_root: Path) -> None:
    """Write helper scripts for web search, URL extraction, post search, analytics, validation, image search, and draft/edit/memory."""
    tools_dir = workspace_root / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "web_search.py").write_text(_WEB_SEARCH_SCRIPT, encoding="utf-8")
    (tools_dir / "fetch_url.py").write_text(_FETCH_URL_SCRIPT, encoding="utf-8")
    (tools_dir / "query_posts.py").write_text(_QUERY_POSTS_SCRIPT, encoding="utf-8")
    (tools_dir / "ordinal_analytics.py").write_text(_ORDINAL_ANALYTICS_SCRIPT, encoding="utf-8")
    (tools_dir / "semantic_search_posts.py").write_text(_SEMANTIC_SEARCH_SCRIPT, encoding="utf-8")
    (tools_dir / "validate_draft.py").write_text(_VALIDATE_DRAFT_SCRIPT, encoding="utf-8")
    (tools_dir / "image_search.py").write_text(_IMAGE_SEARCH_SCRIPT, encoding="utf-8")

    draft_sh = workspace_root / "draft.sh"
    draft_sh.write_text(_DRAFT_SH_SCRIPT, encoding="utf-8")
    draft_sh.chmod(0o755)

    edit_sh = workspace_root / "edit.sh"
    edit_sh.write_text(_EDIT_SH_SCRIPT, encoding="utf-8")
    edit_sh.chmod(0o755)

    memory_sh = workspace_root / "memory.sh"
    memory_sh.write_text(_MEMORY_SH_SCRIPT, encoding="utf-8")
    memory_sh.chmod(0o755)


def _write_agents_md(workspace_root: Path, company_keyword: str) -> None:
    """Write AGENTS.md to workspace root for Pi to discover."""
    directives = _build_dynamic_directives(company_keyword)
    overrides = _build_overrides(company_keyword)
    content = _PI_AGENTS_TEMPLATE.format(dynamic_directives=directives)
    content = _apply_overrides_to_prompt(content, overrides)
    (workspace_root / "AGENTS.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pi-based agentic loop (primary — matches Jacquard architecture)
# ---------------------------------------------------------------------------

def _run_pi_agent(
    workspace_root: Path,
    user_prompt: str,
    company_keyword: str,
    event_callback: Any = None,
    model: str = "claude-opus-4-6",
) -> tuple[dict | None, list[dict]]:
    """Run the ghostwriter via Pi CLI with automatic context compaction."""
    session_log: list[dict[str, Any]] = []
    session_start = time.time()

    _write_agents_md(workspace_root, company_keyword)
    _write_tool_scripts(workspace_root)

    session_dir = P.memory_dir(company_keyword) / ".pi-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    has_sessions = any(session_dir.glob("*.jsonl"))

    provider = "anthropic"
    if "/" in model:
        provider, model = model.split("/", 1)

    pi_cmd = [
        "pi",
        "--mode", "json",
        "-p",
        "--provider", provider,
        "--model", model,
        "--thinking", "high",
        "--session-dir", str(session_dir),
        "--tools", "read,bash,edit,write,grep,find,ls",
    ]
    if has_sessions:
        pi_cmd.append("--continue")
    pi_cmd.append(user_prompt)

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    env["PARALLEL_API_KEY"] = PARALLEL_API_KEY
    env["SUPABASE_URL"] = SUPABASE_URL
    env["SUPABASE_KEY"] = SUPABASE_KEY
    env["APIMAESTRO_API_KEY"] = APIMAESTRO_KEY
    env["APIMAESTRO_HOST"] = APIMAESTRO_HOST
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
    env["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
    env["SERPER_BASE_URL"] = os.getenv("SERPER_BASE_URL", "https://google.serper.dev/search")
    ordinal_key = _get_ordinal_api_key(company_keyword)
    if ordinal_key:
        env["ORDINAL_API_KEY"] = ordinal_key

    session_log.append({
        "type": "session_start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": "pi",
        "pi_version": "0.63.1",
        "has_prior_session": has_sessions,
        "workspace": str(workspace_root),
    })

    pi_timeout = 900  # 15 minutes max for a full generation run

    logger.info("[Stelle/Pi] Starting Pi agent (session_dir=%s, continue=%s)...", session_dir, has_sessions)
    print(f"[Stelle] Running Pi agent for {company_keyword} (timeout={pi_timeout}s)...")

    jsonl_out = workspace_root / "output" / "pi_events.jsonl"

    events_seen = 0
    compaction_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    all_lines: list[str] = []
    exit_code = -1

    def _process_event(event: dict) -> None:
        nonlocal events_seen, compaction_count, total_input_tokens, total_output_tokens, total_cost
        events_seen += 1
        etype = event.get("type", "")

        if etype == "message_update":
            ae = event.get("assistantMessageEvent", {})
            ae_type = ae.get("type", "")
            msg = event.get("message", ae.get("message", {}))

            if ae_type == "text_delta":
                delta = ae.get("textDelta", "")
                if delta and event_callback:
                    event_callback("text_delta", {"text": delta})

            elif ae_type == "thinking_delta":
                delta = ae.get("thinkingDelta", "")
                if delta and event_callback:
                    event_callback("thinking", {"text": delta})

            elif ae_type.startswith("toolcall"):
                for block in msg.get("content", []):
                    if block.get("type") == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments", {})
                        summary = ""
                        if isinstance(args, dict):
                            summary = args.get("path", args.get("command", str(args)))[:80]
                        logger.info("[Stelle/Pi] tool: %s(%s)", name, summary)
                        print(f"[Stelle/Pi] tool: {name}({summary})")
                        if event_callback:
                            event_callback("tool_call", {"name": name, "arguments": summary})

            usage = msg.get("usage", {})
            if usage:
                total_input_tokens = max(total_input_tokens, usage.get("input", 0))
                total_output_tokens += usage.get("output", 0)
                cost = usage.get("cost", {})
                if isinstance(cost, dict):
                    total_cost = max(total_cost, cost.get("total", 0))

        elif etype == "tool_result":
            result_text = str(event.get("result", ""))[:500]
            if event_callback:
                event_callback("tool_result", {"name": event.get("tool", ""), "result": result_text, "is_error": False})

        elif etype == "turn_end":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            if usage:
                cost_info = usage.get("cost", {})
                in_tok = usage.get("input", 0)
                out_tok = usage.get("output", 0)
                cache_tok = usage.get("cacheRead", 0)
                cost_val = cost_info.get("total", 0) if isinstance(cost_info, dict) else 0
                logger.info(
                    "[Stelle/Pi] turn end — in=%d out=%d cache_read=%d cost=$%.4f",
                    in_tok, out_tok, cache_tok, cost_val,
                )
                print(f"[Stelle/Pi] turn end — in={in_tok} out={out_tok} cache={cache_tok} cost=${cost_val:.4f}")
                if event_callback:
                    event_callback("status", {"message": f"Turn complete — in={in_tok} out={out_tok} cache={cache_tok} cost=${cost_val:.4f}"})

        elif etype == "auto_compaction_start":
            compaction_count += 1
            logger.info("[Stelle/Pi] Context compaction #%d", compaction_count)
            print(f"[Stelle/Pi] Context compaction #{compaction_count}")
            if event_callback:
                event_callback("compaction", {"message": f"Context compaction #{compaction_count}"})

        elif etype == "auto_retry_start":
            logger.info("[Stelle/Pi] Retry %s/%s...", event.get("attempt", "?"), event.get("maxAttempts", "?"))
            print(f"[Stelle/Pi] Retry {event.get('attempt', '?')}/{event.get('maxAttempts', '?')}...")
            if event_callback:
                event_callback("status", {"message": f"Retry {event.get('attempt', '?')}/{event.get('maxAttempts', '?')}..."})

        elif etype == "error":
            err_msg = event.get("message", str(event))[:300]
            logger.error("[Stelle/Pi] Error: %s", err_msg)
            print(f"[Stelle/Pi] ERROR: {err_msg}")
            if event_callback:
                event_callback("error", {"message": err_msg})

        session_log.append({
            "type": "pi_event",
            "event_type": etype,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": str(event)[:500],
        })

    try:
        proc = subprocess.Popen(
            pi_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(workspace_root),
            env=env,
            text=True,
            bufsize=1,
        )

        import threading

        stderr_chunks: list[str] = []

        def _read_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        file_poll_stop = threading.Event()

        def _poll_files():
            known: set[str] = set()
            for p in workspace_root.rglob("*"):
                if p.is_file():
                    known.add(str(p))
            while not file_poll_stop.is_set():
                file_poll_stop.wait(3.0)
                if file_poll_stop.is_set():
                    break
                current: set[str] = set()
                for p in workspace_root.rglob("*"):
                    if p.is_file():
                        current.add(str(p))
                new_files = current - known
                if new_files and event_callback:
                    relative = [str(Path(f).relative_to(workspace_root)) for f in sorted(new_files)]
                    event_callback("status", {"message": f"Files changed: {', '.join(relative[:10])}"})
                known = current

        file_poll_thread = threading.Thread(target=_poll_files, daemon=True)
        file_poll_thread.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            all_lines.append(line)
            try:
                event = json.loads(line)
                _process_event(event)
            except json.JSONDecodeError:
                logger.debug("[Stelle/Pi] Non-JSON line: %s", line[:200])

        proc.wait(timeout=30)
        exit_code = proc.returncode
        file_poll_stop.set()
        file_poll_thread.join(timeout=2)
        stderr_thread.join(timeout=5)
        stderr_output = "".join(stderr_chunks)

    except subprocess.TimeoutExpired:
        logger.error("[Stelle/Pi] Pi timed out after %ds", pi_timeout)
        print(f"[Stelle] Pi timed out after {pi_timeout}s")
        session_log.append({"type": "timeout", "timeout_seconds": pi_timeout})
        file_poll_stop.set()
        if proc:
            proc.kill()
        stderr_output = ""
    except FileNotFoundError:
        logger.error("[Stelle/Pi] Pi CLI not found — is it installed?")
        return None, session_log

    if all_lines:
        try:
            jsonl_out.write_text("\n".join(all_lines), encoding="utf-8")
        except Exception:
            pass

    total_elapsed = time.time() - session_start

    session_log.append({
        "type": "session_end",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exit_code": exit_code,
        "events_seen": events_seen,
        "compaction_count": compaction_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "total_elapsed_seconds": round(total_elapsed, 1),
    })

    logger.info(
        "[Stelle/Pi] Pi finished: exit=%d, events=%d, compactions=%d, "
        "in=%d out=%d cost=$%.4f elapsed=%.1fs",
        exit_code, events_seen, compaction_count,
        total_input_tokens, total_output_tokens, total_cost, total_elapsed,
    )
    print(
        f"[Stelle/Pi] Done: {events_seen} events, {compaction_count} compactions, "
        f"cost=${total_cost:.4f}, elapsed={total_elapsed:.0f}s"
    )

    if exit_code != 0:
        logger.error("[Stelle/Pi] Pi exited with code %d. stderr: %s", exit_code, stderr_output[:500])

    result = _extract_pi_result(workspace_root)

    if result is None:
        logger.warning("[Stelle/Pi] No result.json found — trying to extract from scratch/drafts/")
        result = _extract_result_from_scratch(workspace_root)

    return result, session_log


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
    """Fallback: reconstruct result JSON from draft files in memory/draft-posts/ or scratch/drafts/."""
    search_dirs = [
        workspace_root / "memory" / "draft-posts",
        workspace_root / "scratch" / "drafts",
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


def _run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    workspace_root: Path,
) -> tuple[dict | None, list[dict]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]
    session_log: list[dict[str, Any]] = []
    session_start = time.time()

    session_log.append({
        "type": "session_start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_prompt_chars": len(system_prompt),
        "user_prompt": user_prompt,
    })

    result_json: str | None = None
    turn = 0

    while turn < MAX_AGENT_TURNS:
        turn += 1
        t0 = time.time()
        logger.info("[Stelle] Turn %d — calling Claude...", turn)

        try:
            with _client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=128000,
                thinking={"type": "adaptive"},
                system=system_prompt,
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
            if turn < 3:
                time.sleep(5)
                continue
            raise

        elapsed = time.time() - t0

        usage_raw = response.usage
        usage = {
            "input_tokens": getattr(usage_raw, "input_tokens", 0),
            "output_tokens": getattr(usage_raw, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage_raw, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(usage_raw, "cache_read_input_tokens", 0),
        }

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

        logger.info(
            "[Stelle] Turn %d done in %.1fs — stop_reason=%s, blocks=%d, in=%d out=%d",
            turn, elapsed, response.stop_reason, len(response.content),
            usage["input_tokens"], usage["output_tokens"],
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    logger.info("[Stelle] Agent finished with text response on turn %d", turn)
            break

        tool_results = []
        tool_result_log = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            args = block.input if isinstance(block.input, dict) else {}
            logger.info("[Stelle]   tool: %s(%s)", name, _summarize_args(args))

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

                result_json = raw_json
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
            elif name in _TOOL_HANDLERS:
                try:
                    output = _TOOL_HANDLERS[name](workspace_root, args)
                    is_error = False
                except Exception as e:
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

    total_elapsed = time.time() - session_start
    total_input = sum(e.get("usage", {}).get("input_tokens", 0) for e in session_log if e.get("type") == "turn")
    total_output = sum(e.get("usage", {}).get("output_tokens", 0) for e in session_log if e.get("type") == "turn")

    session_log.append({
        "type": "session_end",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_turns": turn,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "has_result": result_json is not None,
    })

    logger.info(
        "[Stelle] Session complete: %d turns, %.1fs, %d in / %d out tokens",
        turn, total_elapsed, total_input, total_output,
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

    try:
        resp = _call_with_retry(lambda: _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
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
                ),
            }],
        ))
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("[Stelle] Why-post generation failed: %s", e)
        return ""


def _generate_image_suggestion(post_text: str, hook: str) -> str:
    """Generate a simple, easy-to-produce image suggestion for the post."""
    try:
        resp = _call_with_retry(lambda: _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Post hook: {hook}\n\n"
                    f"Post:\n{post_text}\n\n"
                    f"Suggest ONE simple image a single graphic designer could make in "
                    f"under 30 minutes. Think: a clean quote card, a bold stat highlight, "
                    f"a minimal photo with a text overlay, or a simple before/after. "
                    f"No intricate infographics, multi-panel illustrations, or complex "
                    f"diagrams. Keep it to one sentence describing the visual and one "
                    f"sentence describing any text on it. If the post works better as "
                    f"text-only, just say 'Text-only'. No preamble."
                ),
            }],
        ))
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("[Stelle] Image suggestion generation failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Result processing + fact-check
# ---------------------------------------------------------------------------

def _compute_cv_thresholds(company: str) -> dict:
    """Compute data-driven constitutional verification gating thresholds.

    Returns {skip, ensemble_upper, ensemble_lower}. Falls back to
    hard-coded defaults (4.0, 3.8, 3.0) when insufficient data.
    """
    defaults = {"skip": 4.0, "ensemble_upper": 3.8, "ensemble_lower": 3.0}

    # Check cache
    cache_path = P.memory_dir(company) / "cv_thresholds.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            computed = cached.get("computed_at", "")
            if computed:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(computed.replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                if age_h < 24 and cached.get("observation_count", 0) >= 15:
                    return {k: cached[k] for k in ("skip", "ensemble_upper", "ensemble_lower") if k in cached}
        except Exception:
            pass

    # Need observations with both cyrene_composite and engagement reward
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        obs_with_scores = [
            o for o in rm._state.get("observations", [])
            if o.get("status") == "scored"
            and o.get("cyrene_composite") is not None
            and o.get("reward", {}).get("immediate") is not None
        ]
    except Exception:
        return defaults

    if len(obs_with_scores) < 15:
        return defaults

    perm_scores = [o["cyrene_composite"] for o in obs_with_scores]
    rewards = [o["reward"]["immediate"] for o in obs_with_scores]
    median_reward = sorted(rewards)[len(rewards) // 2]

    # Skip threshold: 75th percentile of Cyrene scores for above-median-engagement posts
    good_perms = sorted([p for p, r in zip(perm_scores, rewards) if r > median_reward])
    skip = good_perms[int(len(good_perms) * 0.75)] if good_perms else 4.0

    # Ensemble upper: median Cyrene score across all posts
    all_sorted = sorted(perm_scores)
    ensemble_upper = all_sorted[len(all_sorted) // 2]

    # Ensemble lower: 25th percentile
    ensemble_lower = all_sorted[int(len(all_sorted) * 0.25)]

    # Sanity: ensure lower < upper < skip
    ensemble_lower = min(ensemble_lower, ensemble_upper - 0.1)
    skip = max(skip, ensemble_upper + 0.1)

    result = {
        "skip": round(skip, 2),
        "ensemble_upper": round(ensemble_upper, 2),
        "ensemble_lower": round(ensemble_lower, 2),
        "observation_count": len(obs_with_scores),
        "computed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    # Cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        tmp.rename(cache_path)
    except Exception:
        pass

    return result


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

        # Save original draft before Cyrene revision
        pre_revision_text = text

        output_lines.append("### Original Draft\n")
        output_lines.append(text + "\n")

        # -----------------------------------------------------------
        # SELF-REFINE: Critique-revise loop (Cyrene)
        # Runs BEFORE fact-checking so Castorice gets the best version.
        # -----------------------------------------------------------
        print(f"[Stelle] SELF-REFINE post {i}/{len(posts)}: {hook[:50]}...")
        refine_report_lines: list[str] = []
        _refine_result = None
        try:
            from backend.src.agents.cyrene import refine_post as _refine
            _refine_result = _refine(
                company=company_keyword,
                draft_text=text,
                transcript_excerpt=origin[:1000] if origin else "",
                max_iterations=3,
            )
            if _refine_result.final_text != text:
                text = _refine_result.final_text
                refine_report_lines.append(f"**SELF-REFINE:** {_refine_result.total_iterations} iterations, "
                                           f"best score {_refine_result.best_score}/5.0 ({_refine_result.method})")
                for it in _refine_result.iterations:
                    weak = it.get("weak_dimensions", [])
                    if weak:
                        weak_str = ", ".join(f"{d['name']}={d['score']}" for d in weak)
                        refine_report_lines.append(f"  Iter {it['iteration']}: weak=[{weak_str}]")
            else:
                refine_report_lines.append(f"**SELF-REFINE:** Passed on first critique "
                                           f"(score {_refine_result.best_score}/5.0)")
        except Exception as _refine_err:
            logger.debug("[Stelle] SELF-REFINE skipped for post %d: %s", i, _refine_err)

        if refine_report_lines:
            output_lines.append("### Quality Refinement\n")
            output_lines.extend(refine_report_lines)
            output_lines.append("")

        # -----------------------------------------------------------
        # Fact-check (Castorice) — runs on the refined version
        # -----------------------------------------------------------
        print(f"[Stelle] Fact-checking post {i}/{len(posts)}: {hook[:50]}...")
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

        # -----------------------------------------------------------
        # Constitutional Verifier — gated to borderline Cyrene scores only.
        # Full ensemble (3 models) on ambiguous posts; single-model on others.
        # Skipped entirely when SELF-REFINE scored ≥4.0 (high confidence).
        # -----------------------------------------------------------
        _cyrene_score = _refine_result.best_score if _refine_result is not None else 0.0
        _cv_thresholds = _compute_cv_thresholds(company_keyword)
        _cv_borderline = _cv_thresholds["ensemble_lower"] <= _cyrene_score < _cv_thresholds["ensemble_upper"]
        _cv_low = _cyrene_score < _cv_thresholds["ensemble_lower"] and _cyrene_score > 0
        if _cv_borderline or _cv_low:
            _cv_models = ["claude", "gemini", "gpt"] if _cv_borderline else ["claude"]
            print(f"[Stelle] Constitutional verification post {i}/{len(posts)} "
                  f"(Cyrene {_cyrene_score:.1f}, {'ensemble' if _cv_borderline else 'single'})...")
            try:
                from backend.src.utils.constitutional_verifier import verify_post as _verify
                _cv_result = _verify(corrected, company=company_keyword, models=_cv_models)
                _cv_score = _cv_result.get("constitutional_score")
                _cv_mode = _cv_result.get("mode", "binary")
                if not _cv_result.get("passed"):
                    violations = _cv_result.get("violations", [])
                    score_str = f" score={_cv_score:.2f}" if _cv_score is not None else ""
                    output_lines.append(f"### Constitutional Verification: FAIL ({_cv_mode}{score_str})\n")
                    for v in violations:
                        conf = v.get("confidence")
                        weight = v.get("weight")
                        if conf is not None and weight is not None:
                            output_lines.append(f"- **{v['name']}** (confidence: {conf:.0%}, weight: {weight:.2f})")
                        else:
                            notes = " | ".join(v.get("notes", []))
                            output_lines.append(f"- **{v['name']}** ({v.get('votes_fail', 0)}/{v.get('votes_pass', 0) + v.get('votes_fail', 0)} models failed) {notes}")
                        for note in v.get("notes", []):
                            output_lines.append(f"  {note}")
                    output_lines.append(f"\nModel agreement: {_cv_result.get('model_agreement', 0):.0%}\n")
                else:
                    score_str = f" score={_cv_score:.2f}" if _cv_score is not None else ""
                    output_lines.append(f"### Constitutional Verification: PASS ({_cv_mode}{score_str}, {_cv_result.get('model_agreement', 0):.0%} agreement)\n")
            except Exception as _cv_err:
                logger.debug("[Stelle] Constitutional verification skipped for post %d: %s", i, _cv_err)
        elif _cyrene_score >= _cv_thresholds["skip"]:
            output_lines.append(f"### Constitutional Verification: SKIPPED (Cyrene score {_cyrene_score:.1f} — high confidence)\n")

        print(f"[Stelle] Generating why-post + image for post {i}/{len(posts)}...")
        why_post = _generate_why_post(corrected, origin, client_name, company_keyword)
        if why_post:
            output_lines.append(f"### Why Post\n\n{why_post}\n")

        img_sug = _generate_image_suggestion(corrected, hook)
        if img_sug:
            output_lines.append(f"### Image Suggestion\n\n{img_sug}\n")
        elif image_suggestion:
            output_lines.append(f"### Image Suggestion\n\n{image_suggestion}\n")

        print(f"[Stelle] Validating draft {i}/{len(posts)}...")
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

            # Persist Cyrene scores and constitutional results on the observation.
            # Persist Cyrene scores and constitutional results on the observation.
            # existing adaptive config code and readiness checks.
            _extra_fields = {}
            if _refine_result is not None and _refine_result.iterations:
                _last_iter = _refine_result.iterations[-1]
                _dim_scores = _last_iter.get("all_dimensions", {})
                if not _dim_scores:
                    _dim_scores = {d["name"]: d["score"] for d in _last_iter.get("weak_dimensions", [])}
                if _dim_scores:
                    _extra_fields["cyrene_dimensions"] = _dim_scores
                _extra_fields["cyrene_composite"] = _refine_result.best_score
                _extra_fields["cyrene_iterations"] = _refine_result.total_iterations
                # Per-iteration composite scores — enables the adaptive config
                # to learn the marginal value of each additional revision cycle
                # and set the iteration ceiling per-client.
                _extra_fields["cyrene_iteration_scores"] = [
                    it.get("composite_score", 0) for it in _refine_result.iterations
                ]
                _extra_fields["cyrene_weights_tier"] = _refine_result.adaptive_tier
                _extra_fields["cyrene_dimension_set"] = _refine_result.dimension_set

            # Constitutional results — _cv_result is only set when verifier ran
            try:
                if _cv_result and isinstance(_cv_result, dict):
                    _cv_principles = {}
                    for _pr in _cv_result.get("principles", []):
                        if _pr.get("id"):
                            _cv_principles[_pr["id"]] = _pr.get("passed", True)
                    if _cv_principles:
                        _extra_fields["constitutional_results"] = _cv_principles
            except NameError:
                pass  # _cv_result not defined — verifier was skipped

            # Alignment score — compute and store for threshold learning
            try:
                from backend.src.utils.alignment_scorer import score_draft_alignment as _score_align
                _align_result = _score_align(company_keyword, corrected)
                if _align_result and _align_result.get("score") is not None:
                    _extra_fields["alignment_score"] = _align_result["score"]
            except Exception:
                pass

            # Active learned directives — stamp which directives were in the
            # system prompt at generation time so compute_directive_efficacy()
            # can retrospectively classify them as validated / neutral /
            # counterproductive based on the resulting engagement.
            try:
                from backend.src.utils.feedback_distiller import get_active_directive_ids
                _active_ids = get_active_directive_ids(company_keyword)
                # Always set the field (even to empty list) so attribution can
                # distinguish "directive was not active" from "observation
                # created before tracking shipped".
                _extra_fields["active_directives"] = _active_ids
            except Exception:
                pass

            # Strategy brief tracking — stamp the brief version active at
            # generation time. The strategy brief is injected into the user
            # prompt via build_stelle_strategy_context (the compact builder);
            # if a brief exists, it was seen by the LLM. strategy_tracker uses
            # this field to compare data-informed vs uninformed post performance.
            try:
                from backend.src.utils.strategy_brief import get_brief_version
                _brief_version = get_brief_version(company_keyword)
                if _brief_version:
                    _extra_fields["strategy_brief_version"] = _brief_version
            except Exception:
                pass

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
                # Store both pre-revision and post-revision content
                _pre_rev = pre_revision_text if pre_revision_text != corrected else None
                _perm_score = _refine_result.best_score if _refine_result is not None else None
                # Build generation metadata for draft_map persistence
                _gen_meta = dict(_extra_fields) if _extra_fields else {}
                _save_post(
                    post_id=_draft_id,
                    company=company_keyword,
                    content=corrected,
                    title=hook[:200] if hook else None,
                    status="draft",
                    why_post=why_post or None,
                    citation_comments=citation_comments,
                    pre_revision_content=_pre_rev,
                    cyrene_score=_perm_score,
                    generation_metadata=_gen_meta if _gen_meta else None,
                )
            except Exception as _e:
                logger.warning("[Stelle] Could not save post %d to local SQLite: %s", i, _e)

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
    print(f"[Stelle] Starting agentic ghostwriter for {client_name}...")

    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        raise FileNotFoundError(
            f"Missing memory/{company_keyword}/linkedin_username.txt — "
            f"create this file with the client's LinkedIn username "
            f"(the part after linkedin.com/in/) before running the pipeline."
        )

    P.ensure_dirs(company_keyword)

    print("[Stelle] Setting up workspace...")
    workspace_root = _setup_workspace(company_keyword)

    # RuanMei: generate performance insight context (soft, non-prescriptive).
    # This is additive context that Stelle can use or ignore. If RuanMei
    # has insufficient data (< 5 scored posts), this is silently empty.
    ruan_mei_insight_context = ""
    try:
        import asyncio
        from backend.src.agents.ruan_mei import RuanMei
        _rm = RuanMei(company_keyword)
        if _rm.scored_count() >= 5:
            try:
                _loop = asyncio.get_running_loop()
            except RuntimeError:
                _loop = None
            if _loop and _loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as _pool_exec:
                    _insights = _pool_exec.submit(
                        lambda: asyncio.run(_rm.generate_insights())
                    ).result(timeout=30)
            else:
                _insights = asyncio.run(_rm.generate_insights())
            if _insights:
                ruan_mei_insight_context = (
                    "\n\nPERFORMANCE INSIGHTS (from engagement data):\n"
                    "The following patterns have been observed from this client's "
                    "post performance history. Use your judgment on whether these "
                    "apply to the current source material. These are observations, "
                    "not directives.\n\n"
                    + _insights
                )
        else:
            _cross = RuanMei.generate_cross_client_insights()
            if _cross:
                ruan_mei_insight_context = (
                    "\n\nCROSS-CLIENT INSIGHTS (this client has limited history — "
                    "these patterns are from other clients' top-performing posts, anonymized):\n"
                    "Use your judgment on which apply. These are general observations.\n\n"
                    + _cross
                )
    except Exception as _e:
        logger.debug("[Stelle] RuanMei insight generation skipped: %s", _e)

    # LOLA: topic/format bandit recommendations.
    lola_context = ""
    try:
        from backend.src.agents.lola import LOLA
        _lola = LOLA(company_keyword)
        # Auto-seed arms: cross-client auto-seed is primary path.
        # topic_arms.json is manual override only.
        if not _lola._state.arms:
            _arms_path = P.memory_dir(company_keyword) / "topic_arms.json"
            if _arms_path.exists():
                # Manual override takes precedence
                import json as _json
                _seed_arms = _json.loads(_arms_path.read_text(encoding="utf-8"))
                _lola.seed_arms(_seed_arms)
            else:
                # Primary: auto-seed from universal patterns + client ICP
                try:
                    from backend.src.services.cross_client_learning import auto_seed_lola as _auto_seed
                    _auto_seed(company_keyword)
                except Exception as _seed_err:
                    logger.debug("[Stelle] LOLA auto-seed skipped: %s", _seed_err)

        # Auto-plan series from hot LOLA arms (no human gate)
        try:
            from backend.src.services.market_intelligence import auto_plan_series_from_lola as _auto_series
            _auto_series(company_keyword)
        except Exception as _series_err:
            logger.debug("[Stelle] Auto series plan skipped: %s", _series_err)
        lola_context = _lola.recommend_context()
    except Exception as _e:
        logger.debug("[Stelle] LOLA context skipped: %s", _e)

    # Feedback learning is handled entirely through RuanMei observations:
    # draft (post_body) → client edits → final (posted_body) → engagement (reward).
    # No separate feedback ingestion pipeline needed.

    # Market intelligence: trending topics, hook shifts, whitespace from vertical monitoring.
    market_context = ""
    try:
        from backend.src.services.market_intelligence import build_market_context as _mktctx
        market_context = _mktctx(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Market intelligence skipped: %s", _e)

    # Cross-client hook library: top-performing hooks as reference exemplars.
    hook_library_context = ""
    try:
        from backend.src.services.cross_client_learning import load_hook_library_for_stelle as _hooks
        hook_library_context = _hooks(company=company_keyword, limit=10)
    except Exception as _e:
        logger.debug("[Stelle] Hook library skipped: %s", _e)

    # Series Engine: inject series context if a series post is due.
    series_context = ""
    try:
        from backend.src.services.series_engine import get_stelle_series_context as _series_ctx
        series_context = _series_ctx(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Series context skipped: %s", _e)

    # Temporal Orchestrator: scheduling intelligence for generation context.
    scheduling_context = ""
    try:
        from backend.src.services.temporal_orchestrator import build_scheduling_context as _sched_ctx
        scheduling_context = _sched_ctx(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Scheduling context skipped: %s", _e)

    # Analyst findings: hypothesis-driven engagement analysis from the analyst
    # agent. Replaces the old fixed strategy brief with findings the agent
    # discovered by forming and testing its own hypotheses using statistical
    # tools + LinkedIn-wide data. Injected as raw data with caveats — the
    # LLM reads the findings and decides how to apply them.
    analyst_context = ""
    try:
        import json as _json_analyst_ctx
        _findings_path = P.memory_dir(company_keyword) / "analyst_findings.json"
        if _findings_path.exists():
            _af = _json_analyst_ctx.loads(_findings_path.read_text(encoding="utf-8"))
            _findings = _af.get("findings", [])
            if _findings:
                _lines = [
                    "",
                    "",
                    "ENGAGEMENT ANALYSIS (from hypothesis-driven analyst agent):",
                    "The following findings were discovered by testing statistical hypotheses "
                    "against this client's engagement history and 200K+ LinkedIn posts. "
                    "They are data-driven observations, not directives. Use them to inform "
                    "topic selection and content approach when multiple valid angles exist. "
                    "Respect the confidence levels — 'suggestive' means directional, not proven.",
                    "",
                ]
                for _f in _findings:
                    _conf = _f.get("confidence", "suggestive")
                    _claim = _f.get("claim", "")
                    _evidence = _f.get("evidence", "")
                    _lines.append(f"[{_conf.upper()}] {_claim}")
                    if _evidence:
                        _lines.append(f"  Evidence: {_evidence[:200]}")
                    _lines.append("")
                analyst_context = "\n".join(_lines)
    except Exception as _e:
        logger.debug("[Stelle] Analyst context skipped: %s", _e)

    # 360Brew alignment scorer: pre-generation semantic consistency check.
    alignment_context = ""
    try:
        from backend.src.utils.alignment_scorer import build_stelle_context as _align
        # Sample source material from transcripts for alignment check.
        _transcripts_dir = P.transcripts_dir(company_keyword)
        _source_sample = ""
        if _transcripts_dir.exists():
            for _tf in sorted(_transcripts_dir.iterdir()):
                if _tf.suffix in (".txt", ".md") and _tf.stat().st_size < 30_000:
                    try:
                        _source_sample += _tf.read_text(encoding="utf-8")[:1500] + "\n\n"
                        if len(_source_sample) > 3000:
                            break
                    except Exception:
                        pass
        if _source_sample:
            alignment_context = _align(company_keyword, _source_sample, client_name)
    except Exception as _e:
        logger.debug("[Stelle] Alignment scoring skipped: %s", _e)

    user_prompt = prompt or (
        f"Write up to {num_posts} LinkedIn posts for {client_name}. "
        f"The transcripts are from content interviews — conversations designed "
        f"to surface post material. Mine them for everything worth writing about. "
        f"Only write as many posts as the transcripts can genuinely support with "
        f"distinct insights — if the material supports 7, write 7, not {num_posts}. "
        f"Quality and distinctness over quantity."
    )
    if ruan_mei_insight_context:
        user_prompt += ruan_mei_insight_context
    if lola_context:
        user_prompt += lola_context
    if alignment_context:
        user_prompt += alignment_context

    if scheduling_context:
        user_prompt += scheduling_context
    if series_context:
        user_prompt += series_context
    if market_context:
        user_prompt += market_context
    if hook_library_context:
        user_prompt += hook_library_context
    if analyst_context:
        user_prompt += analyst_context

    if _PI_AVAILABLE:
        print(f"[Stelle] Using Pi agent (context compaction enabled)...")
        result, session_log = _run_pi_agent(
            workspace_root, user_prompt, company_keyword,
            event_callback=event_callback, model=model,
        )
    else:
        logger.warning("[Stelle] Pi not installed — falling back to direct API loop (higher token usage)")
        print(f"[Stelle] Pi not found. Using direct API loop (max {MAX_AGENT_TURNS} turns)...")
        directives = _build_dynamic_directives(company_keyword)
        overrides = _build_overrides(company_keyword)
        system_prompt = _DIRECT_SYSTEM_TEMPLATE.format(dynamic_directives=directives)
        system_prompt = _apply_overrides_to_prompt(system_prompt, overrides)
        result, session_log = _run_agent_loop(system_prompt, user_prompt, workspace_root)

    session_path = output_filepath.replace(".md", "_session.jsonl")
    Path(session_path).parent.mkdir(parents=True, exist_ok=True)
    with open(session_path, "w", encoding="utf-8") as f:
        for entry in session_log:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"[Stelle] Session log saved to {session_path}")

    if result is None:
        print("[Stelle] Agent did not produce a result. Writing empty output.")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"# {client_name} — One-Shot Posts\n\nAgent failed to produce output.\n")
        return output_filepath

    passed, val_errors, val_warnings = _validate_output(result)
    if not passed:
        logger.warning("[Stelle] Final output validation failed: %s", val_errors)

    post_count = len(result.get("posts", []))
    print(f"[Stelle] Agent produced {post_count} posts. Running fact-check...")

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
        print("[Stelle] Inline edit via Pi --continue...")
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
                print(f"[Stelle] Inline edit complete ({len(output)} chars)")
                return output

            logger.warning("[Stelle] Pi inline edit produced no usable output")
        except subprocess.TimeoutExpired:
            logger.warning("[Stelle] Pi inline edit timed out after 120s")
            if proc:
                proc.kill()
        except Exception as e:
            logger.warning("[Stelle] Pi inline edit failed: %s", e)

    print("[Stelle] Inline edit via direct Claude Haiku call...")
    try:
        client = Anthropic()
        resp = _call_with_retry(lambda: client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": edit_prompt}],
        ))
        text = resp.content[0].text.strip() if resp.content else None
        if text and len(text) > 100:
            print(f"[Stelle] Inline edit complete ({len(text)} chars)")
            if event_callback:
                event_callback("text_delta", {"text": text})
            return text
    except Exception as e:
        logger.error("[Stelle] Direct inline edit failed: %s", e)

    return None
