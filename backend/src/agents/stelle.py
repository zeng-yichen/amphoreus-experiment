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

MAX_AGENT_TURNS = 100
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
- `memory/draft-posts/` — **Authoritative pushed-but-not-yet-live drafts.** \
Pre-populated at workspace setup from Ordinal's /posts endpoint, filtered \
to non-Posted statuses. These are drafts that have already been committed \
to Ordinal and will publish on LinkedIn imminently. READ ONLY — do not \
write to this directory. These count for dedup; your own in-run drafts \
do NOT count for dedup until they are pushed to Ordinal by the publisher \
after your run completes.
- `memory/feedback/edits/` — auto-captured draft→posted diffs for your \
own past work (what you wrote vs. what the client actually published). \
Pure delta data, no editorial commentary. Study these BEFORE writing.
- `memory/plan.md` — content calendar (if it exists).
- `scratch/` — your working space.
- `context/research/` — deep research on client and company.
- `context/org/` — company context — industry, positioning, competitors.

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

- `list_directory` / `read_file` — explore the workspace
- `write_file` / `edit_file` — write scratch notes, draft posts, content plans
- `bash` — run shell commands. Every command starts with your workspace as the current directory. Use relative paths like `tools/validate_draft.py` or `memory/plan.md`.
- `write_result` — submit your final posts (ends the session)
- `python3 tools/validate_draft.py "text"` — self-check a draft for AI patterns, banned phrases, and structural issues BEFORE submitting. Also accepts `--file path/to/draft.md`. Returns JSON with issues. If `needs_correction` is true, revise and re-validate with `--attempt N`. After attempt 2, escape hatch activates (issues downgraded, proceeds).

### Past performance data

`memory/post-history.md` contains this client's top-performing posts by engagement — full text, full numbers, draft vs published versions. Read this before drafting. It is your baseline — everything you write will be compared to this distribution.
## Process

1. If `memory/plan.md` doesn't exist, create it first. Read every \
transcript and published post, mine them for candidate angles, cross-check \
against `memory/published-posts/` (LinkedIn-live) and `memory/draft-posts/` \
(Ordinal-pushed, not yet live) to avoid collisions, and write the plan. \
**Do NOT read your own prior-run scratch files as dedup source** — only \
posts that are actually in Ordinal or on LinkedIn count. A draft you saved \
locally last run that was never pushed does not exist for dedup purposes.
1a. Re-read the plan you just wrote. For each pair of posts, ask: do \
these share the same core insight? If yes, kill one and replace it with \
a genuinely different angle from the transcripts. Two posts can share a \
topic domain but must have different underlying insights. Repeat until \
every post in the plan is distinct.
1b. Read `memory/story-inventory.md`. If it is empty or missing, build \
it now: scan every transcript and every authoritative post (published \
on LinkedIn OR pushed to Ordinal), and write a list of every story, \
anecdote, or specific moment you find — one bullet per story, with file \
+ timestamp + one-sentence description, and whether it has been used. \
**Authoritative posts only**: `memory/published-posts/` and \
`memory/draft-posts/` (which is pre-populated from Ordinal). Never scan \
your own scratch writes as "used" — your in-run drafts don't exist until \
the publisher pushes them after your run. If the inventory already exists, \
consult it before picking angles: do not draft a post around a story \
already marked as used. Do not mark stories as used yourself — the \
publisher marks them after confirmed Ordinal push.
2. Pick the next unwritten topic from the plan. Identify the specific \
source material (file + timestamps) you'll draw from.
3. Draft in `scratch/`. After each paragraph with a factual claim, \
add a citation comment: `<!-- [filename, timestamp] "quote" -->`. \
Read it back.
4. Call `get_reader_reaction` on your draft. Read the reaction \
string carefully. You are looking for REAL positive engagement, \
not tolerance. Ship ONLY if the reaction contains felt engagement \
— phrases like `"felt real"`, `"line stays"`, `"been here"`, \
`"oh that's a good one"`, `"gonna forward this"`. Passive \
tolerance (`"nodding along"`, `"fine"`, `"smart flex"`, `"okay \
that's a line"`, `"reasonable take"`) means the post failed to \
land — keep revising. Make ONE surgical edit targeting the \
anchored span using `edit_file` (one sentence, one word, one \
line break — not a rewrite), then call `get_reader_reaction` \
again. The response also includes `_prior_reactions` — the last \
few reactions from this session, each with the draft's first \
line and length so you can see which ones were iterations on \
the same post. Use this trajectory to check whether your edits \
are moving the signal: if prior reactions for this post were \
\"lecture\" → \"sermon\" → \"sermon\", your edits aren't \
helping and you need a bigger change (cut more aggressively, \
target a different span, or pivot the angle). If the same \
reaction appears 2-3 times after different edits, your theory \
is wrong. If you've made 6+ edits and the reaction is still \
\"meh\", the topic may not land for this audience — consider \
dropping the post rather than shipping mediocre. Cap at 12 \
cycles per post.
5. Save the final draft to `scratch/final/` — NOT to `memory/draft-posts/`. \
That directory is authoritative for Ordinal-pushed content and is \
read-only during your run. `scratch/final/` is also where you read \
back your own earlier drafts to avoid self-collision across posts \
in the same batch.
6. Repeat steps 2-5 for all planned posts. When every post is \
complete, call `write_result` with the final JSON. The order of \
posts in the `posts` array IS the publication order — put the \
post you want published first at index 0.

## TIME BUDGET — CRITICAL

You have a hard wall-clock limit. **You MUST call `write_result` before \
running out of turns.** Budget your turns:
- Planning + research: ~5 turns
- Per post (draft + simulate + revise): ~8-10 turns
- Final assembly + `write_result`: ~3 turns
- **Reserve at least 5 turns as buffer for `write_result`.**

If you find yourself on revision v3+ for multiple posts, STOP REVISING. \
Take your best versions, save them to `scratch/final/`, and call \
`write_result`. Shipping 7 good posts beats perfecting 3 and timing out \
with nothing. **A run that doesn't call `write_result` produces ZERO \
output — every revision was wasted.**

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

## LinkedIn feed

On mobile, only ~140 characters are visible before the "see more" fold. \
On desktop, ~210 characters.

## Hard constraints

- Up to 3000 characters (LinkedIn platform cap).
- Every claim traces to a source file. No fabrication.
- No markdown formatting in posts (no #, **, etc.)

Everything else — length, cadence, diction, voice — is learnable from the \
client's own published posts, voice examples, and engagement history. Do \
not apply global stylistic rules. If a phrase is a problem for this client, \
you'll see it in the raw (draft, published) deltas and in the per-reactor \
data.

## Planning mode

When asked to create a content plan:
1. Read the memory files listed above (transcripts, voice examples, profile, published posts, plan, research, org context).
2. Mine transcripts for every usable story.
3. Check the EXISTING POSTS list injected into your prompt — don't \
repeat any topic, angle, or hook that's already been written or scheduled.
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
            "Irontomb returns ONE short visceral reaction from a "
            "LinkedIn reader (a working professional, not a writing "
            "teacher) plus an anchor pointing to where they reacted. "
            "Not a critique, not a prescription — just what they "
            "felt and where. Use this to stress-test drafts before "
            "shipping.\n\n"
            "Response shape:\n"
            "  reaction — under-15-word reader-voice reaction\n"
            "  anchor   — where in the post they reacted (quoted "
            "phrase, \"paragraph N\", \"at the end\", \"hook\")\n\n"
            "You interpret the reaction and decide whether/how to "
            "revise. Irontomb does not know craft. You do."
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
        "name": "submit_draft",
        "description": (
            "ATOMIC FINAL-DRAFT SUBMISSION. Use this instead of "
            "`write_file(posts/drafts/<id>/content.md, text)` to avoid "
            "leaving rough-draft files in the workspace. One call per "
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
            "on the Lineage calendar.\n"
            "  approver_user_ids (list[uuid], optional): explicit approver "
            "list. If omitted, defaults to the company's assigned AM, or "
            "the FOC user themselves if no AM is set.\n"
            "  publication_order (int, optional): sequencing hint when "
            "producing multiple drafts in one run (1, 2, 3…).\n"
            "  why_post (string, optional): your rationale — why this "
            "post, this angle, this hook. Stored as a draft_comment."
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
            "ICP match rate, and engagement breakdown. Use this instead of "
            "reading memory/post-history.md — same data, filtered server-side.\n\n"
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


def _dispatch_list_directory(root, args):
    """Path-aware routing:
    - Lineage mount paths → HTTP proxy to virio-api workspace
    - Everything else (scratch/, loose dirs) → fly-local SandboxFs

    Same routing rule as _dispatch_write_file so scratch files Stelle
    wrote with ``write_file`` are listable via the same path."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lfs.exec_list_directory(root, args)
    return _exec_list_directory(root, args)


def _dispatch_read_file(root, args):
    """Path-aware routing:
    - Lineage mount paths → HTTP proxy (transcripts, engagement JSON, etc.)
    - Scratch paths → fly-local SandboxFs (reads the v1/v2/v3 drafts
      Stelle wrote via write_file in the same run)"""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lfs.exec_read_file(root, args)
    return _exec_read_file(root, args)


def _lineage_write_blocked_message(path: str) -> str:
    """Explain why a write into a Lineage mount is refused and where
    Stelle should write instead (scratch paths work fine)."""
    return (
        f"Error: cannot write to `{path}` — that path is inside Lineage's "
        "workspace, which is READ-ONLY to Stelle (it's the client's space).\n\n"
        "Your options:\n"
        "  • FINAL drafts: call `submit_draft(user_slug, content, "
        "scheduled_date, why_post)` — the only write tool that reaches Lineage.\n"
        "  • Scratch / working notes / draft iterations: write to any path "
        "OUTSIDE the Lineage mount tree. Good choices: `scratch/post1-v1.md`, "
        "`scratch/plan.md`, `notes/brainstorm.md`. Those land on your "
        "fly-local SandboxFs, persist for the duration of the run, and "
        "you can `read_file` them back normally.\n\n"
        "Paths that route to Lineage (read-only for write): anything under "
        "`transcripts/`, `research/`, `engagement/`, `reports/`, `context/`, "
        "`posts/`, `edits/`, `tone/`, `strategy/`, or the shared "
        "`conversations/`, `slack/`, `tasks/`, `.pi/`."
    )


def _dispatch_write_file(root, args):
    """Writes route by path:

    - Paths inside a Lineage mount (transcripts/, posts/, strategy/, …) →
      refused with an error redirecting to ``submit_draft`` or a scratch
      path.
    - Scratch paths (anything NOT a Lineage mount — e.g. ``scratch/*``,
      ``notes/*``, loose files) → fly-local SandboxFs.
    - In local (non-Lineage) mode, always fly-local.

    Preserves Stelle's classic scratch-file iteration pattern
    (``scratch/post1-v1.md`` → ``scratch/post1-v2.md`` → …) while
    keeping the client's Lineage workspace read-only from her
    perspective. Only ``submit_draft`` reaches Lineage's drafts table."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lineage_write_blocked_message(path)
    return _exec_write_file(root, args)


def _dispatch_edit_file(root, args):
    """Same path-aware routing as _dispatch_write_file."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        path = args.get("path", "") or ""
        if _lfs.is_lineage_path(path):
            return _lineage_write_blocked_message(path)
    return _exec_edit_file(root, args)


def _dispatch_search_files(root, args):
    """Path-aware routing based on the ``directory`` arg. Searches over
    Lineage mounts use the HTTP endpoint; searches over scratch paths
    use the local ripgrep."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        directory = args.get("directory", "") or ""
        if _lfs.is_lineage_path(directory):
            return _lfs.exec_search_files(root, args)
    return _exec_search_files(root, args)


def _dispatch_bash(root, args):
    """Bash in Lineage mode is intentionally disabled — the remote workspace
    is unreachable via a local shell. Direct agents to structured tools."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        return _lfs.exec_bash_lineage_stub(root, args)
    return _exec_bash(root, args)


def _dispatch_mention_resolve(root, args):
    """Always goes through HTTP — whether local or Lineage mode. The
    Lineage endpoint wraps the same Supabase + APImaestro resolver Pi
    used, so behavior is identical. Falls through to a local error when
    not in Lineage mode (Stelle never had this tool outside Lineage)."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        return _lfs.exec_mention_resolve(root, args)
    return "Error: mention_resolve is only available in Lineage mode"


def _dispatch_submit_draft(root, args):
    """Final-draft submission — writes to Amphoreus's local posts store.

    Amphoreus drafts NEVER go back to Jacquard. Whether Stelle is running
    pure-local or in lineage-read mode (reading Jacquard's workspace data),
    submit_draft lands a row in the local ``local_posts`` table and a
    markdown file in ``output/{company}/drafts/``. The operator reviews
    from Amphoreus's Posts tab and pushes to Ordinal from there.

    The session-level wrapper in stelle.py (_stelle_submit_draft_with_castorice)
    runs Castorice fact-check on ``content`` before forwarding here, so by
    the time we land, the content is already corrected and why_post carries
    the fact-check report.

    Args (from the submit_draft tool schema):
        user_slug (str, required)   FOC user this draft is attributed to.
                                    In pure-local mode the slug == company
                                    keyword; in lineage-read mode it's
                                    the Jacquard FOC-user slug.
        content (str, required)     Final post markdown.
        scheduled_date (str, opt)   ISO YYYY-MM-DD calendar slot.
        publication_order (int, opt)
        why_post (str, opt)         Rationale + fact-check report.
        approver_user_ids (list, opt)  Ignored locally — approvals happen
                                       in the Amphoreus Posts tab, not
                                       through a separate approver UUID.
    """
    import json as _json
    import os as _os
    import uuid as _uuid
    from pathlib import Path as _Path

    user_slug = (args.get("user_slug") or "").strip()
    content = args.get("content") or ""
    if not content:
        return "Error: content is required"

    # Determine company for bookkeeping. Pull from env set by stelle_runner
    # (LINEAGE_COMPANY_ID present in lineage-read mode, otherwise fall back
    # to the lineage_user_slug env which in local mode equals company).
    # LAST resort: if neither is set, use user_slug itself.
    company = (
        _os.environ.get("STELLE_COMPANY_KEYWORD", "").strip()
        or user_slug
        or "unknown"
    )

    draft_id = str(_uuid.uuid4())
    scheduled_date = args.get("scheduled_date") or None
    publication_order = args.get("publication_order")
    why_post = args.get("why_post") or None

    # Title = first non-empty line stripped of markdown headers, max 200 chars.
    # Matches the heuristic Jacquard's /submit-draft uses so titles look
    # identical across modes.
    title = None
    for _line in content.splitlines():
        stripped = _line.strip()
        if stripped:
            title = stripped.lstrip("#").strip()[:200]
            break

    # 1) local_posts row — this is what Amphoreus's Posts tab reads.
    try:
        from backend.src.db.local import create_local_post
        create_local_post(
            post_id=draft_id,
            company=company,
            content=content,
            title=title,
            status="draft",
            why_post=why_post,
            scheduled_date=scheduled_date,
            publication_order=(
                publication_order if isinstance(publication_order, int) else None
            ),
        )
    except Exception as exc:
        logger.exception("[submit_draft] create_local_post failed: %s", exc)
        return f"Error: failed to persist draft to local_posts: {exc}"

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
        header_lines.extend(["## Why this post", "", why_post, ""])
    header_lines.extend(["## Content", "", content])
    try:
        md_path.write_text("\n".join(header_lines), encoding="utf-8")
    except Exception as exc:
        logger.debug("[submit_draft] failed to write markdown mirror: %s", exc)

    return (
        "Draft submitted to Amphoreus's Posts tab.\n"
        f"  draft_id: {draft_id}\n"
        f"  company: {company}\n"
        f"  user_slug: {user_slug or '(none)'}\n"
        f"  scheduled_date: {scheduled_date or '(unset)'}\n"
        f"  title: {(title or '')[:80]}\n"
        "\n"
        "The operator will review in the Posts tab and push to Ordinal "
        "from there. Nothing has been sent to Jacquard / LinkedIn yet."
    )


def _dispatch_query_observations(root, args):
    """Query scored observations. In Lineage mode goes through the remote
    workspace's ``/observations`` endpoint (backed by linkedin_posts +
    linkedin_reactions + ICP reports). Outside Lineage mode, falls back
    to the in-process Analyst implementation when scored_observations are
    pre-computed for this run — else returns empty."""
    from backend.src.agents import lineage_fs_client as _lfs
    if _lfs.is_lineage_mode():
        return _lfs.exec_query_observations(root, args)
    # Non-Lineage fallback — preserve Amphoreus's local behavior.
    try:
        from backend.src.agents.analyst import _tool_query_observations as _q
        # If the agent loop wired scored_observations into module state,
        # use them; otherwise return an empty result rather than crash.
        scored = globals().get("_scored_observations_for_run") or []
        return _q(args, scored)
    except Exception as e:
        import json as _json
        return _json.dumps({"count": 0, "observations": [], "error": str(e)})


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


def _fetch_all_ordinal_hooks(company_keyword: str) -> str:
    """Fetch hooks/titles of ALL posts in Ordinal (any status) for topic dedup.

    Returns a formatted string injected into Stelle's user prompt so she
    knows every topic already written, scheduled, or in-review and avoids
    duplication. No LLM call — just an API fetch + text formatting.
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
        return ""

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
        "\n\nEXISTING POSTS (all posts in Ordinal and locally generated — "
        f"{len(entries)} total). Full text included so you can judge "
        "thematic overlap accurately. DO NOT write a post that covers "
        "the same TOPIC as any existing post, even from a different "
        "angle. Two posts about the same activity, setting, or subject "
        "(e.g. climbing, piano, a specific trial) will read as "
        "repetition to the audience regardless of angle distinction. "
        "Every post you write must occupy genuinely new thematic "
        "territory:\n\n"
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
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        return []

    username = username_path.read_text().strip()
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



def _build_observation_digest(company_keyword: str, n: int = 10) -> str:
    """Compact raw digest of the client's top-N posts by engagement.

    Replaces the `query_observations` tool as Stelle's access to her own
    performance history. Dumps the N best-reacted-to scored observations
    as full text + full numbers. No aggregation, no summary: the same raw
    (draft, published, engagement) triples she used to fetch via the
    tool, just delivered up-front at workspace-stage time.
    """
    try:
        from backend.src.db.local import ruan_mei_load
    except Exception:
        return ""

    state = ruan_mei_load(company_keyword)
    if not state:
        return "# Post history\n\n(no scored observations available)\n"

    scored = [
        o for o in state.get("observations", [])
        if ((o.get("reward") or {}).get("raw_metrics") or {}).get("impressions")
    ]
    if not scored:
        return "# Post history\n\n(no scored observations available)\n"

    def _reactions(o: dict) -> int:
        return int(((o.get("reward") or {}).get("raw_metrics") or {}).get("reactions", 0) or 0)

    scored.sort(key=_reactions, reverse=True)
    top = scored[:n]

    lines: list[str] = []
    lines.append("# Post history — top performers")
    lines.append("")
    lines.append(
        f"The {len(top)} highest-reacted posts from this client's scored "
        f"history. Each entry shows the text as Stelle drafted it, the "
        f"text as the client ultimately published it (may differ after "
        f"client edits), and the engagement numbers as of the last sync. "
        f"This is your baseline — everything you write will be compared "
        f"to this distribution."
    )
    lines.append("")

    for i, o in enumerate(top, 1):
        raw = ((o.get("reward") or {}).get("raw_metrics") or {})
        r = int(raw.get("reactions", 0) or 0)
        c = int(raw.get("comments", 0) or 0)
        rp = int(raw.get("reposts", 0) or 0)
        imp = int(raw.get("impressions", 0) or 0)
        posted_at = (o.get("posted_at") or "")[:10]
        draft = (o.get("post_body") or "").strip()
        published = (o.get("posted_body") or "").strip()

        lines.append("---")
        lines.append(f"## Post {i} — {posted_at}")
        lines.append("")
        lines.append(f"**Engagement:** {r} reactions · {c} comments · {rp} reposts · {imp} impressions")
        lines.append("")

        if published and published != draft:
            lines.append("**Stelle draft:**")
            lines.append("")
            lines.append(draft)
            lines.append("")
            lines.append("**Client-published version:**")
            lines.append("")
            lines.append(published)
        else:
            lines.append("**Post text:**")
            lines.append("")
            lines.append(draft or published)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workspace setup (Gap 3, 10 — jacquard directory structure)
# ---------------------------------------------------------------------------

def _setup_workspace(company_keyword: str) -> Path:
    """Stage workspace matching Jacquard's memory/ layout.

    Uses a persistent workspace under data/workspaces/ instead of a temp dir
    so that snapshots, rollback, and feedback all share the same root.
    Symlinks and Supabase-sourced files are rebuilt on each run.

    Under the stripped architecture (2026-04-10), no RuanMei-derived artifact
    (memory/strategy.md) is written into the workspace. Stelle operates on
    raw workspace inputs only.
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
    # Also wipe scratch/ at run start. Without this, stale draft files from
    # prior runs (posts that were never pushed to Ordinal) linger and can:
    #   1. Collide with new drafts that get the same slug-based filename
    #   2. Pollute scratch/final/ so Stelle reads ghost drafts as her own
    #      in-progress output when avoiding self-collision across posts
    #   3. Feed _extract_result_from_scratch stale files if write_result
    #      fails and the fallback fires
    # Anything unpushed "does not exist" per the new dedup model, so wiping
    # scratch at setup start enforces that invariant at the filesystem level.
    scratch = workspace / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
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

    # memory/draft-posts/ — READ-ONLY pre-populated from Ordinal's /posts endpoint
    # (authoritative "pushed-but-not-yet-live" drafts). Stelle's in-run outputs
    # go to scratch/final/, NOT here. This directory is the dedup source; her
    # own writes don't "exist" until the publisher pushes them after the run.
    draft_posts_dir = mem / "draft-posts"
    draft_posts_dir.mkdir()
    try:
        ordinal_drafts = _fetch_ordinal_drafts(company_keyword, exclude_dates=_published_dates)
        for od in ordinal_drafts:
            (draft_posts_dir / od["filename"]).write_text(od["content"], encoding="utf-8")
    except Exception as e:
        logger.debug("[Stelle] Ordinal draft prefill skipped: %s", e)

    # memory/feedback/edits/ — auto-saved draft→posted edit diffs only.
    # Operator-typed client feedback files are NOT symlinked in — they
    # encode the operator's interpretation of client reactions, which is
    # prescriptive and out-of-policy. Client-direct content belongs in
    # transcripts/.
    feedback_dir = mem / "feedback"
    feedback_dir.mkdir()
    edits_dir = feedback_dir / "edits"
    edits_dir.mkdir()

    # context/research/ — fetch BEFORE profile.md so we can fallback
    research_files = _fetch_research_files(company_keyword)
    research_dir = ctx / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    _person_research_content: str = ""
    for rf in research_files:
        (research_dir / rf["filename"]).write_text(rf["content"], encoding="utf-8")
        if rf["filename"] == "person.md":
            _person_research_content = rf["content"]

    # memory/profile.md — LinkedIn profile data (APIMaestro, or person.md fallback)
    profile_summary = _fetch_linkedin_profile(company_keyword)
    if profile_summary:
        (mem / "profile.md").write_text(profile_summary, encoding="utf-8")
    elif _person_research_content:
        (mem / "profile.md").write_text(
            "# Client Profile (from deep research)\n\n" + _person_research_content[:8000],
            encoding="utf-8",
        )
        logger.info("[Stelle] Using person.md research as profile.md fallback")

    # Cyrene's strategic brief — treated as source material (like a transcript)
    # so Stelle can read content_priorities, content_avoid, etc. without
    # being prescriptively told what to write.
    cyrene_brief_path = client_mem / "cyrene_brief.json"
    if cyrene_brief_path.exists():
        import json as _json
        try:
            brief = _json.loads(cyrene_brief_path.read_text(encoding="utf-8"))
            brief_text = (
                "# Strategic Brief (from Cyrene's account review)\n\n"
                + _json.dumps(brief, indent=2, default=str)
            )
            (source_mat / "cyrene-strategic-brief.txt").write_text(
                brief_text, encoding="utf-8",
            )
            logger.info("[Stelle] Loaded Cyrene brief into source-material/")
        except Exception as e:
            logger.debug("[Stelle] Cyrene brief load skipped: %s", e)

    # memory/constraints.md — voice/tone rules from accepted posts (placeholder if empty)
    accepted_src = client_mem / "accepted"
    if accepted_src.exists() and any(accepted_src.iterdir()):
        constraint_lines = ["Voice and tone reference from accepted posts:\n"]
        for f in sorted(accepted_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                constraint_lines.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')}\n")
        (mem / "constraints.md").write_text("\n".join(constraint_lines), encoding="utf-8")
    else:
        (mem / "constraints.md").write_text(
            "No accepted posts yet. Infer voice and tone from voice-examples/.",
            encoding="utf-8",
        )

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
    # ctx dir and ctx/research/ already created above (before profile.md)
    ctx.mkdir(exist_ok=True)

    # context/org/ — company context from research
    org_dir = ctx / "org"
    org_dir.mkdir(exist_ok=True)
    company_research = research_dir / "company.md"
    if company_research.exists():
        os.symlink(company_research.resolve(), org_dir / "company.md")

    # context/topic-velocity.md — intentionally NOT symlinked. Previously
    # Perplexity-generated trending-topic summary auto-injected into
    # Stelle's context, which violated the "transcripts/ + deltas +
    # engagement only" input policy. If Stelle needs trending-topic
    # context during drafting she reaches for the web_search tool.

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
    (scratch_dir / "final").mkdir(exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)

    # Observation digest: dump top-N scored posts (full text + full
    # engagement numbers) into memory/post-history.md. Replaces the
    # query_observations tool — same raw data, delivered up-front,
    # so Stelle reads it like any other memory file.
    digest = _build_observation_digest(company_keyword, n=10)
    if digest:
        (mem / "post-history.md").write_text(digest, encoding="utf-8")

    _write_tool_scripts(workspace)

    logger.info(
        "[Stelle] Workspace staged at %s (%d published posts, %d voice examples, %d research files)",
        workspace, len(pub_posts), len(voice_examples), len(research_files),
    )

    return workspace


# ---------------------------------------------------------------------------
# Learned overrides — graduated defaults from data
# ---------------------------------------------------------------------------

_OVERRIDE_MIN_EDITS_FOR_CHAR_LIMIT = 5


# ---------------------------------------------------------------------------
# Dynamic directives
# ---------------------------------------------------------------------------

_LINEAGE_DIRECTIVES_TOOL_OVERRIDES = """\
## Tool semantics that differ from the default Amphoreus prompt

Several instructions in the default system prompt referenced paths or files
that **do not exist in the Lineage workspace**, or suggested workflows that
create duplicate drafts. Here's how to do the same things correctly:

- `memory/post-history.md` → does not exist. Call the
  `query_observations` tool instead. Args: ``{min_reward, max_reward,
  limit, summary_only}``. Output shape is identical to what the default
  prompt expects — each observation carries ``reward.immediate`` (an
  engagement score), per-reaction breakdown, topic/format tags,
  ICP match rate, and the post text itself. Use ``summary_only: true``
  to get distributions (reward_mean, reward_std, topic_distribution,
  format_distribution).

- `memory/voice-examples/` → use `posts/published/` instead. Every post
  file already carries engagement metadata in its header (Reactions /
  Comments / Reposts / per-reaction breakdown / ICP rate), so you can
  rank them yourself. For a curated slice, also check `tone/` which
  exposes ``tone_references`` picks.

- `bash` is DISABLED in Lineage mode (scratch filesystem isn't wired).

- `write_file` / `edit_file` route BY PATH:
  * **Lineage mount paths are READ-ONLY.** Any write under
    ``transcripts/``, ``research/``, ``engagement/``, ``reports/``,
    ``context/``, ``posts/``, ``edits/``, ``tone/``, ``strategy/``, or
    the shared ``conversations/``, ``slack/``, ``tasks/``, ``.pi/``
    is refused — those are the client's files. The error message tells
    you exactly where to write instead.
  * **Scratch paths work normally.** Write freely to anything OUTSIDE
    the Lineage mounts: ``scratch/post1-v1.md``, ``scratch/plan.md``,
    ``notes/brainstorm.md``, ``drafts/wip.md`` — whatever you want.
    Those land on your fly-local SandboxFs, persist for the run, and
    you can ``read_file``/``list_directory`` them back normally.

- **Use scratch for draft iteration.** Classic pattern still works:

      write_file("scratch/post1-v1.md", <draft>)
      → get_reader_reaction(draft_text=<draft>)
      → interpret reaction
      → write_file("scratch/post1-v2.md", <revised>)
      → get_reader_reaction(draft_text=<revised>)
      → … iterate until the reader stops complaining.

  You can also keep drafts purely in-context if you prefer —
  ``get_reader_reaction`` takes the full draft text directly, so a
  file round-trip is never required. Use whichever feels natural.

- **FINAL DRAFTS: use `submit_draft`.** One call per finished post:

      submit_draft(
        user_slug="<slug>",          # which FOC user this post is for
        content="<final markdown>",   # the final post, plain markdown
        scheduled_date="YYYY-MM-DD",  # calendar slot (tomorrow or later)
        publication_order=1,          # 1, 2, 3… for multi-post runs
        why_post="<rationale>",       # stored alongside the draft
      )

  **Where the draft lands.** Reads above came from Jacquard's workspace
  (transcripts, engagement, context, etc.) — but ``submit_draft`` writes
  to **Amphoreus's own Posts tab**, not to Jacquard. Nothing you submit
  goes back to Jacquard or Lineage. The operator reviews the draft in
  Amphoreus's UI and pushes to Ordinal from there.

  ``submit_draft`` also runs Castorice fact-check on your content before
  persisting — the fact-check report + citations are appended to
  ``why_post`` and visible to the operator during review.

- **Multi-post runs: vary angles, don't riff one topic twice.**
  When the user asks for N posts, cover N DIFFERENT angles/topics —
  not the same topic with wording variations. Use
  ``query_observations({summary_only: true})`` to see the client's
  topic/format distribution and pick underrepresented angles. Space the
  publication dates across the cadence (3 posts/week = Mon/Wed/Fri or
  Mon/Tue/Thu). Pass each post's slot as ``scheduled_date``.

"""


_LINEAGE_DIRECTIVES_USER_TARGETED = """\
# Lineage Mode — USER-TARGETED RUN

You are running against a REMOTE workspace served by Lineage (virio-api),
scoped to a single FOC user for this run. Every draft you produce will
be attributed to that user.

**Lineage's workspace is READ-ONLY to you.** Calls to `list_directory`,
`read_file`, `search_files` on Lineage paths proxy through to the
client's workspace over HTTPS. ``write_file`` / ``edit_file`` on those
same paths are refused.

**But scratch paths work normally.** Any path OUTSIDE the Lineage mount
tree (``scratch/``, ``notes/``, ``drafts/``, loose top-level files) is
your own fly-local SandboxFs — read, write, edit, list freely.

- Lineage paths (routed to client's workspace, READ-ONLY for writes):
  ``transcripts/``, ``research/``, ``engagement/``, ``reports/``,
  ``context/``, ``posts/``, ``edits/``, ``tone/``, ``strategy/``, and
  the shared ``conversations/``, ``slack/``, ``tasks/``, ``.pi/``.
- Scratch paths (fly-local, read/write): anything else.

Classic iteration pattern still works:

    write_file("scratch/post1-v1.md", <draft>)
    → get_reader_reaction(draft_text=<draft>)
    → write_file("scratch/post1-v2.md", <revised>)
    → get_reader_reaction(draft_text=<revised>)
    → … until Irontomb stops complaining, then submit_draft.

You can also keep drafts purely in-context — ``get_reader_reaction``
takes the full draft directly. Use whichever feels natural.

Only ``submit_draft`` writes to Lineage (finished posts only).

## Workspace layout (read-only; paths auto-prefixed to the target user)

- ``transcripts/``       — raw client-direct transcripts
- ``research/``          — deep research
- ``engagement/``        — LinkedIn engagement observations.
                           Files: ``posts.json``, ``reactions.json``,
                           ``comments.json``, ``profiles.json``,
                           ``work_experiences.json``, ``client_info.json``
- ``reports/``           — latest ICP report JSON + Typst template. The
                           ICP report names specific engagers and highlights
                           recurring content themes — read it before
                           deciding angles.
- ``context/``           — company context. ``account.md`` is auto-built
                           (client name, company, posts_per_month target,
                           Slack channels). Other files are operator-
                           uploaded brand docs / positioning PDFs.
- ``posts/published/``   — already-published LinkedIn posts
- ``posts/drafts/``      — unpushed drafts (do NOT attempt to write here —
                           use ``submit_draft``)
- ``edits/``             — FEEDBACK SIGNAL. For each published draft, a
                           markdown file showing the earliest snapshot,
                           the final published text, AND threaded
                           operator comments. Read these before drafting
                           — they reveal how operators actually revise
                           your output.
- ``tone/``              — style references / voice calibration examples
- ``strategy/``          — persistent cross-run strategy memory. Read
                           it; you cannot write to it in Lineage mode.

Shared (not user-scoped; don't prepend slug):
- ``tasks/``             — pending review tasks (read-only)
- ``slack/``             — Slack channel context (read-only)
- ``conversations/``     — ``trigger-log.jsonl``: replay of every prior
                           trigger (interviews, CE feedback with diffs,
                           manual runs) in chronological order. This is
                           your history of prior interactions with this
                           client — scan it at session start.
- ``.pi/``               — Jacquard's own agent's skill files. IGNORE.

## Draft write contract

**Use ``submit_draft`` for finished posts.** One call per finished post.
Reads above came from Jacquard's workspace, but ``submit_draft`` lands in
**Amphoreus's own Posts tab** — NOT in Jacquard. Nothing you submit goes
back to Jacquard or Lineage. The operator reviews in Amphoreus's UI and
pushes to Ordinal from there.

``submit_draft`` runs Castorice fact-check on your content before
persisting. The fact-check report + citations are appended to
``why_post`` so the operator sees them during review.

## Ingestion order at session start (Lineage mode)

1. ``list_directory("")`` — confirm the target slug and what's available.
2. ``read_file("conversations/trigger-log.jsonl")`` — your history with
   this company (interviews, CE feedback diffs, prior runs). Scan it.
3. ``read_file("strategy/strategy.md")`` if it exists — cross-run memory
   left by your previous selves.
4. ``list_directory("edits/")`` — operator-edit feedback signal.
5. ``list_directory("transcripts/")`` + read the latest 2-3 transcripts.
6. ``list_directory("engagement/")`` — the JSON files are your data
   substrate for what's actually landing with this user's audience.
7. ``list_directory("reports/")`` — the ICP report names specific
   engagers and themes. Read it once.
8. Spot-read ``tone/``, ``posts/published/``, ``context/account.md`` as
   needed for voice calibration and company facts.
"""


_LINEAGE_DIRECTIVES_COMPANY_WIDE = """\
# Lineage Mode — COMPANY-WIDE RUN

You are running against a REMOTE workspace served by Lineage (virio-api).
Filesystem tool calls route BY PATH:

- **Lineage mount paths** (``transcripts/``, ``research/``, ``engagement/``,
  ``reports/``, ``context/``, ``posts/``, ``edits/``, ``tone/``,
  ``strategy/``, and the shared ``conversations/``, ``slack/``, ``tasks/``,
  ``.pi/``) hit the client's remote workspace over HTTPS. These are
  READ-ONLY for writes — ``write_file``/``edit_file`` on them is refused.
- **Scratch paths** (``scratch/``, ``notes/``, ``drafts/``, any loose
  top-level file) land on your fly-local SandboxFs. Read/write freely.
  Classic scratch-file draft iteration works exactly as in local mode.

Only ``submit_draft`` writes to Lineage (finished posts only).

## Workspace layout (all read-only)

The workspace root contains one top-level directory per FOC user of the
company. Each user has the same subtree structure:

    /                                  (workspace root)
    ├── <user-slug>/
    │   ├── transcripts/               (client-direct interview text)
    │   ├── research/                  (deep research)
    │   ├── engagement/                (posts.json / reactions.json /
    │   │                                comments.json / profiles.json /
    │   │                                work_experiences.json / client_info.json)
    │   ├── reports/                   (latest ICP report + Typst template)
    │   ├── context/                   (account.md + uploaded brand docs)
    │   ├── posts/published/           (published LinkedIn posts)
    │   ├── posts/drafts/              (existing unpushed drafts — use submit_draft
    │   │                                for new ones)
    │   ├── edits/                     (FEEDBACK SIGNAL: per-draft first-snapshot
    │   │                                vs. final-published diffs WITH threaded
    │   │                                operator comments)
    │   ├── tone/                      (voice/style references)
    │   └── strategy/                  (persistent cross-run strategy memory —
    │                                    you can read it, you cannot write to it)
    ├── tasks/                         (shared — pending review tasks, read-only)
    ├── slack/                         (shared, read-only)
    ├── conversations/                 (shared — trigger-log.jsonl: replay of every
    │                                    prior interview / CE feedback / manual run
    │                                    for this company, chronological)
    └── .pi/                           (Jacquard-agent skill files — IGNORE)

You MUST use explicit user-slug prefixes in every filesystem call — do not
guess. Start by calling ``list_directory("")`` to discover the available
FOC-user slugs.

## Per-draft author attribution

Each finished post belongs to exactly ONE user. You decide which user by
passing their slug to ``submit_draft``:

    submit_draft(user_slug="<user-slug>", content=<post text>, ...)

The slug determines attribution — there is no separate ``author`` field.

## Draft write contract

**Use ``submit_draft`` for finished posts.** One call per finished post.
Reads above came from Jacquard's workspace, but ``submit_draft`` lands in
**Amphoreus's own Posts tab** — NOT in Jacquard. Nothing you submit goes
back to Jacquard or Lineage. The operator reviews in Amphoreus's UI and
pushes to Ordinal from there.

``submit_draft`` runs Castorice fact-check on your content before
persisting. The fact-check report + citations are appended to
``why_post`` so the operator sees them during review.

## Ingestion order at session start (Lineage mode)

1. ``list_directory("")`` — discover the FOC-user slugs.
2. ``read_file("conversations/trigger-log.jsonl")`` — history of prior
   triggers for this company (interviews, CE feedback diffs, manual runs).
3. For each slug you plan to write for:
   a. ``read_file("<slug>/strategy/strategy.md")`` if it exists.
   b. ``list_directory("<slug>/edits/")`` — operator-edit feedback.
   c. ``list_directory("<slug>/transcripts/")`` + read latest 2-3.
   d. ``list_directory("<slug>/engagement/")`` — JSON data substrate.
   e. ``list_directory("<slug>/reports/")`` — ICP report if present.
   f. Spot-read ``<slug>/tone/``, ``<slug>/posts/published/``,
      ``<slug>/context/account.md`` as needed.
"""


def _today_preamble() -> str:
    """Inject today's date so Stelle schedules posts in the future, not
    the past. Claude's training cutoff means she has no inherent sense
    of what day it is — without this she'll hallucinate dates from
    whatever year felt plausible in training data. Applies in both
    Lineage and local mode."""
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
        "forward. The posts_per_month target in `context/account.md` "
        "tells you the cadence density.\n"
    )


def _build_dynamic_directives(company_keyword: str) -> str:
    """Return dynamic directives for the system prompt.

    Always emits the today-preamble so Stelle knows what day it is.

    Additionally emits one of two Lineage-mode overlays when running
    against the remote workspace:

    - ``USER-TARGETED`` when the runner was given ``--lineage-user-slug``.
      Every draft is attributed to that user and paths are auto-prefixed.
    - ``COMPANY-WIDE``  when no user slug was supplied. Stelle sees the
      full workspace with all FOC-user subtrees and must include the
      slug in every filesystem call.

    In local mode (no Lineage env vars), only the today-preamble is
    emitted.
    """
    preamble = _today_preamble()
    try:
        from backend.src.agents import lineage_fs_client as _lfs
        if not _lfs.is_lineage_mode():
            return preamble
        layout = (
            _LINEAGE_DIRECTIVES_USER_TARGETED
            if _lfs.is_user_targeted()
            else _LINEAGE_DIRECTIVES_COMPANY_WIDE
        )
        # Always append the tool-override block so Stelle doesn't try to read
        # memory/post-history.md or memory/voice-examples/ which don't exist.
        return (
            preamble
            + "\n"
            + layout
            + "\n\n"
            + _LINEAGE_DIRECTIVES_TOOL_OVERRIDES
        )
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
        pass
    else:
        turn_start_idx = _find_turn_start(messages, cut_idx)
        if turn_start_idx >= 0 and turn_start_idx > 0:
            is_split_turn = True
        else:
            while cut_idx < len(messages) and messages[cut_idx].get("role") != "user":
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
            returns {reaction, anchor}. The reaction is reader-voice
            (not critique, not prescription). Stelle interprets what
            caused the reaction and decides how to revise.
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

                # Irontomb now outputs {reaction, anchor}. No
                # scalar prediction, so no gradient block.
                return json.dumps(result, default=str)
            except Exception as _e:
                logger.warning("[Stelle] Irontomb reader-reaction call failed: %s", _e)
                return json.dumps({"_error": f"simulate failed: {str(_e)[:200]}"})

        # Register the session-scoped handler. It closes over
        # simulate_call_count + simulate_results for iteration discipline,
        # so it must live in the per-run dict, not module-level _TOOL_HANDLERS.
        run_handlers["get_reader_reaction"] = _stelle_get_reader_reaction_handler

        # Wrap submit_draft with a Castorice fact-check gate. The wrapper
        # runs Castorice.fact_check_post on `content` first, replaces the
        # content with the corrected post, and appends the fact-check
        # report to why_post so the reviewer in Lineage sees Castorice's
        # findings in the draft_comment thread. Failures in Castorice
        # surface as a visible error back to Stelle — she can retry or
        # submit unchecked after acknowledging the failure.
        _original_submit_draft = run_handlers.get("submit_draft")

        def _stelle_submit_draft_with_castorice(_root: Path, args: dict) -> str:
            if not company_keyword:
                return "Error: company not set"
            content = args.get("content") or ""
            if not content:
                return "Error: content is required"
            # Run Castorice on the clean post text before the Lineage POST.
            try:
                from backend.src.agents.castorice import Castorice
                fc = Castorice().fact_check_post(company_keyword, content)
                corrected = fc.get("corrected_post") or content
                report = fc.get("report") or ""
                citation_comments = fc.get("citation_comments") or []
            except Exception as _e:
                logger.warning("[Stelle] Castorice fact-check failed: %s", _e)
                corrected = content
                report = f"[Castorice unavailable: {str(_e)[:200]}]"
                citation_comments = []

            # Build the enriched payload. Keep Stelle's own why_post text
            # and append the fact-check summary so the reviewer sees both.
            forwarded = dict(args)
            forwarded["content"] = corrected
            existing_why = forwarded.get("why_post") or ""
            fc_summary_parts: list[str] = []
            if existing_why:
                fc_summary_parts.append(existing_why)
            if report:
                fc_summary_parts.append("## Castorice fact-check\n\n" + report)
            if citation_comments:
                fc_summary_parts.append(
                    "## Citations (Castorice)\n\n"
                    + "\n".join(f"- {c}" for c in citation_comments)
                )
            if fc_summary_parts:
                forwarded["why_post"] = "\n\n---\n\n".join(fc_summary_parts)

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

            messages.append({"role": "assistant", "content": response.content})

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
                        # Lineage ingestion failures are FATAL. Don't convert
                        # to a tool error and let Stelle muddle on with
                        # missing data — re-raise so the subprocess crashes
                        # cleanly and job_manager records a failed run.
                        from backend.src.agents.lineage_fs_client import (
                            LineageIngestionError as _LineageErr,
                        )
                        if isinstance(e, _LineageErr):
                            logger.error(
                                "[Stelle] FATAL: Lineage ingestion failed during "
                                "%s — aborting run. %s", name, e,
                            )
                            if event_callback:
                                event_callback("error", {
                                    "message": f"Lineage ingestion failed: {e}",
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

            # Prediction tracking — score this post with the draft scorer
            # BEFORE it's published so we can compare predicted vs actual
            # engagement after the post is scored. This closes the validation
            # loop: does the model actually get better over time?
            try:
                from backend.src.utils.draft_scorer import score_drafts
                _draft_scores = score_drafts(company_keyword, [{"text": corrected}])
                if _draft_scores and _draft_scores[0].model_source != "no_model":
                    _extra_fields["predicted_engagement"] = _draft_scores[0].predicted_score
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
                _gen_meta = dict(_extra_fields) if _extra_fields else {}
                _save_post(
                    post_id=_draft_id,
                    company=company_keyword,
                    content=corrected,
                    title=hook[:200] if hook else None,
                    status="draft",
                    why_post=why_post or None,
                    citation_comments=citation_comments,
                    pre_revision_content=None,
                    cyrene_score=None,
                    generation_metadata=_gen_meta if _gen_meta else None,
                )
            except Exception as _e:
                logger.warning("[Stelle] Could not save post %d to local SQLite: %s", i, _e)

        # DELETED: the old Lineage-parallel-write path.
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
    # In Lineage mode the workspace lives remotely; there's no local
    # ``memory/<company>/linkedin_username.txt`` to read. The display name
    # flows in through Lineage's ``context/account.md`` which Stelle reads
    # via a tool call during the agent loop. Skip the local preamble.
    try:
        from backend.src.agents.lineage_fs_client import is_lineage_mode
        _lineage_active = is_lineage_mode()
    except Exception:
        _lineage_active = False

    if not _lineage_active:
        username_path = P.linkedin_username_path(company_keyword)
        if not username_path.exists():
            raise FileNotFoundError(
                f"Missing memory/{company_keyword}/linkedin_username.txt — "
                f"create this file with the client's LinkedIn username "
                f"(the part after linkedin.com/in/) before running the pipeline."
            )

        # Resolve proper display name from Supabase (falls back to slug)
        username = username_path.read_text().strip()
        if username:
            _, _, display_name = _resolve_supabase_ids(username)
            if display_name:
                client_name = display_name

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

    # Purge any unpushed draft rows in local_posts from prior runs for this
    # company. Per the project's dedup model, a draft that was never pushed
    # to Ordinal "doesn't exist" — it must not persist across runs, because
    # Stelle's dedup reads pushed/published content only. This is the DB-side
    # companion to the memory/ + scratch/ filesystem wipe in _setup_workspace.
    # Rows with a non-empty ordinal_post_id are preserved (they are owned by
    # Ordinal), as are non-draft statuses (posted/scheduled/failed).
    try:
        from backend.src.db.local import purge_unpushed_drafts as _purge_drafts
        _purged = _purge_drafts(company_keyword)
        if _purged:
            logger.info(
                "[Stelle] Purged %d unpushed draft(s) from local_posts for %s",
                _purged, company_keyword,
            )
    except Exception as _purge_err:
        logger.warning("[Stelle] local_posts purge skipped: %s", _purge_err)

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

    # Existing posts from Ordinal — all statuses — for topic dedup.
    # Operational (not prescriptive), so it stays: prevents Stelle from
    # writing the same post twice.
    existing_posts_context = ""
    try:
        existing_posts_context = _fetch_all_ordinal_hooks(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Ordinal post dedup fetch skipped: %s", _e)

    # Series Engine: inject series context if a series post is due (operational).
    series_context = ""
    try:
        from backend.src.services.series_engine import get_stelle_series_context as _series_ctx
        series_context = _series_ctx(company_keyword)
    except Exception as _e:
        logger.debug("[Stelle] Series context skipped: %s", _e)

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
