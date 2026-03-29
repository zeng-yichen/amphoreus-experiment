"""
Stelle — Jacquard-style agentic LinkedIn ghostwriter.

Delegates the agentic loop to Pi (pi.dev CLI), which handles context
compaction, session persistence, and tool execution natively.  Falls back
to a direct Anthropic API loop if Pi is not installed.

The agent explores the client workspace (transcripts, published posts,
content strategy, ABM profiles, feedback, revisions), drafts posts grounded
in transcripts, self-reviews against a "magic moment" quality bar, and
outputs structured JSON.  Posts are then fact-checked by Permansor Terrae.
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

import vortex as P

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stelle")

_client = Anthropic()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APIMAESTRO_KEY = os.getenv("APIMAESTRO_API_KEY", "")
APIMAESTRO_HOST = os.getenv("APIMAESTRO_HOST", "")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

_PI_AVAILABLE = shutil.which("pi") is not None

MAX_AGENT_TURNS = 60
MAX_TOOL_OUTPUT_CHARS = 80_000
MAX_FETCH_CHARS = 12_000
MAX_BASH_OUTPUT_CHARS = 30_000

# ---------------------------------------------------------------------------
# Langfuse observability (Gap 12)
# ---------------------------------------------------------------------------

_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_BASE_URL,
        )
        return _langfuse
    except ImportError:
        logger.info("[Stelle] langfuse not installed — tracing disabled")
        return None


# ---------------------------------------------------------------------------
# System prompt — Pi-native AGENTS.md (primary path)
# ---------------------------------------------------------------------------

_PI_AGENTS_TEMPLATE = """\
# Executive LinkedIn Ghostwriter

You turn meeting transcripts into posts that sound like the client wrote \
them — their vocabulary, their energy, their perspective, but better.

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

## Workspace

Explore the workspace using your tools. Key paths:

| Path | Contents |
|------|----------|
| `context/user/profile-linkedin.md` | Client's LinkedIn profile (headline, about, experience) |
| `transcripts/primary.md` | Latest interview transcript — your main source material |
| `transcripts/history/` | Previous content interviews — additional stories and context |
| `notes/` | Interview prep notes, topic banks, planning docs |
| `context/published-posts/` | Published posts with engagement data for this client |
| `context/draft-posts/` | Posts already written for this client — do NOT duplicate |
| `context/research/` | Deep research on client and company |
| `context/org/` | Company context — industry, positioning, competitors |
| `accepted/` | Published / approved posts — study these for voice |
| `content_strategy/` | Content strategy documents |
| `abm_profiles/` | ABM target briefings |
| `feedback/` | Client feedback on previous drafts |
| `revisions/` | Before/after revision pairs |
| `past_posts/` | Post history for redundancy checking |
| `scratch/` | Your working area — write plans, drafts, notes here |

Everything here is relevant. Read it all before you write.

**Content strategy from data**: Analyze the client's published posts in \
`context/published-posts/`. Each file includes engagement metrics (reactions, \
comments, reposts, engagement score, outlier flags). Identify which topics, \
formats, angles, and hooks drive the strongest engagement. Use these patterns \
as your de facto content strategy — write more of what resonates, less of \
what falls flat. If no `content_strategy/` document exists, intuit one \
entirely from the engagement data — you have everything you need. If a \
strategy document does exist, treat it as an intent layer (e.g. pivots, \
ABM targets, compliance) that can override the data, but default to what \
the numbers show.

**Critical**: Check `context/draft-posts/` and `past_posts/` before writing. \
Never duplicate an idea that already exists.

## Web Research

To search the web (Parallel API):
```bash
python3 tools/web_search.py "your search query here"
```

To extract content from a URL:
```bash
python3 tools/fetch_url.py "https://example.com/article"
```

## Process

1. Read everything in the workspace before writing anything
2. Write a content plan to `scratch/plan.md`
3. Draft posts to `scratch/drafts/` — one file per post
4. Re-read your drafts. Apply the magic moment test. Revise until they pass.
5. When every post passes, write the final result JSON to `output/result.json`

## Output

When you're done, write a JSON file to `output/result.json`:

```json
{{
  "posts": [
    {{
      "hook": "First sentence or opening line (required, <=200 chars)",
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

## Hard Constraints

- LinkedIn hard limit: 3,000 characters per post. Aim for 1,300-3,000.
- No markdown formatting in posts (no #, **, etc.)
- No emojis unless the client's published posts use them
- Banned phrases: "game-changer", "deep dive", "let that sink in", \
"here's the thing", "in today's [anything]", "buckle up", "the reality is"

{dynamic_directives}
"""

# ---------------------------------------------------------------------------
# System prompt — direct API loop (fallback when Pi is unavailable)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM_TEMPLATE = """\
# Executive LinkedIn Ghostwriter

You turn meeting transcripts into posts that sound like the client wrote \
them — their vocabulary, their energy, their perspective, but better.

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

## Workspace

Use the `list_directory` and `read_file` tools to explore the workspace. \
The top-level directory is the client's memory folder. Key paths:

| Path | Contents |
|------|----------|
| `context/user/profile-linkedin.md` | Client's LinkedIn profile (headline, about, experience) |
| `transcripts/primary.md` | Latest interview transcript — your main source material |
| `transcripts/history/` | Previous content interviews — additional stories and context |
| `notes/` | Interview prep notes, topic banks, planning docs |
| `context/published-posts/` | Published posts with engagement data for this client |
| `context/draft-posts/` | Posts already written for this client — do NOT duplicate |
| `context/research/` | Deep research on client and company |
| `context/org/` | Company context — industry, positioning, competitors |
| `accepted/` | Published / approved posts — study these for voice |
| `content_strategy/` | Content strategy documents |
| `abm_profiles/` | ABM target briefings |
| `feedback/` | Client feedback on previous drafts |
| `revisions/` | Before/after revision pairs |
| `past_posts/` | Post history for redundancy checking |
| `scratch/` | Your working area — write plans, drafts, notes here |

Everything here is relevant. Read it all before you write.

**Content strategy from data**: Analyze the client's published posts in \
`context/published-posts/`. Each file includes engagement metrics (reactions, \
comments, reposts, engagement score, outlier flags). Identify which topics, \
formats, angles, and hooks drive the strongest engagement. Use these patterns \
as your de facto content strategy — write more of what resonates, less of \
what falls flat. If no `content_strategy/` document exists, intuit one \
entirely from the engagement data — you have everything you need. If a \
strategy document does exist, treat it as an intent layer (e.g. pivots, \
ABM targets, compliance) that can override the data, but default to what \
the numbers show.

**Critical**: Check `context/draft-posts/` and `past_posts/` before writing. \
Never duplicate an idea that already exists.

## Tools

- `list_directory` / `read_file` / `search_files` — explore the workspace
- `write_file` / `edit_file` — write scratch notes, draft posts, content plans
- `web_search` — search the web (Parallel API — returns ranked excerpts)
- `web_research` — deep research on a topic (Parallel API — synthesized analysis)
- `fetch_url` — extract content from a URL
- `bash` — run shell commands (curl, jq, scripts, etc.)
- `subagent` — spawn a fast, cheap sub-agent for focused research tasks
- `write_result` — submit your final posts (ends the session)

## Process

1. Read everything in the workspace before writing anything
2. Write a content plan to `scratch/plan.md`
3. Draft posts to `scratch/drafts/` — one file per post
4. Re-read your drafts. Apply the magic moment test. Revise with `edit_file`
5. When every post passes, call `write_result` with the final JSON

## Output

When you're done, call the `write_result` tool with a JSON string:

```json
{{{{
  "posts": [
    {{{{
      "hook": "First sentence or opening line (required, <=200 chars)",
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

## Hard Constraints

- LinkedIn hard limit: 3,000 characters per post. Aim for 1,300-3,000.
- No markdown formatting in posts (no #, **, etc.)
- No emojis unless the client's published posts use them
- Banned phrases: "game-changer", "deep dive", "let that sink in", \
"here's the thing", "in today's [anything]", "buckle up", "the reality is"

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
        resp = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
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
    """Stage a virtual workspace matching jacquard's /workspace/ layout."""
    workspace = Path(tempfile.mkdtemp(prefix="stelle_"))
    memory = P.memory_dir(company_keyword)

    # Transcripts: symlink as transcripts/ with primary.md alias
    transcripts_src = memory / "transcripts"
    if transcripts_src.exists():
        transcripts_dst = workspace / "transcripts"
        transcripts_dst.mkdir()
        # History = all transcripts
        history_dst = transcripts_dst / "history"
        os.symlink(transcripts_src.resolve(), history_dst)
        # Primary = latest transcript (or first found)
        latest = transcripts_src / "latest.txt"
        if latest.exists():
            os.symlink(latest.resolve(), transcripts_dst / "primary.md")
        else:
            txts = sorted(transcripts_src.glob("*.txt"))
            if txts:
                os.symlink(txts[-1].resolve(), transcripts_dst / "primary.md")

    # Direct symlinks for flat memory dirs
    for subdir in ("accepted", "feedback", "revisions",
                    "abm_profiles", "content_strategy", "past_posts"):
        src = memory / subdir
        dst = workspace / subdir
        if src.exists():
            os.symlink(src.resolve(), dst)

    # Notes directory (Gap 3)
    notes_src = memory / "notes"
    if notes_src.exists():
        os.symlink(notes_src.resolve(), workspace / "notes")
    else:
        (workspace / "notes").mkdir()

    # context/ hierarchy matching jacquard (Gap 10)
    ctx = workspace / "context"
    ctx.mkdir()

    # context/user/ — LinkedIn profile
    user_dir = ctx / "user"
    user_dir.mkdir()
    profile_summary = _fetch_linkedin_profile(company_keyword)
    if profile_summary:
        (user_dir / "profile-linkedin.md").write_text(profile_summary, encoding="utf-8")

    # context/org/ — company context (content strategy also goes here)
    org_dir = ctx / "org"
    org_dir.mkdir()
    cs_src = memory / "content_strategy"
    if cs_src.exists():
        for f in sorted(cs_src.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                os.symlink(f.resolve(), org_dir / f.name)

    # context/published-posts/ — from Supabase
    pub_posts, published_dates = _fetch_published_posts(company_keyword)
    pub_dir = ctx / "published-posts"
    pub_dir.mkdir()
    for post in pub_posts:
        (pub_dir / post["filename"]).write_text(post["content"], encoding="utf-8")

    # context/draft-posts/ — from Ordinal (Gap 11: excludes published dates)
    drafts = _fetch_ordinal_drafts(company_keyword, exclude_dates=published_dates)
    draft_dir = ctx / "draft-posts"
    draft_dir.mkdir()
    for d in drafts:
        (draft_dir / d["filename"]).write_text(d["content"], encoding="utf-8")

    # context/research/ — from Supabase parallel_research_results (Gap 8)
    research_files = _fetch_research_files(company_keyword)
    research_dir = ctx / "research"
    research_dir.mkdir()
    for rf in research_files:
        (research_dir / rf["filename"]).write_text(rf["content"], encoding="utf-8")

    # scratch/ and output/ directories
    scratch_dir = workspace / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "drafts").mkdir()
    (workspace / "output").mkdir()

    logger.info(
        "[Stelle] Workspace staged at %s (%d published posts, %d drafts, %d research files)",
        workspace, len(pub_posts), len(drafts), len(research_files),
    )

    return workspace


# ---------------------------------------------------------------------------
# Dynamic directives
# ---------------------------------------------------------------------------

def _build_dynamic_directives(company_keyword: str) -> str:
    sections = []

    cs_dir = P.content_strategy_dir(company_keyword)
    if cs_dir.exists():
        for f in sorted(cs_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md"):
                sections.append(f"## Content Strategy\n\n{f.read_text(encoding='utf-8', errors='replace')}")

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
# Pi workspace helpers
# ---------------------------------------------------------------------------

_WEB_SEARCH_SCRIPT = '''\
#!/usr/bin/env python3
"""Search the web via Parallel API. Usage: python3 web_search.py "query" """
import json, os, sys
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python3 web_search.py \\"your query\\""); return
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
        for r in resp.json().get("results", []):
            print(f"Title: {r.get('title', 'N/A')}")
            print(f"URL: {r.get('url', '')}")
            d = r.get("publish_date")
            if d: print(f"Date: {d}")
            for exc in r.get("excerpts", [])[:2]:
                print(f"  {exc[:500]}")
            print()
    except Exception as e:
        print(f"Search error: {e}")

if __name__ == "__main__":
    main()
'''

_FETCH_URL_SCRIPT = '''\
#!/usr/bin/env python3
"""Extract content from a URL via Parallel API. Usage: python3 fetch_url.py "url" """
import json, os, sys
try:
    import httpx
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

def main():
    url = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not url:
        print("Usage: python3 fetch_url.py \\"https://example.com\\""); return
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
                print(content[:12000]); return
        except Exception:
            pass
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15.0)
        print(resp.text[:12000])
    except Exception as e:
        print(f"Fetch error: {e}")

if __name__ == "__main__":
    main()
'''


def _write_tool_scripts(workspace_root: Path) -> None:
    """Write helper Python scripts for web search and URL extraction."""
    tools_dir = workspace_root / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "web_search.py").write_text(_WEB_SEARCH_SCRIPT, encoding="utf-8")
    (tools_dir / "fetch_url.py").write_text(_FETCH_URL_SCRIPT, encoding="utf-8")


def _write_agents_md(workspace_root: Path, company_keyword: str) -> None:
    """Write AGENTS.md to workspace root for Pi to discover."""
    directives = _build_dynamic_directives(company_keyword)
    content = _PI_AGENTS_TEMPLATE.format(dynamic_directives=directives)
    (workspace_root / "AGENTS.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pi-based agentic loop (primary — matches Jacquard architecture)
# ---------------------------------------------------------------------------

def _run_pi_agent(
    workspace_root: Path,
    user_prompt: str,
    company_keyword: str,
    trace: Any = None,
) -> tuple[dict | None, list[dict]]:
    """Run the ghostwriter via Pi CLI with automatic context compaction."""
    session_log: list[dict[str, Any]] = []
    session_start = time.time()

    _write_agents_md(workspace_root, company_keyword)
    _write_tool_scripts(workspace_root)

    session_dir = P.memory_dir(company_keyword) / ".pi-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    has_sessions = any(session_dir.glob("*.jsonl"))

    pi_cmd = [
        "pi",
        "--mode", "json",
        "-p",
        "--provider", "anthropic",
        "--model", "claude-opus-4-6",
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

    try:
        proc = subprocess.run(
            pi_cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=str(workspace_root),
            env=env,
            timeout=pi_timeout,
        )
        stdout_raw = proc.stdout or ""
        stderr_output = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        logger.error("[Stelle/Pi] Pi timed out after %ds", pi_timeout)
        print(f"[Stelle] Pi timed out after {pi_timeout}s")
        session_log.append({"type": "timeout", "timeout_seconds": pi_timeout})
        stdout_raw = ""
        stderr_output = ""
        exit_code = -1
    except FileNotFoundError:
        logger.error("[Stelle/Pi] Pi CLI not found — is it installed?")
        return None, session_log

    events_seen = 0
    compaction_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for line in stdout_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[Stelle/Pi] Non-JSON line: %s", line[:200])
            continue

        events_seen += 1
        etype = event.get("type", "")

        if etype == "message_update":
            ae = event.get("assistantMessageEvent", {})
            ae_type = ae.get("type", "")
            msg = event.get("message", ae.get("message", {}))

            if ae_type.startswith("toolcall"):
                for block in msg.get("content", []):
                    if block.get("type") == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments", {})
                        summary = ""
                        if isinstance(args, dict):
                            summary = args.get("path", args.get("command", str(args)))[:80]
                        logger.info("[Stelle/Pi] tool: %s(%s)", name, summary)

            usage = msg.get("usage", {})
            if usage:
                total_input_tokens = max(total_input_tokens, usage.get("input", 0))
                total_output_tokens += usage.get("output", 0)
                cost = usage.get("cost", {})
                if isinstance(cost, dict):
                    total_cost = max(total_cost, cost.get("total", 0))

        elif etype == "turn_end":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            if usage:
                cost_info = usage.get("cost", {})
                logger.info(
                    "[Stelle/Pi] turn end — in=%d out=%d cache_read=%d cost=$%.4f",
                    usage.get("input", 0), usage.get("output", 0),
                    usage.get("cacheRead", 0),
                    cost_info.get("total", 0) if isinstance(cost_info, dict) else 0,
                )

        elif etype == "auto_compaction_start":
            compaction_count += 1
            logger.info("[Stelle/Pi] Context compaction #%d", compaction_count)

        elif etype == "auto_retry_start":
            logger.info("[Stelle/Pi] Retry %s/%s...", event.get("attempt", "?"), event.get("maxAttempts", "?"))

        elif etype == "error":
            logger.error("[Stelle/Pi] Error: %s", event.get("message", str(event))[:300])

        session_log.append({
            "type": "pi_event",
            "event_type": etype,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": str(event)[:500],
        })

        if trace:
            try:
                if etype in ("message_update", "turn_end", "auto_compaction_start", "error"):
                    trace.event(name=f"pi/{etype}", metadata={"event": str(event)[:500]})
            except Exception:
                pass

    if stdout_raw:
        try:
            jsonl_out.write_text(stdout_raw, encoding="utf-8")
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


def _extract_result_from_scratch(workspace_root: Path) -> dict | None:
    """Fallback: reconstruct result JSON from individual draft files in scratch/drafts/."""
    drafts_dir = workspace_root / "scratch" / "drafts"
    if not drafts_dir.exists():
        return None

    posts = []
    for f in sorted(drafts_dir.iterdir()):
        if not f.is_file() or f.suffix not in (".md", ".txt"):
            continue
        text = f.read_text(encoding="utf-8").strip()
        if not text or len(text) < 100:
            continue

        lines = text.split("\n")
        hook = lines[0].lstrip("#").strip()[:200] if lines else "Untitled"
        posts.append({
            "hook": hook,
            "text": text,
            "origin": f"Extracted from {f.name}",
            "citations": [],
        })

    if posts:
        logger.info("[Stelle/Pi] Reconstructed %d posts from scratch/drafts/", len(posts))
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
    trace: Any = None,
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

        # Langfuse generation logging (Gap 12)
        if trace:
            try:
                trace.generation(
                    name=f"turn-{turn}",
                    model=response.model,
                    usage={
                        "input": usage["input_tokens"],
                        "output": usage["output_tokens"],
                    },
                    metadata={
                        "stop_reason": response.stop_reason,
                        "tool_calls": [tc["name"] for tc in tool_calls] if tool_calls else None,
                    },
                )
            except Exception:
                pass

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

def _generate_why_post(post_text: str, origin: str, client_name: str) -> str:
    """Generate a concise explanation of why this post should be published."""
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a content strategist explaining to a client why a specific "
                    f"LinkedIn post is worth publishing.\n\n"
                    f"Client: {client_name}\n"
                    f"Post origin: {origin}\n\n"
                    f"Post:\n{post_text}\n\n"
                    f"Write 2-3 sentences explaining why this post is valuable for the "
                    f"client's LinkedIn presence. Cover: what makes it compelling to their "
                    f"audience, what strategic purpose it serves (thought leadership, "
                    f"engagement, ABM, credibility), and why NOW is a good time to post it. "
                    f"Be specific — no generic praise. Write in second person ('you')."
                ),
            }],
        )
        return resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        logger.warning("[Stelle] Why-post generation failed: %s", e)
        return ""


def _generate_image_suggestion(post_text: str, hook: str) -> str:
    """Generate a concrete image suggestion for the post."""
    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a visual content strategist for LinkedIn.\n\n"
                    f"Post hook: {hook}\n\n"
                    f"Post:\n{post_text}\n\n"
                    f"Suggest ONE specific image that would complement this post on LinkedIn. "
                    f"Be concrete: describe the visual (photo, graphic, screenshot, diagram), "
                    f"its composition, colors, and what text overlay (if any) it should have. "
                    f"The image should stop the scroll and reinforce the post's message. "
                    f"If the post is better without an image, say 'Text-only recommended' "
                    f"and briefly explain why. Reply with just the suggestion, no preamble."
                ),
            }],
        )
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

    from permansor_terrae import PermansorTerrae
    permansor = PermansorTerrae()
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

        output_lines.append(f"## Post {i}: {hook}\n")
        output_lines.append(f"**Origin:** {origin}\n")
        output_lines.append(f"**Characters:** {len(text)}\n")

        if citations:
            output_lines.append("**Citations:**")
            for c in citations:
                output_lines.append(f"- {c.get('claim', '')}: {c.get('source', '')}")
            output_lines.append("")

        output_lines.append("### Draft\n")
        output_lines.append(text + "\n")

        print(f"[Stelle] Fact-checking post {i}/{len(posts)}: {hook[:50]}...")
        corrected = text
        try:
            fc_report = permansor.fact_check_post(company_keyword, text)
            corrected_match = re.search(
                r"\[CORRECTED POST\]\s*\n([\s\S]+?)(?:\n##|\Z)", fc_report
            )
            if corrected_match:
                corrected = corrected_match.group(1).strip()
                fc_header = fc_report[:fc_report.index("[CORRECTED POST]")].strip()
            else:
                fc_header = fc_report.strip()

            output_lines.append(f"### Fact-Check Report\n\n{fc_header}\n")
            output_lines.append(f"### Final Post\n\n{corrected}\n")
        except Exception as e:
            logger.warning("[Stelle] Fact-check failed for post %d: %s", i, e)
            output_lines.append(f"### Fact-Check\n\nFact-check failed: {e}\n")
            output_lines.append(f"### Final Post\n\n{corrected}\n")

        print(f"[Stelle] Generating why-post + image for post {i}/{len(posts)}...")
        why_post = _generate_why_post(corrected, origin, client_name)
        if why_post:
            output_lines.append(f"### Why Post\n\n{why_post}\n")

        img_sug = _generate_image_suggestion(corrected, hook)
        if img_sug:
            output_lines.append(f"### Image Suggestion\n\n{img_sug}\n")
        elif image_suggestion:
            output_lines.append(f"### Image Suggestion\n\n{image_suggestion}\n")

        output_lines.append("---\n")

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
) -> str:
    print(f"[Stelle] Starting agentic ghostwriter for {client_name}...")

    P.ensure_dirs(company_keyword)

    langfuse = _get_langfuse()
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name=f"stelle/{client_name}",
                metadata={
                    "company_keyword": company_keyword,
                    "num_posts": num_posts,
                    "runner": "pi" if _PI_AVAILABLE else "direct",
                },
            )
            logger.info("[Stelle] Langfuse trace: %s/traces/%s", LANGFUSE_BASE_URL, trace.id)
        except Exception as e:
            logger.warning("[Stelle] Langfuse trace creation failed: %s", e)

    print("[Stelle] Setting up workspace...")
    workspace_root = _setup_workspace(company_keyword)

    user_prompt = (
        f"Write {num_posts} LinkedIn posts for {client_name}. "
        f"The transcripts are from content interviews — conversations designed "
        f"to surface post material. Mine them for everything worth writing about."
    )

    if _PI_AVAILABLE:
        print(f"[Stelle] Using Pi agent (context compaction enabled)...")
        result, session_log = _run_pi_agent(workspace_root, user_prompt, company_keyword, trace=trace)
    else:
        logger.warning("[Stelle] Pi not installed — falling back to direct API loop (higher token usage)")
        print(f"[Stelle] Pi not found. Using direct API loop (max {MAX_AGENT_TURNS} turns)...")
        directives = _build_dynamic_directives(company_keyword)
        system_prompt = _DIRECT_SYSTEM_TEMPLATE.format(dynamic_directives=directives)
        result, session_log = _run_agent_loop(system_prompt, user_prompt, workspace_root, trace=trace)

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
        if trace:
            trace.update(output={"status": "failed", "reason": "no result"})
        if langfuse:
            langfuse.flush()
        return output_filepath

    passed, val_errors, val_warnings = _validate_output(result)
    if not passed:
        logger.warning("[Stelle] Final output validation failed: %s", val_errors)

    post_count = len(result.get("posts", []))
    print(f"[Stelle] Agent produced {post_count} posts. Running fact-check...")

    output_path = _process_result(result, client_name, company_keyword, output_filepath)

    if trace:
        trace.update(output={
            "status": "completed",
            "posts_count": post_count,
            "validation_passed": passed,
            "validation_errors": val_errors,
            "validation_warnings": val_warnings,
        })
    if langfuse:
        langfuse.flush()

    return output_path
