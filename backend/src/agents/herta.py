"""
herta.py — Content Strategy Agent
======================================
Wraps Claude Opus (claude-opus-4-6) with three self-research tools:

  1. ordinal_list_profiles      — resolves a company keyword → Ordinal profile UUID
  2. ordinal_follower_analytics — LinkedIn follower count + growth history
  3. ordinal_post_analytics     — recent post impressions, engagement, frequency
  4. google_search              — Gemini-grounded web search for ICP / industry research

The agent gathers all context it needs autonomously from these tools and the
client's transcript files in memory/{company}/. No manual input required beyond
the company keyword.

Two modes of use
----------------
1. Programmatic (called by amphoreus.py):
       from herta import run_programmatic
       run_programmatic("hensley-biostats", output_callback=fn)

2. CLI (standalone):
       python herta.py
       python herta.py --client hensley-biostats

Requirements
------------
    pip install anthropic google-genai requests

Environment Variables
---------------------
    ANTHROPIC_API_KEY   — Required.
    GEMINI_API_KEY      — Required for Google Search grounding.
    ORDINAL_API_KEY     — Fallback if memory/ordinal_auth_rows.csv has no entry for client.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

# Load .env from the script's own directory (for standalone CLI use)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Anthropic ──────────────────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    sys.exit("❌  anthropic not installed. Run: pip install anthropic")

# ── Requests (Ordinal REST calls) ──────────────────────────────────────────────
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── Google GenAI (Gemini Search) ───────────────────────────────────────────────
try:
    from google import genai as _google_genai
    from google.genai import types as _genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Vortex paths ───────────────────────────────────────────────────────────────
try:
    from backend.src.db import vortex as P
    MEMORY_ROOT      = P.MEMORY_ROOT
    _AUTH_CSV_PATH   = P.ordinal_auth_csv()
except Exception:
    MEMORY_ROOT    = Path("memory")
    _AUTH_CSV_PATH = MEMORY_ROOT / "ordinal_auth_rows.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_MODEL     = "claude-sonnet-4-6"
HTML_MODEL       = "claude-sonnet-4-6"
# Gemini model tried in order — first success wins
GEMINI_MODELS    = [
    "gemini-3.1-flash",     # newest; may or may not be available
    "gemini-2.5-flash",     # fallback
    "gemini-2.0-flash",     # widely available baseline
    "gemini-1.5-flash",     # last-resort legacy
]
MAX_TOKENS       = 8192
ORDINAL_BASE_URL = "https://app.tryordinal.com/api/v1"
SCRIPT_DIR       = Path(__file__).parent
PLAYBOOK_PATH    = SCRIPT_DIR / "content-strategy-playbook.md"

OutputCallback = Callable[[str], None]


# ══════════════════════════════════════════════════════════════════════════════
# Ordinal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ordinal_api_key(client_name: str) -> str:
    """
    Look up the Ordinal API key for client_name from the auth CSV.
    Falls back to the ORDINAL_API_KEY env var if no CSV entry exists.
    """
    if _AUTH_CSV_PATH.exists():
        try:
            with open(_AUTH_CSV_PATH, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    c_id  = row.get("company_id", "").strip().lower()
                    slug  = row.get("provider_org_slug", "").strip().lower()
                    target = client_name.strip().lower()
                    if c_id == target or slug == target:
                        key = row.get("api_key", "").strip()
                        if key:
                            return key
        except Exception:
            pass
    return os.environ.get("ORDINAL_API_KEY", "")


def _ordinal_headers(client_name: str) -> dict:
    key = _ordinal_api_key(client_name)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _ordinal_list_profiles(client_name: str) -> str:
    """
    Call GET /profiles/scheduling and return JSON string of all scheduling
    profiles in the workspace. The agent uses this to identify the correct
    profileId (UUID) for the client before calling analytics endpoints.
    """
    if not REQUESTS_AVAILABLE:
        return "[requests not installed — pip install requests]"

    api_key = _ordinal_api_key(client_name)
    if not api_key:
        return f"[No Ordinal API key found for '{client_name}'. Set ORDINAL_API_KEY or add to ordinal_auth_rows.csv]"

    try:
        resp = _requests.get(
            f"{ORDINAL_BASE_URL}/profiles/scheduling",
            headers=_ordinal_headers(client_name),
            timeout=15,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except Exception as exc:
        return f"[Ordinal list_profiles error: {exc}]"


def _ordinal_follower_analytics(profile_id: str, client_name: str,
                                 start_date: str = "", end_date: str = "") -> str:
    """
    Call GET /analytics/linkedin/{profileId}/followers.
    Returns follower count history as a JSON string.
    """
    if not REQUESTS_AVAILABLE:
        return "[requests not installed — pip install requests]"

    api_key = _ordinal_api_key(client_name)
    if not api_key:
        return f"[No Ordinal API key found for '{client_name}']"

    if not start_date:
        start_date = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    try:
        resp = _requests.get(
            f"{ORDINAL_BASE_URL}/analytics/linkedin/{profile_id}/followers",
            headers=_ordinal_headers(client_name),
            params={"startDate": start_date, "endDate": end_date},
            timeout=15,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except Exception as exc:
        return f"[Ordinal follower_analytics error: {exc}]"


def _ordinal_post_analytics(profile_id: str, client_name: str,
                             start_date: str = "", end_date: str = "") -> str:
    """
    Call GET /analytics/linkedin/{profileId}/posts.
    Returns post-level impressions, engagement, and frequency data.
    """
    if not REQUESTS_AVAILABLE:
        return "[requests not installed — pip install requests]"

    api_key = _ordinal_api_key(client_name)
    if not api_key:
        return f"[No Ordinal API key found for '{client_name}']"

    if not start_date:
        start_date = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    try:
        resp = _requests.get(
            f"{ORDINAL_BASE_URL}/analytics/linkedin/{profile_id}/posts",
            headers=_ordinal_headers(client_name),
            params={"startDate": start_date, "endDate": end_date},
            timeout=15,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except Exception as exc:
        return f"[Ordinal post_analytics error: {exc}]"


# ══════════════════════════════════════════════════════════════════════════════
# Gemini Google Search
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_search(query: str) -> str:
    """
    Run a Google Search via Gemini grounding; return the synthesised answer.
    Tries each model in GEMINI_MODELS in order, stopping at the first success.
    """
    if not GEMINI_AVAILABLE:
        return f"[google-genai not installed — pip install google-genai]\nQuery: {query}"

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return f"[GEMINI_API_KEY not set — add it to .env or export GEMINI_API_KEY=...]\nQuery: {query}"

    client = _google_genai.Client(api_key=api_key)
    last_exc: Exception | None = None

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=query,
                config=_genai_types.GenerateContentConfig(
                    tools=[_genai_types.Tool(google_search=_genai_types.GoogleSearch())],
                    temperature=0.2,
                ),
            )
            return response.text or "[Gemini returned an empty response]"
        except Exception as exc:
            last_exc = exc
            # Try the next model in the fallback chain
            continue

    return (
        f"[Gemini search failed on all models ({', '.join(GEMINI_MODELS)}). "
        f"Last error: {last_exc}]\nQuery: {query}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool schemas for Claude
# ══════════════════════════════════════════════════════════════════════════════

_TOOLS = [
    {
        "name": "ordinal_list_profiles",
        "description": (
            "List all Ordinal scheduling profiles in the client's workspace. "
            "Call this FIRST to find the correct profileId (UUID) for the client — "
            "you need it before calling follower or post analytics. "
            "Match the returned profile names against the company keyword to identify "
            "the right profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "ordinal_follower_analytics",
        "description": (
            "Fetch LinkedIn follower count history for the client from Ordinal. "
            "Returns total follower count over time, letting you determine: "
            "current follower count, growth trajectory, and audience size context. "
            "Requires profileId from ordinal_list_profiles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "string",
                    "description": "The Ordinal profile UUID (from ordinal_list_profiles).",
                },
                "start_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Defaults to 90 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Defaults to today.",
                },
            },
            "required": ["profile_id"],
        },
    },
    {
        "name": "ordinal_post_analytics",
        "description": (
            "Fetch LinkedIn post analytics for the client from Ordinal. "
            "Returns per-post impressions, engagement rates, likes, comments, and "
            "publish dates — letting you determine posting frequency, content performance, "
            "which post types resonate, and whether the client is active or sporadic. "
            "Requires profileId from ordinal_list_profiles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "string",
                    "description": "The Ordinal profile UUID (from ordinal_list_profiles).",
                },
                "start_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Defaults to 90 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Defaults to today.",
                },
            },
            "required": ["profile_id"],
        },
    },
    {
        "name": "google_search",
        "description": (
            "Search the web via Google (powered by Gemini) to research the client's "
            "ICP pain points, industry trends, competitor LinkedIn content, and "
            "engagement benchmarks. Use after reviewing Ordinal data to fill knowledge gaps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific, targeted search query.",
                }
            },
            "required": ["query"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# File helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_playbook() -> str:
    for candidate in (PLAYBOOK_PATH, Path("content-strategy-playbook.md")):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return (
        "[Playbook not found — apply general LinkedIn content strategy best practices: "
        "TOFU/MOFU/BOFU framework, pipeline vs. brand north star, storytelling-based "
        "thought leadership over generic thought leadership.]"
    )


_MAX_FILE_CHARS = 120_000


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf."""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        return f"[PDF not readable — install pymupdf: pip install pymupdf]"
    except Exception as exc:
        return f"[Could not extract PDF text: {exc}]"


def _load_client_files(client_name: str) -> dict[str, str]:
    client_dir = MEMORY_ROOT / client_name
    results: dict[str, str] = {}
    for directory in (client_dir / "transcripts", client_dir / "references", client_dir):
        if not directory.exists():
            continue
        for pattern in ("*.txt", "*.md", "*.pdf"):
            for fp in sorted(directory.glob(pattern)):
                rel = str(fp.relative_to(client_dir))
                if rel in results:
                    continue
                try:
                    if fp.suffix.lower() == ".pdf":
                        text = _read_pdf(fp)
                    else:
                        text = fp.read_text(encoding="utf-8", errors="replace")
                    results[rel] = text[:_MAX_FILE_CHARS]
                except Exception as exc:
                    results[rel] = f"[Could not read: {exc}]"
    return results


def _format_client_files(files: dict[str, str]) -> str:
    if not files:
        return "_No client files found. Rely on Ordinal data and Google Search._"
    parts = [f"### {name}\n\n{content.strip()}" for name, content in files.items()]
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_TEMPLATE = """\
You are a senior LinkedIn content strategist embedded in a post-generation pipeline.

Your job is to produce a complete, personalized LinkedIn content strategy for a client
across three time horizons: SHORT-TERM (Month 1), MEDIUM-TERM (Months 2–3), and
LONG-TERM (Months 4–6+).

You have four tools available. USE THEM — do not skip the research phase.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY RESEARCH SEQUENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 — ordinal_list_profiles
  Call this immediately to get the client's Ordinal profile UUID.
  Match the profile name to the company keyword. If no Ordinal data is available,
  skip Steps 2–3 and note the gap.

Step 2 — ordinal_follower_analytics (requires profileId from Step 1)
  Fetch the last 90 days of follower data. Derive:
  • Current follower count
  • Growth trajectory (accelerating / flat / declining)
  • Absolute audience size context (small / mid / large)

Step 3 — ordinal_post_analytics (requires profileId from Step 1)
  Fetch the last 90 days of post data. Derive:
  • Posting frequency (daily / weekly / sporadic / inactive)
  • Average impressions and engagement rates
  • Which post types performed best
  • Whether ICP-relevant content is already present or absent

Step 4 — google_search (run at least 3–5 queries)
  Use what you learned from the transcripts and Ordinal data to search for:
  • The client's ICP role: day-to-day pain points, KPIs, frustrations
  • Industry trends and hot-button topics the client can credibly speak to
  • Peer/competitor LinkedIn posts that resonate with that ICP
  • Engagement benchmarks for the client's content category

After completing research, DERIVE the following autonomously — do NOT ask the user:
  • Audience bucket (A = has following but wrong ICP / B = ICP already present / C = inactive)
  • ICP profile — from transcripts + search
  • Voice fingerprint — from transcripts + any existing posts

The PRIMARY GOAL (pipeline / brand / mixed) is provided directly by the operator in the
user message and must be treated as a confirmed directive — do not infer or second-guess it.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTENT STRATEGY PLAYBOOK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{playbook}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LINKEDIN POST QUALITY STANDARDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Great Posts
  • One Main Idea (OMI): every element serves one intellectual point
  • Meaty, substantive, opinionated — not announcements or diary entries
  • Cohesive body: a skimmer follows the narrative arc
  • No outright selling unless the OMI is about the poster's own company
  • Formatting (bullets, →) used to accentuate a point, not fill space
  • Rarely has a CTA unless it is a genuine lead magnet

Hook Standards (<200 chars, usually)
  • Piques curiosity AND hints at the OMI
  • Can use: impressive numbers, subverted assumptions, ICP title callout, story climax

Funnel Classification (only Great Posts qualify)
  • TOFU — educational, broad authority-building
  • MOFU — must (1) name specific ICP title in hook AND (2) be primarily about that role
  • BOFU/ABM — tags specific companies/people positively; list ABM is highest leverage

AI Writing Anti-Patterns — NEVER use
  • "It's not just X; it's Y." / "Not A. But B."
  • Casual hyperbole ("This one thing changes everything")
  • Manufactured drama ("The quiet part no one says aloud…")
  • "But here's the thing:" / excessive em dashes
  • Short dramatic lead-up questions ("The number that matters? Their churn rate:")
  • "What do you think? Let me know in the comments."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT (clean Markdown)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Content Strategy: [Client Name]
_Generated: [date]_

## Client Intelligence Summary
- ICP Profile (title, company type, pain points, KPIs, what they rarely hear)
- Voice Fingerprint (tone, vocabulary, story patterns, what to avoid)
- Bucket Classification (A / B / C + rationale drawn from Ordinal data)
- Goal Alignment (pipeline / brand / mixed + how you inferred it)
- LinkedIn Snapshot (follower count, growth trend, posting frequency, avg engagement)
- Key Research Findings

## Short-Term Strategy — Month 1
- Objective
- Content Mix (TOFU / MOFU / BOFU %)
- Priority Post Types (with rationale)
- 5 Specific Post Ideas (OMI · hook direction · funnel position · post type)
- Key Performance Signals
- Expectation-Setting Language for the client

## Medium-Term Strategy — Months 2–3
- Objective · Content Mix · Priority Post Types
- 5 Specific Post Ideas
- Key Performance Signals · Potential Pivots

## Long-Term Strategy — Months 4–6+
- Objective · Content Mix (steady-state) · Priority Post Types
- 5 Specific Post Ideas
- Success Definition · Iteration Framework

## Operational Notes for Post-Generation Agents
Voice rules, ICP framing preferences, topics to lean into / avoid, hook style guidance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _build_system_prompt(playbook: str) -> str:
    return _SYSTEM_TEMPLATE.format(playbook=playbook)


def _build_user_message(
    client_name: str,
    files_str: str,
    primary_goal: str = "",
    follower_count: str = "",
    icp_pct: str = "",
    extra_context: str = "",
) -> str:
    lines = [f"**Client / Company Keyword:** {client_name}"]
    if primary_goal:
        lines.append(
            f"**Primary Goal (operator-confirmed):** {primary_goal} — "
            "treat this as a hard directive, not something to infer or override."
        )
    if follower_count:
        lines.append(
            f"**LinkedIn Follower Count (operator-provided):** {follower_count} — "
            "use this exact figure; do not attempt to look it up via Ordinal."
        )
    if icp_pct:
        lines.append(
            f"**% ICP Among Followers (operator-provided):** {icp_pct}% — "
            "use this exact figure; do not attempt to infer or override it."
        )
    if extra_context:
        lines.append(f"**Additional Context from operator:** {extra_context}")
    header = "\n".join(lines)

    # Adjust the research sequence hint based on what's already provided
    if follower_count:
        research_hint = (
            "Begin research with: ordinal_list_profiles → ordinal_post_analytics → "
            "google_search (×3–5). Follower count and ICP % have been provided above — "
            "skip ordinal_follower_analytics for those figures."
        )
    else:
        research_hint = (
            "Begin with the mandatory research sequence: "
            "ordinal_list_profiles → ordinal_follower_analytics → ordinal_post_analytics → "
            "google_search (×3–5). Derive all client context from these sources, then write "
            "the full strategy document."
        )

    # Strategy brief context — injected as a dedicated section so Herta can
    # reason over the learned engagement data alongside its own live research.
    # Includes the full brief plus raw topic_transitions.json and
    # causal_dimensions.json appendices when they exist.
    strategy_ctx = ""
    try:
        from backend.src.utils.strategy_brief import build_herta_strategy_context
        strategy_ctx = build_herta_strategy_context(client_name)
    except Exception:
        pass

    base = (
        f"{header}\n\n---\n\n"
        f"## Client Source Files (transcripts + memory)\n\n{files_str}\n\n---\n\n"
    )
    if strategy_ctx:
        base += strategy_ctx + "\n---\n\n"
    base += research_hint
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Tool dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def _dispatch_tool(tool_name: str, tool_input: dict,
                   client_name: str, output_cb: OutputCallback) -> str:
    if tool_name == "ordinal_list_profiles":
        output_cb(f"\n\n📋 **Ordinal:** listing profiles…\n\n")
        return _ordinal_list_profiles(client_name)

    elif tool_name == "ordinal_follower_analytics":
        profile_id = tool_input.get("profile_id", "")
        start      = tool_input.get("start_date", "")
        end        = tool_input.get("end_date", "")
        output_cb(f"\n\n📈 **Ordinal:** fetching follower analytics…\n\n")
        return _ordinal_follower_analytics(profile_id, client_name, start, end)

    elif tool_name == "ordinal_post_analytics":
        profile_id = tool_input.get("profile_id", "")
        start      = tool_input.get("start_date", "")
        end        = tool_input.get("end_date", "")
        output_cb(f"\n\n📊 **Ordinal:** fetching post analytics…\n\n")
        return _ordinal_post_analytics(profile_id, client_name, start, end)

    elif tool_name == "google_search":
        query = tool_input.get("query", "")
        output_cb(f"\n\n🔍 **Search:** {query[:80]}{'…' if len(query) > 80 else ''}\n\n")
        return _gemini_search(query)

    else:
        return f"[Unknown tool '{tool_name}']"


# ══════════════════════════════════════════════════════════════════════════════
# Streaming agentic loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_agent_streaming(system_prompt: str, user_message: str,
                         client_name: str, output_cb: OutputCallback) -> str:
    """
    Token-streaming Claude ↔ tools loop.
    Calls output_cb(chunk) for every text token and every tool status line.
    Safe to run on a background thread — output_cb must be thread-safe.
    """
    client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages: list[dict] = [{"role": "user", "content": user_message}]
    accumulated: list[str] = []

    while True:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=_TOOLS,
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                output_cb(chunk)
                accumulated.append(chunk)
            final = stream.get_final_message()

        messages.append({"role": "assistant", "content": final.content})

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if final.stop_reason == "end_turn" or not tool_uses:
            break

        tool_results: list[dict] = []
        for tu in tool_uses:
            result = _dispatch_tool(tu.name, tu.input, client_name, output_cb)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu.id,
                "content":     result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "".join(accumulated)


# ══════════════════════════════════════════════════════════════════════════════
# HTML Report Generation
# ══════════════════════════════════════════════════════════════════════════════

_HTML_SYSTEM_PROMPT = """\
You are a designer creating a clean, presentation-ready one-pager for a client strategy meeting.

Your job is to convert a detailed content strategy document into a minimal, visually clear HTML
slide that can be shown on screen during a presentation. It must be easy to absorb in 60 seconds.

═══════════════════════════════════════════════
CONTENT RULES — what to include and what to cut
═══════════════════════════════════════════════

INCLUDE only:
  1. The client's explicitly stated goal (extract the exact quote or close paraphrase from the
     strategy document — this is what the client said they want to achieve, not the operator's
     pipeline/brand label). Display this prominently at the top as the north star.
  2. Three timeline sections — Short-Term, Medium-Term, Long-Term — each containing:
       • 3–5 punchy bullet points describing the core content themes / strategic moves
       • One sentence on the dominant content type for that period
       • A simple CSS content-split bar showing the TOFU / MOFU / BOFU percentage breakdown
         for that period (derive reasonable percentages from the strategy; use whole numbers
         that sum to 100)
  3. A single "Pipeline Timeline" row: a minimal horizontal milestone strip showing roughly
     when the client should expect to see early pipeline signals, then consistent pipeline,
     using months (e.g. Month 1–2 → Brand awareness; Month 3–4 → First inbound signals;
     Month 5–6 → Consistent pipeline). Keep it to 3–4 milestones.

EXCLUDE everything else:
  • No post idea cards, no hook directions, no OMI scores
  • No operational notes, no appendices
  • No lists of 10+ items
  • No section about "client intelligence" or raw research findings
  • No lengthy paragraphs

═════════════════
DESIGN RULES
═════════════════

Layout & typography
  • Single self-contained HTML file — all CSS in one <style> block, zero external dependencies
  • Font stack: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
  • Max-width 820px, centred, white background, generous padding
  • Body text: 15px / 1.6 line-height; all text #1a1a2a

Masthead (top of page)
  • Client name — large (28px), bold, #1a1a2a
  • Subtitle: "LinkedIn Content Strategy" — 14px, muted (#6b7280)
  • Date — 13px, muted
  • Client goal — displayed in a lightly tinted call-out box directly below the masthead
    (background #f0f6ff, left border 4px solid #1a5a8a, padding 14px 18px, italic, 16px)
    Label it "Their Goal" in small caps above the quote.

Timeline sections — one per period
  • Section accent colours:
      Short-Term  : emerald  #0d6e4f  /  tint #f0faf5
      Medium-Term : navy     #1a5a8a  /  tint #f0f6fc
      Long-Term   : violet   #5c2d91  /  tint #f8f4ff
  • Section header: period label only ("Short-Term", "Medium-Term", "Long-Term") in the accent colour, 18px bold — do NOT include month ranges (e.g. "Month 1" or "Months 2–3") in the header
  • Bullets: concise, max 12 words each; use a custom CSS bullet in the accent colour
  • Content split bar — three labelled segments side by side using flexbox:
      Each segment is a coloured rectangle (height 28px) whose flex-basis equals its percentage,
      with the label (e.g. "TOFU 60%") centred inside in white 12px bold.
      Colours: TOFU #16a34a, MOFU #2563eb, BOFU #7c3aed
      Show the bar below the bullets with a small "Content Mix" label above it.

Pipeline Timeline strip
  • A horizontal flex row of 3–4 milestone boxes
  • Each box: white background, 1px border #e5e7eb, 8px border-radius, centred text
  • Month label in bold above a one-line description
  • Connect boxes with a thin #e5e7eb line (use border-right or a pseudo-element)
  • Section label "When to Expect Results" in small caps, muted, above the strip

General
  • Subtle section separator: 1px solid #f0f0f0, 24px vertical margin
  • No box shadows heavier than 0 1px 4px rgba(0,0,0,0.06)
  • @media print: remove backgrounds from masthead only; keep section tints; page-break-inside avoid
  • NO JavaScript. NO animations. NO images. NO external fonts.

Output ONLY the complete HTML document — nothing before <!DOCTYPE html>, nothing after </html>.
"""

_HTML_USER_TEMPLATE = """\
Convert the following content strategy into the presentation one-pager described in your instructions.

Client name: {client_name}
Date: {date}
Operator-confirmed primary goal type: {primary_goal}

---

{strategy}
"""


def _generate_html_report(
    strategy: str,
    client_name: str,
    output_cb: OutputCallback,
    primary_goal: str = "",
) -> Path | None:
    """
    Use claude-sonnet-4-6 to convert the strategy markdown into a clean
    presentation one-pager. Returns the saved Path, or None on failure.
    """
    if not strategy.strip():
        return None

    output_cb("\n\n📄 Generating HTML presentation…\n")

    try:
        ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        user_msg = _HTML_USER_TEMPLATE.format(
            client_name=client_name,
            date=datetime.now().strftime("%B %d, %Y"),
            primary_goal=primary_goal or "not specified",
            strategy=strategy,
        )
        response = ac.messages.create(
            model=HTML_MODEL,
            max_tokens=8192,
            system=_HTML_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        html_content = response.content[0].text.strip()

        # Ensure it's valid HTML (starts with doctype)
        if "<!DOCTYPE" not in html_content[:200] and "<html" not in html_content[:200]:
            output_cb("   ⚠️  HTML output looks malformed — saving anyway.\n")

        # Save alongside the strategy markdown
        try:
            from backend.src.db import vortex as _P
            out_dir = _P.content_strategy_dir(client_name)
        except ImportError:
            out_dir = MEMORY_ROOT / client_name / "content_strategy"

        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        html_path = out_dir / f"content-strategy-{timestamp}.html"
        html_path.write_text(html_content, encoding="utf-8")
        html_path = html_path.resolve()  # always return an absolute path
        output_cb(f"   Saved HTML report to: {html_path}\n")
        return html_path

    except Exception as exc:
        output_cb(f"   ⚠️  HTML generation failed: {exc}\n")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Public API — programmatic entry point (called by amphoreus.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_programmatic(
    client_name: str,
    output_callback: OutputCallback,
    *,
    primary_goal: str = "",
    follower_count: str = "",
    icp_pct: str = "",
    extra_context: str = "",
    save_output: bool = True,
) -> str:
    """
    Generate a content strategy and stream every token to output_callback.

    Parameters
    ----------
    client_name     : company keyword — matches memory/ folder and Ordinal profile name
    output_callback : called with each text/status chunk; must be thread-safe
    primary_goal    : "pipeline" | "brand" | "mixed" — operator-confirmed hard directive.
    follower_count  : LinkedIn follower count provided by the operator. If given, the
                      agent skips ordinal_follower_analytics for this figure.
    icp_pct         : % of followers who match the ICP, operator-provided. Not derivable
                      from any API — this is a strategic human assessment.
    extra_context   : optional free-form operator notes passed verbatim to the agent
    save_output     : if True, writes timestamped markdown to
                      memory/{client}/content_strategy/
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        output_callback("❌  ANTHROPIC_API_KEY is not set.\n")
        return ""

    if not GEMINI_AVAILABLE:
        output_callback(
            "⚠️  google-genai not installed — Google Search unavailable.\n"
            "    pip install google-genai\n\n"
        )
    elif not os.environ.get("GEMINI_API_KEY"):
        output_callback("⚠️  GEMINI_API_KEY not set — Google Search unavailable.\n\n")

    if not REQUESTS_AVAILABLE:
        output_callback(
            "⚠️  requests not installed — Ordinal analytics unavailable.\n"
            "    pip install requests\n\n"
        )

    # ── Load assets ───────────────────────────────────────────────────────────
    output_callback("📖 Loading playbook…\n")
    playbook = _load_playbook()

    output_callback(f"📁 Loading client files for '{client_name}'…\n")
    client_files = _load_client_files(client_name)
    if client_files:
        output_callback(f"   {len(client_files)} file(s): {', '.join(client_files)}\n\n")
    else:
        output_callback("   No files found — relying on Ordinal + Search.\n\n")

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = _build_system_prompt(playbook)
    user_message  = _build_user_message(
        client_name,
        _format_client_files(client_files),
        primary_goal=primary_goal,
        follower_count=follower_count,
        icp_pct=icp_pct,
        extra_context=extra_context,
    )

    # ── Run agent ─────────────────────────────────────────────────────────────
    output_callback("🚀 Agent starting research…\n\n" + "─" * 60 + "\n\n")
    strategy = _run_agent_streaming(system_prompt, user_message,
                                    client_name, output_callback)
    output_callback("\n\n" + "─" * 60 + "\n✅ Strategy generation complete.\n")

    # ── Save markdown to disk ─────────────────────────────────────────────────
    html_path: Path | None = None
    if save_output and strategy.strip():
        try:
            from backend.src.db import vortex as _P
            out_dir = _P.content_strategy_dir(client_name)
        except ImportError:
            out_dir = MEMORY_ROOT / client_name / "content_strategy"

        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_path  = out_dir / f"content-strategy-{timestamp}.md"
        out_path.write_text(strategy, encoding="utf-8")
        output_callback(f"💾 Markdown saved: {out_path}\n")

        # ── Generate HTML report ──────────────────────────────────────────────
        html_path = _generate_html_report(strategy, client_name, output_callback,
                                           primary_goal=primary_goal)

    return {"strategy": strategy, "html_path": str(html_path) if html_path else ""}


# ══════════════════════════════════════════════════════════════════════════════
# Short-term section extractor (used by amphoreus.py for the save dialog)
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

def extract_short_term_section(strategy: str) -> str:
    """
    Extract the Short-Term Strategy section from a generated strategy document.

    Looks for a heading matching /short.?term/i (##, ###, or **…**) and returns
    everything from that heading up to (but not including) the next same-level
    heading, or the end of the string.

    Returns the extracted section, or the full strategy string if no matching
    heading is found (so the user always has something to edit).
    """
    # Try to match any markdown heading level containing "short" and "term"
    pattern = _re.compile(
        r'((?:#{1,4}\s+[^\n]*short[^\n]*term[^\n]*|#{1,4}\s+[^\n]*term[^\n]*short[^\n]*)\n.*?)(?=\n#{1,4}\s|\Z)',
        _re.IGNORECASE | _re.DOTALL,
    )
    match = pattern.search(strategy)
    if match:
        return match.group(1).strip()

    # Fallback: look for bold heading **Short-Term …** style
    bold_pattern = _re.compile(
        r'(\*\*[^*]*short[^*]*term[^*]*\*\*.*?)(?=\n\*\*[A-Z]|\Z)',
        _re.IGNORECASE | _re.DOTALL,
    )
    bold_match = bold_pattern.search(strategy)
    if bold_match:
        return bold_match.group(1).strip()

    # Nothing found — return full strategy so user can still edit
    return strategy.strip()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"  {prompt}{suffix}: ").strip()
    return raw if raw else default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LinkedIn content strategy (agent researches all context).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python herta.py
              python herta.py --client hensley-biostats
        """),
    )
    parser.add_argument("--client",  default="", help="Client folder / company keyword.")
    parser.add_argument("--context", default="", help="Optional extra context string.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("❌  ANTHROPIC_API_KEY is not set.")

    client_name = args.client or _ask("Client folder name (matches memory/{name})")
    if not client_name:
        sys.exit("Client name is required.")

    print()
    print("  ── Strategic Direction ──")
    print("    1  Pipeline   2  Brand   3  Both (weighted pipeline)")
    goal_raw = _ask("Primary goal [1/2/3]", "1")
    goal = {"1": "pipeline", "2": "brand", "3": "mixed"}.get(goal_raw, "pipeline")

    extra = args.context or _ask("Any extra context? (press Enter to skip)", "")

    result = run_programmatic(
        client_name=client_name,
        output_callback=lambda t: print(t, end="", flush=True),
        primary_goal=goal,
        extra_context=extra,
    )
    if result.get("html_path"):
        print(f"\n🌐 HTML report: {result['html_path']}")


if __name__ == "__main__":
    main()
