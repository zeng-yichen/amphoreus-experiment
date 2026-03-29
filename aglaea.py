"""
Aglaea — Interview briefing generator (Pi-based agentic approach).

Prepares a ghostwriter for their next content interview by generating
questions that extract compelling, novel personal stories from the client.
Delegates the work to Pi (pi.dev CLI) — same architecture as stelle.py.
Falls back to a single Claude message if Pi is not installed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import vortex as P
from stelle import (
    _setup_workspace,
    _write_tool_scripts,
    _build_dynamic_directives,
    _PI_AVAILABLE,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aglaea")

_client = Anthropic()

PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

# ---------------------------------------------------------------------------
# Langfuse observability
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
        logger.info("[Aglaea] langfuse not installed — tracing disabled")
        return None


# ---------------------------------------------------------------------------
# AGENTS.md template — Pi-native system prompt for interview prep
# ---------------------------------------------------------------------------

_BRIEFING_AGENTS_TEMPLATE = """\
# Interview Question Generator

You prepare a ghostwriter for their next content interview with a client.

## The Goal

Generate interview questions that extract compelling, novel personal stories \
and experiences the client hasn't shared before — stories that can power \
LinkedIn posts readers would hit "Save" on.

The ghostwriter walks into the interview with your briefing. If the questions \
surface rich, untold stories that become great posts, you succeeded. If the \
interview retreads old ground or yields only generic product talk, you failed.

## What Makes a Great Interview Question

A great question gets the client to tell a story they haven't told before. \
It should:

- Reference something specific from a past interview and probe deeper into \
  the human side of it
- Ask about a *moment*, not a concept ("Walk me through the morning you \
  realized X" not "What do you think about X?")
- Make the client pause and think, not recite talking points
- Yield material that reads as a lived experience, not an industry take

## What Makes a Bad Interview Question

- Generic questions about their industry or product
- Questions they've already answered thoroughly in past interviews
- Questions that sound like a podcast host trying too hard
- Questions whose answers would sound like ChatGPT wrote them
- Banned phrases: "pull on that thread", "unpack that", "dive in", \
  "tell me a story about", "yelling at the TV"

## Workspace

Explore the workspace using your tools. Key paths:

| Path | Contents |
|------|----------|
| `context/user/profile-linkedin.md` | Client's LinkedIn profile |
| `transcripts/primary.md` | Latest interview transcript — see what's been covered recently |
| `transcripts/history/` | Previous content interviews — full history of covered ground |
| `notes/` | Interview prep notes, topic banks, planning docs |
| `context/published-posts/` | Published posts with engagement data |
| `context/draft-posts/` | Posts already drafted — topics to avoid duplicating |
| `context/research/` | Deep research on client and company |
| `context/org/` | Company context — industry, positioning, competitors |
| `accepted/` | Published / approved posts — study for voice and topics covered |
| `content_strategy/` | Content strategy documents |
| `abm_profiles/` | ABM target briefings |
| `feedback/` | Client feedback on previous drafts — reveals preferences |
| `revisions/` | Before/after revision pairs — reveals style preferences |
| `past_posts/` | Post history for redundancy checking |
| `scratch/` | Your working area — write analysis, notes here |

Read everything before you write. The transcripts are your primary source — \
you need to know exactly what ground has been covered to avoid it.

**Content strategy from data**: Analyze the client's published posts in \
`context/published-posts/`. Each file includes engagement metrics (reactions, \
comments, reposts, engagement score, outlier flags). Identify which topics, \
formats, angles, and hooks drive the strongest engagement. Use these patterns \
to inform which *types* of stories to pursue in the interview — dig for more \
of what resonates with the client's audience, less of what falls flat. If no \
`content_strategy/` document exists, intuit one entirely from the engagement \
data — you have everything you need. If a strategy document does exist, \
treat it as an intent layer (e.g. pivots, ABM targets, compliance) that can \
override the data, but default to what the numbers show.

## Web Research

To search the web (Parallel API):
```bash
python3 tools/web_search.py "your search query here"
```

To extract content from a URL:
```bash
python3 tools/fetch_url.py "https://example.com/article"
```

Use web research to find timely hooks — recent news about the client's \
company, industry shifts, competitor moves — anything that could spark a \
question the client would be excited to answer.

## Process

1. Read everything in the workspace before writing anything
2. Write a topic audit to `scratch/audit.md` — catalog every story, \
   anecdote, and theme already covered in past interviews
3. Research the client's space for timely angles and hooks
4. Write the final briefing to `output/briefing.md`

## Output

Write a markdown briefing to `output/briefing.md`. The interview questions \
are the core deliverable. Include whatever supporting analysis (topic audit, \
domain context, ICP notes) genuinely helps the ghostwriter in the room — \
but don't pad. If something doesn't make the interview better, cut it.

{dynamic_directives}
"""

# ---------------------------------------------------------------------------
# Direct-API fallback system prompt (no tool use — context stuffed inline)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM_PROMPT = """\
You are an elite interview prep specialist for a LinkedIn ghostwriting agency.

Your job: generate interview questions that extract compelling, novel personal \
stories from the client — stories that become LinkedIn posts readers hit "Save" on.

Great questions ask about *moments* — failures, surprises, conflicts, decisions, \
realizations. Not product features. Not industry opinions. The human stories \
behind the work.

You will receive the client's transcripts, published posts, feedback, and other \
context. Read it all carefully, then:

1. Identify every topic and story already covered — these are OFF LIMITS
2. Find gaps: untold stories, unexplored angles, timely hooks
3. Write a briefing with interview questions targeting those gaps

Every question should pass this test: if the client answers honestly, would \
the answer make a LinkedIn post worth saving?
"""


# ---------------------------------------------------------------------------
# Pi-based briefing agent
# ---------------------------------------------------------------------------

def _run_briefing_agent(
    workspace_root: Path,
    user_prompt: str,
    company_keyword: str,
    trace=None,
) -> tuple[str | None, list[dict]]:
    """Run the interview prep agent via Pi CLI."""
    session_log: list[dict] = []
    session_start = time.time()

    directives = _build_dynamic_directives(company_keyword)
    agents_md = _BRIEFING_AGENTS_TEMPLATE.format(dynamic_directives=directives)
    (workspace_root / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    _write_tool_scripts(workspace_root)

    session_dir = P.memory_dir(company_keyword) / ".pi-briefing-sessions"
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

    session_log.append({
        "type": "session_start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": "pi",
        "has_prior_session": has_sessions,
        "workspace": str(workspace_root),
    })

    pi_timeout = 600
    logger.info("[Aglaea/Pi] Starting Pi agent (session_dir=%s)...", session_dir)
    print(f"[Aglaea] Running Pi agent for {company_keyword} (timeout={pi_timeout}s)...")

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
        logger.error("[Aglaea/Pi] Pi timed out after %ds", pi_timeout)
        print(f"[Aglaea] Pi timed out after {pi_timeout}s")
        session_log.append({"type": "timeout", "timeout_seconds": pi_timeout})
        stdout_raw, stderr_output, exit_code = "", "", -1
    except FileNotFoundError:
        logger.error("[Aglaea/Pi] Pi CLI not found")
        return None, session_log

    events_seen = 0
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
            continue

        events_seen += 1
        etype = event.get("type", "")

        if etype == "message_update":
            ae = event.get("assistantMessageEvent", {})
            msg = event.get("message", ae.get("message", {}))
            if ae.get("type", "").startswith("toolcall"):
                for block in msg.get("content", []):
                    if block.get("type") == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments", {})
                        summary = args.get("path", args.get("command", str(args)))[:80] if isinstance(args, dict) else ""
                        logger.info("[Aglaea/Pi] tool: %s(%s)", name, summary)
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
                    "[Aglaea/Pi] turn end — in=%d out=%d cost=$%.4f",
                    usage.get("input", 0), usage.get("output", 0),
                    cost_info.get("total", 0) if isinstance(cost_info, dict) else 0,
                )

        elif etype == "error":
            logger.error("[Aglaea/Pi] Error: %s", event.get("message", str(event))[:300])

        session_log.append({
            "type": "pi_event",
            "event_type": etype,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": str(event)[:500],
        })

        if trace:
            try:
                if etype in ("message_update", "turn_end", "error"):
                    trace.event(name=f"pi/{etype}", metadata={"event": str(event)[:500]})
            except Exception:
                pass

    if stdout_raw:
        try:
            (workspace_root / "output" / "pi_events.jsonl").write_text(stdout_raw, encoding="utf-8")
        except Exception:
            pass

    total_elapsed = time.time() - session_start
    session_log.append({
        "type": "session_end",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exit_code": exit_code,
        "events_seen": events_seen,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "total_elapsed_seconds": round(total_elapsed, 1),
    })

    logger.info(
        "[Aglaea/Pi] Finished: exit=%d, events=%d, in=%d out=%d cost=$%.4f elapsed=%.1fs",
        exit_code, events_seen, total_input_tokens, total_output_tokens, total_cost, total_elapsed,
    )

    if exit_code != 0:
        logger.error("[Aglaea/Pi] Pi exited with code %d. stderr: %s", exit_code, stderr_output[:500])

    briefing = _extract_briefing(workspace_root)
    return briefing, session_log


def _extract_briefing(workspace_root: Path) -> str | None:
    """Read the agent's output/briefing.md file."""
    briefing_path = workspace_root / "output" / "briefing.md"
    if briefing_path.exists():
        text = briefing_path.read_text(encoding="utf-8").strip()
        if text:
            logger.info("[Aglaea] Loaded briefing from output/briefing.md (%d chars)", len(text))
            return text

    for candidate in (workspace_root / "output").glob("*.md"):
        text = candidate.read_text(encoding="utf-8").strip()
        if text:
            logger.info("[Aglaea] Loaded briefing from %s (%d chars)", candidate.name, len(text))
            return text

    for candidate in sorted((workspace_root / "scratch").rglob("*.md")):
        text = candidate.read_text(encoding="utf-8").strip()
        if len(text) > 500:
            logger.info("[Aglaea] Recovered briefing from scratch: %s", candidate)
            return text

    return None


# ---------------------------------------------------------------------------
# Direct-API fallback (single Claude message, no tool use)
# ---------------------------------------------------------------------------

def _run_direct_fallback(company_keyword: str, client_name: str) -> str | None:
    """Stuff all workspace context into a single Claude message."""
    memory = P.memory_dir(company_keyword)
    context_parts = []

    for subdir_name in ("transcripts", "accepted", "feedback", "revisions",
                        "content_strategy", "abm_profiles", "notes"):
        subdir = memory / subdir_name
        if not subdir.exists():
            continue
        for f in sorted(subdir.rglob("*")):
            if f.is_file() and f.suffix in (".txt", ".md"):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        context_parts.append(f"--- {subdir_name}/{f.name} ---\n{text}")
                except Exception:
                    pass

    if not context_parts:
        logger.warning("[Aglaea] No context files found for %s", company_keyword)
        return None

    context_blob = "\n\n".join(context_parts)
    if len(context_blob) > 180_000:
        context_blob = context_blob[:180_000] + "\n\n[... truncated ...]"

    directives = _build_dynamic_directives(company_keyword)

    user_msg = (
        f"Prepare a briefing for the ghostwriter's next interview with {client_name} "
        f"({company_keyword}). Generate questions that extract novel personal stories "
        f"for LinkedIn ghostwriting.\n\n"
        f"CLIENT CONTEXT:\n\n{context_blob}"
    )
    if directives:
        user_msg += f"\n\nADDITIONAL DIRECTIVES:\n\n{directives}"

    logger.info("[Aglaea] Running direct fallback (%d chars context)", len(context_blob))
    print(f"[Aglaea] Pi not available — using direct Claude call for {company_keyword}...")

    try:
        resp = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            system=_DIRECT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    except Exception as e:
        logger.error("[Aglaea] Direct fallback failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_briefing(client_name: str, company_keyword: str) -> str:
    """Generate an interview briefing for the next content interview."""
    print(f"[Aglaea] Starting interview prep for {client_name}...")

    P.ensure_dirs(company_keyword)
    output_dir = P.brief_dir(company_keyword)
    output_filepath = str(output_dir / f"{company_keyword}_briefing.md")

    langfuse = _get_langfuse()
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name=f"aglaea/{client_name}",
                metadata={
                    "company_keyword": company_keyword,
                    "runner": "pi" if _PI_AVAILABLE else "direct",
                },
            )
        except Exception as e:
            logger.warning("[Aglaea] Langfuse trace creation failed: %s", e)

    user_prompt = (
        f"Prepare a briefing for the ghostwriter's next content interview with "
        f"{client_name}. Read all transcripts to know what ground has been covered, "
        f"then generate questions that will surface fresh, untold personal stories "
        f"worth turning into LinkedIn posts."
    )

    briefing_text = None

    if _PI_AVAILABLE:
        print("[Aglaea] Using Pi agent...")
        workspace_root = _setup_workspace(company_keyword)
        briefing_text, session_log = _run_briefing_agent(
            workspace_root, user_prompt, company_keyword, trace=trace,
        )

        session_path = output_filepath.replace(".md", "_session.jsonl")
        Path(session_path).parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w", encoding="utf-8") as f:
            for entry in session_log:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        print(f"[Aglaea] Session log saved to {session_path}")
    else:
        logger.warning("[Aglaea] Pi not installed — falling back to direct API")
        briefing_text = _run_direct_fallback(company_keyword, client_name)

    if briefing_text is None:
        print("[Aglaea] Agent did not produce a briefing. Writing empty output.")
        briefing_text = f"# {client_name} — Interview Briefing\n\nAgent failed to produce output.\n"

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(briefing_text)

    if trace:
        trace.update(output={
            "status": "completed" if "failed" not in briefing_text.lower() else "failed",
            "briefing_length": len(briefing_text),
        })
    if langfuse:
        langfuse.flush()

    print(f"\n[Aglaea] Briefing complete! Output at: {output_filepath}")
    return output_filepath
