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

from backend.src.db import vortex as P
from backend.src.agents.stelle import (
    _setup_workspace,
    _write_tool_scripts,
    _build_dynamic_directives,
    _get_ordinal_api_key,
    _PI_AVAILABLE,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aglaea")

_client = Anthropic()

PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")


# ---------------------------------------------------------------------------
# AGENTS.md template — Pi-native system prompt for interview prep
# ---------------------------------------------------------------------------

_BRIEFING_AGENTS_TEMPLATE = """\
# Interview Question Generator

You prepare a ghostwriter for their next content interview with a client.

## The Goal

Generate interview questions that surface source material capable of producing \
posts that maximise ICP audience engagement. Your job is to study the client's \
past posts and engagement data, reverse-engineer what kind of source material \
has been working, and design questions to surface more of it.

The ghostwriter walks into the interview with your briefing. If the questions \
surface material that produces posts the ICP audience engages with, you \
succeeded. If the interview retreads covered ground or yields material that \
looks like what has already underperformed, you failed.

Prioritise breadth across the client's experiences, relationships, and \
knowledge domains — the system learns what resonates only if it has diverse \
material to generate from.

## What Makes a Great Interview Question

A great question surfaces source material that the data says drives ICP \
engagement for this client. Study past posts and engagement data before \
writing any questions.

<!-- HARD-CODED BY DESIGN: The instruction below is interview methodology,
     not content theory. It describes HOW to elicit usable material —
     "ask about moments not concepts" is a measurement technique that
     applies regardless of what the data says to look for. It is the
     difference between a calibrated instrument and a hypothesis about
     what you will find. The WHAT (which topics, stories, and angles to
     target) is fully data-driven via the interview objectives injected at
     runtime. The HOW stays fixed. Do not make this dynamic. -->
Methodologically: ask about *moments*, not concepts. "Walk me through \
the morning you realised X" yields specific, usable material. "What do \
you think about X?" yields generic takes. Make the client recall a \
specific situation rather than form an opinion.

## What Makes a Bad Interview Question

- Questions that cover ground already in past transcripts
- Questions whose answers would look like what has already underperformed
- Questions that produce industry-generic answers with no specificity \
  to this client's actual situation

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
| `references/` | Client-provided articles, URLs, and reference material — treat as supplementary source material |
| `scratch/` | Your working area — write analysis, notes here |

Read everything before you write. The transcripts and references are your \
primary sources — you need to know exactly what ground has been covered to \
avoid it.

**Let the data define what to look for**: Study the client's actual engagement \
history (past posts + reactor data) to see what has driven ICP engagement. \
Do not apply a pre-loaded theory of what makes content good. The system \
learns what works by observing ICP outcomes; your job is to surface source \
material that could produce more of what has actually worked, and broad \
enough material that the system can learn from what hasn't been tried yet.

## Web Research

To search the web (Parallel API):
```bash
python3 tools/web_search.py "your search query here"
```

To extract content from a URL:
```bash
python3 tools/fetch_url.py "https://example.com/article"
```

## LinkedIn Post Database

Search 200K+ real LinkedIn posts with engagement metrics to understand \
what resonates in the client's space:
```bash
python3 tools/query_posts.py "topic or keyword"
```

Search a specific creator's posts by username to study their style:
```bash
python3 tools/query_posts.py --creator "username"
```

Use this to identify which *types* of stories drive engagement — then \
craft interview questions that will surface those types of stories from \
the client.

## Semantic Post Search

Search the post database by meaning, not just keywords:
```bash
python3 tools/semantic_search_posts.py "vulnerability and failure stories in leadership"
```

Finds conceptually similar posts even with different wording. Use this to \
discover what *kinds* of personal stories drive engagement — then design \
questions that pull those stories out of the client.

## Ordinal Analytics

Get real LinkedIn performance data for the client:
```bash
python3 tools/ordinal_analytics.py profiles                # list scheduling profiles
python3 tools/ordinal_analytics.py followers <profileId>   # follower count + growth
python3 tools/ordinal_analytics.py posts <profileId>       # post impressions + engagement
python3 tools/ordinal_analytics.py cadence <profileId>     # posting frequency + gap analysis
```

Use this to understand what's actually working for this client — which \
topics and formats drive the most engagement — and steer interview \
questions toward surfacing more of those stories.

## Timely Research

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
You are an interview prep specialist for a LinkedIn ghostwriting agency.

Your job: generate interview questions that surface source material capable \
of producing posts that maximise ICP audience engagement. Study the client's \
transcripts, published posts, and engagement data before writing a single \
question.

# HARD-CODED BY DESIGN: interview methodology, not content theory.
# This is a measurement technique — HOW to elicit specific, usable material
# regardless of WHAT the data says to target. The "what" is data-driven
# (see interview objectives in the user prompt). The "how" stays fixed.
Methodologically: ask about *moments*, not concepts. Specific situations \
yield usable material. Opinions yield generic takes.

Prioritise breadth — cover diverse experiences, relationships, and knowledge \
domains. The system learns what resonates only from material it has actually \
generated from.

You will receive the client's transcripts, published posts, and other context. \
Read it all carefully, then:

1. Identify every topic and story already covered — these are OFF LIMITS
2. Study the ICP exemplars in the objectives — what kind of source material \
   could produce more posts like those?
3. Write a briefing with questions targeting that material and unexplored ground
"""


# ---------------------------------------------------------------------------
# Pi-based briefing agent
# ---------------------------------------------------------------------------

def _run_briefing_agent(
    workspace_root: Path,
    user_prompt: str,
    company_keyword: str,
    event_callback=None,
) -> tuple[str | None, list[dict]]:
    """Run the interview prep agent via Pi CLI, streaming events in real time."""
    import threading

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
    env["SUPABASE_URL"] = os.getenv("SUPABASE_URL", "")
    env["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY", "")
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
    ordinal_key = _get_ordinal_api_key(company_keyword)
    if ordinal_key:
        env["ORDINAL_API_KEY"] = ordinal_key

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

    events_seen = 0
    compaction_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    stdout_lines: list[str] = []
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
                        summary = args.get("path", args.get("command", str(args)))[:80] if isinstance(args, dict) else ""
                        logger.info("[Aglaea/Pi] tool: %s(%s)", name, summary)
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
                logger.info("[Aglaea/Pi] turn end — in=%d out=%d cache=%d cost=$%.4f", in_tok, out_tok, cache_tok, cost_val)
                if event_callback:
                    event_callback("status", {"message": f"Turn complete — in={in_tok} out={out_tok} cache={cache_tok} cost=${cost_val:.4f}"})

        elif etype == "auto_compaction_start":
            compaction_count += 1
            logger.info("[Aglaea/Pi] Context compaction #%d", compaction_count)
            if event_callback:
                event_callback("compaction", {"message": f"Context compaction #{compaction_count}"})

        elif etype == "auto_retry_start":
            if event_callback:
                event_callback("status", {"message": f"Retry {event.get('attempt', '?')}/{event.get('maxAttempts', '?')}..."})

        elif etype == "error":
            err_msg = event.get("message", str(event))[:300]
            logger.error("[Aglaea/Pi] Error: %s", err_msg)
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

        stderr_chunks: list[str] = []

        def _read_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        assert proc.stdout is not None
        deadline = time.time() + pi_timeout
        for line in proc.stdout:
            if time.time() > deadline:
                proc.kill()
                logger.error("[Aglaea/Pi] Timed out after %ds", pi_timeout)
                if event_callback:
                    event_callback("status", {"message": f"Timed out after {pi_timeout}s"})
                break
            line = line.strip()
            if not line:
                continue
            stdout_lines.append(line)
            try:
                event = json.loads(line)
                _process_event(event)
            except json.JSONDecodeError:
                logger.debug("[Aglaea/Pi] Non-JSON line: %s", line[:200])

        proc.wait(timeout=10)
        exit_code = proc.returncode
        stderr_thread.join(timeout=5)
        stderr_output = "".join(stderr_chunks)

    except subprocess.TimeoutExpired:
        logger.error("[Aglaea/Pi] Pi timed out after %ds", pi_timeout)
        session_log.append({"type": "timeout", "timeout_seconds": pi_timeout})
        stderr_output = ""
    except FileNotFoundError:
        logger.error("[Aglaea/Pi] Pi CLI not found")
        return None, session_log

    stdout_raw = "\n".join(stdout_lines)
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

    for subdir_name in ("transcripts", "references", "accepted", "feedback",
                        "revisions", "content_strategy", "abm_profiles", "notes"):
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

def generate_briefing(client_name: str, company_keyword: str, event_callback=None) -> str:
    """Generate an interview briefing for the next content interview."""
    print(f"[Aglaea] Starting interview prep for {client_name}...")

    P.ensure_dirs(company_keyword)
    output_dir = P.brief_dir(company_keyword)
    output_filepath = str(output_dir / f"{company_keyword}_briefing.md")

    user_prompt = (
        f"Prepare a briefing for the ghostwriter's next content interview with "
        f"{client_name}. Read all transcripts to know what ground has been covered, "
        f"then generate questions that will surface at least 12 fresh, distinct "
        f"personal stories worth turning into LinkedIn posts. Each story should be "
        f"capable of standing alone as its own post."
    )

    briefing_text = None

    if _PI_AVAILABLE:
        print("[Aglaea] Using Pi agent...")
        workspace_root = _setup_workspace(company_keyword)
        briefing_text, session_log = _run_briefing_agent(
            workspace_root, user_prompt, company_keyword,
            event_callback=event_callback,
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

    print(f"\n[Aglaea] Briefing complete! Output at: {output_filepath}")
    return output_filepath
