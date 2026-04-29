"""Claude CLI adapter — run agent loops through `claude` instead of the API.

Drop-in replacements for the API-based agent entry points that invoke
the Claude CLI with MCP servers for tool access. Uses the user's
Claude Max subscription instead of per-token API billing.

ZERO changes to existing agent code. The adapter builds the same
system prompt + user message, writes a temp MCP config, and invokes:

    claude -p "user message" \
        --system-prompt-file /tmp/system.txt \
        --mcp-config /tmp/mcp.json \
        --strict-mcp-config \
        --output-format json \
        --model opus \
        --permission-mode bypassPermissions \
        --max-turns 8

Then parses the JSON output to extract the submit_reaction call.

Activate by setting AMPHOREUS_USE_CLI=1 in the environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------
# Claude Max's rate limits are per-account and shared across every CLI
# subprocess that auths with the same OAuth session. Running Stelle +
# Cyrene + an Irontomb critique in parallel against the same Max
# account burns through the 5h window way faster than running them
# serially — same messages, much tighter pack. Worse: two subprocesses
# that write to the same ``~/.claude/`` session dir can corrupt each
# other's state (rare but observed).
#
# AMPHOREUS_CLI_MAX_CONCURRENCY caps how many CLI subprocesses we'll
# have in flight at once. Default is unlimited (0) for local dev where
# a single human is the only caller anyway. Production Fly should set
# it to 1 or 2 — see DEPLOY.md "Claude CLI on Fly" for the full trade.
# Counter is process-local; multi-machine fly deploys would bypass it,
# but we currently run a single machine.

def _max_concurrency() -> int:
    raw = os.environ.get("AMPHOREUS_CLI_MAX_CONCURRENCY", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    return max(0, n)


_cli_semaphore: Optional[threading.Semaphore] = None
_cli_semaphore_lock = threading.Lock()


def _get_cli_semaphore() -> Optional[threading.Semaphore]:
    """Lazily create the semaphore on first access. None = unlimited."""
    global _cli_semaphore
    cap = _max_concurrency()
    if cap <= 0:
        return None
    with _cli_semaphore_lock:
        if _cli_semaphore is None:
            _cli_semaphore = threading.Semaphore(cap)
    return _cli_semaphore


class _cli_concurrency_guard:
    """Context manager that blocks when the configured CLI concurrency
    ceiling is hit. No-op when unlimited. Timeouts raise so one stuck
    subprocess can't stall every future CLI call forever."""

    TIMEOUT_SECONDS = 30 * 60  # 30 min — longer than a typical Stelle run

    def __enter__(self):
        sem = _get_cli_semaphore()
        self._sem = sem
        if sem is None:
            return self
        t0 = time.time()
        got = sem.acquire(timeout=self.TIMEOUT_SECONDS)
        if not got:
            raise RuntimeError(
                "[claude_cli] concurrency semaphore timeout — another CLI "
                "subprocess held the slot for >30m"
            )
        waited = time.time() - t0
        if waited > 1:
            logger.info("[claude_cli] waited %.1fs for concurrency slot", waited)
        return self

    def __exit__(self, *args):
        if self._sem is not None:
            self._sem.release()


def _gated_cli_call(fn):
    """Decorator: wrap the decorated entry point so it acquires the
    concurrency semaphore for its whole duration. Applied to the four
    public CLI entry points (``run_stelle_cli``, ``run_cyrene_cli``,
    ``simulate_flame_chase_journey_cli``, ``cli_single_shot``) so every
    path that calls ``claude -p`` goes through the same cap."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _cli_concurrency_guard():
            return fn(*args, **kwargs)
    return wrapper


def _cli_env() -> dict:
    """Return the env dict used for every `claude` subprocess we spawn.

    Honours ``AMPHOREUS_CLAUDE_CONFIG_DIR`` so the production pipeline can
    use a separate Max account from the developer's interactive Claude
    Code sessions. The variable is forwarded to the CLI as
    ``CLAUDE_CONFIG_DIR``, which the CLI uses to locate OAuth credentials,
    settings, and session history. If unset, the subprocess inherits the
    parent's environment unchanged (i.e. uses ``~/.claude/`` like normal),
    so this is backward-compatible.

    One-time setup for the dedicated account::

        mkdir -p ~/.claude-amphoreus
        CLAUDE_CONFIG_DIR=~/.claude-amphoreus claude login
        export AMPHOREUS_CLAUDE_CONFIG_DIR=~/.claude-amphoreus  # in backend env
    """
    env = os.environ.copy()
    # Strip ANTHROPIC_API_KEY so the CLI uses OAuth (Max plan) instead of
    # falling back to API-key billing. The backend needs the key for its
    # own direct Anthropic SDK calls, but CLI subprocesses must authenticate
    # via the Max plan's OAuth token in the keychain / CLAUDE_CONFIG_DIR.
    env.pop("ANTHROPIC_API_KEY", None)
    # Also strip the billing helper that the SDK sometimes reads.
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    override = os.environ.get("AMPHOREUS_CLAUDE_CONFIG_DIR", "").strip()
    if override:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(override)
    # The Fly image runs as root (docker-entrypoint.sh needs root to
    # symlink the persistent volume into /app/* paths). Claude Code
    # refuses ``--dangerously-skip-permissions`` / ``--permission-mode
    # bypassPermissions`` for root unless IS_SANDBOX=1 is set — it's
    # the standard container-escape-hatch the CLI recognises. Without
    # this, every CLI subprocess dies with
    # "--dangerously-skip-permissions cannot be used with root/sudo
    # privileges for security reasons". Safe to set unconditionally;
    # no effect outside the CLI's own permission check.
    env.setdefault("IS_SANDBOX", "1")
    return env


# ---------------------------------------------------------------------------
# Usage tracking (equivalent cost, not real spend)
# ---------------------------------------------------------------------------

# Map CLI --model shorthand to the billing-grade model id used by price_call.
# "opus" / "sonnet" / "haiku" are the CLI's aliases; they resolve to whatever
# the CLI considers the current default for that tier. We bill against our
# current canonical id for that tier.
_CLI_MODEL_TO_BILLING = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _parse_cli_usage(stdout: str) -> dict | None:
    """Extract token usage from CLI stream-json or json stdout.

    The CLI emits a terminal `result` event (stream-json) or a single
    result object (json) with a ``usage`` block shaped like:

        {"input_tokens": N, "output_tokens": N,
         "cache_creation_input_tokens": N, "cache_read_input_tokens": N}

    Returns None if no usage info is parseable. Never raises.
    """
    if not stdout:
        return None

    usage = None
    try:
        obj = json.loads(stdout.strip())
        if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
    except json.JSONDecodeError:
        pass

    if usage is None:
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "result" and isinstance(event.get("usage"), dict):
                usage = event["usage"]

    if not usage:
        return None

    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_tokens": int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_creation_tokens")
            or 0
        ),
        "cache_read_tokens": int(
            usage.get("cache_read_input_tokens")
            or usage.get("cache_read_tokens")
            or 0
        ),
    }


def _record_cli_usage(
    *,
    stdout: str,
    cli_model: str,
    call_kind: str,
    client_slug: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    """Parse CLI stdout for usage and insert an equivalent-cost row.

    Writes with provider='anthropic_cli' so the admin dashboard can
    distinguish zero-dollar Max-plan usage from actual API spend. Cost
    is computed as if the same tokens had gone through the API, so ops
    can see "savings from CLI = sum(anthropic_cli.cost_usd)".
    """
    try:
        usage = _parse_cli_usage(stdout)
        if usage is None:
            logger.debug("[CLI-usage] No usage block in stdout for %s", call_kind)
            return

        billing_model = _CLI_MODEL_TO_BILLING.get(cli_model, cli_model)

        from backend.src.usage.recorder import record_usage_event
        record_usage_event(
            provider="anthropic_cli",
            model=billing_model,
            call_kind="messages",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            client_slug=client_slug,
            duration_ms=duration_ms,
            error=error,
        )
    except Exception as e:
        logger.warning("[CLI-usage] Failed to record usage for %s: %s", call_kind, e)


# ---------------------------------------------------------------------------
# Irontomb via CLI
# ---------------------------------------------------------------------------

@_gated_cli_call
def simulate_flame_chase_journey_cli(
    company: str,
    draft_text: str,
) -> dict[str, Any]:
    """Drop-in replacement for irontomb.simulate_flame_chase_journey().

    Builds the same system prompt and user message, then runs the
    simulation through `claude` CLI with the Irontomb MCP server
    instead of the Anthropic API.
    """
    from backend.src.agents.irontomb import (
        _build_system_prompt,
        _draft_hash,
        _format_calibration_block,
        _format_cross_client_block,
        _load_audience_context,
        _load_scored_observations,
        _IRONTOMB_MAX_TURNS,
    )

    draft_text = (draft_text or "").strip()
    if not draft_text:
        return {"_error": "draft_text is required"}

    t0 = time.time()

    # Build the same context that the API path uses
    audience_context = _load_audience_context(company)
    observations = _load_scored_observations(company)
    calibration_block = _format_calibration_block(observations)
    cross_client_block = _format_cross_client_block(company)
    system_prompt = _build_system_prompt(
        audience_context,
        n_scored_obs=len(observations),
        calibration_block=calibration_block,
        cross_client_block=cross_client_block,
    )

    user_message = (
        "Here is the draft LinkedIn post you are evaluating. "
        "You've already read the calibration examples from this "
        "client's real history above. Predict how this specific "
        "audience will react to THIS draft, anchored in what you "
        "saw happen to comparable past posts. If the draft is in "
        "territory the examples don't cover, retrieve more "
        "comparables first; otherwise submit_reaction directly.\n\n"
        "=== DRAFT ===\n"
        f"{draft_text}\n"
        "=== END DRAFT ==="
    )

    # Write temp files for the CLI
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="irontomb_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    mcp_config = {
        "mcpServers": {
            "irontomb-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "irontomb_server.py"),
                ],
                "env": {
                    "IRONTOMB_COMPANY": company,
                    # Forward relevant env vars
                    "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
                    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                    "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
                },
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="irontomb_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    try:
        cmd = [
            "claude",
            "-p", user_message,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--strict-mcp-config",
            "--output-format", "stream-json",
            "--verbose",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(_IRONTOMB_MAX_TURNS),
            # NOTE: do NOT use --bare here. --bare disables OAuth and
            # falls back to API key auth, bypassing the Max plan.
        ]

        logger.info("[CLI] Running Irontomb simulation for %s via claude CLI...", company)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env=_cli_env(),
        )

        elapsed = time.time() - t0
        dh = _draft_hash(draft_text)

        _record_cli_usage(
            stdout=proc.stdout or "",
            cli_model="opus",
            call_kind="irontomb_cli",
            client_slug=company,
            duration_ms=int(elapsed * 1000),
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
        )

        if proc.returncode != 0:
            logger.warning("[CLI] claude exited %d: %s", proc.returncode, proc.stderr[:500])
            return {
                "_error": f"claude CLI exited {proc.returncode}: {proc.stderr[:200]}",
                "_draft_hash": dh,
                "_elapsed_s": round(elapsed, 1),
                "_via": "cli",
            }

        # Parse stream-json output to find submit_reaction tool call
        reaction = _extract_reaction_from_stream(proc.stdout)

        if reaction is None:
            # Fallback: try single JSON format
            reaction = _extract_reaction_from_json(proc.stdout)

        if reaction is None:
            logger.warning("[CLI] Could not extract reaction from output")
            return {
                "_error": "No submit_reaction found in CLI output",
                "_draft_hash": dh,
                "_elapsed_s": round(elapsed, 1),
                "_via": "cli",
                "_raw_stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            }

        # Build the same return shape as the API path. ``anchors`` is
        # the new first-class list (zero, one, or many inline anchors);
        # ``anchor`` (singular) is synthesized from the first anchor's
        # quote for back-compat with the convergence_log column and
        # trajectory snapshots.
        anchors = reaction.get("anchors")
        if not isinstance(anchors, list):
            anchors = []
        legacy_anchor = reaction.get("anchor")
        if not legacy_anchor:
            legacy_anchor = (anchors[0].get("quote", "") if anchors else "")
        result = {
            "reaction": reaction.get("reaction", ""),
            "anchors": anchors,
            "anchor":  legacy_anchor,
            "_draft_hash": dh,
            "_via": "cli",
            "_elapsed_s": round(elapsed, 1),
            "_cost_usd": 0.0,  # Max plan — no per-token cost
        }
        return result

    except subprocess.TimeoutExpired:
        return {
            "_error": "claude CLI timed out after 300s",
            "_draft_hash": _draft_hash(draft_text),
            "_via": "cli",
        }
    except FileNotFoundError:
        return {
            "_error": "claude CLI not found — is Claude Code installed?",
            "_draft_hash": _draft_hash(draft_text),
            "_via": "cli",
        }
    finally:
        # Clean up temp files
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_reaction_from_stream(stdout: str) -> Optional[dict]:
    """Parse stream-json output to find submit_reaction tool call.

    The Claude CLI prefixes MCP tool names with the server name:
    e.g. "mcp__irontomb-tools__submit_reaction". We match on the
    suffix "submit_reaction" to handle this.
    """
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Look for tool_use blocks in assistant messages
        if event.get("type") == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if (
                    block.get("type") == "tool_use"
                    and (block.get("name", "").endswith("submit_reaction"))
                ):
                    return block.get("input", {})

        # Also check content_block_start events
        if event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if (
                block.get("type") == "tool_use"
                and (block.get("name", "").endswith("submit_reaction"))
            ):
                return block.get("input", {})

    return None


def _extract_reaction_from_json(stdout: str) -> Optional[dict]:
    """Parse CLI JSON output and extract submit_reaction arguments.

    The CLI's JSON output has a `result` field with the model's final
    text. But the submit_reaction tool call happened during the tool
    loop — its arguments were passed to the MCP server which echoed
    them back. The model's `result` text often summarizes its prediction.

    Strategy: look for the reaction/anchor fields in the result text
    (the MCP server echoes back the submitted fields as JSON).
    """
    try:
        cli_result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return None

    result_text = cli_result.get("result", "")

    # The new schema has nested objects under ``anchors`` (a list of
    # {quote, reaction}), so the old ``\{[^{}]*\}`` matcher would
    # truncate on the inner braces. Use a depth-aware brace walker
    # instead. Look for the first JSON object that contains the
    # ``"reaction"`` key.
    import re
    for m in re.finditer(r'"reaction"\s*:', result_text):
        # Walk backward from the match to the opening brace, then
        # forward through balanced braces to find the closing one.
        i = m.start()
        # Find the enclosing '{' (scan back over balanced braces)
        depth = 0
        start = -1
        for j in range(i, -1, -1):
            ch = result_text[j]
            if ch == '}':
                depth += 1
            elif ch == '{':
                if depth == 0:
                    start = j
                    break
                depth -= 1
        if start < 0:
            continue
        # Forward-scan for the matching close '}'
        depth = 0
        end = -1
        in_str = False
        esc = False
        for k in range(start, len(result_text)):
            ch = result_text[k]
            if esc:
                esc = False; continue
            if ch == '\\' and in_str:
                esc = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            continue
        try:
            data = json.loads(result_text[start:end+1])
            if isinstance(data, dict) and "reaction" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Try parsing the entire result as JSON
    try:
        data = json.loads(result_text)
        if isinstance(data, dict) and "reaction" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # No prose-extraction fallback — if the tool call didn't surface
    # structured reaction/anchor fields, return None so the caller can
    # error cleanly rather than synthesize spurious content.
    return None


# ---------------------------------------------------------------------------
# Single-shot CLI call (for compaction, why-post, image suggestion, etc.)
# ---------------------------------------------------------------------------

@_gated_cli_call
def cli_single_shot(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "sonnet",
    max_tokens: int = 4096,
    timeout: int = 60,
) -> str | None:
    """Run a single-shot Claude CLI call and return the text response.

    Drop-in replacement for:
        resp = client.messages.create(model=..., messages=[...])
        text = resp.content[0].text

    ``timeout`` is seconds to wait for the subprocess. Default 60s suits
    short helpers (why-post, image-suggestion). Pass a longer value for
    large-output calls like the progress-report HTML generator.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--tools", "",  # no tools needed for single-shot
    ]

    if system_prompt:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="cli_sys_"
        ) as f:
            f.write(system_prompt)
            sys_file = f.name
        cmd.extend(["--system-prompt-file", sys_file])
    else:
        sys_file = None

    try:
        _t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env=_cli_env(),
        )
        _elapsed_ms = int((time.time() - _t0) * 1000)

        _record_cli_usage(
            stdout=proc.stdout or "",
            cli_model=model,
            call_kind="single_shot",
            duration_ms=_elapsed_ms,
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
        )

        if proc.returncode != 0:
            logger.warning("[CLI] single-shot failed: %s", proc.stderr[:200])
            return None

        result = json.loads(proc.stdout)
        return result.get("result", "")

    except Exception as e:
        logger.warning("[CLI] single-shot error: %s", e)
        return None
    finally:
        if sys_file:
            try:
                os.unlink(sys_file)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Stelle via CLI
# ---------------------------------------------------------------------------

@_gated_cli_call
def run_stelle_cli(
    client_name: str,
    company_keyword: str,
    output_filepath: str,
    num_posts: int = 12,
    prompt: str | None = None,
    event_callback: Any = None,
) -> str:
    """Run Stelle's full generation loop through Claude CLI.

    Drop-in replacement for the API-based generate_one_shot().
    Builds the same system prompt and user message, sets up the
    workspace, launches `claude -p` with the Stelle MCP server
    for custom tools (the CLI provides filesystem/web tools natively),
    then reads the result from .stelle_cli_result.json and runs
    _process_result() for fact-checking and output generation.
    """
    import shutil
    from pathlib import Path as _Path

    # --- Lazy imports from stelle.py (avoid circular at module level) ---
    from backend.src.agents.stelle import (
        _build_dynamic_directives,
        _DIRECT_SYSTEM_TEMPLATE,
        _process_result,
        _resolve_supabase_ids,
        _setup_workspace,
        _validate_output,
        MAX_AGENT_TURNS,
        P,
    )

    # Resolve LinkedIn handle via the shared resolver — Jacquard
    # ``users.linkedin_url`` first (works for every FOC slug + bare
    # company UUID in the current dropdown), legacy
    # ``memory/<slug>/linkedin_username.txt`` as fallback.
    #
    # DATABASE_USER_UUID is set by stelle_runner when the ghostwriter
    # endpoint resolved a specific FOC target. Passing it here lets
    # _resolve_linkedin_username short-circuit past the multi-FOC
    # ambiguity check — Trimble / Commenda / Virio runs used to hit
    # that check and return None (the resolver refused to auto-pick
    # one FOC out of many), leaving Stelle running without voice
    # calibration for every run at shared-account companies. 2026-04-23.
    import os as _os
    _target_user_uuid = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
    from backend.src.agents.stelle import _resolve_linkedin_username
    username = _resolve_linkedin_username(
        company_keyword, user_id=_target_user_uuid,
    ) or ""
    if username:
        _, _, display_name = _resolve_supabase_ids(username)
        if display_name:
            client_name = display_name
    else:
        logger.warning(
            "[CLI] no LinkedIn handle resolved for %s (user_id=%s) — "
            "Stelle will run without voice-calibration input from published posts",
            company_keyword, _target_user_uuid,
        )

    logger.info("[CLI-Stelle] Starting CLI-based generation for %s...", client_name)

    # --- Setup (same as generate_one_shot) ---
    P.ensure_dirs(company_keyword)

    # 2026-04-23: DISABLED. Unpushed drafts are now retained across runs
    # and flow into Stelle's dedup via the post_bundle UNPUSHED section
    # (with any operator-added comments attached). See the twin note in
    # ``agents/stelle.py`` for the full rationale.

    # database mode: pre-populate a workspace with Jacquard's data (Supabase
    # + GCS) so Claude CLI's native filesystem tools read the same files
    # Stelle would see on the direct-API path. Skip the old Ordinal-fed
    # local-mode setup entirely — no Ordinal API calls, no memory/<company>/
    # building, no Amphoreus-side Supabase queries. Purely Jacquard data.
    from backend.src.agents.database_client import is_database_mode as _lm
    if _lm():
        from backend.src.agents.jacquard_direct import populate_lineage_workspace
        import shutil as _shutil
        workspace_root = _PROJECT_ROOT / "scratch" / f"database-{company_keyword}"
        # Fresh workspace per run — Jacquard data is authoritative, no
        # incremental merge semantics to preserve across runs.
        if workspace_root.exists():
            _shutil.rmtree(workspace_root)
        workspace_root.mkdir(parents=True, exist_ok=True)
        company_id = os.environ.get("DATABASE_COMPANY_ID", "").strip()
        target_slug = os.environ.get("DATABASE_USER_SLUG", "").strip() or None
        try:
            counts = populate_lineage_workspace(workspace_root, company_id, target_slug)
            total = sum(counts.values())
            logger.info(
                "[CLI-Stelle] Populated database workspace %s (%d files across %d mounts)",
                workspace_root, total, len(counts),
            )
            if event_callback:
                event_callback("status", {
                    "message": f"Loaded {total} files from Amphoreus (mirror + Storage) for company_id={company_id[:8]}…",
                })
        except Exception as exc:
            logger.exception("[CLI-Stelle] FATAL: failed to populate database workspace")
            raise RuntimeError(
                f"database workspace population failed: {exc}"
            ) from exc
    else:
        # Legacy local-mode workspace (Ordinal-fed). Shouldn't normally
        # fire given the database-mode guard in generate_one_shot.
        workspace_root = _setup_workspace(company_keyword)

    # --- Unified post bundle (body + engagement + comments + delta) ---
    # See backend/src/services/post_bundle.py for the contract.
    existing_posts_context = ""
    try:
        from backend.src.services.post_bundle import build_post_bundle
        # Per-FOC scoping: DATABASE_USER_UUID is set by stelle_runner.
        # Without this, at shared-company clients the bundle pulled
        # every FOC's drafts and the prompt argv hit Linux ARG_MAX
        # (2026-04-23 Virio incident — 758 posts, 1.2 MB argv).
        import os as _os
        _bundle_user_uuid = (_os.environ.get("DATABASE_USER_UUID") or "").strip() or None
        existing_posts_context = build_post_bundle(company_keyword, user_id=_bundle_user_uuid)
    except Exception:
        pass

    # Series Engine retired 2026-04-22 (BL cleanup). Scheduling context
    # from temporal_orchestrator stays — it's operational (timing), not
    # prescriptive (narrative arc theory).
    series_context = ""

    scheduling_context = ""
    try:
        from backend.src.services.temporal_orchestrator import build_scheduling_context as _sched_ctx
        scheduling_context = _sched_ctx(company_keyword)
    except Exception:
        pass

    # --- Build user prompt ---
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

    # --- Build system prompt ---
    directives = _build_dynamic_directives(company_keyword)
    system_prompt = _DIRECT_SYSTEM_TEMPLATE.format(dynamic_directives=directives)

    # --- Write temp files ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="stelle_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    # MCP config: stelle-tools server for custom tools
    # The CLI provides filesystem (Read/Write/Edit/Bash/Grep/Glob) and
    # web (WebSearch/WebFetch) tools natively.
    env_vars = {
        "STELLE_COMPANY": company_keyword,
        "STELLE_USE_CLI_IRONTOMB": "1",
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
        "SERPER_API_KEY": os.environ.get("SERPER_API_KEY", ""),
        "ORDINAL_API_KEY": os.environ.get("ORDINAL_API_KEY", ""),
        # GCS + database — irontomb/castorice/jacquard_direct all need these
        # so the MCP server's in-process Irontomb calls can read transcripts
        # from GCS and observations from Jacquard's Supabase.
        "GCS_CREDENTIALS_B64": os.environ.get("GCS_CREDENTIALS_B64", ""),
        "GCS_BUCKET": os.environ.get("GCS_BUCKET", ""),
        "DATABASE_COMPANY_ID": os.environ.get("DATABASE_COMPANY_ID", ""),
        "DATABASE_USER_SLUG": os.environ.get("DATABASE_USER_SLUG", ""),
    }
    # Filter out empty values
    env_vars = {k: v for k, v in env_vars.items() if v}

    # CRITICAL: also export into this process's environment so the vars
    # chain-propagate through Claude CLI → stdio MCP server by normal
    # subprocess inheritance. Claude CLI's mcp_config.env block is
    # inconsistently honored across versions — relying on it alone caused
    # stelle_server to crash on startup with "STELLE_COMPANY env var not
    # set", which left Claude with zero registered tools (Stelle then
    # spent every turn running ToolSearch against her missing tools and
    # falling back to native Glob).
    for k, v in env_vars.items():
        os.environ[k] = v

    mcp_config = {
        "mcpServers": {
            "stelle-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "stelle_server.py"),
                ],
                "env": env_vars,
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="stelle_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    # Clear any previous result file
    result_file = _PROJECT_ROOT / ".stelle_cli_result.json"
    if result_file.exists():
        result_file.unlink()

    try:
        # Keep the prompt on argv. A very large prompt here (e.g. from
        # multi-user workspace scoping escaping and populating 19 FOC
        # users' worth of context) will fail fast with ARG_MAX — that
        # is a correctness canary, not a problem to paper over with a
        # stdin pipe. If this trips, the right fix is upstream:
        # scope the workspace to the targeted user and filter internal
        # users to Amphoreus-only sources.
        cmd = [
            "claude",
            "-p", user_prompt,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--output-format", "stream-json",
            "--verbose",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(MAX_AGENT_TURNS),
            "--add-dir", str(workspace_root),
        ]

        logger.info("[CLI-Stelle] Launching claude CLI (max %d turns)...", MAX_AGENT_TURNS)
        t0 = time.time()

        # Streaming subprocess: read stdout line-by-line so the web UI
        # gets live progress via event_callback. Drain stderr on a
        # background thread to avoid pipe-buffer deadlock.
        import threading
        TIMEOUT_SEC = 3600  # 60 min

        # cwd = workspace root so the agent's relative paths (memory/config.md
        # etc.) resolve correctly against the staged workspace layout.
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
                cwd=str(workspace_root),
                env=_cli_env(),
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — is Claude Code installed?")

        stderr_chunks: list[str] = []
        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, ""):
                    if not line:
                        break
                    stderr_chunks.append(line)
            except Exception:
                pass
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        if event_callback:
            event_callback("status", {"message": "Stelle CLI running (Max plan)..."})

        stdout_lines: list[str] = []
        timed_out = False
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                stdout_lines.append(raw_line)
                if time.time() - t0 > TIMEOUT_SEC:
                    timed_out = True
                    proc.kill()
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _translate_cli_event(event, event_callback)

            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            stderr_thread.join(timeout=5)

        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_chunks)
        returncode = proc.returncode
        elapsed = time.time() - t0
        logger.info("[CLI-Stelle] CLI finished in %.1fs (exit %s)", elapsed, returncode)

        _record_cli_usage(
            stdout=full_stdout,
            cli_model="opus",
            call_kind="stelle_cli",
            client_slug=company_keyword,
            duration_ms=int(elapsed * 1000),
            error=None if returncode == 0 else f"exit {returncode}",
        )

        if timed_out:
            raise RuntimeError(f"claude CLI timed out after {TIMEOUT_SEC}s")

        if returncode != 0:
            logger.error("[CLI-Stelle] CLI failed: %s", full_stderr[:500])
            raise RuntimeError(
                f"claude CLI exited {returncode}: {full_stderr[:300] or full_stdout[-300:]}"
            )

        # Read the result written by write_result handler
        if not result_file.exists():
            logger.error("[CLI-Stelle] No result file found at %s", result_file)
            logger.info("[CLI-Stelle] stdout tail: %s", full_stdout[-1000:])
            raise RuntimeError("Stelle CLI did not produce a result file")

        result = json.loads(result_file.read_text(encoding="utf-8"))
        logger.info(
            "[CLI-Stelle] Got result with %d posts. Running post-processing...",
            len(result.get("posts", [])),
        )

        passed, val_errors, val_warnings = _validate_output(result)
        if not passed:
            logger.warning("[CLI-Stelle] Output validation issues: %s", val_errors)

        output_path = _process_result(result, client_name, company_keyword, output_filepath)

        session_path = output_filepath.replace(".md", "_session.jsonl")
        _Path(session_path).parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w", encoding="utf-8") as f:
            f.write(full_stdout)
        logger.info("[CLI-Stelle] Session log saved to %s", session_path)

        return output_path

    finally:
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Cyrene via CLI
# ---------------------------------------------------------------------------

@_gated_cli_call
def run_cyrene_cli(
    company: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Run Cyrene's strategic review through Claude CLI.

    Drop-in replacement for cyrene.run_strategic_review(). Launches
    `claude -p` with the cyrene-tools MCP server, lets it run the full
    tool-use loop, then reads the brief from .cyrene_cli_result.json
    and persists it.

    ``user_id`` (optional) scopes the review to a specific FOC user at
    a multi-FOC company — the saved brief is keyed by (company, user_id)
    and the user prompt is prefixed with that user's identity so Cyrene
    can tailor recommendations to their role/voice rather than averaging
    across the whole company. When None the run is company-wide
    (legacy path for single-FOC clients).

    Returns the brief dict on success; returns {"_error": ...} on
    failure (same shape as the API version's error path).
    """
    import threading
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    # Lazy imports to avoid circular
    from backend.src.agents.cyrene import _SYSTEM_PROMPT
    from backend.src.agents.irontomb import _load_icp_context

    logger.info(
        "[CLI-Cyrene] Starting CLI-based strategic review for %s (user_id=%s)...",
        company, user_id or "<company-wide>",
    )

    # Resolve target user info if user_id provided — used to inject a
    # user-specific context block into the prompt so Cyrene tailors the
    # brief to this individual rather than averaging across the company.
    target_user_info: dict | None = None
    if user_id:
        try:
            from backend.src.agents.jacquard_direct import list_foc_users
            for u in list_foc_users(company) or []:
                if u.get("id") == user_id:
                    target_user_info = u
                    break
        except Exception as _exc:
            logger.debug("[CLI-Cyrene] user_id lookup failed (non-fatal): %s", _exc)

    # --- Gather context, same as the API version ---
    try:
        client_context = _load_icp_context(company)
    except Exception:
        client_context = "(no client context found)"

    # Count scored observations via the rebuilt ledger (pulls from
    # linkedin_posts + matched local_posts). Replaces ruan_mei_load
    # which always returned 0 for multi-FOC companies after the
    # ruan_mei_state wipe. 2026-04-24.
    try:
        from backend.src.agents.cyrene import _load_cyrene_observations
        n_scored = len(_load_cyrene_observations(company, user_id=user_id))
    except Exception:
        n_scored = 0

    # Previous brief — Amphoreus Supabase (cyrene_briefs) is the canonical
    # source. Fly-local memory/{company}/cyrene_brief.json is deprecated;
    # those files still exist on the volume but are stale and mislead
    # Cyrene into treating active clients as cold-start.
    previous_brief = "No previous brief exists. This is the first Cyrene run for this client."
    try:
        from backend.src.db.amphoreus_supabase import get_latest_cyrene_brief
        prev_data = get_latest_cyrene_brief(company, user_id=user_id)
        if prev_data:
            previous_brief = json.dumps(prev_data, indent=2, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.debug("[CLI-Cyrene] previous_brief load failed (non-fatal): %s", exc)

    system_prompt = _SYSTEM_PROMPT.format(
        client_context=client_context,
        n_scored=n_scored,
        company=company,
        previous_brief=previous_brief,
    )

    # Build the user prompt. When we have a target user, prefix the
    # task with identity context so Cyrene's tool-calls + brief focus
    # on THIS person's voice/role/ICP rather than a company average.
    if target_user_info:
        _name = " ".join(filter(None, [
            (target_user_info.get("first_name") or "").strip(),
            (target_user_info.get("last_name") or "").strip(),
        ])) or target_user_info.get("email") or user_id
        _role_hint = (target_user_info.get("linkedin_url") or "").strip()
        _target_block = (
            f"TARGET FOC USER: {_name} (user_id={user_id}).\n"
            f"LinkedIn: {_role_hint or '(none on file)'}.\n"
            f"This brief is scoped to THIS individual's content at "
            f"{company} — not a company-wide strategy. Their role, voice, "
            f"network, and ICP are distinct from other FOCs at the same "
            f"company; average-the-whole-company recommendations will be "
            f"strategically useless for them. Use the transcript/observation "
            f"tools to ground the brief in what this specific person has "
            f"said, posted, and what's landed for them.\n\n"
        )
    else:
        _target_block = ""

    user_prompt = (
        f"{_target_block}"
        f"Run a strategic review for {company}"
        f"{' (scoped to ' + (target_user_info.get('first_name') or '').strip() + ')' if target_user_info else ''}. "
        f"Study the data across your tools, form your strategy from evidence, "
        f"and produce a comprehensive brief via submit_brief."
    )

    # --- Write temp files ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="cyrene_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    env_vars = {
        "CYRENE_COMPANY":         company,
        # Amphoreus-owned Supabase (cyrene_briefs, local_posts,
        # linkedin_posts, meetings) — primary data path for all of
        # Cyrene's rewired tools. Without these the MCP subprocess
        # can't reach the mirror and every tool returns empty.
        "AMPHOREUS_SUPABASE_URL": os.environ.get("AMPHOREUS_SUPABASE_URL", ""),
        "AMPHOREUS_SUPABASE_KEY": os.environ.get("AMPHOREUS_SUPABASE_KEY", ""),
        # Jacquard mirror creds — still used by a few helpers
        # (get_client_transcripts joins meeting_participants).
        "SUPABASE_URL":           os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_KEY":           os.environ.get("SUPABASE_KEY", ""),
        # FOC scoping — _resolve_linkedin_username reads this when no
        # explicit user_id kwarg is passed, which is how the query
        # tools pick the right creator at a multi-FOC company.
        "DATABASE_USER_UUID":     (user_id or ""),
        # External APIs used by a few tools.
        "OPENAI_API_KEY":         os.environ.get("OPENAI_API_KEY", ""),
        "PARALLEL_API_KEY":       os.environ.get("PARALLEL_API_KEY", ""),
    }
    env_vars = {k: v for k, v in env_vars.items() if v}

    mcp_config = {
        "mcpServers": {
            "cyrene-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "cyrene_server.py"),
                ],
                "env": env_vars,
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="cyrene_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    result_file = _PROJECT_ROOT / ".cyrene_cli_result.json"
    if result_file.exists():
        result_file.unlink()

    # Pull max turns from Cyrene constants to stay in sync.
    from backend.src.agents.cyrene import _CYRENE_MAX_TURNS

    # Model selection — env-configurable so bulk refreshes can route
    # around UPP refusals on Opus by switching to Sonnet without a
    # code change. The CLI's UPP error message itself recommends this:
    # "try running /model claude-sonnet-4-20250514 to switch models".
    # Default sonnet-4-5 — strong enough for Cyrene's analysis and
    # currently does not refuse the same engagement-data patterns.
    _cyrene_model = (os.environ.get("CYRENE_CLI_MODEL") or "sonnet").strip() or "sonnet"

    try:
        cmd = [
            "claude",
            "-p", user_prompt,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--output-format", "stream-json",
            "--verbose",
            "--model", _cyrene_model,
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(_CYRENE_MAX_TURNS),
        ]

        logger.info(
            "[CLI-Cyrene] Launching claude CLI (model=%s, max %d turns)...",
            _cyrene_model, _CYRENE_MAX_TURNS,
        )
        t0 = time.time()
        TIMEOUT_SEC = 3600  # Cyrene does 15-30 turns of deep analysis; 1 hour cap.

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(_PROJECT_ROOT),
                env=_cli_env(),
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — is Claude Code installed?")

        stderr_chunks: list[str] = []
        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, ""):
                    if not line:
                        break
                    stderr_chunks.append(line)
            except Exception:
                pass
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        stdout_lines: list[str] = []
        timed_out = False
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                stdout_lines.append(raw_line)
                if time.time() - t0 > TIMEOUT_SEC:
                    timed_out = True
                    proc.kill()
                    break
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            stderr_thread.join(timeout=5)

        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_chunks)
        returncode = proc.returncode
        elapsed = time.time() - t0
        logger.info("[CLI-Cyrene] CLI finished in %.1fs (exit %s)", elapsed, returncode)

        _record_cli_usage(
            stdout=full_stdout,
            cli_model=_cyrene_model,
            call_kind="cyrene_cli",
            client_slug=company,
            duration_ms=int(elapsed * 1000),
            error=None if returncode == 0 else f"exit {returncode}",
        )

        if timed_out:
            return {"_error": f"claude CLI timed out after {TIMEOUT_SEC}s"}

        # Diagnostic: extract assistant text + tool-use names from the
        # stream-json events so a failed run leaves enough breadcrumbs
        # in the log to debug. Without this, "exit 1" + 300 chars of
        # JSON metadata is unactionable.
        def _summarize_session(stdout: str) -> str:
            """Pull human-readable summary from claude CLI stream-json
            output: assistant text + tool calls in order. Truncated."""
            tool_calls: list[str] = []
            assistant_texts: list[str] = []
            for line in stdout.splitlines()[-200:]:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("type")
                if t == "assistant":
                    msg = obj.get("message") or {}
                    for blk in msg.get("content") or []:
                        if blk.get("type") == "text":
                            txt = (blk.get("text") or "").strip()
                            if txt:
                                assistant_texts.append(txt[:1000])
                        elif blk.get("type") == "tool_use":
                            name = blk.get("name") or "?"
                            inp_keys = list((blk.get("input") or {}).keys())
                            tool_calls.append(f"{name}({','.join(inp_keys[:5])})")
            parts = [f"tool_calls={tool_calls}"]
            if assistant_texts:
                parts.append(f"last_assistant_text={assistant_texts[-1][:600]!r}")
            return " | ".join(parts)

        if returncode != 0:
            session_summary = _summarize_session(full_stdout)
            logger.error(
                "[CLI-Cyrene] CLI failed (exit %s): %s",
                returncode, session_summary,
            )
            logger.info(
                "[CLI-Cyrene] stderr tail: %s | stdout tail: %s",
                full_stderr[-500:], full_stdout[-500:],
            )
            return {
                "_error": (
                    f"claude CLI exited {returncode}: "
                    f"{full_stderr[:300] or session_summary[:500]}"
                ),
            }

        if not result_file.exists():
            session_summary = _summarize_session(full_stdout)
            logger.error(
                "[CLI-Cyrene] No result file (submit_brief not called). %s",
                session_summary,
            )
            logger.info("[CLI-Cyrene] stdout tail: %s", full_stdout[-1500:])
            return {
                "_error": (
                    "Cyrene CLI did not call submit_brief. "
                    f"{session_summary[:300]}"
                ),
            }

        brief = json.loads(result_file.read_text(encoding="utf-8"))

        # Stamp metadata (same fields as run_strategic_review)
        brief["_company"] = company
        brief["_computed_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        brief["_duration_sec"] = round(elapsed, 1)
        if user_id:
            brief["_user_id"] = user_id

        # --- Persist to Amphoreus Supabase (primary) ---
        # The non-CLI path in cyrene.py saves here; the CLI path
        # historically only wrote to fly-local. That meant CLI-run
        # briefs (our default) never reached the Supabase store, so
        # Stelle/Tribbie/report couldn't read them. Mirror the API
        # path: save to Supabase with user_id scoping, then fall back
        # to fly-local if Supabase fails.
        try:
            from backend.src.db.amphoreus_supabase import save_cyrene_brief
            saved = save_cyrene_brief(
                company=company,
                brief=brief,
                created_by=None,
                user_id=user_id,
            )
            if saved:
                logger.info(
                    "[CLI-Cyrene] %s (user=%s): brief saved to Supabase (id=%s)",
                    company, user_id or "<company-wide>", saved.get("id"),
                )
        except Exception as _exc:
            logger.warning(
                "[CLI-Cyrene] Supabase save failed (%s, user=%s): %s",
                company, user_id, _exc,
            )

        # 2026-04-29: removed fly-local fallback persistence. Briefs
        # live exclusively in Amphoreus Supabase (``cyrene_briefs``);
        # the Supabase save above is the only persistence path.
        logger.info(
            "[CLI-Cyrene] %s: brief generated. "
            "topics_to_probe=%d, strategic_themes=%d, "
            "industry_signals=%d, dm_targets=%d",
            company,
            len(brief.get("topics_to_probe", [])),
            len(brief.get("strategic_themes", [])),
            len((brief.get("industry_context") or {}).get("current_signals") or []),
            len(brief.get("dm_targets", [])),
        )
        return brief

    finally:
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def use_cli() -> bool:
    """Check if CLI mode is enabled."""
    return os.environ.get("AMPHOREUS_USE_CLI", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Stream-json event translation
# ---------------------------------------------------------------------------

_CYCLE_COUNTERS: dict[int, int] = {}


def _translate_cli_event(event: dict, event_callback: Any) -> None:
    """Map a single CLI stream-json event to the existing event_callback.

    The web UI's event_callback expects ("status"|"text_delta"|"tool_call"|
    "tool_result", payload_dict). The CLI emits coarser chunks than the
    Anthropic SDK's per-token stream — you'll see one text_delta per
    assistant message, not per token — but it's enough for live progress.

    Emits a synthetic ``CLI cycle N`` status message on every new
    ``assistant`` turn so the web UI has the same "Stelle is thinking"
    cadence it gets in API mode (API-mode logs "Amphoreus cycle N done
    in Xs"). Each assistant message under stream-json corresponds to
    exactly one agent-loop iteration — receive messages, emit text +
    tool calls — so counting them gives the cycle count. Counter is
    keyed by id(event_callback) so multiple concurrent runs don't
    share the tally.
    """
    if not event_callback or not isinstance(event, dict):
        return
    etype = event.get("type")
    try:
        if etype == "assistant":
            msg = event.get("message") or {}
            blocks = msg.get("content") or []
            # Bump the cycle counter and emit it BEFORE processing the
            # blocks, so the UI sees "cycle N started" above its
            # text/tool events and can group visually.
            key = id(event_callback)
            n = _CYCLE_COUNTERS.get(key, 0) + 1
            _CYCLE_COUNTERS[key] = n
            # Count what's in this assistant message for a useful summary.
            n_text = sum(1 for b in blocks if b.get("type") == "text")
            n_tools = sum(1 for b in blocks if b.get("type") == "tool_use")
            tool_names = ", ".join(
                (b.get("name") or "?") for b in blocks if b.get("type") == "tool_use"
            )
            # Pull token usage if the event carries it.
            usage_bits: list[str] = []
            try:
                u = (event.get("message") or {}).get("usage") or {}
                if u:
                    inp = u.get("input_tokens", 0)
                    outp = u.get("output_tokens", 0)
                    cread = u.get("cache_read_input_tokens", 0)
                    cwrite = u.get("cache_creation_input_tokens", 0)
                    pct = int(100 * cread / (cread + cwrite + inp)) if (cread + cwrite + inp) else 0
                    usage_bits.append(
                        f"in={inp} out={outp} cache_read={cread} cache_write={cwrite} ({pct}% cached)"
                    )
            except Exception:
                pass
            summary = f"Amphoreus cycle {n} — text={n_text} tools={n_tools}"
            if tool_names:
                summary += f" [{tool_names}]"
            if usage_bits:
                summary += " — " + " ".join(usage_bits)
            event_callback("status", {"message": summary})

            for block in blocks:
                btype = block.get("type")
                if btype == "text":
                    txt = block.get("text") or ""
                    if txt:
                        event_callback("text_delta", {"text": txt})
                elif btype == "tool_use":
                    inp = block.get("input") or {}
                    try:
                        input_summary = json.dumps(inp, default=str)[:300]
                    except Exception:
                        input_summary = str(inp)[:300]
                    event_callback("tool_call", {
                        "name": block.get("name", "") or "",
                        "arguments": input_summary,
                    })
        elif etype == "user":
            msg = event.get("message") or {}
            for block in msg.get("content") or []:
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                # 8000-char cap (was 500, raised 2026-04-28). Irontomb's
                # tool_result contains the gestalt + anchors[] + a
                # _prior_reactions trajectory of the last 5 calls; even
                # a single multi-anchor response easily exceeds 500.
                # The model itself receives the full content — this cap
                # only affects what we LOG to run_events for post-hoc
                # audit. The 500-char cap was making truncated JSON
                # impossible to parse during diagnosis runs and made
                # the anchor schema-floor look unenforced when it was
                # actually fine. 8000 holds a typical reaction +
                # trajectory + headroom.
                event_callback("tool_result", {"result": str(content)[:8000]})
        elif etype == "system":
            sub = event.get("subtype") or ""
            if sub:
                event_callback("status", {"message": f"CLI system: {sub}"})
    except Exception:
        # Event translation must never break the run
        pass
