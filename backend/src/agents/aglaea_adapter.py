"""Aglaea adapter — wraps aglaea.py for the API layer."""

import logging
import os

from backend.src.utils.langfuse_tracing import traced

logger = logging.getLogger(__name__)


@traced(name="aglaea.generate_briefing", kind="generation")
def run_aglaea(client_name: str, company: str, job_id: str | None = None) -> str | None:
    """Run Aglaea briefing generation, streaming Pi events to the SSE queue."""
    from backend.src.agents.aglaea import generate_briefing
    from backend.src.db import vortex

    vortex.ensure_dirs(company)

    event_callback = None
    if job_id:
        from backend.src.services.job_manager import emit_event
        from backend.src.core.events import (
            status_event,
            text_delta_event,
            thinking_event,
            tool_call_event,
            tool_result_event,
        )

        def event_callback(kind: str, data: dict) -> None:  # type: ignore[misc]
            if kind == "status":
                emit_event(job_id, status_event(data.get("message", "")))
            elif kind == "compaction":
                emit_event(job_id, status_event(data.get("message", "Context compaction…")))
            elif kind == "text_delta":
                emit_event(job_id, text_delta_event(data.get("text", "")))
            elif kind == "thinking":
                emit_event(job_id, thinking_event(data.get("text", "")))
            elif kind == "tool_call":
                emit_event(
                    job_id,
                    tool_call_event(
                        name=data.get("name", ""),
                        args={"summary": data.get("arguments", "")},
                    ),
                )
            elif kind == "tool_result":
                emit_event(
                    job_id,
                    tool_result_event(
                        name=data.get("name", ""),
                        result=data.get("result", ""),
                        is_error=data.get("is_error", False),
                    ),
                )
            elif kind == "error":
                emit_event(job_id, status_event(f"Error: {data.get('message', '')}"))

    result_path = generate_briefing(client_name, company, event_callback=event_callback)

    if result_path and os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            return f.read()
    return result_path
