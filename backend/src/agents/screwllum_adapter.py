"""Screwllum adapter — wraps screwllum.py for the API layer."""

import logging

from backend.src.utils.langfuse_tracing import traced

logger = logging.getLogger(__name__)


@traced(name="screwllum.run_programmatic", kind="generation")
def run_screwllum(company: str, prompt: str | None = None, job_id: str | None = None) -> str | None:
    """Run Screwllum content strategy generation."""
    from backend.src.agents.screwllum import run_programmatic

    if job_id:
        from backend.src.services.job_manager import emit_event
        from backend.src.core.events import status_event, text_delta_event
        emit_event(job_id, status_event(f"Generating content strategy for {company}..."))

        def output_callback(chunk: str) -> None:
            emit_event(job_id, text_delta_event(chunk))
    else:
        def output_callback(chunk: str) -> None:
            print(chunk, end="", flush=True)

    result = run_programmatic(
        client_name=company,
        output_callback=output_callback,
        primary_goal=prompt or "",
    )

    if isinstance(result, dict):
        return result.get("strategy", str(result))
    return str(result) if result else None
