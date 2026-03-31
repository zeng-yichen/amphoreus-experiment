"""LLM-as-judge grader — uses Claude to evaluate output quality against rubrics."""

import asyncio
import logging
import threading
from typing import Any, Literal

from pydantic import BaseModel

from backend.src.models.eval import EvalTrace, GradeResult

logger = logging.getLogger(__name__)


class RubricScore(BaseModel):
    item: str
    passed: bool
    reasoning: str


class GradingResponse(BaseModel):
    overall_passed: bool
    overall_reasoning: str
    rubric_scores: list[RubricScore]


_client = None
_client_lock = threading.Lock()


def _get_grader_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from anthropic import Anthropic
        _client = Anthropic()
        return _client


def grade_output_quality(
    trace: EvalTrace,
    rubric: list[str],
    grading_mode: Literal["strict", "lenient"] = "lenient",
) -> GradeResult:
    """Grade agent output against a rubric using Claude as judge."""
    if not rubric:
        return GradeResult(passed=True, reason="No rubric items specified", grader="llm")

    prompt = _build_grading_prompt(trace, rubric, grading_mode)

    try:
        client = _get_grader_client()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text

        passed = "PASS" in text.upper().split("\n")[0]
        return GradeResult(
            passed=passed,
            reason=text[:500],
            grader="llm",
            rubric_scores={item: True for item in rubric} if passed else None,
        )
    except Exception as e:
        return GradeResult(passed=False, reason=f"LLM grading failed: {e}", grader="llm")


def _build_grading_prompt(
    trace: EvalTrace,
    rubric: list[str],
    grading_mode: Literal["strict", "lenient"],
) -> str:
    rubric_text = "\n".join(f"- {item}" for item in rubric)
    mode_instruction = (
        "STRICT mode: ALL rubric items must be fully satisfied for a PASS."
        if grading_mode == "strict"
        else "LENIENT mode: The output should reasonably address the rubric items. Minor issues are acceptable."
    )

    return f"""You are an AI output quality grader. Evaluate the following agent output against the rubric.

## Agent Output
{trace.user_response or "(No response generated)"}

## Rubric Items
{rubric_text}

## Grading Mode
{mode_instruction}

## Instructions
1. Evaluate each rubric item individually
2. Start your response with either PASS or FAIL on the first line
3. Provide reasoning for each score
4. Give an overall assessment

Respond with your evaluation."""
