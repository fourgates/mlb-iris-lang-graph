"""
Verification module for judging if answers are complete.

Provides a reusable judge_answer() function that uses LLM-as-judge
to determine if a user's question was fully answered.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .services import llm_langchain

CURRENT_YEAR = datetime.now().year

JUDGE_PROMPT = f"""You are evaluating whether an AI assistant's answer fully addresses a user's question.

IMPORTANT CONTEXT: The current year is {CURRENT_YEAR}. When evaluating answers about MLB statistics, references to the {CURRENT_YEAR} season are VALID and ACCURATE. Do NOT reject answers that correctly reference the {CURRENT_YEAR} season.

User Question: {{query}}

Assistant Answer: {{answer}}

Evaluate if the assistant's answer COMPLETELY addresses the user's question. Consider:
- Are all parts of the question answered?
- Is the information accurate and relevant?
- Is the answer complete (not partial or vague)?
- Does it include appropriate temporal context (season/year)?

CRITICAL: If the answer references the {CURRENT_YEAR} season, this is CORRECT - do not reject it based on the year.

If the answer is incomplete, provide a brief explanation of what is missing or unclear."""


class VerificationResult(BaseModel):
    """Structured output schema for verification judgment."""

    status: Literal["OK", "REPLAN"] = Field(
        description="'OK' if answer is complete, 'REPLAN' if incomplete or missing information"
    )
    reason: str | None = Field(
        default=None,
        description="Brief explanation of why REPLAN is needed (required if status is REPLAN, optional if OK)",
    )


def judge_answer(
    query: str, answer: str
) -> dict[str, Literal["OK", "REPLAN"] | str | None]:
    """
    Use LLM-as-judge to determine if an answer fully addresses the query.

    Args:
        query: The original user question
        answer: The assistant's answer to evaluate

    Returns:
        dict with "status" key: "OK" if answer is complete, "REPLAN" if incomplete.
        Also includes "reason" key with explanation if REPLAN.
    """
    logging.info(
        "[verification] Judging answer for query=%r",
        query[:100] if len(query) > 100 else query,
    )

    prompt = JUDGE_PROMPT.format(query=query, answer=answer)

    try:
        # Use structured output for reliable JSON parsing
        model_with_structure = llm_langchain.with_structured_output(VerificationResult)
        result = model_with_structure.invoke(prompt)

        # result is a VerificationResult Pydantic model
        # Handle both dict and Pydantic model return types
        if isinstance(result, VerificationResult):
            status = result.status
            reason = result.reason
        elif isinstance(result, dict):
            status = result.get("status", "REPLAN")
            reason = result.get("reason")
        else:
            status = getattr(result, "status", "REPLAN")
            reason = getattr(result, "reason", None)

        logging.info(
            "[verification] Judge result: status=%s, reason=%r",
            status,
            reason,
        )
        return {"status": status, "reason": reason}

    except Exception as e:
        logging.error("[verification] Failed to judge answer: %s", e, exc_info=True)
        # On error, default to REPLAN to be safe
        return {"status": "REPLAN", "reason": f"Verification error: {e!s}"}


__all__ = ["judge_answer"]
