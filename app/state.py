"""
State schema definition for the MLB assistant agent.

This module defines the State TypedDict that represents the graph's state.
Separated from graph.py to avoid circular import dependencies.
"""

from typing import Literal, TypedDict

from langchain_core.messages import AnyMessage


class State(TypedDict):
    """State schema for the MLB assistant LangGraph agent."""

    messages: list[AnyMessage]
    # 'snippets' is no longer needed as we get a synthesized answer directly
    player_id: int | None
    stats: dict | None
    extracted_name: str | None
    extracted_team: str | None
    route: str  # Either "PLAYER_STATS", "DOCUMENT_QA", "MULTI_DOMAIN", or "HELLO"
    replan_attempts: int  # Track number of replan attempts
    verification_status: Literal["OK", "REPLAN"] | None  # Verification result
    verification_reason: (
        str | None
    )  # Explanation of why verification failed (if REPLAN)
