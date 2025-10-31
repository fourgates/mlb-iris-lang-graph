"""
Agent tools wrapping shared logic functions for the planner agent.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from .logic import (
    find_player_id,
    fetch_player_stats,
    generate_grounded_answer,
)


@tool
def search_for_player(player_name: str) -> int | str:
    """Searches for an MLB player by name and returns their unique player ID."""
    pid = find_player_id(player_name)
    return pid if pid is not None else "not_found"


@tool
def get_player_statistics(player_id: int) -> Dict[str, Any] | str:
    """Fetches season statistics for a given player ID."""
    stats = fetch_player_stats(player_id)
    return stats if stats is not None else "no_stats"


@tool
def query_document_knowledge_base(query: str) -> str:
    """Answers questions about MLB rules, policies, and definitions using a document knowledge base."""
    return generate_grounded_answer(query)


__all__ = [
    "search_for_player",
    "get_player_statistics",
    "query_document_knowledge_base",
]


