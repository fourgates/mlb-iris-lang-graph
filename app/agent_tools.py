"""
Agent tools wrapping shared logic functions for the planner agent.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from .logic import (
    fetch_player_stats,
    find_player_id,
    generate_grounded_answer,
)


@tool
def search_for_player(player_name: str) -> int | str:
    """Searches for an MLB player by name and returns their unique player ID."""
    logging.info("[tool] search_for_player called with player_name=%r", player_name)
    pid = find_player_id(player_name)
    result = pid if pid is not None else "not_found"
    logging.info("[tool] search_for_player result: %s", result)
    return result


@tool
def get_player_statistics(player_id: int) -> dict[str, Any] | str:
    """Fetches season statistics for a given player ID."""
    logging.info("[tool] get_player_statistics called with player_id=%s", player_id)
    stats = fetch_player_stats(player_id)
    result = stats if stats is not None else "no_stats"
    if isinstance(result, dict):
        # Log a summary instead of the full dict
        hitting = (
            result.get("stats", {}).get("hitting_season", {})
            if isinstance(result, dict)
            else {}
        )
        logging.info(
            "[tool] get_player_statistics result: player_id=%s, has_stats=%s, avg=%s, hr=%s",
            player_id,
            isinstance(result, dict),
            hitting.get("avg") if hitting else None,
            hitting.get("home_runs") if hitting else None,
        )
    else:
        logging.info("[tool] get_player_statistics result: %s", result)
    return result


@tool
def query_document_knowledge_base(query: str) -> str:
    """Answers questions about MLB rules, policies, and definitions using a document knowledge base."""
    logging.info(
        "[tool] query_document_knowledge_base called with query=%r",
        query[:100] if len(query) > 100 else query,
    )
    result = generate_grounded_answer(query)
    logging.info(
        "[tool] query_document_knowledge_base result: length=%d chars", len(result)
    )
    return result


__all__ = [
    "get_player_statistics",
    "query_document_knowledge_base",
    "search_for_player",
]
