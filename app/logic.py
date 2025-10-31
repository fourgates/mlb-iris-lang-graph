"""
Core business logic for MLB assistant, decoupled from LangGraph node wrappers.

These functions are pure (or side-effect limited to service calls) and can be
reused by subgraphs or agent tools.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from google.api_core.exceptions import ResourceExhausted
from vertexai import rag

from app.utils.mlb_tools import search_player, get_player_stats
from .services import grounding_tool, llm_langchain, llm_native_grounding


def find_player_id(player_name: str) -> Optional[int]:
    """
    Search for a player by name and choose the best match.

    Selection heuristic:
    - Exact case-insensitive full-name match
    - Partial substring match
    - First result as fallback
    """
    result = search_player(player_name, only_active=True)
    players = result.get("players", []) if isinstance(result, dict) else []
    if not players:
        return None

    target = player_name.lower().strip()
    # Exact match
    for p in players:
        if str(p.get("full_name", "")).lower().strip() == target:
            return int(p["id"])  # type: ignore[index]

    # Partial match
    for p in players:
        if target in str(p.get("full_name", "")).lower():
            return int(p["id"])  # type: ignore[index]

    # Fallback
    return int(players[0]["id"])  # type: ignore[index]


def fetch_player_stats(player_id: int) -> Dict[str, Any] | None:
    """
    Fetch season statistics for a given player id.
    Returns a dictionary (or None on failure).
    """
    try:
        return get_player_stats(player_id)
    except Exception as exc:  # Defensive: keep node wrappers simple
        logging.warning("fetch_player_stats failed for %s: %s", player_id, exc)
        return None


def generate_player_stats_answer(query: str, stats: Dict[str, Any] | None) -> str:
    """
    Construct a concise answer using provided hitting stats as context.
    Falls back to answering the query directly if no stats are provided.
    """
    hitting = (
        (stats or {}).get("stats", {}).get("hitting_season", {})
        if isinstance(stats, dict)
        else {}
    )

    if hitting:
        prompt = (
            "You are an expert MLB analyst. Here is the player's season hitting data:\n"
            f"AVG: {hitting.get('avg', '.000')}\n"
            f"HR: {hitting.get('home_runs', 0)}\n"
            f"OPS: {hitting.get('ops', '.000')}\n"
            f"RBI: {hitting.get('rbi', 0)}\n\n"
            "Based on this data, answer the user's question.\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        prompt = query

    response = llm_langchain.invoke(prompt)
    content = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    return content


def generate_grounded_answer(query: str) -> str:
    """
    Generate a grounded answer using Vertex AI native grounding, with basic retries
    and inline citations formatting.
    """
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = llm_native_grounding.generate_content(
                query, tools=[grounding_tool]
            )

            if not response.candidates:
                return (
                    response.text or "I could not find any information on that topic."
                )

            candidate = response.candidates[0]

            if not getattr(candidate, "grounding_metadata", None):
                return (
                    response.text or "I could not find any information on that topic."
                )

            grounding_supports = list(candidate.grounding_metadata.grounding_supports)
            grounding_chunks = list(candidate.grounding_metadata.grounding_chunks)
            if not grounding_supports:
                return (
                    response.text or "I could not find any information on that topic."
                )

            if not getattr(candidate, "content", None) or not getattr(
                candidate.content, "parts", None
            ):
                return (
                    response.text or "I could not find any information on that topic."
                )

            rag_response = rag.add_inline_citations_and_references(
                original_text_str=candidate.content.parts[0].text,
                grounding_supports=grounding_supports,
                grounding_chunks=grounding_chunks,
            )

            final_content = rag_response.cited_text
            if rag_response.final_bibliography:
                final_content += "\n\n**Sources:**\n" + rag_response.final_bibliography
            return final_content

        except ResourceExhausted:
            if attempt + 1 == max_retries:
                return (
                    "The service is currently busy. Please try again in a few moments."
                )
            time.sleep(base_delay * (2**attempt))

    return "An unexpected error occurred after multiple retries."


__all__ = [
    "find_player_id",
    "fetch_player_stats",
    "generate_player_stats_answer",
    "generate_grounded_answer",
]
