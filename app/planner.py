"""
Planner agent using LangChain's create_agent with shared tools.

This module avoids importing LangChain at import time to keep tests working
without the optional `langchain` package installed. Use `get_planner_agent()`
to lazily construct the agent.
"""

from __future__ import annotations

import logging
from typing import Any

from .agent_tools import (
    get_player_statistics,
    query_document_knowledge_base,
    search_for_player,
)
from .services import llm_langchain

SYSTEM_PROMPT = (
    "You are an MLB assistant planner. Use tools to answer multi-domain questions "
    "that may require both player statistics and policy/rules knowledge. "
    "Be concise and accurate.\n\n"
    "IMPORTANT: When answering questions about player statistics:\n"
    "- Always include the specific statistic requested (e.g., batting average, home runs)\n"
    "- Always include the season/context (e.g., 'for the 2024 season' or 'this season')\n"
    "- Answer the user's question directly - don't ask follow-up questions or provide unrelated stats\n"
    "- If stats are provided in context, use them directly without calling tools again\n\n"
    "CRITICAL: When using query_document_knowledge_base tool:\n"
    "- The tool returns answers with inline citations (e.g., [0], [1]) and a Sources section\n"
    "- You MUST preserve ALL citations and the Sources section in your final answer\n"
    "- Do NOT remove inline citations (e.g., [0], [1]) from the text\n"
    "- Do NOT remove the Sources section at the end\n"
    "- Citations are formatted as: [number] in text and 'Sources: [0] document.pdf, p.23' at the end\n"
    "- When combining information from multiple sources, preserve all citations from all sources"
)

_PLANNER_CACHE: Any = None


def get_planner_agent() -> Any:
    """Lazily build and cache the planner agent. Falls back to a stub only if LangChain is missing."""
    global _PLANNER_CACHE
    if _PLANNER_CACHE is not None:
        return _PLANNER_CACHE

    agent: Any
    try:
        from langchain.agents import create_agent

        logging.info("[planner] Creating agent with create_agent (LangChain v1)...")
        agent = create_agent(
            model=llm_langchain,
            tools=[
                search_for_player,
                get_player_statistics,
                query_document_knowledge_base,
            ],
            system_prompt=SYSTEM_PROMPT,
        )
        logging.info("[planner] Agent created successfully")
    except ImportError as e:
        # Only catch ImportError for missing langchain package
        logging.warning(
            "[planner] LangChain not available (ImportError: %s), using stub agent",
            e,
        )

        # Minimal stub agent for environments without `langchain` installed
        class _Stub:
            def invoke(self, x: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"messages": []}

        agent = _Stub()
    except Exception as e:
        # For any other exception, log and re-raise - fail fast
        logging.error(
            "[planner] Failed to create agent (unexpected error: %s)",
            e,
            exc_info=True,
        )
        raise

    _PLANNER_CACHE = agent
    return agent


__all__ = ["get_planner_agent"]
