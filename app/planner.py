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
    "Be concise and accurate."
)

_PLANNER_CACHE: Any = None


def get_planner_agent() -> Any:
    """Lazily build and cache the planner agent. Falls back to a stub only if LangChain is missing."""
    global _PLANNER_CACHE
    if _PLANNER_CACHE is not None:
        return _PLANNER_CACHE

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
