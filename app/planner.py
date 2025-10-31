"""
Planner agent using LangChain's create_agent with shared tools.

This module avoids importing LangChain at import time to keep tests working
without the optional `langchain` package installed. Use `get_planner_agent()`
to lazily construct the agent.
"""

from __future__ import annotations

from .services import llm_langchain
from .agent_tools import (
    search_for_player,
    get_player_statistics,
    query_document_knowledge_base,
)


SYSTEM_PROMPT = (
    "You are an MLB assistant planner. Use tools to answer multi-domain questions "
    "that may require both player statistics and policy/rules knowledge. "
    "Be concise and accurate."
)

_PLANNER_CACHE = None


def get_planner_agent():
    """Lazily build and cache the planner agent. Falls back to a stub if LangChain is missing."""
    global _PLANNER_CACHE
    if _PLANNER_CACHE is not None:
        return _PLANNER_CACHE

    try:
        from langchain.agents import create_agent  # type: ignore[import-not-found]

        agent = create_agent(
            model=llm_langchain,
            tools=[search_for_player, get_player_statistics, query_document_knowledge_base],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception:
        # Minimal stub agent for environments without `langchain` installed
        class _Stub:
            def invoke(self, x, *args, **kwargs):
                return {"messages": []}

        agent = _Stub()

    _PLANNER_CACHE = agent
    return agent


__all__ = ["get_planner_agent"]


