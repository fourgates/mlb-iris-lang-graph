"""
Subgraph builders for MLB assistant flows.

We expose builder functions that accept the `State` TypedDict class from the
caller to avoid circular imports with `app.graph`.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

# Import node functions directly
from .nodes import (
    player_search_node,
    player_stats_node,
    answer_player_stats_query,
    generate_rag_answer,
)


def build_player_stats_subgraph(State):
    """Create PLAYER_STATS subgraph: search -> stats -> answer -> END."""
    builder = StateGraph(State)
    builder.add_node("player_search", player_search_node)
    builder.add_node("player_stats", player_stats_node)
    builder.add_node("answer_player_stats_query", answer_player_stats_query)

    builder.add_edge(START, "player_search")
    builder.add_edge("player_search", "player_stats")
    builder.add_edge("player_stats", "answer_player_stats_query")
    builder.add_edge("answer_player_stats_query", END)

    return builder.compile()


def build_document_qa_subgraph(State):
    """Create DOCUMENT_QA subgraph: generate_rag_answer -> END."""
    builder = StateGraph(State)
    builder.add_node("generate_rag_answer", generate_rag_answer)
    builder.add_edge(START, "generate_rag_answer")
    builder.add_edge("generate_rag_answer", END)
    return builder.compile()


