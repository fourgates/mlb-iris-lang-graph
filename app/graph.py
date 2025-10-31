"""
LangGraph graph definition and State schema for the MLB assistant agent.

This module defines the graph structure, state schema, and wires together
all the node functions from nodes.py.
"""

from typing import TypedDict
from langchain_core.messages import AnyMessage

from langgraph.graph import END, StateGraph


class State(TypedDict):
    messages: list[AnyMessage]
    # 'snippets' is no longer needed as we get a synthesized answer directly
    player_id: int | None
    stats: dict | None
    extracted_name: str | None
    extracted_team: str | None
    route: str  # Either "PLAYER_STATS", "DOCUMENT_QA", or "HELLO"


# Import nodes after State is defined (needed for type hints and to avoid circular import issues)
from .nodes import (
    decide_route,
    hello_node,
    route_query_node,
)
from .subgraphs import build_player_stats_subgraph, build_document_qa_subgraph


_graph = StateGraph(State)
_graph.add_node("router", route_query_node)
_graph.add_node("hello", hello_node)

# Compile subgraphs and add as nodes
player_stats_sg = build_player_stats_subgraph(State)
document_qa_sg = build_document_qa_subgraph(State)
_graph.add_node("player_stats_sg", player_stats_sg)
_graph.add_node("document_qa_sg", document_qa_sg)

# Set entry point to router
_graph.set_entry_point("router")

# Add conditional routing from router
_graph.add_conditional_edges(
    "router",
    decide_route,
    {
        "DOCUMENT_QA": "document_qa_sg",
        "PLAYER_STATS": "player_stats_sg",
        "HELLO": "hello",
    },
)

# Define the hello path (error handling)
_graph.add_edge("hello", END)


agent = _graph.compile(name="Grounding Chat Graph")
