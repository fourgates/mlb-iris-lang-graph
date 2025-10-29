"""
LangGraph graph definition and State schema for the MLB assistant agent.

This module defines the graph structure, state schema, and wires together
all the node functions from nodes.py.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .nodes import (
    chat_player,
    decide_route,
    generate_rag_answer,
    hello_node,
    player_search_node,
    player_stats_node,
    route_query_node,
)


class State(TypedDict):
    messages: list
    # 'snippets' is no longer needed as we get a synthesized answer directly
    player_id: int | None
    stats: dict | None
    extracted_name: str | None
    extracted_team: str | None
    route: str  # Either "PLAYER_STATS", "DOCUMENT_QA", or "HELLO"


# Build the graph
_graph = StateGraph(State)
_graph.add_node("router", route_query_node)
_graph.add_node("hello", hello_node)
_graph.add_node("generate_rag_answer", generate_rag_answer)
_graph.add_node("player_search", player_search_node)
_graph.add_node("player_stats", player_stats_node)
_graph.add_node("chat_player", chat_player)

# Set entry point to router
_graph.set_entry_point("router")

# Add conditional routing from router
_graph.add_conditional_edges(
    "router",
    decide_route,
    {
        "DOCUMENT_QA": "generate_rag_answer",
        "PLAYER_STATS": "player_search",
        "HELLO": "hello",
    },
)

# Define the player tool-use path
_graph.add_edge("player_search", "player_stats")
_graph.add_edge("player_stats", "chat_player")
_graph.add_edge("chat_player", END)

# Define the RAG path
_graph.add_edge("generate_rag_answer", END)

# Define the hello path (error handling)
_graph.add_edge("hello", END)


agent = _graph.compile(name="Grounding Chat Graph")
