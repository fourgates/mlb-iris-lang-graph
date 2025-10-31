"""
LangGraph graph definition for the MLB assistant agent.

This module defines the graph structure and wires together
all the node functions from nodes.py.
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    decide_route,
    hello_node,
    planner_node,
    route_query_node,
)
from .state import State
from .subgraphs import build_document_qa_subgraph, build_player_stats_subgraph

_graph = StateGraph(State)
_graph.add_node("router", route_query_node)
_graph.add_node("hello", hello_node)
_graph.add_node("planner", planner_node)

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
        "MULTI_DOMAIN": "planner",
        "HELLO": "hello",
    },
)

# Define the hello path (error handling)
_graph.add_edge("hello", END)
_graph.add_edge("planner", END)


agent = _graph.compile(name="Grounding Chat Graph")
