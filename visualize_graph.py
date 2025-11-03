#!/usr/bin/env python3
"""
Visualize the LangGraph agent graph structure.

This script generates a visual representation of the graph using LangGraph's
built-in visualization capabilities. Outputs both Mermaid syntax and PNG image.
Also visualizes subgraphs separately to show their internal structure.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agent import agent
from app.planner import get_planner_agent
from app.subgraphs import build_document_qa_subgraph, build_player_stats_subgraph
from app.state import State


def visualize_graph():
    """Generate and save graph visualization."""
    try:
        # Get the drawable graph
        drawable_graph = agent.get_graph()

        # Generate Mermaid syntax for main graph
        mermaid_syntax = drawable_graph.draw_mermaid()
        print("=" * 80)
        print("MAIN GRAPH - MERMAID SYNTAX:")
        print("=" * 80)
        print(mermaid_syntax)
        print()

        # Generate PNG image for main graph
        output_path = project_root / "graph_visualization.png"
        print(f"Generating PNG visualization for main graph...")
        png_data = drawable_graph.draw_mermaid_png()

        # Save PNG
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"✅ Main graph PNG saved to: {output_path}")
        print()

        # Visualize subgraphs separately
        print("=" * 80)
        print("SUBGRAPH VISUALIZATIONS:")
        print("=" * 80)
        print()

        # Player Stats Subgraph
        player_stats_sg = build_player_stats_subgraph(State)
        player_stats_graph = player_stats_sg.get_graph()
        player_stats_mermaid = player_stats_graph.draw_mermaid()

        print("PLAYER_STATS Subgraph - Mermaid:")
        print("-" * 80)
        print(player_stats_mermaid)
        print()

        player_stats_png_path = (
            project_root / "graph_visualization_player_stats_subgraph.png"
        )
        player_stats_png = player_stats_graph.draw_mermaid_png()
        with open(player_stats_png_path, "wb") as f:
            f.write(player_stats_png)
        print(f"✅ Player Stats subgraph PNG saved to: {player_stats_png_path}")
        print()

        # Document QA Subgraph
        document_qa_sg = build_document_qa_subgraph(State)
        document_qa_graph = document_qa_sg.get_graph()
        document_qa_mermaid = document_qa_graph.draw_mermaid()

        print("DOCUMENT_QA Subgraph - Mermaid:")
        print("-" * 80)
        print(document_qa_mermaid)
        print()

        document_qa_png_path = (
            project_root / "graph_visualization_document_qa_subgraph.png"
        )
        document_qa_png = document_qa_graph.draw_mermaid_png()
        with open(document_qa_png_path, "wb") as f:
            f.write(document_qa_png)
        print(f"✅ Document QA subgraph PNG saved to: {document_qa_png_path}")
        print()

        # Visualize Planner Agent
        print("=" * 80)
        print("PLANNER AGENT VISUALIZATION:")
        print("=" * 80)
        print()

        planner_agent = get_planner_agent()
        if hasattr(planner_agent, "get_graph"):
            planner_graph = planner_agent.get_graph()
            planner_mermaid = planner_graph.draw_mermaid()

            print("Planner Agent (create_agent) - Mermaid:")
            print("-" * 80)
            print(planner_mermaid)
            print()

            planner_png_path = project_root / "graph_visualization_planner_agent.png"
            planner_png = planner_graph.draw_mermaid_png()
            with open(planner_png_path, "wb") as f:
                f.write(planner_png)
            print(f"✅ Planner agent PNG saved to: {planner_png_path}")
            print()

            # Extract tool information
            print("Planner Agent Tools:")
            print("-" * 80)
            from app.agent_tools import (
                search_for_player,
                get_player_statistics,
                query_document_knowledge_base,
            )

            tools = [
                ("search_for_player", search_for_player),
                ("get_player_statistics", get_player_statistics),
                ("query_document_knowledge_base", query_document_knowledge_base),
            ]

            for tool_name, tool_func in tools:
                if hasattr(tool_func, "name"):
                    name = tool_func.name
                else:
                    name = tool_name

                if hasattr(tool_func, "description"):
                    desc = tool_func.description
                else:
                    desc = "No description available"

                print(f"  • {name}")
                print(f"    {desc}")
                print()
        else:
            print("⚠️  Planner agent does not have get_graph() method")
            print("    (May be a stub agent if LangChain is not installed)")
            print()

        print("=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print("✅ All single-domain paths (PLAYER_STATS, DOCUMENT_QA) include")
        print("   verification nodes INSIDE their subgraphs.")
        print("✅ Multi-domain path (MULTI_DOMAIN) includes verification")
        print("   AFTER the planner node in the main graph.")
        print("✅ After subgraphs complete, the main graph checks verification_status")
        print("   and routes to END (if OK) or planner (if REPLAN needed).")
        print("✅ Planner agent (create_agent) uses a ReAct loop:")
        print("   model → tools → model (loop until finish)")
        print("   Tools: search_for_player, get_player_statistics,")
        print("          query_document_knowledge_base")
        print()
        print("You can also view the graph interactively in LangGraph Studio:")
        print("  langgraph dev")
        print("  Then visit: https://smith.langchain.com/studio/")

    except Exception as e:
        print(f"❌ Error visualizing graph: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    visualize_graph()
