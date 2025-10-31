# MLB IRIS LangGraph – Idealized Flow Diagram

## Graph Flow

```mermaid
flowchart TD
    Start([User Query]) --> Router["router\nLLM route + entity extract"]
    Router -->|decide_route| Route{Route Decision}

    %% Primary routes (target entry nodes, not subgraph ids)
    Route -->|DOCUMENT_QA| RAG
    Route -->|PLAYER_STATS| PS
    Route -->|"EBIS"| TXExtract
    Route
    Route -->|FALLBACK| Hello["Help/Recovery"]

    %% RAG path
    subgraph RAGSG [DOCUMENT_QA Subgraph]
        RAG["generate_rag_answer - Vertex AI Grounding"]
        RAG --> VerifyRAG{Answer OK?}
        VerifyRAG -->|OK| RAG_END([END])
        VerifyRAG -->|CLARIFY| Clarify
        VerifyRAG
    end

    %% Player Stats path
    subgraph STATSSG [PLAYER_STATS Subgraph]
        PS["player_search"]
        PS -->|multiple matches| ConfirmPlayer["confirm_player - interrupt"]
        PS -->|single match| PlayerStats["player_stats"]
        ConfirmPlayer --> PlayerStats
        PlayerStats --> AnswerStats["answer_player_stats_query"]
        AnswerStats --> VerifyStats{Answer OK?}
        VerifyStats -->|OK| STATS_END([END])
        VerifyStats -->|CLARIFY| Clarify
        VerifyStats
    end

    subgraph TXSG["EBIS Subgraph"]
        TXExtract["extract_tx_entities"]
        TXExtract -->|ambiguous| ConfirmTx["confirm_transaction - interrupt"]
        TXExtract -->|unambiguous| FetchTx["fetch_transactions/contracts"]
        ConfirmTx --> FetchTx
        FetchTx --> AnswerTx["answer_transactions_query"]
        AnswerTx --> VerifyTx{Answer OK?}
        VerifyTx -->|OK| TX_END([END])
        VerifyTx -->|CLARIFY| Clarify
        VerifyTx
    end

    %% Multi-domain orchestrator (agent with tools)
    subgraph DD [Multi Domain Subgraph]
		Planner["Planner Agent Node \n Tools: rag, stats, tx"]
        ToolRAG["rag_query tool"]
        ToolStats["search_player tool\nget_player_stats tool"]****
        ToolTx["find_transactions tool\nget_contract tool"]
        Compose["compose_and_ground_answer"]
        Compose --> VerifyPlan{Answer OK?}
        VerifyPlan -->|OK| PLAN_END([END])
        VerifyPlan -->|CLARIFY| Clarify
        VerifyPlan
   end

    %% Shared clarify (interrupt) node and approvals
    Clarify["clarify - interrupt"] --> Router["Router - LLM route + entity extract"]

    %% HITL approvals (policy-based)
    ToolTx -->|sensitive action| ApproveTx["HITL approval - interrupt"]
    ApproveTx --> ToolTx

    Hello --> HELLO_END([END])
	VerifyPlan
	Compose
	ToolTx
	ToolStats
	ToolRAG
	VerifyTx
	VerifyStats
	VerifyRAG
	Route
	Route -->|"MULTI_DOMAIN"| Planner
	VerifyRAG -.->|"REPLAN"| Planner
	VerifyStats -.->|"REPLAN"| Planner
	VerifyTx -.->|"REPLAN"| Planner
	Planner -->|"may call"| ToolRAG
	Planner -->|"may call"| ToolStats["search_player tool -  get_player_stats tool"]
	Planner -->|"may call"| ToolTx
	Planner --> Compose
	VerifyPlan -.->|"REPLAN"| Planner["Planner Agent Node Tools: rag, stats, EBIS"]
```

## Notes
- Compile the graph with a checkpointer and invoke with a stable `thread_id` to support interrupts/HITL and long runs.
- Interrupt nodes (`confirm_player`, `confirm_transaction`, `HITL approval`, `Clarify`) pause and resume using `interrupt()` / `Command(resume=...)`.
- Planner Agent can plan across tools (RAG/stats/tx) for multi-domain queries; deterministic subgraphs cover single-domain paths.
- Each domain remains a subgraph with isolated state to minimize cross-topic coupling.

## Patterns & Snippets

### Reuse code between subgraphs and agents
Keep domain logic in shared services; wrap once, use everywhere.

```python
# services.py (shared)
def search_player_service(name: str, only_active: bool = True) -> dict: ...
def get_player_stats_service(player_id: int) -> dict: ...
def rag_query_service(query: str) -> str: ...
```

```python
# agent tools
from langchain_core.tools import tool

@tool
def tool_search_player(name: str) -> dict:
    return search_player_service(name, only_active=True)

@tool
def tool_get_player_stats(player_id: int) -> dict:
    return get_player_stats_service(player_id)

@tool
def tool_rag_query(query: str) -> str:
    return rag_query_service(query)
```

```python
# graph nodes
def player_search_node(state: State) -> dict:
    name = state.get("extracted_name") or ""
    res = search_player_service(name)
    return {"candidate_players": res.get("players", [])}

def player_stats_node(state: State) -> dict:
    stats = get_player_stats_service(state["player_id"])  # type: ignore[index]
    return {"stats": stats}
```

### Combine agents and StateGraph (Pattern A: agent node inside graph)
```python
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

mlb_agent = create_agent(model, tools=[tool_search_player, tool_get_player_stats, tool_rag_query])

def planner_node(state: State) -> dict:
    out = mlb_agent.invoke({"messages": state["messages"]}, config=state.get("config"))
    return {"messages": state["messages"] + out["messages"]}

graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_edge("planner", END)
agent = graph.compile(name="MLB Graph", checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "session-123"}}
result = agent.invoke({"messages": [...]}, config=config)
```

### Agents-as-tools (Pattern B)
```python
stats_agent = create_agent(model, tools=[tool_search_player, tool_get_player_stats])

from langchain_core.tools import tool

@tool
def tool_stats_agent(messages: list) -> dict:
    return stats_agent.invoke({"messages": messages})
```

### Graph-as-tool (Pattern C)
```python
from langgraph.checkpoint.memory import MemorySaver

stats_subgraph = StateGraph(State)
# ... add stats nodes ...
stats = stats_subgraph.compile(name="stats", checkpointer=MemorySaver())

@tool
def tool_run_stats(messages: list) -> dict:
    cfg = {"configurable": {"thread_id": "tool-stats-1"}}
    return stats.invoke({"messages": messages}, config=cfg)
```

### Interrupt disambiguation + resume
```python
from langgraph.types import interrupt, Command

def confirm_player_node(state: State) -> dict:
    candidates = state.get("candidate_players", [])
    if len(candidates) <= 1:
        return {}
    selection = interrupt({
        "type": "confirm_player",
        "options": [{"id": p["id"], "name": p["full_name"]} for p in candidates],
    })
    return {"player_id": int(selection["id"])}

# resume from client
resume_cmd = Command(resume={"id": selected_id})
agent.invoke(resume_cmd, config={"configurable": {"thread_id": "session-123"}})
```
