# Technical Debt

This document tracks known technical debt items that need to be addressed, including their cost estimates and expected benefits.

---

## 1. Multi-Player Support in Single-Domain Paths

**Status:** 🔴 High Priority  
**Impact:** Medium  
**Cost:** Medium (2-3 days)

### Current State
The `PLAYER_STATS` subgraph currently only handles a single player at a time:
- State schema has `player_id: int | None` (single value)
- `player_search_node` extracts one player name via regex or router
- `player_stats_node` fetches stats for one player
- `answer_player_stats_query` generates answer for one player's stats

### Problem
- Multi-player queries like "Compare Aaron Judge and Mike Trout's stats" or "Give me stats for three players" are routed to `MULTI_DOMAIN` planner, which is less efficient
- The planner agent must make multiple tool calls sequentially, even though the deterministic path could handle this more efficiently
- Each player query requires separate API calls and LLM invocations in the planner

### Proposed Solution
- Extend `State` schema to support `player_ids: list[int]` alongside `player_id`
- Update `find_player_id` to `find_player_ids` that can extract multiple names from a query
- Modify `player_stats_node` to batch fetch stats for multiple players
- Update `answer_player_stats_query` to handle multi-player comparisons

### Benefits
- **Efficiency:** Single deterministic path handles multi-player queries without planner overhead
- **Latency:** Batch API calls reduce round-trips
- **Consistency:** More predictable behavior for common multi-player queries
- **Cost:** Fewer LLM calls (planner vs. direct path)

### Implementation Considerations
- Router needs to detect multi-player queries and route to `PLAYER_STATS` instead of `MULTI_DOMAIN`
- Backward compatibility: Keep `player_id` for single-player queries
- `player_stats_node` needs to handle both single and batch scenarios gracefully

---

## 2. Multi-Turn Conversation Context Handling

**Status:** 🔴 High Priority  
**Impact:** High  
**Cost:** Medium (2-4 days)

### Current State
Single-domain paths (`PLAYER_STATS` and `DOCUMENT_QA`) only examine the most recent user message:
- `player_search_node` extracts player name from `state["messages"][-1]` only
- `generate_rag_answer` uses `state["messages"][-1]` for the query
- Follow-up questions like "what is aaron judge's batting avg?" → "how about home runs?" lose context

### Problem
**Example Failure Scenario:**
1. User: "What is Aaron Judge's batting average?"
2. System: "Aaron Judge's batting average is .331"
3. User: "How about home runs?" 
4. **Current behavior:** System doesn't know "Aaron Judge" from previous turn, may fail or ask for clarification

**Root Cause:**
- Each invocation starts fresh at the router
- No conversation history awareness in single-domain nodes
- Router only sees the latest message, not the conversation context

### Proposed Solutions

#### Option A: Context-Aware Routing (Recommended)
- Router examines recent conversation history (last 2-3 turns)
- Extract implicit references (e.g., "how about X" → previous player)
- Update `route_query_node` to consider conversation context
- **Cost:** Medium (2-3 days)
- **Benefit:** Handles most common follow-up patterns

#### Option B: State Persistence with Checkpointer
- Use `MemorySaver` checkpointer to persist state across invocations
- Track `previous_player_id` in state for follow-up queries
- Router checks state for recent context before routing
- **Cost:** Medium-High (3-4 days, requires checkpointer setup)
- **Benefit:** Full conversation history available

#### Option C: LLM Context Window Management
- Pass full conversation history to nodes (up to token limit)
- Use LangGraph's `trimMessages` utility to manage context window
- Let LLM nodes naturally handle context from message history
- **Cost:** Low-Medium (1-2 days)
- **Benefit:** Leverages existing LangGraph patterns, handles arbitrary follow-ups

### Recommended Approach
**Hybrid:** Option A (context-aware routing) + Option C (message history in nodes)

1. **Router enhancement:** Look at last 2-3 messages for implicit references
2. **Node enhancement:** Pass recent message history to LLM nodes (within token limits)
3. **Fallback:** If context unclear, route to planner which has better context handling

### Benefits
- **User Experience:** Natural follow-up questions work correctly
- **Efficiency:** Single-domain paths handle follow-ups without planner overhead
- **Robustness:** Better handling of conversational queries

### Documentation References
- LangGraph memory guide: https://docs.langchain.com/oss/python/langgraph/add-memory
- LangGraph summarization middleware: https://docs.langchain.com/oss/javascript/langchain/middleware
- LangGraph trimMessages utility: https://docs.langchain.com/oss/javascript/langgraph/add-memory

---

## Summary

| Issue | Priority | Cost | Benefit | Next Steps |
|-------|----------|------|---------|------------|
| Multi-player support | High | Medium | Medium | Design state schema extensions |
| Multi-turn context | High | Medium | High | Research LangGraph memory patterns, design router enhancement |

---

## Notes

- Both issues are related to making single-domain paths more capable and efficient
- Addressing these will reduce reliance on the planner agent for common queries
- Consider batching these improvements together to minimize refactoring overhead

### 3. Interrupt Infrastructure Refinements (NEW)

**Status:** 🟠 Medium Priority  
**Impact:** Medium  
**Cost:** Low-Medium (1–2 days)

| Sub-Issue | Problem | Proposed Solution | Benefit |
|-----------|---------|-------------------|---------|
| EventProcessor coupling | `EventProcessor` currently mixes streaming, tool-call rendering, and interrupt logic, making future maintenance harder | Extract an `InterruptHandler` (or `StreamEventRouter`) class responsible only for interrupt detection & dispatch | Cleaner separation of concerns; easier to extend & unit-test |
| Hard-coded player selection UI | ********`display_interrupt_selection`******** only handles `player_selection` with ad-hoc button rendering | Introduce an **Interrupt Renderer Registry**: `INTERRUPT_RENDERERS = {"player_selection": render_player_selection, "tool_approval": render_tool_approval, ...}`.  UI dispatches to the appropriate renderer based on `payload["type"]`. Provide a **generic `render_choice_list`** helper reusable across interrupt types. | Supports additional interrupt kinds (e.g. tool approval, multi-turn clarification) with minimal changes |
| Duplicate interrupt checks in backend | `AgentEngineApp.stream_query` contains two separate paths for interrupt detection & final answer extraction | Refactor: always call `final_state = self.runnable.invoke(...)` once after streaming; branch on `"__interrupt__" in final_state` | Simpler logic, one code path to test |

---

### 4. Integration Tests for Interrupt Flow (NEW)

**Status:** 🟡 Medium Priority  
**Impact:** Medium  
**Cost:** Medium (1–2 days)

Add an **end-to-end integration test** that:
1. Launches the local agent & Streamlit UI headlessly (Streamlit `testing` module or Playwright)
2. Sends a query that triggers a `player_selection` interrupt
3. Simulates clicking a candidate button
4. Asserts that the final AI message appears in chat history and `pending_interrupt` state is cleared.

Provides confidence that regressions in interrupt handling or resume logic are caught automatically.

#### Verification Loop Side-Effect  
*Because the verification node tries to rediscover the user’s query by scanning `state["messages"]`, interruption/resume and follow-up turns can reorder messages so that the first/last element is an `AIMessage`.  When no `HumanMessage` is found the node falls back to the wrong content (often the previous answer), so the LLM-judge is evaluating **answer vs. answer** and returns `OK` even if the user question was never addressed.*

**Fix comes “for free” once multi-turn context is implemented:**  whichever option we pick (A/B/C) should guarantee that the latest user utterance can be reliably accessed (e.g. `state["last_user_query"]`).  Verification node will then read that field directly instead of searching the message list, eliminating this bug.


#### Regression Test Scenario (to add once multi-turn context is implemented)

```
# test_multi_turn_verification.py
from app.agent_engine_app import AgentEngineApp
from langgraph.types import Command

app = AgentEngineApp(...  # local runnable)
thread_id = "test-thread"

# 1️⃣ initial query -> should answer batting average
steps = app.runnable.invoke(
    input={"messages": [{"role": "user", "content": "What is Aaron Judge's batting average?"}]},
    config={"configurable": {"thread_id": thread_id}},
)
assert "AVG" in steps["messages"][-1].content

# 2️⃣ follow-up query referring to same player
steps = app.runnable.invoke(
    input={"messages": [{"role": "user", "content": "How about home runs?"}]},
    config={"configurable": {"thread_id": thread_id}},
)
# ensure no interrupt is triggered (should reuse context)
assert "__interrupt__" not in steps
# ensure answer mentions a numeric home-run total
assert any(word.isdigit() for word in steps["messages"][-1].content.split())
```

This test fails today (home-run question loses context) and should pass once multi-turn routing + verification fixes are delivered.


### 5. Parallel Interrupt Handling (Optional / Future)  

**Status:** 🟣 Low Priority  
**Impact:** Low (quality-of-life)  
**Cost:** Low-Medium (≤1 day)

| Problem | Proposed Solution | Benefit |
|---------|-------------------|---------|
| Current UI/backend assume **one** pending interrupt per conversation.  If a second interrupt fires before the first is resolved it simply overwrites the previous payload. | Migrate to<br>`st.session_state.pending_interrupts: dict[str, list[Interrupt]]` keyed by `thread_id`.  UI pops from the list and can show a queue/stack of actions.  Backend logic unchanged (still emits `{type:"interrupt"}` events). | Enables nested/workflow-style approvals (e.g. player selection **and** tool approval) without dropping earlier requests.

This enhancement can be tackled after the primary interrupt UX is solid and multi-turn context is complete.


> **Note on parallel interrupts**  
> Today the UI assumes **one pending interrupt per conversation** (`pending_interrupt`, `interrupt_thread_id`).  If we later support multiple concurrent actions (e.g., nested tool approvals while a player selection is still open) we can migrate to a structure like:
>
> ```python
> st.session_state.pending_interrupts: dict[str, list[Interrupt]]  # keyed by thread_id
> ```
>
> The interrupt renderer would pop from the queue and the UI could surface a stack/queue of actions.  Not required for the current MVP, but documented so the next refactor considers it.