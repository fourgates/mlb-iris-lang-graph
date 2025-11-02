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

