# Phase 4 Implementation Plan: Interrupts for Disambiguation

## Overview
Handle ambiguous player name searches (e.g., "Will Smith" matching multiple players). 

**Decision Point:** Two approaches are possible:
1. **LangGraph Interrupts** (formal pause/resume mechanism)
2. **Message-based Clarification** (LLM generates clarification question, user responds naturally)

**Recommendation:** Start with **Message-based Clarification** - simpler, no special UI handling needed, works with any deployment.

## Approach Comparison

### Option A: LangGraph Interrupts (Complex)
**Flow:**
```
User: "Tell me about Will Smith"
→ player_search_node: finds multiple matches
→ interrupt(): pauses graph, returns candidates
→ [PAUSE - waiting for user input]
→ User selects: player_id=123
→ Resume with Command(resume=123)
→ player_stats_node: uses player_id=123
→ answer: returns stats for correct Will Smith
```

**Pros:**
- Explicit pause/resume mechanism
- State persisted automatically
- Clean separation of clarification from normal flow

**Cons:**
- Requires checkpointer (`MemorySaver`)
- Requires special UI handling (`__interrupt__` detection)
- May not work with Vertex AI Agent Engine (see ADK issue #2620)
- More complex implementation

### Option B: Message-based Clarification (Simple) ⭐ RECOMMENDED
**Flow:**
```
User: "Tell me about Will Smith"
→ player_search_node: finds multiple matches
→ answer_player_stats_query: generates clarification message
→ "Multiple players found: 1) Will Smith (Dodgers), 2) Will Smith (Marlins). Which one?"
→ [User responds in next message]
→ Router detects clarification response
→ player_search_node: extracts selected player from message
→ Continue normally
```

**Pros:**
- ✅ No checkpointer needed
- ✅ Works with any UI (no special interrupt handling)
- ✅ Natural conversation flow
- ✅ No Vertex AI Agent Engine compatibility concerns
- ✅ Simpler implementation (just generate a message)
- ✅ Aligns with solving multi-turn conversation context (tech debt #2)

**Cons:**
- User has to send another message (vs. clicking a button)
- Need to parse/understand clarification response
- Requires router/node to handle follow-up context

**Recommendation:** Use **Option B** (Message-based) for Phase 4, especially since you need to solve multi-turn context handling anyway.

---

## Implementation Steps (Message-Based Approach)

### Step 1: Update `find_player_id` Logic (`app/logic.py`)

**Current Signature:**
```python
def find_player_id(player_name: str) -> int | None
```

**New Signature:**
```python
def find_player_id(player_name: str) -> int | list[dict[str, Any]] | None
```

**Logic Changes:**
1. After searching, check if multiple good matches exist
2. **Ambiguity threshold:** If 2+ matches with similar confidence (exact/partial matches)
3. **Single match:** Return `int` (player ID)
4. **Multiple matches:** Return `list[dict]` with format:
   ```python
   [
       {"id": 123, "name": "Will Smith", "team": "Los Angeles Dodgers"},
       {"id": 456, "name": "Will Smith", "team": "Miami Marlins"},
   ]
   ```
5. **No matches:** Return `None`

**Implementation Notes:**
- Keep existing exact/partial match logic
- When multiple exact matches OR multiple partial matches exist, return list
- Include team info in candidates to help user distinguish

---

### Step 2: Update `player_search_node` (`app/nodes.py`)

**Current Implementation:**
```python
def player_search_node(state: State) -> dict:
    player_id = find_player_id(name)
    return {"player_id": int(player_id) if player_id is not None else None}
```

**New Implementation (Message-based):**
```python
def player_search_node(state: State) -> dict:
    result = find_player_id(name)
    
    if isinstance(result, int):
        # Single match - proceed normally
        return {"player_id": result}
    
    elif isinstance(result, list):
        # Multiple matches - store candidates and let answer node handle clarification
        candidates = result
        logging.info(
            "[player_search] Multiple matches found: %d candidates", len(candidates)
        )
        
        # Store candidates in state for answer node to use
        # We'll return None for player_id to signal ambiguity
        return {
            "player_id": None,
            "pending_player_candidates": candidates,  # Add to state schema
        }
    
    else:
        # No matches
        return {"player_id": None}
```

**Key Points:**
- No interrupts needed - just detect ambiguity
- Store candidates in state
- Let the answer node generate clarification message

### Step 2b: Update `answer_player_stats_query` to Handle Ambiguity (`app/nodes.py`)

**Current Implementation:**
```python
def answer_player_stats_query(state: State) -> dict:
    stats = state.get("stats")
    query = extract_message_content(state["messages"][-1])
    answer = generate_player_stats_answer(query, stats)
    return {"messages": [AIMessage(content=answer)]}
```

**New Implementation:**
```python
def answer_player_stats_query(state: State) -> dict:
    # Check if we have pending candidates (ambiguity detected)
    candidates = state.get("pending_player_candidates")
    
    if candidates and not state.get("player_id"):
        # Generate clarification message
        candidate_list = "\n".join(
            [f"{i+1}. {c['name']} ({c.get('team', 'Unknown Team')})" 
             for i, c in enumerate(candidates)]
        )
        clarification = (
            f"I found multiple players named '{state.get('extracted_name', 'this name')}'. "
            f"Please specify which one you'd like information about:\n\n{candidate_list}\n\n"
            f"You can reply with the number (1, 2, etc.) or the player's full name and team."
        )
        return {"messages": [AIMessage(content=clarification)]}
    
    # Normal flow - player_id resolved
    stats = state.get("stats")
    query = extract_message_content(state["messages"][-1])
    answer = generate_player_stats_answer(query, stats)
    return {"messages": [AIMessage(content=answer)]}
```

**Key Points:**
- Check for `pending_player_candidates` in state
- Generate user-friendly clarification message
- User responds naturally in next turn

---

### Step 3: Update Router to Handle Clarification Responses (`app/nodes.py`)

**Enhancement Needed:**
When user responds to clarification (e.g., "1" or "Will Smith Dodgers"), the router needs to:
1. Detect this is a clarification response
2. Extract the selected player from the message
3. Route back to `PLAYER_STATS` with the resolved player

**Option A: Simple Pattern Matching**
```python
def route_query_node(state: State) -> dict:
    # Check if this looks like a clarification response
    last_msg = state["messages"][-1]
    query = extract_message_content(last_msg)
    
    # Check if previous message was a clarification (has pending_player_candidates)
    # This requires multi-turn context - see tech_debt.md #2
    
    # For now, check if query is just a number or contains player name + team
    if re.match(r'^\d+$', query.strip()):
        # User selected by number - need to resolve from previous context
        # This requires multi-turn context handling
        pass
    
    # Normal routing logic...
```

**Option B: Use LLM Router (Recommended)**
Update router prompt to detect clarification responses:
```python
system_prompt = """...
If the user is responding to a clarification question (e.g., "1" or "Will Smith Dodgers"),
extract the player name/team from the response and route to PLAYER_STATS.

Examples:
- User: "1" (after clarification) -> {"route": "PLAYER_STATS", "entities": {"name": "[resolved from context]", "team": null}}
...
"""
```

**Note:** This requires solving multi-turn context handling (tech debt #2). Consider implementing together.

### Step 3b: Update State Schema (`app/state.py`)

**Add Field:**
```python
class State(TypedDict):
    # ... existing fields ...
    pending_player_candidates: list[dict[str, Any]] | None  # Candidates awaiting selection
```

**Note:** This allows nodes to check if clarification is needed.

---

### Step 4: No Client Changes Needed! 🎉

**Current Implementation Works:**
```python
def invoke_agent(query: str) -> str:
    initial_state = {"messages": [HumanMessage(content=query)]}
    final_state = agent.invoke(initial_state)
    return final_state["messages"][-1].content
```

**No changes needed!** The clarification message is just a normal AI message. User responds naturally in the next turn, and the router handles it.

**Example Flow:**
```python
# First turn
result1 = invoke_agent("Tell me about Will Smith")
# Returns: "I found multiple players... 1) Will Smith (Dodgers), 2) Will Smith (Marlins)..."

# Second turn (user responds)
result2 = invoke_agent("1")  # or "Will Smith Dodgers"
# Router detects clarification, resolves player, continues normally
# Returns: "Will Smith (Dodgers) has a batting average of..."
```

**Key Benefit:** No special interrupt handling needed in client code!

---

### Step 5: Update State Schema (`app/state.py`)

**Add Field:**
```python
class State(TypedDict):
    # ... existing fields ...
    pending_player_candidates: list[dict[str, Any]] | None  # Candidates awaiting selection
```

**Usage:**
- Set by `player_search_node` when ambiguity detected
- Checked by `answer_player_stats_query` to generate clarification
- Cleared when user provides clarification

---

### Step 6: No Agent Engine App Changes Needed! 🎉

**Current Implementation Works:**
```python
def query(self, *, input: str | Mapping, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
    config = ensure_valid_config(config)
    self.set_tracing_properties(config=config)
    return dumpd(self.runnable.invoke(input=input, config=config, **kwargs))
```

**No changes needed!** Clarification messages are just normal AI messages in the response.

---

### Step 7: No Streamlit UI Changes Needed! 🎉

**Current Implementation Works:**
The clarification message is just a normal AI message that gets displayed like any other response. User responds naturally in the next turn.

**Example in Streamlit:**
```
User: "Tell me about Will Smith"
→ AI: "I found multiple players... 1) Will Smith (Dodgers), 2) Will Smith (Marlins)..."
[User sees this as normal message]
User: "1" or "Will Smith Dodgers"
→ Router handles clarification, continues normally
→ AI: "Will Smith (Dodgers) has..."
```

**Optional Enhancement:** You could add UI to make selection easier (e.g., clickable buttons), but it's not required - the message-based approach works with plain text.

---

## Why Message-Based is Better for Your Use Case

1. ✅ **No checkpointer needed** - simpler deployment
2. ✅ **Works with Vertex AI Agent Engine** - no ADK interrupt concerns
3. ✅ **Natural conversation flow** - user responds like normal
4. ✅ **No special UI handling** - clarification is just another message
5. ✅ **Aligns with multi-turn context** - solves tech debt #2 simultaneously
6. ✅ **Simpler implementation** - less code, fewer edge cases

**Trade-off:** User has to type response instead of clicking, but this is more natural for conversational interfaces anyway.

---

## Testing Strategy

### Unit Tests (`tests/unit/test_interrupts.py`)

1. **Test `find_player_id` ambiguity detection:**
   ```python
   def test_find_player_id_ambiguous():
       result = find_player_id("Will Smith")
       assert isinstance(result, list)
       assert len(result) >= 2
       assert all("id" in c and "name" in c for c in result)
   ```

2. **Test `find_player_id` single match:**
   ```python
   def test_find_player_id_single():
       result = find_player_id("Aaron Judge")
       assert isinstance(result, int)
   ```

### Integration Tests (`tests/integration/test_interrupts.py`)

1. **Test interrupt flow:**
   ```python
   def test_player_search_interrupt():
       config = {"configurable": {"thread_id": "test-interrupt-1"}}
       
       # First invocation - should interrupt
       result = agent.invoke(
           {"messages": [HumanMessage(content="Tell me about Will Smith")]},
           config=config
       )
       
       assert "__interrupt__" in result
       interrupt_data = result["__interrupt__"][0].value
       assert "candidates" in interrupt_data
       assert len(interrupt_data["candidates"]) >= 2
       
       # Resume with selection
       selected_id = interrupt_data["candidates"][0]["id"]
       final_result = agent.invoke(
           Command(resume=selected_id),
           config=config
       )
       
       assert "__interrupt__" not in final_result
       assert final_result["player_id"] == selected_id
   ```

2. **Test single match (no interrupt):**
   ```python
   def test_player_search_no_interrupt():
       config = {"configurable": {"thread_id": "test-no-interrupt-1"}}
       
       result = agent.invoke(
           {"messages": [HumanMessage(content="Tell me about Aaron Judge")]},
           config=config
       )
       
       assert "__interrupt__" not in result
       assert result["player_id"] is not None
   ```

---

## Edge Cases & Considerations

### 1. Invalid Resume Value
- **Problem:** User passes invalid player_id (not in candidates)
- **Solution:** Validate in `player_search_node` and use first candidate as fallback

### 2. No Matches Found
- **Problem:** `find_player_id` returns `None`
- **Solution:** Current behavior (return `{"player_id": None}`) is fine, will be handled downstream

### 3. Thread ID Management
- **Problem:** Need unique thread_id per conversation
- **Solution:** Generate UUID or use session_id from client

### 4. Interrupt Timeout
- **Problem:** What if user never responds?
- **Solution:** LangGraph interrupts wait indefinitely until resume. Consider adding timeout at application layer if needed.

### 5. Subgraph Interrupts
- **Consideration:** Interrupt happens inside `player_stats_sg` subgraph
- **Impact:** Should work fine - interrupts work at any node level

### 6. Planner Path
- **Consideration:** Planner agent uses tools that call `find_player_id`
- **Impact:** Tools themselves don't interrupt (nodes do). If planner needs disambiguation, we'd need to handle it differently (maybe planner asks user directly).

---

## Verification Checklist

- [ ] `find_player_id` returns `int | list[dict] | None`
- [ ] `player_search_node` detects list and calls `interrupt()`
- [ ] Graph compiled with `checkpointer=MemorySaver()`
- [ ] `invoke_agent` supports `thread_id` and `resume_with`
- [ ] Interrupt payload includes user-friendly message and candidates
- [ ] Resume validates selected ID
- [ ] Unit tests cover ambiguity detection
- [ ] Integration tests cover full interrupt/resume flow
- [ ] Single-match queries still work (no interrupt)
- [ ] No-match queries still work (no interrupt)

---

## Deployment Considerations

### Development/Testing
- `MemorySaver` is sufficient (in-memory, resets on restart)

### Production
- Consider persistent checkpointer (`AsyncPostgresSaver`, `SqliteSaver`)
- Thread IDs should be managed per user session
- Client UI needs to handle interrupt display and selection
- Consider timeout mechanism for abandoned interrupts

---

## Next Steps After Phase 4

1. Update `evals.md` with interrupt test cases
2. Add interrupt handling to playground/UI
3. Document interrupt behavior for users
4. Consider extending to other ambiguous queries (team names, etc.)

