# Phase 4 UI Interrupt Implementation Plan

**Branching from:** `phase4-implementation-plan.md`  
**Goal:** Implement LangGraph interrupt support in the Streamlit UI for player disambiguation  
**Inspiration:** [LangGraph agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) - `use-interrupted-actions.tsx`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current State Analysis](#current-state-analysis)
3. [LangGraph UI Pattern Analysis](#langgraph-ui-pattern-analysis)
4. [Implementation Approach](#implementation-approach)
5. [Edge Cases & Error Handling](#edge-cases--error-handling)
6. [Alternative: Use LangGraph UI Directly](#alternative-use-langgraph-ui-directly)
7. [Testing Strategy](#testing-strategy)
8. [Migration Path](#migration-path)

---

## Architecture Overview

### Current Flow (No Interrupts)
```
User Input → handle_user_input()
  → generate_ai_response()
    → get_chain_response()
      → EventProcessor.process_events()
        → client.stream_messages()
          → agent.stream_query()
            → agent.stream(stream_mode="messages")
              → Yields message chunks
        → Process chunks, accumulate content
        → Add final message to chat history
```

### Target Flow (With Interrupts)
```
User Input → handle_user_input()
  → generate_ai_response()
    → get_chain_response()
      → EventProcessor.process_events()
        → client.stream_messages()
          → agent.stream_query()
            → Stream completes
        → Check final state for __interrupt__
        → IF interrupt detected:
           → Store interrupt data in session state
           → Display interrupt UI (player selection)
           → WAIT for user selection
           → resume_interrupt(selected_player_id)
             → client.stream_messages(resume_command=Command(resume=...))
             → Continue normal flow
        → ELSE: Add final message to chat history
```

---

## Current State Analysis

### Backend (`app/agent_engine_app.py`)

**Current `stream_query` Implementation:**
```python
def stream_query(self, *, input: str | Mapping, config: RunnableConfig | None = None, **kwargs: Any) -> Iterable[Any]:
    config = ensure_valid_config(config)
    self.set_tracing_properties(config=config)
    input_chat = InputChat.model_validate(input)
    
    for chunk in self.runnable.stream(
        input=input_chat, config=config, **kwargs, stream_mode="messages"
    ):
        dumped_chunk = dumpd(chunk)
        yield dumped_chunk
```

**Problem:** Only streams chunks, doesn't return final state where `__interrupt__` lives.

**Solution Needed:** After streaming completes, invoke once to get final state and check for interrupts.

### Frontend (`frontend/utils/stream_handler.py`)

**Current `EventProcessor.process_events()`:**
- Processes streaming chunks
- Accumulates `final_content`
- Adds final message to chat history
- **Missing:** Interrupt detection

**Current `Client.stream_messages()`:**
- Calls `agent.stream_query(**data)`
- **Missing:** Support for `Command(resume=...)` pattern

### Graph (`app/graph.py`)

**Current:**
```python
agent = _graph.compile(name="Grounding Chat Graph", cache=InMemoryCache())
```

**Missing:** Checkpointer (required for interrupts)

---

## LangGraph UI Pattern Analysis

### Key Insights from `use-interrupted-actions.tsx`

1. **Interrupt Detection:**
   - Interrupts come from `HumanInterrupt` type
   - Accessed via `thread.meta` or final state `__interrupt__`

2. **Resume Pattern:**
   ```typescript
   thread.submit(
     {},
     {
       command: {
         resume: response,  // HumanResponse[] array
       },
     },
   );
   ```

3. **Response Types:**
   - `"response"` - User provides text input
   - `"edit"` - User edits existing values
   - `"accept"` - User accepts default/pre-filled values
   - `"ignore"` - User ignores the interrupt

4. **State Management:**
   - Loading states (`loading`, `streaming`, `streamFinished`)
   - Response state (`humanResponse`)
   - UI state (`hasEdited`, `hasAddedResponse`)

5. **User Experience:**
   - Toast notifications for feedback
   - Validation before submit
   - Multiple submission methods supported

### Adapting to Streamlit

**Differences:**
- Streamlit uses session state (not React hooks)
- Streamlit reruns on interaction (not reactive updates)
- No built-in toast system (use `st.error`, `st.success`, `st.warning`)

**Similarities:**
- Need to detect interrupts
- Need to display selection UI
- Need to resume with `Command(resume=...)`
- Need loading states

---

## Implementation Approach

### Phase 1: Backend Foundation

#### Step 1.1: Add Checkpointer to Graph

**File:** `app/graph.py`

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = _graph.compile(
    name="Grounding Chat Graph",
    cache=InMemoryCache(),
    checkpointer=checkpointer,  # Required for interrupts
)
```

**Impact:** Enables state persistence for interrupts.

---

#### Step 1.2: Implement Interrupt in `player_search_node`

**File:** `app/nodes.py`

```python
from langgraph.types import interrupt

def player_search_node(state: State) -> dict:
    log_start("player_search")
    name = state.get("extracted_name")
    if not name:
        log_end("player_search", error=True, reason="no_name")
        return {"player_id": None}
    
    result = find_player_id(name)
    
    if isinstance(result, int):
        # Single match - proceed normally
        log_end("player_search", player_id=result)
        return {"player_id": result}
    
    elif isinstance(result, list):
        # Multiple matches - interrupt for user selection
        candidates = result
        logging.info(
            "[player_search] Multiple matches found: %d candidates", len(candidates)
        )
        
        # Format interrupt payload
        interrupt_payload = {
            "message": f"I found {len(candidates)} players named '{name}'. Please select one:",
            "candidates": candidates,
            "type": "player_selection",  # Custom type for our use case
        }
        
        # Interrupt and wait for user selection
        selected_id = interrupt(interrupt_payload)
        
        # Validate selected ID
        valid_ids = [c["id"] for c in candidates]
        if selected_id not in valid_ids:
            logging.warning(
                "[player_search] Invalid player_id %s selected, using first candidate",
                selected_id,
            )
            selected_id = valid_ids[0]
        
        log_end("player_search", player_id=selected_id, interrupted=True)
        return {"player_id": int(selected_id)}
    
    else:
        # No matches
        log_end("player_search", error=True, reason="no_matches")
        return {"player_id": None}
```

**Key Points:**
- `interrupt()` pauses graph execution
- Returns value passed to `Command(resume=...)`
- Interrupt payload includes user-friendly message and candidates

---

#### Step 1.3: Modify `agent_engine_app.py` to Return Final State

**File:** `app/agent_engine_app.py`

**Current `stream_query`:**
```python
def stream_query(self, *, input: str | Mapping, config: RunnableConfig | None = None, **kwargs: Any) -> Iterable[Any]:
    # ... existing code ...
    for chunk in self.runnable.stream(...):
        yield dumped_chunk
```

**New `stream_query` (Check for Interrupts):**
```python
def stream_query(
    self,
    *,
    input: str | Mapping,
    config: RunnableConfig | None = None,
    resume_command: Any | None = None,  # NEW: Command(resume=...)
    **kwargs: Any,
) -> Iterable[Any]:
    """Stream responses from the agent, optionally resuming from interrupt."""
    config = ensure_valid_config(config)
    self.set_tracing_properties(config=config)
    input_chat = InputChat.model_validate(input)
    
    # If resuming from interrupt, use Command pattern
    if resume_command is not None:
        input_data = resume_command
    else:
        input_data = input_chat
    
    # Stream chunks
    final_state = None
    for chunk in self.runnable.stream(
        input=input_data, config=config, **kwargs, stream_mode="messages"
    ):
        dumped_chunk = dumpd(chunk)
        yield dumped_chunk
    
    # After stream completes, get final state to check for interrupts
    final_state = self.runnable.invoke(input=input_data, config=config, **kwargs)
    
    # Yield interrupt event if present
    if "__interrupt__" in final_state:
        interrupt_data = final_state["__interrupt__"]
        yield {
            "type": "interrupt",
            "data": interrupt_data,
            "thread_id": config.get("configurable", {}).get("thread_id"),
        }
```

**Alternative Approach (Lighter Weight):**
Only check for interrupts when not resuming (new queries):

```python
def stream_query(self, *, input: str | Mapping, config: RunnableConfig | None = None, resume_command: Any | None = None, **kwargs: Any) -> Iterable[Any]:
    config = ensure_valid_config(config)
    self.set_tracing_properties(config=config)
    
    if resume_command is not None:
        # Resuming - stream directly
        input_data = resume_command
    else:
        # New query - validate input
        input_chat = InputChat.model_validate(input)
        input_data = input_chat
    
    # Stream chunks
    for chunk in self.runnable.stream(
        input=input_data, config=config, **kwargs, stream_mode="messages"
    ):
        yield dumpd(chunk)
    
    # Only check for interrupts on new queries (not resumes)
    if resume_command is None:
        final_state = self.runnable.invoke(input=input_data, config=config, **kwargs)
        if "__interrupt__" in final_state:
            yield {
                "type": "interrupt",
                "data": final_state["__interrupt__"],
                "thread_id": config.get("configurable", {}).get("thread_id"),
            }
```

**Trade-off:** Slightly less efficient (extra `invoke()` call), but simpler to reason about.

---

### Phase 2: Frontend Interrupt Detection

#### Step 2.1: Update `EventProcessor` to Detect Interrupts

**File:** `frontend/utils/stream_handler.py`

```python
class EventProcessor:
    def __init__(self, st: Any, client: Client, stream_handler: StreamHandler) -> None:
        # ... existing fields ...
        self.interrupt_data: dict[str, Any] | None = None
        self.interrupt_thread_id: str | None = None
    
    def process_events(self, resume_command: Any | None = None) -> None:
        """Process events from the stream, handling interrupts."""
        messages = self.st.session_state.user_chats[
            self.st.session_state["session_id"]
        ]["messages"]
        self.current_run_id = str(uuid.uuid4())
        self.st.session_state["run_id"] = self.current_run_id
        
        # Get thread_id (use session_id)
        thread_id = self.st.session_state.get("thread_id") or self.st.session_state["session_id"]
        if "thread_id" not in self.st.session_state:
            self.st.session_state["thread_id"] = thread_id
        
        stream = self.client.stream_messages(
            data={
                "input": {"messages": messages},
                "config": {
                    "configurable": {"thread_id": thread_id},  # NEW: thread_id
                    "run_id": self.current_run_id,
                    "metadata": {
                        "user_id": self.st.session_state["user_id"],
                        "session_id": self.st.session_state["session_id"],
                    },
                },
                "resume_command": resume_command,  # NEW: for resuming
            }
        )
        
        # Process stream events
        for message in stream:
            if isinstance(message, dict):
                # Check for interrupt event
                if message.get("type") == "interrupt":
                    self.interrupt_data = message.get("data")
                    self.interrupt_thread_id = message.get("thread_id")
                    self.st.session_state.pending_interrupt = self.interrupt_data
                    self.st.session_state.interrupt_thread_id = self.interrupt_thread_id
                    logging.info("[EventProcessor] Interrupt detected: %s", self.interrupt_data)
                    return  # Stop processing, wait for user input
                
                # ... existing message processing logic ...
        
        # Normal completion - no interrupt
        if self.final_content:
            # ... existing final message logic ...
```

**Key Changes:**
- Detect `type == "interrupt"` events
- Store interrupt data in session state
- Return early if interrupt detected (don't add final message)

---

#### Step 2.2: Update `Client.stream_messages` to Support Resume

**File:** `frontend/utils/stream_handler.py`

```python
class Client:
    def stream_messages(
        self, data: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]:
        """Stream events from the server, yielding parsed event data."""
        resume_command = data.pop("resume_command", None)  # Extract resume command
        
        if self.url:
            # Remote URL - add resume_command to request
            request_data = {**data}
            if resume_command:
                request_data["resume_command"] = resume_command
            # ... existing URL streaming logic ...
        elif self.agent is not None:
            # Local agent - pass resume_command to stream_query
            yield from self.agent.stream_query(
                **data,
                resume_command=resume_command,
            )
```

**Note:** Need to handle `resume_command` serialization for remote URLs.

---

### Phase 3: UI Components

#### Step 3.1: Add Interrupt Display Function

**File:** `frontend/streamlit_app.py`

```python
def display_interrupt_selection() -> bool:
    """
    Display interrupt UI and handle user selection.
    
    Returns:
        bool: True if interrupt was handled (selection made), False otherwise
    """
    if "pending_interrupt" not in st.session_state:
        return False
    
    interrupt_data = st.session_state.pending_interrupt
    
    # Extract interrupt payload (LangGraph wraps it)
    if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
        interrupt_payload = interrupt_data[0].get("value", {})
    elif isinstance(interrupt_data, dict):
        interrupt_payload = interrupt_data
    else:
        st.error("Invalid interrupt data format")
        return False
    
    # Check if this is a player selection interrupt
    if interrupt_payload.get("type") != "player_selection":
        st.warning(f"Unknown interrupt type: {interrupt_payload.get('type')}")
        return False
    
    candidates = interrupt_payload.get("candidates", [])
    if not candidates:
        st.error("No candidates found in interrupt")
        return False
    
    # Display interrupt message
    st.info(f"🔍 {interrupt_payload.get('message', 'Multiple matches found')}")
    
    # Display candidates as buttons
    st.markdown("**Please select a player:**")
    
    # Create columns for buttons (max 3 per row)
    cols_per_row = 3
    num_rows = (len(candidates) + cols_per_row - 1) // cols_per_row
    
    selected_index = None
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            candidate_idx = row * cols_per_row + col_idx
            if candidate_idx >= len(candidates):
                break
            
            candidate = candidates[candidate_idx]
            label = f"{candidate['name']}\n({candidate.get('team', 'Unknown Team')})"
            
            with col:
                if st.button(label, key=f"interrupt_btn_{candidate_idx}"):
                    selected_index = candidate_idx
    
    # Alternative: Use selectbox (simpler, less visual)
    # selected_label = st.selectbox(
    #     "Select a player:",
    #     options=[f"{c['name']} ({c.get('team', 'Unknown')})" for c in candidates],
    #     key="interrupt_selection"
    # )
    # if selected_label:
    #     selected_index = [f"{c['name']} ({c.get('team', 'Unknown')})" for c in candidates].index(selected_label)
    
    # Handle selection
    if selected_index is not None:
        selected_id = candidates[selected_index]["id"]
        resume_interrupt(selected_id)
        return True
    
    # Cancel button
    if st.button("Cancel", key="interrupt_cancel"):
        # Clear interrupt state, add error message
        del st.session_state.pending_interrupt
        del st.session_state.interrupt_thread_id
        st.session_state.user_chats[st.session_state["session_id"]]["messages"].append({
            "type": "ai",
            "content": "Player selection cancelled. Please try again with a more specific query.",
        })
        st.rerun()
    
    return False  # Still waiting for selection
```

---

#### Step 3.2: Add Resume Function

**File:** `frontend/streamlit_app.py`

```python
def resume_interrupt(selected_player_id: int) -> None:
    """Resume graph execution with selected player ID."""
    from langgraph.types import Command
    
    thread_id = st.session_state.interrupt_thread_id
    if not thread_id:
        st.error("Missing thread_id for interrupt resume")
        return
    
    # Create resume command
    resume_command = Command(resume=selected_player_id)
    
    # Clear interrupt state BEFORE resuming (prevent loops)
    del st.session_state.pending_interrupt
    del st.session_state.interrupt_thread_id
    
    # Display loading state
    with st.spinner("Processing your selection..."):
        # Resume execution
        stream_handler = StreamHandler(st=st)
        client = Client(
            remote_agent_engine_id=st.session_state.get("remote_agent_engine_id"),
            agent_callable_path=st.session_state.get("agent_callable_path"),
            url=st.session_state.get("url"),
            authenticate_request=st.session_state.get("authenticate_request", False),
        )
        
        # Modify get_chain_response to accept resume_command
        get_chain_response(
            st=st,
            client=client,
            stream_handler=stream_handler,
            resume_command=resume_command,
        )
    
    st.rerun()
```

---

#### Step 3.3: Update `get_chain_response` to Accept Resume Command

**File:** `frontend/utils/stream_handler.py`

```python
def get_chain_response(
    st: Any,
    client: Client,
    stream_handler: StreamHandler,
    resume_command: Any | None = None,  # NEW
) -> None:
    """Process the chain response, optionally resuming from interrupt."""
    processor = EventProcessor(st, client, stream_handler)
    processor.process_events(resume_command=resume_command)
```

---

#### Step 3.4: Update `main()` to Check for Interrupts

**File:** `frontend/streamlit_app.py`

```python
def main() -> None:
    """Main function to set up and run the Streamlit app."""
    setup_page()
    initialize_session_state()
    side_bar = SideBar(st=st)
    side_bar.init_side_bar()
    
    # Check for pending interrupt BEFORE displaying messages
    interrupt_handled = display_interrupt_selection()
    if interrupt_handled:
        return  # Interrupt was handled, wait for rerun
    
    display_messages()
    handle_user_input(side_bar=side_bar)
    display_feedback(side_bar=side_bar)
```

**Key Point:** Check interrupts **before** displaying messages to prevent normal flow from continuing.

---

#### Step 3.5: Disable Input During Interrupt

**File:** `frontend/streamlit_app.py`

```python
def handle_user_input(side_bar: SideBar) -> None:
    """Process user input, generate AI response, and update chat history."""
    # Disable input if interrupt is pending
    if "pending_interrupt" in st.session_state:
        st.chat_input(disabled=True, placeholder="Please select a player above to continue...")
        return
    
    prompt = st.chat_input() or st.session_state.modified_prompt
    # ... rest of existing logic ...
```

---

## Edge Cases & Error Handling

### 1. Multiple Interrupts
**Problem:** What if user triggers another interrupt while one is pending?  
**Solution:** Only allow one pending interrupt at a time. Clear existing interrupt if new query comes in.

```python
def handle_user_input(side_bar: SideBar) -> None:
    # Clear any pending interrupt if user sends new message
    if "pending_interrupt" in st.session_state:
        logging.warning("Clearing pending interrupt due to new user input")
        del st.session_state.pending_interrupt
        del st.session_state.interrupt_thread_id
    
    # ... rest of logic ...
```

---

### 2. Invalid Selected ID
**Problem:** User somehow selects invalid player_id (shouldn't happen, but defense in depth).  
**Solution:** Validate in backend `player_search_node` (already implemented), also validate in frontend.

```python
def resume_interrupt(selected_player_id: int) -> None:
    # Validate selected_id is in candidates
    interrupt_data = st.session_state.get("pending_interrupt")
    if interrupt_data:
        interrupt_payload = interrupt_data[0].get("value", {}) if isinstance(interrupt_data, list) else interrupt_data
        candidates = interrupt_payload.get("candidates", [])
        valid_ids = [c["id"] for c in candidates]
        if selected_player_id not in valid_ids:
            st.error(f"Invalid player ID selected: {selected_player_id}")
            return
    
    # ... rest of resume logic ...
```

---

### 3. Session Refresh During Interrupt
**Problem:** User refreshes page while interrupt is pending.  
**Solution:** Interrupt state persists in `st.session_state`, UI re-renders correctly.

**Consideration:** Add timeout/expiration? Maybe not needed for MVP.

---

### 4. Thread ID Mismatch
**Problem:** Resume uses wrong thread_id (shouldn't happen, but validate).  
**Solution:** Store thread_id with interrupt, validate on resume.

```python
def resume_interrupt(selected_player_id: int) -> None:
    thread_id = st.session_state.interrupt_thread_id
    if not thread_id:
        st.error("Missing thread_id for interrupt resume")
        return
    
    # Ensure thread_id matches current session
    current_thread_id = st.session_state.get("thread_id") or st.session_state["session_id"]
    if thread_id != current_thread_id:
        logging.warning(
            "Thread ID mismatch: interrupt=%s, current=%s",
            thread_id,
            current_thread_id,
        )
        # Use current thread_id instead
        thread_id = current_thread_id
    
    # ... rest of resume logic ...
```

---

### 5. Network Errors During Resume
**Problem:** Network fails when resuming interrupt.  
**Solution:** Show error message, keep interrupt state so user can retry.

```python
def resume_interrupt(selected_player_id: int) -> None:
    try:
        # ... resume logic ...
    except Exception as e:
        st.error(f"Failed to resume: {e}")
        # Restore interrupt state so user can retry
        # (But this is tricky - we already deleted it)
        # Better: Don't delete until after successful resume
        logging.error("Resume failed, interrupt state lost", exc_info=True)
```

**Better Approach:** Only clear interrupt state AFTER successful resume.

```python
def resume_interrupt(selected_player_id: int) -> None:
    # Store interrupt state temporarily
    interrupt_data = st.session_state.pending_interrupt
    interrupt_thread_id = st.session_state.interrupt_thread_id
    
    try:
        # ... resume logic ...
        
        # Only clear AFTER success
        del st.session_state.pending_interrupt
        del st.session_state.interrupt_thread_id
    except Exception as e:
        st.error(f"Failed to resume: {e}")
        # Interrupt state still exists, user can retry
        logging.error("Resume failed", exc_info=True)
```

---

### 6. Interrupt in Subgraph
**Problem:** Interrupt happens inside `player_stats_sg` subgraph.  
**Impact:** Should work fine - interrupts work at any node level. The final state will still have `__interrupt__`.

---

### 7. Interrupt During Streaming
**Problem:** What if interrupt happens mid-stream?  
**Solution:** LangGraph interrupts happen at node boundaries, not during streaming. Stream completes normally, then we check final state.

---

### 8. User Never Selects
**Problem:** User abandons interrupt, never selects.  
**Solution:** Interrupt persists until:
- User selects a candidate
- User clicks "Cancel"
- User sends new message (clears interrupt)

**Future Enhancement:** Add timeout mechanism (not needed for MVP).

---

### 9. Resume with Wrong Thread ID
**Problem:** User somehow has stale thread_id.  
**Solution:** Always use current session's thread_id, validate in backend.

---

### 10. Interrupt Data Corruption
**Problem:** Interrupt data in session state is malformed.  
**Solution:** Validate interrupt data structure, show error and clear if invalid.

```python
def display_interrupt_selection() -> bool:
    if "pending_interrupt" not in st.session_state:
        return False
    
    interrupt_data = st.session_state.pending_interrupt
    
    # Validate structure
    try:
        if isinstance(interrupt_data, list):
            if len(interrupt_data) == 0:
                raise ValueError("Empty interrupt list")
            interrupt_payload = interrupt_data[0].get("value", {})
        elif isinstance(interrupt_data, dict):
            interrupt_payload = interrupt_data
        else:
            raise ValueError(f"Invalid interrupt data type: {type(interrupt_data)}")
        
        if not isinstance(interrupt_payload, dict):
            raise ValueError("Interrupt payload is not a dict")
        
        candidates = interrupt_payload.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError("No valid candidates in interrupt")
        
    except Exception as e:
        st.error(f"Invalid interrupt data: {e}")
        # Clear corrupted interrupt state
        del st.session_state.pending_interrupt
        del st.session_state.interrupt_thread_id
        return False
    
    # ... rest of display logic ...
```

---

## Alternative: Use LangGraph UI Directly

### Option: Replace Streamlit with LangGraph UI

**Pros:**
- ✅ Built-in interrupt support (already implemented)
- ✅ Better UX (React-based, more responsive)
- ✅ Actively maintained by LangChain team
- ✅ Production-ready features (auth, deployment, etc.)
- ✅ Less code to maintain

**Cons:**
- ❌ Need to set up Next.js/React development environment
- ❌ Need to configure LangGraph server endpoint
- ❌ Lose Streamlit-specific features (if any)
- ❌ Migration effort (but could be gradual)

### Implementation Steps

#### Step 1: Set Up LangGraph Server

**File:** `app/langgraph_server.py` (NEW)

```python
from fastapi import FastAPI
from langgraph.graph import StateGraph
from app.graph import agent

app = FastAPI()

@app.post("/stream_messages")
async def stream_messages(input: dict, config: dict):
    """Stream messages from the agent."""
    # Proxy to agent.stream() with proper config
    # Return SSE stream
    pass

@app.post("/feedback")
async def feedback(feedback_dict: dict):
    """Log feedback."""
    pass
```

**Or:** Use LangGraph's built-in server:
```bash
langgraph dev
```

#### Step 2: Update Makefile

**File:** `Makefile`

```makefile
# Run LangGraph server locally
langgraph-server:
	langgraph dev --port 2024

# Run LangGraph UI (requires Node.js)
langgraph-ui:
	cd chat/mlb-iris && npm run dev
```

#### Step 3: Configure LangGraph UI

**File:** `.env` (in LangGraph UI directory)

```env
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=agent
```

### Recommendation

**For MVP:** Stick with Streamlit (you already have it working, less migration risk).  
**For Production:** Consider migrating to LangGraph UI for better interrupt support and UX.

**Hybrid Approach:** Keep Streamlit for now, implement interrupts manually. Migrate to LangGraph UI later if needed.

---

## Testing Strategy

### Unit Tests

**File:** `tests/unit/test_interrupt_detection.py`

```python
def test_event_processor_detects_interrupt():
    """Test that EventProcessor correctly detects interrupt events."""
    # Mock stream with interrupt event
    # Verify interrupt_data is stored in session state
    pass

def test_resume_command_serialization():
    """Test that Command(resume=...) is properly serialized."""
    pass
```

### Integration Tests

**File:** `tests/integration/test_interrupt_flow.py`

```python
def test_full_interrupt_resume_flow():
    """Test complete interrupt → selection → resume flow."""
    # 1. Invoke agent with ambiguous query
    # 2. Verify interrupt detected
    # 3. Resume with selected player_id
    # 4. Verify final answer uses correct player
    pass

def test_interrupt_cancellation():
    """Test that canceling interrupt clears state."""
    pass

def test_interrupt_with_new_message():
    """Test that new message clears pending interrupt."""
    pass
```

### Manual Testing Checklist

- [ ] Query with ambiguous player name triggers interrupt
- [ ] Interrupt UI displays correctly with candidates
- [ ] Selecting a candidate resumes execution
- [ ] Final answer uses selected player
- [ ] Cancel button clears interrupt
- [ ] New message clears pending interrupt
- [ ] Session refresh preserves interrupt state
- [ ] Network error during resume shows error message
- [ ] Invalid interrupt data shows error and clears state

---

## Migration Path

### Phase 1: Backend (Day 1)
1. Add checkpointer to graph
2. Implement interrupt in `player_search_node`
3. Modify `agent_engine_app.py` to return final state
4. Test backend interrupt/resume with unit tests

### Phase 2: Frontend Detection (Day 2)
5. Update `EventProcessor` to detect interrupts
6. Update `Client.stream_messages` to support resume
7. Test interrupt detection (no UI yet)

### Phase 3: UI Components (Day 3)
8. Add `display_interrupt_selection()` function
9. Add `resume_interrupt()` function
10. Update `main()` to check for interrupts
11. Disable input during interrupt
12. Test full flow manually

### Phase 4: Edge Cases & Polish (Day 4)
13. Handle all edge cases
14. Add error messages
15. Add loading states
16. Final testing

---

## Success Criteria

- ✅ User queries ambiguous player name → interrupt UI appears
- ✅ User selects player → graph resumes with correct player
- ✅ User cancels → interrupt clears, error message shown
- ✅ New message during interrupt → interrupt clears
- ✅ All edge cases handled gracefully
- ✅ No UI bugs or state corruption

---

## Future Enhancements

1. **Timeout Mechanism:** Auto-cancel interrupt after X minutes
2. **Better UI:** Use Streamlit's `st.selectbox` with custom styling
3. **Multiple Interrupt Types:** Extend to team disambiguation, etc.
4. **Interrupt History:** Show previous interrupt selections
5. **Keyboard Shortcuts:** Select candidate with number keys (1, 2, 3...)
6. **Migrate to LangGraph UI:** For production deployment

---

## References

- [LangGraph Interrupts Documentation](https://langchain-ai.github.io/langgraph/how-tos/interrupts/)
- [LangGraph agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui)
- [LangGraph Types: Command](https://github.com/langchain-ai/langgraph/blob/main/langgraph/types/interrupt.py)

