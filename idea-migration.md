Plan: Evolving the MLB Assistant to an Agentic Architecture
This plan outlines the steps to transition the current LangGraph implementation to the target ideal_flow.md architecture. We will proceed in logical phases to ensure stability and testability at each stage.
Phase 0: Foundational Refactoring for Reusability
State Schema (unified): Use LangGraph's `MessagesState` for `messages`, and extend with our fields via a TypedDict.
- messages: List of LangChain `BaseMessage` objects (from `MessagesState`)
- player_id: int | None
- stats: dict | None
- extracted_name: str | None
- extracted_team: str | None
- route: Literal["PLAYER_STATS", "DOCUMENT_QA", "FALLBACK", "MULTI_DOMAIN"] | None
- replan_attempts: int (default 0)

Goal: Decouple the core application logic (calling APIs, formatting prompts) from the graph structure (nodes.py). This is the most critical prerequisite for creating both subgraphs and agent tools.
Agent Instructions:
Create a New Logic File: Create a new file app/logic.py. This file will house the core business logic functions that are currently embedded within the graph nodes.
Refactor player_search_node:
In app/logic.py, create a function find_player_id(player_name: str) -> int | None.
Move the implementation logic from nodes.py:player_search_node (calling search_player, choosing the best match) into this new function. The function should take a name string and return a single player ID or None.
Update nodes.py:player_search_node to be a thin wrapper that calls logic.find_player_id() and updates the state.
Refactor player_stats_node:
In app/logic.py, create a function fetch_player_stats(player_id: int) -> dict | None.
Move the call to get_player_stats(pid) from nodes.py:player_stats_node into this new function.
Update nodes.py:player_stats_node to call logic.fetch_player_stats() with the ID from the state.
Refactor answer_player_stats_query:
In app/logic.py, create a function generate_player_stats_answer(query: str, stats: dict) -> str.
Move the prompt construction and the llm_langchain.invoke() call from nodes.py:answer_player_stats_query into this new function. It should return the generated string content.
Update nodes.py:answer_player_stats_query to call this function and format the result into an AIMessage.
Refactor generate_rag_answer:
In app/logic.py, create a function generate_grounded_answer(query: str) -> str.
Move the entire try...except block, including the retry logic and the call to llm_native_grounding.generate_content, into this new function. This function should handle all API logic and return the final, formatted string with citations.
Update nodes.py:generate_rag_answer to be a simple wrapper around this new function.
Verification: After this phase, run your existing tests. The application's external behavior should be identical, but the code is now modular and ready for the next steps.
Phase 1: Building Isolated Subgraphs
Goal: Encapsulate the single-domain flows (PLAYER_STATS and DOCUMENT_QA) into their own self-contained, compilable graphs.
Agent Instructions:
Create a Subgraphs File: Create a new file named app/subgraphs.py.
Define the PLAYER_STATS Subgraph:
In app/subgraphs.py, import the State TypedDict and the refactored player nodes (player_search_node, player_stats_node, answer_player_stats_query).
Create a new StateGraph(State) instance for the player stats flow.
Add the three player nodes and chain them together with .add_edge().
Set the entry point to player_search.
Set the final node (answer_player_stats_query) to connect to END.
Compile the graph: player_stats_graph = _graph.compile().
Define the DOCUMENT_QA Subgraph:
In app/subgraphs.py, import the State TypedDict and generate_rag_answer.
Create a new StateGraph(State) instance.
Add the generate_rag_answer node.
Set it as the entry point and connect it to END.
Compile the graph: document_qa_graph = _graph.compile().
Update the Main Graph (app/graph.py):
Import the compiled subgraphs from app.subgraphs.
In the main StateGraph definition, remove the individual nodes that are now inside the subgraphs (e.g., player_search, player_stats, etc.).
Add the compiled subgraphs as nodes: _graph.add_node("player_stats_sg", player_stats_graph) and _graph.add_node("document_qa_sg", document_qa_graph).
Update the add_conditional_edges call from the router to point to the new subgraph nodes: "PLAYER_STATS": "player_stats_sg" and "DOCUMENT_QA": "document_qa_sg".
Add edges from the new subgraph nodes to END.
Verification: The application flow should still be identical. You have now modularized the graph itself, which will make adding the planner agent much cleaner.
Phase 2: Implementing the Planner Agent
Goal: Create a ReAct-style agent that can use the core logic functions as tools to answer multi-domain or complex queries.
Agent Instructions:
Create an Agent Tools File: Create app/agent_tools.py.
Define Tools:
In app/agent_tools.py, import the tool decorator from langchain_core.tools and the logic functions from app.logic.
Wrap your logic functions with the @tool decorator. Add clear docstrings to each, as the agent will use these for tool selection.
@tool\ndef search_for_player(player_name: str) -> int | str: (docstring: "Searches for an MLB player by name and returns their unique player ID.")
@tool\ndef get_player_statistics(player_id: int) -> dict | str: (docstring: "Fetches season statistics for a given player ID.")
@tool\ndef query_document_knowledge_base(query: str) -> str: (docstring: "Answers questions about MLB rules, policies, and definitions using a document knowledge base.")
Create the Planner Agent:
Create a new file app/planner.py.
Import the tools from app.agent_tools.
Use LangChain's standard agent creation API `langchain.agents.create_agent` (with a `system_prompt`, optionally fetched from the LangChain Hub) to create an agent that uses your defined tools.
Invoke the agent directly (e.g., `agent.invoke({"messages": state["messages"]}, config=...)`); no separate AgentExecutor is needed.
Integrate Planner into Main Graph:
In app/graph.py, create a new node function planner_node(state: State) -> dict. This node will invoke the agent created with `create_agent` and return updated `messages`; no AgentExecutor is used.
Modify app/nodes.py:route_query_node. Update the system prompt to include a new route option: "MULTI_DOMAIN". This route should be chosen for queries that require combining player stats with document information or are too complex for a single path.
In app/graph.py, add the new planner_node to the main graph.
Update the add_conditional_edges call to include the new route: "MULTI_DOMAIN": "planner_node". Add an edge from planner_node to END.
Verification: Test with both single-domain queries (which should still work) and a new multi-domain query like: "According to the official rules, what is the injured list, and can you also tell me Aaron Judge's current home run total?"
Phase 3: Verification (LLM-as-Judge with Structured JSON for All Paths)
Goal: Use a single reusable judge to decide if the user's question was actually answered. Keep it simple; if answered → END, else → REPLAN.
Agent Instructions:
- Add `app/verification.py` with a function `judge_answer(query: str, answer: str) -> dict` that calls the existing model with a strict prompt to return a single JSON object.
- JSON schema (minimal): `{ "status": "OK" | "REPLAN" }`.
- Prompt requirement: Instruct the model to output only the JSON object with no prose.

Integration points:
- Planner (MULTI_DOMAIN): add `create_agent` middleware that, after the final AIMessage, calls `judge_answer`. If `status == "OK"`, finish; if `REPLAN`, trigger another planner iteration (or set a flag to loop once more).
- Single-domain subgraphs (PLAYER_STATS, DOCUMENT_QA): in their final step, call `judge_answer` on `(original_query, final_text)`. If `status == "OK"`, go to END; if `REPLAN`, route to `planner_node` for a second attempt.

Notes:
- Uses the same model as the rest of the graph.
- Works with `MemorySaver` and `thread_id` for durable middleware flows.
- Replan cap (configurable): Track `replan_attempts` in state and enforce a configurable `max_replans` (default 3). On each REPLAN: increment `replan_attempts`; if it reaches `max_replans`, stop replanning and either return the best-effort answer or route to FALLBACK. Expose `max_replans` via a simple setting in `app/config.py` (no runtime override needed for the POC).
Phase 4: Implementing Interrupts for Disambiguation
Goal: Pause the graph execution to ask the user for clarifying information when needed, such as when multiple players match a name search.
Agent Instructions:
Modify the Logic and Node for Ambiguity:
Update the app/logic.py:find_player_id function. Instead of always choosing one player, change its return signature to -> int | list[dict] | None. If there's one clear match, return the int ID. If there are multiple good matches, return a list of dictionaries, e.g., [{"id": 123, "name": "Will Smith"}, {"id": 456, "name": "Will Smith"}].
In app/nodes.py:player_search_node, check the type of the result from find_player_id.
If it's an int, update the state with player_id and continue.
If it's a list, this is where the interrupt happens.
Implement the Interrupt:
In player_search_node, when you detect the list of candidates, return interrupt(). Import this from `langgraph.types`. The agent framework must be compiled with a checkpointer (e.g., `MemorySaver`) for this to work.
The object returned to the user/client will be the list of players for them to choose from.
Handle Resumption:
The client code (e.g., in agent.py) will receive the interrupt. After the user makes a selection (e.g., player ID 456), the client will reinvoke the graph with the selected value.
When the graph resumes, the output of the interrupt() call inside player_search_node will be the value passed in by the client. The node can then use this ID to update the state and the graph will continue to the next step (player_stats).
Verification: Test by searching for a common name like "Will Smith". The agent should pause and return a list of choices. After providing a choice, the agent should resume and fetch the correct player's stats. Check the LangGraph documentation on "Human-in-the-loop" for detailed examples. Note that deploying this to a managed service like Vertex AI Agent Builder may have specific constraints on how interrupts are handled.
Phase 5: Finalizing Testing Strategy
Goal: Implement a comprehensive testing suite to validate the new, complex flows.
Agent Instructions:
Update Integration Tests:
Write test cases in a file like tests/integration_test.py.
Single-Domain (Player Stats): Use invoke_agent("what is aaron judge's batting average"). Assert that the response is a string containing a three-digit decimal (e.g., using a regex r"\.\d{3}").
Single-Domain (RAG): Use invoke_agent("tell me what the Injured List is"). This is harder to assert directly.
Programmatic Check: Assert that the response string contains "[1]" and "\n\n**Sources:**". This confirms the citation formatting is present.
LLM-as-Judge: For a more robust check, create a test function that calls another LLM. Provide it with the question, the generated answer, and the source text. Ask it to evaluate if the answer is factually supported by the sources and if the citations are correctly placed.
Multi-Domain: Use invoke_agent("Compare Shohei Ohtani's OPS to the policy on team travel."). Assert that the response contains both a statistic (e.g., a decimal > .700) and keywords related to the policy ("travel", "policy", etc.).
Interrupt: Write a test that specifically searches for "Will Smith" and asserts that the returned object is an interrupt/list of candidates.