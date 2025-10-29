import json
import logging
import re
import time
from typing import TypedDict

from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from vertexai import rag

from app.utils.log_utils import log_end, log_start
from app.utils.mlb_tools import get_player_stats, search_player

from .services import grounding_tool, llm_langchain, llm_native_grounding


class State(TypedDict):
    messages: list
    # 'snippets' is no longer needed as we get a synthesized answer directly
    player_id: int | None
    stats: dict | None
    extracted_name: str | None
    extracted_team: str | None
    do_rag: bool
    do_player: bool


# --- DELETED: The old `retrieve_rag` node is no longer needed. ---
# --- It's replaced by `generate_rag_answer`.
# Under the hood, Google's infrastructure does everything for you:
#
# It takes your query.
# It uses the query to perform a vector search on your RAG corpus.
# It retrieves the most relevant document chunks.
# It then internally feeds those chunks to the Gemini model along with your original query.
# The Gemini model synthesizes an answer.
# Critically, as it writes the answer, it keeps track of which sentence came from which document chunk. This creates the grounding_metadata that is returned with the response.
def generate_rag_answer(state: State) -> dict:
    log_start("generate_rag_answer")
    last_message = state["messages"][-1]
    query = (
        last_message.content
        if isinstance(last_message.content, str)
        else str(last_message.content)
    )
    logging.info("[generate_rag_answer] query=%r", query)

    # --- START of new retry logic ---
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Make a single, grounded API call
            response = llm_native_grounding.generate_content(
                query,
                tools=[grounding_tool],
            )
            if not response.candidates[0].grounding_metadata:
                log_end("generate_rag_answer", note="No grounding metadata returned.")
                raw_text = (
                    response.text or "I could not find any information on that topic."
                )
                return {"messages": [AIMessage(content=raw_text)]}

            # Use the helper function from `vertexai.rag`
            rag_response = rag.add_inline_citations_and_references(
                original_text_str=response.candidates[0].content.parts[0].text,
                grounding_supports=list(
                    response.candidates[0].grounding_metadata.grounding_supports
                ),
                grounding_chunks=list(
                    response.candidates[0].grounding_metadata.grounding_chunks
                ),
            )

            # Combine the answer and sources into a single, clean message
            final_content = rag_response.cited_text
            if rag_response.final_bibliography:
                final_content += "\n\n**Sources:**\n" + rag_response.final_bibliography

            log_end(
                "generate_rag_answer", response_chars=len(final_content), success=True
            )
            return {
                "messages": [AIMessage(content=final_content)]
            }  # Success! Exit the loop and function.

        except ResourceExhausted as e:
            logging.warning(
                f"Attempt {attempt + 1}/{max_retries} failed with ResourceExhausted error: {e}"
            )
            if attempt + 1 == max_retries:
                logging.error("Max retries reached. Failing the request.")
                # You could return a user-friendly error message here
                error_message = (
                    "The service is currently busy. Please try again in a few moments."
                )
                return {"messages": [AIMessage(content=error_message)]}

            # Exponential backoff: wait 5s, 10s, etc. before the next try
            time.sleep(base_delay * (2**attempt))

    # This part should ideally not be reached, but as a fallback:
    return {
        "messages": [
            AIMessage(content="An unexpected error occurred after multiple retries.")
        ]
    }
    # --- END of new retry logic ---


def extract_player_node(state: State) -> dict:
    log_start("extract_player")
    last = state["messages"][-1]
    q = last.content if isinstance(last.content, str) else str(last.content)
    system = {
        "role": "system",
        "content": (
            "Extract MLB player name and optional team from the user question. "
            'Respond ONLY as minified JSON: {"name":"...","team":null|"..."}. '
            'If no player name, return {"name":null,"team":null}.'
        ),
    }
    human = {"role": "user", "content": f"Question: {q}"}
    out = llm_langchain.invoke(
        [system, human]
    )  # Using the langchain wrapper here is fine
    try:
        content = out.content if isinstance(out.content, str) else str(out.content)
        data = json.loads(content)
    except Exception:
        data = {"name": None, "team": None}
    logging.info("[extract_player] q=%r -> %r", q, data)
    result = {"extracted_name": data.get("name"), "extracted_team": data.get("team")}
    log_end("extract_player", **result)
    return result


def player_search_node(state: State) -> dict:
    # ... (this node remains the same)
    log_start("player_search", do_player=state.get("do_player", False))
    if not state.get("do_player", False):
        log_end("player_search", skipped=True)
        return {}
    last = state["messages"][-1]
    q = last.content if isinstance(last.content, str) else str(last.content)

    # Prefer LLM-extracted name if present; fallback to heuristic
    name = state.get("extracted_name") or None
    if not name:
        m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", q)
        name = m.group(1) if m else q
    logging.info("[player_search] query=%r extracted_name=%r", q, name)

    res = search_player(name, only_active=True)
    players = res.get("players", [])
    logging.info(
        "[player_search] matches=%d top=%s",
        len(players),
        [(p.get("full_name"), p.get("id")) for p in players[:3]],
    )

    if not players:
        return {"player_id": None}

    name_lc = name.lower().strip()
    chosen = None
    for p in players:
        if str(p.get("full_name", "")).lower().strip() == name_lc:
            chosen = p
            break
    if chosen is None:
        for p in players:
            if name_lc in str(p.get("full_name", "")).lower():
                chosen = p
                break
    if chosen is None:
        chosen = players[0]

    logging.info(
        "[player_search] chosen=%r id=%s", chosen.get("full_name"), chosen.get("id")
    )
    out = {"player_id": int(chosen["id"])}
    log_end("player_search", **out)
    return out


def player_stats_node(state: State) -> dict:
    # ... (this node remains the same)
    log_start("player_stats", do_player=state.get("do_player", False))
    if not state.get("do_player", False):
        log_end("player_stats", skipped=True)
        return {"stats": None}
    pid = state.get("player_id")
    if not pid:
        log_end("player_stats", no_player_id=True)
        return {"stats": None}
    stats = get_player_stats(pid)
    try:
        hitting = (
            stats.get("stats", {}).get("hitting_season", {})
            if isinstance(stats, dict)
            else {}
        )
        logging.info(
            "[player_stats] player_id=%s avg=%s ops=%s hr=%s",
            pid,
            hitting.get("avg"),
            hitting.get("ops"),
            hitting.get("home_runs"),
        )
    except Exception:
        pass
    out = {"stats": stats}
    log_end("player_stats", has_stats=bool(stats))
    return out


def chat_player(state: State) -> dict:
    # RENAMED from chat to chat_player to be specific
    log_start("chat_player")
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)

    # This node now only handles player-related chat
    st = state.get("stats") or {}
    hitting = (
        st.get("stats", {}).get("hitting_season", {}) if isinstance(st, dict) else {}
    )
    if hitting:
        prompt = (
            f"You are an expert MLB analyst. Here is the player's season hitting data:\n"
            f"AVG: {hitting.get('avg', '.000')}\n"
            f"HR: {hitting.get('home_runs', 0)}\n"
            f"OPS: {hitting.get('ops', '.000')}\n"
            f"RBI: {hitting.get('rbi', 0)}\n\n"
            f"Based on this data, answer the user's question.\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        prompt = query
    logging.info("[chat_player] prompt_len=%d preview=%r", len(prompt), prompt)
    out = llm_langchain.invoke(prompt)
    res = {"messages": [AIMessage(content=out.content)]}
    log_end(
        "chat_player",
        response_chars=len(out.content) if hasattr(out, "content") else None,
    )
    return res


def classify_node(state: State) -> dict:
    """
    Uses an LLM call to classify the user's intent into 'PLAYER_STATS' or 'DOCUMENT_QA'.
    """
    log_start("classify")
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)
    extracted_name = state.get("extracted_name")

    # System prompt to guide the LLM classifier
    system_prompt = f"""You are an expert classification agent. Your task is to determine the user's primary intent based on their question.
    You must choose one of the following two categories:
    1.  `PLAYER_STATS`: The user is asking for statistics, performance, or biographical information about a specific baseball player.
    2.  `DOCUMENT_QA`: The user is asking a question that should be answered by consulting a knowledge base of documents (e.g., policies, rules, "how-to" guides, explanations).

    You must respond ONLY with a minified JSON object in the format: {{"category": "YOUR_CHOICE"}}.

    ---
    Here are some examples:
    - User Question: "How many home runs did Shohei Ohtani hit last year?" -> {{"category": "PLAYER_STATS"}}
    - User Question: "What is the official policy on team travel?" -> {{"category": "DOCUMENT_QA"}}
    - User Question: "Tell me about Aaron Judge" -> {{"category": "PLAYER_STATS"}}
    - User Question: "explain how the RAG system works" -> {{"category": "DOCUMENT_QA"}}
    - User Question: "Hello there" -> {{"category": "DOCUMENT_QA"}}
    ---

    Additional context to help you decide:
    - A potential player name has already been extracted from the query: '{extracted_name if extracted_name else "None"}'. This is a strong hint towards `PLAYER_STATS`.
    """

    # Make the LLM call
    try:
        response = llm_langchain.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {query}"},
            ]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        data = json.loads(content)
        category = data.get(
            "category", "DOCUMENT_QA"
        )  # Default to RAG if key is missing
    except Exception as e:
        logging.warning(
            f"[classify] LLM classification failed or returned invalid JSON: {e}. Defaulting to DOCUMENT_QA."
        )
        category = "DOCUMENT_QA"

    # Map the category back to the state booleans
    do_player = category == "PLAYER_STATS"
    do_rag = category == "DOCUMENT_QA"

    logging.info("[classify] LLM decided category=%s for q=%r", category, query)
    out = {"do_player": do_player, "do_rag": do_rag}
    log_end("classify", **out)
    return out


# --- KEY CHANGE: Add a conditional routing function ---
def route_after_classification(state: State) -> str:
    if state.get("do_rag", False):
        return "rag_path"
    elif state.get("do_player", False):
        return "player_path"
    return END  # Should not happen with current logic, but a safe fallback


_graph = StateGraph(State)
_graph.add_node("extract_player", extract_player_node)
_graph.add_node("classify", classify_node)
_graph.add_node("generate_rag_answer", generate_rag_answer)  # New RAG node
_graph.add_node("player_search", player_search_node)
_graph.add_node("player_stats", player_stats_node)
_graph.add_node("chat_player", chat_player)  # Renamed chat node

# --- KEY CHANGE: Re-wire the graph with conditional logic ---
_graph.set_entry_point("extract_player")
_graph.add_edge("extract_player", "classify")

# Add the conditional edge for routing
_graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "rag_path": "generate_rag_answer",
        "player_path": "player_search",
    },
)

# Define the player tool-use path
_graph.add_edge("player_search", "player_stats")
_graph.add_edge("player_stats", "chat_player")
_graph.add_edge("chat_player", END)

# Define the RAG path
_graph.add_edge("generate_rag_answer", END)


agent = _graph.compile(name="Grounding Chat Graph")
