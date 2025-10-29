"""
LangGraph node implementations for the MLB assistant agent.

All node functions follow the signature: (state: State) -> dict
They update the state dictionary and return the updated fields.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import AIMessage
from vertexai import rag

from app.utils.log_utils import log_end, log_start
from app.utils.mlb_tools import get_player_stats, search_player

from .services import grounding_tool, llm_langchain, llm_native_grounding

if TYPE_CHECKING:
    from .graph import State


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


def hello_node(state: State) -> dict:
    """
    Handles cases where the query couldn't be properly routed.
    Explains agent capabilities and informs the user their query couldn't be processed.
    """
    log_start("hello")
    message = (
        "I'm an MLB assistant agent with two main capabilities:\n\n"
        "1. **Player Statistics**: I can help you find statistics, performance data, "
        "and biographical information about specific MLB players. "
        "For example: 'Tell me about Aaron Judge' or 'What are Shohei Ohtani's stats?'\n\n"
        "2. **Document Q&A**: I can answer questions by consulting a knowledge base of "
        "documents including policies, rules, guides, and explanations. "
        "For example: 'What is the policy on team travel?' or 'Explain how the draft works.'\n\n"
        "I'm sorry, but I wasn't able to process your query. Please try rephrasing your question "
        "or ask about a specific player or topic from my knowledge base."
    )
    log_end("hello", response_chars=len(message))
    return {"messages": [AIMessage(content=message)]}


def route_query_node(state: State) -> dict:
    """
    Combines entity extraction and intent classification in a single LLM call.
    Determines the route (PLAYER_STATS or DOCUMENT_QA) and extracts player entities if present.
    """
    log_start("route_query")
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)

    system_prompt = """You are an expert routing agent for an MLB assistant. Your task is to analyze the user's query and return a JSON object that specifies the routing decision and any extracted entities.

    **You must respond ONLY with a single, minified JSON object and nothing else. Do not include any text, explanations, or markdown formatting before or after the JSON object.**

    The JSON object must have this exact format:
    {"route": "PLAYER_STATS_PATH" | "DOCUMENT_QA", "entities": {"name": "..." | null, "team": "..." | null}}

    Route Options:
    - "PLAYER_STATS": The user is asking for statistics, performance, or biographical information about a specific baseball player.
    - "DOCUMENT_QA": The user is asking for a definition, explanation, or information that would be found in a knowledge base (e.g., policies, rules, "how-to" guides).

    Examples:
    - User Query: "Tell me about Aaron Judge of the Yankees"
    -> {"route": "PLAYER_STATS_PATH", "entities": {"name": "Aaron Judge", "team": "Yankees"}}
    - User Query: "What is the policy on team travel?"
    -> {"route": "DOCUMENT_QA", "entities": {"name": null, "team": null}}
    - User Query: "tell me what the Injured List is"  # <-- NEW EXAMPLE
    -> {"route": "DOCUMENT_QA", "entities": {"name": null, "team": null}}
    - User Query: "Hello there"
    -> {"route": "DOCUMENT_QA", "entities": {"name": null, "team": null}}
    """

    try:
        response = llm_langchain.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Query: {query}"},
            ]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        data = json.loads(content)
        route = data.get("route", "HELLO")
        entities = data.get("entities", {})
        extracted_name = entities.get("name") if entities else None
        extracted_team = entities.get("team") if entities else None

        # Validate route is one of the expected values
        if route not in ["PLAYER_STATS", "DOCUMENT_QA"]:
            logging.warning(
                f"[route_query] Invalid route '{route}' returned. Defaulting to HELLO."
            )
            route = "HELLO"
            extracted_name = None
            extracted_team = None

    except Exception as e:
        logging.warning(
            f"[route_query] LLM routing failed or returned invalid JSON: {e}. Defaulting to DOCUMENT_QA."
        )
        route = "DOCUMENT_QA"
        extracted_name = None
        extracted_team = None

    logging.info(
        "[route_query] q=%r -> route=%s name=%r team=%r",
        query,
        route,
        extracted_name,
        extracted_team,
    )
    result = {
        "route": route,
        "extracted_name": extracted_name,
        "extracted_team": extracted_team,
    }
    log_end("route_query", **result)
    return result


def player_search_node(state: State) -> dict:
    log_start("player_search")
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
    log_start("player_stats")
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


# --- Routing function for conditional edges ---
def decide_route(state: State) -> str:
    """
    Returns the route string from state, defaulting to HELLO if missing or invalid.
    """
    route = state.get("route", "HELLO")
    if route not in ["PLAYER_STATS", "DOCUMENT_QA", "HELLO"]:
        logging.warning(
            f"[decide_route] Invalid route '{route}' in state. Defaulting to HELLO."
        )
        return "HELLO"
    return route
