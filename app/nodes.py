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

from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import AIMessage
from vertexai import rag

from app.utils.log_utils import log_end, log_start
from .logic import (
    find_player_id,
    fetch_player_stats,
    generate_player_stats_answer,
    generate_grounded_answer,
)
from .services import llm_langchain
from .graph import State


def extract_message_content(message: dict | object) -> str:
    """
    Extract content from a message, handling both dict and LangChain message objects.

    Args:
        message: Either a dict with 'content' key or a LangChain message object with .content attribute

    Returns:
        The message content as a string

    Raises:
        ValueError: If message is a dict but missing 'content' key
        TypeError: If message is neither a dict nor has a 'content' attribute
    """
    if isinstance(message, dict):
        content = message.get("content")
        if content is None:
            raise ValueError(f"Message dict missing 'content': {message}")
        return str(content) if not isinstance(content, str) else content
    elif hasattr(message, "content"):
        return (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")


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
    query = extract_message_content(last_message)
    logging.info("[generate_rag_answer] query=%r", query)
    final_content = generate_grounded_answer(query)
    log_end("generate_rag_answer", response_chars=len(final_content), success=True)
    return {"messages": [AIMessage(content=final_content)]}


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
    query = extract_message_content(last)

    system_prompt = """You are an expert routing agent for an MLB assistant. Your task is to analyze the user's query and return a JSON object that specifies the routing decision and any extracted entities.

    **You must respond ONLY with a single, minified JSON object and nothing else. Do not include any text, explanations, or markdown formatting before or after the JSON object.**

    The JSON object must have this exact format:
    {"route": "PLAYER_STATS" | "DOCUMENT_QA", "entities": {"name": "..." | null, "team": "..." | null}}

    Route Options:
    - "PLAYER_STATS": The user is asking for statistics, performance, or biographical information about a specific baseball player.
    - "DOCUMENT_QA": The user is asking for a definition, explanation, or information that would be found in a knowledge base (e.g., policies, rules, "how-to" guides).

    Examples:
    - User Query: "Tell me about Aaron Judge of the Yankees"
    -> {"route": "PLAYER_STATS", "entities": {"name": "Aaron Judge", "team": "Yankees"}}
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
    q = extract_message_content(last)

    # Prefer LLM-extracted name if present; fallback to heuristic
    name = state.get("extracted_name") or None
    if not name:
        m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", q)
        name = m.group(1) if m else q
    logging.info("[player_search] query=%r extracted_name=%r", q, name)

    player_id = find_player_id(name)
    logging.info("[player_search] chosen_id=%r", player_id)
    out = {"player_id": int(player_id) if player_id is not None else None}
    log_end("player_search", **out)
    return out


def player_stats_node(state: State) -> dict:
    log_start("player_stats")
    pid = state.get("player_id")
    if not pid:
        log_end("player_stats", no_player_id=True)
        return {"stats": None}
    stats = fetch_player_stats(pid)
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


def answer_player_stats_query(state: State) -> dict:
    """
    Generates an answer to the user's query using the player's statistics as context.
    Constructs a prompt with the player's hitting stats and asks the LLM to answer the query.
    """
    log_start("answer_player_stats_query")
    last = state["messages"][-1]
    query = extract_message_content(last)

    # This node generates responses using player stats as context
    st = state.get("stats") or {}
    content = generate_player_stats_answer(query, st)
    res = {"messages": [AIMessage(content=content)]}
    log_end(
        "answer_player_stats_query",
        response_chars=len(content),
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
