"""
LangGraph node implementations for the MLB assistant agent.

All node functions follow the signature: (state: State) -> dict
They update the state dictionary and return the updated fields.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from app.utils.log_utils import log_end, log_start

from . import config
from .logic import (
    fetch_player_stats,
    find_player_id,
    generate_grounded_answer,
    generate_player_stats_answer,
)
from .planner import get_planner_agent
from .services import llm_langchain
from .state import State
from .verification import judge_answer


def extract_message_content(message: dict | object) -> str:
    """
    Extract content from a message, handling both dict and LangChain message objects.
    Handles list content (parts format) like [{'type': 'text', 'text': '...'}].

    Args:
        message: Either a dict with 'content' key or a LangChain message object with .content attribute

    Returns:
        The message content as a string

    Raises:
        ValueError: If message is a dict but missing 'content' key
        TypeError: If message is neither a dict nor has a 'content' attribute
    """

    def extract_from_content(content: str | list | object) -> str:
        """Extract text from content, handling both string and list formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle parts format: [{'type': 'text', 'text': '...'}, ...]
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(str(part["text"]))
                    elif "text" in part:
                        # Fallback: if there's a 'text' key, use it
                        text_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts) if text_parts else str(content)
        else:
            return str(content)

    if isinstance(message, dict):
        content = message.get("content")
        if content is None:
            raise ValueError(f"Message dict missing 'content': {message}")
        return extract_from_content(content)
    elif hasattr(message, "content"):
        return extract_from_content(message.content)
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
    {"route": "PLAYER_STATS" | "DOCUMENT_QA" | "MULTI_DOMAIN", "entities": {"name": "..." | null, "team": "..." | null}}

    Route Options:
    - "PLAYER_STATS": The user is asking for statistics, performance, or biographical information about a specific baseball player.
    - "DOCUMENT_QA": The user is asking for a definition, explanation, or information that would be found in a knowledge base (e.g., policies, rules, "how-to" guides).
    - "MULTI_DOMAIN": The user needs an answer that requires combining player stats with document/policy/rules knowledge, or spans multiple domains.

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
        if route not in ["PLAYER_STATS", "DOCUMENT_QA", "MULTI_DOMAIN"]:
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
        "last_user_query": query,  # Store query for verification
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

    # TESTING MODE: Always confirm player selection (even for single matches)
    # Controlled by config.ALWAYS_CONFIRM_PLAYER flag
    result = find_player_id(name, always_return_candidates=config.ALWAYS_CONFIRM_PLAYER)

    if config.ALWAYS_CONFIRM_PLAYER and isinstance(result, list):
        logging.info(
            "[player_search] TESTING MODE: Always confirming player (single match wrapped)"
        )

    if isinstance(result, int):
        # Single match - proceed normally
        logging.info("[player_search] Single match found: player_id=%r", result)
        out = {"player_id": result}
        log_end("player_search", **out)
        return out

    elif isinstance(result, list):
        # Multiple matches - interrupt for user selection
        candidates = result
        logging.info(
            "[player_search] Multiple matches found: %d candidates", len(candidates)
        )

        # Import interrupt here to avoid circular imports
        from langgraph.types import interrupt

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

        logging.info("[player_search] Resume with selected_id=%r", selected_id)
        out = {"player_id": int(selected_id)}
        log_end("player_search", player_id=selected_id, interrupted=True)
        return out

    else:
        # No matches
        logging.info("[player_search] No matches found")
        out = {"player_id": None}
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
    if route not in ["PLAYER_STATS", "DOCUMENT_QA", "MULTI_DOMAIN", "HELLO"]:
        logging.warning(
            f"[decide_route] Invalid route '{route}' in state. Defaulting to HELLO."
        )
        return "HELLO"
    return route


def planner_node(state: State) -> dict:
    """Invoke the planner agent for multi-domain queries and append its output messages.

    Fallback: if the agent returns no messages (e.g., missing LangChain at runtime),
    synthesize a combined answer by calling the logic functions directly.
    """
    log_start("planner")
    try:
        agent = get_planner_agent()
        logging.info("[planner] Invoking planner agent...")

        # Extract only the user's query message to avoid message format issues with Gemini
        # Pass only the first user message to start fresh (avoids tool call/response mismatches)
        user_messages: list[AnyMessage] = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not user_messages:
            # Fallback: use the first message if no HumanMessage found
            user_messages = [state["messages"][0]] if state["messages"] else []

        # Build context for replan attempts
        verification_reason = state.get("verification_reason")
        replan_attempts = state.get("replan_attempts", 0)
        context_parts = []

        # If this is a replan attempt, add feedback about what was missing
        if verification_reason and replan_attempts > 0:
            context_parts.append(
                f"Previous attempt was incomplete. Feedback: {verification_reason}. "
                f"Please provide a more complete answer addressing all parts of the question."
            )
            logging.info(
                "[planner] REPLAN ATTEMPT #%d: Adding feedback context to guide improvement",
                replan_attempts,
            )
            logging.info(
                "[planner] Replan feedback: %s",
                verification_reason[:200]
                if len(verification_reason) > 200
                else verification_reason,
            )
        elif replan_attempts > 0:
            logging.warning(
                "[planner] REPLAN ATTEMPT #%d but no verification_reason found in state",
                replan_attempts,
            )

        # IMPORTANT: If stats were already fetched, pass them to planner so it doesn't need to fetch again
        # This is critical for replan attempts from PLAYER_STATS subgraph
        player_id = state.get("player_id")
        stats = state.get("stats")
        if player_id and stats:
            # Format stats for context - include ALL relevant stats with season info
            hitting = (
                stats.get("stats", {}).get("hitting_season", {})
                if isinstance(stats, dict)
                else {}
            )
            stats_summary = []
            season_info = ""
            if hitting:
                # Include season first if available
                if hitting.get("season"):
                    season_info = f"for the {hitting.get('season')} season"
                if hitting.get("avg") is not None:
                    stats_summary.append(f"Batting Average: {hitting.get('avg')}")
                if hitting.get("home_runs") is not None:
                    stats_summary.append(f"Home Runs: {hitting.get('home_runs')}")
                if hitting.get("ops") is not None:
                    stats_summary.append(f"OPS: {hitting.get('ops')}")
                if hitting.get("rbi") is not None:
                    stats_summary.append(f"RBI: {hitting.get('rbi')}")
                if hitting.get("at_bats") is not None:
                    stats_summary.append(f"At Bats: {hitting.get('at_bats')}")
                if hitting.get("hits") is not None:
                    stats_summary.append(f"Hits: {hitting.get('hits')}")

            if stats_summary:
                season_text = f" {season_info}" if season_info else ""
                context_parts.append(
                    f"CRITICAL: Player statistics have already been fetched for player_id={player_id}{season_text}. "
                    f"Complete stats: {', '.join(stats_summary)}. "
                    f"You MUST use these stats directly - do NOT call get_player_statistics or search_for_player again. "
                    f"Answer the user's question using these exact statistics."
                )
                logging.info(
                    "[planner] Including pre-fetched stats in context: player_id=%s, season=%s, stats_summary=%s",
                    player_id,
                    hitting.get("season") if hitting else None,
                    stats_summary,
                )

        # Combine all context and add as system message
        if context_parts:
            replan_context = "\n\n".join(context_parts)
            user_messages.insert(0, SystemMessage(content=replan_context))
            logging.info(
                "[planner] Full replan context message: %s",
                replan_context[:300] if len(replan_context) > 300 else replan_context,
            )

        logging.info(
            "[planner] Passing %d message(s) to agent (filtered from %d total)",
            len(user_messages),
            len(state["messages"]),
        )

        out = agent.invoke({"messages": user_messages})
        new_messages = out.get("messages", [])

        # Log tool calls found in the messages
        for i, msg in enumerate(new_messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    logging.info(
                        "[planner] Agent made tool call #%d: tool=%s, args=%s",
                        i,
                        tc.get("name")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "unknown"),
                        str(tc.get("args", {}))[:200]
                        if isinstance(tc, dict)
                        else str(getattr(tc, "args", {}))[:200],
                    )

        if new_messages:
            merged = state["messages"] + new_messages
            log_end("planner", added=len(new_messages))
            return {"messages": merged}
        else:
            logging.warning(
                "[planner] Agent returned empty messages list, using fallback"
            )
    except Exception as exc:
        logging.warning("[planner] Agent invocation failed: %s", exc, exc_info=True)

    # --- Fallback path: synthesize answer directly ---
    logging.info("[planner] Fallback: synthesizing combined answer directly")
    try:
        last = state["messages"][-1]
        query = extract_message_content(last)
        logging.info(
            "[planner] Fallback: extracted query=%r",
            query[:100] if len(query) > 100 else query,
        )

        # Reuse extracted name if present
        name = state.get("extracted_name") or None
        if not name:
            m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", query)
            name = m.group(1) if m else None
        logging.info("[planner] Fallback: player_name=%r", name)

        logging.info("[planner] Fallback: fetching player stats...")
        pid = find_player_id(name) if name else None
        stats = fetch_player_stats(pid) if pid else None
        logging.info("[planner] Fallback: generating stats answer...")
        stats_answer = generate_player_stats_answer(query, stats)

        logging.info("[planner] Fallback: generating RAG answer...")
        doc_answer = generate_grounded_answer(query)

        combined = f"Answer (combined):\n\n{doc_answer}\n\n---\n\n{stats_answer}"
        res = {"messages": [AIMessage(content=combined)]}
        log_end("planner", fallback=True)
        return res
    except Exception as exc:
        logging.error("[planner] Fallback failed: %s", exc, exc_info=True)
        log_end("planner", error=True)
        return {
            "messages": [
                AIMessage(content="Sorry, I hit an error planning the answer.")
            ]
        }


def verify_answer_node(state: State) -> dict:
    """
    Verify if the final answer fully addresses the user's query.

    Extracts the original query and final answer, then uses judge_answer()
    to determine if the answer is complete. Updates replan_attempts and
    verification_status accordingly.

    Uses last_user_query from state if available, otherwise searches messages.
    """
    log_start("verify_answer")

    try:
        # PRIMARY: Use last_user_query from state (set by route_query_node)
        query: str | None = state.get("last_user_query")

        if query:
            logging.info(
                "[verify_answer] Using last_user_query from state: %s",
                query[:100] if len(query) > 100 else query,
            )
        else:
            # FALLBACK: Search messages for HumanMessage
            if not state["messages"]:
                logging.warning("[verify_answer] No messages in state")
                result = {
                    "verification_status": "OK",
                    "replan_attempts": state.get("replan_attempts", 0),
                }
                log_end("verify_answer", status="OK", reason="no_messages")
                return result

            # Log message types for debugging
            message_types = [type(msg).__name__ for msg in state["messages"]]
            logging.debug("[verify_answer] Message types in state: %s", message_types)

            # Try to find HumanMessage in messages
            # Handle both LangChain message objects and serialized dicts
            for msg in reversed(state["messages"]):
                # Check if it's a HumanMessage object
                if isinstance(msg, HumanMessage):
                    query = extract_message_content(msg)
                    logging.info(
                        "[verify_answer] Found HumanMessage (reverse search): %s",
                        query[:100] if len(query) > 100 else query,
                    )
                    break
                # Check if it's a serialized HumanMessage dict (not a LangChain object)
                elif not hasattr(msg, "content") and isinstance(msg, dict):
                    msg_type = msg.get("type") or msg.get("_type")
                    if msg_type == "human":
                        query = extract_message_content(msg)
                        logging.info(
                            "[verify_answer] Found HumanMessage dict (reverse search): %s",
                            query[:100] if len(query) > 100 else query,
                        )
                        break

            # If still not found, search from the beginning
            if query is None:
                logging.warning(
                    "[verify_answer] No HumanMessage found in reverse order; searching from start"
                )
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        query = extract_message_content(msg)
                        logging.info(
                            "[verify_answer] Found HumanMessage (forward search): %s",
                            query[:100] if len(query) > 100 else query,
                        )
                        break
                    elif not hasattr(msg, "content") and isinstance(msg, dict):
                        msg_type = msg.get("type") or msg.get("_type")
                        if msg_type == "human":
                            query = extract_message_content(msg)
                            logging.info(
                                "[verify_answer] Found HumanMessage dict (forward search): %s",
                                query[:100] if len(query) > 100 else query,
                            )
                            break

            # Last resort: use first message if it has content (but log warning)
            if query is None:
                logging.warning(
                    "[verify_answer] No HumanMessage found in state; using first message as fallback"
                )
                if state["messages"]:
                    first_msg = state["messages"][0]
                    query = extract_message_content(first_msg)
                    # Validate it's not an AIMessage (which would be wrong)
                    is_ai = isinstance(first_msg, AIMessage) or (
                        isinstance(first_msg, dict) and first_msg.get("type") == "ai"
                    )
                    if is_ai:
                        logging.error(
                            "[verify_answer] First message is AIMessage, not HumanMessage! "
                            "This will cause incorrect verification."
                        )
                else:
                    query = "Unknown query"

        # Extract final answer from last AIMessage
        # Handle both LangChain message objects and serialized dicts
        final_answer = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                final_answer = extract_message_content(msg)
                break
            elif not hasattr(msg, "content") and isinstance(msg, dict):
                msg_type = msg.get("type") or msg.get("_type")
                if msg_type == "ai":
                    final_answer = extract_message_content(msg)
                    break

        if not final_answer:
            logging.warning("[verify_answer] No AIMessage found in state")
            fallback_message = "I wasn't able to generate a full answer yet. Could you clarify or try again?"
            result = {
                "verification_status": "OK",
                "replan_attempts": state.get("replan_attempts", 0),
            }
            log_end("verify_answer", status="OK", reason="no_answer")
            return {
                **result,
                "messages": [AIMessage(content=fallback_message)],
            }

        # Judge the answer
        judge_result = judge_answer(query, final_answer)
        status = judge_result["status"]
        reason = judge_result.get("reason")

        # Update replan_attempts
        current_attempts = state.get("replan_attempts", 0)
        if status == "REPLAN":
            new_attempts = current_attempts + 1
            logging.info(
                "[verify_answer] Answer incomplete, incrementing replan_attempts: %d -> %d",
                current_attempts,
                new_attempts,
            )
            if reason:
                logging.info("[verify_answer] Replan reason: %s", reason)
        else:
            new_attempts = current_attempts
            logging.info("[verify_answer] Answer verified as complete")

        result = {
            "verification_status": status,
            "replan_attempts": new_attempts,
            "verification_reason": reason,
        }

        log_end("verify_answer", status=status, attempts=new_attempts, reason=reason)
        return result

    except Exception as exc:
        logging.error("[verify_answer] Verification failed: %s", exc, exc_info=True)
        log_end("verify_answer", error=True)
        # On error, default to OK to avoid infinite loops
        return {
            "verification_status": "OK",
            "replan_attempts": state.get("replan_attempts", 0),
            "messages": [
                AIMessage(
                    content=(
                        "I ran into an issue while double-checking the answer, but I'm "
                        "returning the latest response I have."
                    )
                )
            ],
        }


def decide_verification_route(state: State) -> str:
    """
    Route decision function for verification results.

    Returns:
        "end" if verification passed or max replans reached
        "planner" if verification failed and we should replan
    """
    status = state.get("verification_status")
    attempts = state.get("replan_attempts", 0)

    if status == "OK":
        logging.info("[verify_answer] Routing to END (verification passed)")
        return "end"

    if status == "REPLAN":
        if attempts >= config.MAX_REPLANS:
            logging.warning(
                "[verify_answer] Max replans (%d) reached, routing to END",
                config.MAX_REPLANS,
            )
            return "end"
        else:
            reason = state.get("verification_reason")
            logging.info(
                "[verify_answer] Routing to planner (attempt %d/%d). Reason: %s",
                attempts,
                config.MAX_REPLANS,
                reason[:100]
                if reason and len(reason) > 100
                else reason or "no reason provided",
            )
            return "planner"

    # Default to end if status is None or unexpected
    logging.warning("[verify_answer] Unexpected status %r, routing to END", status)
    return "end"
