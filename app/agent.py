# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="union-attr"
import logging
import re
import json
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from app.utils.vertex_rag import VertexRAGClient
from app.utils.mlb_tools import search_player, get_player_stats

LOCATION = "global"
LLM = "gemini-2.5-flash"

llm = ChatVertexAI(model=LLM, location=LOCATION, temperature=0)


class State(TypedDict):
    messages: list
    snippets: list[str]
    player_id: int | None
    stats: dict | None
    extracted_name: str | None
    extracted_team: str | None


_rag = VertexRAGClient.from_env()


def retrieve_rag(state: State) -> dict:
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)
    text = _rag.search(query)
    # Keep only the cited lines for compact prompts; fall back to whole text
    lines = [ln for ln in text.splitlines() if ln.startswith("(")] or [text]
    return {"snippets": lines}


def extract_player_node(state: State) -> dict:
    last = state["messages"][-1]
    q = last.content if isinstance(last.content, str) else str(last.content)
    system = {
        "role": "system",
        "content": (
            "Extract MLB player name and optional team from the user question. "
            "Respond ONLY as minified JSON: {\"name\":\"...\",\"team\":null|\"...\"}. "
            "If no player name, return {\"name\":null,\"team\":null}."
        ),
    }
    human = {"role": "user", "content": f"Question: {q}"}
    out = llm.invoke([system, human])
    try:
        data = json.loads(out.content)
    except Exception:
        data = {"name": None, "team": None}
    logging.info("[extract_player] q=%r -> %r", q, data)
    return {"extracted_name": data.get("name"), "extracted_team": data.get("team")}
def player_search_node(state: State) -> dict:
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
    logging.info("[player_search] matches=%d top=%s", len(players),
                 [(p.get("full_name"), p.get("id")) for p in players[:3]])

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

    logging.info("[player_search] chosen=%r id=%s", chosen.get("full_name"), chosen.get("id"))
    return {"player_id": int(chosen["id"])}


def player_stats_node(state: State) -> dict:
    pid = state.get("player_id")
    if not pid:
        return {"stats": None}
    stats = get_player_stats(pid)
    try:
        hitting = stats.get("stats", {}).get("hitting_season", {}) if isinstance(stats, dict) else {}
        logging.info("[player_stats] player_id=%s avg=%s ops=%s hr=%s",
                     pid, hitting.get("avg"), hitting.get("ops"), hitting.get("home_runs"))
    except Exception:
        pass
    return {"stats": stats}


def chat(state: State) -> dict:
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)
    snippets = state.get("snippets", [])
    if snippets:
        prompt = (
            "Use the cited snippets to answer the question. Cite tags like (1) when used.\n\n"
            + "\n".join(snippets)
            + f"\n\nQuestion: {query}\nAnswer:"
        )
    else:
        # If we have stats, surface basics inline
        st = state.get("stats") or {}
        hitting = st.get("stats", {}).get("hitting_season", {}) if isinstance(st, dict) else {}
        if hitting:
            prompt = (
                f"Player hitting (season): AVG {hitting.get('avg', '.000')}, "
                f"HR {hitting.get('home_runs', 0)}, OPS {hitting.get('ops', '.000')}.\n\n"
                f"Question: {query}\nAnswer:"
            )
        else:
            prompt = query
    out = llm.invoke(prompt)
    return {"messages": [AIMessage(content=out.content)]}


_graph = StateGraph(State)
#_graph.add_node("retrieve_rag", retrieve_rag)
_graph.add_node("extract_player", extract_player_node)
_graph.add_node("player_search", player_search_node)
_graph.add_node("player_stats", player_stats_node)
_graph.add_node("chat", chat)
_graph.add_edge(START, "extract_player")
_graph.add_edge("extract_player", "player_search")
#_graph.add_edge("retrieve_rag", "player_search")
_graph.add_edge("player_search", "player_stats")
_graph.add_edge("player_stats", "chat")
_graph.add_edge("chat", END)

agent = _graph.compile(name="Minimal Chat Graph")
