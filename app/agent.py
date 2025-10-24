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
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from app.utils.vertex_rag import VertexRAGClient

LOCATION = "global"
LLM = "gemini-2.5-flash"

llm = ChatVertexAI(model=LLM, location=LOCATION, temperature=0)


class State(TypedDict):
    messages: list
    snippets: list[str]


_rag = VertexRAGClient.from_env()


def retrieve_rag(state: State) -> dict:
    last = state["messages"][-1]
    query = last.content if isinstance(last.content, str) else str(last.content)
    text = _rag.search(query)
    # Keep only the cited lines for compact prompts; fall back to whole text
    lines = [ln for ln in text.splitlines() if ln.startswith("(")] or [text]
    return {"snippets": lines}


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
        prompt = query
    out = llm.invoke(prompt)
    return {"messages": [AIMessage(content=out.content)]}


_graph = StateGraph(State)
_graph.add_node("retrieve_rag", retrieve_rag)
_graph.add_node("chat", chat)
_graph.add_edge(START, "retrieve_rag")
_graph.add_edge("retrieve_rag", "chat")
_graph.add_edge("chat", END)

agent = _graph.compile(name="Minimal Chat Graph")
