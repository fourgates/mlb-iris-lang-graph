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
from langgraph.prebuilt import create_react_agent

from app.utils.vertex_rag import VertexRAGClient

LOCATION = "global"
LLM = "gemini-2.5-flash"

llm = ChatVertexAI(model=LLM, location=LOCATION, temperature=0)



_rag_client = None


def _get_rag_client() -> VertexRAGClient:
    global _rag_client
    if _rag_client is None:
        _rag_client = VertexRAGClient.from_env()
    return _rag_client


def search_corpus(query: str) -> str:
    """Searches Vertex RAG corpus and returns concise, cited snippets."""
    return _get_rag_client().search(query)


agent = create_react_agent(
    model=llm,
    tools=[search_corpus],
    prompt=(
        "You are a helpful assistant. Use search_corpus for questions needing external knowledge. "
        "When you use it, include brief citations like (1) (2) referring to sources."
    ),
)
