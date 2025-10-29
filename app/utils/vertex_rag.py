import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Any

from google.api_core.exceptions import ServiceUnavailable
from vertexai import init as vertex_init
from vertexai.preview import rag as vertex_rag

# Defaults (override via env vars at runtime)
DEFAULT_PROJECT_ID = "mlb-iris-production"
DEFAULT_LOCATION = "us-east4"
DEFAULT_RAG_CORPUS_NAME = (
    "projects/mlb-iris-production/locations/us-east4/ragCorpora/4611686018427387904"
)
DEFAULT_TOP_K = 15


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@dataclass(slots=True)
class VertexRAGClient:
    project_id: str
    location: str
    rag_corpus_name: str
    top_k: int = 15

    @classmethod
    def from_env(cls) -> "VertexRAGClient":
        project_id = os.getenv("VERTEX_PROJECT_ID", DEFAULT_PROJECT_ID)
        location = os.getenv("VERTEX_LOCATION", DEFAULT_LOCATION)
        rag_corpus_name = os.getenv("VERTEX_RAG_CORPUS_NAME", DEFAULT_RAG_CORPUS_NAME)
        try:
            top_k = int(os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K)))
        except ValueError:
            top_k = DEFAULT_TOP_K
        return cls(
            project_id=project_id,
            location=location,
            rag_corpus_name=rag_corpus_name,
            top_k=top_k,
        )

    def _ensure_initialized(self) -> None:
        # Safe to call multiple times; underlying client caches
        vertex_init(project=self.project_id, location=self.location)

    def _retrieve(self, query_text: str) -> Any:
        self._ensure_initialized()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = vertex_rag.retrieval_query(
                    text=query_text,
                    rag_corpora=[self.rag_corpus_name],
                    similarity_top_k=self.top_k,
                    vector_distance_threshold=0.6,
                )
                # Log raw result shape and a few previews
                try:
                    contexts = getattr(getattr(result, "contexts", None), "contexts", []) or []
                    from_len = len(contexts)
                    print(f"[RAG:_retrieve] query={query_text!r} corpora={self.rag_corpus_name} top_k={self.top_k} contexts={from_len}")
                    for i, c in enumerate(contexts[:3], 1):
                        src = getattr(c, "source_uri", None)
                        dist = getattr(c, "distance", None)
                        txt = getattr(c, "text", "")
                        print(f"[RAG:_retrieve] #{i} source={src!r} distance={dist} preview={txt[:500]!r}")
                except Exception:
                    pass
                return result
            except ServiceUnavailable as e:
                last_exc = e
                time.sleep(1.0 * (2**attempt))
            except Exception as e:  # noqa: BLE001 - surface message to caller
                last_exc = e
                break
        if last_exc is not None:
            raise last_exc
        return None

    @staticmethod
    def _build_citations(raw_contexts: List[object]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        # Returns (tagged_snippets, sources)
        # tagged_snippets: [(tag, snippet)]
        # sources: [(tag, uri)]
        citation_map: dict[str, str] = {}
        tagged_snippets: List[Tuple[str, str]] = []
        sources: List[Tuple[str, str]] = []

        if not raw_contexts:
            return tagged_snippets, sources

        for ctx in raw_contexts:
            uri = getattr(ctx, "source_uri", None) or ""
            if uri and uri not in citation_map:
                citation_map[uri] = str(len(citation_map) + 1)
                sources.append((citation_map[uri], uri))
            tag = citation_map.get(uri) or str(len(citation_map) + 1)
            text = getattr(ctx, "text", "").strip()
            if text:
                tagged_snippets.append((tag, text))
        return tagged_snippets, sources

    def search(self, query: str) -> str:
        """Query Vertex RAG and return concise, cited snippets in text form.

        Output format:
            (1) snippet text...
            (2) snippet text...

            Sources:
            (1) <uri>
            (2) <uri>
        """
        result = self._retrieve(query_text=query)
        contexts = []
        if (
            result is not None
            and getattr(result, "contexts", None) is not None
            and getattr(result.contexts, "contexts", None)
        ):
            contexts = list(result.contexts.contexts)

        tagged_snippets, sources = self._build_citations(contexts)

        if not tagged_snippets:
            return "No relevant information found in the RAG corpus."

        lines: List[str] = [
            f"({tag}) {text}" for tag, text in tagged_snippets
        ]

        if sources:
            lines.append("")
            lines.append("Sources:")
            for tag, uri in sources:
                lines.append(f"({tag}) {uri}")

        return "\n".join(lines)


