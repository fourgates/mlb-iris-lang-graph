# app/services.py
from langchain_google_vertexai import ChatVertexAI
from vertexai import rag
from vertexai.preview.generative_models import GenerativeModel, Tool

from . import config

# --- LLM Clients ---
llm_langchain = ChatVertexAI(
    model_name=config.LLM, location=config.LOCATION, temperature=0.1
)

llm_native_grounding = GenerativeModel(config.LLM)


# --- Tools ---
grounding_tool = Tool.from_retrieval(
    rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=config.RAG_CORPUS_NAME,
                )
            ]
        ),
    )
)
