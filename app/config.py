import os

# --- Configuration ---
LOCATION = "us-east4"  # Grounding is not available in "global"
LLM = "gemini-2.5-flash"

RAG_CORPUS_NAME = os.getenv(
    "VERTEX_RAG_CORPUS_NAME",
    "projects/mlb-iris-production/locations/us-east4/ragCorpora/4611686018427387904"
)