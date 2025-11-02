import os

# --- Configuration ---
LOCATION = "us-east4"  # Grounding is not available in "global"
LLM = "gemini-2.5-flash"

RAG_CORPUS_NAME = os.getenv(
    "VERTEX_RAG_CORPUS_NAME",
    "projects/mlb-iris-production/locations/us-east4/ragCorpora/4611686018427387904",
)

# --- Verification Configuration ---
MAX_REPLANS = 3  # Maximum number of replan attempts before giving up

# --- Interrupt Testing Configuration ---
# When True, always trigger player confirmation interrupt (even for single matches)
# Useful for testing interrupt flow. Set to False for production behavior.
ALWAYS_CONFIRM_PLAYER = os.getenv("ALWAYS_CONFIRM_PLAYER", "false").lower() == "true"
