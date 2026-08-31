"""Central runtime configuration for the submitted Agent 1-5 backend."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv


BE_DIR = Path(__file__).resolve().parent
SOURCE_CODE_DIR = BE_DIR.parent
SUBMISSION_DIR = SOURCE_CODE_DIR.parent

# Never search parent directories for configuration. The submission is a
# self-contained application and reads only its own optional .env file.
load_dotenv(SUBMISSION_DIR / ".env", override=False, encoding="utf-8-sig")


# Language-model configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# The E2E proposed-system model is shared by default. Optional per-Agent
# variables allow controlled stage-specific ablations without editing source.
E2E_PROPOSED_GEMINI_MODEL = os.getenv(
    "E2E_PROPOSED_GEMINI_MODEL",
    "models/gemini-3.1-flash-lite",
)
AGENT1_GEMINI_MODEL = os.getenv(
    "E2E_AGENT1_GEMINI_MODEL", E2E_PROPOSED_GEMINI_MODEL
)
AGENT2_GEMINI_MODEL = os.getenv(
    "E2E_AGENT2_GEMINI_MODEL", E2E_PROPOSED_GEMINI_MODEL
)
AGENT3_GEMINI_MODEL = os.getenv(
    "E2E_AGENT3_GEMINI_MODEL", E2E_PROPOSED_GEMINI_MODEL
)
AGENT4_GEMINI_MODEL = os.getenv(
    "E2E_AGENT4_GEMINI_MODEL", E2E_PROPOSED_GEMINI_MODEL
)
AGENT5_GEMINI_MODEL = os.getenv(
    "E2E_AGENT5_GEMINI_MODEL", E2E_PROPOSED_GEMINI_MODEL
)

AGENT_MODELS = {
    "agent1": AGENT1_GEMINI_MODEL,
    "agent2": AGENT2_GEMINI_MODEL,
    "agent3": AGENT3_GEMINI_MODEL,
    "agent4": AGENT4_GEMINI_MODEL,
    "agent5": AGENT5_GEMINI_MODEL,
}


# Retrieval configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
_requested_reranker_device = os.getenv("RERANKER_DEVICE", "auto").strip().lower()
if _requested_reranker_device == "auto":
    RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    RERANKER_DEVICE = _requested_reranker_device
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_documents")


def configure_agent_runtime() -> None:
    """Apply the Gemini-only production model assignment."""

    runtime_values = {
        "CLARIFIER_PYTHON_GUARDS": "false",
        "CLARIFIER_GEMINI_MODEL": AGENT1_GEMINI_MODEL,
        "AGENT2_GEMINI_MODEL": AGENT2_GEMINI_MODEL,
        "AGENT3_GEMINI_MODEL": AGENT3_GEMINI_MODEL,
        "AGENT4_GEMINI_MODEL": AGENT4_GEMINI_MODEL,
        "AGENT5_GEMINI_MODEL": AGENT5_GEMINI_MODEL,
    }
    os.environ.update(runtime_values)
