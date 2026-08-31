"""Central configuration for the frozen experimental runtime snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv


RUNTIME_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = RUNTIME_DIR.parent
SUBMISSION_DIR = EXPERIMENTS_DIR.parent

# Never search parent directories for configuration. The submission is a
# self-contained experiment and reads only the submission's shared .env file.
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

# Compatibility model used by the Retriever base class. Answer generation is
# still performed by Agent 5 with AGENT5_GEMINI_MODEL.
GEMINI_MODEL = AGENT3_GEMINI_MODEL

AGENT_MODELS = {
    "agent1": AGENT1_GEMINI_MODEL,
    "agent2": AGENT2_GEMINI_MODEL,
    "agent3": AGENT3_GEMINI_MODEL,
    "agent4": AGENT4_GEMINI_MODEL,
    "agent5": AGENT5_GEMINI_MODEL,
}


# Retrieval configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = 1024
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
    """Select the submitted runtime profile."""

    profile = os.getenv("SUBMISSION_RUNTIME_PROFILE", "gemini").strip().lower()
    if profile == "qwen":
        qwen_url = os.getenv("QWEN_OLLAMA_URL", "http://127.0.0.1:11434")
        qwen_token = os.getenv("QWEN_OLLAMA_TOKEN", "").strip()
        qwen_model = os.getenv("QWEN_MODEL", "qwen3.5:27b-q4_K_M")
        qwen_num_ctx = os.getenv("QWEN_NUM_CTX", "32768")
        qwen_timeout = os.getenv("QWEN_TIMEOUT", "600")
        qwen_max_output = os.getenv("QWEN_MAX_OUTPUT", "2048")
        qwen_fact_max_output = os.getenv("QWEN_FACT_MAX_OUTPUT", "4096")
        qwen_keep_alive = os.getenv("QWEN_KEEP_ALIVE", "30m")
        runtime_values = {
            "CLARIFIER_BACKEND": "ollama",
            "AGENT2_BACKEND": "ollama_function_choice",
            "AGENT34_BACKEND": "ollama_sequential",
            "AGENT5_BACKEND": "ollama",
            "CLARIFIER_OLLAMA_URL": qwen_url,
            "CLARIFIER_OLLAMA_TOKEN": qwen_token,
            "CLARIFIER_OLLAMA_MODEL": qwen_model,
            "CLARIFIER_OLLAMA_NUM_CTX": qwen_num_ctx,
            "CLARIFIER_OLLAMA_MAX_OUTPUT": qwen_max_output,
            "CLARIFIER_OLLAMA_TIMEOUT": qwen_timeout,
            "CLARIFIER_OLLAMA_KEEP_ALIVE": qwen_keep_alive,
            "CLARIFIER_PYTHON_GUARDS": os.getenv(
                "QWEN_CLARIFIER_PYTHON_GUARDS", "false"
            ),
            "AGENT2_OLLAMA_URL": qwen_url,
            "AGENT2_OLLAMA_TOKEN": qwen_token,
            "AGENT2_OLLAMA_MODEL": qwen_model,
            "AGENT2_OLLAMA_NUM_CTX": qwen_num_ctx,
            "AGENT2_OLLAMA_MAX_OUTPUT": qwen_max_output,
            "AGENT2_2B_MAX_OUTPUT": qwen_fact_max_output,
            "AGENT2_OLLAMA_TIMEOUT": qwen_timeout,
            "AGENT2_OLLAMA_KEEP_ALIVE": qwen_keep_alive,
            "AGENT34_OLLAMA_URL": qwen_url,
            "AGENT34_OLLAMA_TOKEN": qwen_token,
            "AGENT34_OLLAMA_MODEL": qwen_model,
            "AGENT34_OLLAMA_NUM_CTX": qwen_num_ctx,
            "AGENT34_OLLAMA_MAX_OUTPUT": qwen_max_output,
            "AGENT34_OLLAMA_TIMEOUT": qwen_timeout,
            "AGENT5_OLLAMA_URL": qwen_url,
            "AGENT5_OLLAMA_TOKEN": qwen_token,
            "AGENT5_OLLAMA_MODEL": qwen_model,
            "AGENT5_OLLAMA_NUM_CTX": qwen_num_ctx,
            "AGENT5_OLLAMA_MAX_OUTPUT": qwen_max_output,
            "AGENT5_OLLAMA_TIMEOUT": qwen_timeout,
            "AGENT5_OLLAMA_KEEP_ALIVE": qwen_keep_alive,
        }
    else:
        runtime_values = {
            "CLARIFIER_BACKEND": "gemini",
            "CLARIFIER_PYTHON_GUARDS": "false",
            "AGENT2_BACKEND": "gemini",
            "AGENT34_BACKEND": "gemini_sequential",
            "AGENT5_BACKEND": "gemini",
            "CLARIFIER_GEMINI_MODEL": AGENT1_GEMINI_MODEL,
            "AGENT2_GEMINI_MODEL": AGENT2_GEMINI_MODEL,
            "AGENT3_GEMINI_MODEL": AGENT3_GEMINI_MODEL,
            "AGENT4_GEMINI_MODEL": AGENT4_GEMINI_MODEL,
            "AGENT5_GEMINI_MODEL": AGENT5_GEMINI_MODEL,
        }
    os.environ.update(runtime_values)
