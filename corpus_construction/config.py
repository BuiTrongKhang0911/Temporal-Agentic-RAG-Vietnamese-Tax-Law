"""Self-contained configuration for the offline corpus pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


CORPUS_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = CORPUS_DIR.parent
load_dotenv(SUBMISSION_DIR / ".env", override=False, encoding="utf-8-sig")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CORPUS_ANALYSIS_GEMINI_MODEL = os.getenv(
    "CORPUS_ANALYSIS_GEMINI_MODEL",
    "models/gemini-3.1-flash-lite",
)
SAC_TIER1_GEMINI_MODEL = os.getenv(
    "SAC_TIER1_GEMINI_MODEL",
    "models/gemini-3.1-flash-lite",
)
SAC_TIER2_GEMINI_MODEL = os.getenv(
    "SAC_TIER2_GEMINI_MODEL",
    "models/gemini-3.1-flash-lite",
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = 1024
HF_HUB_OFFLINE = os.getenv("CORPUS_HF_OFFLINE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SIZES_TO_TEST = [500, 800, 1000]

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_documents")

INPUT_DIR = str(CORPUS_DIR / "documents")
OUTPUT_DIR = str(CORPUS_DIR / "output")
DOCUMENT_ARTIFACTS_DIR = str(CORPUS_DIR / "document_artifacts")
CHUNKS_DIR = str(CORPUS_DIR / "chunks")
CORPUS_MANIFEST_PATH = str(
    CORPUS_DIR / "corpus_manifest" / "corpus_manifest.json"
)
