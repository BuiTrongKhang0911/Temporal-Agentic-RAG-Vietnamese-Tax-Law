"""Interactive API for the latest Gemini Agent 1-5 legal RAG pipeline."""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
FE_DIR = ROOT / "frontend"

from .config import (  # noqa: E402
    AGENT_MODELS,
    EMBEDDING_DEVICE,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    configure_agent_runtime,
)


configure_agent_runtime()

from .agents.agent1 import TaxLawClarifier  # noqa: E402
from .orchestrator import PlannerOrchestrator  # noqa: E402


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    session_id: str | None = None


class ResetRequest(BaseModel):
    session_id: str


class SessionState:
    def __init__(self) -> None:
        self.clarifier = TaxLawClarifier(
            python_guards_enabled=False,
        )
        self.updated_at = time.time()


app = FastAPI(title="Vietnamese Tax Law RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()
_pipeline_lock = threading.Lock()
_orchestrator: PlannerOrchestrator | None = None
PIPELINE_VERBOSE = os.getenv("APP_VERBOSE_PIPELINE", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _log(message: str = "") -> None:
    if PIPELINE_VERBOSE:
        print(message, flush=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _get_session(session_id: str | None) -> tuple[str, SessionState]:
    key = session_id or str(uuid.uuid4())
    with _sessions_lock:
        state = _sessions.get(key)
        if state is None:
            state = SessionState()
            _sessions[key] = state
        state.updated_at = time.time()
    return key, state


def _get_orchestrator() -> PlannerOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PlannerOrchestrator()
    return _orchestrator


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "pipeline_initialized": _orchestrator is not None,
        "backends": {
            "agent1": "gemini",
            "agent2": "gemini_two_stage",
            "agent3": "gemini_sequential",
            "agent4": "gemini_sequential",
            "agent5": "gemini",
        },
        "agent1_python_guards": False,
        "models": AGENT_MODELS,
        "retrieval_devices": {
            "embedding": EMBEDDING_DEVICE,
            "reranker": RERANKER_DEVICE,
            "reranker_batch_size": RERANKER_BATCH_SIZE,
        },
        "verbose_pipeline_log": PIPELINE_VERBOSE,
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    session_id, session = _get_session(request.session_id)
    started = time.perf_counter()
    original_query = request.message.strip()

    try:
        _log("\n" + "=" * 88)
        _log(f"[REQUEST] session={session_id}")
        _log(f"[Agent 1 - Clarifier] Query goc: {original_query}")

        clarification = session.clarifier.process(original_query)
        _log(
            "[Agent 1 - Clarifier] Trang thai: "
            f"{clarification.get('status', 'unknown')}"
        )
        if clarification.get("status") == "ask":
            _log(
                "[Agent 1 - Clarifier] Cau hoi lam ro: "
                f"{clarification.get('clarification_question', '')}"
            )
            _log(
                f"[REQUEST] Ket thuc sau "
                f"{time.perf_counter() - started:.2f}s (cho nguoi dung lam ro)"
            )
            return {
                "status": "ask",
                "session_id": session_id,
                "message": clarification.get("clarification_question"),
                "clarifier": clarification,
                "duration_seconds": round(time.perf_counter() - started, 2),
            }

        normalized_query = str(
            clarification.get("normalized_query") or request.message
        ).strip()
        _log(f"[Agent 1 - Clarifier] Query chuan hoa: {normalized_query}")
        _log("[Pipeline] Bat dau Agent 2 -> Agent 3/4 -> Agent 5")
        # The production retriever and model adapters keep mutable state. A
        # process-wide lock prevents two requests from mixing search memories.
        with _pipeline_lock:
            result = _get_orchestrator().execute_workflow(
                normalized_query,
                verbose=PIPELINE_VERBOSE,
            )

        evidences = _jsonable(result.get("all_evidences", []))
        duration = time.perf_counter() - started
        _log(
            f"[Pipeline] Hoan tat: {len(evidences)} evidence, "
            f"{duration:.2f}s"
        )
        _log("=" * 88)
        return {
            "status": "completed",
            "session_id": session_id,
            "message": result.get("final_answer", ""),
            "original_query": request.message,
            "normalized_query": normalized_query,
            "clarifier": clarification,
            "plan": _jsonable(result.get("plan", [])),
            "steps": _jsonable(result.get("step_results", [])),
            "evidences": evidences,
            "evidence_count": len(evidences),
            "failed_steps": _jsonable(result.get("failed_steps", [])),
            "incomplete_searches": _jsonable(
                result.get("incomplete_searches", [])
            ),
            "duration_seconds": round(duration, 2),
        }
    except Exception as exc:
        _log(f"[Pipeline] LOI: {type(exc).__name__}: {exc}")
        _log("=" * 88)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict[str, str]:
    with _sessions_lock:
        _sessions.pop(request.session_id, None)
    return {"status": "reset", "session_id": request.session_id}


if FE_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FE_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FE_DIR / "index.html")
