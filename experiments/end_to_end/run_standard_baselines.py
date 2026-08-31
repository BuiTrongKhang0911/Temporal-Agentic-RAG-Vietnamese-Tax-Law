"""Run all four simple-RAG baselines and evaluate them end to end."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from experiments.runtime.agents._search_runtime import ChunkEvidence
from experiments.runtime.orchestrator import PlannerOrchestrator
from experiments.runtime.retriever.standard import StandardBaselineRetriever
from experiments.end_to_end.llm_judge import LLMJudge  # noqa: E402
from experiments.end_to_end.call_pacing import pause_after_call  # noqa: E402
from experiments.end_to_end.metrics import (  # noqa: E402
    jsonable,
    load_keyed_csv,
    load_keyed_jsonl,
    read_jsonl,
    save_configuration_results,
)
from experiments.end_to_end.temporal_evaluator import TemporalEvaluator  # noqa: E402


METHODS = ("dense", "bm25", "hybrid_rrf", "hybrid_rerank")
DEFAULT_GOLD = HERE / "data" / "gold_set_300_e2e.jsonl"
DEFAULT_OUTPUT = HERE / "result" / "rerun" / "baselines"


def _fetch_bm25(
    retriever: StandardBaselineRetriever,
    ranked_ids: list[tuple[object, float]],
) -> list[dict[str, Any]]:
    chunks = []
    for chunk_id, _ in ranked_ids[: retriever.semantic_candidate_limit]:
        chunk = retriever._get_chunk(chunk_id)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def retrieve_all(
    retriever: StandardBaselineRetriever,
    query: str,
) -> dict[str, list[dict[str, Any]]]:
    dense = retriever._vector_retrieve(
        query,
        limit=retriever.semantic_candidate_limit,
    )
    bm25_ranked = retriever._bm25_search(
        query,
        limit=retriever.semantic_candidate_limit,
    )
    bm25 = _fetch_bm25(retriever, bm25_ranked)
    fused_ids = retriever._reciprocal_rank_fusion(dense, bm25_ranked, k=60)
    hybrid = retriever.fetch_fused_candidates(dense, fused_ids)
    pairs = [
        [query, str(item.get("content") or item.get("original_content") or "")]
        for item in hybrid
    ]
    scores = (
        retriever.reranker.predict(
            pairs,
            batch_size=retriever.reranker_batch_size,
        )
        if pairs
        else []
    )
    reranked = sorted(
        (
            {**item, "rerank_score": float(score)}
            for item, score in zip(hybrid, scores)
        ),
        key=lambda item: item["rerank_score"],
        reverse=True,
    )
    return {
        "dense": dense[:5],
        "bm25": bm25[:5],
        "hybrid_rrf": hybrid[:5],
        "hybrid_rerank": reranked[:5],
    }


def to_evidence(chunk: dict[str, Any]) -> ChunkEvidence:
    metadata = chunk.get("metadata") or {}
    document_id = str(metadata.get("document_id") or "")
    title = str(
        metadata.get("document_title")
        or metadata.get("title")
        or document_id
    )
    return ChunkEvidence(
        chunk_id=str(chunk.get("id") or metadata.get("chunk_id") or ""),
        buoc_so=1,
        van_ban=title,
        document_id=document_id,
        provision_key=str(metadata.get("provision_key") or "") or None,
        dieu_khoan=str(metadata.get("location_label") or ""),
        tinh_trang_hieu_luc=metadata.get("status"),
        valid_from=metadata.get("valid_from"),
        valid_to=metadata.get("valid_to"),
        version=metadata.get("version"),
        noi_dung_goc=str(chunk.get("original_content") or ""),
        amended_by_document_id=metadata.get("amended_by_document_id"),
        amended_by_source_location=metadata.get("amended_by_source_location"),
        change_type=metadata.get("change_type"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--generator-model",
        default=os.getenv(
            "E2E_BASELINE_GEMINI_MODEL",
            "models/gemini-3.1-flash-lite",
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "E2E_JUDGE_GEMINI_MODEL",
            os.getenv(
                "EVALUATION_JUDGE_MODEL",
                "models/gemini-3.5-flash-lite",
            ),
        ),
    )
    parser.add_argument("--as-of", default="2026-08-30")
    parser.add_argument(
        "--call-delay",
        type=float,
        default=float(os.getenv("E2E_LLM_CALL_DELAY_SECONDS", "0")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in the repository-root .env file")
    cases = read_jsonl(args.gold)
    if len(cases) != 300:
        raise ValueError(f"Expected 300 E2E cases, found {len(cases)}")
    end = args.end or len(cases)
    selected = cases[max(0, args.start - 1):end]
    if args.limit:
        selected = selected[: args.limit]

    outputs = {
        method: (
            load_keyed_jsonl(args.output_dir / method / "outputs.jsonl")
            if args.resume
            else {}
        )
        for method in METHODS
    }
    scores: dict[str, dict[str, dict[str, Any]]] = {
        method: (
            load_keyed_csv(args.output_dir / method / "per_case_results.csv")
            if args.resume
            else {}
        )
        for method in METHODS
    }
    retriever = StandardBaselineRetriever(disable_temporal_filtering=True)
    generator = PlannerOrchestrator.agent5_only(
        backend="gemini",
        model=args.generator_model,
    )
    judge = LLMJudge(
        api_key=api_key,
        model=args.judge_model,
        call_delay=args.call_delay,
    )
    temporal = TemporalEvaluator(
        api_key=api_key,
        model=args.judge_model,
        as_of=date.fromisoformat(args.as_of),
        call_delay=args.call_delay,
    )

    print(f"Cases      : {len(selected)}")
    print(f"Generator  : {args.generator_model}")
    print(f"Judge      : {args.judge_model}")
    print(f"Call delay : {args.call_delay:g}s")
    print("Methods    : Dense | BM25 | Hybrid RRF | Hybrid + Rerank")
    for position, case in enumerate(selected, 1):
        case_id = str(case["case_id"])
        if args.resume and all(
            outputs[method].get(case_id, {}).get("inference_status") == "ok"
            for method in METHODS
        ):
            print(f"[{position}/{len(selected)}] {case_id}: inference cached")
        else:
            print(f"[{position}/{len(selected)}] {case_id}: retrieving")
            ranked = retrieve_all(retriever, str(case["query"]))
            for method in METHODS:
                started = time.perf_counter()
                evidence_models = [to_evidence(item) for item in ranked[method]]
                try:
                    answer = generator._generate_final_answer(
                        original_query=str(case["query"]),
                        evidences=evidence_models,
                        failed_steps=[],
                        incomplete_searches=[],
                        plan=[],
                        verbose=not args.quiet,
                        raise_on_error=True,
                    )
                    pause_after_call(args.call_delay)
                    status, error = "ok", ""
                except Exception as exc:
                    answer = ""
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                outputs[method][case_id] = {
                    "schema_version": 1,
                    "case_id": case_id,
                    "input_query": case["query"],
                    "retrieved_evidence": jsonable(evidence_models),
                    "final_answer": answer,
                    "inference_status": status,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "error": error,
                }
                save_configuration_results(
                    args.output_dir / method,
                    outputs[method],
                    scores[method],
                    configuration=method,
                )

        # Evaluation starts only after every baseline answer for this case exists.
        for method in METHODS:
            output = outputs[method][case_id]
            if (
                args.resume
                and scores[method].get(case_id, {}).get("judge_status") == "ok"
            ):
                print(f"  {method}: judge cached")
                continue
            row = {
                "case_id": case_id,
                "inference_status": output.get("inference_status"),
                "judge_status": "not_run",
                "duration_seconds": output.get("duration_seconds", 0),
                "error": output.get("error", ""),
            }
            if output.get("inference_status") == "ok":
                try:
                    evidence = list(output.get("retrieved_evidence") or [])
                    quality = judge.evaluate(
                        case=case,
                        answer=str(output.get("final_answer") or ""),
                        evidences=evidence,
                    )
                    temporal_result = temporal.evaluate(
                        case=case,
                        answer=str(output.get("final_answer") or ""),
                        evidences=evidence,
                    )
                    row.update(
                        {
                            **quality,
                            **temporal_result,
                            "evidence_count": len(evidence),
                            "judge_status": "ok",
                        }
                    )
                except Exception as exc:
                    row["judge_status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            scores[method][case_id] = row
            save_configuration_results(
                args.output_dir / method,
                outputs[method],
                scores[method],
                configuration=method,
            )
            print(f"  {method}: {row['judge_status']}")

    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
