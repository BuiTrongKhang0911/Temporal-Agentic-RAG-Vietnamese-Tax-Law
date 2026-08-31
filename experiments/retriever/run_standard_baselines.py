"""Evaluate four retrieval configurations in one pass over the locked 100 cases.

For each temporal branch, Dense and BM25 retrieval are executed once. Their
outputs are reused to evaluate Dense, BM25, Hybrid RRF, and Hybrid + Rerank
under the same eligible corpus, candidate limits, and scoring procedure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from experiments.retriever.run_proposed_retriever import (  # noqa: E402
    collapse_ws,
    ndcg,
    provision_identity,
    qrel_identity,
    read_jsonl,
    retrieval_scopes,
    write_csv,
)


DEFAULT_GOLD = HERE / "data" / "gold_set_100_locked.jsonl"
DEFAULT_RESULT_DIR = HERE / "result" / "standard_baselines"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "latest_per_case_results.csv"
DEFAULT_SUMMARY = DEFAULT_RESULT_DIR / "latest_summary.json"

METHODS = ("dense", "bm25", "hybrid_rrf", "hybrid_rerank")
QUALITY_METRICS = (
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "fact_coverage",
    "evidence_precision",
)
COUNT_METRICS = (
    "ranked_subchunks",
    "context_subchunks",
    "context_chars",
    "context_words",
    "estimated_context_tokens",
    "candidate_pool",
)


def fetch_by_ids(
    retriever: StandardBaselineRetriever,
    chunk_ids: list[object],
) -> list[dict[str, Any]]:
    """Fetch cached chunks while preserving the BM25 rank order."""

    chunks: list[dict[str, Any]] = []
    for chunk_id in chunk_ids[: retriever.semantic_candidate_limit]:
        chunk = retriever._get_chunk(chunk_id)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def rerank_text(chunk: dict[str, Any]) -> str:
    """Return the candidate passage supplied to the Cross-Encoder baseline."""

    return str(
        chunk.get("content")
        or chunk.get("original_content")
        or ""
    )


def cross_encoder_rerank(
    retriever: StandardBaselineRetriever,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rerank fused subchunks without structural or provision expansion."""

    if not candidates:
        return []
    scores = retriever.reranker.predict(
        [[query, rerank_text(candidate)] for candidate in candidates],
        batch_size=retriever.reranker_batch_size,
    )
    ranked = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    return sorted(
        ranked,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )


def retrieve_four_configurations(
    retriever: StandardBaselineRetriever,
    query: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Create all four rankings from one Dense and one BM25 retrieval call."""

    dense_candidates = retriever._vector_retrieve(
        query,
        limit=retriever.semantic_candidate_limit,
    )
    bm25_results = retriever._bm25_search(
        query,
        limit=retriever.semantic_candidate_limit,
    )
    bm25_candidates = fetch_by_ids(
        retriever,
        [chunk_id for chunk_id, _ in bm25_results],
    )
    fused_ids = retriever._reciprocal_rank_fusion(
        dense_candidates,
        bm25_results,
    )
    hybrid_candidates = retriever.fetch_fused_candidates(
        dense_candidates,
        fused_ids,
    )
    reranked_candidates = cross_encoder_rerank(
        retriever,
        query,
        hybrid_candidates,
    )

    candidate_lists = {
        "dense": dense_candidates,
        "bm25": bm25_candidates,
        "hybrid_rrf": hybrid_candidates,
        "hybrid_rerank": reranked_candidates,
    }
    selected = {
        method: list(candidates[:5])
        for method, candidates in candidate_lists.items()
    }
    pool_sizes = {
        method: len(candidates)
        for method, candidates in candidate_lists.items()
    }
    return selected, pool_sizes


def evaluate_top_chunks(
    chunks: list[dict[str, Any]],
    gold: dict[str, Any],
) -> dict[str, Any]:
    """Score the raw top-five subchunks against provision-version qrels.

    A relevant provision version receives location-level credit at most once.
    Later subchunks from the same version remain in the top-five denominator but
    do not create additional relevance. Evidence metrics inspect each returned
    subchunk independently for a locked continuous support span.
    """

    qrels = {
        qrel_identity(qrel): qrel
        for qrel in gold["qrels"]
        if int(qrel["relevance"]) > 0
    }
    fact_ids = {fact["fact_id"] for fact in gold["gold_facts"]}
    seen_relevant: set[str] = set()
    observed_relevance: list[int] = []

    for chunk in chunks:
        identity = provision_identity(chunk)
        qrel = qrels.get(identity)
        if qrel is None or identity in seen_relevant:
            observed_relevance.append(0)
            continue
        seen_relevant.add(identity)
        observed_relevance.append(int(qrel["relevance"]))

    recall = len(seen_relevant) / len(qrels) if qrels else 0.0
    precision = len(seen_relevant) / len(chunks) if chunks else 0.0
    ideal_relevance = sorted(
        (int(qrel["relevance"]) for qrel in qrels.values()),
        reverse=True,
    )
    score_ndcg = ndcg(
        observed_relevance,
        ideal_relevance[: len(observed_relevance)],
    )

    covered_facts: set[str] = set()
    useful_chunks = 0
    context_parts: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("original_content") or "")
        context_parts.append(text)
        qrel = qrels.get(provision_identity(chunk))
        if qrel is None:
            continue
        normalized_text = collapse_ws(text)
        chunk_is_useful = False
        for support in qrel.get("support_spans") or []:
            if collapse_ws(support.get("text")) not in normalized_text:
                continue
            supported = set(support.get("fact_ids") or []) & fact_ids
            if supported:
                covered_facts.update(supported)
                chunk_is_useful = True
        if chunk_is_useful:
            useful_chunks += 1

    fact_coverage = len(covered_facts) / len(fact_ids) if fact_ids else 0.0
    evidence_precision = useful_chunks / len(chunks) if chunks else 0.0
    context = "\n\n".join(context_parts)
    return {
        "ranked_subchunks": len(chunks),
        "context_subchunks": len(chunks),
        "recall_at_k": round(recall, 6),
        "precision_at_k": round(precision, 6),
        "ndcg_at_k": round(score_ndcg, 6),
        "fact_coverage": round(fact_coverage, 6),
        "evidence_precision": round(evidence_precision, 6),
        "context_chars": len(context),
        "context_words": len(context.split()),
        "estimated_context_tokens": math.ceil(len(context) / 4),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average every metric over successful locked test cases."""

    successful = [row for row in rows if row.get("status") == "ok"]
    methods: dict[str, dict[str, float]] = {}
    for method in METHODS:
        values: dict[str, float] = {}
        for metric in QUALITY_METRICS + COUNT_METRICS:
            observations = [
                float(row[f"{method}_{metric}"])
                for row in successful
            ]
            values[metric] = (
                round(sum(observations) / len(observations), 6)
                if observations
                else 0.0
            )
        methods[method] = values

    return {
        "schema_version": 1,
        "cases": len(rows),
        "successful_cases": len(successful),
        "metric_unit": "0_to_1_except_counts_and_context_size",
        "configuration": {
            "candidate_limit_per_dense_or_bm25_branch": 20,
            "top_subchunks_per_temporal_scope": 5,
            "dense_encoder": "BAAI/bge-m3",
            "lexical_retriever": "BM25 with Vietnamese word segmentation",
            "fusion": "RRF(k=60)",
            "cross_encoder_reranker": "BAAI/bge-reranker-base",
            "indexed_content": "SAC Tier 1 + SAC Tier 2 + original subchunk",
            "reranker_passage": "candidate subchunk content",
            "disabled_for_all_methods": (
                "same-Article expansion, complete-provision restoration, "
                "required-fact-aware reranking, and sibling expansion"
            ),
        },
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dense, BM25, Hybrid RRF, and Hybrid + Rerank together."
        )
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N locked cases for a quick smoke test.",
    )
    args = parser.parse_args()

    from experiments.runtime.retriever.standard import StandardBaselineRetriever

    cases = read_jsonl(args.gold)
    if len(cases) != 100:
        raise ValueError(
            f"Expected exactly 100 locked Retriever cases, found {len(cases)}."
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero.")
        cases = cases[: args.limit]
    print(f"Gold set       : {args.gold}", flush=True)
    print(f"Selected cases : {len(cases)}", flush=True)
    print(
        "Methods        : Dense | BM25 | Hybrid RRF | Hybrid + Rerank",
        flush=True,
    )

    retriever = StandardBaselineRetriever()
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        scopes = retrieval_scopes(case["expected_temporal"])
        method_items: dict[str, list[dict[str, Any]]] = {
            method: [] for method in METHODS
        }
        pool_totals = {method: 0 for method in METHODS}

        print("\n" + "=" * 88, flush=True)
        print(
            f"[{index}/{len(cases)}] {case['benchmark_id']} | "
            f"branches={len(scopes)}",
            flush=True,
        )
        print(f"Query: {case['query']}", flush=True)

        for scope_index, scope in enumerate(scopes, 1):
            retriever.set_temporal_scope(scope)
            selected, pool_sizes = retrieve_four_configurations(
                retriever,
                case["query"],
            )
            for method in METHODS:
                method_items[method].extend(selected[method])
                pool_totals[method] += pool_sizes[method]
            print(
                f"  Branch {scope_index}/{len(scopes)}: "
                + " | ".join(
                    f"{method} pool={pool_sizes[method]} "
                    f"top={len(selected[method])}"
                    for method in METHODS
                ),
                flush=True,
            )

        row: dict[str, Any] = {
            "benchmark_id": case["benchmark_id"],
            "case_id": case["case_id"],
            "dataset_group": case["dataset_group"],
            "difficulty": case["difficulty"],
            "query": case["query"],
            "gold_fact_count": len(case["gold_facts"]),
            "gold_qrel_count": len(case["qrels"]),
            "status": "ok",
        }
        for method in METHODS:
            result = evaluate_top_chunks(method_items[method], case)
            row[f"{method}_candidate_pool"] = pool_totals[method]
            for metric, value in result.items():
                row[f"{method}_{metric}"] = value

        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        write_csv(args.output, rows)
        print(
            f"[{index}/{len(cases)}] completed in "
            f"{row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    report = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"CSV detail : {args.output}", flush=True)
    print(f"JSON report: {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
