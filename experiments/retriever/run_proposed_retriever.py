"""Evaluate the two reported configurations of the Structured Retriever."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_GOLD = HERE / "data" / "gold_set_100_locked.jsonl"
DEFAULT_RESULT_DIR = HERE / "result" / "proposed_retriever"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "latest_per_case_results.csv"
DEFAULT_SUMMARY = DEFAULT_RESULT_DIR / "latest_summary.json"
STAGES = (
    (
        "full_provision",
        "stage_b_ranked_locations",
        "stage_b_top_locations_full_provisions",
    ),
    (
        "full_structured",
        "stage_c_ranked_locations_with_siblings",
        "stage_c_full_context_with_siblings",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collapse_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def provision_identity(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    version_id = str(metadata.get("version_id") or item.get("version_id") or "").strip()
    if version_id:
        return version_id
    provision_key = str(metadata.get("provision_key") or "").strip()
    version = metadata.get("version")
    if provision_key:
        return f"{provision_key}|V{version or 1}"
    return f"POINT|{item.get('id')}"


def qrel_identity(qrel: dict[str, Any]) -> str:
    version_id = str(qrel.get("version_id") or "").strip()
    if version_id:
        return version_id
    return f"{qrel['provision_key']}|V{qrel.get('version') or 1}"


def dedupe_ranked(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        identity = provision_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def ndcg(relevances: list[int], ideal: list[int]) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    denominator = dcg(ideal)
    return dcg(relevances) / denominator if denominator else 0.0


def stage_content(stage_name: str, item: dict[str, Any]) -> str:
    return str(item.get("provision_content") or item.get("original_content") or "")


def evaluate_stage(
    stage_name: str,
    ranking_items: list[dict[str, Any]],
    context_items: list[dict[str, Any]],
    gold: dict[str, Any],
) -> dict[str, Any]:
    ranked = dedupe_ranked(ranking_items)
    context_ranked = dedupe_ranked(context_items)
    qrels = {qrel_identity(qrel): qrel for qrel in gold["qrels"] if int(qrel["relevance"]) > 0}
    ranked_ids = [provision_identity(item) for item in ranked]
    retrieved_gold = [identity for identity in ranked_ids if identity in qrels]

    recall = len(set(retrieved_gold)) / len(qrels) if qrels else 0.0
    precision = len(retrieved_gold) / len(ranked_ids) if ranked_ids else 0.0
    observed_rel = [int(qrels.get(identity, {}).get("relevance") or 0) for identity in ranked_ids]
    ideal_rel = sorted((int(qrel["relevance"]) for qrel in qrels.values()), reverse=True)
    stage_ndcg = ndcg(observed_rel, ideal_rel[: len(observed_rel)])

    fact_ids = {fact["fact_id"] for fact in gold["gold_facts"]}
    covered_facts: set[str] = set()
    useful_ids: set[str] = set()
    context_parts: list[str] = []
    for item in context_ranked:
        identity = provision_identity(item)
        text = stage_content(stage_name, item)
        context_parts.append(text)
        normalized_text = collapse_ws(text)
        qrel = qrels.get(identity)
        if not qrel:
            continue
        item_useful = False
        for support in qrel.get("support_spans") or []:
            if collapse_ws(support.get("text")) in normalized_text:
                supported = set(support.get("fact_ids") or []) & fact_ids
                if supported:
                    covered_facts.update(supported)
                    item_useful = True
        if item_useful:
            useful_ids.add(identity)

    fact_coverage = len(covered_facts) / len(fact_ids) if fact_ids else 0.0
    context_ids = [provision_identity(item) for item in context_ranked]
    utility_precision = len(useful_ids) / len(context_ids) if context_ids else 0.0
    context = "\n\n".join(context_parts)
    return {
        "ranked_provisions": len(ranked_ids),
        "context_provisions": len(context_ids),
        "returned_provisions": len(context_ids),
        "recall_at_k": round(recall, 6),
        "precision_at_k": round(precision, 6),
        "ndcg_at_k": round(stage_ndcg, 6),
        "fact_coverage": round(fact_coverage, 6),
        "evidence_precision": round(utility_precision, 6),
        "context_chars": len(context),
        "context_words": len(context.split()),
        "estimated_context_tokens": math.ceil(len(context) / 4),
    }


def merge_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    trace_keys = {
        trace_key
        for _, ranking_key, context_key in STAGES
        for trace_key in (ranking_key, context_key)
    }
    merged: dict[str, Any] = {trace_key: [] for trace_key in trace_keys}
    for trace in traces:
        for trace_key in trace_keys:
            merged[trace_key].extend(trace.get(trace_key) or [])
    return merged


def retrieval_scopes(raw_scope: dict[str, Any]) -> list[dict[str, Any]]:
    from experiments.runtime.retriever.temporal import normalize_temporal_scope

    normalized = normalize_temporal_scope(raw_scope)
    if normalized["type"] != "comparison":
        return [normalized]
    result = []
    for child in normalized["comparison_scopes"]:
        item = dict(child)
        item.pop("label", None)
        result.append(item)
    return result


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {}
    for stage_name, _, _ in STAGES:
        stage_metrics = {}
        for metric in (
            "recall_at_k", "precision_at_k", "ndcg_at_k",
            "fact_coverage", "evidence_precision",
            "ranked_provisions", "context_provisions", "returned_provisions",
            "context_chars", "context_words",
            "estimated_context_tokens",
        ):
            key = f"{stage_name}_{metric}"
            values = [float(row[key]) for row in rows]
            stage_metrics[metric] = round(sum(values) / len(values), 6) if values else 0.0
        metrics[stage_name] = stage_metrics
    return {
        "schema_version": 1,
        "cases": len(rows),
        "metric_unit": "0_to_1_except_counts_and_context_size",
        "configurations": metrics,
        "notes": {
            "full_provision": "Article expansion, complete-provision restoration and required-fact-aware reranking without sibling expansion.",
            "full_structured": "The preceding configuration followed by bounded reranked sibling expansion and final context assembly.",
            "ranking_metrics": "Recall@K, Precision@K and nDCG@K use only ranked legal locations; supplemental article context is excluded.",
            "context_metrics": "Fact Coverage, Evidence Precision and context size use the complete context actually constructed for downstream agents.",
            "fact_coverage": "A fact is covered only when a locked contiguous support span occurs in returned context.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete-provision and Full Structured Retriever "
            "configurations on the locked Retriever set."
        )
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from experiments.runtime.retriever.structured import ArticleExpansionRAG

    cases = read_jsonl(args.gold)
    if len(cases) != 100:
        raise ValueError(
            f"Expected exactly 100 locked Retriever cases, found {len(cases)}."
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero.")
        cases = cases[: args.limit]
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {row["benchmark_id"]: row for row in csv.DictReader(handle)}

    retriever = ArticleExpansionRAG(use_hybrid=True)
    output_rows = dict(existing)
    for index, case in enumerate(cases, 1):
        benchmark_id = case["benchmark_id"]
        if benchmark_id in output_rows:
            print(f"[{index}/{len(cases)}] {benchmark_id}: skip", flush=True)
            continue
        fact_descriptions = [fact["description"] for fact in case["gold_facts"]]
        scopes = retrieval_scopes(case["expected_temporal"])
        started = time.perf_counter()
        print("\n" + "=" * 88, flush=True)
        print(
            f"[{index}/{len(cases)}] {benchmark_id} | case={case['case_id']} | "
            f"difficulty={case['difficulty']}",
            flush=True,
        )
        print(f"Query: {case['query']}", flush=True)
        print(
            "Temporal: "
            + json.dumps(case["expected_temporal"], ensure_ascii=False),
            flush=True,
        )
        print(
            f"Gold: {len(fact_descriptions)} facts, {len(case['qrels'])} qrels; "
            f"retrieval branches={len(scopes)}",
            flush=True,
        )
        for fact_index, description in enumerate(fact_descriptions, 1):
            print(f"  F{fact_index}: {description}", flush=True)
        retriever.set_required_facts(fact_descriptions)
        traces = []
        for scope_index, scope in enumerate(scopes, 1):
            print(
                f"  -> Branch {scope_index}/{len(scopes)}: "
                f"{json.dumps(scope, ensure_ascii=False)}",
                flush=True,
            )
            branch_started = time.perf_counter()
            retriever.set_temporal_scope(scope)
            retriever.retrieve_chunks(case["query"], limit=5)
            traces.append(dict(retriever.last_retrieval_trace))
            trace = retriever.last_retrieval_trace
            print(
                "     Retrieved: "
                f"candidate pool={len(trace.get('stage_a_hybrid_candidate_pool') or [])}, "
                f"stage A={len(trace.get('stage_a_hybrid_reranked_seed_locations') or [])}, "
                f"stage B={len(trace.get('stage_b_top_locations_full_provisions') or [])}, "
                f"stage C={len(trace.get('stage_c_full_context_with_siblings') or [])}; "
                f"{time.perf_counter() - branch_started:.1f}s",
                flush=True,
            )
        merged = merge_traces(traces)

        row: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "case_id": case["case_id"],
            "dataset_group": case["dataset_group"],
            "difficulty": case["difficulty"],
            "query": case["query"],
            "gold_fact_count": len(case["gold_facts"]),
            "gold_qrel_count": len(case["qrels"]),
        }
        for stage_name, ranking_key, context_key in STAGES:
            result = evaluate_stage(
                stage_name,
                merged[ranking_key],
                merged[context_key],
                case,
            )
            for key, value in result.items():
                row[f"{stage_name}_{key}"] = value
        output_rows[benchmark_id] = row
        write_csv(args.output, list(output_rows.values()))
        print(
            f"[{index}/{len(cases)}] {benchmark_id}: "
            f"Full provision R={row['full_provision_recall_at_k']:.2f} "
            f"P={row['full_provision_precision_at_k']:.2f}; "
            f"Full structured R={row['full_structured_recall_at_k']:.2f} "
            f"P={row['full_structured_precision_at_k']:.2f}; "
            f"total={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    ordered = [output_rows[case["benchmark_id"]] for case in cases if case["benchmark_id"] in output_rows]
    write_csv(args.output, ordered)
    report = summary(ordered)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
