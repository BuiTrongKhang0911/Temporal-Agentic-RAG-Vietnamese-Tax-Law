"""Run and evaluate the complete proposed Agentic RAG workflow."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(
    SUBMISSION_ROOT / ".env",
    override=False,
    encoding="utf-8-sig",
)

from experiments.end_to_end.llm_judge import LLMJudge  # noqa: E402
from experiments.end_to_end.call_pacing import (  # noqa: E402
    paced_gemini_calls,
    pause_after_call,
)
from experiments.end_to_end.metrics import (  # noqa: E402
    jsonable,
    load_keyed_csv,
    load_keyed_jsonl,
    read_jsonl,
    save_configuration_results,
)
from experiments.end_to_end.temporal_evaluator import TemporalEvaluator  # noqa: E402


DEFAULT_GOLD = HERE / "data" / "gold_set_300_e2e.jsonl"
DEFAULT_RESPONSES = HERE / "data" / "agent1_simulated_responses.jsonl"
DEFAULT_OUTPUT = HERE / "result" / "rerun" / "proposed_system"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("gemini", "qwen"), default="gemini")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--configuration",
        default="",
        help="Optional configuration name written to summary.json.",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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


def required_facts(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts = []
    for step in plan:
        facts.extend(list(step.get("fact_requirements") or []))
    return facts


def execution_error_steps(result: Mapping[str, Any]) -> list[int]:
    """Return workflow steps that failed because of an execution error."""

    failures: list[int] = []
    for index, step in enumerate(result.get("step_results") or [], 1):
        if not isinstance(step, Mapping):
            continue
        status = str(step.get("final_status") or "").strip().lower()
        error = str(step.get("error") or "").strip()
        if status == "error" or error:
            failures.append(int(step.get("step_number") or index))
    return failures


def main() -> int:
    args = parse_args()
    os.environ["SUBMISSION_RUNTIME_PROFILE"] = args.system

    # Import after selecting the frozen experimental runtime profile.
    from experiments.runtime.agents.agent1 import TaxLawClarifier
    from experiments.runtime.orchestrator import PlannerOrchestrator

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required for the independent LLM Judge"
        )
    cases = read_jsonl(args.gold)
    if len(cases) != 300:
        raise ValueError(f"Expected 300 E2E cases, found {len(cases)}")
    response_by_id = {
        str(row["case_id"]): str(row["simulated_user_response"])
        for row in read_jsonl(args.responses)
    }
    end = args.end or len(cases)
    selected = cases[max(0, args.start - 1):end]
    if args.limit:
        selected = selected[: args.limit]

    output_dir = args.output_dir / args.system
    configuration = args.configuration or f"proposed_{args.system}"
    outputs = (
        load_keyed_jsonl(output_dir / "outputs.jsonl")
        if args.resume
        else {}
    )
    scores: dict[str, dict[str, Any]] = (
        load_keyed_csv(output_dir / "per_case_results.csv")
        if args.resume
        else {}
    )
    orchestrator = PlannerOrchestrator()
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

    print(f"System : {args.system}")
    print(f"Cases  : {len(selected)}")
    print(f"Judge  : {args.judge_model}")
    print(f"Delay  : {args.call_delay:g}s")
    for position, case in enumerate(selected, 1):
        case_id = str(case["case_id"])
        output = outputs.get(case_id)
        if (
            args.resume
            and output
            and output.get("inference_status") == "ok"
        ):
            print(f"[{position}/{len(selected)}] {case_id}: inference cached")
        else:
            started = time.perf_counter()
            status = "error"
            error = ""
            normalized_query = str(case["query"])
            clarification: dict[str, Any] = {
                "status": "ready",
                "normalized_query": normalized_query,
            }
            result: dict[str, Any] = {}
            try:
                # Q001-Q100 are already direct benchmark queries. Later cases
                # preserve the original user-style input and run Agent 1.
                if int(case_id[1:]) > 100:
                    clarifier = TaxLawClarifier()
                    clarification = clarifier.process(str(case["query"]))
                    pause_after_call(args.call_delay)
                    if clarification.get("status") == "ask":
                        response = response_by_id.get(case_id)
                        if response:
                            clarification = clarifier.process(response)
                            pause_after_call(args.call_delay)
                    if clarification.get("status") == "ask":
                        status = "clarification_required"
                    else:
                        normalized_query = str(
                            clarification.get("normalized_query") or case["query"]
                        )

                if status != "clarification_required":
                    gemini_delay = args.call_delay if args.system == "gemini" else 0
                    with paced_gemini_calls(gemini_delay):
                        result = orchestrator.execute_workflow(
                            normalized_query,
                            verbose=not args.quiet,
                        )
                    failed_steps = execution_error_steps(result)
                    if failed_steps:
                        status = "error"
                        error = (
                            "Workflow returned execution errors in steps: "
                            + ", ".join(str(step) for step in failed_steps)
                        )
                    else:
                        status = "ok"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            evidence = jsonable(result.get("all_evidences", []))
            plan = jsonable(result.get("plan", []))
            output = {
                "schema_version": 1,
                "case_id": case_id,
                "input_query": case["query"],
                "normalized_query": normalized_query,
                "clarifier": jsonable(clarification),
                "tasks": plan,
                "required_facts": required_facts(plan),
                "verified_evidence": evidence,
                "calculation_results": jsonable(
                    getattr(orchestrator, "last_verified_calculations", [])
                ),
                "final_answer": str(result.get("final_answer") or ""),
                "inference_status": status,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "failed_steps": jsonable(result.get("failed_steps", [])),
                "error": error,
            }
            outputs[case_id] = output
            save_configuration_results(
                output_dir,
                outputs,
                scores,
                configuration=configuration,
            )

        if args.resume and scores.get(case_id, {}).get("judge_status") == "ok":
            print(f"[{position}/{len(selected)}] {case_id}: judge cached")
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
                evidence = list(output.get("verified_evidence") or [])
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
        scores[case_id] = row
        save_configuration_results(
            output_dir,
            outputs,
            scores,
            configuration=configuration,
        )
        print(f"[{position}/{len(selected)}] {case_id}: {row['judge_status']}")

    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
