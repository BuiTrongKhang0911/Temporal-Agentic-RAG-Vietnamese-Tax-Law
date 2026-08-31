"""Shared persistence and aggregation for end-to-end experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


METRICS = (
    "fact_coverage",
    "evidence_precision",
    "temporal_accuracy",
    "answer_correctness",
    "groundedness",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                )
                for field, value in row.items()
            }
            for row in rows
        )
    temporary.replace(path)


def load_keyed_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row.get("case_id") or ""): row
        for row in read_jsonl(path)
        if row.get("case_id")
    }


def load_keyed_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("case_id") or ""): dict(row)
            for row in csv.DictReader(handle)
            if row.get("case_id")
        }


def save_configuration_results(
    directory: Path,
    outputs: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    configuration: str,
) -> None:
    ordered_ids = sorted(outputs)
    output_rows = [dict(outputs[case_id]) for case_id in ordered_ids]
    score_rows = [dict(scores[case_id]) for case_id in ordered_ids if case_id in scores]
    write_jsonl(directory / "outputs.jsonl", output_rows)
    write_csv(directory / "per_case_results.csv", score_rows)

    summary_metrics: dict[str, float] = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in score_rows
            if row.get(metric) not in (None, "")
        ]
        summary_metrics[metric] = round(mean(values), 6) if values else 0.0
    successful_statuses = {"ok", "passed", "completed"}
    successful = sum(
        str(row.get("inference_status") or "").strip().lower()
        in successful_statuses
        for row in output_rows
    )
    judged = sum(row.get("judge_status") == "ok" for row in score_rows)
    summary = {
        "schema_version": 1,
        "configuration": configuration,
        "cases": len(output_rows),
        "successful_cases": successful,
        "judged_cases": judged,
        "metric_unit": "0_to_1",
        "metrics": summary_metrics,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evidence_text(evidence: Mapping[str, Any]) -> str:
    return str(
        evidence.get("noi_dung_goc")
        or evidence.get("original_content")
        or evidence.get("provision_content")
        or evidence.get("content")
        or ""
    )


def evidence_identity(evidence: Mapping[str, Any]) -> tuple[str, str]:
    metadata = evidence.get("metadata") or {}
    provision_key = str(
        evidence.get("provision_key") or metadata.get("provision_key") or ""
    )
    version_id = str(
        evidence.get("version_id") or metadata.get("version_id") or ""
    )
    if "|SUBCHUNK|" in provision_key:
        provision_key = provision_key.split("|SUBCHUNK|", 1)[0]
    version = evidence.get("version") or metadata.get("version")
    if not version_id and provision_key and version not in (None, ""):
        version_id = f"{provision_key}|V{version}"
    return provision_key, version_id
