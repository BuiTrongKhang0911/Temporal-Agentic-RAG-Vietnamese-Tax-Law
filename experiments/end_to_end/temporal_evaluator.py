"""Temporal evidence audit and answer evaluation for end-to-end runs."""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import google.generativeai as genai

from .metrics import evidence_identity, evidence_text


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Temporal Judge response must be a JSON object")
    return value


def expected_intervals(
    scope: Mapping[str, Any],
    *,
    as_of: date,
) -> list[tuple[date, date]]:
    branches = list(scope.get("comparison_scopes") or [])
    if branches:
        intervals: list[tuple[date, date]] = []
        for branch in branches:
            intervals.extend(expected_intervals(branch, as_of=as_of))
        return intervals

    scope_type = str(scope.get("type") or "current")
    if scope_type == "current":
        return [(as_of, as_of + timedelta(days=1))]
    if scope_type == "exact_date":
        point = _parse_date(scope.get("date") or scope.get("from"))
        return [(point, point + timedelta(days=1))] if point else []
    if scope_type == "year":
        year = int(scope.get("year"))
        return [(date(year, 1, 1), date(year + 1, 1, 1))]
    if scope_type == "month":
        year = int(scope.get("year"))
        month = int(scope.get("month"))
        end = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        return [(date(year, month, 1), end)]
    start = _parse_date(scope.get("from") or scope.get("date"))
    end_value = _parse_date(scope.get("to") or scope.get("date"))
    if start and end_value:
        return [(start, end_value + timedelta(days=1))]
    return []


def _evidence_interval(evidence: Mapping[str, Any]) -> tuple[date, date]:
    metadata = evidence.get("metadata") or {}
    start = _parse_date(evidence.get("valid_from") or metadata.get("valid_from"))
    end = _parse_date(evidence.get("valid_to") or metadata.get("valid_to"))
    return start or date.min, end or date.max


def _overlaps(left: tuple[date, date], right: tuple[date, date]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


class TemporalEvaluator:
    """Audit evidence intervals and judge the temporal answer conclusion."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        as_of: date,
        call_delay: float = 0.0,
        retries: int = 2,
        timeout: float = 180.0,
    ) -> None:
        genai.configure(api_key=api_key)
        self.model_name = model
        self.as_of = as_of
        self.call_delay = max(0.0, call_delay)
        self.retries = retries
        self.timeout = timeout
        self.model = genai.GenerativeModel(
            model,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )

    def _call(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    request_options={"timeout": self.timeout},
                )
                result = _parse_json(response.text)
                if self.call_delay > 0:
                    time.sleep(self.call_delay)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Temporal Judge failed: {last_error}")

    def evaluate(
        self,
        *,
        case: Mapping[str, Any],
        answer: str,
        evidences: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        scope = case.get("expected_temporal") or {"type": "current"}
        intervals = expected_intervals(scope, as_of=self.as_of)
        evidence_audit = []
        for evidence in evidences:
            provision_key, version_id = evidence_identity(evidence)
            interval = _evidence_interval(evidence)
            evidence_audit.append(
                {
                    "provision_key": provision_key,
                    "version_id": version_id,
                    "valid_from": None if interval[0] == date.min else interval[0].isoformat(),
                    "valid_to": None if interval[1] == date.max else interval[1].isoformat(),
                    "matches_expected_scope": any(
                        _overlaps(interval, expected) for expected in intervals
                    ),
                    "text_excerpt": evidence_text(evidence)[:1200],
                }
            )

        prompt = f"""You are judging only temporal correctness in Vietnamese legal QA.
Use only the supplied expected scope, reference answer, actual answer, and
evidence validity metadata. Do not use memorized legal knowledge.

CURRENT AS-OF DATE: {self.as_of.isoformat()}
QUESTION: {case.get('query', '')}
EXPECTED TEMPORAL SCOPE:
{json.dumps(scope, ensure_ascii=False)}
REFERENCE ANSWER:
{case.get('reference_answer', '')}
ACTUAL ANSWER:
{answer}
EVIDENCE AUDIT:
{json.dumps(evidence_audit, ensure_ascii=False, indent=2)}

Score temporal_answer_accuracy from 0 to 10. A score of 10 requires the legal
version and conclusion to be correct for every requested period. A score of 5
means the general direction is correct but one important period is absent. A
score of 0 means the answer applies the wrong legal period or reverses the
old/new rule. Return JSON only:
{{"temporal_answer_accuracy":0,"uses_wrong_legal_version":false,
"missing_temporal_branch":false,"reason":"short reason"}}"""
        judged = self._call(prompt)
        score = max(
            0.0,
            min(1.0, float(judged.get("temporal_answer_accuracy") or 0) / 10),
        )
        branch_coverage = (
            sum(
                any(
                    _overlaps(_evidence_interval(evidence), expected)
                    for evidence in evidences
                )
                for expected in intervals
            )
            / len(intervals)
            if intervals
            else 0.0
        )
        evidence_precision = (
            sum(item["matches_expected_scope"] for item in evidence_audit)
            / len(evidence_audit)
            if evidence_audit
            else 0.0
        )
        return {
            "temporal_accuracy": round(score, 6),
            "temporal_evidence_precision": round(evidence_precision, 6),
            "temporal_branch_coverage": round(branch_coverage, 6),
            "temporal_reason": str(judged.get("reason") or ""),
            "temporal_judge_detail": judged,
            "temporal_evidence_audit": evidence_audit,
            "temporal_judge_model": self.model_name,
            "as_of": self.as_of.isoformat(),
        }
