"""Source-validated LLM Judge for end-to-end answer quality metrics."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping, Sequence

import google.generativeai as genai

from .metrics import evidence_identity, evidence_text


def _collapse(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


class LLMJudge:
    """Judge evidence support, answer correctness, and groundedness."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        call_delay: float = 0.0,
        retries: int = 2,
        timeout: float = 180.0,
    ) -> None:
        genai.configure(api_key=api_key)
        self.model_name = model
        self.call_delay = max(0.0, call_delay)
        self.retries = retries
        self.timeout = timeout
        self.model = genai.GenerativeModel(
            model,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 8192,
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
        raise RuntimeError(f"LLM Judge failed: {last_error}")

    def evaluate(
        self,
        *,
        case: Mapping[str, Any],
        answer: str,
        evidences: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        facts = list(case.get("gold_facts") or [])
        evidence_rows = []
        for index, evidence in enumerate(evidences, 1):
            provision_key, version_id = evidence_identity(evidence)
            evidence_rows.append(
                {
                    "evidence_id": f"E{index}",
                    "provision_key": provision_key,
                    "version_id": version_id,
                    "text": evidence_text(evidence),
                }
            )

        prompt = f"""You are evaluating a Vietnamese legal QA system.
Use only the supplied reference answer, locked facts, and returned evidence.
Do not use memorized legal knowledge.

QUESTION:
{case.get('query', '')}

REFERENCE ANSWER:
{case.get('reference_answer', '')}

LOCKED FACTS:
{json.dumps(facts, ensure_ascii=False, indent=2)}

RETURNED EVIDENCE:
{json.dumps(evidence_rows, ensure_ascii=False, indent=2)}

ACTUAL ANSWER:
{answer}

Tasks:
1. Map an evidence item to a fact only when its text directly supports that fact.
2. For every mapping, copy one continuous exact support span from that evidence.
3. Split the actual answer into atomic factual or legal claims. Mark each claim
   supported only when one or more returned evidence items directly support it.
4. Score answer_correctness from 0 to 10 against the reference conclusion and
   locked facts. Do not award correctness for unsupported outside knowledge.

Return JSON only:
{{
  "evidence_fact_support": [
    {{"evidence_id":"E1","fact_id":"F1","supported":true,"support_span":"exact text"}}
  ],
  "answer_claims": [
    {{"claim":"atomic claim","supported":true,"evidence_ids":["E1"]}}
  ],
  "answer_correctness": 0,
  "reason": "short reason"
}}"""
        judged = self._call(prompt)
        evidence_by_id = {
            row["evidence_id"]: row for row in evidence_rows
        }
        fact_ids = {str(fact.get("fact_id") or "") for fact in facts}

        supported_by_evidence: dict[str, set[str]] = {
            evidence_id: set() for evidence_id in evidence_by_id
        }

        # Reviewed qrels receive deterministic credit.
        for index, evidence in enumerate(evidences, 1):
            evidence_id = f"E{index}"
            provision_key, version_id = evidence_identity(evidence)
            for qrel in case.get("qrels") or []:
                qrel_version = str(qrel.get("version_id") or "")
                exact = (
                    version_id == qrel_version
                    if version_id and qrel_version
                    else bool(
                        provision_key
                        and provision_key
                        == str(qrel.get("provision_key") or "")
                    )
                )
                if exact:
                    supported_by_evidence[evidence_id].update(
                        str(item) for item in qrel.get("supports_fact_ids") or []
                    )

        accepted_mappings = []
        for mapping in judged.get("evidence_fact_support") or []:
            evidence_id = str(mapping.get("evidence_id") or "")
            fact_id = str(mapping.get("fact_id") or "")
            span = _collapse(mapping.get("support_span"))
            row = evidence_by_id.get(evidence_id)
            valid = bool(
                mapping.get("supported")
                and row
                and fact_id in fact_ids
                and span
                and span in _collapse(row["text"])
            )
            if valid:
                supported_by_evidence[evidence_id].add(fact_id)
                accepted_mappings.append(
                    {
                        "evidence_id": evidence_id,
                        "fact_id": fact_id,
                        "support_span": span,
                    }
                )

        covered = set().union(*supported_by_evidence.values()) if evidences else set()
        useful = sum(bool(items) for items in supported_by_evidence.values())
        fact_coverage = len(covered & fact_ids) / len(fact_ids) if fact_ids else 0.0
        evidence_precision = useful / len(evidences) if evidences else 0.0

        claims = list(judged.get("answer_claims") or [])
        supported_claims = 0
        normalized_claims = []
        for claim in claims:
            cited = [
                str(item)
                for item in claim.get("evidence_ids") or []
                if str(item) in evidence_by_id
            ]
            supported = bool(claim.get("supported") and cited)
            supported_claims += int(supported)
            normalized_claims.append(
                {
                    "claim": str(claim.get("claim") or ""),
                    "supported": supported,
                    "evidence_ids": cited,
                }
            )
        groundedness = supported_claims / len(claims) if claims else 0.0
        correctness = max(
            0.0,
            min(1.0, float(judged.get("answer_correctness") or 0) / 10),
        )
        return {
            "fact_coverage": round(fact_coverage, 6),
            "evidence_precision": round(evidence_precision, 6),
            "answer_correctness": round(correctness, 6),
            "groundedness": round(groundedness, 6),
            "accepted_evidence_fact_support": accepted_mappings,
            "answer_claims": normalized_claims,
            "judge_reason": str(judged.get("reason") or ""),
            "judge_model": self.model_name,
        }
