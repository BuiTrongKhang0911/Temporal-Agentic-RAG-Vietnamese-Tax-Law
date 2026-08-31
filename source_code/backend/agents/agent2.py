"""Two-stage Gemini Agent 2 for task planning and fact decomposition."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import google.generativeai as genai

from .agent2_support import (
    AGENT2_BATCH_FACT_PROMPT,
    AGENT2_FUNCTION_CHOICE_PROMPT,
    FACT_TYPES,
    MAX_FACTS_PER_TASK,
    MAX_TASKS,
    Agent2TaskNormalizer,
)


BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "granularity": {
            "type": "string",
            "enum": ["year", "month", "day", "current"],
        },
        "value": {"type": "string"},
    },
    "required": ["granularity", "value"],
}

COMPARISON_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["exact_date", "month", "year", "date_range"],
        },
        "label": {"type": "string", "nullable": True},
        "date": {"type": "string", "nullable": True},
        "month": {"type": "integer", "nullable": True},
        "year": {"type": "integer", "nullable": True},
        "from": {**BOUNDARY_SCHEMA, "nullable": True},
        "to": {**BOUNDARY_SCHEMA, "nullable": True},
    },
    "required": ["type"],
}

TEMPORAL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "current",
                "exact_date",
                "month",
                "year",
                "date_range",
                "comparison",
            ],
        },
        "date": {"type": "string", "nullable": True},
        "month": {"type": "integer", "nullable": True},
        "year": {"type": "integer", "nullable": True},
        "from": {**BOUNDARY_SCHEMA, "nullable": True},
        "to": {**BOUNDARY_SCHEMA, "nullable": True},
        "comparison_scopes": {
            "type": "array",
            "items": COMPARISON_SCOPE_SCHEMA,
        },
    },
    "required": ["type"],
}

STAGE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "query": {"type": "string"},
                    "temporal": TEMPORAL_SCHEMA,
                },
                "required": ["task", "query", "temporal"],
            },
        }
    },
    "required": ["tasks"],
}

STAGE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "task_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": FACT_TYPES},
                                "description": {"type": "string"},
                            },
                            "required": ["type", "description"],
                        },
                    },
                },
                "required": ["task_id", "facts"],
            },
        }
    },
    "required": ["task_facts"],
}


class GeminiTwoStagePlannerLLM:
    """Run Gemini 2A once, then Gemini 2B once for all locked tasks."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int = 2048,
        fact_max_output_tokens: int = 4096,
    ) -> None:
        genai.configure(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.fact_max_output_tokens = fact_max_output_tokens
        self.llm = genai.GenerativeModel(model)
        self.last_stage_a: dict[str, Any] = {}
        self.last_stage_b: dict[str, Any] = {}

    def _generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        response = self.llm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = str(response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty structured response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {text[:1000]}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError("Gemini structured response must be a JSON object")
        return data

    @staticmethod
    def _stage_a_prompt(original_query: str) -> str:
        instructions = AGENT2_FUNCTION_CHOICE_PROMPT.replace(
            "Chỉ gọi submit_task_plan đúng một lần. Không trả văn bản ngoài JSON.",
            "Chỉ trả JSON object có field tasks. Không giải thích ngoài JSON.",
        ).replace(
            "Chỉ gọi submit_task_plan. Không tạo facts và không trả lời câu hỏi.",
            "Chỉ tạo field tasks. Không tạo facts và không trả lời câu hỏi.",
        )
        return f"{instructions}\n\nCâu hỏi: {original_query}"

    @staticmethod
    def _stage_b_prompt(
        original_query: str,
        tasks: list[dict[str, Any]],
    ) -> str:
        instructions = AGENT2_BATCH_FACT_PROMPT.replace(
            "Chỉ gọi submit_facts. Không trả văn bản giải thích.",
            "Chỉ tạo facts theo schema. Không trả văn bản giải thích.",
        ).replace(
            "Chỉ gọi submit_task_facts đúng một lần. Không trả văn bản ngoài JSON.",
            "Chỉ trả JSON object có field task_facts. Không giải thích ngoài JSON.",
        )
        return (
            f"{instructions}\n\n"
            "INPUT JSON:\n"
            + json.dumps(
                {"original_query": original_query, "tasks": tasks},
                ensure_ascii=False,
            )
        )

    @staticmethod
    def _normalize_tasks(
        raw_tasks: Any,
        *,
        original_query: str,
    ) -> list[dict[str, Any]]:
        return Agent2TaskNormalizer._normalize_tasks(
            raw_tasks,
            original_query=original_query,
        )

    def _generate_facts(
        self,
        *,
        original_query: str,
        tasks: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, str]]]:
        prompt = self._stage_b_prompt(
            original_query,
            tasks,
        )
        data = self._generate_json(
            prompt=prompt,
            response_schema=STAGE_B_SCHEMA,
            max_output_tokens=self.fact_max_output_tokens,
        )
        raw_groups = data.get("task_facts")
        if not isinstance(raw_groups, list):
            raise RuntimeError("Gemini Agent 2B returned no task_facts")

        expected_ids = [task["task_id"] for task in tasks]
        groups: dict[str, Any] = {}
        for group in raw_groups:
            if not isinstance(group, dict):
                raise RuntimeError("Gemini Agent 2B returned an invalid group")
            task_id = str(group.get("task_id") or "").strip()
            if not task_id or task_id in groups:
                raise RuntimeError(
                    f"Gemini Agent 2B duplicated/omitted task_id: {task_id!r}"
                )
            groups[task_id] = group.get("facts")

        if set(groups) != set(expected_ids):
            raise RuntimeError(
                "Gemini Agent 2B task IDs do not match Agent 2A: "
                f"expected={expected_ids}, actual={sorted(groups)}"
            )

        facts_by_task: dict[str, list[dict[str, str]]] = {}
        for task_id in expected_ids:
            raw_facts = groups[task_id]
            if not isinstance(raw_facts, list) or not raw_facts:
                raise RuntimeError(f"Gemini Agent 2B returned no facts for {task_id}")
            facts: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for raw_fact in raw_facts:
                if not isinstance(raw_fact, dict):
                    continue
                fact_type = str(raw_fact.get("type") or "").strip()
                description = str(raw_fact.get("description") or "").strip()
                if fact_type not in FACT_TYPES or not description:
                    continue
                key = (fact_type, re.sub(r"\s+", " ", description).casefold())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "fact_id": f"{task_id}_F{len(facts) + 1}",
                        "task_id": task_id,
                        "type": fact_type,
                        "description": description,
                    }
                )
                if len(facts) >= MAX_FACTS_PER_TASK:
                    break
            if not facts:
                raise RuntimeError(
                    f"Gemini Agent 2B returned no valid facts for {task_id}"
                )
            facts_by_task[task_id] = facts

        self.last_stage_b = {
            "model": self.model,
            "backend": "gemini",
            "batch": True,
            "task_facts": facts_by_task,
        }
        return facts_by_task

    def _complete_from_tasks(
        self,
        *,
        original_query: str,
        tasks: list[dict[str, Any]],
        architecture: str,
    ) -> str:
        facts_by_task = self._generate_facts(
            original_query=original_query,
            tasks=tasks,
        )
        plan = [
            {
                **task,
                "fact_requirements": facts_by_task[task["task_id"]],
                "_selected_temporal_function": "submit_task_plan",
                "_gemini_model": self.model,
                "_agent2_architecture": architecture,
            }
            for task in tasks
        ]
        return json.dumps(plan, ensure_ascii=False)

    def complete_query(self, original_query: str) -> str:
        stage_a_prompt = self._stage_a_prompt(original_query)
        data = self._generate_json(
            prompt=stage_a_prompt,
            response_schema=STAGE_A_SCHEMA,
            max_output_tokens=self.max_output_tokens,
        )
        tasks = self._normalize_tasks(
            data.get("tasks"),
            original_query=original_query,
        )
        self.last_stage_a = {
            "model": self.model,
            "backend": "gemini",
            "tasks": tasks,
        }
        return self._complete_from_tasks(
            original_query=original_query,
            tasks=tasks,
            architecture="2A_per_task_temporal__2B_batch_facts",
        )

    def complete_locked_plan(
        self,
        original_query: str,
        locked_plan: list[dict[str, Any]],
    ) -> str:
        if not locked_plan:
            raise ValueError("Locked Agent 2A plan must not be empty")
        tasks = self._normalize_tasks(
            [
                {
                    "task": step.get("task"),
                    "query": step.get("query"),
                    "temporal": step.get("temporal"),
                }
                for step in locked_plan
            ],
            original_query=original_query,
        )
        return self._complete_from_tasks(
            original_query=original_query,
            tasks=tasks,
            architecture="2A_frozen_task_temporal__2B_batch_facts",
        )
