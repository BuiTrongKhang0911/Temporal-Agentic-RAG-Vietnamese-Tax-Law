"""Gemini Agent 3 retrieval and batched per-task Agent 4 verification."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
import os
import re
import time
from typing import Any

from ._search_runtime import (
    LegalSearchTool,
    StructuredChunk,
    StructuredSearchMemory,
    _build_evidences,
    fact_descriptions,
    normalize_fact_requirements,
)


AGENT3_QUERY_PROMPT = """Bạn là Agent 3, chuyên viết query để gọi công cụ tra cứu pháp luật thuế.

Bạn nhận câu hỏi/task gốc, temporal đã bị Python khóa, các fact còn thiếu và có thể có
retry_query từ Agent 4. Hãy gọi search_legal_chunks đúng một lần:
- query ngắn, giữ đúng loại thuế, đối tượng, hoạt động/thu nhập và thuật ngữ pháp lý;
- tập trung vào các fact còn thiếu;
- khi retry, xem gợi ý của Agent 4 là MÔ TẢ DỮ KIỆN CÒN THIẾU, không phải đáp án hay
  căn cứ đã được xác nhận; tự viết lại thành query độc lập và không lặp nguyên query cũ;
- không sao chép mã văn bản, Điều/Khoản hoặc giá trị pháp lý do retry_query tự nêu nếu
  chúng không xuất hiện trong câu hỏi/task hoặc fact đã khóa;
- không thêm ngày hoặc đổi phạm vi thời gian vì Python đã áp dụng filter;
- không trả lời câu hỏi và không tự điền giá trị pháp lý."""


AGENT3_SUMMARY_PROMPT = """Bạn là Agent 3 trong hệ thống tra cứu pháp luật thuế.

Python đã thực thi query do bạn viết. Hãy đọc toàn bộ provision đã tích lũy và tạo
bản trả lời tra cứu tạm thời cho task và các fact requirements:
- Không dùng kiến thức nhớ sẵn, không tự thêm giá trị pháp lý.
- Mỗi kết luận phải ghi Candidate ID hỗ trợ (C1, C2, ...).
- Nếu nhắc mã văn bản, phải chép đúng mã hiển thị trong candidate, không tự nhớ hoặc đổi QH/QH khóa.
- Nêu rõ fact nào chưa có căn cứ trong các chunk.
- Không gọi tìm kiếm trong bước tổng hợp này.

Chỉ gọi submit_search_summary."""


AGENT4_FACT_PROMPT = """Bạn là Agent 4, kiểm chứng TOÀN BỘ fact requirements của một task trong một lần.

Đầu vào gồm câu hỏi, danh sách facts, câu trả lời tạm thời của Agent 3 và toàn bộ provision ứng viên.
Với MỖI fact_id đầu vào, phải trả đúng một phần tử evaluation có cùng fact_id. Không bỏ sót, không tạo fact mới.
Đánh giá từng fact độc lập bằng evidence trực tiếp:
- covered: evidence đủ để xác định fact.
- missing_evidence: evidence chưa đủ hoặc không đúng trọng tâm.
- not_applicable: evidence trực tiếp chứng minh fact không áp dụng cho tình huống.
- support_span phải là một đoạn liên tục, chép NGUYÊN VĂN TỪNG KÝ TỰ từ đúng chunk.
- Không được diễn giải, rút gọn, thêm dấu ba chấm, đổi dấu câu hoặc ghép hai đoạn rời nhau.
- candidate_id phải đúng một bí danh C1, C2, ... xuất hiện trong danh sách provision ứng viên.
- Không tự chép hoặc suy đoán ID Qdrant dạng số dài.
- Chỉ giữ tối đa 3 chunk trực tiếp cần cho mỗi fact.
- Một chunk có thể hỗ trợ nhiều facts, nhưng mỗi fact phải có support_span riêng.
- Không dùng kiến thức ngoài evidence và không yêu cầu tìm lại thời gian/version.
- Nếu missing_evidence, retry_query phải là một CỤM TRA CỨU ĐỘC LẬP mô tả đúng dữ kiện
  còn thiếu, giữ nguyên loại thuế, đối tượng, hoạt động/thu nhập và thuật ngữ trong fact.
- Chỉ tìm phần còn thiếu: tax_rate -> thuế suất/tỷ lệ; tax_base -> căn cứ tính thuế;
  threshold_or_exemption -> ngưỡng/điều kiện; deadline -> thời hạn; procedure -> thủ tục.
- Không viết "tìm thêm", "evidence chưa đủ", không nhắc Candidate ID và không lặp lại
  toàn bộ câu hỏi nếu chỉ thiếu một khía cạnh.
- Tuyệt đối không đoán mã văn bản, Điều/Khoản, giá trị, ngưỡng, đáp án hoặc đưa ra các
  phương án kiểu "A hay B" khi các chi tiết đó không có trong câu hỏi/fact.
- Không thêm ngày vào retry_query vì temporal đã được Python khóa.
- Nếu covered hoặc not_applicable, retry_query phải là chuỗi rỗng.

Chỉ gọi submit_fact_evaluations."""

AGENT4_SPAN_REPAIR_PROMPT = """Bạn là Agent 4 đang sửa lỗi trích dẫn cho một hoặc nhiều facts, không đánh giá lại nội dung.

Python đã xác định các evaluation trước có ý định dùng evidence nhưng support_span không tồn tại
nguyên văn trong candidate chunk. Hãy gọi submit_support_span_repairs và trả đúng fact_id cần sửa:
- Chỉ dùng candidate_id dạng C1, C2, ... có trong danh sách candidate.
- Không tự chép hoặc suy đoán ID Qdrant dạng số dài.
- Chép một đoạn LIÊN TỤC, NGUYÊN VĂN TỪNG KÝ TỰ từ nội dung chunk.
- Đoạn trích phải trực tiếp hỗ trợ fact.
- Không diễn giải, không thêm dấu ba chấm, không ghép nhiều đoạn rời nhau.
- Nếu một câu ngắn đã đủ thì chỉ chép câu đó.

Không gọi tìm kiếm và không trả văn bản ngoài function."""


def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }



def _gemini_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema keywords unsupported by the legacy Gemini SDK."""

    unsupported = {
        "additionalProperties",
        "default",
        "maxItems",
        "minItems",
        "pattern",
    }
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, dict):
            cleaned[key] = _gemini_response_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _gemini_response_schema(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


class GeminiToolClient:
    """Gemini client for structured Agent 4 responses."""

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GOOGLE_API_KEY cho Agent 4 Gemini")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = os.getenv(
            "AGENT4_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
        )
        self.model = genai.GenerativeModel(self.model_name)
        self.api_attempts = max(
            1,
            int(os.getenv("AGENT4_GEMINI_ATTEMPTS", "2")),
        )
        self.max_output = int(os.getenv("AGENT4_GEMINI_MAX_OUTPUT", "4096"))
        print(f"[Agent 4 Gemini sequential] Model: {self.model_name}")

    def call_tool(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        function = tool.get("function") or {}
        schema = _gemini_response_schema(function.get("parameters") or {})
        prompt = (
            f"{system_prompt}\n\n{user_prompt}\n\n"
            f"Hãy thực hiện function {tool_name!r}. Chỉ xuất JSON object là "
            "phần arguments của function, không bọc trong tool_calls và không "
            "thêm giải thích."
        )

        last_error: Exception | None = None
        for attempt in range(1, self.api_attempts + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=schema,
                        max_output_tokens=self.max_output,
                    ),
                )
                text = str(response.text or "").strip()
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Gemini không trả JSON object cho {tool_name!r}"
                    )
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.api_attempts:
                    time.sleep(
                        float(os.getenv("AGENT4_GEMINI_RETRY_SLEEP", "2"))
                    )
        raise RuntimeError(f"Gemini Agent 4 lỗi: {last_error}") from last_error


def _candidate_map(chunks: list[StructuredChunk]) -> dict[str, StructuredChunk]:
    return {
        f"C{index}": chunk
        for index, chunk in enumerate(chunks, start=1)
    }


_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:\s*%)?", re.UNICODE)
_NEGATION_RE = re.compile(
    r"\b(?:không|chưa|chẳng|trừ|ngoại\s+trừ)\b",
    re.IGNORECASE | re.UNICODE,
)
_FUZZY_SPAN_THRESHOLD = 0.95
_LEGAL_ANCHOR_RE = re.compile(
    r"(?:\bđiều\s+\d+[a-z]?\b|\b(?:luật|nghị\s+định|thông\s+tư)\s+số\b|"
    r"\b\d{1,4}/\d{4}/[A-ZĐ-]+\b)",
    re.IGNORECASE | re.UNICODE,
)
_CANDIDATE_ID_RE = re.compile(r"\bC[1-9][0-9]*\b", re.IGNORECASE)


def _safe_retry_hint(
    retry_query: str,
    *,
    query: str,
    fact: dict[str, Any],
) -> str:
    """Reject hallucinated retrieval anchors without inventing legal semantics."""
    hint = re.sub(r"\s+", " ", str(retry_query or "")).strip()
    description = re.sub(
        r"\s+",
        " ",
        str(fact.get("description") or ""),
    ).strip()
    if not hint or len(hint.split()) < 3 or _CANDIDATE_ID_RE.search(hint):
        return description

    locked_text = f"{query}\n{description}".casefold()
    for match in _LEGAL_ANCHOR_RE.finditer(hint):
        if match.group(0).casefold() not in locked_text:
            return description

    # Agent 3 needs a compact retrieval hint, not an essay from Agent 4.
    return hint[:360].rstrip()


def _normalize_whitespace_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining indexes into the original text."""
    normalized: list[str] = []
    source_indexes: list[int] = []
    pending_space_index: int | None = None

    for index, character in enumerate(text):
        if character.isspace():
            if normalized:
                pending_space_index = index
            continue
        if pending_space_index is not None and normalized:
            normalized.append(" ")
            source_indexes.append(pending_space_index)
        pending_space_index = None
        normalized.append(character)
        source_indexes.append(index)

    return "".join(normalized), source_indexes


def _protected_signature(text: str) -> tuple[Counter[str], Counter[str]]:
    numbers = Counter(
        re.sub(r"\s+", "", match.group(0)).casefold()
        for match in _NUMBER_RE.finditer(text)
    )
    negations = Counter(
        re.sub(r"\s+", " ", match.group(0)).casefold()
        for match in _NEGATION_RE.finditer(text)
    )
    return numbers, negations


def _source_excerpt(
    normalized_start: int,
    normalized_end: int,
    source_indexes: list[int],
    source: str,
) -> str:
    start = source_indexes[normalized_start]
    end = source_indexes[normalized_end - 1] + 1
    return source[start:end]


def _match_support_span(span: str, source: str) -> dict[str, Any] | None:
    """Find a safe source-backed equivalent of a model-produced support span."""
    if not span or not source:
        return None

    exact_start = source.find(span)
    if exact_start >= 0:
        return {
            "support_span": source[exact_start : exact_start + len(span)],
            "match_type": "exact",
            "match_score": 1.0,
        }

    normalized_source, source_indexes = _normalize_whitespace_with_map(source)
    normalized_span, _ = _normalize_whitespace_with_map(span)
    if not normalized_source or not normalized_span:
        return None

    source_folded = normalized_source.casefold()
    span_folded = normalized_span.casefold()
    normalized_start = source_folded.find(span_folded)
    if normalized_start >= 0:
        normalized_end = normalized_start + len(normalized_span)
        return {
            "support_span": _source_excerpt(
                normalized_start,
                normalized_end,
                source_indexes,
                source,
            ),
            "match_type": "whitespace_normalized",
            "match_score": 1.0,
        }

    span_words = list(re.finditer(r"\S+", normalized_span))
    source_words = list(re.finditer(r"\S+", normalized_source))
    if len(normalized_span) < 32 or len(span_words) < 6 or not source_words:
        return None

    expected_words = len(span_words)
    tolerance = max(2, min(8, round(expected_words * 0.15)))
    min_words = max(1, expected_words - tolerance)
    max_words = expected_words + tolerance
    protected_span = _protected_signature(normalized_span)
    best: dict[str, Any] | None = None

    for start_word in range(len(source_words)):
        for word_count in range(min_words, max_words + 1):
            end_word = start_word + word_count
            if end_word > len(source_words):
                break
            candidate_start = source_words[start_word].start()
            candidate_end = source_words[end_word - 1].end()
            candidate = normalized_source[candidate_start:candidate_end]
            if _protected_signature(candidate) != protected_span:
                continue
            score = SequenceMatcher(
                None,
                span_folded,
                candidate.casefold(),
                autojunk=False,
            ).ratio()
            if best is None or score > best["match_score"]:
                best = {
                    "support_span": _source_excerpt(
                        candidate_start,
                        candidate_end,
                        source_indexes,
                        source,
                    ),
                    "match_type": "fuzzy",
                    "match_score": score,
                }

    if best is not None and best["match_score"] >= _FUZZY_SPAN_THRESHOLD:
        return best
    return None


def _format_candidates(chunks: list[StructuredChunk]) -> str:
    blocks: list[str] = []
    for candidate_id, chunk in _candidate_map(chunks).items():
        blocks.append(
            "\n".join(
                [
                    f"[Candidate ID: {candidate_id}]",
                    f"Văn bản: {chunk.document_id}",
                    f"Vị trí: {chunk.dieu_khoan}",
                    f"Hiệu lực: [{chunk.valid_from or 'không rõ'}, {chunk.valid_to or 'chưa kết thúc'})",
                    chunk.noi_dung_goc,
                ]
            )
        )
    return "\n\n".join(blocks)



class GeminiSequentialSearcher:
    """Use Gemini for Agent 3 while retaining the locked Python retriever."""

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GOOGLE_API_KEY cho Agent 3 Gemini")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = os.getenv(
            "AGENT3_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
        )
        self.model = genai.GenerativeModel(self.model_name)
        self.legal_tool = LegalSearchTool()
        self.api_attempts = max(1, int(os.getenv("AGENT3_GEMINI_ATTEMPTS", "2")))
        print(f"[Agent 3 Gemini sequential] Model: {self.model_name}")

    def _call_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.api_attempts + 1):
            try:
                response = self.model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}\n\nChỉ trả một JSON object hợp lệ.",
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                text = (response.text or "").strip()
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini Agent 3 không trả JSON object")
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.api_attempts:
                    time.sleep(float(os.getenv("AGENT3_GEMINI_RETRY_SLEEP", "2")))
        raise RuntimeError(f"Gemini Agent 3 lỗi: {last_error}") from last_error

    def write_query(
        self,
        *,
        original_query: str,
        temporal_scope: Any,
        fact_requirements: list[dict[str, Any]],
        retry_queries: list[str] | None = None,
        previous_queries: list[str] | None = None,
    ) -> str:
        prompt = (
            f"CÂU HỎI/TASK GỐC:\n{original_query}\n\n"
            f"TEMPORAL ĐÃ KHÓA:\n{json.dumps(temporal_scope, ensure_ascii=False)}\n\n"
            f"FACT CÒN THIẾU:\n{json.dumps(fact_requirements, ensure_ascii=False)}\n\n"
            f"GỢI Ý TÌM LẠI TỪ AGENT 4:\n"
            f"{json.dumps(retry_queries or [], ensure_ascii=False)}\n\n"
            f"QUERY ĐÃ DÙNG:\n"
            f"{json.dumps(previous_queries or [], ensure_ascii=False)}\n\n"
            'Trả JSON dạng {"query":"một semantic query ngắn"}.'
        )
        result = self._call_json(AGENT3_QUERY_PROMPT, prompt)
        query = str(result.get("query") or "").strip()
        if not query:
            raise ValueError("Gemini Agent 3 không tạo query tra cứu")
        return query

    def retrieve(
        self,
        *,
        search_query: str,
        step_number: int,
        temporal_scope: Any,
        fact_requirements: list[dict[str, Any]],
    ) -> list[StructuredChunk]:
        self.legal_tool.set_temporal_scope(temporal_scope)
        self.legal_tool.begin_turn(
            step_number,
            fact_requirements=fact_requirements,
            required_facts=fact_descriptions(fact_requirements),
        )
        self.legal_tool.search(search_query)
        return list(self.legal_tool.current_chunks.values())

    def summarize(
        self,
        *,
        original_query: str,
        search_query: str,
        chunks: list[StructuredChunk],
        fact_requirements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            f"CÂU HỎI/TASK GỐC:\n{original_query}\n\n"
            f"QUERY TOOL VỪA DÙNG:\n{search_query}\n\n"
            f"FACT REQUIREMENTS:\n{json.dumps(fact_requirements, ensure_ascii=False)}\n\n"
            f"PROVISION ỨNG VIÊN:\n{_format_candidates(chunks)}\n\n"
            "Trả JSON gồm summary, cited_chunk_ids (các mã C1, C2, ...) và "
            "missing_fact_ids. Không dùng kiến thức ngoài provision."
        )
        gemini_summary_prompt = AGENT3_SUMMARY_PROMPT.replace(
            "Chỉ gọi submit_search_summary.",
            "Chỉ trả một JSON object theo đúng cấu trúc được yêu cầu.",
        )
        result = self._call_json(gemini_summary_prompt, prompt)
        candidates = _candidate_map(chunks)
        cited_candidate_ids = [
            str(value)
            for value in result.get("cited_chunk_ids") or []
            if str(value) in candidates
        ]
        result["summary"] = str(result.get("summary") or "").strip()
        result["missing_fact_ids"] = [
            str(value) for value in result.get("missing_fact_ids") or []
        ]
        result["cited_candidate_ids"] = cited_candidate_ids
        result["cited_chunk_ids"] = [
            candidates[candidate_id].chunk_id
            for candidate_id in cited_candidate_ids
        ]
        return result


class GeminiSequentialFactEvaluator:
    """Evaluate every pending fact in one Gemini call over shared candidates."""

    def __init__(self, client: GeminiToolClient) -> None:
        self.client = client

    @staticmethod
    def _evidence_schema(*, require_one: bool = False) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["candidate_id", "support_span"],
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "pattern": "^C[1-9][0-9]*$",
                    },
                    "support_span": {"type": "string"},
                },
            },
        }
        if require_one:
            schema["minItems"] = 1
        return schema

    def _fact_batch_tool(self, fact_ids: list[str]) -> dict[str, Any]:
        return _function_tool(
            "submit_fact_evaluations",
            "Evaluate all required legal facts against one shared candidate set.",
            {
                "type": "object",
                "required": ["evaluations"],
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "minItems": len(fact_ids),
                        "maxItems": len(fact_ids),
                        "items": {
                            "type": "object",
                            "required": [
                                "fact_id",
                                "status",
                                "reason",
                                "evidence",
                                "retry_query",
                            ],
                            "properties": {
                                "fact_id": {
                                    "type": "string",
                                    "enum": fact_ids,
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "covered",
                                        "missing_evidence",
                                        "not_applicable",
                                    ],
                                },
                                "reason": {"type": "string"},
                                "retry_query": {"type": "string"},
                                "evidence": self._evidence_schema(),
                            },
                        },
                    }
                },
            },
        )

    def _span_repair_batch_tool(self, fact_ids: list[str]) -> dict[str, Any]:
        return _function_tool(
            "submit_support_span_repairs",
            "Repair invalid support spans for all listed fact IDs in one call.",
            {
                "type": "object",
                "required": ["repairs"],
                "properties": {
                    "repairs": {
                        "type": "array",
                        "minItems": len(fact_ids),
                        "maxItems": len(fact_ids),
                        "items": {
                            "type": "object",
                            "required": ["fact_id", "reason", "evidence"],
                            "properties": {
                                "fact_id": {
                                    "type": "string",
                                    "enum": fact_ids,
                                },
                                "reason": {"type": "string"},
                                "evidence": self._evidence_schema(
                                    require_one=True
                                ),
                            },
                        },
                    }
                },
            },
        )

    @staticmethod
    def _validate_evidence(
        evidence: list[dict[str, Any]],
        chunks: list[StructuredChunk],
    ) -> list[dict[str, Any]]:
        candidates = _candidate_map(chunks)
        valid_evidence: list[dict[str, Any]] = []
        for item in evidence:
            candidate_id = str(item.get("candidate_id") or "")
            chunk = candidates.get(candidate_id)
            span = str(item.get("support_span") or "").strip()
            match = (
                _match_support_span(span, chunk.noi_dung_goc)
                if chunk is not None
                else None
            )
            if chunk is not None and match is not None:
                valid_evidence.append(
                    {
                        "candidate_id": candidate_id,
                        "chunk_id": chunk.chunk_id,
                        "support_span": match["support_span"],
                        "span_match_type": match["match_type"],
                        "span_match_score": match["match_score"],
                    }
                )
        return valid_evidence

    def evaluate_facts(
        self,
        *,
        query: str,
        facts: list[dict[str, Any]],
        agent3_summary: str,
        chunks: list[StructuredChunk],
    ) -> dict[str, dict[str, Any]]:
        if not facts:
            return {}

        facts_by_id = {
            str(fact.get("fact_id") or "").strip(): fact
            for fact in facts
            if str(fact.get("fact_id") or "").strip()
        }
        fact_ids = list(facts_by_id)
        prompt = (
            f"CÂU HỎI/TASK:\n{query}\n\n"
            "TOÀN BỘ FACT REQUIREMENTS CẦN ĐÁNH GIÁ:\n"
            f"{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"CÂU TRẢ LỜI TẠM CỦA AGENT 3:\n{agent3_summary}\n\n"
            f"PROVISION ỨNG VIÊN:\n{_format_candidates(chunks)}"
        )
        result = self.client.call_tool(
            system_prompt=AGENT4_FACT_PROMPT,
            user_prompt=prompt,
            tool=self._fact_batch_tool(fact_ids),
            tool_name="submit_fact_evaluations",
        )

        accepted_statuses = {"covered", "not_applicable"}
        evaluations: dict[str, dict[str, Any]] = {}
        invalid_for_repair: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()

        raw_evaluations = list(result.get("evaluations") or [])
        if not raw_evaluations and len(fact_ids) == 1 and result.get("status"):
            raw_evaluations = [{**result, "fact_id": fact_ids[0]}]
        for raw in raw_evaluations:
            if not isinstance(raw, dict):
                continue
            fact_id = str(raw.get("fact_id") or "").strip()
            if fact_id not in facts_by_id or fact_id in seen_ids:
                continue
            seen_ids.add(fact_id)
            raw_evidence = list(raw.get("evidence") or [])
            valid_evidence = self._validate_evidence(raw_evidence, chunks)
            evaluation = dict(raw)
            evaluation["fact_id"] = fact_id
            evaluation["evidence"] = valid_evidence
            evaluation["span_repair"] = {
                "attempted": False,
                "success": False,
                "invalid_evidence": raw_evidence,
            }
            evaluations[fact_id] = evaluation
            if (
                evaluation.get("status") in accepted_statuses
                and not valid_evidence
            ):
                invalid_for_repair[fact_id] = evaluation

        # Missing/duplicate output is isolated to that fact instead of shifting
        # another evaluation onto it.
        for fact_id, fact in facts_by_id.items():
            if fact_id in evaluations:
                continue
            evaluations[fact_id] = {
                "fact_id": fact_id,
                "status": "missing_evidence",
                "reason": "Agent 4 không trả đúng một evaluation cho fact_id này.",
                "evidence": [],
                "retry_query": str(fact.get("description") or "").strip(),
                "span_repair": {
                    "attempted": False,
                    "success": False,
                    "invalid_evidence": [],
                },
            }

        # Formatting errors are repaired for all affected facts in one call.
        if invalid_for_repair:
            repair_ids = list(invalid_for_repair)
            for fact_id in repair_ids:
                evaluations[fact_id]["span_repair"]["attempted"] = True
            repair_prompt = (
                f"CÂU HỎI/TASK:\n{query}\n\n"
                "CÁC FACT VÀ EVIDENCE KHÔNG HỢP LỆ CẦN SỬA:\n"
                f"{json.dumps([{'fact': facts_by_id[fact_id], 'evaluation': invalid_for_repair[fact_id]} for fact_id in repair_ids], ensure_ascii=False)}\n\n"
                f"PROVISION ỨNG VIÊN:\n{_format_candidates(chunks)}"
            )
            repair_result = self.client.call_tool(
                system_prompt=AGENT4_SPAN_REPAIR_PROMPT,
                user_prompt=repair_prompt,
                tool=self._span_repair_batch_tool(repair_ids),
                tool_name="submit_support_span_repairs",
            )
            raw_repairs = list(repair_result.get("repairs") or [])
            if (
                not raw_repairs
                and len(repair_ids) == 1
                and repair_result.get("evidence")
            ):
                raw_repairs = [
                    {**repair_result, "fact_id": repair_ids[0]}
                ]
            seen_repair_ids: set[str] = set()
            for repair in raw_repairs:
                if not isinstance(repair, dict):
                    continue
                fact_id = str(repair.get("fact_id") or "").strip()
                if fact_id not in invalid_for_repair or fact_id in seen_repair_ids:
                    continue
                seen_repair_ids.add(fact_id)
                repaired_evidence = self._validate_evidence(
                    list(repair.get("evidence") or []),
                    chunks,
                )
                if repaired_evidence:
                    evaluations[fact_id]["evidence"] = repaired_evidence
                    evaluations[fact_id]["span_repair"]["success"] = True
                    evaluations[fact_id]["reason"] = str(
                        repair.get("reason")
                        or evaluations[fact_id].get("reason")
                        or ""
                    )

        for fact_id, fact in facts_by_id.items():
            evaluation = evaluations[fact_id]
            if (
                evaluation.get("status") in accepted_statuses
                and not evaluation.get("evidence")
            ):
                evaluation["status"] = "missing_evidence"
                evaluation["reason"] = (
                    "Agent 4 đánh dấu đã đủ nhưng support_span không hợp lệ; "
                    "lượt sửa span trên cùng candidate chunks cũng thất bại."
                )
            if evaluation.get("status") == "missing_evidence":
                evaluation["retry_query"] = _safe_retry_hint(
                    str(evaluation.get("retry_query") or ""),
                    query=query,
                    fact=fact,
                )
            else:
                evaluation["retry_query"] = ""
        return evaluations

    def evaluate_fact(
        self,
        *,
        query: str,
        fact: dict[str, Any],
        agent3_summary: str,
        chunks: list[StructuredChunk],
    ) -> dict[str, Any]:
        """Backward-compatible single-fact wrapper around batch evaluation."""
        fact_id = str(fact.get("fact_id") or "F1")
        normalized_fact = dict(fact)
        normalized_fact["fact_id"] = fact_id
        return self.evaluate_facts(
            query=query,
            facts=[normalized_fact],
            agent3_summary=agent3_summary,
            chunks=chunks,
        )[fact_id]


def search_once_with_sequential_fact_evaluation(
    *,
    searcher: GeminiSequentialSearcher,
    evaluator: GeminiSequentialFactEvaluator,
    query: str,
    max_steps: int = 1,
    memory: StructuredSearchMemory | None = None,
    temporal_scope: Any = None,
    fact_requirements: list[dict[str, Any]] | None = None,
    required_facts: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Agent 3 tool loop plus sequential Agent 4 fact verification."""
    memory = memory or StructuredSearchMemory()
    facts = normalize_fact_requirements(fact_requirements, required_facts)
    facts_by_id = {str(fact["fact_id"]): fact for fact in facts}
    pending_ids = list(facts_by_id)
    accumulated_chunks = dict(memory.accumulated_chunks)
    evaluations_by_id: dict[str, dict[str, Any]] = {}
    relevant_ids: list[int | str] = []
    previous_queries: list[str] = []
    retry_queries: list[str] = []
    history: list[dict[str, Any]] = []
    quality_issues: list[str] = []

    # Three rounds preserve retry behavior while bounding Agent 3 and Agent 4
    # calls.
    round_limit = max(1, min(int(max_steps or 1), 3))
    for step_number in range(1, round_limit + 1):
        pending_facts = [facts_by_id[fact_id] for fact_id in pending_ids]
        search_query = searcher.write_query(
            original_query=query,
            temporal_scope=temporal_scope,
            fact_requirements=pending_facts,
            retry_queries=retry_queries,
            previous_queries=previous_queries,
        )
        previous_queries.append(search_query)
        if verbose:
            print(f"\n[Agent 3] Vong {step_number}/{round_limit}")
            print(f"[Agent 3] Query tool: {search_query}")

        round_chunks = searcher.retrieve(
            search_query=search_query,
            step_number=step_number,
            temporal_scope=temporal_scope,
            fact_requirements=pending_facts,
        )
        if verbose:
            print(
                f"[Retriever] Tra ve {len(round_chunks)} candidate chunks "
                f"trong vong {step_number}:"
            )
            for index, chunk in enumerate(round_chunks, 1):
                print("\n" + "-" * 80)
                print(
                    f"[Candidate {index}/{len(round_chunks)}] "
                    f"Chunk ID: {chunk.chunk_id}"
                )
                print(f"Van ban: {chunk.document_id}")
                print(f"Vi tri: {chunk.dieu_khoan}")
                print(
                    "Hieu luc: "
                    f"[{chunk.valid_from or 'khong ro'}, "
                    f"{chunk.valid_to or 'chua ket thuc'})"
                )
                print("Noi dung:")
                print(chunk.noi_dung_goc)
            print("\n" + "-" * 80)

        before_count = len(accumulated_chunks)
        for chunk in round_chunks:
            accumulated_chunks[str(chunk.chunk_id)] = chunk
        chunks = list(accumulated_chunks.values())
        if step_number > 1 and len(accumulated_chunks) == before_count:
            quality_issues.append(
                f"Vòng {step_number} không bổ sung candidate chunk mới."
            )

        summary = searcher.summarize(
            original_query=query,
            search_query=search_query,
            chunks=chunks,
            fact_requirements=pending_facts,
        )
        if verbose:
            print("\n[Agent 3] Cau tra loi tra cuu tam thoi:")
            print(str(summary.get("summary") or "(rong)"))
            print(
                "[Agent 3] Cited chunk IDs: "
                f"{summary.get('cited_chunk_ids') or []}"
            )
            print(
                "[Agent 3] Missing fact IDs: "
                f"{summary.get('missing_fact_ids') or []}"
            )

        retry_queries = []
        next_pending_ids: list[str] = []
        round_evaluations: list[dict[str, Any]] = []
        batched_evaluations = evaluator.evaluate_facts(
            query=query,
            facts=pending_facts,
            agent3_summary=str(summary.get("summary") or ""),
            chunks=chunks,
        )
        if verbose:
            print(
                "\n[Agent 4] Da danh gia "
                f"{len(pending_facts)} facts trong 1 LLM call."
            )
        for fact in pending_facts:
            fact_id = str(fact["fact_id"])
            evaluation = batched_evaluations[fact_id]
            evaluations_by_id[fact_id] = evaluation
            round_evaluations.append(evaluation)
            if verbose:
                print("\n[Agent 4] Danh gia fact:")
                print(
                    f"  Fact: [{fact_id}] "
                    f"{fact.get('type', 'unspecified')}: "
                    f"{fact.get('description', '')}"
                )
                print(f"  Status: {evaluation.get('status')}")
                print(f"  Reason: {evaluation.get('reason') or '(rong)'}")
                print(
                    "  Retry query: "
                    f"{evaluation.get('retry_query') or '(khong)'}"
                )
                span_repair = evaluation.get("span_repair") or {}
                if span_repair.get("attempted"):
                    print(
                        "  Span repair: "
                        f"{'thanh cong' if span_repair.get('success') else 'that bai'}"
                    )
                    print(
                        "  Invalid span truoc repair: "
                        f"{span_repair.get('invalid_evidence') or []}"
                    )
                evidence_items = evaluation.get("evidence") or []
                print(f"  Support evidence: {len(evidence_items)}")
                for evidence_index, item in enumerate(evidence_items, 1):
                    print(
                        f"    [{evidence_index}] Chunk "
                        f"{item.get('chunk_id')}"
                    )
                    print(
                        "        Match: "
                        f"{item.get('span_match_type', 'exact')} "
                        f"({float(item.get('span_match_score', 1.0)):.3f})"
                    )
                    print(f"        Span: {item.get('support_span')}")
            if evaluation.get("status") in {"covered", "not_applicable"}:
                for item in evaluation.get("evidence") or []:
                    chunk_id = item["chunk_id"]
                    if chunk_id not in relevant_ids:
                        relevant_ids.append(chunk_id)
            else:
                next_pending_ids.append(fact_id)
                retry_query = str(evaluation.get("retry_query") or "").strip()
                retry_queries.append(
                    retry_query or str(fact.get("description") or "").strip()
                )

        history.append(
            {
                "step": step_number,
                "query": search_query,
                "candidate_chunk_ids": [
                    chunk.chunk_id for chunk in round_chunks
                ],
                "accumulated_candidate_count": len(accumulated_chunks),
                "agent3_summary": summary,
                "fact_evaluations": round_evaluations,
                "retry_queries": retry_queries,
            }
        )
        pending_ids = next_pending_ids
        if not pending_ids:
            break

    accepted = {"covered", "not_applicable"}
    evaluations = [
        evaluations_by_id[fact_id]
        for fact_id in facts_by_id
        if fact_id in evaluations_by_id
    ]
    covered_count = sum(item.get("status") in accepted for item in evaluations)
    if facts and covered_count == len(facts):
        final_status = "success"
    elif covered_count:
        final_status = "partial"
    else:
        final_status = "not_found"

    all_evidences = _build_evidences(
        accumulated_chunks,
        {str(chunk_id) for chunk_id in relevant_ids},
    )
    missing = [
        str(facts_by_id[fact_id].get("description") or fact_id)
        for fact_id in pending_ids
    ]
    memory.accumulated_chunks.update(accumulated_chunks)
    memory.relevant_ids.update(str(chunk_id) for chunk_id in relevant_ids)
    memory.total_search_turns += len(history)
    if verbose:
        print("\n[Agent 3+4] Ket qua cuoi:")
        print(f"  Status: {final_status}")
        print(f"  Candidate chunks da doc: {len(accumulated_chunks)}")
        print(f"  Evidence chunks duoc chap nhan: {len(all_evidences)}")
        print(f"  Facts con thieu: {missing or []}")
    return {
        "final_status": final_status,
        "all_evidences": all_evidences,
        "total_searches": len(history),
        "search_history": history,
        "quality_issues": quality_issues,
        "missing_info": missing,
        "required_facts": fact_descriptions(facts),
        "fact_evaluations": evaluations,
        "candidate_count": len(accumulated_chunks),
    }

