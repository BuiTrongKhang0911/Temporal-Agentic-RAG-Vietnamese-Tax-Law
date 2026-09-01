"""Language-only clarifier placed before the legal planner.

The clarifier only resolves linguistic ambiguity and produces a concise query.
It does not plan retrieval, create search terms, decide legal branches, or
answer the query.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rank_bm25 import BM25Okapi


from ..config import configure_agent_runtime
from ..model_registry import get_embedding_model


configure_agent_runtime()

DEFAULT_GEMINI_MODEL = os.getenv(
    "CLARIFIER_GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
)
BE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY = BE_DIR / "resources" / "clarifier_dictionary.json"
DEFAULT_DICTIONARY_RETRIEVAL = os.getenv(
    "CLARIFIER_DICTIONARY_RETRIEVAL", "bge_m3"
).strip().casefold()
DEFAULT_DICTIONARY_BGE_MODEL = os.getenv(
    "CLARIFIER_DICTIONARY_BGE_MODEL", "BAAI/bge-m3"
).strip()


_DICTIONARY_COMMON_TOKENS = {
    "ai", "ban", "bi", "ca", "cac", "can", "cho", "co", "cua", "den",
    "duoc", "gia", "hoat", "kinh", "la", "lam", "mot", "muc", "nao",
    "nguoi", "nhan", "nhieu", "nhung", "nop", "phai", "quy", "tai",
    "the", "theo", "thi", "thue", "tien", "to", "trong", "tu", "va",
    "ve", "viet", "doanh", "kinh doanh", "doanh thu", "hoat dong",
}


def _dictionary_fold(text: str) -> str:
    """Normalize text only for candidate retrieval, not for the final query."""
    folded = text.casefold()
    replacements = {
        "shoppe": "shopee",
        "tik tok": "tiktok",
        "tiktokshop": "tiktok shop",
        "lazadaa": "lazada",
    }
    for source, target in replacements.items():
        folded = folded.replace(source, target)
    return re.sub(r"[^0-9a-zA-Z\u00c0-\u024f\u1e00-\u1eff]+", " ", folded).strip()


def _dictionary_tokens(text: str) -> list[str]:
    return [token for token in _dictionary_fold(text).split() if token]


FOLLOW_UP_PROMPT = """

LƯỢT TRẢ LỜI MỐC THỜI GIAN
- Nếu ASSISTANT vừa hỏi về năm hoặc khoảng thời gian, kết hợp câu trả lời mới với
  truy vấn gốc để tạo normalized_query hoàn chỉnh.
- Nếu người dùng đã cung cấp năm hoặc khoảng thời gian rõ thì trả ready.
- Chỉ tiếp tục ask khi câu trả lời vẫn không xác định được năm hoặc khoảng thời gian.
- Không hỏi thêm dữ kiện pháp lý khác.
"""


CLARIFICATION_POLICY_PROMPT = """

QUY TẮC PHÂN BIỆT ASK VÀ READY
- Mặc định luôn chọn ready.
- Chỉ chọn ask khi người dùng ĐÃ nêu một mốc thời gian dùng để chọn phiên bản pháp luật
  nhưng mốc đó thiếu năm, hoặc chỉ dùng mốc tương đối không xác định như "hồi đó",
  "trước đây", "mấy năm trước" hay "sau Covid-19".
- Các mốc sau đều đầy đủ và bắt buộc ready: một năm; tháng/năm; ngày/tháng/năm;
  khoảng thời gian có hai đầu rõ; hai hay nhiều mốc đầy đủ dùng để so sánh.
- Năm dùng chung cho nhiều ngày hoặc tháng trong cùng một cụm cũng được xem là đầy đủ.
- Không hỏi lại khi câu không nêu thời gian; Agent 2 sẽ xử lý là truy vấn hiện hành.
- Không hỏi năm cho lịch lặp hay thời hạn tương đối như "ngày 15 hằng tháng",
  "trong 10 ngày", "từ năm thứ hai trở đi" hoặc "ngày cuối cùng của hạn nộp".
- Không hỏi bất kỳ dữ kiện nào khác ngoài mốc thời gian bị thiếu.
- Khi ask, hỏi đúng một câu ngắn về năm hoặc khoảng thời gian cần tra cứu.
"""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}

CLARIFIER_PROMPT = """Bạn là Agent 1 (Clarifier) đứng trước Agent 2 trong hệ thống RAG pháp luật thuế Việt Nam.

CHỈ LÀM BA VIỆC
1. Giải quyết tham chiếu hội thoại như "cái đó", "thuế này" khi lịch sử đã cho biết đối tượng.
2. Chuẩn hóa và rút gọn câu hỏi thành một truy vấn pháp lý rõ, ngắn, đủ ý.
3. Chọn ready, ask hoặc out_of_scope; nếu ask thì đặt đúng một câu hỏi làm rõ.

RANH GIỚI TUYỆT ĐỐI
- Không trả lời câu hỏi pháp luật.
- Không tạo từ khóa tìm kiếm, candidate concept, fact requirement hoặc kế hoạch truy xuất.
- Không xác định nhánh chịu thuế, miễn thuế, không chịu thuế hay phương pháp tính.
- Không tự thêm chủ thể, số liệu, doanh thu, ngưỡng, thuế suất, thủ tục hoặc điều kiện.
- Không hỏi thêm dữ kiện pháp lý. Agent 2-5 sẽ tra cứu và xử lý các nhánh áp dụng.
- Ngoài trường hợp mốc thời gian bị thiếu theo chính sách ASK, luôn chọn ready.
- Thuật ngữ không có trong gợi ý thì giữ nguyên, không đoán.
- Không thay một khái niệm bằng khái niệm gần nghĩa: doanh thu != doanh số;
  miễn thuế != không chịu thuế; tiền chậm nộp != tiền phạt.

CÁCH CHUẨN HÓA
- Rút gọn lời dẫn và kể chuyện nhưng giữ đủ mọi vế hỏi cùng dữ kiện có thể ảnh hưởng kết quả.
- Giữ nguyên chủ thể, hành động, số liệu, đơn vị, phủ định, ngoại lệ và quan hệ thời gian.
- Không gộp các hành động khác nhau như "kê khai và nộp" thành một hành động.
- Có thể thay cách nói đời thường bằng thuật ngữ chuẩn khi nghĩa tương đương chắc chắn.
- Có thể sửa lỗi chính tả rõ ràng nhưng không sửa tên riêng hay số liệu.
- Giữ nguyên độ chính xác và thứ tự của mọi mốc thời gian, nhất là các mốc so sánh.
- Không chia một câu thành danh sách từ khóa. Chỉ trả lại một normalized_query hoàn chỉnh.

CHỌN STATUS
- ready: đã đủ rõ về mặt ngôn ngữ để chuyển nguyên câu sang Agent 2.
- ask: chỉ khi mốc thời gian dùng để chọn phiên bản pháp luật thiếu năm hoặc hoàn toàn tương đối.
- out_of_scope: rõ ràng không liên quan đến thuế Việt Nam.

TỰ KIỂM TRA TRƯỚC KHI XUẤT
1. Đã giữ đủ các vế hỏi, chủ thể, số liệu, đơn vị và mốc thời gian chưa?
2. Có tự thêm dữ kiện hoặc kết luận pháp luật không?
3. Nếu ask, mốc thời gian có thật sự thiếu năm hoặc chỉ là mốc tương đối không?
4. Nếu câu có năm, tháng/năm hoặc ngày/tháng/năm đầy đủ, status đã là ready chưa?

VÍ DỤ
- Query: "Người nước ngoài ở Việt Nam 190 ngày thì tính thuế lương ra sao?"
  -> ready; "Người nước ngoài ở Việt Nam 190 ngày phải tính thuế thu nhập cá nhân từ tiền
  lương, tiền công như thế nào?" Không thêm hoặc kết luận "cá nhân cư trú".
- Query: "Từ ngày 1 tháng 7 dùng số định danh cá nhân thay mã số thuế thì có cần cập
  nhật thông tin không?"
  -> ask: "Bạn muốn hỏi ngày 1 tháng 7 năm nào?" Không được tự thêm "năm hiện tại".
- Query: "Tháng 7/2026 hộ kinh doanh phải khai thuế thế nào?" -> ready.
- Query: "Ngày 1/7/2026 quy định nào được áp dụng?" -> ready.
- Query: "So sánh quy định vào 31/12/2014 và 1/1/2015" -> ready; giữ đủ hai ngày.
- Query: "So sánh quy định vào 31/01 và 01/02 năm 2018" -> ready; năm 2018 dùng chung.
- Query: "Từ năm thứ hai trở đi nộp tiền thuê đất khi nào?" -> ready; đây là lịch tương đối.
- Query: "Sau Covid-19 tôi làm thêm nhưng quyết toán muộn có bị phạt không?"
  -> ask: "Thu nhập phát sinh trong năm hoặc khoảng thời gian nào?"

GỢI Ý TỪ ĐIỂN ĐƯỢC PYTHON CHỌN CHO CÂU NÀY
- Chỉ dùng canonical term khi `definition`, `note` và toàn câu xác nhận nghĩa tương đương.
- Alias chỉ tạo ứng viên, không tự động quyết định mapping và không phải căn cứ pháp luật.
- Không sao chép danh sách alias hoặc thêm kết luận nghĩa vụ thuế vào truy vấn.
- Nhà, căn hộ, phòng trọ, mặt bằng hoặc đất được cho thuê theo quan hệ thuê thông thường
  chuẩn hóa thành `cho thuê bất động sản`; xe, máy móc dùng `cho thuê tài sản`.
- Nếu có dấu hiệu cung cấp chỗ ở ngắn hạn hoặc dịch vụ đi kèm như Airbnb, homestay, khách sạn,
  thuê theo ngày hoặc theo giờ, dùng `kinh doanh lưu trú`.
- Nếu chưa chọn được duy nhất một canonical term, giữ cách nói gốc và vẫn trả ready.

{dictionary_hints}

LỊCH SỬ HỘI THOẠI
{conversation_history}

CÂU HIỆN TẠI
{user_query}

Chỉ xuất một JSON object theo đúng cấu trúc:
{{
  "status": "ready|ask|out_of_scope",
  "normalized_query": "string hoặc null",
  "clarification_question": "string hoặc null"
}}"""


class ClarifierDecision(BaseModel):
    """Strict three-field contract returned by the language model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "ask", "out_of_scope"]
    normalized_query: str | None = None
    clarification_question: str | None = None


class ClarifierOutput(ClarifierDecision):
    """Runtime envelope enriched only by deterministic Python checks."""

    status: Literal["ready", "ask", "out_of_scope"]
    original_query: str = ""
    preservation_warnings: list[str] = Field(default_factory=list)


def _number_tokens(text: str) -> list[str]:
    """Extract user-entered numeric forms without interpreting their value."""
    return re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", text)


_CURRENT_TIME_CUES = (
    "hiện nay",
    "hiện hành",
    "hiện tại",
    "bây giờ",
    "năm nay",
    "tháng này",
)

_PROTECTED_QUERY_PHRASES = (
    "cá nhân cư trú",
    "cá nhân không cư trú",
    "hộ kinh doanh",
    "cá nhân kinh doanh",
    "người lao động",
    "người nộp thuế",
    "doanh nghiệp",
    "nhà cung cấp nước ngoài",
    "tổ chức trả thu nhập",
    "thuế thu nhập cá nhân",
    "thuế giá trị gia tăng",
    "thuế thu nhập doanh nghiệp",
    "thuế tiêu thụ đặc biệt",
    "lệ phí môn bài",
    "tiền chậm nộp",
    "cho thuê nhà",
    "cho thuê bất động sản",
    "chuyển nhượng bất động sản",
    "chuyển nhượng vốn",
    "chuyển nhượng chứng khoán",
    "chuyển nhượng nhà",
    "chuyển nhượng đất",
    "bán hàng online",
    "bán hàng trực tuyến",
    "kinh doanh bất động sản",
    "kinh doanh vàng",
    "thương mại điện tử",
)

_PROTECTED_QUALIFIERS = (
    "hoàn toàn",
    "duy nhất",
    "toàn bộ",
    "chưa",
    "chỉ",
    "tối đa",
    "ít nhất",
)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(phrase.casefold() in folded for phrase in phrases)


def _temporal_scope_warnings(
    query: str,
    normalized: str,
    allowed_context: str,
) -> list[str]:
    """Detect only explicit temporal changes, not harmless wording changes."""
    warnings: list[str] = []
    query_folded = query.casefold()
    normalized_folded = normalized.casefold()
    context_folded = allowed_context.casefold()

    if _contains_any(query, _CURRENT_TIME_CUES) and not _contains_any(
        normalized, _CURRENT_TIME_CUES
    ):
        warnings.append("Model làm mất phạm vi thời gian hiện hành")
    if not _contains_any(allowed_context, _CURRENT_TIME_CUES) and _contains_any(
        normalized, _CURRENT_TIME_CUES
    ):
        warnings.append("Model tự thêm phạm vi thời gian hiện hành")

    directional_cues = {
        "trước": r"\btrước\s+(?:ngày|tháng|năm|thời điểm|khi)\b",
        "sau": r"\bsau\s+(?:ngày|tháng|năm|thời điểm|khi)\b",
        "kể từ": r"\b(?:kể\s+từ|từ\s+ngày)\b",
        "đến": r"\b(?:đến|tới)\s+ngày\b",
    }
    for label, pattern in directional_cues.items():
        query_has_cue = bool(re.search(pattern, query_folded))
        normalized_has_cue = bool(re.search(pattern, normalized_folded))
        context_has_cue = bool(re.search(pattern, context_folded))
        if query_has_cue and not normalized_has_cue:
            warnings.append(f"Model làm mất quan hệ thời gian '{label}'")
        elif not context_has_cue and normalized_has_cue:
            warnings.append(f"Model tự thêm quan hệ thời gian '{label}'")
    return warnings


def _residency_conflict(source: str, candidate: str) -> bool:
    source_folded = source.casefold()
    candidate_folded = candidate.casefold()
    resident = "cá nhân cư trú"
    non_resident = "cá nhân không cư trú"
    if non_resident in source_folded:
        return resident in candidate_folded and non_resident not in candidate_folded
    if resident in source_folded:
        return non_resident in candidate_folded
    return False


def _bare_money_like_number(text: str) -> str | None:
    # Only inspect numbers that syntactically follow a monetary concept. A
    # money-related word elsewhere in the sentence must not turn years, dates,
    # ages, legal references, or identifiers such as COVID-19 into amounts.
    amount_pattern = re.compile(
        r"\b(?:lãi|lương|doanh\s+thu|thu\s+nhập|tiền|giá|trị\s+giá)\b"
        r"(?:\s+(?:là|mức|khoảng|tầm|gần|hơn|được|nhận|phát\s+sinh))*"
        r"\s+(?P<number>\d+(?:[.,]\d+)*)",
        flags=re.IGNORECASE,
    )
    unit_pattern = re.compile(
        r"^\s*(?:%|phần\s+trăm\b|đồng\b|nghìn\b|ngàn\b|triệu\b|tỷ\b|"
        r"usd\b|eur\b)",
        flags=re.IGNORECASE,
    )
    for match in amount_pattern.finditer(text):
        if not unit_pattern.search(text[match.end() : match.end() + 24]):
            return match.group("number")
    return None


class TaxLawClarifier:
    _embedding_models: dict[tuple[str, str], SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str | None = None,
        dictionary_path: str | Path = DEFAULT_DICTIONARY,
        *,
        python_guards_enabled: bool | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self.python_guards_enabled = (
            _env_bool("CLARIFIER_PYTHON_GUARDS", True)
            if python_guards_enabled is None
            else python_guards_enabled
        )
        self.dictionary_path = Path(dictionary_path)
        self.dictionary = self._load_dictionary()
        self._build_dictionary_index()
        self.conversation_history: list[dict[str, str]] = []

    def _load_dictionary(self) -> dict[str, Any]:
        if not self.dictionary_path.exists():
            return {"entries": []}
        with self.dictionary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data.get("entries"), list):
            raise ValueError("clarifier_dictionary.json must contain an entries array")
        return data

    def _build_dictionary_index(self) -> None:
        """Build BM25 and full BGE-M3 dense indexes for dictionary retrieval."""
        self._dictionary_entries = list(self.dictionary.get("entries", []))
        documents: list[str] = []
        for entry in self._dictionary_entries:
            parts = [
                str(entry.get("legal_term") or entry.get("canonical_term") or ""),
                *[str(item) for item in entry.get("aliases", [])],
                *[str(item) for item in entry.get("strong_signals", [])],
                str(entry.get("definition") or ""),
                str(entry.get("note") or ""),
            ]
            documents.append(_dictionary_fold(" ".join(parts)))

        self._dictionary_documents = documents
        tokenized = [_dictionary_tokens(document) for document in documents]
        self._dictionary_bm25 = BM25Okapi(tokenized) if tokenized else None
        self._dictionary_dense_matrix = None
        self._dictionary_embedding_model = None
        self._dictionary_retrieval = DEFAULT_DICTIONARY_RETRIEVAL
        if self._dictionary_retrieval not in {"bge_m3", "lexical"}:
            raise ValueError(
                "CLARIFIER_DICTIONARY_RETRIEVAL must be 'bge_m3' or 'lexical'"
            )
        if documents and self._dictionary_retrieval == "bge_m3":
            self._build_dictionary_dense_index(documents)

    def _build_dictionary_dense_index(self, documents: list[str]) -> None:
        model_name = DEFAULT_DICTIONARY_BGE_MODEL
        device = os.getenv("CLARIFIER_DICTIONARY_BGE_DEVICE", "cpu").strip()
        model_key = (model_name, device)
        model = self._embedding_models.get(model_key)
        if model is None:
            model = get_embedding_model(
                model_name,
                device=device,
            )
            self._embedding_models[model_key] = model
        self._dictionary_embedding_model = model

        fingerprint_payload = json.dumps(
            {"model": model_name, "documents": documents},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:20]
        cache_dir = Path(
            os.getenv(
                "CLARIFIER_DICTIONARY_CACHE_DIR",
                str(BE_DIR.parent / ".cache"),
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"clarifier_dictionary_bge_{fingerprint}.npy"
        if cache_path.exists():
            matrix = np.load(cache_path)
        else:
            matrix = model.encode(
                documents,
                batch_size=min(16, max(len(documents), 1)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            matrix = np.asarray(matrix, dtype=np.float32)
            np.save(cache_path, matrix)
        if matrix.shape[0] != len(documents):
            raise ValueError("Cached BGE dictionary embeddings have invalid shape")
        self._dictionary_dense_matrix = matrix

    def _dictionary_hints(self, query: str) -> list[dict[str, Any]]:
        folded = _dictionary_fold(query)
        query_tokens = _dictionary_tokens(query)
        query_token_set = set(query_tokens)
        distinctive_query_tokens = {
            token
            for token in query_token_set
            if token not in _DICTIONARY_COMMON_TOKENS and len(token) > 2
        }

        bm25_scores = [0.0] * len(self._dictionary_entries)
        if self._dictionary_bm25 is not None and query_tokens:
            raw_scores = self._dictionary_bm25.get_scores(query_tokens)
            positive_max = max((float(score) for score in raw_scores), default=0.0)
            if positive_max > 0:
                bm25_scores = [
                    max(float(score), 0.0) / positive_max for score in raw_scores
                ]

        dense_scores = [0.0] * len(self._dictionary_entries)
        if self._dictionary_embedding_model is not None and folded:
            query_vector = self._dictionary_embedding_model.encode(
                [folded],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            similarities = self._dictionary_dense_matrix @ query_vector
            dense_scores = [max(float(value), 0.0) for value in similarities]

        minimum_score = float(
            os.getenv("CLARIFIER_DICTIONARY_MIN_SCORE", "0.42")
        )
        require_lexical = _env_bool(
            "CLARIFIER_DICTIONARY_REQUIRE_LEXICAL", False
        )

        matches: list[dict[str, Any]] = []
        for index, entry in enumerate(self._dictionary_entries):
            aliases = [str(item) for item in entry.get("aliases", [])]
            legal_term = str(
                entry.get("legal_term") or entry.get("canonical_term") or ""
            )
            strong_signals = [
                str(item) for item in entry.get("strong_signals", [])
            ]
            negative_signals = [
                str(item) for item in entry.get("negative_signals", [])
            ]
            matched = [
                alias for alias in aliases if _dictionary_fold(alias) in folded
            ]
            matched_signals = [
                signal
                for signal in strong_signals
                if _dictionary_fold(signal) in folded
            ]
            matched_negative_signals = [
                signal
                for signal in negative_signals
                if _dictionary_fold(signal) in folded
            ]
            canonical_match = bool(
                legal_term and _dictionary_fold(legal_term) in folded
            )
            document_tokens = set(
                _dictionary_tokens(self._dictionary_documents[index])
            )
            distinctive_overlap = distinctive_query_tokens & document_tokens
            bm25_score = bm25_scores[index]
            dense_score = dense_scores[index]
            lexical_match = bool(distinctive_overlap) or bm25_score >= 0.15
            inferred_match = dense_score >= minimum_score or (
                bool(distinctive_overlap) and bm25_score >= 0.28
            )
            if require_lexical and not lexical_match:
                inferred_match = False
            if (
                matched_negative_signals
                and not matched
                and not canonical_match
                and not matched_signals
            ):
                inferred_match = False
            if not matched and not canonical_match and not inferred_match:
                continue

            exact_match = any(
                _dictionary_fold(alias) == folded for alias in matched
            )
            hybrid_score = min(
                1.0,
                (0.55 if matched or canonical_match else 0.0)
                + (0.08 if matched and matched_signals else 0.0)
                + (0.12 if exact_match else 0.0)
                + 0.20 * bm25_score
                + 0.70 * dense_score,
            )
            if not matched and not canonical_match and hybrid_score < minimum_score:
                continue
            methods: list[str] = []
            if matched:
                methods.append("exact_alias")
            if canonical_match:
                methods.append("exact_legal_term")
            if bm25_score >= 0.15:
                methods.append("bm25")
            if dense_score >= minimum_score:
                methods.append("bge_m3_dense")
            matches.append(
                {
                    "matched_aliases": matched,
                    "matched_strong_signals": matched_signals,
                    "matched_negative_signals": matched_negative_signals,
                    "legal_term_candidate": legal_term,
                    "definition": entry.get("definition", ""),
                    "note": entry.get("note", ""),
                    "source_refs": entry.get("source_refs", []),
                    "ambiguity_group": entry.get("ambiguity_group"),
                    "retrieval_methods": methods,
                    "retrieval_score": round(hybrid_score, 4),
                    "dense_score": round(dense_score, 4),
                    "bm25_score": round(bm25_score, 4),
                    "_rank": (hybrid_score, exact_match, -index),
                }
            )

        matches.sort(key=lambda item: item["_rank"], reverse=True)
        selected: list[dict[str, Any]] = []
        group_counts: dict[str, int] = {}
        for item in matches:
            group = item.get("ambiguity_group")
            if group and group_counts.get(group, 0) >= 3:
                continue
            item.pop("_rank", None)
            selected.append(item)
            if group:
                group_counts[group] = group_counts.get(group, 0) + 1
            if len(selected) >= 3:
                break
        return selected

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        if not history:
            return "Không có."
        lines = []
        for message in history[-8:]:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) or "Không có."

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY không tồn tại trong .env")
        model = self.model_name.removeprefix("models/")
        response = requests.post(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            ),
            headers={"x-goog-api-key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0,
                },
            },
            timeout=float(os.getenv("CLARIFIER_TIMEOUT", "90")),
        )
        response.raise_for_status()
        payload = response.json()
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Gemini response thiếu JSON content: {payload}") from exc
        return json.loads(text)

    @staticmethod
    def _guard_output(
        query: str,
        output: ClarifierOutput,
        source_context: str | None = None,
    ) -> ClarifierOutput:
        output.original_query = query
        if output.status == "ask":
            output.normalized_query = None
            return output
        if output.status == "out_of_scope":
            output.normalized_query = None
            output.clarification_question = None
            return output

        output.clarification_question = None

        normalized = (output.normalized_query or query).strip()
        warnings: list[str] = []
        allowed_context = source_context or query
        allowed_numbers = set(_number_tokens(allowed_context))
        normalized_numbers = set(_number_tokens(normalized))
        missing_numbers = [
            token for token in _number_tokens(query) if token not in normalized_numbers
        ]
        if missing_numbers:
            warnings.append(
                "Model làm rơi số liệu: " + ", ".join(missing_numbers)
            )
        added_numbers = sorted(normalized_numbers - allowed_numbers)
        if added_numbers:
            warnings.append(
                "Model tự thêm giá trị cụ thể: " + ", ".join(added_numbers)
            )

        warnings.extend(
            _temporal_scope_warnings(query, normalized, allowed_context)
        )
        if _residency_conflict(allowed_context, normalized):
            warnings.append("Model đổi nhánh cá nhân cư trú/không cư trú")

        normalized_folded = normalized.casefold()
        missing_phrases = [
            phrase
            for phrase in _PROTECTED_QUERY_PHRASES
            if phrase in query.casefold() and phrase not in normalized_folded
        ]
        if missing_phrases:
            warnings.append(
                "Model làm mất chủ thể/hoạt động: " + ", ".join(missing_phrases)
            )
        missing_qualifiers = [
            qualifier
            for qualifier in _PROTECTED_QUALIFIERS
            if query.casefold().count(qualifier)
            > normalized_folded.count(qualifier)
        ]
        if missing_qualifiers:
            warnings.append(
                "Model làm mất lượng từ/phủ định quan trọng: "
                + ", ".join(missing_qualifiers)
            )

        if warnings:
            normalized = query
        output.normalized_query = normalized
        output.preservation_warnings = warnings
        return output

    def normalize(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ClarifierOutput:
        query = user_query.strip()
        if not query:
            return ClarifierOutput(
                status="ask",
                original_query=user_query,
                clarification_question="Bạn muốn hỏi vấn đề thuế nào?",
            )
        history = conversation_history or []
        reference_markers = ("cái đó", "việc đó", "thuế này", "bên sàn")
        hint_text = query
        if any(marker in query.casefold() for marker in reference_markers):
            hint_text += " " + " ".join(
                str(message.get("content", "")) for message in history[-4:]
            )
        hints = self._dictionary_hints(hint_text)
        prompt = CLARIFIER_PROMPT.format(
            dictionary_hints=json.dumps(hints, ensure_ascii=False, indent=2),
            conversation_history=self._format_history(history),
            user_query=query,
        )
        prompt += CLARIFICATION_POLICY_PROMPT
        if history and any(
            str(message.get("role", "")).casefold() == "assistant"
            for message in history[-2:]
        ):
            prompt += FOLLOW_UP_PROMPT
        try:
            raw = self._call_llm(prompt)
            decision = ClarifierDecision.model_validate(raw)
            parsed = ClarifierOutput(**decision.model_dump())
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            raise ValueError(f"Clarifier trả JSON không hợp lệ: {exc}") from exc
        if (
            parsed.status == "ask"
            and history
            and any(marker in query.casefold() for marker in reference_markers)
        ):
            correction_prompt = (
                prompt
                + "\n\nSỬA LẠI: Lịch sử đã xác định được tham chiếu. "
                "Không hỏi xác nhận; hãy trả status=ready và viết rõ tham chiếu "
                "trong normalized_query."
            )
            corrected = self._call_llm(correction_prompt)
            decision = ClarifierDecision.model_validate(corrected)
            parsed = ClarifierOutput(**decision.model_dump())
        source_context = query + " " + " ".join(
            str(message.get("content", "")) for message in history[-8:]
        )
        if self.python_guards_enabled:
            return self._guard_output(query, parsed, source_context)

        # Raw-LLM benchmark mode: retain only schema normalization. Python does
        # not rewrite the query, ask on the model's behalf, or reject semantics.
        parsed.original_query = query
        parsed.preservation_warnings = []
        if parsed.status != "ready":
            parsed.normalized_query = None
        if parsed.status != "ask":
            parsed.clarification_question = None
        return parsed

    def process(self, user_input: str) -> dict[str, Any]:
        """Process one conversational turn using the minimal Agent 1 contract."""
        output = self.normalize(user_input, self.conversation_history)
        self.conversation_history.append({"role": "user", "content": user_input})
        if output.status == "ask" and output.clarification_question:
            self.conversation_history.append(
                {"role": "assistant", "content": output.clarification_question}
            )
        return output.model_dump()

    def reset(self) -> None:
        self.conversation_history = []


if __name__ == "__main__":
    agent = TaxLawClarifier()
    print("Clarifier ngôn ngữ. Gõ exit để thoát.")
    while True:
        text = input("\nUSER: ").strip()
        if text.casefold() in {"exit", "quit"}:
            break
        print(json.dumps(agent.process(text), ensure_ascii=False, indent=2))
