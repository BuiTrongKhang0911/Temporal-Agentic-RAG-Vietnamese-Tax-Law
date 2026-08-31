"""
Shared Agent 3 and Agent 4 runtime using structured chunk IDs.

The public Agent 3 and Agent 4 interfaces are exposed by ``agent3.py`` and
``agent4.py``. This module keeps their shared candidate, memory, and evidence
types in one place.

Design:
- The search tool stores exact Qdrant chunks keyed by chunk_id.
- Agent 3 returns short prose with chunk_id citations; there is no LLM parser.
- Agent 4 selects/excludes chunk IDs and never rewrites legal text.
- Python builds ChunkEvidence deterministically from selected chunks.
- ReAct memory is reset before every retry. Cross-retry memory is explicit.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Set

import google.generativeai as genai
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.gemini import Gemini
from pydantic import BaseModel, Field

from ..retriever.structured import ArticleExpansionRAG
from ..retriever.temporal import (
    describe_temporal_scope,
    metadata_overlaps_scope,
    normalize_temporal_scope,
)
from ..config import configure_agent_runtime


configure_agent_runtime()


def normalize_fact_requirements(
    fact_requirements: Optional[List[Dict]] = None,
    required_facts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Preserve typed requirements while supporting legacy string facts."""

    normalized: List[Dict[str, str]] = []
    seen = set()
    for index, requirement in enumerate(fact_requirements or [], 1):
        if not isinstance(requirement, dict):
            continue
        fact_type = str(requirement.get("type") or "unspecified").strip()
        description = str(requirement.get("description") or "").strip()
        task_id = str(requirement.get("task_id") or "T1").strip()
        fact_id = str(
            requirement.get("fact_id") or f"{task_id}_F{index}"
        ).strip()
        if not description:
            continue
        key = (task_id.casefold(), fact_type.casefold(), description.casefold())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "fact_id": fact_id,
                "task_id": task_id,
                "type": fact_type,
                "description": description,
            }
        )

    if normalized:
        return normalized

    return [
        {
            "fact_id": f"T1_F{index}",
            "task_id": "T1",
            "type": "unspecified",
            "description": str(fact).strip(),
        }
        for index, fact in enumerate(dict.fromkeys(required_facts or []), 1)
        if str(fact).strip()
    ]


def fact_descriptions(
    fact_requirements: Optional[List[Dict[str, str]]],
) -> List[str]:
    return [
        str(requirement.get("description") or "").strip()
        for requirement in fact_requirements or []
        if str(requirement.get("description") or "").strip()
    ]


def compact_query_label(query: str) -> str:
    """Return the semantic line for readable console logs."""

    lines = [line.strip() for line in str(query).splitlines() if line.strip()]
    for prefix in ("Semantic query:", "SEMANTIC QUERY GỢI Ý:"):
        for line in lines:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return lines[0] if lines else ""


def fact_requirement_key(requirement: Dict[str, str]) -> tuple[str, str]:
    return (
        str(requirement.get("type") or "unspecified").strip().casefold(),
        re.sub(
            r"\s+",
            " ",
            str(requirement.get("description") or "").strip(),
        ).casefold(),
    )


def is_gemini_empty_candidate_error(error: Exception) -> bool:
    error_text = str(error)
    error_type = type(error).__name__
    return (
        "StopCandidateException" in error_type
        or "finish_reason" in error_text
        or "content {\n}" in error_text
    )


class BienSoPhapLy(BaseModel):
    """Kept for compatibility with PlannerOrchestrator and Agent 5."""

    thue_suat: Optional[str] = None
    nguong_mien_thue: Optional[str] = None
    nguong_chiu_thue: Optional[str] = None
    giam_tru_gia_canh: Optional[str] = None
    thoi_han: Optional[str] = None
    dieu_kien_ap_dung: Optional[str] = None
    cach_tinh: Optional[str] = None


class ChunkEvidence(BaseModel):
    """Final evidence built by Python, never rewritten by an LLM."""

    chunk_id: str
    buoc_so: int
    van_ban: str
    document_id: str
    provision_key: Optional[str] = None
    dieu_khoan: str
    tinh_trang_hieu_luc: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    version: Optional[int] = None
    noi_dung_goc: Optional[str] = None
    cac_bien_so_phap_ly: Optional[BienSoPhapLy] = None
    amended_by_document_id: Optional[str] = None
    amended_by_source_location: Optional[str] = None
    change_type: Optional[str] = None
    temporal_scope: Optional[Dict] = None


class StructuredChunk(BaseModel):
    """Canonical representation of one result returned by Qdrant."""

    chunk_id: str
    first_seen_step: int
    query: str
    van_ban: str
    document_id: str
    provision_key: Optional[str] = None
    document_type: Optional[str] = None
    legal_rank: Optional[int] = None
    dieu_khoan: str
    tinh_trang_hieu_luc: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    version: Optional[int] = None
    noi_dung_goc: str
    amended_by_document_id: Optional[str] = None
    amended_by_source_location: Optional[str] = None
    change_type: Optional[str] = None
    temporal_scope: Optional[Dict] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    score: float = 0.0


class SearchResult(BaseModel):
    """Agent 3 output plus exact chunks captured from its tool calls."""

    status: str
    message: Optional[str] = None
    search_query: str = ""
    agent_output: str = ""
    chunks: List[StructuredChunk] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    fact_requirements: List[Dict[str, str]] = Field(default_factory=list)
    required_facts: List[str] = Field(default_factory=list)
    raw_chunks_text: Optional[str] = None


class ExcludedChunkDecision(BaseModel):
    chunk_id: str
    reason: str


class FactCoverageDecision(BaseModel):
    fact_id: str = Field(
        description="Giữ nguyên fact_id do Agent 2 cung cấp."
    )
    task_id: str = Field(
        description="Giữ nguyên task_id do Agent 2 cung cấp."
    )
    fact_type: str = Field(
        description="Giữ nguyên type của fact requirement do Agent 2 cung cấp."
    )
    required_fact: str
    status: str = Field(
        description=(
            "covered | missing_evidence | not_applicable"
        )
    )
    chunk_id: Optional[str]
    support_span: Optional[str] = Field(
        description=(
            "Một đoạn nguyên văn LIÊN TỤC sao chép chính xác từ nội_dung_gốc; "
            "không ghép các đoạn rời bằng ...; null nếu missing."
        )
    )


class EvaluationDecision(BaseModel):
    decision: str = Field(description="pass | retry | stop")
    reason: str
    quality_issues: Optional[List[str]] = Field(
        description="Các vấn đề chất lượng; nếu không có thì trả null."
    )
    relevant_chunk_ids: Optional[List[str]] = Field(
        description="Danh sách đầy đủ chunk ID cần giữ; nếu không có thì trả []."
    )
    excluded_chunk_ids: Optional[List[str]] = Field(
        description="Danh sách chunk ID bị loại; nếu không có thì trả []."
    )
    exclusion_reasons: Optional[List[str]] = Field(
        description=(
            "Lý do tương ứng với từng phần tử trong excluded_chunk_ids; "
            "nếu không có chunk bị loại thì trả []."
        )
    )
    missing_info: Optional[str] = Field(
        description="Thông tin pháp lý thiết yếu còn thiếu; nếu đủ thì trả null."
    )
    suggested_retry_queries: Optional[List[str]] = Field(
        description="Query đề xuất khi retry; nếu không retry thì trả []."
    )
    coverage: List[FactCoverageDecision] = Field(
        description="Một phần tử cho từng required_fact.",
    )


class LegalSearchTool:
    """Hybrid search wrapper that preserves exact structured results."""

    def __init__(self):
        print("[Agent 3 Structured] Dang khoi tao RAG system...")
        self.rag = ArticleExpansionRAG(
            use_hybrid=True,
            seed_location_limit=5,
            final_location_limit=5,
        )
        self.current_step = 0
        self.current_chunks: Dict[str, StructuredChunk] = {}
        self.current_raw_observations: List[str] = []
        self.fact_requirements: List[Dict[str, str]] = []
        self.required_facts: List[str] = []
        self.temporal_scope = normalize_temporal_scope({"type": "current"})
        print("[Agent 3 Structured] RAG system da san sang\n")

    def set_temporal_scope(self, scope: Dict) -> None:
        self.temporal_scope = normalize_temporal_scope(scope)
        self.rag.set_temporal_scope(self.temporal_scope)

    def begin_turn(
        self,
        step_number: int,
        fact_requirements: Optional[List[Dict[str, str]]] = None,
        required_facts: Optional[List[str]] = None,
    ) -> None:
        self.current_step = step_number
        self.current_chunks = {}
        self.current_raw_observations = []
        self.fact_requirements = normalize_fact_requirements(
            fact_requirements,
            required_facts,
        )
        self.required_facts = fact_descriptions(self.fact_requirements)
        self.rag.set_fact_requirements(
            self.fact_requirements,
            self.required_facts,
        )

    def search(self, query: str) -> str:
        print(f"\n[Agent 3 Structured TOOL] Dang tra cuu: '{query}'")
        try:
            chunks = self.rag.retrieve_chunks(query, limit=5)
        except Exception as exc:
            message = f"Lỗi khi tra cứu: {exc}"
            print(f"[Agent 3 Structured TOOL] {message}")
            return message

        if not chunks:
            observation = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu luật."
            self.current_raw_observations.append(
                f"QUERY: {query}\n\n{observation}"
            )
            return observation

        result_parts = [f"Tìm thấy {len(chunks)} đoạn luật liên quan:\n"]
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata") or {}
            chunk_id = str(chunk.get("id"))
            doc_id = str(metadata.get("document_id") or "N/A")
            title = str(metadata.get("title") or "N/A")
            location = str(metadata.get("location_label") or "Không xác định")
            status = metadata.get("status") or "ConHieuLuc"
            valid_from = metadata.get("valid_from") or metadata.get("effective_date")
            valid_to = metadata.get("valid_to")
            version = metadata.get("version")
            original_content = str(
                chunk.get("provision_content")
                or chunk.get("original_content")
                or ""
            )
            amended_by_document_id = metadata.get("amended_by_document_id")
            amended_by_source_location = metadata.get("amended_by_source_location")
            change_type = metadata.get("change_type")

            structured = StructuredChunk(
                chunk_id=chunk_id,
                first_seen_step=self.current_step,
                query=query,
                van_ban=f"{title} ({doc_id})",
                document_id=doc_id,
                provision_key=str(
                    metadata.get("provision_key") or f"{doc_id}|{location}"
                ),
                document_type=(
                    str(metadata.get("document_type"))
                    if metadata.get("document_type") is not None
                    else None
                ),
                legal_rank=(
                    int(metadata.get("legal_rank"))
                    if metadata.get("legal_rank") is not None
                    else None
                ),
                dieu_khoan=location,
                tinh_trang_hieu_luc=str(status) if status is not None else None,
                valid_from=str(valid_from) if valid_from else None,
                valid_to=str(valid_to) if valid_to else None,
                version=int(version) if version is not None else None,
                noi_dung_goc=original_content,
                amended_by_document_id=(
                    str(amended_by_document_id) if amended_by_document_id else None
                ),
                amended_by_source_location=(
                    str(amended_by_source_location)
                    if amended_by_source_location
                    else None
                ),
                change_type=str(change_type) if change_type else None,
                temporal_scope=self.temporal_scope,
                source_chunk_ids=[
                    str(chunk_id)
                    for chunk_id in (chunk.get("source_chunk_ids") or [chunk_id])
                ],
                score=float(chunk.get("score") or 0.0),
            )
            self.current_chunks[chunk_id] = structured

            result_parts.extend(
                [
                    "\n" + "=" * 60,
                    f"[Doan {index}] [Chunk ID: {chunk_id}]",
                    f"Văn bản: {structured.van_ban}",
                    f"Loại văn bản: {structured.document_type or 'Chưa xác định'}",
                    f"Cấp hiệu lực: {structured.legal_rank if structured.legal_rank is not None else 'Chưa xác định'}",
                    f"Vị trí: {location}",
                    f"Tình trạng: {status}",
                    f"Phiên bản: {structured.version or 1}",
                    f"Khoảng hiệu lực: [{structured.valid_from or 'không rõ'}, {structured.valid_to or 'chưa kết thúc'})",
                ]
            )
            if structured.amended_by_document_id:
                result_parts.extend(
                    [
                        f"Được sửa bởi văn bản: {structured.amended_by_document_id}",
                        f"Vị trí sửa đổi: {structured.amended_by_source_location or 'Không rõ'}",
                        f"Loại thay đổi: {structured.change_type or 'Không rõ'}",
                    ]
                )
            else:
                result_parts.append("Được sửa bởi văn bản: Không có")
            result_parts.extend(
                [
                    f"Score: {structured.score:.4f}",
                    "Nội dung:",
                    original_content,
                    "=" * 60,
                ]
            )

        result_text = "\n".join(result_parts)
        self.current_raw_observations.append(f"QUERY: {query}\n\n{result_text}")
        print(
            f"[Agent 3 Structured TOOL] Da tim thay {len(chunks)} chunks"
        )
        return result_text


AGENT3_SYSTEM_PROMPT = """Bạn là Agent 3, chuyên gia TRA CỨU pháp luật thuế
Việt Nam. Bạn thu thập căn cứ cho Agent 4 đánh giá; bạn không phải Agent trả
lời cuối cùng.

===============================================================================
1. KHÓA NGUỒN DỮ LIỆU VÀ NGỮ CẢNH
===============================================================================
- Tool tra_cuu_luat_thue là nguồn pháp lý DUY NHẤT. Không dùng trí nhớ huấn
  luyện để chèn văn bản, ngưỡng, thuế suất, điều kiện hoặc cách tính.
- Luôn giữ các neo của QUERY GỐC: đối tượng, trạng thái cư trú, loại thu
  nhập/hoạt động, loại thuế và thông tin cần tìm.
- Phần "NHIỆM VỤ TRA CỨU" không được mở rộng phạm vi của phần "BỐI CẢNH CÂU
  HỎI GỐC". Không tự tìm thêm loại thuế mà người dùng không hỏi.
- Trùng từ khóa, trùng con số hoặc trùng Điều nhưng khác đối tượng/loại thu
  nhập không được xem là cùng chủ đề.

===============================================================================
2. GIAO THỨC TRA CỨU
===============================================================================
- Mỗi lượt BẮT BUỘC gọi tool ít nhất một lần.
- Nếu query có nhiều thông tin cần tìm, tách thành các mục rõ ràng và gọi tool
  riêng cho mục còn thiếu. Không tìm lại mục đã được chunk cũ chứng minh đủ.
- Đọc cả `type` và `description` của từng typed fact requirement. Không xem một
  chunk chỉ đúng chủ đề là đã đủ:
  * applicability cần căn cứ về đối tượng, phạm vi hoặc điều kiện áp dụng;
  * tax_base cần căn cứ về đại lượng làm căn cứ tính thuế;
  * tax_rate cần thuế suất hoặc tỷ lệ áp dụng;
  * threshold_or_exemption cần ngưỡng, miễn thuế hoặc trường hợp không phải nộp;
  * calculation_method cần công thức hoặc phương pháp tính;
  * obligation/procedure/deadline/authority phải đúng loại nghĩa vụ, thủ tục,
    thời hạn hoặc cơ quan được yêu cầu.
- Nếu chunk dẫn chiếu đến Điểm/Khoản/Điều khác và nội dung dẫn chiếu cần thiết
  để trả lời, phải tra cứu tiếp vị trí đó.
- Nếu chunk chỉ PHÂN LOẠI thu nhập/hoạt động mà chưa có thuế suất, căn cứ tính
  thuế, ngưỡng hoặc điều kiện, phải truy vết điều khoản khác dẫn chiếu CHÍNH XÁC
  về vị trí phân loại.
- Query truy vết phải có đồng thời:
  [thông tin còn thiếu] + [chủ đề gốc] + [đối tượng] +
  [đầy đủ Điểm/Khoản/Điều đã được chunk chứng minh].
- Không đổi Khoản, không tự đoán Điểm và không tìm vị trí pháp lý đơn độc.
- Nếu một cách diễn đạt chưa đủ, thử cách diễn đạt cụ thể hơn, nhóm pháp lý rộng
  hơn hoặc từ đồng nghĩa, nhưng vẫn phải giữ các neo ngữ nghĩa của QUERY GỐC.
- Không kết luận "không có quy định" chỉ sau một query hoặc chỉ vì thông tin
  không nằm ngay trong điều khoản phân loại.

Ví dụ đúng:
- Đã có chunk xác định tài sản số tại điểm d khoản 10 Điều 3, nhưng thiếu thuế
  suất: tìm "thuế suất chuyển nhượng tài sản số cá nhân cư trú quy định tại
  điểm d khoản 10 Điều 3".
- Kết quả có thể nằm tại một Điều khác dẫn chiếu ngược về điểm d khoản 10 Điều 3.

Ví dụ sai:
- "thuế suất điểm d Điều 3" vì mất Khoản 10, chủ đề và đối tượng.
- Đổi từ "kinh doanh thương mại điện tử" sang "chuyển nhượng tài sản số".

===============================================================================
3. ĐỌC VÀ ĐỐI CHIẾU CHUNK
===============================================================================
- Mỗi kết luận pháp lý phải được nội dung chunk chứng minh trực tiếp.
- Phân biệt chức năng của các ngưỡng: ngưỡng miễn/không phải nộp thuế; ngưỡng
  lựa chọn phương pháp; ngưỡng hóa đơn/kê khai. Không thay thế lẫn nhau.
- Với cách tính, phân biệt doanh thu, thu nhập tính thuế, giá chuyển nhượng và
  phần vượt ngưỡng. Không tự suy ra toán hạng không có trong chunk.
- Nếu chunk có amended_by_document_id, nội dung chunk đã là version được tái tạo
  bởi văn bản đó. Dùng trực tiếp nội dung hiện hành và không phục dựng lại từ
  văn bản cũ.
- Không xem score là bằng chứng nội dung đúng; phải đọc nội dung gốc.

===============================================================================
4. ĐẦU RA CHO AGENT 4
===============================================================================
- Không trả JSON và không tự dựng evidence.
- Khi kết thúc, BẮT BUỘC bắt đầu câu trả lời cuối bằng đúng từ khóa `Answer:`
  để bộ ReAct parser nhận diện. Không viết `Thought:` sau khi đã bắt đầu Answer.
- Trả lời ngắn gọn theo ba mục:
  KẾT LUẬN TẠM THỜI: điều các chunk thực sự chứng minh.
  CĂN CỨ: mỗi căn cứ kèm đúng [Chunk ID: ...].
  CÒN THIẾU: thông tin chưa đủ hoặc "Không".
- Không tạo Chunk ID, không sửa nội dung chunk, không chèn con số ngoài chunk.
- Nếu nhiều chunk tạo thành chuỗi dẫn chiếu, trích dẫn TẤT CẢ chunk trong chuỗi.
- Không gọi lại tool bằng query giống hệt query đã gọi trong cùng lượt. Khi đã
  đủ căn cứ hoặc không còn query mới hợp lý, phải kết thúc bằng `Answer:`.
- Với nhánh comparison, mỗi lần Agent 3 chỉ xử lý MỘT temporal scope Python đã
  khóa. Không tự tra cứu mốc còn lại, không so sánh hai mốc và không cố sửa
  temporal scope dựa trên ngày xuất hiện trong câu hỏi cha. Nhánh còn lại được
  Orchestrator gọi độc lập; Agent 5 mới tổng hợp phép so sánh.

CHECKLIST TRƯỚC KHI TRẢ LỜI:
- Đã gọi tool?
- Còn đúng đối tượng, cư trú, loại thu nhập và thông tin cần tìm?
- Mỗi kết luận có Chunk ID chứng minh?
- Đã truy vết dẫn chiếu cần thiết?
- Đã chọn đúng version và khoảng hiệu lực của chunk?
"""


class LegalSearcher:
    """Agent 3 without an LLM parser."""

    def __init__(self, api_retry_attempts: int = 2):
        print("[Agent 3 Structured] Dang khoi tao...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY khong ton tai trong .env")

        self.api_retry_attempts = max(1, api_retry_attempts)
        self.model_name = os.getenv(
            "AGENT3_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
        )
        self.llm = Gemini(
            model=self.model_name,
            api_key=api_key,
            temperature=0.0,
        )
        self.legal_tool = LegalSearchTool()
        self.search_tool = FunctionTool.from_defaults(
            fn=self.legal_tool.search,
            name="tra_cuu_luat_thue",
            description=(
                "Tra cuu co so du lieu Luat thue Viet Nam va tra ve cac "
                "structured chunks co Chunk ID."
            ),
        )
        self.react_agent = self._new_react_agent()
        print("[Agent 3 Structured] San sang\n")

    def set_temporal_scope(self, scope: Dict) -> None:
        self.legal_tool.set_temporal_scope(scope)

    def _new_react_agent(self):
        return ReActAgent.from_tools(
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            context=AGENT3_SYSTEM_PROMPT,
        )

    def _reset_react_agent(self) -> None:
        """Reset implicit chat memory while retaining memory inside one chat."""
        try:
            reset = getattr(self.react_agent, "reset", None)
            if callable(reset):
                reset()
                return
        except Exception:
            pass
        self.react_agent = self._new_react_agent()

    @staticmethod
    def _format_previous_context(
        relevant_chunks: Optional[List[StructuredChunk]],
        missing_info: Optional[str],
        query: str,
        original_query: Optional[str] = None,
        temporal_scope: Optional[Dict] = None,
        fact_requirements: Optional[List[Dict[str, str]]] = None,
        required_facts: Optional[List[str]] = None,
    ) -> str:
        normalized_scope = normalize_temporal_scope(
            temporal_scope or {"type": "current"}
        )
        original = str(original_query or query).strip()
        current = str(query).strip()
        parts = ["TRA CỨU PHÁP LÝ:"]
        if original == current:
            parts.extend(["Query:", current])
        else:
            parts.extend(["Bối cảnh:", original, "Query lượt này:", current])
        parts.extend(
            [
                "Thời gian (Python đã khóa): "
                + describe_temporal_scope(normalized_scope),
                "Không được tự đổi hoặc mở rộng phạm vi thời gian.",
                "Facts cần tìm:",
            ]
        )
        for requirement in normalize_fact_requirements(
            fact_requirements,
            required_facts,
        ):
            parts.append(
                f"- [{requirement.get('fact_id')}] "
                f"{requirement.get('type')}: "
                f"{requirement.get('description')}"
            )
        parts.append("=" * 60)
        if not relevant_chunks:
            parts.extend(
                [
                    "CHƯA CÓ CHUNK ĐƯỢC AGENT 4 XÁC NHẬN TỪ LƯỢT TRƯỚC.",
                    "Bắt buộc gọi tool để tra cứu.",
                ]
            )
            return "\n".join(parts)

        parts.append("CÁC CHUNK ĐÃ ĐƯỢC AGENT 4 XÁC NHẬN TỪ LƯỢT TRƯỚC:")
        for chunk in relevant_chunks:
            parts.extend(
                [
                    f"\n[Chunk ID: {chunk.chunk_id}]",
                    f"Văn bản: {chunk.van_ban}",
                    f"Vị trí: {chunk.dieu_khoan}",
                    f"Nội dung: {chunk.noi_dung_goc}",
                ]
            )
            if chunk.amended_by_document_id:
                parts.extend(
                    [
                        f"Được sửa bởi văn bản: {chunk.amended_by_document_id}",
                        f"Vị trí sửa đổi: {chunk.amended_by_source_location or 'Không rõ'}",
                        f"Loại thay đổi: {chunk.change_type or 'Không rõ'}",
                    ]
                )

        parts.extend(
            [
                "\n" + "=" * 60,
                f"THÔNG TIN CÒN THIẾU: {missing_info or 'Chưa xác định'}",
                f"QUERY RETRY: {query}",
                "Hãy gọi tool để tìm bổ sung; không chỉ dựa vào context cũ.",
            ]
        )
        return "\n".join(parts)

    @staticmethod
    def _extract_cited_chunk_ids(text: str) -> List[str]:
        patterns = [
            r"\[Chunk ID:\s*([^\]]+)\]",
            r"\[chunk_id:\s*([^\]]+)\]",
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        return list(dict.fromkeys(item.strip() for item in found if item.strip()))

    def search_and_structure(
        self,
        step_number: int,
        query: str,
        previous_evidences: Optional[List[StructuredChunk]] = None,
        missing_info: Optional[str] = None,
        original_query: Optional[str] = None,
        temporal_scope: Optional[Dict] = None,
        fact_requirements: Optional[List[Dict[str, str]]] = None,
        required_facts: Optional[List[str]] = None,
    ) -> SearchResult:
        print(
            f"\n[Agent 3 Structured] Bước {step_number}: "
            f"{compact_query_label(query)}"
        )
        final_query = self._format_previous_context(
            previous_evidences,
            missing_info,
            query,
            original_query,
            temporal_scope,
            fact_requirements,
            required_facts,
        )

        last_error: Optional[Exception] = None
        captured_chunks: Dict[str, StructuredChunk] = {}
        captured_raw_observations: List[str] = []
        for attempt in range(1, self.api_retry_attempts + 1):
            self._reset_react_agent()
            self.legal_tool.begin_turn(
                step_number,
                fact_requirements,
                required_facts,
            )
            try:
                print(
                    f"[Agent 3 Structured] Goi ReAct Agent "
                    f"(API attempt {attempt}/{self.api_retry_attempts})..."
                )
                response = self.react_agent.chat(final_query)
                agent_output = str(response)
                captured_chunks.update(self.legal_tool.current_chunks)
                captured_raw_observations.extend(
                    self.legal_tool.current_raw_observations
                )
                chunks = list(captured_chunks.values())
                raw_text = "\n\n".join(captured_raw_observations)

                if not chunks:
                    return SearchResult(
                        status="error",
                        message=(
                            "Agent 3 did not retrieve any structured chunk; "
                            "a tool call is required."
                        ),
                        search_query=query,
                        agent_output=agent_output,
                        chunks=[],
                        cited_chunk_ids=[],
                        fact_requirements=list(
                            self.legal_tool.fact_requirements
                        ),
                        required_facts=list(self.legal_tool.required_facts),
                        raw_chunks_text=raw_text,
                    )

                cited_ids = self._extract_cited_chunk_ids(agent_output)
                print("\n[Agent 3 Structured] CÂU TRẢ LỜI AGENT 3:")
                print(agent_output)
                print(
                    "[Agent 3 Structured] CHUNK IDS ĐÃ THU ĐƯỢC: "
                    f"{[chunk.chunk_id for chunk in chunks]}"
                )
                return SearchResult(
                    status="retrieved",
                    message=f"Retrieved {len(chunks)} structured chunks",
                    search_query=query,
                    agent_output=agent_output,
                    chunks=chunks,
                    cited_chunk_ids=cited_ids,
                    fact_requirements=list(
                        self.legal_tool.fact_requirements
                    ),
                    required_facts=list(self.legal_tool.required_facts),
                    raw_chunks_text=raw_text,
                )
            except Exception as exc:
                last_error = exc
                captured_chunks.update(self.legal_tool.current_chunks)
                captured_raw_observations.extend(
                    self.legal_tool.current_raw_observations
                )
                if is_gemini_empty_candidate_error(exc):
                    print(
                        "[Agent 3 Structured] Gemini empty candidate; "
                        "retrying the API call instead of returning not_found."
                    )
                    if attempt < self.api_retry_attempts:
                        time.sleep(2 * attempt)
                        continue
                break

        if captured_chunks:
            chunks = list(captured_chunks.values())
            captured_ids = [chunk.chunk_id for chunk in chunks]
            fallback_output = (
                "Agent 3 không hoàn tất được câu trả lời theo định dạng ReAct, "
                "nhưng tool đã thu được các structured chunks. Agent 4 phải "
                "đánh giá trực tiếp nội dung các chunk này; không xem đây là "
                "kết luận không tìm thấy.\n"
                f"CHUNK IDS ĐÃ THU ĐƯỢC: {captured_ids}"
            )
            print("\n[Agent 3 Structured] CÂU TRẢ LỜI AGENT 3 (FALLBACK):")
            print(fallback_output)
            return SearchResult(
                status="retrieved_partial",
                message=(
                    "Đã thu được structured chunks nhưng Agent 3 không hoàn "
                    f"tất câu trả lời: {last_error}"
                ),
                search_query=query,
                agent_output=fallback_output,
                chunks=chunks,
                cited_chunk_ids=[],
                fact_requirements=list(self.legal_tool.fact_requirements),
                required_facts=list(self.legal_tool.required_facts),
                raw_chunks_text="\n\n".join(captured_raw_observations),
            )

        return SearchResult(
            status="error",
            message=f"Agent 3 API/system error: {last_error}",
            search_query=query,
            agent_output="",
            chunks=[],
            cited_chunk_ids=[],
            fact_requirements=normalize_fact_requirements(
                fact_requirements,
                required_facts,
            ),
            required_facts=fact_descriptions(
                normalize_fact_requirements(
                    fact_requirements,
                    required_facts,
                )
            ),
            raw_chunks_text=None,
        )


EVALUATOR_SYSTEM_RULES = """Bạn là Agent 4, chuyên gia QA cho hệ thống RAG
pháp luật. Bạn đánh giá và CHỌN CHUNK ID; bạn không viết lại evidence.

===============================================================================
1. KHÓA NGUỒN DỮ LIỆU (CORPUS LOCK)
===============================================================================
- TOÀN BỘ ACCUMULATED STRUCTURED CHUNKS là nguồn pháp lý duy nhất của lần đánh
  giá. Corpus có thể mới hơn dữ liệu huấn luyện của bạn.
- Không dùng trí nhớ bên ngoài để phủ nhận văn bản trong corpus, và không chèn
  văn bản, ngưỡng, thuế suất hay điều kiện không có trong chunk.
- Câu trả lời và Chunk ID Agent 3 trích dẫn chỉ là gợi ý. Phải tự đọc nội dung
  gốc của tất cả chunk trước khi chọn.
- Có quyền cứu chunk Agent 3 bỏ sót nếu chunk đó trực tiếp chứng minh một mắt
  xích cần thiết. Không được sửa, tóm tắt lại hay tự tạo nội dung chunk.

===============================================================================
2. KHÓA NGỮ CẢNH (CONTEXT LOCK)
===============================================================================
Trước khi đánh giá, xác định từ CÂU HỎI GỐC:
- đối tượng và trạng thái cư trú;
- loại thu nhập/hoạt động;
- loại thuế hoặc nghĩa vụ đang hỏi;
- thông tin cần tìm: phân loại, ngưỡng, thuế suất, căn cứ tính, điều kiện hay
  thủ tục.

Chunk chỉ liên quan khi phù hợp các neo cần thiết trên. Trùng từ "thuế suất",
trùng con số hoặc trùng Điều nhưng khác loại thu nhập không được xem là khớp.
Phần BỐI CẢNH CÂU HỎI GỐC có ưu tiên cao hơn nhiệm vụ Planner tự thêm.

===============================================================================
3. CHỌN VÀ LOẠI CHUNK ID
===============================================================================
- relevant_chunk_ids phải là danh sách ĐẦY ĐỦ cần đưa cho Agent 5, không chỉ là
  chunk có đáp án cuối cùng.
- Giữ cả chuỗi căn cứ khi cần: chunk phân loại/khái niệm, chunk dẫn chiếu, chunk
  thuế suất/cách tính và relation sửa đổi đúng phạm vi.
- Không loại chunk phân loại chỉ vì nó chưa có thuế suất nếu chunk đó là mắt
  xích chứng minh quy định thuế suất áp dụng đúng chủ đề.
- Chỉ giữ chunk nền/phạm vi khi Agent 5 cần nó để hiểu hoặc áp dụng chunk chính.
- Chunk relevant từ lượt trước mặc định được giữ. Muốn loại, phải đưa ID vào
  excluded_chunk_ids và ghi lý do cùng vị trí trong exclusion_reasons: sai đối
  tượng, sai chủ đề, trùng lặp, bị thay thế, hoặc không còn cần cho chuỗi căn cứ.
- Hai mảng excluded_chunk_ids và exclusion_reasons phải có cùng số phần tử.
- relevant_chunk_ids và excluded_chunk_ids CHỈ được dùng ID xuất hiện trong input.
- Không đưa cùng một ID vào cả relevant_chunk_ids và excluded_chunk_ids.

===============================================================================
4. KIỂM TRA CHẤT LƯỢNG PHÁP LÝ
===============================================================================
- Mỗi kết luận của Agent 3 phải có chunk chứng minh. Nội dung không được chunk
  chứng minh là quality issue, không phải lý do tự động loại các chunk đúng.
- Phân biệt các loại ngưỡng: miễn/không phải nộp thuế; lựa chọn phương pháp;
  hóa đơn/kê khai. Không dùng ngưỡng thủ tục để kết luận nghĩa vụ thuế.
- Với cách tính, cần đủ căn cứ tính thuế và thuế suất đúng loại hoạt động. Không
  tự suy ra doanh thu tính thuế là toàn bộ hay phần vượt ngưỡng.
- Nếu câu hỏi cần tính toán và Luật chỉ nêu "mức do Chính phủ quy định", chunk
  Luật đó CHƯA đủ để pass. Phải giữ thêm chunk Nghị định đang có hiệu lực chứa
  con số cụ thể như ngưỡng, mức trừ hoặc thời hạn. Không được loại chunk duy nhất
  chứa giá trị cụ thể chỉ vì chunk đó có cấp_hiệu_lực thấp hơn Luật.
- Nếu có amended_by_document_id, nội dung chunk đã là version được Step 3 tái
  tạo. Đánh giá trực tiếp nội dung đó và dùng provenance để truy vết nguồn sửa.
- Luật giao Chính phủ quy định chi tiết và Nghị định cung cấp giá trị cụ thể là
  quan hệ BỔ SUNG CĂN CỨ, không phải mâu thuẫn. Trong trường hợp này phải giữ cả
  chunk Luật và chunk Nghị định; legal_rank không được dùng để loại Nghị định.
- Tương tự, Nghị định giao Bộ trưởng/cơ quan hướng dẫn và Thông tư quy định chi
  tiết đúng phạm vi được giao là quan hệ BỔ SUNG CĂN CỨ. Phải giữ cả Nghị định
  và Thông tư. Chỉ áp dụng NGHỊ ĐỊNH > THÔNG TƯ nếu hai nội dung mâu thuẫn trực
  tiếp sau khi đã kiểm tra đúng đối tượng, vấn đề, điều kiện và thời điểm áp dụng.
- Nếu chunk Nghị định hoặc Thông tư có amended_by_document_id, nội_dung_gốc đã
  chứa giá trị của version mới. Phải giữ chunk khi nội dung hiện hành cung cấp
  biến pháp lý cần cho câu trả lời.
- Kiểm tra chính xác từng cấp Điểm/Khoản/Điều. Không tự đoán Điểm hoặc đổi Khoản.
- Chỉ xác định mâu thuẫn trực tiếp khi các chunk cùng điều chỉnh: đúng đối tượng
  và trạng thái cư trú; đúng loại thu nhập/hoạt động; cùng nghĩa vụ hoặc vấn đề
  pháp lý; cùng điều kiện; cùng thời điểm áp dụng; nhưng đưa ra kết quả không thể
  đồng thời đúng. Trước đó phải loại trừ trường hợp một quy định là ngoại lệ,
  điều khoản chuyển tiếp, nội dung sửa đổi hoặc thuộc phương pháp tính khác.
- Khi đã xác định mâu thuẫn trực tiếp theo các tiêu chí trên, bắt buộc áp dụng
  thứ bậc hiệu lực: LUẬT cao hơn NGHỊ ĐỊNH, NGHỊ ĐỊNH cao hơn THÔNG TƯ. Văn bản
  hợp nhất của Luật được đánh giá theo cấp của Luật gốc.
- Mỗi chunk có metadata loại_văn_bản và cấp_hiệu_lực. Khi giải quyết mâu thuẫn,
  BẮT BUỘC so sánh cấp_hiệu_lực: số lớn hơn có hiệu lực pháp lý cao hơn. Không
  tự đoán lại cấp văn bản nếu metadata đã cung cấp.
- Khi giải quyết mâu thuẫn trực tiếp, retrieval score, thứ tự xuất hiện, mức độ
  chi tiết và kết luận của Agent 3 KHÔNG được dùng để ưu tiên chunk có
  cấp_hiệu_lực thấp hơn.
- Văn bản cấp dưới mới hơn hoặc quy định chi tiết hơn KHÔNG được tự động ưu tiên
  nếu nội dung trái với văn bản cấp trên. Không loại chunk cấp trên chỉ để giữ
  kết luận thuận tiện từ văn bản cấp dưới.
- Nếu chưa chắc hai quy định có cùng đối tượng, loại thu nhập/hoạt động, vấn đề
  pháp lý, điều kiện, phương pháp tính hoặc thời điểm áp dụng, phải giữ cả hai và
  retry để tìm phạm vi áp dụng, hiệu lực, chuyển tiếp hoặc quan hệ sửa đổi. Nếu
  đã xác định mâu thuẫn trực tiếp, giữ căn cứ cấp trên và loại căn cứ cấp dưới
  với lý do rõ.
- Khi hai chuỗi căn cứ mâu thuẫn về cùng một biến pháp lý như thuế suất, ngưỡng,
  căn cứ tính hoặc thời hạn, KHÔNG được tạo chuỗi evidence hỗn hợp bằng cách lấy
  biến đó từ văn bản cấp thấp và lấy các thành phần còn lại từ văn bản cấp cao.
  CHỈ SAU KHI đã chứng minh có mâu thuẫn trực tiếp mới chọn biến pháp lý từ văn
  bản có cấp_hiệu_lực cao hơn. Nếu Luật giao Chính phủ quy định chi tiết thì lấy
  giá trị từ Nghị định đang có hiệu lực; nếu Nghị định giao Bộ trưởng/cơ quan
  hướng dẫn thì lấy chi tiết từ Thông tư đúng phạm vi. Đây không phải chuỗi
  evidence hỗn hợp mâu thuẫn.
- Nếu relevant_chunk_ids giữ chunk cấp thấp nhưng excluded_chunk_ids loại chunk
  cấp cao đang quy định cùng đối tượng và cùng vấn đề, quyết định pass là KHÔNG
  HỢP LỆ, trừ khi reason chỉ rõ chunk cấp cao thuộc ngoại lệ, phương pháp, thời
  điểm hoặc phạm vi khác và trích đúng nội dung chứng minh sự khác biệt đó.
- Ví dụ: cùng là thuế suất nhượng quyền thương mại của cá nhân cư trú, Luật có
  cấp_hiệu_lực=3 quy định 5%, Nghị định có cấp_hiệu_lực=2 quy định 10%. Nếu không
  có điều khoản sửa đổi, chuyển tiếp, ngoại lệ hoặc khác phương pháp thì phải giữ
  chunk Luật 5%, loại chunk Nghị định 10%; không được chọn 10% vì Nghị định đứng
  trước hoặc có retrieval score cao hơn.
- Tương tự, nếu Nghị định quy định 5% nhưng Thông tư quy định 10% cho đúng cùng
  đối tượng, vấn đề, điều kiện và thời điểm mà không có căn cứ ngoại lệ, phải giữ
  Nghị định 5% và loại Thông tư 10%.
- Ví dụ BỔ SUNG, không mâu thuẫn: Khoản 1 Điều 7 Luật quy định mức doanh thu do
  Chính phủ quy định; version hiện hành của Khoản 1 Điều 4 Nghị định
  68/2026/NĐ-CP đã được Nghị định 141/2026/NĐ-CP sửa thành 01 tỷ đồng. Phải giữ
  cả chunk Luật và chunk Nghị định 68, áp dụng giá trị mới 01 tỷ đồng.
  Không được loại chunk Nghị định 68 chỉ vì legal_rank=2 thấp hơn Luật=3.
- Status error/lỗi API/lỗi Agent 3 không có nghĩa là không có quy định. Nếu còn
  lượt tìm, BẮT BUỘC retry với query rõ ràng.
- Câu "chưa tìm thấy trong các chunk đã tra cứu" chỉ mô tả kết quả retrieval,
  KHÔNG phải căn cứ để kết luận pháp luật chưa có quy định.
- Nếu Agent 3 nói "không có/chưa có quy định" nhưng không có chunk nào khẳng
  định trực tiếp điều đó, ghi đây là quality issue và chọn retry.

===============================================================================
5. QUYẾT ĐỊNH
===============================================================================
- pass: chuỗi relevant chunks đủ trả lời CÂU HỎI GỐC, đúng đối tượng, chủ đề,
  căn cứ và không còn thông tin pháp lý thiết yếu bị thiếu.
- retry: CHỈ dùng khi coverage có status=missing_evidence, nghĩa là chưa có
  chunk phù hợp để chứng minh fact.
- Nếu chunk phù hợp đã có nhưng support_span sai, đây là lỗi đánh giá của
  Agent 4. Không tạo query tìm kiếm mới; hệ thống sẽ yêu cầu Agent 4 đánh giá
  lại trên chính evidence hiện có.
- not_applicable chỉ hợp lệ khi chunk và support_span chứng minh trực tiếp rằng
  fact không áp dụng cho tình huống.
- stop trước khi đạt giới hạn retry: CHỈ được dùng khi một chunk trong input
  khẳng định trực tiếp rằng trường hợp đang hỏi không được quy định/không thuộc
  phạm vi áp dụng. Phải giữ chunk đó trong relevant_chunk_ids và nêu Chunk ID
  trong reason.
- stop khi đã đạt giới hạn retry: được dùng nếu các hướng tìm khác nhau đều đã
  thất bại; reason phải tóm tắt các hướng đã thử.
- Không pass chỉ vì Agent 3 viết một kết luận tự tin.
- Khi retry_count còn nhỏ hơn max_retries và relevant_chunk_ids rỗng, mặc định
  phải chọn retry, KHÔNG chọn stop.
- Top-k trả về toàn chunk sai chủ đề là dấu hiệu query retrieval chưa tốt, không
  phải bằng chứng rằng corpus không có đáp án.

===============================================================================
6. TẠO RETRY QUERY
===============================================================================
- Mỗi query mới phải giữ: đối tượng + cư trú nếu có + loại thu nhập/hoạt động +
  thông tin còn thiếu.
- Không lặp nguyên văn query đã thử và không đổi chủ đề.
- Chỉ thêm Điểm/Khoản/Điều khi chunk chứng minh vị trí đó thuộc đúng chủ đề.
- Giữ ĐẦY ĐỦ cấp vị trí. Không đổi Khoản và không tự đoán Điểm.
- Nếu chunk phân loại dẫn chiếu, query theo mẫu:
  [thông tin còn thiếu] + [chủ đề gốc] + [đối tượng] +
  "quy định tại" + [đầy đủ Điểm/Khoản/Điều].
- Nếu chưa có vị trí đúng, tìm theo thông tin còn thiếu + chủ đề + đối tượng;
  có thể đổi thuật ngữ hẹp/rộng nhưng không bỏ neo ngữ nghĩa.
- Khi chưa tìm thấy bất kỳ chunk liên quan nào, lần lượt thử các hướng sau:
  1. [phân loại pháp lý/thu nhập chịu thuế] + [chủ đề] + [tên luật nếu có];
  2. [thuế suất/căn cứ tính/cách tính] + [chủ đề] + [đối tượng];
  3. [nhóm pháp lý rộng hơn] + [chủ đề gốc], nhưng không đổi sang loại thu nhập
     khác chỉ vì có cùng từ "chuyển nhượng" hoặc cùng một mức thuế suất.
- suggested_retry_queries là BẮT BUỘC và phải có từ 1 đến 3 query khi decision
  là retry. Các query phải khác query vừa thử và khác nhau về hướng retrieval.
- Không được lặp lại nguyên văn một suggested_retry_query đã xuất hiện trong
  trường retrieval_query của các accumulated chunks.

Ví dụ khi retrieval tài sản số chỉ trả chuyển nhượng vốn/bất động sản:
- SAI: stop và kết luận "chưa có quy định về tài sản số".
- ĐÚNG: retry "thu nhập từ chuyển nhượng tài sản số thuộc loại thu nhập chịu
  thuế nào Điều 3 Luật Thuế thu nhập cá nhân".
- Sau khi có vị trí phân loại, tiếp tục truy vết thuế suất bằng đầy đủ vị trí đó.

Ví dụ truy vết đúng:
- Câu hỏi: thuế suất chuyển nhượng tài sản số của cá nhân cư trú.
- Chunk phân loại: điểm d khoản 10 Điều 3, chưa có thuế suất.
- Retry: "thuế suất chuyển nhượng tài sản số cá nhân cư trú quy định tại điểm d
  khoản 10 Điều 3". Mục tiêu là tìm Điều khác dẫn chiếu chính xác về vị trí này.

Ví dụ sai:
- "thuế suất điểm d Điều 3" vì mất chủ đề và Khoản 10.
- Đổi "kinh doanh thương mại điện tử" thành "chuyển nhượng tài sản số".

Trước khi trả JSON, tự kiểm tra:
- relevant IDs đã đủ cả chuỗi căn cứ chưa?
- Nếu Luật/Nghị định giao quy định chi tiết, relevant IDs đã giữ đúng Nghị định
  hoặc Thông tư chứa giá trị đang có hiệu lực và mắt xích chứng minh phạm vi
  hướng dẫn chưa? Nếu chưa thì không được pass.
- Có đang giữ chunk cấp thấp nhưng loại chunk cấp cao cùng vấn đề không? Nếu có,
  pass chỉ hợp lệ khi reason chứng minh chúng khác phạm vi hoặc thuộc ngoại lệ.
- Chuỗi evidence có đang ghép biến pháp lý từ hai văn bản mâu thuẫn không?
- ID bị loại đã có lý do chưa?
- retry query còn trả lời đúng CÂU HỎI GỐC không?
"""


class SearchEvaluator:
    """Agent 4 selects IDs; it never creates ChunkEvidence."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY khong ton tai trong .env")
        genai.configure(api_key=api_key)
        self.model_name = os.getenv(
            "AGENT4_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite"),
        )
        self.llm = genai.GenerativeModel(self.model_name)
        print(
            "[Agent 4 Structured] LLM Evaluator san sang; "
            f"model={self.model_name}"
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value)).strip().casefold()

    @classmethod
    def _valid_support_span(cls, support_span: str, source_text: str) -> bool:
        """Accept only one exact, contiguous excerpt from the cited chunk."""

        span = str(support_span or "").strip()
        if not span or "..." in span or "…" in span:
            return False
        return cls._normalize_text(span) in cls._normalize_text(source_text)

    @staticmethod
    def _format_chunk(chunk: StructuredChunk, label: str) -> str:
        return (
            f"\n[{label}]\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"provision_key: {chunk.provision_key}\n"
            f"retrieval_query: {chunk.query}\n"
            f"văn_bản: {chunk.van_ban}\n"
            f"loại_văn_bản: {chunk.document_type}\n"
            f"cấp_hiệu_lực: {chunk.legal_rank}\n"
            f"điều_khoản: {chunk.dieu_khoan}\n"
            f"tình_trạng: {chunk.tinh_trang_hieu_luc}\n"
            f"phiên_bản: {chunk.version}\n"
            f"valid_from: {chunk.valid_from}\n"
            f"valid_to: {chunk.valid_to}\n"
            f"nội_dung_gốc:\n{chunk.noi_dung_goc}\n"
            f"amended_by_document_id: {chunk.amended_by_document_id}\n"
            f"amended_by_source_location: {chunk.amended_by_source_location}\n"
            f"change_type: {chunk.change_type}\n"
        )

    @staticmethod
    def _temporal_gate_issues(
        chunks: List[StructuredChunk],
        temporal_scope: Dict,
    ) -> List[str]:
        normalized = normalize_temporal_scope(temporal_scope)
        issues: List[str] = []
        versions_by_key: Dict[str, Set[int]] = {}
        for chunk in chunks:
            metadata = {
                "valid_from": chunk.valid_from,
                "valid_to": chunk.valid_to,
            }
            if not metadata_overlaps_scope(metadata, normalized):
                issues.append(
                    f"Chunk {chunk.chunk_id} nằm ngoài phạm vi "
                    f"{describe_temporal_scope(normalized)}."
                )
            if chunk.provision_key:
                versions_by_key.setdefault(chunk.provision_key, set()).add(
                    int(chunk.version or 1)
                )

        if normalized["type"] in {"current", "exact_date"}:
            for provision_key, versions in versions_by_key.items():
                if len(versions) > 1:
                    issues.append(
                        f"Temporal point query có nhiều version của "
                        f"{provision_key}: {sorted(versions)}."
                    )
        return issues

    def evaluate(
        self,
        search_result: SearchResult,
        original_query: str,
        retry_count: int,
        accumulated_chunks: Dict[str, StructuredChunk],
        previous_relevant_ids: Set[str],
        previous_excluded: Dict[str, str],
        temporal_scope: Optional[Dict] = None,
        fact_requirements: Optional[List[Dict[str, str]]] = None,
        required_facts: Optional[List[str]] = None,
        repair_feedback: Optional[str] = None,
    ) -> dict:
        normalized_scope = normalize_temporal_scope(
            temporal_scope or {"type": "current"}
        )
        print(
            f"\n[Agent 4 Structured] DANH GIA "
            f"(Retry {retry_count}/{self.max_retries})"
        )

        previous_relevant_chunks = [
            accumulated_chunks[chunk_id]
            for chunk_id in previous_relevant_ids
            if chunk_id in accumulated_chunks
        ]
        previous_relevant_set = {chunk.chunk_id for chunk in previous_relevant_chunks}
        # Always show full current retrieval context, even if IDs were excluded in
        # prior turns. Exclusion registry is history, not a permanent blacklist.
        current_chunks = list(search_result.chunks)
        temporal_issues = self._temporal_gate_issues(
            previous_relevant_chunks + current_chunks,
            normalized_scope,
        )
        visible_ids = previous_relevant_set | {
            chunk.chunk_id for chunk in current_chunks
        }
        fact_requirements = normalize_fact_requirements(
            fact_requirements or search_result.fact_requirements,
            required_facts or search_result.required_facts,
        )
        required_facts = fact_descriptions(fact_requirements)

        previous_relevant_text = "".join(
            self._format_chunk(chunk, f"Previous Relevant {index}")
            for index, chunk in enumerate(previous_relevant_chunks, 1)
        )
        current_chunk_text = "".join(
            self._format_chunk(chunk, f"Current Candidate {index}")
            for index, chunk in enumerate(current_chunks, 1)
        )
        print(
            "[Agent 4 Structured] Context: "
            f"{len(previous_relevant_chunks)} relevant cũ + "
            f"{len(current_chunks)} chunk hiện tại; "
            f"ẩn {len(accumulated_chunks) - len(visible_ids)} chunk tích lũy cũ"
        )
        prompt = f"""{EVALUATOR_SYSTEM_RULES}

CÂU HỎI GỐC:
{original_query}

TYPED FACT REQUIREMENTS DO AGENT 2 XÁC ĐỊNH:
{json.dumps(fact_requirements, ensure_ascii=False, indent=2)}

PHẠM VI THỜI GIAN DO PYTHON KHÓA:
{json.dumps(normalized_scope, ensure_ascii=False)}
Diễn giải: {describe_temporal_scope(normalized_scope)}

TEMPORAL GATE DO PYTHON KIỂM TRA:
{json.dumps(temporal_issues, ensure_ascii=False) if temporal_issues else 'PASS - không phát hiện chunk sai khoảng hiệu lực'}

Quy tắc thời gian:
- current/exact_date: một provision_key chỉ được giữ một version bao phủ ngày.
- month/year/date_range: được giữ nhiều version cùng provision_key nếu từng
  version giao với khoảng hỏi; không xem chúng là trùng lặp chỉ vì cùng vị trí.
- Không dùng issued_date để thay valid_from/valid_to.
- Không pass nếu TEMPORAL GATE báo lỗi.

AGENT 3 STATUS: {search_result.status}
AGENT 3 MESSAGE: {search_result.message}
QUERY AGENT 3 VỪA TRA CỨU: {search_result.search_query}
CÂU TRẢ LỜI CỦA AGENT 3:
{search_result.agent_output or '(không có)'}

CHUNK IDS AGENT 3 ĐÃ TRÍCH DẪN:
{json.dumps(search_result.cited_chunk_ids, ensure_ascii=False)}

PREVIOUS RELEVANT IDS:
{json.dumps(sorted(previous_relevant_ids), ensure_ascii=False)}

PREVIOUS EXCLUDED REGISTRY:
{json.dumps(previous_excluded, ensure_ascii=False, indent=2)}

CONFIRMED RELEVANT CHUNKS TỪ CÁC LƯỢT TRƯỚC:
{previous_relevant_text or '(không có)'}

CURRENT TURN STRUCTURED CHUNKS:
{current_chunk_text or '(không có)'}

Các chunk tích lũy cũ không relevant không được đưa lại vào prompt.
Lưu ý: chunk từng excluded ở lượt trước nếu được Agent 3 tìm lại trong
CURRENT TURN STRUCTURED CHUNKS thì vẫn hiển thị đầy đủ nội dung để Agent 4 có
thể chọn lại khi cần.

RETRY COUNT: {retry_count}/{self.max_retries}

{f"YÊU CẦU SỬA ĐÁNH GIÁ TRƯỚC: {repair_feedback}" if repair_feedback else ""}

Trả JSON theo EvaluationDecision. relevant_chunk_ids phải là danh sách đầy đủ
evidence cần giữ. Nếu loại relevant ID cũ, bắt buộc đưa ID đó vào
excluded_chunk_ids và đưa lý do tại cùng vị trí trong exclusion_reasons.
coverage phải có đúng một mục cho mỗi typed fact requirement. `fact_type` và
`required_fact` phải giữ nguyên chính xác `type` và `description` tương ứng;
`fact_id` và `task_id` cũng phải giữ nguyên.
Không được dùng evidence của một type khác để đánh dấu covered: ví dụ chunk chỉ
nêu đối tượng áp dụng không đủ cho tax_rate; chunk chỉ nêu thời hạn không đủ cho
procedure.

Mỗi coverage chỉ được dùng một trong ba status:
- covered: evidence trực tiếp chứng minh fact;
- missing_evidence: chưa có chunk thích hợp, cần Agent 3 tìm thêm;
- not_applicable: evidence trực tiếp chứng minh fact không áp dụng.

Với status=covered hoặc not_applicable, chunk_id phải xuất hiện trong input và
support_span phải là một đoạn ngắn sao chép nguyên văn từ nội_dung_gốc của
chính chunk đó. support_span phải là MỘT chuỗi ký tự liên tục: tuyệt đối không
ghép nhiều câu/đoạn rời bằng "...", "…" hoặc lời lược bỏ. Nếu một fact gồm nhiều
điều kiện trong cùng Khoản, có thể trích cả dải văn bản liên tục chứa các điều
kiện và phần nằm giữa chúng. Không được diễn giải support_span. Chỉ đặt
missing_evidence khi thực sự thiếu chunk; không dùng missing_evidence để che lỗi
trích support_span.
Không giải thích ngoài JSON.
"""

        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=EvaluationDecision,
                    temperature=0.0,
                ),
            )
            print("\n[Agent 4 Structured] RAW OUTPUT:")
            print(response.text)
            decision = EvaluationDecision(**json.loads(response.text))
        except Exception as exc:
            print(f"[Agent 4 Structured] LLM error: {exc}")
            cited = [
                chunk_id
                for chunk_id in search_result.cited_chunk_ids
                if chunk_id in visible_ids
            ]
            return {
                "decision": "stop",
                "reason": f"Agent 4 LLM error: {exc}",
                "quality_issues": ["LLM evaluation failed"],
                "relevant_chunk_ids": sorted(previous_relevant_ids | set(cited)),
                "excluded_chunks": [],
                "missing_info": None,
                "retry_queries": [],
                "covered_facts": [],
                "uncovered_facts": list(required_facts),
                "covered_fact_requirements": [],
                "uncovered_fact_requirements": list(fact_requirements),
            }

        # Agent 4 may only select or exclude IDs whose full content appears in
        # this prompt. Hidden accumulated/excluded chunks cannot be revived by
        # an accidental ID-only selection.
        valid_ids = visible_ids
        selected = {
            chunk_id
            for chunk_id in decision.relevant_chunk_ids or []
            if chunk_id in valid_ids
        }
        excluded_ids = decision.excluded_chunk_ids or []
        exclusion_reasons = decision.exclusion_reasons or []
        exclusions = []
        for index, chunk_id in enumerate(excluded_ids):
            if chunk_id not in valid_ids:
                continue
            reason = (
                exclusion_reasons[index]
                if index < len(exclusion_reasons)
                else "Agent 4 không cung cấp lý do loại chunk."
            )
            exclusions.append(
                ExcludedChunkDecision(chunk_id=chunk_id, reason=reason)
            )
        explicitly_excluded = {item.chunk_id for item in exclusions}

        # Previous relevant chunks survive unless Agent 4 explicitly excludes them.
        selected |= previous_relevant_ids - explicitly_excluded
        selected -= explicitly_excluded

        final_decision = decision.decision.lower().strip()
        retry_queries = decision.suggested_retry_queries or []
        reason = decision.reason
        missing_info = decision.missing_info
        quality_issues = list(decision.quality_issues or [])

        coverage_by_id = {
            str(item.fact_id or "").strip(): item
            for item in decision.coverage
            if str(item.fact_id or "").strip()
        }
        coverage_by_fact = {
            (
                str(item.fact_type or "unspecified").strip().casefold(),
                self._normalize_text(item.required_fact),
            ): item
            for item in decision.coverage
        }
        uncovered_facts: List[str] = []
        uncovered_fact_requirements: List[Dict[str, str]] = []
        invalid_coverage_requirements: List[Dict[str, str]] = []
        not_applicable_requirements: List[Dict[str, str]] = []
        for requirement in fact_requirements:
            fact = requirement["description"]
            coverage = coverage_by_id.get(
                str(requirement.get("fact_id") or "")
            ) or coverage_by_fact.get(
                fact_requirement_key(requirement)
            )
            if coverage is None:
                invalid_coverage_requirements.append(requirement)
                uncovered_facts.append(fact)
                uncovered_fact_requirements.append(requirement)
                continue

            status = coverage.status.lower().strip()
            if status == "missing_evidence":
                uncovered_facts.append(fact)
                uncovered_fact_requirements.append(requirement)
                continue
            if status not in {"covered", "not_applicable"}:
                uncovered_facts.append(fact)
                uncovered_fact_requirements.append(requirement)
                invalid_coverage_requirements.append(requirement)
                continue

            chunk_id = str(coverage.chunk_id or "")
            support_span = str(coverage.support_span or "").strip()
            chunk = accumulated_chunks.get(chunk_id)
            if (
                chunk_id not in visible_ids
                or chunk_id in explicitly_excluded
                or chunk is None
                or not self._valid_support_span(
                    support_span,
                    chunk.noi_dung_goc,
                )
            ):
                uncovered_facts.append(fact)
                uncovered_fact_requirements.append(requirement)
                invalid_coverage_requirements.append(requirement)
                quality_issues.append(
                    "Support span không hợp lệ cho "
                    f"{requirement.get('fact_id')}: {fact}"
                )
                continue
            selected.add(chunk_id)
            if status == "not_applicable":
                not_applicable_requirements.append(requirement)

        if invalid_coverage_requirements and not temporal_issues:
            missing_info = "; ".join(
                f"{item.get('fact_id')}: support span cần sửa"
                for item in invalid_coverage_requirements
            )
            retry_queries = []
            if repair_feedback:
                final_decision = "stop"
                reason = (
                    f"{reason} Agent 4 vẫn trả coverage/span không hợp lệ "
                    "sau lượt sửa đánh giá."
                )
            else:
                final_decision = "reevaluate"
                reason = (
                    f"{reason} Evidence đã có nhưng coverage/support span "
                    "không hợp lệ; chỉ đánh giá lại Agent 4."
                )
        elif uncovered_facts and not temporal_issues:
            final_decision = "retry"
            missing_info = "; ".join(
                f"{requirement.get('fact_id')}: "
                f"{requirement['type']}: {requirement['description']}"
                for requirement in uncovered_fact_requirements
            )
            if not retry_queries:
                retry_queries = [
                    (
                        f"{original_query}\n"
                        f"Loại fact còn thiếu: {requirement['type']}\n"
                        f"Thông tin còn thiếu: {requirement['description']}"
                    )
                    for requirement in uncovered_fact_requirements
                ]
            reason = (
                f"{reason} Python coverage gate còn thiếu "
                f"{len(uncovered_facts)} required fact."
            )
        elif not temporal_issues:
            final_decision = "pass"
            retry_queries = []
            missing_info = None
        if temporal_issues:
            quality_issues.extend(temporal_issues)
            final_decision = "stop"
            missing_info = "Retriever returned a version outside the locked temporal scope."
            retry_queries = []
            reason = f"{reason} Python temporal gate failed; retry cannot repair it."

        # Evaluate the final attempt first. Only prevent another retry afterwards.
        if final_decision == "retry" and retry_count >= self.max_retries:
            final_decision = "stop"
            retry_queries = []
            reason = (
                f"{reason} Đã đạt giới hạn {self.max_retries} retry."
            )

        if final_decision not in {"pass", "retry", "reevaluate", "stop"}:
            final_decision = "stop"
            reason = f"Invalid Agent 4 decision: {decision.decision}"

        print(f"[Agent 4 Structured] Decision: {final_decision}")
        print(f"[Agent 4 Structured] Relevant IDs: {sorted(selected)}")
        if required_facts:
            print(
                "[Agent 4 Structured] Required facts chưa đủ: "
                f"{uncovered_facts or 'không'}"
            )
        if exclusions:
            print(
                "[Agent 4 Structured] Excluded IDs: "
                f"{[item.chunk_id for item in exclusions]}"
            )

        return {
            "decision": final_decision,
            "reason": reason,
            "quality_issues": quality_issues,
            "relevant_chunk_ids": sorted(selected),
            "excluded_chunks": exclusions,
            "missing_info": missing_info,
            "retry_queries": retry_queries,
            "covered_facts": [
                fact for fact in required_facts if fact not in uncovered_facts
            ],
            "uncovered_facts": uncovered_facts,
            "covered_fact_requirements": [
                requirement
                for requirement in fact_requirements
                if requirement not in uncovered_fact_requirements
            ],
            "uncovered_fact_requirements": uncovered_fact_requirements,
            "invalid_coverage_requirements": invalid_coverage_requirements,
            "not_applicable_fact_requirements": not_applicable_requirements,
        }


def _build_evidences(
    accumulated_chunks: Dict[str, StructuredChunk],
    relevant_ids: Set[str],
) -> List[ChunkEvidence]:
    evidences = []
    for chunk_id in relevant_ids:
        chunk = accumulated_chunks.get(chunk_id)
        if chunk is None:
            continue
        evidences.append(
            ChunkEvidence(
                chunk_id=chunk.chunk_id,
                buoc_so=chunk.first_seen_step,
                van_ban=chunk.van_ban,
                document_id=chunk.document_id,
                provision_key=chunk.provision_key,
                dieu_khoan=chunk.dieu_khoan,
                tinh_trang_hieu_luc=chunk.tinh_trang_hieu_luc,
                valid_from=chunk.valid_from,
                valid_to=chunk.valid_to,
                version=chunk.version,
                noi_dung_goc=chunk.noi_dung_goc,
                cac_bien_so_phap_ly=None,
                amended_by_document_id=chunk.amended_by_document_id,
                amended_by_source_location=chunk.amended_by_source_location,
                change_type=chunk.change_type,
                temporal_scope=chunk.temporal_scope,
            )
        )
    return sorted(
        evidences,
        key=lambda evidence: (evidence.buoc_so, evidence.van_ban, evidence.dieu_khoan),
    )


class StructuredSearchMemory:
    """Shared structured memory for all Planner steps of one user query."""

    def __init__(self):
        self.accumulated_chunks: Dict[str, StructuredChunk] = {}
        self.relevant_ids: Set[str] = set()
        self.excluded_registry: Dict[str, str] = {}
        self.total_search_turns = 0


def search_with_retry(
    searcher: LegalSearcher,
    evaluator: SearchEvaluator,
    query: str,
    max_steps: int = 5,
    memory: Optional[StructuredSearchMemory] = None,
    temporal_scope: Optional[Dict] = None,
    fact_requirements: Optional[List[Dict[str, str]]] = None,
    required_facts: Optional[List[str]] = None,
) -> dict:
    """Structured Agent 3 + Agent 4 workflow, compatible with Planner."""

    print("\n" + "=" * 80)
    print("WORKFLOW: STRUCTURED CHUNKS WITH RETRY")
    print("=" * 80)
    normalized_scope = normalize_temporal_scope(
        temporal_scope or {"type": "current"}
    )
    searcher.set_temporal_scope(normalized_scope)
    fact_requirements = normalize_fact_requirements(
        fact_requirements,
        required_facts,
    )
    required_facts = fact_descriptions(fact_requirements)
    print(f"Query: {compact_query_label(query)}")
    print(f"Thời gian: {describe_temporal_scope(normalized_scope)}")
    print("Facts:")
    for requirement in fact_requirements:
        print(
            f"  - [{requirement.get('fact_id')}] "
            f"{requirement.get('type')}: "
            f"{requirement.get('description')}"
        )

    memory = memory or StructuredSearchMemory()
    accumulated_chunks = memory.accumulated_chunks
    relevant_ids = memory.relevant_ids
    excluded_registry = memory.excluded_registry
    quality_issues: List[str] = []
    search_history = []

    if relevant_ids:
        print(
            "[Workflow Structured] Memory từ các bước Planner trước: "
            f"{len(accumulated_chunks)} chunks, "
            f"{len(relevant_ids)} relevant IDs"
        )

    current_query = query
    missing_info: Optional[str] = None
    retry_count = 0

    for step in range(1, max_steps + 1):
        memory.total_search_turns += 1
        previous_relevant_chunks = [
            accumulated_chunks[chunk_id]
            for chunk_id in relevant_ids
            if chunk_id in accumulated_chunks
        ]

        search_result = searcher.search_and_structure(
            memory.total_search_turns,
            current_query,
            previous_relevant_chunks,
            missing_info,
            original_query=query,
            temporal_scope=normalized_scope,
            fact_requirements=fact_requirements,
            required_facts=required_facts,
        )

        new_chunk_ids = []
        for chunk in search_result.chunks:
            new_chunk_ids.append(chunk.chunk_id)
            if chunk.chunk_id not in accumulated_chunks:
                accumulated_chunks[chunk.chunk_id] = chunk

        eval_result = evaluator.evaluate(
            search_result=search_result,
            original_query=query,
            retry_count=retry_count,
            accumulated_chunks=accumulated_chunks,
            previous_relevant_ids=relevant_ids,
            previous_excluded=excluded_registry,
            temporal_scope=normalized_scope,
            fact_requirements=fact_requirements,
            required_facts=required_facts,
        )
        evaluator_repairs = 0
        while eval_result["decision"] == "reevaluate" and evaluator_repairs < 1:
            evaluator_repairs += 1
            repair_feedback = (
                "Giữ nguyên tập chunks hiện tại. Không đề xuất search query mới. "
                "Hãy trả lại coverage với đúng fact_id/task_id và một "
                "support_span nguyên văn, liên tục cho các fact sau: "
                + "; ".join(
                    f"{item.get('fact_id')}: {item.get('description')}"
                    for item in eval_result.get(
                        "invalid_coverage_requirements",
                        [],
                    )
                )
            )
            print(
                "[Workflow Structured] Agent 4 sửa coverage/span trên cùng "
                "evidence (không gọi lại Agent 3)."
            )
            eval_result = evaluator.evaluate(
                search_result=search_result,
                original_query=query,
                retry_count=retry_count,
                accumulated_chunks=accumulated_chunks,
                previous_relevant_ids=relevant_ids,
                previous_excluded=excluded_registry,
                temporal_scope=normalized_scope,
                fact_requirements=fact_requirements,
                required_facts=required_facts,
                repair_feedback=repair_feedback,
            )

        relevant_ids.clear()
        relevant_ids.update(eval_result["relevant_chunk_ids"])

        # Reactivate IDs if Agent 4 selected them again in a later turn.
        for chunk_id in list(relevant_ids):
            excluded_registry.pop(chunk_id, None)

        for excluded in eval_result["excluded_chunks"]:
            excluded_registry[excluded.chunk_id] = excluded.reason
            relevant_ids.discard(excluded.chunk_id)

        quality_issues.extend(eval_result["quality_issues"])
        search_history.append(
            {
                "step": step,
                "query": current_query,
                "agent3_status": search_result.status,
                "new_chunk_ids": new_chunk_ids,
                "agent3_cited_chunk_ids": search_result.cited_chunk_ids,
                "agent4_decision": eval_result["decision"],
                "agent4_span_repairs": evaluator_repairs,
                "agent4_reason": eval_result["reason"],
                "missing_info": eval_result["missing_info"],
                "covered_facts": eval_result.get("covered_facts", []),
                "uncovered_facts": eval_result.get("uncovered_facts", []),
                "covered_fact_requirements": eval_result.get(
                    "covered_fact_requirements",
                    [],
                ),
                "uncovered_fact_requirements": eval_result.get(
                    "uncovered_fact_requirements",
                    [],
                ),
                "relevant_chunk_ids": sorted(relevant_ids),
                "excluded_chunk_ids": sorted(excluded_registry),
            }
        )

        if eval_result["decision"] == "pass":
            return {
                "final_status": "success",
                "all_evidences": _build_evidences(
                    accumulated_chunks,
                    relevant_ids,
                ),
                "total_searches": step,
                "search_history": search_history,
                "quality_issues": quality_issues,
                "missing_info": None,
                "chunk_memory": {
                    "accumulated_chunk_ids": sorted(accumulated_chunks),
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "excluded_chunks": excluded_registry,
                },
            }

        if eval_result["decision"] == "stop":
            evidences = _build_evidences(accumulated_chunks, relevant_ids)
            return {
                "final_status": "partial" if evidences else "not_found",
                "all_evidences": evidences,
                "total_searches": step,
                "search_history": search_history,
                "quality_issues": quality_issues,
                "missing_info": eval_result["missing_info"],
                "chunk_memory": {
                    "accumulated_chunk_ids": sorted(accumulated_chunks),
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "excluded_chunks": excluded_registry,
                },
            }

        retry_queries = eval_result["retry_queries"]
        if not retry_queries:
            evidences = _build_evidences(accumulated_chunks, relevant_ids)
            return {
                "final_status": "partial" if evidences else "not_found",
                "all_evidences": evidences,
                "total_searches": step,
                "search_history": search_history,
                "quality_issues": quality_issues + [
                    "Agent 4 requested retry without a retry query"
                ],
                "missing_info": eval_result["missing_info"],
                "chunk_memory": {
                    "accumulated_chunk_ids": sorted(accumulated_chunks),
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "excluded_chunks": excluded_registry,
                },
            }

        current_query = retry_queries[0]
        missing_info = eval_result["missing_info"]
        retry_count += 1
        print(f"[Workflow Structured] Retry {retry_count}: {current_query}")

    evidences = _build_evidences(accumulated_chunks, relevant_ids)
    return {
        "final_status": "partial" if evidences else "not_found",
        "all_evidences": evidences,
        "total_searches": max_steps,
        "search_history": search_history,
        "quality_issues": quality_issues,
        "missing_info": missing_info,
        "chunk_memory": {
            "accumulated_chunk_ids": sorted(accumulated_chunks),
            "relevant_chunk_ids": sorted(relevant_ids),
            "excluded_chunks": excluded_registry,
        },
    }
