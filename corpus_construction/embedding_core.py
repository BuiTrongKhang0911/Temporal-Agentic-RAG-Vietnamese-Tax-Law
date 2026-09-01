"""Shared embedding and Qdrant support for the version-aware Step 3 pipeline.

This module is not a standalone corpus-construction step. The runnable Step 3
entry point is ``step3_embedding_qdrant_versioned.py``.

- Chunking: MarkdownHeaderTextSplitter (bảo toàn metadata phân cấp) + RecursiveCharacterTextSplitter
- SAC Tier 1: Document-level summary (từ step 2)
- SAC Tier 2: Chunk-level summary (batch 10 chunks/lần)
- Inject SAC Tier 1 + Tier 2 vào chunk
- Generate embeddings bằng BAAI/bge-m3 (multilingual, 8192 tokens)
- KHÔNG cần word segmentation (bge-m3 tự xử lý)
- Lưu trực tiếp vào Qdrant
"""
import os

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from typing import List, Dict, Optional
from datetime import date
import google.generativeai as genai
import hashlib
import json
import re
import time
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OUTPUT_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    GOOGLE_API_KEY,
    SAC_TIER2_GEMINI_MODEL
)


class EmbeddingQdrantProcessor:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("Thiếu GOOGLE_API_KEY trong file .env ở thư mục gốc")

        # Load HuggingFace model - bge-m3 (multilingual, 8192 tokens)
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Get embedding dimension từ config
        self.embedding_dimension = EMBEDDING_DIMENSION
        print(f"Embedding dimension: {self.embedding_dimension}")
        
        # Gemini model cho SAC Tier 2
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(SAC_TIER2_GEMINI_MODEL)
        
        # Qdrant client
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        
        # Chunking settings
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        
        # SAC Tier 2 batch size
        self.tier2_batch_size = 10
        
        # MarkdownHeaderTextSplitter - tự động kế thừa metadata phân cấp
        self.headers_to_split_on = [
            ("#", "Chuong"),
            ("##", "Dieu"),
            ("###", "Khoan"),
        ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # RecursiveCharacterTextSplitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
        print(f"SAC Tier 2 batch size: {self.tier2_batch_size}")
    
    def extract_location_label(self, hierarchical_metadata: dict) -> str:
        """
        Extract location label từ hierarchical metadata
        Hỗ trợ: Chương, Điều, Khoản, Phụ lục, Mục
        
        Input: {'Chuong': 'Chương II: CĂN CỨ...', 'Dieu': 'Điều 10. Giảm trừ...', 'Khoan': '1. Mức giảm...'}
        Output: 'Chương II > Điều 10 > Khoản 1'
        
        Input: {'Chuong': 'Phụ lục I: DANH MỤC...', 'Dieu': 'Mục II. Việc xác định...'}
        Output: 'Phụ lục I > Mục II'
        """
        parts = []
        
        # Kiểm tra nếu "Chuong" thực ra là Phụ lục
        chuong_val = hierarchical_metadata.get('Chuong', '')
        if 'Phụ lục' in chuong_val or 'PHỤ LỤC' in chuong_val:
            # Đây là Phụ lục, không phải Chương
            match = re.search(r'Phụ lục\s+([IVXLCDM\d]+)', chuong_val, re.IGNORECASE)
            if match:
                parts.append(f"Phụ lục {match.group(1)}")
        else:
            # Chương bình thường
            if chuong_val:
                match = re.search(r'Chương\s+([IVXLCDM]+)', chuong_val)
                if match:
                    parts.append(f"Chương {match.group(1)}")
        
        # Kiểm tra nếu "Dieu" thực ra là Mục (trong Phụ lục)
        dieu_val = hierarchical_metadata.get('Dieu', '')
        if 'Mục' in dieu_val or 'MỤC' in dieu_val:
            # Đây là Mục, không phải Điều
            match = re.search(r'Mục\s+([IVXLCDM\d]+)', dieu_val, re.IGNORECASE)
            if match:
                parts.append(f"Mục {match.group(1)}")
        else:
            # Điều bình thường
            if dieu_val:
                match = re.search(r'Điều\s+(\d+)', dieu_val)
                if match:
                    parts.append(f"Điều {match.group(1)}")
        
        # Extract Khoản (Arabic numbers)
        # Lưu ý: khoan_val có dạng "Khoản 4. Content" (do step1 tạo heading)
        khoan_val = hierarchical_metadata.get('Khoan', '')
        if khoan_val:
            match = re.search(r'Khoản\s+(\d+)', khoan_val, re.IGNORECASE)
            if match:
                parts.append(f"Khoản {match.group(1)}")
        
        return " > ".join(parts) if parts else "Không xác định"
    
    def generate_tier2_summaries_batch(self, chunks_batch: List[Dict]) -> List[Dict]:
        """
        Generate SAC Tier 2 summaries cho một batch chunks (batch=10)
        
        Returns: List of tier2 summaries với cùng số lượng như input
        """
        batch_size = len(chunks_batch)
        
        # Build prompt với CHÍNH XÁC batch_size chunks
        prompt_parts = [
            f"You are a Vietnamese legal text analyzer. Your task is to analyze EXACTLY {batch_size} legal text chunks and return a JSON array with EXACTLY {batch_size} objects.",
            "",
            "CRITICAL RULES:",
            f"1. Return EXACTLY {batch_size} objects in the JSON array",
            "2. DO NOT merge multiple chunks into one summary",
            "3. DO NOT skip any chunk",
            '4. Each object must have these 3 keys: "doi_tuong", "noi_dung", "so_lieu_quy_dinh"',
            "5. Output ONLY valid JSON array, no explanation, no markdown code blocks",
            "",
            "For each chunk, extract:",
            '- "doi_tuong": Who/what entity (người nộp thuế, doanh nghiệp, cá nhân cư trú, etc.)',
            '- "noi_dung": Main regulation content (what is being regulated)',
            '- "so_lieu_quy_dinh": Specific numbers/rates/provisions (15.5 triệu, 5-35%, miễn thuế 5 năm, etc.)',
            "",
            "INPUT CHUNKS:",
            ""
        ]
        
        # Thêm từng chunk vào prompt
        for i, chunk_data in enumerate(chunks_batch, 1):
            location = chunk_data['location_label']
            # Limit 800 chars để:
            # 1. Tránh vượt quá context window của LLM (Gemini có giới hạn)
            # 2. SAC Tier 2 chỉ cần summary ngắn gọn, không cần toàn bộ chunk
            # 3. Giảm chi phí API call (ít tokens hơn)
            text = chunk_data['original_content'][:800]
            
            prompt_parts.append(f"--- CHUNK {i} ---")
            prompt_parts.append(f"Location: {location}")
            prompt_parts.append(f"Text: {text}")
            prompt_parts.append("")
        
        prompt_parts.append(f"OUTPUT FORMAT (JSON array with {batch_size} objects):")
        prompt_parts.append('[')
        prompt_parts.append('  {"doi_tuong": "...", "noi_dung": "...", "so_lieu_quy_dinh": "..."},')
        prompt_parts.append('  {"doi_tuong": "...", "noi_dung": "...", "so_lieu_quy_dinh": "..."}')
        prompt_parts.append(']')
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON
            # Loại bỏ markdown code blocks nếu có
            if response_text.startswith('```'):
                response_text = re.sub(r'^```json?\s*', '', response_text)
                response_text = re.sub(r'```\s*$', '', response_text)
            
            parsed = json.loads(response_text)
            
            # Validate số lượng
            if len(parsed) != batch_size:
                print(f"    ⚠ LLM trả về {len(parsed)} objects thay vì {batch_size}")
                print(f"    Retry với temperature=0...")
                
                # Retry với temperature thấp hơn
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0)
                )
                response_text = response.text.strip()
                if response_text.startswith('```'):
                    response_text = re.sub(r'^```json?\s*', '', response_text)
                    response_text = re.sub(r'```\s*$', '', response_text)
                parsed = json.loads(response_text)
                
                if len(parsed) != batch_size:
                    print(f"    ❌ Retry thất bại! Dùng fallback...")
                    # Fallback: tạo summaries rỗng
                    parsed = [{"doi_tuong": "N/A", "noi_dung": "N/A", "so_lieu_quy_dinh": "N/A"} for _ in range(batch_size)]
            
            return parsed
            
        except Exception as e:
            print(f"    ❌ Lỗi khi tạo SAC Tier 2: {e}")
            # Fallback
            return [{"doi_tuong": "N/A", "noi_dung": "N/A", "so_lieu_quy_dinh": "N/A"} for _ in range(batch_size)]

    def _strip_json_fence(self, text: str) -> str:
        """Remove markdown fences around JSON output."""
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```json?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _safe_parse_json_list(self, text: str) -> List[Dict]:
        """Parse a JSON list from LLM output."""
        try:
            parsed = json.loads(self._strip_json_fence(text))
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            print(f"  ⚠ Không parse được JSON amendment: {e}")
            return []

    def extract_amendments_from_markdown(
        self,
        markdown_text: str,
        document_metadata: Dict
    ) -> List[Dict]:
        """
        Extract amendment/change operations after chunking and before embedding.
        The review JSON is saved for inspection before it is trusted downstream.
        """
        document_text = markdown_text.strip()
        if not document_text:
            return []

        source_doc = document_metadata.get("document_id") or document_metadata.get("source_file") or "unknown"
        prompt = f"""Bạn là chuyên gia trích xuất quan hệ sửa đổi văn bản pháp luật Việt Nam.

Nhiệm vụ: đọc TOÀN BỘ MARKDOWN BODY và tìm TẤT CẢ nội dung có tính chất sửa đổi, bổ sung, thay thế, bãi bỏ văn bản khác.

Văn bản hiện tại:
- source_document_id: {source_doc}
- source_title: {document_metadata.get("title") or ""}
- source_file: {document_metadata.get("source_file") or ""}

QUY TẮC:
- Chỉ trích xuất nội dung nằm trong MARKDOWN BODY.
- Không lấy phần "Căn cứ..." làm quan hệ sửa đổi.
- CHỈ trích xuất khi văn bản hiện tại đang sửa đổi/bổ sung/thay thế/bãi bỏ văn bản khác.
- BỎ QUA các câu chỉ viện dẫn văn bản khác bằng các cụm như "theo quy định tại", "thực hiện theo", "tiếp tục thực hiện theo", "được quy định tại", "quy định tại".
- BỎ QUA trường hợp văn bản được viện dẫn có tên chứa "sửa đổi, bổ sung". Ví dụ "theo quy định tại Nghị định 70/2025/NĐ-CP ... sửa đổi, bổ sung Nghị định 123/2020/NĐ-CP" KHÔNG có nghĩa văn bản hiện tại sửa Nghị định 123/2020/NĐ-CP.
- Không suy diễn văn bản bị sửa nếu nội dung không nêu rõ.
- "van_ban_bi_sua" là mã của VĂN BẢN KHÁC đang bị văn bản hiện tại tác động. Chỉ lấy mã được nêu rõ trong nội dung.
- Nếu một quy định sửa nhiều Điều/Khoản của cùng một văn bản, giữ trong một object và liệt kê tất cả vị trí tại mảng "dieu_khoan_bi_sua".
- Nếu một quy định tác động đến nhiều văn bản khác nhau, tách thành một object cho mỗi "van_ban_bi_sua".
- "dieu_khoan_sua" là Điều/Khoản trong VĂN BẢN HIỆN TẠI chứa quy định sửa đổi. Hãy tự xác định từ các heading Markdown gần nhất của chính nội dung đó.
- Chuẩn hóa "dieu_khoan_sua" theo dạng "Chương I > Điều 1 > Khoản 1". Nếu văn bản không có heading Chương thì dùng "Điều 1 > Khoản 1"; nếu không có Khoản thì dùng "Chương I > Điều 1" hoặc "Điều 1".
- Không lấy Điều/Khoản được nhắc bên trong câu sửa đổi làm "dieu_khoan_sua"; các vị trí đó thuộc "dieu_khoan_bi_sua".
- "dieu_khoan_bi_sua" là vị trí bị sửa trong văn bản khác, phải normalize theo format của Step 3:
  Input: {{"Chuong": "Chương II: CĂN CỨ...", "Dieu": "Điều 10. Giảm trừ...", "Khoan": "1. Mức giảm..."}}
  Output: "Chương II > Điều 10 > Khoản 1"
- Nếu không có Chương thì chỉ ghi "Điều X" hoặc "Điều X > Khoản Y".
- "noi_dung_moi" chỉ chứa nội dung/cụm từ mới được văn bản hiện tại quy định rõ; không tự bổ sung hoặc diễn giải thành nội dung mới khác.
- Với mẫu "Sửa đổi cụm từ X thành Y tại Điều A, Điều B...", lấy "noi_dung_moi" = Y, "loai_thay_doi" = "sua_doi_cum_tu", và đưa Điều A, Điều B vào "dieu_khoan_bi_sua".
- Với trường hợp bãi bỏ không có nội dung thay thế, để "noi_dung_moi" là chuỗi rỗng.
- Nếu không có quan hệ sửa đổi, trả về [].

VÍ DỤ PHẢI BỎ QUA:
- "thực hiện theo quy định tại Nghị định số 117/2025/NĐ-CP..." -> chỉ là viện dẫn áp dụng.
- "thực hiện thủ tục ... theo quy định tại Nghị định số 168/2025/NĐ-CP..." -> chỉ là viện dẫn áp dụng.
- "theo quy định tại khoản 8 Điều 1 Nghị định số 70/2025/NĐ-CP ... sửa đổi, bổ sung một số điều của Nghị định số 123/2020/NĐ-CP" -> văn bản hiện tại chỉ viện dẫn Nghị định 70, KHÔNG sửa Nghị định 123.

VÍ DỤ PHẢI TRÍCH XUẤT:
- "Sửa đổi cụm từ '500 triệu đồng' thành '01 tỷ đồng' tại Điều 3, Điều 4 Nghị định số 68/2026/NĐ-CP."
- "Bổ sung khoản 15 Điều 4 Nghị định số 320/2025/NĐ-CP như sau: ..."
- "Bãi bỏ Điều 12 Nghị định số ..."

JSON schema:
[
  {{
    "van_ban_sua": "{source_doc}",
    "dieu_khoan_sua": "Điều/Khoản trong văn bản hiện tại chứa quy định sửa đổi, format: Chương II > Điều 10 > Khoản 1",
    "van_ban_bi_sua": "mã văn bản bị sửa nếu có",
    "dieu_khoan_bi_sua": ["Chương/Điều/Khoản bị sửa, format: Chương II > Điều 10 > Khoản 1"],
    "noi_dung_moi": "nội dung/cụm từ mới nếu có",
    "loai_thay_doi": "sua_doi|sua_doi_cum_tu|bo_sung|thay_the|bai_bo",
    "can_cu_trich_doan": "trích đoạn nguyên văn chứng minh"
  }}
]

MARKDOWN BODY:
{document_text}

Chỉ trả về JSON list, không giải thích."""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0)
            )
            amendments = self._safe_parse_json_list(response.text)
            for item in amendments:
                if isinstance(item, dict):
                    item.setdefault("van_ban_sua", source_doc)
            return [item for item in amendments if isinstance(item, dict)]
        except Exception as e:
            print(f"  ⚠ Lỗi amendment extraction: {e}")
            return []

    def save_amendments_review(self, markdown_file: str, amendments: List[Dict]) -> str:
        review_path = os.path.splitext(markdown_file)[0] + "_amendments_review.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(amendments, f, ensure_ascii=False, indent=2)
        print(f"  Saved amendment review: {review_path} ({len(amendments)} relations)")
        return review_path

    def load_amendments_review(self, markdown_file: str) -> Optional[List[Dict]]:
        """Load an existing review file; an empty list is a valid final result."""
        review_path = os.path.splitext(markdown_file)[0] + "_amendments_review.json"
        if not os.path.exists(review_path):
            return None

        try:
            with open(review_path, "r", encoding="utf-8") as f:
                amendments = json.load(f)
            if not isinstance(amendments, list):
                print(f"  ⚠ Amendment review không phải JSON list: {review_path}")
                return None
            print(f"  Loaded amendment review: {review_path} ({len(amendments)} relations)")
            return amendments
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ⚠ Không đọc được amendment review {review_path}: {e}")
            return None

    def _canonical_location(self, location: str) -> str:
        """Normalize a legal location so it can be matched to Step 3 labels."""
        if not isinstance(location, str):
            return ""

        parts = []
        chuong_match = re.search(r"Chương\s+([IVXLCDM\d]+)", location, re.IGNORECASE)
        dieu_match = re.search(r"Điều\s+(\d+)", location, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\d+)", location, re.IGNORECASE)

        if chuong_match:
            parts.append(f"chương {chuong_match.group(1).upper()}")
        if dieu_match:
            parts.append(f"điều {dieu_match.group(1)}")
        if khoan_match:
            parts.append(f"khoản {khoan_match.group(1)}")

        return " > ".join(parts)

    def _canonical_document_id(self, document_id: str) -> str:
        """Normalize document IDs such as 68/2026/NĐ-CP for exact matching."""
        if not isinstance(document_id, str):
            return ""
        normalized = document_id.upper().replace("Đ", "D")
        id_match = re.search(
            r"\d+\s*[/\-]\s*\d+\s*[/\-]\s*[A-Z0-9\-]+",
            normalized
        )
        if id_match:
            normalized = id_match.group(0)
        return re.sub(r"[^A-Z0-9]", "", normalized)

    def _location_is_same_or_descendant(self, chunk_location: str, target_location: str) -> bool:
        """Return True when a chunk is at or below an amended legal location."""
        canonical_chunk = self._canonical_location(chunk_location)
        canonical_target = self._canonical_location(target_location)
        if not canonical_chunk or not canonical_target:
            return False

        chunk_parts = canonical_chunk.split(" > ")
        target_parts = canonical_target.split(" > ")
        target_size = len(target_parts)
        return any(
            chunk_parts[index:index + target_size] == target_parts
            for index in range(len(chunk_parts) - target_size + 1)
        )

    def group_amendments_by_location(
        self,
        amendments: List[Dict],
        chunks: List[Dict],
        document_metadata: Dict
    ) -> Dict[str, List[Dict]]:
        """Attach incoming amendments to chunks of the document being amended."""
        grouped: Dict[str, List[Dict]] = {}
        location_labels = {
            chunk.get("location_label")
            for chunk in chunks
            if chunk.get("location_label") and chunk.get("location_label") != "Không xác định"
        }
        current_document_id = self._canonical_document_id(
            document_metadata.get("document_id", "")
        )

        for amendment in amendments:
            if not isinstance(amendment, dict):
                continue

            amended_document_id = self._canonical_document_id(
                amendment.get("van_ban_bi_sua", "")
            )
            if not amended_document_id or amended_document_id != current_document_id:
                continue

            amended_locations = amendment.get("dieu_khoan_bi_sua", [])
            if isinstance(amended_locations, str):
                amended_locations = [amended_locations]
            if not isinstance(amended_locations, list) or not amended_locations:
                print(
                    f"  ⚠ Quan hệ sửa {document_metadata.get('document_id', '')} "
                    "thiếu dieu_khoan_bi_sua"
                )
                continue

            matched_any = False
            for amended_location in amended_locations:
                matches = [
                    label
                    for label in location_labels
                    if self._location_is_same_or_descendant(label, amended_location)
                ]
                if not matches:
                    print(
                        f"  ⚠ Không tìm thấy chunk {document_metadata.get('document_id', '')} "
                        f"tại {amended_location}"
                    )
                    continue

                matched_any = True
                for label in matches:
                    relations = grouped.setdefault(label, [])
                    relation_for_target = {
                        key: value
                        for key, value in amendment.items()
                        if key not in {"van_ban_bi_sua", "dieu_khoan_bi_sua"}
                    }
                    if relation_for_target not in relations:
                        relations.append(relation_for_target)

            if not matched_any:
                print(
                    f"  ⚠ Không gắn được quan hệ từ {amendment.get('van_ban_sua', '')} "
                    f"vào {amendment.get('van_ban_bi_sua', '')}"
                )

        return grouped

    def _scroll_document_points(self, document_id: str) -> List:
        """Read all existing Qdrant points for one document ID."""
        points = []
        offset = None
        document_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        while True:
            batch, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=document_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break

        return points

    def remove_document_from_qdrant(self, document_id: str) -> int:
        """Remove stale points for a document disabled by the corpus manifest."""
        points = self._scroll_document_points(document_id)
        if not points:
            return 0
        point_ids = [point.id for point in points]
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
            wait=True,
        )
        print(f"Removed non-embedded document: {document_id} ({len(point_ids)} points)")
        return len(point_ids)

    def update_existing_qdrant_amendments(self, amendments: List[Dict]) -> int:
        """Apply outgoing amendments directly to target documents already in Qdrant."""
        amendments_by_target: Dict[str, List[Dict]] = {}
        for amendment in amendments:
            if not isinstance(amendment, dict):
                continue
            target_document_id = amendment.get("van_ban_bi_sua")
            if not target_document_id:
                continue
            amendments_by_target.setdefault(target_document_id, []).append(amendment)

        updated_points = 0
        for target_document_id, target_amendments in amendments_by_target.items():
            points = self._scroll_document_points(target_document_id)
            if not points:
                print(f"  ⚠ Qdrant chưa có chunks của {target_document_id}")
                continue

            source_ids = {
                self._canonical_document_id(amendment.get("van_ban_sua", ""))
                for amendment in target_amendments
            }

            target_updates = 0
            pending_updates = []
            for point in points:
                payload = point.payload or {}
                metadata = dict(payload.get("metadata") or {})
                location_label = metadata.get("location_label", "")
                existing_relations = metadata.get("amendment_relations") or []

                # Rebuild relations from the current source reviews to avoid duplicates
                # and remove stale locations from an older version of the same review.
                new_relations = [
                    relation
                    for relation in existing_relations
                    if self._canonical_document_id(relation.get("van_ban_sua", ""))
                    not in source_ids
                ]

                for amendment in target_amendments:
                    amended_locations = amendment.get("dieu_khoan_bi_sua", [])
                    if isinstance(amended_locations, str):
                        amended_locations = [amended_locations]
                    if not isinstance(amended_locations, list):
                        continue
                    if not any(
                        self._location_is_same_or_descendant(location_label, location)
                        for location in amended_locations
                    ):
                        continue

                    relation_for_target = {
                        key: value
                        for key, value in amendment.items()
                        if key not in {"van_ban_bi_sua", "dieu_khoan_bi_sua"}
                    }
                    if relation_for_target not in new_relations:
                        new_relations.append(relation_for_target)

                amendment_text = self._format_applied_amendments(new_relations)
                existing_amendment_text = str(
                    payload.get("applied_amendment_text") or ""
                ).strip()
                if (
                    new_relations == existing_relations
                    and amendment_text == existing_amendment_text
                ):
                    continue

                metadata["amendment_relations"] = new_relations
                marker = "\n\n[CẬP NHẬT PHÁP LÝ ÁP DỤNG CHO VỊ TRÍ NÀY]\n"
                base_content = str(payload.get("content") or "").split(marker, 1)[0]
                updated_content = base_content
                if amendment_text:
                    updated_content += f"\n\n{amendment_text}"

                updated_payload = dict(payload)
                updated_payload["content"] = updated_content
                updated_payload["applied_amendment_text"] = amendment_text
                updated_payload["metadata"] = metadata
                pending_updates.append(
                    (point.id, updated_content, updated_payload)
                )

            if pending_updates:
                vectors = self.batch_embed(
                    [item[1] for item in pending_updates]
                )
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(id=item[0], vector=vector, payload=item[2])
                        for item, vector in zip(pending_updates, vectors)
                    ],
                    wait=True,
                )
                target_updates = len(pending_updates)
                updated_points += target_updates

            print(
                f"  Updated amendment metadata: {target_document_id} "
                f"({target_updates}/{len(points)} points)"
            )

        return updated_points

    def _format_applied_amendments(self, relations: List[Dict]) -> str:
        blocks = []
        for relation in relations:
            source_id = relation.get("van_ban_sua", "")
            source_location = relation.get("dieu_khoan_sua", "")
            change_type = relation.get("loai_thay_doi", "")
            effective_date = relation.get("effective_date", "")
            new_content = str(relation.get("noi_dung_moi") or "").strip()
            block = (
                f"Nguồn: {source_id} > {source_location}\n"
                f"Loại thay đổi: {change_type}\n"
                f"Hiệu lực: {effective_date or 'Không xác định'}"
            )
            if new_content:
                block += f"\n{new_content}"
            blocks.append(block)
        if not blocks:
            return ""
        return (
            "[CẬP NHẬT PHÁP LÝ ÁP DỤNG CHO VỊ TRÍ NÀY]\n"
            + "\n\n".join(blocks)
        )

    def update_existing_qdrant_document_effects(
        self,
        document_effects: List[Dict],
        amendments: List[Dict],
    ) -> int:
        """Apply document relations and expiration status to target chunks."""
        effects_by_target: Dict[str, List[Dict]] = {}
        for effect in document_effects:
            if not isinstance(effect, dict):
                continue
            target_id = effect.get("target_document_id")
            source_id = effect.get("source_document_id")
            if target_id and source_id:
                effects_by_target.setdefault(target_id, []).append(effect)

        amendment_locations: Dict[tuple, List[str]] = {}
        for amendment in amendments:
            if not isinstance(amendment, dict):
                continue
            key = (
                self._canonical_document_id(amendment.get("van_ban_sua", "")),
                self._canonical_document_id(amendment.get("van_ban_bi_sua", "")),
            )
            locations = amendment.get("dieu_khoan_bi_sua") or []
            if isinstance(locations, str):
                locations = [locations]
            amendment_locations.setdefault(key, []).extend(
                str(location).strip()
                for location in locations
                if str(location).strip()
            )

        today = date.today().isoformat()
        updated_points = 0
        for target_id, target_effects in effects_by_target.items():
            points = self._scroll_document_points(target_id)
            if not points:
                print(f"  ⚠ Qdrant chưa có chunks của {target_id}")
                continue

            source_ids = {
                self._canonical_document_id(effect.get("source_document_id", ""))
                for effect in target_effects
            }
            target_updates = 0

            for point in points:
                payload = point.payload or {}
                metadata = dict(payload.get("metadata") or {})
                location_label = str(metadata.get("location_label") or "")
                existing_affected_by = metadata.get("affected_by_documents") or []
                new_affected_by = [
                    relation
                    for relation in existing_affected_by
                    if self._canonical_document_id(
                        relation.get("source_document_id", "")
                    ) not in source_ids
                ]

                terminal_dates = []
                for effect in target_effects:
                    source_id = self._canonical_document_id(
                        effect.get("source_document_id", "")
                    )
                    target_canonical = self._canonical_document_id(target_id)
                    impact_scope = effect.get("impact_scope", "partial")
                    locations = amendment_locations.get(
                        (source_id, target_canonical),
                        [],
                    )
                    applies_to_chunk = impact_scope == "whole_document" or any(
                        self._location_is_same_or_descendant(location_label, location)
                        for location in locations
                    )
                    if not applies_to_chunk:
                        continue

                    inverse = {
                        "source_document_id": effect.get("source_document_id"),
                        "relation_types": effect.get("relation_types") or [],
                        "impact_scope": impact_scope,
                        "effective_date": effect.get("effective_date"),
                        "status": effect.get("status", "active"),
                    }
                    if inverse not in new_affected_by:
                        new_affected_by.append(inverse)

                    relation_types = set(effect.get("relation_types") or [])
                    effective_date = effect.get("effective_date")
                    ends_old_content = bool(
                        relation_types & {"thay_the", "bai_bo"}
                    )
                    if (
                        ends_old_content
                        and effective_date
                        and effective_date <= today
                    ):
                        terminal_dates.append(effective_date)

                changed = new_affected_by != existing_affected_by
                metadata["affected_by_documents"] = new_affected_by
                if terminal_dates:
                    expiration_date = min(terminal_dates)
                    if metadata.get("status") != "inactive":
                        metadata["status"] = "inactive"
                        changed = True
                    if metadata.get("expiration_date") != expiration_date:
                        metadata["expiration_date"] = expiration_date
                        changed = True

                if not changed:
                    continue

                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"metadata": metadata},
                    points=[point.id],
                    wait=True,
                )
                target_updates += 1
                updated_points += 1

            print(
                f"  Updated document effects: {target_id} "
                f"({target_updates}/{len(points)} points)"
            )

        return updated_points

    def create_collection(self, recreate: bool = False):
        """Tạo Qdrant collection"""
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            if recreate:
                print(f"Xóa collection cũ: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
            else:
                print(f"Collection đã tồn tại: {self.collection_name}")
                return
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Đã tạo collection: {self.collection_name}")
    
    def chunk_markdown(self, markdown_text: str) -> List[Dict]:
        """
        Chunking markdown VÀ extract location labels
        CHƯA inject SAC Tier 1 & 2 (sẽ làm sau)
        
        Returns: List of raw chunks với metadata phân cấp và location_label
        """
        # Bước 1: Split theo header - tự động kế thừa metadata phân cấp
        md_header_splits = self.markdown_splitter.split_text(markdown_text)
        
        # Bước 2: Split thêm nếu chunk quá dài
        final_chunks = []
        
        for doc in md_header_splits:
            content = doc.page_content
            hierarchical_metadata = doc.metadata  # Tự động có Chuong/Dieu/Khoan
            
            # Extract location label
            location_label = self.extract_location_label(hierarchical_metadata)
            
            # Nếu chunk nhỏ hơn limit, giữ nguyên
            if len(content) <= self.chunk_size:
                sub_chunks = [content]
            else:
                # Split thêm bằng RecursiveCharacterTextSplitter
                sub_chunks = self.recursive_splitter.split_text(content)
            
            # Lưu từng sub-chunk với metadata
            for i, sub_chunk in enumerate(sub_chunks):
                final_chunks.append({
                    "original_content": sub_chunk,
                    "hierarchical_metadata": hierarchical_metadata,
                    "location_label": location_label,
                    "sub_chunk_index": i if len(sub_chunks) > 1 else None,
                    "chunk_size": len(sub_chunk)
                })
        
        return final_chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector từ text
        bge-m3 KHÔNG cần word segmentation - tự xử lý subword tokenization
        """
        try:
            # Encode trực tiếp (không cần pyvi)
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"    Lỗi khi tạo embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed nhiều texts theo batch
        bge-m3 KHÔNG cần word segmentation
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"    Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            # Encode trực tiếp (không cần tách từ)
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.extend([emb.tolist() for emb in batch_embeddings])
        
        return embeddings

    def collect_all_amendments(self, documents: List[tuple]) -> List[Dict]:
        """Collect amendment relations already extracted into Step 1 metadata."""
        all_amendments = []
        print(f"\n{'='*60}")
        print("Loading amendment relations from document metadata")
        print(f"{'='*60}")

        for markdown_file, _, metadata_file in documents:
            with open(metadata_file, "r", encoding="utf-8") as f:
                document_metadata = json.load(f)

            document_name = document_metadata.get(
                "document_id",
                os.path.basename(markdown_file)
            )
            amendments = document_metadata.get("amendment_relations") or []
            if not isinstance(amendments, list):
                raise ValueError(
                    f"amendment_relations của {document_name} phải là JSON list."
                )
            if not amendments:
                print(f"  No amendment relations: {document_name}")
            else:
                print(f"  Loaded {len(amendments)} relations: {document_name}")
            all_amendments.extend(amendments)

        print(f"  Total amendment relations: {len(all_amendments)}")
        return all_amendments

    def collect_all_document_effects(self, documents: List[tuple]) -> List[Dict]:
        """Collect outgoing document-level effects from Step 1 metadata."""
        effects = []
        seen = set()
        for markdown_file, _, metadata_file in documents:
            with open(metadata_file, "r", encoding="utf-8") as f:
                document_metadata = json.load(f)

            source_id = document_metadata.get("document_id")
            for relation in document_metadata.get("affects_documents") or []:
                if not isinstance(relation, dict) or not relation.get("target_document_id"):
                    continue
                effect = {
                    "source_document_id": source_id,
                    "target_document_id": relation.get("target_document_id"),
                    "relation_types": relation.get("relation_types") or [],
                    "impact_scope": relation.get("impact_scope", "partial"),
                    "effective_date": relation.get("effective_date"),
                    "status": relation.get("status", "active"),
                }
                key = json.dumps(effect, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    effects.append(effect)
        print(f"  Total document-level effects: {len(effects)}")
        return effects
    
    def process_document(
        self,
        markdown_file: str,
        summary_file: str,
        metadata_file: str,
        all_amendments: List[Dict] = None
    ) -> int:
        """
        Process một văn bản: chunking + SAC Tier 2 + embedding + upload Qdrant
        
        Returns: số lượng chunks đã thêm vào Qdrant
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(markdown_file)}")
        print(f"{'='*60}")
        
        # Đọc markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        print(f"  Document length: {len(markdown_text)} chars")
        
        # Đọc summary (SAC Tier 1)
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        sac_tier1 = summary_data.get("document_summary", "")
        print(f"  SAC Tier 1: {sac_tier1}")
        
        # Đọc document metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            document_metadata = json.load(f)
        print(f"  Document ID: {document_metadata.get('document_id', 'N/A')}")
        
        # Step 1: Chunking (chưa inject SAC)
        print(f"  Chunking...")
        chunks = self.chunk_markdown(markdown_text)
        print(f"    Total chunks: {len(chunks)}")

        # Step 1.5: Match incoming amendments to this document's chunks.
        # Direct calls still work, but the main workflow supplies corpus-wide relations.
        if all_amendments is None:
            all_amendments = document_metadata.get("amendment_relations") or []
            if not isinstance(all_amendments, list):
                raise ValueError("amendment_relations trong metadata phải là JSON list.")

        amendments_by_location = self.group_amendments_by_location(
            all_amendments,
            chunks,
            document_metadata
        )
        
        # Step 2: Generate SAC Tier 2 summaries (batch processing)
        print(f"  Generating SAC Tier 2 summaries (batch={self.tier2_batch_size})...")
        tier2_summaries = []
        
        for i in range(0, len(chunks), self.tier2_batch_size):
            batch = chunks[i:i + self.tier2_batch_size]
            batch_num = i // self.tier2_batch_size + 1
            total_batches = (len(chunks) + self.tier2_batch_size - 1) // self.tier2_batch_size
            
            print(f"    Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            summaries = self.generate_tier2_summaries_batch(batch)
            tier2_summaries.extend(summaries)
            
            # Sleep 10s sau mỗi batch để tránh rate limit
            if batch_num < total_batches:  # Không sleep sau batch cuối
                print(f"    Sleeping 10s to avoid rate limit...")
                time.sleep(10)
        
        # Step 3: Inject SAC Tier 1 + Tier 2 vào chunks
        print(f"  Injecting SAC Tier 1 + Tier 2...")
        augmented_chunks = []
        
        for chunk, tier2 in zip(chunks, tier2_summaries):
            # Format SAC Tier 2 với FULL location
            # Hỗ trợ: Chương/Điều/Khoản (luật) và Phụ lục/Mục (nghị định)
            metadata = chunk['hierarchical_metadata']
            location_parts = []
            
            # Thêm Document ID vào đầu
            doc_id = document_metadata.get('document_id', '')
            if doc_id:
                location_parts.append(doc_id)
            
            # Kiểm tra nếu "Chuong" thực ra là Phụ lục
            chuong_val = metadata.get('Chuong', '')
            if 'Phụ lục' in chuong_val or 'PHỤ LỤC' in chuong_val:
                # Đây là Phụ lục
                match = re.search(r'Phụ lục\s+([IVXLCDM\d]+)', chuong_val, re.IGNORECASE)
                if match:
                    location_parts.append(f"Phụ lục {match.group(1)}")
            else:
                # Chương bình thường
                if chuong_val:
                    match = re.search(r'Chương\s+([IVXLCDM]+)', chuong_val)
                    if match:
                        location_parts.append(f"Chương {match.group(1)}")
            
            # Kiểm tra nếu "Dieu" thực ra là Mục
            dieu_val = metadata.get('Dieu', '')
            if 'Mục' in dieu_val or 'MỤC' in dieu_val:
                # Đây là Mục
                match = re.search(r'Mục\s+([IVXLCDM\d]+)', dieu_val, re.IGNORECASE)
                if match:
                    location_parts.append(f"Mục {match.group(1)}")
            else:
                # Điều bình thường
                if dieu_val:
                    match = re.search(r'Điều\s+(\d+)', dieu_val)
                    if match:
                        location_parts.append(f"Điều {match.group(1)}")
            
            # Extract Khoản
            # Lưu ý: khoan_val có dạng "Khoản 4. Content" (do step1 tạo heading)
            khoan_val = metadata.get('Khoan', '')
            if khoan_val:
                match = re.search(r'Khoản\s+(\d+)', khoan_val, re.IGNORECASE)
                if match:
                    location_parts.append(f"Khoản {match.group(1)}")
            
            full_location = " > ".join(location_parts) if location_parts else "Không xác định"
            
            # Format SAC Tier 2 với full location
            tier2_text = f"[{full_location}]\n"
            tier2_text += f"ĐỐI TƯỢNG: {tier2['doi_tuong']}\n"
            tier2_text += f"NỘI DUNG: {tier2['noi_dung']}\n"
            tier2_text += f"SỐ LIỆU/QUY ĐỊNH: {tier2['so_lieu_quy_dinh']}"

            applied_relations = amendments_by_location.get(
                chunk["location_label"],
                [],
            )
            amendment_text = self._format_applied_amendments(applied_relations)

            # Embed cả nội dung gốc và nguyên section sửa đổi đã được Python trích.
            augmented_content = (
                f"{sac_tier1}\n\n{tier2_text}\n\n"
                f"{chunk['original_content']}"
            )
            if amendment_text:
                augmented_content += f"\n\n{amendment_text}"
            
            augmented_chunks.append({
                **chunk,
                "sac_tier1": sac_tier1,
                "sac_tier2": tier2,
                "tier2_text": tier2_text,
                "applied_amendment_text": amendment_text,
                "content": augmented_content,
                "augmented_size": len(augmented_content)
            })
        
        # Stats
        avg_orig = sum(c['chunk_size'] for c in augmented_chunks) / len(augmented_chunks)
        avg_aug = sum(c['augmented_size'] for c in augmented_chunks) / len(augmented_chunks)
        print(f"    Avg original size: {avg_orig:.0f} chars")
        print(f"    Avg augmented size (with Tier 1+2): {avg_aug:.0f} chars")
        
        # Step 4: Generate embeddings
        print(f"  Generating embeddings...")
        texts = [chunk["content"] for chunk in augmented_chunks]
        embeddings = self.batch_embed(texts)
        
        # Step 5: Prepare Qdrant points
        print(f"  Preparing Qdrant points...")
        points = []
        document_id = document_metadata.get("document_id", "")
        document_type = document_metadata.get("document_type")
        legal_rank = document_metadata.get("legal_rank")
        if not document_type or not isinstance(legal_rank, int):
            raise ValueError(
                f"Metadata của {document_id!r} thiếu document_type/legal_rank hợp lệ. "
                "Hãy chạy lại Step 1 cho văn bản này."
            )
        
        for idx, (chunk, embedding) in enumerate(zip(augmented_chunks, embeddings)):
            # Merge hierarchical metadata + document metadata
            full_metadata = {
                **chunk.get("hierarchical_metadata", {}),  # Chương/Điều/Khoản
                "location_label": chunk["location_label"],
                "document_id": document_id,
                "title": document_metadata.get("title", ""),
                "document_type": document_type,
                "legal_rank": legal_rank,
                "issued_date": document_metadata.get("issued_date"),
                "effective_date": document_metadata.get("effective_date", ""),
                "expiration_date": document_metadata.get("expiration_date"),
                "status": document_metadata.get("status", ""),
                "affects_documents": document_metadata.get("affects_documents") or [],
                "affected_by_documents": document_metadata.get("affected_by_documents") or [],
                # Bỏ "amended_laws" - đã được xóa ở step 1
                "source_file": os.path.basename(markdown_file),
                "sub_chunk_index": chunk.get("sub_chunk_index"),
                "chunk_size": chunk.get("chunk_size"),
                "augmented_size": chunk.get("augmented_size"),
                "amendment_relations": amendments_by_location.get(chunk["location_label"], [])
            }
            
            point_key = f"{os.path.basename(markdown_file)}:{idx}"
            point_id = int(hashlib.sha256(point_key.encode("utf-8")).hexdigest()[:16], 16)
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "original_content": chunk["original_content"],
                    "sac_tier1": chunk["sac_tier1"],
                    "sac_tier2": chunk["sac_tier2"],
                    "tier2_text": chunk["tier2_text"],
                    "applied_amendment_text": chunk["applied_amendment_text"],
                    "metadata": full_metadata
                }
            )
            points.append(point)
        
        # Step 6: Upload to Qdrant
        print(f"  Uploading to Qdrant...")
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"  ✓ Đã thêm {len(points)} chunks vào Qdrant")
        return len(points)
    
    def get_collection_info(self):
        """Hiển thị thông tin collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            print(f"\n{'='*60}")
            print(f"Collection Info")
            print(f"{'='*60}")
            print(f"  Name: {self.collection_name}")
            print(f"  Vector size: {info.config.params.vectors.size}")
            print(f"  Distance: {info.config.params.vectors.distance}")
            print(f"  Points count: {info.points_count}")
        except Exception as e:
            print(f"Không thể lấy thông tin collection: {e}")
    
    def test_search(self, query: str, limit: int = 3):
        """Test search"""
        print(f"\n{'='*60}")
        print(f"Test Search")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search - dùng query_points thay vì search
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        
        for i, hit in enumerate(results.points, 1):
            print(f"\n[{i}] Score: {hit.score:.4f}")
            print(f"Content: {hit.payload['content'][:200]}...")
            metadata = hit.payload['metadata']
            print(f"Document: {metadata.get('title', 'N/A')}")
            print(f"Document ID: {metadata.get('document_id', 'N/A')}")
            if metadata.get('Dieu'):
                print(f"Location: {metadata.get('Chuong', '')} > {metadata.get('Dieu', '')}")


