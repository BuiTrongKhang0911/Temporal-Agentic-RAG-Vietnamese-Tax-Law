"""
Bước 1: Phân cấp Markdown (CẬP NHẬT: Extract document_id từ văn bản)
Chuyển đổi văn bản luật sang format Markdown với cấu trúc phân cấp

THAY ĐỔI:
- Tên file mới: #effective_date#status.docx (bỏ doc_id vì dấu / không hợp lệ)
- Document ID được extract từ văn bản (sau chữ "Số:")
"""
import argparse
from datetime import date, datetime
import hashlib
import re
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from config import (
    CORPUS_ANALYSIS_GEMINI_MODEL,
    CORPUS_MANIFEST_PATH,
    GOOGLE_API_KEY,
    INPUT_DIR,
    OUTPUT_DIR,
)


LEGAL_RANK_BY_TYPE = {
    "hien_phap": 4,
    "luat": 3,
    "bo_luat": 3,
    "nghi_quyet_quoc_hoi": 3,
    "nghi_dinh": 2,
    "quyet_dinh": 2,
    "thong_tu": 1,
    "unknown": 0,
}


class MarkdownHierarchyConverter:
    def __init__(
        self,
        enable_llm: bool = True,
        manifest_path: Optional[str] = "corpus_manifest.json",
    ):
        # Patterns để detect cấu trúc văn bản luật
        self.chuong_pattern = re.compile(
            r'^(Chương\s+(?:[IVXLCDM]+|\d+))\s*[:\.\-]?\s*(.*)',
            re.IGNORECASE,
        )
        self.dieu_pattern = re.compile(
            r'^(Điều\s+\d+[a-zđ]?)\s*[\.\:]?\s*(.*)$',
            re.IGNORECASE,
        )
        
        # Khoản pattern - số 1-50 ở đầu dòng, bỏ qua citation [số]
        self.khoan_pattern = re.compile(r'^(\d{1,2}[a-zđ]?)\.\s*(?:\[\d+\]\s*)?(.+)', re.IGNORECASE)
        
        # Điểm có thể mang hậu tố số liền hoặc phân cấp:
        # a), a1), đ1), b.3), b.3.1).
        self.diem_pattern = re.compile(
            r'^\s*([a-zđ](?:\d+|(?:\.\d+)+)?)[\.)]\s+(.+)',
            re.IGNORECASE,
        )
        
        # PHỤ LỤC pattern - Match "Phụ lục" với hoặc không có số
        # VD: "Phụ lục I", "PHỤ LỤC", "Phụ lục 1", "Phụ lục"
        # Capture: group(1) = "Phụ lục", group(2) = số (optional), group(3) = title (optional)
        self.phu_luc_pattern = re.compile(
            r'^(Phụ\s*lục)(?:\s+([IVXLCDM]+|\d+))?(?:[\s:.\-]+(.*))?$',
            re.IGNORECASE
        )
        
        # MỤC pattern trong Phụ lục - Roman numerals hoặc số
        # VD: "I. Danh mục...", "II. Việc xác định...", "1. Danh mục..."
        self.muc_pattern = re.compile(
            r'^([IVXLCDM]+|\d+)\.\s+(.+)$'
        )
        
        # Pattern để detect bảng (dòng có nhiều | hoặc tab)
        self.table_pattern = re.compile(r'.*\|.*\|.*')

        self.manifest_path = manifest_path
        self.corpus_manifest = self._load_manifest(manifest_path)

        self.enable_llm = enable_llm
        self.amendment_batch_size = max(
            1,
            int(os.getenv("LEGAL_EFFECTS_SECTION_BATCH_SIZE", "8")),
        )
        self.llm_delay_seconds = max(
            0.0,
            float(os.getenv("LEGAL_EFFECTS_LLM_DELAY_SECONDS", "10")),
        )
        self._last_llm_call_at: Optional[float] = None
        self.gemini_model = None
        self.genai = None
        if enable_llm:
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "Thiếu GOOGLE_API_KEY; dùng --no-llm nếu chỉ kiểm tra deterministic."
                )
            import google.generativeai as genai

            self.genai = genai
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel(
                CORPUS_ANALYSIS_GEMINI_MODEL
            )

    def _load_manifest(self, manifest_path: Optional[str]) -> Dict[str, Dict]:
        if not manifest_path:
            return {}
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy corpus manifest: {manifest_path}")
        with open(path, 'r', encoding='utf-8') as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError("Corpus manifest phải là JSON object.")
        normalized_manifest = {}
        for document_id, entry in manifest.items():
            if not isinstance(document_id, str):
                raise ValueError("Document ID trong manifest phải là chuỗi.")
            if isinstance(entry, list):
                entry = {
                    "include_locations": entry,
                    "embed_in_qdrant": True,
                }
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest của {document_id} phải là JSON object.")
            selectors = entry.get("include_locations") or []
            embed_in_qdrant = entry.get("embed_in_qdrant", True)
            if not isinstance(selectors, list) or not isinstance(embed_in_qdrant, bool):
                raise ValueError(f"Manifest của {document_id} có schema không hợp lệ.")
            if not all(isinstance(selector, str) and selector.strip() for selector in selectors):
                raise ValueError(f"Manifest của {document_id} chứa location không hợp lệ.")
            normalized_manifest[document_id] = {
                "include_locations": selectors,
                "embed_in_qdrant": embed_in_qdrant,
            }
        return normalized_manifest
    
    def extract_metadata(self, markdown_text: str, source_file: str) -> Tuple[Dict, str]:
        """Python trích metadata chắc chắn từ phần đầu văn bản, không đọc tên file."""
        lines = markdown_text.split('\n')
        content_start = next(
            (index for index, line in enumerate(lines) if re.match(r'^#{1,3}\s+', line)),
            None,
        )
        metadata_zone = lines if content_start is None else lines[:content_start]
        header_text = '\n'.join(metadata_zone[:80])

        document_id = self._extract_document_id(header_text)
        document_type = self._extract_document_type(header_text, document_id)
        metadata = {
            "source_file": source_file,
            "document_id": document_id,
            "title": self._extract_title(metadata_zone, document_type),
            "document_type": document_type,
            "legal_rank": LEGAL_RANK_BY_TYPE.get(document_type, 0),
            "issued_date": self._extract_issued_date(header_text),
            "effective_date": None,
            "expiration_date": None,
            "status": "active",
            "affects_documents": [],
            "affected_by_documents": [],
            "amendment_relations": [],
            "embed_in_qdrant": self.corpus_manifest.get(
                document_id,
                {},
            ).get("embed_in_qdrant", True),
        }

        cleaned_markdown = markdown_text if content_start is None else '\n'.join(lines[content_start:])
        return metadata, cleaned_markdown.strip()

    def _extract_document_id(self, header_text: str) -> Optional[str]:
        patterns = [
            r'(?i)(?:luật|nghị\s*định|thông\s*tư|nghị\s*quyết|quyết\s*định)?\s*số\s*:\s*([0-9]+(?:[/\-][0-9]{4})?[/\-][0-9A-ZĐƯƠ\-]+(?:[/\-][A-ZĐƯƠ\-]+)*)',
            r'(?i)\bsố\s*:\s*([0-9]+(?:[/\-][0-9]{4})?[/\-][0-9A-ZĐƯƠ\-]+(?:[/\-][A-ZĐƯƠ\-]+)*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, header_text)
            if match:
                return match.group(1).upper().replace("-NĐ-CP", "/NĐ-CP")
        return None

    def _extract_document_type(self, header_text: str, document_id: Optional[str]) -> str:
        exact_labels = {
            "HIẾN PHÁP": "hien_phap",
            "BỘ LUẬT": "bo_luat",
            "LUẬT": "luat",
            "NGHỊ ĐỊNH": "nghi_dinh",
            "THÔNG TƯ": "thong_tu",
            "NGHỊ QUYẾT": "nghi_quyet_quoc_hoi",
            "QUYẾT ĐỊNH": "quyet_dinh",
        }
        for line in header_text.splitlines():
            label = self._clean_markdown_cell(line).upper()
            if label in exact_labels:
                return exact_labels[label]

        identifier = (document_id or "").upper()
        if "NĐ-CP" in identifier:
            return "nghi_dinh"
        if "TT-" in identifier or identifier.endswith("/TT"):
            return "thong_tu"
        if "/QH" in identifier:
            return "luat"
        return "unknown"

    def _extract_title(self, metadata_zone: List[str], document_type: str) -> Optional[str]:
        type_labels = {
            "hien_phap": "HIẾN PHÁP",
            "bo_luat": "BỘ LUẬT",
            "luat": "LUẬT",
            "nghi_dinh": "NGHỊ ĐỊNH",
            "thong_tu": "THÔNG TƯ",
            "nghi_quyet_quoc_hoi": "NGHỊ QUYẾT",
            "quyet_dinh": "QUYẾT ĐỊNH",
        }
        label = type_labels.get(document_type)
        if not label:
            return None

        cleaned = [self._clean_markdown_cell(line) for line in metadata_zone]
        for index, line in enumerate(cleaned):
            if line.upper() != label:
                continue
            title_parts = [label]
            for candidate in cleaned[index + 1:index + 5]:
                if not candidate:
                    continue
                if re.match(r'(?i)^(căn cứ|quốc hội ban hành|chính phủ ban hành)', candidate):
                    break
                if re.search(r'(?i)\bngày\s+\d{1,2}\s+tháng', candidate):
                    continue
                title_parts.append(candidate)
                break
            return " ".join(title_parts)
        return label

    def _clean_markdown_cell(self, text: str) -> str:
        cleaned = text.strip().strip('|').strip()
        if '|' in cleaned:
            cells = [cell.strip() for cell in cleaned.split('|') if cell.strip()]
            cleaned = ' '.join(cells)
        return re.sub(r'\s+', ' ', cleaned)

    def _extract_issued_date(self, header_text: str) -> Optional[str]:
        matches = re.finditer(
            r'(?i)(?:Hà\s*Nội\s*,\s*)?ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
            header_text,
        )
        for match in matches:
            return self._normalize_date_parts(match.group(1), match.group(2), match.group(3))
        return None

    def _normalize_date_parts(self, day: str, month: str, year: str) -> Optional[str]:
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

    def read_docx(self, file_path: str) -> str:
        """Đọc file .docx và trả về text, xử lý table riêng bằng python-docx"""
        doc = Document(file_path)
        lines = []
        
        # Iterate qua các elements trong document body
        for element in doc.element.body:
            if isinstance(element, CT_Tbl):  # Table element
                # Xử lý bảng bằng python-docx
                table = Table(element, doc)
                table_text = self._extract_table_as_markdown(table)
                if table_text:
                    lines.append(table_text)
                    
            elif isinstance(element, CT_P):  # Paragraph element
                # Tìm paragraph object tương ứng
                for para in doc.paragraphs:
                    if para._element == element:
                        text = para.text.strip()
                        if text:
                            lines.append(text)
                        break
        
        return '\n'.join(lines)
    
    def _extract_table_as_markdown(self, table: Table) -> str:
        """
        Chuyển Table object thành Markdown table
        XÓA FOOTER: Chỉ xóa bảng footer/chữ ký (Nơi nhận, KT., chữ ký)
        GIỮ LẠI: Tất cả bảng nội dung (bảng thuế suất, header văn bản, v.v.)
        """
        if not table.rows or len(table.rows) == 0:
            return ""
        
        # Lấy toàn bộ text trong bảng để check
        table_text = ' '.join([
            ' '.join([cell.text.strip() for cell in row.cells])
            for row in table.rows
        ])
        
        # XÓA FOOTER/CHỮ KÝ: Chỉ xóa khi có ÍT NHẤT 2 TRONG 3 điều kiện
        # (tránh xóa nhầm bảng có từ "Bộ trưởng" trong nội dung)
        footer_conditions = {
            "signature": ["Nơi nhận", "KT.", "CHỦ NHIỆM", "XÁC THỰC", "CHỦ TỊCH"],  # Điều kiện 1: Từ khóa chữ ký
            "archive": ["Lưu:", "Văn phòng", "In "],                     # Điều kiện 2: Từ khóa lưu trữ
            "small_table": len(table.rows) < 3                           # Điều kiện 3: Bảng nhỏ < 3 hàng
        }
        
        has_signature = any(kw in table_text for kw in footer_conditions["signature"])
        has_archive = any(kw in table_text for kw in footer_conditions["archive"])
        is_small = footer_conditions["small_table"]
        
        # Đếm số điều kiện thỏa mãn
        conditions_met = sum([has_signature, has_archive, is_small])
        
        # Xóa khi có ít nhất 2 điều kiện thỏa mãn
        if conditions_met >= 2:
            return ""  # Xóa footer
        
        # GIỮ LẠI: Tất cả bảng khác (bảng nội dung, header văn bản)
        md_lines = []
        md_lines.append("\n<!-- TABLE START -->")
        
        # Header row (first row)
        header_cells = [cell.text.strip().replace('\n', ' ') for cell in table.rows[0].cells]
        
        # Đảm bảo có ít nhất 1 cột
        if not header_cells or all(not cell for cell in header_cells):
            return ""
        
        md_lines.append("| " + " | ".join(header_cells) + " |")
        md_lines.append("|" + "|".join([" --- "] * len(header_cells)) + "|")
        
        # Data rows
        for row in table.rows[1:]:
            cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
            # Đảm bảo số cột khớp với header
            while len(cells) < len(header_cells):
                cells.append("")
            md_lines.append("| " + " | ".join(cells[:len(header_cells)]) + " |")
        
        md_lines.append("<!-- TABLE END -->\n")
        return '\n'.join(md_lines)
    
    def convert_to_markdown(self, text: str) -> str:
        """Convert văn bản sang Markdown với phân cấp và lookahead."""
        lines = text.split('\n')
        markdown_lines = []
        in_table = False
        quote_mode = None
        current_context = None  # Track: chuong, dieu, phu_luc, muc, hoặc None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                markdown_lines.append('')
                i += 1
                continue
            
            # Check nếu đang trong bảng
            if '<!-- TABLE START -->' in line:
                in_table = True
                markdown_lines.append(line)
                i += 1
                continue
            elif '<!-- TABLE END -->' in line:
                in_table = False
                markdown_lines.append(line)
                i += 1
                continue
            elif in_table or self.table_pattern.match(line):
                markdown_lines.append(line)
                i += 1
                continue

            # Replacement text in amendment documents often contains numbered
            # clauses inside quotation marks. Keep that quoted block verbatim;
            # otherwise inner "1.", "2." lines would be mistaken for clauses
            # of the amendment document itself.
            if quote_mode is not None:
                markdown_lines.append(line)
                if quote_mode == "curly" and "”" in line:
                    quote_mode = None
                elif quote_mode == "ascii" and line.count('"') % 2 == 1:
                    quote_mode = None
                i += 1
                continue

            if line.startswith("“"):
                markdown_lines.append(line)
                if "”" not in line[1:]:
                    quote_mode = "curly"
                i += 1
                continue

            if line.startswith('"'):
                markdown_lines.append(line)
                if line.count('"') % 2 == 1:
                    quote_mode = "ascii"
                i += 1
                continue
            
            # 1. PHỤ LỤC - Level 1 (#)
            phu_luc_match = self.phu_luc_pattern.match(line)
            if phu_luc_match:
                # group(1) = "Phụ lục", group(2) = số (optional), group(3) = title (optional)
                phu_luc_num = phu_luc_match.group(2)  # "I", "II", "1"... hoặc None
                phu_luc_title = phu_luc_match.group(3)  # Tiêu đề hoặc None
                
                # Clean title
                if phu_luc_title:
                    phu_luc_title = phu_luc_title.strip()
                
                # Nếu không có tiêu đề trên cùng dòng, nhìn dòng tiếp theo
                if not phu_luc_title:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    
                    if j < len(lines):
                        next_line = lines[j].strip()
                        # Kiểm tra dòng tiếp theo không phải Mục hoặc các pattern khác
                        if (not self.muc_pattern.match(next_line) and
                            not self.phu_luc_pattern.match(next_line) and
                            not self.dieu_pattern.match(next_line)):
                            phu_luc_title = next_line
                            i = j
                
                # Tạo markdown header
                if phu_luc_num:
                    if phu_luc_title:
                        markdown_lines.append(f"# Phụ lục {phu_luc_num}: {phu_luc_title}")
                    else:
                        markdown_lines.append(f"# Phụ lục {phu_luc_num}")
                else:
                    # Không có số, chỉ có "Phụ lục"
                    if phu_luc_title:
                        markdown_lines.append(f"# Phụ lục: {phu_luc_title}")
                    else:
                        markdown_lines.append(f"# Phụ lục")
                
                current_context = "phu_luc"
                i += 1
                continue
            
            # 2. MỤC trong Phụ lục - Level 2 (##)
            # CHỈ apply khi đang trong Phụ lục hoặc Mục (để Mục II có thể đi sau Mục I)
            muc_match = self.muc_pattern.match(line)
            if muc_match and current_context in ["phu_luc", "muc"]:
                muc_num = muc_match.group(1)  # "I", "II", "1", "2"...
                muc_content = muc_match.group(2).strip()
                
                # PHÂN BIỆT MỤC vs ITEM:
                # Mục: CHỈ Roman numerals (I, II, III...)
                # Item: số (1, 2, 3...) → GIỮ NGUYÊN
                is_muc = False
                
                # CHỈ Roman numerals là Mục
                if not muc_num.isdigit() and re.match(r'^[IVXLCDM]+$', muc_num):
                    is_muc = True
                
                if is_muc:
                    markdown_lines.append(f"## Mục {muc_num}. {muc_content}")
                    current_context = "muc"
                    i += 1
                    continue
            
            # 3. Chương — LOOKAHEAD để gộp tên chương
            chuong_match = self.chuong_pattern.match(line)
            if chuong_match:
                chuong_num = chuong_match.group(1)
                chuong_title = chuong_match.group(2).strip()
                
                if not chuong_title:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    
                    if j < len(lines):
                        next_line = lines[j].strip()
                        if (not self.dieu_pattern.match(next_line) and
                            not self.chuong_pattern.match(next_line) and
                            not re.match(r'^\d+\.', next_line)):
                            chuong_title = next_line
                            i = j
                
                if chuong_title:
                    markdown_lines.append(f"# {chuong_num}: {chuong_title}")
                else:
                    markdown_lines.append(f"# {chuong_num}")
                
                current_context = "chuong"
                i += 1
                continue
            
            # 4. Điều
            dieu_match = self.dieu_pattern.match(line)
            if dieu_match:
                dieu_num = dieu_match.group(1)
                dieu_title = dieu_match.group(2).strip()

                if not dieu_title:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    if j < len(lines):
                        next_line = lines[j].strip()
                        if (
                            not self.dieu_pattern.match(next_line)
                            and not self.chuong_pattern.match(next_line)
                            and not self.khoan_pattern.match(next_line)
                        ):
                            dieu_title = next_line
                            i = j

                if dieu_title:
                    markdown_lines.append(f"## {dieu_num}. {dieu_title}")
                else:
                    markdown_lines.append(f"## {dieu_num}")
                current_context = "dieu"
                i += 1
                continue
            
            # 5. Điểm a), b), c) - GIỮ NGUYÊN (check TRƯỚC khoản)
            if self.diem_pattern.match(line):
                markdown_lines.append(line)
                i += 1
                continue
            
            # 6. Khoản - CHỈ apply khi đang trong Điều
            khoan_match = self.khoan_pattern.match(line)
            if khoan_match and current_context == "dieu":
                khoan_num = khoan_match.group(1)
                khoan_content = khoan_match.group(2).strip()
                
                khoan_number_match = re.match(r'\d+', khoan_num)
                if khoan_number_match and int(khoan_number_match.group(0)) <= 50 and khoan_content:
                    first_char = khoan_content[0]
                    if first_char.isalpha() or ord(first_char) > 127:
                        markdown_lines.append(f"### Khoản {khoan_num}. {khoan_content}")
                        i += 1
                        continue
            
            # 7. Các dòng thông thường
            markdown_lines.append(line)
            i += 1
        
        return '\n'.join(markdown_lines)
    
    def _strip_json_fence(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return cleaned.strip()

    def _normalize_iso_date(self, value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        text = str(value).strip()
        try:
            return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

    def _call_llm_json(self, prompt: str, label: str) -> Dict:
        """Call Gemini once, log the exact response, and require a JSON object."""
        if self._last_llm_call_at is not None:
            elapsed = time.monotonic() - self._last_llm_call_at
            wait_seconds = self.llm_delay_seconds - elapsed
            if wait_seconds > 0:
                print(f"Nghỉ {wait_seconds:.1f}s để tránh rate limit...")
                time.sleep(wait_seconds)

        response = self.gemini_model.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(temperature=0),
        )
        self._last_llm_call_at = time.monotonic()
        raw_response_text = response.text or ""
        print("\n" + "=" * 80)
        print(f"LLM RAW RESPONSE: {label}")
        print("=" * 80)
        print(raw_response_text)
        print("=" * 80 + "\n")
        try:
            parsed = json.loads(self._strip_json_fence(raw_response_text))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"LLM không trả JSON object hợp lệ ({label}): {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM phải trả về JSON object ({label}).")
        return parsed

    def _collect_amendment_sections(self, cleaned_markdown: str) -> List[Dict]:
        """Return leaf Article/Clause sections with source locations fixed by Python."""
        lines, sections = self._build_markdown_sections(cleaned_markdown)
        sections_by_start = {section["start"]: section for section in sections}
        legal_sections = [
            section
            for section in sections
            if section.get("location")
            and re.search(r'(?:^| > )(?:Điều|Khoản)\s+', section["location"])
        ]
        result = []
        for section in legal_sections:
            has_legal_child = any(
                section["start"] in child.get("ancestors", [])
                and child.get("location")
                for child in legal_sections
            )
            if has_legal_child:
                continue
            content = "\n".join(
                lines[section["start"]:section["end"]]
            ).strip()
            if content:
                ancestor_sections = [
                    sections_by_start[start]
                    for start in section.get("ancestors", [])
                    if start in sections_by_start
                ]
                chapter = next(
                    (
                        ancestor for ancestor in ancestor_sections
                        if re.match(r"Chương\s+", ancestor.get("title", ""), re.IGNORECASE)
                    ),
                    None,
                )
                article = next(
                    (
                        ancestor for ancestor in reversed(ancestor_sections)
                        if re.match(r"Điều\s+", ancestor.get("title", ""), re.IGNORECASE)
                    ),
                    None,
                )
                if re.match(r"Điều\s+", section.get("title", ""), re.IGNORECASE):
                    article = section
                result.append({
                    "source_location": section["location"],
                    "chapter_heading": chapter.get("title") if chapter else None,
                    "article_heading": article.get("title") if article else None,
                    "section_text": content,
                })
        return result

    def _build_section_amendment_prompt(
        self,
        metadata: Dict,
        document_result: Dict,
        known_targets: List[str],
        batch: List[Dict],
    ) -> str:
        """Build the focused prompt used for one batch of clean Markdown sections."""
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
        targets_json = json.dumps(known_targets, ensure_ascii=False)
        document_effective_date = document_result.get("effective_date")
        return f"""Bạn là chuyên gia trích xuất thao tác sửa đổi văn bản pháp luật Việt Nam.

VĂN BẢN HIỆN TẠI:
- document_id: {metadata.get('document_id') or ''}
- title: {metadata.get('title') or ''}
- effective_date chung: {document_effective_date or ''}
- các văn bản đích đã xác định ở document level: {targets_json}

Python đã chia CLEAN MARKDOWN thành các section và cung cấp chính xác source_location.
Chỉ đọc các section trong batch. Không sửa, rút gọn hoặc tự tạo source_location.
Section không chứa thao tác sửa đổi/bổ sung/thay thế/bãi bỏ thì không tạo operation.
Mỗi section còn có chapter_heading và article_heading do Python lấy từ Markdown.
Hai heading này chỉ cung cấp ngữ cảnh của section, không phải target và không được dùng
để tự suy đoán một operation không được phát biểu trong section_text.

RANH GIỚI VỚI QUAN HỆ CẤP VĂN BẢN:
- LLM document-level đã xử lý việc văn bản hiện tại bãi bỏ hoặc thay thế TOÀN BỘ
  một văn bản khác. LLM hiện tại TUYỆT ĐỐI KHÔNG tạo operation cấp document.
- Chỉ tạo operation khi target_locations xác định được ít nhất một Điều cụ thể;
  target nhỏ hơn phải giữ đủ đường dẫn Điều > Khoản > Điểm.
- Không bao giờ trả "Toàn văn" hoặc "Toàn bộ văn bản" trong target_locations.
- Ví dụ section "Bãi bỏ Pháp lệnh A, Luật B" mà không nêu Điều/Khoản/Điểm:
  trả operations=[] vì đây là quan hệ cấp văn bản.

PHÂN BIỆT ĐIỀU KHOẢN THI HÀNH VỚI OPERATION:
- Section chỉ quy định ngày có hiệu lực, điều khoản chuyển tiếp, trách nhiệm thi hành
  hoặc hướng dẫn thi hành thì KHÔNG tạo operation.
- Section trong Điều "Hiệu lực thi hành" hoặc "Điều khoản thi hành" chỉ tạo
  operation nếu nó tác động rõ tới Điều/Khoản/Điểm cụ thể của văn bản đích.
- Section chỉ bãi bỏ/thay thế tên một hoặc nhiều văn bản thì trả operations=[];
  quan hệ đó đã thuộc kết quả document-level.
- Không tạo operation chỉ vì section nhắc hoặc dẫn chiếu tới một văn bản khác.
- Cụm "Văn bản A đã được sửa đổi, bổ sung theo Văn bản B" chỉ mô tả lịch sử của A;
  không được biến B thành target bị bãi bỏ, thay thế hoặc sửa đổi.

VĂN BẢN MỚI THAY THẾ VĂN BẢN CŨ:
- Khi văn bản hiện tại là một luật/nghị định mới hoàn chỉnh và document-level xác
  định văn bản khác hết hiệu lực hoặc bị thay thế toàn bộ, TUYỆT ĐỐI KHÔNG suy
  diễn rằng từng Điều/Khoản cùng số của văn bản mới thay thế Điều/Khoản cùng số
  của văn bản cũ.
- Việc hai văn bản có cùng tên, cùng lĩnh vực, Điều/Khoản cùng số hoặc nội dung
  tương tự KHÔNG phải bằng chứng tạo amendment operation cấp Điều/Khoản.
- Các Điều/Khoản quy định nội dung độc lập của văn bản mới phải trả operations=[],
  kể cả khi chúng có vị trí tương ứng trong văn bản cũ.
- Quan hệ hết hiệu lực/thay thế toàn văn bản chỉ được xử lý tại document-level;
  không được phân rã quan hệ đó thành operation cho từng Điều/Khoản.
- Chỉ tạo operation khi chính section_text chứa đồng thời: động từ pháp lý rõ
  ràng; vị trí đích cụ thể; và văn bản đích hoặc cách dẫn chiếu xác định được văn
  bản đích.
- Danh sách văn bản đích chỉ dùng để chuẩn hóa mã khi section đã phát biểu
  operation rõ ràng. Không dùng danh sách này để tự suy luận operation.

QUY TẮC TARGET:
- target_document_id là mã văn bản bị tác động. Nếu section dùng "Luật này" hoặc "Nghị định này", đối chiếu danh sách văn bản đích ở trên.
- target_locations phải giữ đầy đủ Điều > Khoản > Điểm được nêu.
- "Bổ sung các khoản 4, 5 và 6 vào Điều 4" phải trả ba target:
  "Điều 4 > Khoản 4", "Điều 4 > Khoản 5", "Điều 4 > Khoản 6".
- "Khoản 14a" là Khoản 14a, không phải Khoản 14 > Điểm a.
- Chỉ dùng target cấp Điều khi toàn bộ Điều bị tác động.
- Nếu section chỉ bãi bỏ một nhóm quy định theo chủ đề nhưng không nêu chính xác
  Điều/Khoản/Điểm, KHÔNG tạo amendment operation; quan hệ partial cấp văn bản đã
  được LLM document-level xử lý.

QUY TẮC OPERATION:
- change_type chỉ là: sua_doi, sua_doi_cum_tu, bo_sung, thay_the, bai_bo.
- target_component chỉ là "title", "preamble", "all" hoặc null:
  * "title": chỉ đổi tên/tiêu đề của toàn Điều, không thay nội dung các Khoản;
  * "preamble": chỉ sửa đoạn mở đầu trực tiếp dưới tiêu đề Điều, giữ nguyên tiêu
    đề và toàn bộ các Khoản thuộc Điều;
  * "all": tác động toàn bộ đơn vị đích được nêu trong target_locations. Với target
    cấp Điều là toàn Điều; với target cấp Khoản/Điểm là toàn Khoản/Điểm đó;
  * null: chỉ dùng cho change_type=sua_doi_cum_tu vì phạm vi đã được xác định bằng
    text_operation cùng anchor_text/operand_text.
- "Sửa đổi tên Điều 101" phải trả target_component="title".
- "Đoạn đầu Điều 3 được sửa đổi" phải trả target_component="preamble".
- "Sửa đổi Điều 101 như sau" hoặc viết lại toàn Điều phải trả target_component="all".
- Mọi operation bổ sung/sửa đổi/thay thế/bãi bỏ Khoản hoặc Điểm phải trả
  target_component="all".
- Nếu một section đồng thời sửa đoạn đầu Điều và sửa Khoản thì phải tách thành
  hai operation riêng tương ứng: preamble và all.
- Nếu một section đồng thời đổi tên Điều, sửa Khoản và bổ sung Khoản thì phải
  tách thành các operation riêng tương ứng: title, all và all.
- Một operation chỉ chứa các target có cùng change_type.
- Một section vừa sửa target cũ vừa bổ sung target mới phải trả các operation riêng.
- "sửa đổi, bổ sung Khoản X như sau" nhưng viết lại nội dung Khoản X: change_type=sua_doi.
- Chỉ dùng change_type=thay_the khi section dùng rõ động từ pháp lý "thay thế".
  Cụm "được sửa đổi, bổ sung như sau" luôn là sua_doi, không phải thay_the,
  kể cả khi section viết lại toàn bộ nội dung đơn vị đích.
- Không chép nội dung thay thế của Điều/Khoản/Điểm vào JSON; Python sẽ lấy nguyên văn từ section.

QUY TẮC CHỨNG CỨ:
- evidence phải là một trích dẫn nguyên văn liên tục có thật trong section_text.
- Không diễn giải, tóm tắt hoặc tự viết evidence. Nếu không thể chép câu chứng
  minh operation từ section_text thì phải trả operations=[].
- Với thay_the, từ "thay thế" phải xuất hiện nguyên văn trong section_text.
- Với sua_doi, từ "sửa đổi" phải xuất hiện nguyên văn trong section_text.
- Với bo_sung, từ "bổ sung" phải xuất hiện nguyên văn trong section_text.
- Với bai_bo, từ "bãi bỏ" hoặc "hết hiệu lực" phải xuất hiện nguyên văn và gắn
  với vị trí đích cụ thể.

SỬA CỤM TỪ:
- Với change_type khác sua_doi_cum_tu, các field text_operation, anchor_text,
  operand_text phải là null.
- Với sua_doi_cum_tu, text_operation phải là một trong:
  * replace: thay anchor_text bằng operand_text;
  * insert_before: chèn operand_text ngay trước anchor_text;
  * insert_after: chèn operand_text ngay sau anchor_text;
  * delete: xóa operand_text; anchor_text phải là null.
- Với replace/insert, anchor_text và operand_text phải là trích dẫn nguyên văn.
- Với delete, chỉ operand_text phải là cụm từ nguyên văn được văn bản yêu cầu
  bãi bỏ; không tự tạo một anchor_text không có trong section.
- Một operation sửa cụm từ chỉ chứa các target dùng cùng text_operation, anchor_text và operand_text.
- Nhiều phép sửa cụm từ trong một section phải tách thành nhiều operation.

VÍ DỤ:
"Bổ sung cụm từ 'X' vào trước cụm từ 'Y' tại Điều 3, Điều 5" trả:
{{
  "change_type": "sua_doi_cum_tu",
  "target_component": null,
  "text_operation": "insert_before",
  "anchor_text": "Y",
  "operand_text": "X",
  "target_locations": ["Điều 3", "Điều 5"]
}}

"Bãi bỏ cụm từ 'thuê mặt nước' tại điểm h khoản 2 Điều 5" trả:
{{
  "change_type": "sua_doi_cum_tu",
  "target_component": null,
  "text_operation": "delete",
  "anchor_text": null,
  "operand_text": "thuê mặt nước",
  "target_locations": ["Điều 5 > Khoản 2 > Điểm h"]
}}

VÍ DỤ KHÔNG TẠO OPERATION:
Văn bản hiện tại là Luật Quản lý thuế mới; document-level cho biết Luật
38/2019/QH14 hết hiệu lực. Section chỉ ghi:
"Khoản 9. Nhiệm vụ, quyền hạn của Kiểm toán nhà nước: ..."
Mặc dù Luật 38/2019/QH14 cũng có Điều 39 Khoản 9, kết quả bắt buộc là:
{{"operations": []}}
Lý do: section là quy định độc lập của luật mới, không phát biểu sửa đổi hoặc
thay thế Điều 39 Khoản 9 của Luật 38/2019/QH14.

VÍ DỤ CÓ OPERATION:
"Điều 51. Bổ sung Điều 70a vào sau Điều 70 của Luật Kế toán số
88/2015/QH13..." tạo operation bo_sung với target_document_id="88/2015/QH13"
và target_locations=["Điều 70a"].

SCHEMA:
{{
  "operations": [
    {{
      "source_location": "chép đúng source_location do Python cung cấp",
      "target_document_id": "mã văn bản bị tác động",
      "target_locations": ["Điều/Khoản/Điểm bị tác động"],
      "change_type": "sua_doi|sua_doi_cum_tu|bo_sung|thay_the|bai_bo",
      "target_component": "title|preamble|all hoặc null",
      "text_operation": "replace|insert_before|insert_after|delete hoặc null",
      "anchor_text": "cụm neo nguyên văn hoặc null",
      "operand_text": "cụm thay/chèn/xóa nguyên văn hoặc null",
      "effective_date": "YYYY-MM-DD nếu section có ngày riêng, ngược lại null",
      "evidence": "trích dẫn nguyên văn liên tục từ section_text chứng minh operation"
    }}
  ]
}}

SECTION BATCH:
{batch_json}

Chỉ trả JSON object, không giải thích."""

    def extract_legal_effects(
        self,
        full_markdown: str,
        cleaned_markdown: str,
        metadata: Dict,
    ) -> Dict:
        """Extract document-level effects once, then amendments by clean sections."""
        if not self.enable_llm or self.gemini_model is None:
            return {
                "effective_date": None,
                "document_relations": [],
                "amendment_relations": [],
                "extraction_status": "skipped",
            }

        document_prompt = f"""Bạn là chuyên gia trích xuất hiệu lực và quan hệ CẤP VĂN BẢN của pháp luật Việt Nam.

Python đã xác định chắc chắn metadata của VĂN BẢN HIỆN TẠI:
- document_id: {metadata.get('document_id') or ''}
- title: {metadata.get('title') or ''}
- document_type: {metadata.get('document_type') or ''}
- issued_date: {metadata.get('issued_date') or ''}

Hãy đọc FULL MARKDOWN ĐÃ ÁP DỤNG MANIFEST và chỉ trả về metadata cấp văn bản.
KHÔNG trích xuất Điều/Khoản/Điểm bị sửa và KHÔNG trả amendment_relations.

I. NGÀY HIỆU LỰC CHUNG
- "effective_date" là ngày hiệu lực chung của VĂN BẢN HIỆN TẠI, định dạng YYYY-MM-DD.
- "từ ngày được thông qua" phải đối chiếu ngày văn bản hiện tại được thông qua trong toàn văn.
- "kể từ ngày ký ban hành" chỉ dùng issued_date của văn bản hiện tại khi câu nói đúng về văn bản hiện tại.
- Không lấy ngày hiệu lực của văn bản được viện dẫn, ngày trong biểu thuế, ngày chuyển tiếp hoặc ngày của một Điều ngoại lệ làm ngày hiệu lực chung.
- Không lấy ngày riêng của một Điều/Khoản làm effective_date chung.
- Nếu không xác định chắc chắn, trả null.

II. QUAN HỆ CẤP VĂN BẢN
- "document_relations" cho biết văn bản hiện tại sửa đổi, bổ sung, thay thế hoặc bãi bỏ văn bản nào.
- Không coi phần "Căn cứ...", câu viện dẫn "theo quy định tại..." hoặc tên của một văn bản được viện dẫn là quan hệ tác động.
- Khi một mục liệt kê có dạng "Bãi bỏ Văn bản A đã được sửa đổi, bổ sung theo
  Văn bản B", đối tượng bị bãi bỏ chỉ là Văn bản A. Văn bản B chỉ mô tả lịch sử
  sửa đổi của A và TUYỆT ĐỐI KHÔNG được tạo thành target_document_id riêng.
- Mỗi điểm a), b), c)... trong danh sách bãi bỏ phải được đọc theo đối tượng pháp
  lý chính đứng sau động từ "bãi bỏ". Không được biến mọi mã văn bản xuất hiện
  trong cùng câu thành các target độc lập.
- Chỉ tạo quan hệ riêng cho văn bản sửa đổi B nếu nội dung tuyên bố trực tiếp và
  độc lập rằng chính B cũng bị sửa đổi, thay thế hoặc bãi bỏ.
- Ví dụ: "Bãi bỏ Pháp lệnh số 35/2001/PL-UBTVQH10 đã được sửa đổi, bổ sung theo
  Pháp lệnh số 14/2004/PL-UBTVQH11" chỉ trả một relation có
  target_document_id="35/2001/PL-UBTVQH10"; không trả relation cho
  "14/2004/PL-UBTVQH11".
- "relation_types" là mảng gồm một hoặc nhiều giá trị: sua_doi, bo_sung, thay_the, bai_bo.
- Phải dùng ĐÚNG động từ pháp lý được quy định trong nội dung tác động:
  * "sửa đổi" -> sua_doi;
  * "bổ sung" -> bo_sung;
  * "sửa đổi, bổ sung" -> ["sua_doi", "bo_sung"];
  * "thay thế" -> thay_the;
  * "bãi bỏ" hoặc quy định rõ văn bản cũ hết hiệu lực -> bai_bo.
- Không được đổi "thay thế"/"bãi bỏ" thành "sửa đổi" hoặc "bổ sung". Hai loại này được Step 3 dùng để xác định nội dung cũ hết hiệu lực.
- "impact_scope" = "whole_document" chỉ khi toàn bộ văn bản đích bị thay thế hoặc bãi bỏ.
- "impact_scope" = "partial" khi chỉ một số Điều/Khoản/Cụm từ bị tác động.
- Phân biệt tuyệt đối "bãi bỏ Điều/Khoản" với "bãi bỏ toàn bộ văn bản".
- "effective_date" của relation chỉ điền khi quan hệ cấp văn bản có ngày riêng; nếu không có trả null.
- Nếu không có quan hệ cấp văn bản, trả document_relations=[].

JSON SCHEMA:
{{
  "effective_date": "YYYY-MM-DD hoặc null",
  "document_relations": [
    {{
      "source_document_id": "{metadata.get('document_id') or ''}",
      "target_document_id": "mã văn bản bị ảnh hưởng",
      "relation_types": ["sua_doi|bo_sung|thay_the|bai_bo"],
      "impact_scope": "partial|whole_document",
      "effective_date": "YYYY-MM-DD hoặc null",
      "can_cu_trich_doan": "trích đoạn nguyên văn chứng minh"
    }}
  ]
}}

FULL MARKDOWN:
{full_markdown}

Chỉ trả về JSON object, không giải thích."""
        document_result = self._call_llm_json(document_prompt, "DOCUMENT LEVEL")

        sections = self._collect_amendment_sections(cleaned_markdown)
        known_targets = [
            str(item.get("target_document_id") or "").strip()
            for item in document_result.get("document_relations") or []
            if isinstance(item, dict) and item.get("target_document_id")
        ]
        amendment_items = []
        total_batches = (len(sections) + self.amendment_batch_size - 1) // self.amendment_batch_size
        for start in range(0, len(sections), self.amendment_batch_size):
            batch = sections[start:start + self.amendment_batch_size]
            batch_number = start // self.amendment_batch_size + 1
            print(
                f"Trích amendment sections batch {batch_number}/{total_batches} "
                f"({len(batch)} sections)..."
            )
            section_prompt = self._build_section_amendment_prompt(
                metadata,
                document_result,
                known_targets,
                batch,
            )
            batch_result = self._call_llm_json(
                section_prompt,
                f"AMENDMENT BATCH {batch_number}/{total_batches}",
            )
            valid_sources = {item["source_location"] for item in batch}
            for operation in batch_result.get("operations") or []:
                if not isinstance(operation, dict):
                    continue
                source_location = str(operation.get("source_location") or "").strip()
                if source_location not in valid_sources:
                    raise ValueError(
                        f"LLM trả source_location ngoài batch: {source_location!r}."
                    )
                target_locations = operation.get("target_locations") or []
                if isinstance(target_locations, str):
                    target_locations = [target_locations]
                target_locations = [
                    str(location).strip()
                    for location in target_locations
                    if str(location).strip()
                ]
                if not target_locations or any(
                    not re.search(r'(?:^|\s*>\s*)Điều\s+\d+', location, re.IGNORECASE)
                    for location in target_locations
                ):
                    print(
                        "  Bỏ operation ngoài phạm vi Điều/Khoản/Điểm của LLM section: "
                        f"{source_location} -> {target_locations}"
                    )
                    continue
                amendment_items.append({
                    "van_ban_sua": metadata.get("document_id"),
                    "dieu_khoan_sua": source_location,
                    "van_ban_bi_sua": operation.get("target_document_id"),
                    "dieu_khoan_bi_sua": target_locations,
                    "loai_thay_doi": operation.get("change_type"),
                    "target_component": operation.get("target_component"),
                    "text_operation": operation.get("text_operation"),
                    "anchor_text": operation.get("anchor_text"),
                    "operand_text": operation.get("operand_text"),
                    "effective_date": operation.get("effective_date"),
                    "can_cu_trich_doan": operation.get("evidence") or "",
                })

        merged_result = {
            "effective_date": document_result.get("effective_date"),
            "document_relations": document_result.get("document_relations") or [],
            "amendment_relations": amendment_items,
        }
        return self._normalize_legal_effects(merged_result, metadata, cleaned_markdown)

    def _infer_target_level(self, target_location: str) -> str:
        """Infer the smallest legal unit explicitly named by a target location."""
        normalized = re.sub(r'\s+', ' ', str(target_location or '')).strip()
        if re.search(
            r'Điểm\s+[a-zđ](?:\d+|(?:\.\d+)+)?\b',
            normalized,
            re.IGNORECASE,
        ):
            return "diem"
        if re.search(r'Khoản\s+\d+[a-zđ]?\b', normalized, re.IGNORECASE):
            return "khoan"
        if re.search(r'Điều\s+\d+[a-zđ]?\b', normalized, re.IGNORECASE):
            return "dieu"
        if re.search(r'Toàn\s+(?:văn|bộ\s+văn\s+bản)', normalized, re.IGNORECASE):
            return "document"
        raise ValueError(f"Không xác định được cấp pháp lý của target: {target_location!r}")

    def _group_target_locations(self, target_locations: List[str]) -> List[List[str]]:
        """Group point targets only when they share the same parent clause."""
        groups: List[List[str]] = []
        point_group_indexes: Dict[str, int] = {}
        for location in target_locations:
            level = self._infer_target_level(location)
            if level != "diem":
                groups.append([location])
                continue

            parent = re.sub(
                r'\s*>\s*Điểm\s+[a-zđ](?:\d+|(?:\.\d+)+)?\s*$',
                '',
                location,
                flags=re.IGNORECASE,
            ).strip()
            # The full parent path keeps points in a different Khoản or Điều
            # isolated even when they occur in the same amendment section.
            key = re.sub(r'\s+', ' ', parent).casefold()
            if key not in point_group_indexes:
                point_group_indexes[key] = len(groups)
                groups.append([])
            groups[point_group_indexes[key]].append(location)
        return groups

    def _extract_replacement_text(self, source_section: str) -> str:
        """Remove the amendment heading and return its authoritative body text."""
        lines = source_section.splitlines()
        if lines and re.match(r'^#{1,3}\s+', lines[0]):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        quoted = re.fullmatch(
            r'[“"](.*)[”"](?:[\.;,:])?',
            body,
            flags=re.DOTALL,
        )
        if quoted:
            body = quoted.group(1).strip()
        return body

    def _target_replacement_pattern(self, target_location: str) -> Optional[re.Pattern]:
        """Build a pattern for the legal label that must open a replacement block."""
        normalized = re.sub(r'\s+', ' ', str(target_location or '')).strip()
        point_match = re.search(
            r'Điểm\s+([a-zđ](?:\d+|(?:\.\d+)+)?)\b',
            normalized,
            re.IGNORECASE,
        )
        if point_match:
            label = re.escape(point_match.group(1))
            return re.compile(rf'^\s*{label}\s*[\.)]', re.IGNORECASE | re.MULTILINE)

        clause_match = re.search(r'Khoản\s+(\d+[a-zđ]?)\b', normalized, re.IGNORECASE)
        if clause_match:
            label = re.escape(clause_match.group(1))
            return re.compile(rf'^\s*{label}\s*[\.)]', re.IGNORECASE | re.MULTILINE)

        article_match = re.search(r'Điều\s+(\d+[a-zđ]?)\b', normalized, re.IGNORECASE)
        if article_match:
            label = re.escape(article_match.group(1))
            return re.compile(
                rf'^\s*(?:##\s+)?Điều\s+{label}\b',
                re.IGNORECASE | re.MULTILINE,
            )
        return None

    def _extract_target_replacement_text(
        self,
        source_section: str,
        target_location: str,
        target_locations: List[str],
    ) -> str:
        """Extract one target unit, including when targets share a quoted block."""
        lines = source_section.splitlines()
        if lines and re.match(r'^#{1,3}\s+', lines[0]):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        opening_pattern = self._target_replacement_pattern(target_location)
        if opening_pattern is None:
            raise ValueError(
                f"Không tạo được nhãn tách nội dung cho target {target_location!r}."
            )

        # Amendment documents may use one quote per target or put several
        # Khoan/Diem targets inside one shared quote.
        quoted_blocks = re.findall(
            r'(?:^|\n)\s*[“"]\s*(.*?)\s*[”"](?:[\.;,:])?(?=\s*(?:\n|$))',
            body,
            flags=re.DOTALL,
        )
        candidate_blocks = quoted_blocks or [body]
        target_patterns = []
        for location in target_locations:
            pattern = self._target_replacement_pattern(location)
            if pattern is None:
                raise ValueError(
                    f"Không tạo được nhãn tách nội dung cho target {location!r}."
                )
            target_patterns.append((location, pattern))

        matches = []
        for block in candidate_blocks:
            boundaries = []
            for location, pattern in target_patterns:
                boundaries.extend(
                    (match.start(), location)
                    for match in pattern.finditer(block)
                )
            boundaries.sort(key=lambda item: item[0])
            for index, (start, location) in enumerate(boundaries):
                if location != target_location:
                    continue
                end = (
                    boundaries[index + 1][0]
                    if index + 1 < len(boundaries)
                    else len(block)
                )
                matches.append(block[start:end].strip())

        # A single-target amendment can safely use the complete replacement
        # body when its legal label is the opening label.
        if not matches and len(target_locations) == 1:
            replacement = self._extract_replacement_text(source_section)
            if opening_pattern.match(replacement):
                matches = [replacement]
        if len(matches) != 1:
            raise ValueError(
                "Không tách duy nhất nội dung mới cho "
                f"{target_location!r} trong amendment có {len(target_locations)} targets "
                f"(matches={len(matches)})."
            )
        return matches[0]

    def _extract_article_title_text(
        self,
        source_section: str,
        target_location: str,
    ) -> str:
        """Extract only the new Điều heading for target_component=title."""
        match = re.search(
            r'Điều\s+(\d+[a-zđ]?)\b',
            str(target_location or ''),
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"Không xác định được số Điều cần đổi tên: {target_location!r}."
            )
        article_number = re.escape(match.group(1))
        heading_pattern = re.compile(
            rf'^\s*(?:##\s+)?[“\"]?\s*(Điều\s+{article_number}\s*[\.:]?\s*[^\n”\"]+)',
            re.IGNORECASE | re.MULTILINE,
        )
        headings = [
            heading.strip().rstrip('”." ').strip()
            for heading in heading_pattern.findall(source_section)
        ]
        headings = list(dict.fromkeys(headings))
        if len(headings) != 1:
            raise ValueError(
                "Không tách duy nhất tiêu đề mới cho "
                f"{target_location!r} (matches={len(headings)})."
            )
        return headings[0]

    def _extract_article_preamble_text(
        self,
        source_section: str,
        target_location: str,
    ) -> str:
        """Extract the direct body below an Điều heading, before its first Khoản."""
        match = re.search(
            r'Điều\s+(\d+[a-zđ]?)\b',
            str(target_location or ''),
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"Không xác định được số Điều có đoạn đầu cần sửa: {target_location!r}."
            )
        article_number = re.escape(match.group(1))
        blocks = re.findall(
            r'(?:^|\n)\s*[“"]\s*(.*?)\s*[”"](?:[\.;,:])?(?=\s*(?:\n|$))',
            source_section,
            flags=re.DOTALL,
        )
        candidates = []
        for block in blocks or [source_section]:
            heading = re.search(
                rf'(?mi)^\s*(?:##\s+)?Điều\s+{article_number}\s*[\.:]?[^\n]*\n',
                block,
            )
            if not heading:
                continue
            body = block[heading.end():]
            first_clause = re.search(
                r'(?m)^\s*(?:Khoản\s+)?1\s*[\.)]\s+',
                body,
                re.IGNORECASE,
            )
            if first_clause:
                body = body[:first_clause.start()]
            body = body.strip().rstrip('”"').strip()
            if body:
                candidates.append(body)
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise ValueError(
                "Không tách duy nhất đoạn mở đầu mới cho "
                f"{target_location!r} (matches={len(candidates)})."
            )
        return candidates[0]

    def _normalize_verbatim_text(self, value: Any) -> str:
        """Normalize quotes and whitespace only for deterministic phrase checks."""
        text = str(value or '').translate(str.maketrans({
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
        }))
        return re.sub(r'\s+', ' ', text).strip().casefold()

    def _build_operation_id(self, operation: Dict) -> str:
        identity = {
            "source_document_id": operation.get("van_ban_sua"),
            "source_location": operation.get("dieu_khoan_sua"),
            "target_document_id": operation.get("van_ban_bi_sua"),
            "target_location": operation.get("target_location"),
            "target_locations": operation.get("dieu_khoan_bi_sua"),
            "change_type": operation.get("loai_thay_doi"),
            "target_component": operation.get("target_component"),
            "text_operation": operation.get("text_operation"),
            "anchor_text": operation.get("anchor_text"),
            "operand_text": operation.get("operand_text"),
            "old_text": operation.get("noi_dung_cu"),
            "new_text": operation.get("noi_dung_moi"),
            "effective_date": operation.get("effective_date"),
        }
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _normalize_phrase_operation(
        self,
        item: Dict,
        source_section: str,
        source_location: str,
    ) -> Tuple[str, Optional[str], str, str, str]:
        """Validate a structured text edit and derive legacy old/new fields."""
        text_operation = str(item.get("text_operation") or "").strip()
        anchor_text = str(item.get("anchor_text") or "").strip()
        operand_text = str(item.get("operand_text") or "").strip()
        allowed = {"replace", "insert_before", "insert_after", "delete"}
        if text_operation not in allowed or not operand_text:
            raise ValueError(
                f"sua_doi_cum_tu tại {source_location} phải có text_operation, "
                "và operand_text hợp lệ."
            )
        if text_operation != "delete" and not anchor_text:
            raise ValueError(
                f"{text_operation} tại {source_location} phải có anchor_text."
            )

        normalized_section = self._normalize_verbatim_text(source_section)
        required_quotes = [operand_text]
        if text_operation != "delete":
            required_quotes.append(anchor_text)
        if any(
            self._normalize_verbatim_text(value) not in normalized_section
            for value in required_quotes
        ):
            raise ValueError(
                f"Các trích dẫn bắt buộc không xuất hiện trong section "
                f"{source_location}."
            )

        def join_text(left: str, right: str) -> str:
            if not left:
                return right
            if not right:
                return left
            if left[-1].isspace() or right[0].isspace():
                return left + right
            return left + " " + right

        if text_operation == "delete":
            return text_operation, None, operand_text, operand_text, ""

        old_text = anchor_text
        if text_operation == "replace":
            new_text = operand_text
        elif text_operation == "insert_before":
            new_text = join_text(operand_text, anchor_text)
        elif text_operation == "insert_after":
            new_text = join_text(anchor_text, operand_text)
        return text_operation, anchor_text, operand_text, old_text, new_text

    def _normalize_legal_effects(
        self,
        result: Dict,
        metadata: Dict,
        full_markdown: str,
    ) -> Dict:
        source_id = metadata.get("document_id")
        effective_date = self._normalize_iso_date(result.get("effective_date"))
        allowed_types = {"sua_doi", "bo_sung", "thay_the", "bai_bo"}

        document_relations = []
        for item in result.get("document_relations") or []:
            if not isinstance(item, dict) or not item.get("target_document_id"):
                continue
            relation_types = item.get("relation_types") or []
            if isinstance(relation_types, str):
                relation_types = [relation_types]
            relation_types = [value for value in relation_types if value in allowed_types]
            if not relation_types:
                continue
            impact_scope = item.get("impact_scope")
            if impact_scope not in {"partial", "whole_document"}:
                impact_scope = "partial"
            document_relations.append({
                "source_document_id": source_id,
                "target_document_id": str(item["target_document_id"]).strip(),
                "relation_types": relation_types,
                "impact_scope": impact_scope,
                "effective_date": self._normalize_iso_date(item.get("effective_date")) or effective_date,
                "can_cu_trich_doan": str(item.get("can_cu_trich_doan") or "").strip(),
                "status": "active",
            })

        amendment_relations = []
        allowed_changes = allowed_types | {"sua_doi_cum_tu"}
        valid_target_ids = {
            relation["target_document_id"]
            for relation in document_relations
        }
        target_effective_dates = {
            relation["target_document_id"]: relation.get("effective_date")
            for relation in document_relations
        }
        for item in result.get("amendment_relations") or []:
            if not isinstance(item, dict):
                continue
            target_id = item.get("van_ban_bi_sua")
            change_type = item.get("loai_thay_doi")
            if not target_id or change_type not in allowed_changes:
                continue
            raw_target_component = item.get("target_component")
            target_component = (
                str(raw_target_component).strip().casefold()
                if raw_target_component is not None
                else None
            )
            if change_type == "sua_doi_cum_tu":
                if target_component not in {None, ""}:
                    raise ValueError(
                        "sua_doi_cum_tu phải có target_component=null, nhận được "
                        f"{raw_target_component!r}."
                    )
                target_component = None
            elif target_component not in {"title", "preamble", "all"}:
                raise ValueError(
                    f"Amendment {change_type} phải có target_component='title' "
                    f"'preamble' hoặc 'all', nhận được {raw_target_component!r}."
                )
            target_id = str(target_id).strip()
            if target_id not in valid_target_ids:
                continue
            target_locations = item.get("dieu_khoan_bi_sua") or []
            if isinstance(target_locations, str):
                target_locations = [target_locations]
            target_locations = [
                str(value).strip()
                for value in target_locations
                if str(value).strip()
            ]
            if not target_locations:
                raise ValueError(
                    f"Amendment tới {target_id} thiếu dieu_khoan_bi_sua."
                )
            source_location = str(item.get("dieu_khoan_sua") or "").strip()
            if not source_location:
                raise ValueError(
                    f"Amendment tới {target_id} thiếu dieu_khoan_sua."
                )
            source_section = self._extract_markdown_section(
                full_markdown,
                source_location,
            )
            source_heading = source_section.splitlines()[0]
            if change_type == "thay_the" and not re.search(
                r'\bthay\s+thế\b',
                source_heading,
                re.IGNORECASE,
            ):
                if re.search(r'\bsửa\s+đổi\b', source_heading, re.IGNORECASE):
                    print(
                        "  Chuẩn hóa change_type thay_the -> sua_doi theo heading: "
                        f"{source_heading}"
                    )
                    change_type = "sua_doi"
                else:
                    raise ValueError(
                        "LLM trả change_type='thay_the' nhưng heading không có "
                        f"động từ 'thay thế': {source_heading}"
                    )
            relation_effective_date = (
                self._normalize_iso_date(item.get("effective_date"))
                or target_effective_dates.get(str(target_id).strip())
                or effective_date
            )

            old_text = None
            new_text = None
            replacement_text = None
            text_operation = None
            anchor_text = None
            operand_text = None
            if change_type == "sua_doi_cum_tu":
                (
                    text_operation,
                    anchor_text,
                    operand_text,
                    old_text,
                    new_text,
                ) = self._normalize_phrase_operation(
                    item,
                    source_section,
                    source_location,
                )
            # Multiple points in the same parent clause are one reconstruction
            # operation. Article/clause targets and points under another parent
            # remain separate operations.
            for target_group in self._group_target_locations(target_locations):
                target_location = target_group[0]
                target_change_type = change_type
                target_level = self._infer_target_level(target_location)
                if target_component == "title" and (
                    target_level != "dieu" or target_change_type != "sua_doi"
                ):
                    raise ValueError(
                        "target_component='title' chỉ hợp lệ với sua_doi ở cấp Điều: "
                        f"{target_location}."
                    )
                if target_component == "preamble" and (
                    target_level != "dieu" or target_change_type != "sua_doi"
                ):
                    raise ValueError(
                        "target_component='preamble' chỉ hợp lệ với sua_doi ở cấp Điều: "
                        f"{target_location}."
                    )
                target_replacement_text = replacement_text
                if target_change_type not in {"sua_doi_cum_tu", "bai_bo"}:
                    if target_component == "title":
                        target_replacement_text = self._extract_article_title_text(
                            source_section,
                            target_location,
                        )
                    elif target_component == "preamble":
                        target_replacement_text = self._extract_article_preamble_text(
                            source_section,
                            target_location,
                        )
                    else:
                        replacement_blocks = [
                            self._extract_target_replacement_text(
                                source_section,
                                grouped_location,
                                target_locations,
                            )
                            for grouped_location in target_group
                        ]
                        target_replacement_text = "\n\n".join(replacement_blocks)
                    if not target_replacement_text:
                        raise ValueError(
                            "Không tách được nội dung thay thế tại "
                            f"{source_location} cho {target_location}."
                        )
                operation = {
                    "van_ban_sua": source_id,
                    "dieu_khoan_sua": source_location,
                    "van_ban_bi_sua": target_id,
                    "target_location": target_location,
                    "target_level": target_level,
                    "target_component": target_component,
                    # Compatibility with the current Step 3 during the transition.
                    "dieu_khoan_bi_sua": target_group,
                    "loai_thay_doi": target_change_type,
                    "text_operation": text_operation,
                    "anchor_text": anchor_text,
                    "operand_text": operand_text,
                    "noi_dung_cu": old_text,
                    "noi_dung_moi": (
                        new_text if target_change_type == "sua_doi_cum_tu"
                        else target_replacement_text
                    ),
                    "replacement_text": target_replacement_text,
                    "amendment_source_section": source_section,
                    "effective_date": relation_effective_date,
                    "can_cu_trich_doan": source_section.splitlines()[0],
                    "status": "active",
                }
                operation["operation_id"] = self._build_operation_id(operation)
                amendment_relations.append(operation)

        return {
            "effective_date": effective_date,
            "document_relations": self._deduplicate_dicts(document_relations),
            "amendment_relations": self._deduplicate_dicts(amendment_relations),
            "extraction_status": "completed",
        }

    def _deduplicate_dicts(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _build_affects_documents(self, document_relations: List[Dict]) -> List[Dict]:
        affects = []
        for relation in document_relations:
            affects.append({
                "target_document_id": relation["target_document_id"],
                "relation_types": relation["relation_types"],
                "impact_scope": relation["impact_scope"],
                "effective_date": relation.get("effective_date"),
                "status": relation.get("status", "active"),
            })
        return self._deduplicate_dicts(affects)

    def _normalize_manifest_location(self, location: str) -> str:
        location = re.sub(r'\s+', ' ', location.strip())
        appendix_match = re.search(
            r'Phụ\s*lục(?:\s+([IVXLCDM]+|\d+))?',
            location,
            re.IGNORECASE,
        )
        if appendix_match:
            suffix = appendix_match.group(1)
            return f"Phụ lục {suffix.upper()}" if suffix else "Phụ lục"

        dieu_match = re.search(r'Điều\s+(\d+[a-zđ]?)', location, re.IGNORECASE)
        khoan_match = re.search(r'Khoản\s+(\d+[a-zđ]?)', location, re.IGNORECASE)
        parts = []
        if dieu_match:
            parts.append(f"Điều {dieu_match.group(1).lower()}")
        if khoan_match:
            if not dieu_match:
                raise ValueError(f"Selector cấp Khoản phải kèm Điều: {location}")
            parts.append(f"Khoản {khoan_match.group(1).lower()}")
        if not parts:
            raise ValueError(f"Manifest location không được hỗ trợ: {location}")
        return " > ".join(parts)

    def _build_markdown_sections(self, markdown_text: str) -> Tuple[List[str], List[Dict]]:
        lines = markdown_text.splitlines()
        sections = []
        current_chuong = None
        current_dieu = None
        current_appendix = None

        for index, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,3})\s+(.+?)\s*$', line)
            if not heading_match:
                continue
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            location = None
            ancestors = []

            chuong_match = re.match(
                r'Chương\s+([IVXLCDM]+|\d+)',
                title,
                re.IGNORECASE,
            )
            appendix_match = re.match(
                r'Phụ\s*lục(?:\s+([IVXLCDM]+|\d+))?',
                title,
                re.IGNORECASE,
            )
            dieu_match = re.match(r'Điều\s+(\d+[a-zđ]?)', title, re.IGNORECASE)
            khoan_match = re.match(r'Khoản\s+(\d+[a-zđ]?)', title, re.IGNORECASE)

            if level == 1 and chuong_match:
                current_chuong = index
                current_dieu = None
                current_appendix = None
            elif level == 1 and appendix_match:
                suffix = appendix_match.group(1)
                location = f"Phụ lục {suffix.upper()}" if suffix else "Phụ lục"
                current_appendix = index
                current_chuong = None
                current_dieu = None
            elif level == 2 and dieu_match:
                location = f"Điều {dieu_match.group(1).lower()}"
                if current_chuong is not None:
                    ancestors.append(current_chuong)
                current_dieu = index
                current_appendix = None
            elif level == 3 and khoan_match and current_dieu is not None:
                article_section = next(
                    section for section in reversed(sections)
                    if section["start"] == current_dieu
                )
                location = (
                    f'{article_section["location"]} > '
                    f'Khoản {khoan_match.group(1).lower()}'
                )
                if current_chuong is not None:
                    ancestors.append(current_chuong)
                ancestors.append(current_dieu)
            elif current_appendix is not None:
                ancestors.append(current_appendix)

            sections.append({
                "start": index,
                "end": len(lines),
                "level": level,
                "title": title,
                "location": location,
                "ancestors": ancestors,
            })

        for position, section in enumerate(sections):
            for next_section in sections[position + 1:]:
                if next_section["level"] <= section["level"]:
                    section["end"] = next_section["start"]
                    break
        return lines, sections

    def _extract_markdown_section(
        self,
        markdown_text: str,
        source_location: str,
    ) -> str:
        """Return the exact heading and body identified by an LLM source location."""
        normalized_location = self._normalize_manifest_location(source_location)
        lines, sections = self._build_markdown_sections(markdown_text)
        matches = [
            section
            for section in sections
            if section.get("location") == normalized_location
        ]
        if len(matches) != 1:
            raise ValueError(
                f"dieu_khoan_sua không match duy nhất một Markdown section: "
                f"{source_location!r} (matches={len(matches)})"
            )
        section = matches[0]
        content = "\n".join(lines[section["start"]:section["end"]]).strip()
        if not content:
            raise ValueError(f"Markdown section rỗng: {source_location!r}")
        return content

    def apply_manifest_scope(self, markdown_text: str, document_id: str) -> str:
        manifest_entry = self.corpus_manifest.get(document_id)
        if manifest_entry is None:
            return markdown_text
        selectors = manifest_entry.get("include_locations") or []
        if not selectors:
            return markdown_text

        lines, sections = self._build_markdown_sections(markdown_text)
        normalized_sections = {
            section["location"]: section
            for section in sections
            if section.get("location")
        }
        normalized_selectors = [
            self._normalize_manifest_location(selector)
            for selector in selectors
        ]
        unmatched = [
            selector for selector, normalized in zip(selectors, normalized_selectors)
            if normalized not in normalized_sections
        ]
        if unmatched:
            raise ValueError(
                f"Manifest không match {document_id}: {', '.join(unmatched)}"
            )

        included_indexes = set()
        first_heading = sections[0]["start"] if sections else len(lines)
        included_indexes.update(range(first_heading))
        for normalized in normalized_selectors:
            section = normalized_sections[normalized]
            included_indexes.update(section["ancestors"])
            included_indexes.update(range(section["start"], section["end"]))

        scoped_markdown = '\n'.join(
            line for index, line in enumerate(lines)
            if index in included_indexes
        ).strip()
        return scoped_markdown

    def process_file(self, input_path: str, output_path: str) -> str:
        """Xử lý tuần tự một DOCX và lưu full Markdown, clean Markdown, JSON."""
        print(f"Đọc file: {input_path}")
        text = self.read_docx(input_path)

        print(f"Chuyển đổi sang Markdown...")
        original_markdown = self.convert_to_markdown(text)

        print("Trích xuất metadata deterministic...")
        metadata, _ = self.extract_metadata(
            original_markdown,
            os.path.basename(input_path),
        )
        if not metadata.get("document_id"):
            raise ValueError("Không tìm thấy document_id trong phần đầu văn bản.")
        if not metadata.get("issued_date"):
            raise ValueError("Không tìm thấy ngày ban hành trong phần đầu văn bản.")

        full_markdown = self.apply_manifest_scope(
            original_markdown,
            metadata["document_id"],
        )
        _, cleaned_markdown = self.extract_metadata(
            full_markdown,
            os.path.basename(input_path),
        )
        full_path = output_path.replace('.md', '_full.md')
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(full_markdown)
        print(f"Lưu full Markdown: {full_path}")

        if self.enable_llm:
            print("Trích xuất document level và amendment theo section bằng LLM...")
        else:
            print("Bỏ qua LLM (--no-llm)...")
        legal_effects = self.extract_legal_effects(
            full_markdown,
            cleaned_markdown,
            metadata,
        )
        metadata["effective_date"] = legal_effects.get("effective_date")
        metadata["amendment_relations"] = legal_effects.get("amendment_relations", [])
        metadata["affects_documents"] = self._build_affects_documents(
            legal_effects.get("document_relations", [])
        )
        metadata_path = output_path.replace('.md', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_markdown)
        print(f"Lưu metadata: {metadata_path}")
        print(f"Lưu Markdown clean: {output_path}")
        return cleaned_markdown

    def inspect_document(self, input_path: str) -> Dict:
        """Đọc metadata cơ bản để sắp batch mà không gọi LLM."""
        text = self.read_docx(input_path)
        markdown = self.convert_to_markdown(text)
        metadata, _ = self.extract_metadata(markdown, os.path.basename(input_path))
        return metadata

    def process_directory(self, input_dir: str, output_dir: str) -> List[str]:
        """Xử lý lần lượt theo ngày ban hành tăng dần, không gọi LLM song song."""
        os.makedirs(output_dir, exist_ok=True)
        files = [
            path for path in Path(input_dir).glob("*.docx")
            if not path.name.startswith("~$")
        ]
        inspected = []
        for path in files:
            metadata = self.inspect_document(str(path))
            inspected.append((metadata.get("issued_date") or "9999-12-31", path, metadata))
        inspected.sort(key=lambda item: (item[0], item[1].name.lower()))

        print("\nThứ tự xử lý theo ngày ban hành:")
        for issued_date, path, metadata in inspected:
            print(f"  {issued_date} | {metadata.get('document_id') or '?'} | {path.name}")

        metadata_paths = []
        for _, path, _ in inspected:
            document_dir = os.path.join(output_dir, path.stem)
            os.makedirs(document_dir, exist_ok=True)
            output_path = os.path.join(document_dir, f"{path.stem}.md")
            self.process_file(str(path), output_path)
            metadata_paths.append(output_path.replace('.md', '_metadata.json'))

        self.sync_inverse_relationships(output_dir)
        return metadata_paths

    def sync_inverse_relationships(self, output_dir: str) -> None:
        """Đồng bộ affected_by và trạng thái toàn văn bản sau khi xử lý batch."""
        metadata_paths = list(Path(output_dir).rglob("*_metadata.json"))
        records = {}
        path_by_id = {}
        for path in metadata_paths:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    metadata = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            document_id = metadata.get("document_id")
            if not document_id:
                continue
            metadata["affected_by_documents"] = []
            records[document_id] = metadata
            path_by_id[document_id] = path

        today = date.today().isoformat()
        for source_id, source in records.items():
            for relation in source.get("affects_documents") or []:
                target_id = relation.get("target_document_id")
                target = records.get(target_id)
                if target is None:
                    continue
                inverse = {
                    "source_document_id": source_id,
                    "relation_types": relation.get("relation_types", []),
                    "impact_scope": relation.get("impact_scope", "partial"),
                    "effective_date": relation.get("effective_date"),
                    "status": relation.get("status", "active"),
                }
                if inverse not in target["affected_by_documents"]:
                    target["affected_by_documents"].append(inverse)

                relation_types = set(relation.get("relation_types") or [])
                relation_date = relation.get("effective_date")
                ends_document = bool(relation_types & {"thay_the", "bai_bo"})
                if (
                    relation.get("impact_scope") == "whole_document"
                    and ends_document
                    and relation_date
                    and relation_date <= today
                ):
                    target["status"] = "inactive"
                    target["expiration_date"] = relation_date

        for document_id, metadata in records.items():
            with open(path_by_id[document_id], 'w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"Đã đồng bộ quan hệ hai chiều cho {len(records)} văn bản.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản pháp luật theo thứ tự thời gian.")
    parser.add_argument(
        "--file",
        help=(
            "Tên văn bản trong documents, có thể bỏ đuôi .docx "
            "(ví dụ: 04_2007_QH12_59652)."
        ),
    )
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--manifest", default=CORPUS_MANIFEST_PATH)
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Giữ toàn bộ Markdown, không áp dụng corpus manifest.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Không gọi Gemini; dùng để kiểm tra đọc DOCX/Markdown/metadata cơ bản.",
    )
    args = parser.parse_args()

    converter = MarkdownHierarchyConverter(
        enable_llm=not args.no_llm,
        manifest_path=None if args.no_manifest else args.manifest,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    if args.file:
        input_path = Path(args.file)
        if input_path.suffix.lower() != ".docx":
            input_path = Path(args.input_dir) / f"{input_path.name}.docx"
        elif not input_path.is_absolute() and not input_path.exists():
            input_path = Path(args.input_dir) / input_path.name
        if not input_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy văn bản: {input_path}")
        document_dir = Path(args.output_dir) / input_path.stem
        document_dir.mkdir(parents=True, exist_ok=True)
        output_path = document_dir / f"{input_path.stem}.md"
        converter.process_file(str(input_path), str(output_path))
        converter.sync_inverse_relationships(args.output_dir)
    else:
        converter.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

