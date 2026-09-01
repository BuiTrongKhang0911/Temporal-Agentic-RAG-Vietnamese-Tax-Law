"""
Bước 2: Generate Document Summary (SAC Tier 1)
- Dùng LLM (Gemini Flash) tạo "document fingerprint"
- Lưu summary để inject vào chunks ở bước 3
"""
import argparse

import google.generativeai as genai
import json
import os
from pathlib import Path
from config import GOOGLE_API_KEY, OUTPUT_DIR, SAC_TIER1_GEMINI_MODEL


class SACGenerator:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("Thiếu GOOGLE_API_KEY trong file .env ở thư mục gốc")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(SAC_TIER1_GEMINI_MODEL)
        
        # Prompt đơn giản theo SAC paper - Generic approach (Section 5.3, 5.2)
        # Generic summaries outperform expert-guided summaries
        # Lý do: broad semantic cues > dense, structured, legally precise summaries
        # LLM được tự do sinh summary, không bắt theo format cứng nhắc
        self.summary_prompt = """
Bạn là chuyên gia tóm tắt văn bản pháp luật Việt Nam.

NHIỆM VỤ: Tạo "dấu vân tay" để phân biệt văn bản này 
với các văn bản khác cùng lĩnh vực trong cùng hệ thống.

METADATA:
{metadata}

NỘI DUNG VĂN BẢN (mẫu):
{document_text}

QUY TẮC THEO LOẠI VĂN BẢN:

Nếu là LUẬT → Nêu: quy định gốc, các mức số liệu 
đặc trưng (thuế suất, ngưỡng tiền, số ngày), 
đối tượng áp dụng, điểm mới so với luật trước.

Nếu là NGHỊ ĐỊNH → Nêu: các nội dung hướng dẫn 
CỤ THỂ mà Luật không có (hồ sơ, thủ tục, điều kiện 
chi tiết, công thức tính, danh mục đính kèm).
KHÔNG lặp lại định nghĩa đã có trong Luật.

Nếu là THÔNG TƯ → Nêu rõ sắc thuế hoặc lĩnh vực được
hướng dẫn và các nhóm quy định cốt lõi, như đối tượng
chịu/không chịu thuế, người nộp thuế, căn cứ và phương
pháp tính, thuế suất, khấu trừ, hoàn thuế, kê khai và
quyết toán. Không giới hạn nội dung ở biểu mẫu, thủ tục
hoặc hướng dẫn kỹ thuật. Chỉ nêu biểu mẫu, hồ sơ và thời
hạn nếu đó là phạm vi chính hoặc đặc điểm phân biệt nổi
bật của văn bản.

YÊU CẦU FORMAT:
- Tối đa 200 ký tự
- Câu hoàn chỉnh, không cắt dở
- Tiếng Việt tự nhiên
- BẮT BUỘC mở đầu bằng tên/chủ đề cụ thể của văn bản lấy từ metadata,
  ví dụ: "Luật Thuế thu nhập cá nhân quy định...", "Luật Quản lý
  thuế quy định...", "Nghị định về chính sách thuế đối với hộ kinh
  doanh quy định...", "Thông tư hướng dẫn thuế giá trị gia tăng về...".
- KHÔNG được mở đầu chung chung bằng "Luật quy định...", "Nghị định
  quy định..." hoặc "Văn bản quy định...".
- Sau phần nhận diện văn bản, ưu tiên các đặc điểm phân biệt và nội
  dung nổi bật của chính văn bản; việc dùng số liệu tuân theo quy tắc
  riêng cho từng loại văn bản dưới đây.
- Với LUẬT hoặc NGHỊ ĐỊNH, ưu tiên 1-2 con số/mức cụ thể
  nếu đó là đặc điểm trung tâm và có trong nội dung.
- Với THÔNG TƯ, KHÔNG bắt buộc đưa số liệu vào SAC Tier 1.
  Không đưa ngưỡng tiền, thuế suất hoặc thời hạn dễ bị sửa
  đổi; các số liệu theo từng phiên bản sẽ nằm ở SAC Tier 2
  và nguyên văn Điều/Khoản.
- KHÔNG suy luận điểm mới so với văn bản trước nếu nội dung
  đầu vào không nêu rõ.
- KHÔNG nêu số hiệu, ngày hiệu lực (đã có trong 
  hard prefix)

Chỉ xuất ra phần tóm tắt:
"""
    
    def generate_summary(self, document_text: str, document_metadata: dict = None) -> str:
        """
        Generate SAC Tier 1 với hard prefix để phân biệt documents
        Format: [document_id | Hiệu lực: date] + LLM generic summary
        """
        # Chuẩn bị metadata dạng text
        metadata_text = "KHÔNG CÓ METADATA"
        if document_metadata:
            metadata_text = json.dumps(document_metadata, ensure_ascii=False, indent=2)
        
        # Build prompt với CẢ .md VÀ .json
        prompt = self.summary_prompt.format(
            metadata=metadata_text,
            document_text=document_text
        )
        
        try:
            response = self.model.generate_content(prompt)
            llm_summary = response.text.strip()
            
            # Tạo hard prefix từ metadata
            prefix = ""
            if document_metadata:
                doc_id = document_metadata.get("document_id", "")
                effective_date = document_metadata.get("effective_date", "")
                if doc_id:
                    prefix = f"[{doc_id}"
                    if effective_date:
                        prefix += f" | Hiệu lực: {effective_date}"
                    prefix += "] "
            
            # GIỮ NGUYÊN output của LLM - KHÔNG CẮT
            final_summary = prefix + llm_summary
            return final_summary
            
        except Exception as e:
            print(f"Lỗi khi tạo summary: {e}")
            # Fallback: tạo summary cơ bản từ metadata
            if document_metadata:
                title = document_metadata.get("title", "Văn bản pháp luật")
                doc_id = document_metadata.get("document_id", "")
                return f"Văn bản: {title}. Số hiệu: {doc_id}."
            return "Văn bản pháp luật."
    
    def process_markdown_file(self, markdown_file: str) -> dict:
        """
        Process một file markdown và tạo summary
        Returns: dict với summary và metadata
        """
        if os.path.basename(markdown_file).endswith('_full.md'):
            raise ValueError(
                "Step 2 chỉ nhận Markdown clean, không nhận file _full.md."
            )

        # Đọc TOÀN BỘ markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # Đọc metadata nếu có
        metadata_file = markdown_file.replace('.md', '_metadata.json')
        document_metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                document_metadata = json.load(f)
        
        print(f"\nProcessing: {os.path.basename(markdown_file)}")
        print(f"  Document length: {len(markdown_text)} chars")
        
        # Generate summary: Hard prefix (code) + Soft content (LLM)
        print(f"  Generating document summary...")
        summary = self.generate_summary(markdown_text, document_metadata)
        print(f"  Summary ({len(summary)} chars): {summary}")
        
        return {
            "source_file": os.path.basename(markdown_file),
            "document_summary": summary,
            "document_metadata": document_metadata
        }
    
    def save_summary(self, summary_data: dict, output_path: str):
        """Lưu summary data"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {output_path}")


def collect_markdown_files(output_dir: str, file_name: str | None) -> list[str]:
    root = Path(output_dir)
    if file_name:
        stem = Path(file_name).stem
        candidates = [root / stem / f"{stem}.md", root / f"{stem}.md"]
        markdown_paths = [path for path in candidates if path.is_file()]
        if not markdown_paths:
            raise FileNotFoundError(
                f"Không tìm thấy output Step 1 cho văn bản: {stem}"
            )
    else:
        markdown_paths = sorted(
            path for path in root.rglob("*.md")
            if not path.name.endswith("_full.md")
        ) if root.exists() else []

    selected = []
    for path in markdown_paths:
        metadata_path = path.with_name(f"{path.stem}_metadata.json")
        if not metadata_path.is_file():
            print(f"⚠ Bỏ qua {path.name}: thiếu metadata.")
            continue
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not metadata.get("embed_in_qdrant", True):
            print(f"Bỏ qua SAC: {path.name} (embed_in_qdrant=false)")
            continue
        selected.append(str(path))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAC Tier 1 summaries.")
    parser.add_argument(
        "--file",
        help="Tên văn bản đã xử lý ở Step 1, không cần đuôi file.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    markdown_files = collect_markdown_files(args.output_dir, args.file)
    if not markdown_files:
        if args.file:
            print(f"Không cần tạo SAC Tier 1 cho {Path(args.file).stem}.")
        else:
            print(f"⚠ Không tìm thấy Markdown cần tạo SAC trong {args.output_dir}")
            print("Hãy chạy step1_markdown_hierarchy.py trước")
        return

    generator = SACGenerator()
    print(f"{'='*60}")
    print("SAC Generation - Step 2")
    print(f"Tìm thấy {len(markdown_files)} file markdown")
    print(f"{'='*60}")

    for markdown_file in markdown_files:
        summary_data = generator.process_markdown_file(markdown_file)
        summary_path = str(Path(markdown_file).with_name(
            f"{Path(markdown_file).stem}_summary.json"
        ))
        generator.save_summary(summary_data, summary_path)
        print()

    print(f"{'='*60}")
    print(f"Hoàn thành! Summary files đã được tạo trong {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

