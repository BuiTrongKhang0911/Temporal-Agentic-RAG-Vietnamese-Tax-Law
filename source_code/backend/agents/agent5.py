"""Grounded Agent 5 model adapters."""

from __future__ import annotations

import re
from typing import Any

import google.generativeai as genai


AGENT5_SYSTEM_PROMPT = """Bạn là Agent 5, chuyên tổng hợp câu trả lời pháp luật thuế Việt Nam.

NGUYÊN TẮC BẮT BUỘC
- Chỉ dùng câu hỏi và evidence được cung cấp. Không dùng kiến thức nhớ sẵn.
- Trả lời trực tiếp, sau đó giải thích ngắn gọn và nêu căn cứ.
- Không tự tạo hoặc sửa tên, số hiệu, Điều, Khoản hay ngày hiệu lực.
- Khi viện dẫn, chỉ dùng nhãn [Evidence N] đã xuất hiện trong đầu vào.
- Mọi kết luận pháp lý và con số phải được một evidence trực tiếp hỗ trợ.
- Không tự thực hiện số học. Chỉ dùng phép tính và kết quả đã được Python
  xác minh trong mục CALCULATION TOOL; trình bày lại phép tính đó khi có.
- Nếu câu hỏi cần tính nhưng CALCULATION TOOL không có kết quả hợp lệ, nêu rõ
  chưa đủ dữ kiện hoặc phép tính chưa được xác minh.
- Nếu thông tin còn thiếu ảnh hưởng đến kết luận, nói rõ chưa đủ căn cứ và
  không tự điền phần thiếu.
- Không dùng evidence nằm ngoài phạm vi thời gian đã khóa.
- Không lặp lại toàn bộ evidence và không thêm lời chào dài.

Trước khi trả lời, tự kiểm tra:
1. Đúng đối tượng và đúng loại thuế hay chưa?
2. Mỗi căn cứ có đúng nhãn [Evidence N] hay chưa?
3. Có con số hoặc điều kiện nào không nằm trong đầu vào hay không?
4. Có bỏ qua thông tin còn thiếu đã được cảnh báo hay không?

Chỉ trả về câu trả lời cuối cùng bằng tiếng Việt."""


_EVIDENCE_LABEL_RE = re.compile(r"\[Evidence\s+(\d+)\]", re.IGNORECASE)

CALCULATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_calculations",
        "description": (
            "Submit only arithmetic that is required to answer the question. "
            "Return an empty list when no arithmetic is required or operands "
            "are incomplete."
        ),
        "parameters": {
            "type": "object",
            "required": ["calculations"],
            "properties": {
                "calculations": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "required": [
                            "label",
                            "expression",
                            "variables",
                            "unit",
                        ],
                        "properties": {
                            "label": {"type": "string"},
                            "expression": {
                                "type": "string",
                                "description": (
                                    "Expression using variable names and only "
                                    "+, -, *, / and parentheses."
                                ),
                            },
                            "variables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "name",
                                        "value",
                                        "source",
                                        "source_text",
                                    ],
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                                            "description": (
                                                "ASCII variable name without Vietnamese "
                                                "diacritics, for example thue_gtgt."
                                            ),
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": (
                                                "Normalized decimal value. "
                                                "Percentages use decimal form."
                                            ),
                                        },
                                        "source": {
                                            "type": "string",
                                            "description": (
                                                "question, evidence_N, or an earlier "
                                                "verified calculation_N."
                                            ),
                                        },
                                        "source_text": {
                                            "type": "string",
                                            "description": (
                                                "Continuous exact text copied "
                                                "from the named source."
                                            ),
                                        },
                                        "occurrence": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "description": (
                                                "1-based position of the intended "
                                                "number when source_text contains "
                                                "multiple numbers. Default 1."
                                            ),
                                        },
                                    },
                                },
                            },
                            "unit": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}


class GeminiFinalGenerator:
    """Expose the Gemini model interface used by the orchestrator."""

    def __init__(self, *, api_key: str, model: str) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.generate_content(*args, **kwargs)



