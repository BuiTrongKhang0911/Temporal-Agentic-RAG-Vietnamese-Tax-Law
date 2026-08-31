"""Grounded Agent 5 model adapters."""

from __future__ import annotations

import os
import re
from typing import Any

import google.generativeai as genai
import requests


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


class OllamaFinalGenerator:
    """Call a remote/local Ollama model for evidence-bound answer synthesis."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        token: str | None = None,
        timeout: int = 600,
        num_ctx: int = 32768,
        max_output: int = 2048,
        keep_alive: str = "30m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token = token
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.max_output = max_output
        self.keep_alive = keep_alive

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def plan_calculations(self, prompt: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Bạn lập kế hoạch số học cho câu trả lời pháp luật. "
                            "Không kết luận pháp lý. Mỗi toán hạng phải có "
                            "source_text nguyên văn liên tục từ question hoặc "
                            "evidence_N. Nếu một span có nhiều số, phải chỉ rõ "
                            "occurrence. Các phép tính phải xếp theo thứ tự; phép "
                            "tính sau được dùng kết quả trước bằng source "
                            "calculation_N và source_text đúng bằng label của phép "
                            "tính đó; N là vị trí 1-based trong mảng calculations. "
                            "Tên biến chỉ dùng ASCII không dấu, chữ số "
                            "và dấu gạch dưới. Không đủ toán hạng thì trả mảng rỗng."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "tools": [CALCULATION_TOOL],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.num_ctx,
                    "num_predict": min(self.max_output, 2048),
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        message = (response.json().get("message") or {})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {"calculations": []}
        arguments = (
            (tool_calls[0].get("function") or {}).get("arguments") or {}
        )
        if isinstance(arguments, str):
            import json

            arguments = json.loads(arguments)
        return arguments if isinstance(arguments, dict) else {"calculations": []}

    def generate(self, prompt: str, *, evidence_count: int) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": AGENT5_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.max_output,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        answer = str((data.get("message") or {}).get("content") or "").strip()
        if not answer:
            raise ValueError("Agent 5 Qwen không trả về nội dung")

        invalid_labels = sorted(
            {
                int(match.group(1))
                for match in _EVIDENCE_LABEL_RE.finditer(answer)
                if int(match.group(1)) < 1
                or int(match.group(1)) > evidence_count
            }
        )
        if invalid_labels:
            raise ValueError(
                "Agent 5 Qwen viện dẫn evidence không tồn tại: "
                + ", ".join(map(str, invalid_labels))
            )
        return answer
