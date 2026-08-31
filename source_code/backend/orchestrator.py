"""
Planner Orchestrator - Agent điều phối đa bước (Multi-Agent State Machine)
Nhiệm vụ: Nhận câu hỏi phức tạp, vạch kế hoạch, điều phối Agent 3+4 (Searcher+Evaluator) thực thi tuần tự
Cuối cùng gọi Agent 5 (Final Generator) để tổng hợp evidences thành câu trả lời
"""
import json
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional
import os
from .tools.calculation import execute_calculation_plan
from .retriever.temporal import (
    describe_temporal_scope,
    expand_comparison_task,
    normalize_temporal_scope,
    temporal_scope_key,
)

from .config import configure_agent_runtime


configure_agent_runtime()

SAFE_FACT_DESCRIPTIONS = {
    "applicability": "đối tượng, phạm vi và điều kiện áp dụng của quy định",
    "tax_base": "căn cứ tính thuế áp dụng cho tình huống",
    "tax_rate": "thuế suất áp dụng cho tình huống",
    "threshold_or_exemption": "ngưỡng và điều kiện miễn hoặc không phải nộp thuế",
    "calculation_method": "phương pháp tính nghĩa vụ thuế",
    "obligation": "nghĩa vụ pháp lý áp dụng cho đối tượng",
    "procedure": "hồ sơ và thủ tục phải thực hiện",
    "deadline": "thời hạn pháp lý áp dụng",
    "authority": "cơ quan có thẩm quyền hoặc nơi thực hiện thủ tục",
}


class MultiAgentState:
    """Class quản lý trạng thái và trí nhớ của phiên làm việc"""
    
    def __init__(self, query: str):
        self.original_query = query
        self.plan: List[Dict] = []
        self.gathered_evidences: List = []  # Lưu tất cả evidences từ các bước
        self.failed_steps: List[int] = []
        self.step_results: List[Dict] = []  # Lưu chi tiết từng bước
        self.quality_issues: List[str] = []  # Lưu các vấn đề chất lượng
        self.incomplete_searches: List[str] = []
        self.search_memories: Dict[str, Any] = {}
        self.candidate_task_sets: List[Dict] = []


class PlannerOrchestrator:
    """
    Orchestrator điều phối tuần tự theo mô hình State Machine
    
    Workflow:
    1. Nhận câu hỏi người dùng
    2. Gọi Planner LLM để vạch kế hoạch (chia thành N bước)
    3. Thực thi tuần tự từng bước bằng Agent 3+4 (Searcher + Evaluator with retry)
    4. Thu thập evidences từ tất cả các bước
    5. Gọi Agent 5 (Final Generator) để tổng hợp thành câu trả lời
    """
    
    def __init__(
        self,
        planner_only: bool = False,
        *,
        planner_model: str | None = None,
    ):
        """Initialize Agent 2 only, or the complete Agent 2-5 workflow."""
        self.max_steps = 3
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY không tồn tại trong .env")

        from .agents.agent2 import GeminiTwoStagePlannerLLM

        self.planner_llm = GeminiTwoStagePlannerLLM(
            api_key=api_key,
            model=planner_model
            or os.getenv("AGENT2_GEMINI_MODEL")
            or "models/gemini-3.1-flash-lite",
            max_output_tokens=int(
                os.getenv("AGENT2_GEMINI_MAX_OUTPUT", "2048")
            ),
            fact_max_output_tokens=int(
                os.getenv("AGENT2_GEMINI_2B_MAX_OUTPUT", "4096")
            ),
        )
        self.last_planner_raw_output = ""

        self.searcher = None
        self.evaluator = None
        self.generator_llm = None
        self.search_memory_class = None
        self.search_with_retry = None
        if planner_only:
            print("[Orchestrator] Chế độ planner-only: chỉ khởi tạo Agent 2 (Gemini)\n")
            return

        from .agents.agent3 import (
            GeminiSequentialSearcher,
            GeminiToolClient,
            StructuredSearchMemory,
        )
        from .agents.agent4 import (
            GeminiSequentialFactEvaluator,
            search_once_with_sequential_fact_evaluation,
        )
        from .agents.agent5 import GeminiFinalGenerator

        client = GeminiToolClient()
        self.search_memory_class = StructuredSearchMemory
        self.search_with_retry = search_once_with_sequential_fact_evaluation
        print("[Orchestrator] Đang khởi tạo Agent 3 (Gemini sequential)...")
        self.searcher = GeminiSequentialSearcher()
        print("[Orchestrator] Đang khởi tạo Agent 4 (Gemini sequential facts)...")
        self.evaluator = GeminiSequentialFactEvaluator(client)
        print("[Orchestrator] Đang khởi tạo Agent 5 (Gemini Final Generator)...")
        self.generator_llm = GeminiFinalGenerator(
            api_key=api_key,
            model=os.getenv(
                "AGENT5_GEMINI_MODEL", "models/gemini-3.1-flash-lite"
            ),
        )
        print("[Orchestrator] ✓ Tất cả Gemini agents đã sẵn sàng\n")
    
    def _fallback_plan(self, original_query: str) -> List[Dict]:
        task_id = "T1"
        fact_requirements = [
            {
                "fact_id": "T1_F1",
                "task_id": task_id,
                "type": "applicability",
                "description": SAFE_FACT_DESCRIPTIONS["applicability"],
            }
        ]
        return [
            {
                "task_id": task_id,
                "task": original_query,
                "query": original_query,
                "original_query": original_query,
                "fact_requirements": fact_requirements,
                "required_facts": [
                    requirement["description"] for requirement in fact_requirements
                ],
                "temporal": normalize_temporal_scope({"type": "current"}),
            }
        ]

    def _clean_and_parse_json(self, json_str: str, original_query: str) -> List[Dict]:
        """
        Làm sạch markdown và parse JSON ra mảng Python
        
        Args:
            json_str: Chuỗi JSON từ LLM (có thể có markdown wrapper)
            original_query: Câu hỏi gốc (fallback nếu parse lỗi)
        
        Returns:
            List các bước tra cứu
        """
        # Loại bỏ markdown code blocks
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            json_str,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        try:
            plan = json.loads(cleaned)
            
            # Validate
            if isinstance(plan, list) and len(plan) > 0:
                normalized_plan: List[Dict] = []
                for index, raw_step in enumerate(plan, 1):
                    if isinstance(raw_step, str):
                        raw_step = {
                            "task": raw_step,
                            "query": raw_step,
                            "temporal": {"type": "current"},
                        }
                    if not isinstance(raw_step, dict):
                        raise ValueError(f"Planner step {index} phải là object")

                    task = str(raw_step.get("task") or "").strip()
                    query = str(raw_step.get("query") or task).strip()
                    if not task or not query:
                        raise ValueError(f"Planner step {index} thiếu task/query")
                    task_id = str(
                        raw_step.get("task_id") or f"T{index}"
                    ).strip()
                    raw_requirements = (
                        raw_step.get("fact_requirements")
                        or raw_step.get("required_facts")
                        or []
                    )
                    fact_requirements = []
                    for fact_index, requirement in enumerate(
                        raw_requirements,
                        1,
                    ):
                        if isinstance(requirement, dict):
                            description = str(
                                requirement.get("description") or ""
                            ).strip()
                            fact_type = str(
                                requirement.get("type") or "unspecified"
                            ).strip()
                            fact_id = str(
                                requirement.get("fact_id")
                                or f"{task_id}_F{fact_index}"
                            ).strip()
                        else:
                            description = str(requirement).strip()
                            fact_type = "unspecified"
                            fact_id = f"{task_id}_F{fact_index}"
                        if not description:
                            continue
                        fact_requirements.append(
                            {
                                "fact_id": fact_id,
                                "task_id": task_id,
                                "type": fact_type,
                                "description": description,
                            }
                        )
                    required_facts = [
                        requirement["description"]
                        for requirement in fact_requirements
                    ]
                    normalized_plan.extend(
                        expand_comparison_task(
                            {
                                "task_id": task_id,
                                "task": task,
                                "query": query,
                                "original_query": original_query,
                                "fact_requirements": fact_requirements,
                                "required_facts": required_facts,
                                "temporal": raw_step.get("temporal")
                                or {"type": "current"},
                            }
                        )
                    )
                return normalized_plan
            
            raise ValueError("Planner output must be a non-empty JSON list")
            
        except json.JSONDecodeError as exc:
            raise ValueError("Planner output is not valid JSON") from exc
    
    def generate_plan(
        self,
        user_query: str,
        allow_fallback: bool = True,
    ) -> List[Dict]:
        """
        Gọi LLM để vạch ra các bước tra cứu tuần tự
        
        Args:
            user_query: Câu hỏi của người dùng
        
        Returns:
            List các bước tra cứu (VD: ["Bước 1: ...", "Bước 2: ..."])
        """
        system_prompt = """Bạn là Kiến trúc sư Lập kế hoạch tra cứu Luật thuế Việt Nam.

Nhiệm vụ: Nhận câu hỏi pháp lý và chia nó thành các bước tra cứu TUẦN TỰ, LOGIC.

=====================================
QUY TẮC LẬP KẾ HOẠCH
=====================================

1. Mỗi bước phải RÕ RÀNG, CỤ THỂ.

2. CHIẾN LƯỢC TỐI ƯU SỐ BƯỚC:
   - BỎ QUA HOÀN TOÀN bước "Tìm định nghĩa" đối với các đối tượng phổ thông (lương, trúng số, bán nhà). Hãy ĐI THẲNG vào tìm Thuế suất và Cách tính.
   - CHỈ tìm định nghĩa khi thuật ngữ pháp lý mang tính trừu tượng hoặc mới mẻ (VD: Tài sản số).

3. KHÔNG tạo quá 3 bước.

3a. KHÓA PHẠM VI LOẠI THUẾ (TAX-SCOPE LOCK):
   - Nếu người dùng nêu RÕ một loại thuế cụ thể như "thuế thu nhập cá nhân",
     "thuế giá trị gia tăng" hoặc "thuế thu nhập doanh nghiệp", CHỈ lập kế hoạch
     cho đúng loại thuế được nêu. TUYỆT ĐỐI không tự thêm loại thuế khác.
   - Nếu người dùng nêu RÕ nhiều loại thuế, chỉ lập kế hoạch cho các loại thuế đó.
   - Chỉ khi người dùng hỏi chung bằng các cụm như "phải đóng thuế gì", "các loại thuế phải nộp", "nghĩa vụ thuế" hoặc chỉ nói "thuế" mà không chỉ rõ loại thuế, mới được xác định và lập kế hoạch cho các loại thuế có khả năng liên quan.
   - Ví dụ: câu hỏi "số thuế thu nhập cá nhân phải nộp là bao nhiêu" chỉ được
     tìm thuế thu nhập cá nhân, không được thêm thuế giá trị gia tăng.

4. QUY TẮC BẢO TOÀN DỮ LIỆU (DATA RETENTION - RẤT QUAN TRỌNG):
   BẮT BUỘC phải BÊ NGUYÊN TOÀN BỘ các con số, thời gian, giá trị tiền (VD: 150 triệu, 5 năm, 50 tỷ, 183 ngày) có trong câu hỏi đầu vào để đưa thẳng vào nội dung của các Bước. Tuyệt đối không được khái quát hóa làm mất số liệu của người dùng.
   - Đồng thời phải giữ các điều kiện pháp lý làm thay đổi kết quả: cá nhân cư trú/không cư trú, loại hoạt động hoặc loại thu nhập, vai trò của người dùng, loại tài sản/hàng hóa, địa điểm và thời kỳ phát sinh.
   - Mỗi bước liên quan phải tự chứa đủ các điều kiện này; không được cho rằng bước sau sẽ tự nhớ từ câu hỏi ban đầu.
   - Không được đổi loại thu nhập. Ví dụ "kinh doanh thương mại điện tử" không được rút gọn thành "thu nhập khác" hoặc "chuyển nhượng tài sản số".

5. ĐỐI VỚI CÂU HỎI TRỰC DIỆN (đã rõ sự kiện và chỉ hỏi một kết luận):
   CHỈ TẠO ĐÚNG 1 BƯỚC DUY NHẤT.

6. ĐỐI VỚI CÂU HỎI VỀ MIỄN THUẾ:
   - Nếu người dùng chỉ hỏi có được miễn thuế hay không: tạo đúng 1 bước tìm
     điều kiện và phạm vi miễn thuế.
   - Chỉ tạo thêm bước tìm thuế suất/cách tính khi người dùng hỏi rõ số thuế
     phải nộp nếu không được miễn hoặc yêu cầu xử lý cả hai kịch bản.

7. NHIỀU SỰ KIỆN ĐỘC LẬP (VD: Vừa bán xe, vừa trúng số):
   BẮT BUỘC tách thành các bước riêng theo từng sự kiện. KHÔNG gộp chung 2 loại thu nhập khác bản chất vào 1 bước.

=====================================
TỰ KIỂM TRA 1: TASK PLAN
=====================================

Soạn danh sách task nháp trước, sau đó kiểm tra CHỈ cấu trúc task:
□ Liệt kê từng KẾT QUẢ người dùng yêu cầu. Mỗi kết quả đã được ít nhất một task
  bao phủ chưa? Ví dụ "xử lý số thuế + thông báo doanh thu + hóa đơn điện tử"
  phải được bao phủ đủ cả ba.
□ Nếu có nhiều sự kiện/giao dịch độc lập, mỗi sự kiện đã có task riêng chưa?
□ Có task nào gộp hai loại thuế, hai đối tượng hoặc hai giao dịch khác bản chất
  khiến semantic query mất trọng tâm không? Nếu có thì tách.
□ Có hai task nào tìm gần như cùng một quy định và cùng dẫn tới một kết luận
  không? Nếu có thì gộp.
□ Có task nào trả lời kịch bản người dùng không hỏi, như tự tính thuế khi người
  dùng chỉ hỏi điều kiện miễn thuế không? Nếu có thì xóa.
□ Mỗi task đã giữ đúng tình trạng cư trú, loại thu nhập/hoạt động, vai trò, địa
  điểm và toàn bộ số liệu tình huống liên quan chưa?
□ Người dùng nêu rõ loại thuế nào thì task có tự thêm loại thuế khác không?
  Nếu có thì xóa phần tự thêm.
□ temporal có phản ánh đúng current, ngày, tháng, năm, khoảng hoặc phép so sánh
  mà người dùng nêu không?
□ Tổng số task có vượt quá 3 không? Nếu có, chỉ gộp các task cùng chủ đề.

Chỉ chuyển sang checklist 2 sau khi TASK PLAN đã đạt tất cả mục trên.

=====================================
TỰ KIỂM TRA 2: TYPED FACT REQUIREMENTS
=====================================

Với TỪNG task đã khóa, kiểm tra riêng danh sách fact_requirements:
□ Mỗi kết quả con mà task cần trả lời đã có ít nhất một fact tương ứng chưa?
□ Mỗi fact có đúng một type trong danh sách cho phép và một description nguyên
  tử, ngắn gọn, có thể được một provision pháp luật chứng minh trực tiếp không?
□ Có fact nào chỉ là mệnh lệnh mơ hồ như "tìm văn bản liên quan", "đọc luật"
  hoặc lặp lại nguyên task không? Nếu có thì viết cụ thể lại.
□ Có fact nào chứa field value, expected_value, answer, result hoặc amount không?
  Nếu có thì XÓA FIELD ĐÓ. Agent 2 chỉ mô tả loại evidence cần tìm.
□ Description có chứa đáp án cuối, kết quả phép tính hoặc kết luận thay vì tiền
  đề pháp lý cần truy xuất không? Nếu có thì viết lại trung tính.
□ Có mức tiền, ngưỡng, thuế suất, thời hạn, giá vốn, chi phí hoặc kết luận pháp
  lý nào do bạn tự nhớ nhưng câu hỏi không cung cấp không? Nếu có, đổi thành mô
  tả trung tính cần tra cứu.
□ Có đổi sai tên khái niệm không? Ví dụ mức doanh thu được trừ của hoạt động
  kinh doanh không được gọi là giảm trừ gia cảnh.
□ Với comparison, facts có chỉ mô tả nội dung chung cần tìm tại MỖI mốc không?
  Không yêu cầu một task con phải chứng minh sự khác nhau giữa cả hai mốc.
□ Danh sách có từ 1 đến 6 facts và không trùng lặp không?

Chỉ xuất JSON List sau khi CẢ checklist TASK PLAN và TYPED FACT REQUIREMENTS đều đạt.

=====================================
VÍ DỤ CHIA NHIỆM VỤ
=====================================

Các ví dụ dưới đây chỉ mô tả DANH SÁCH NHIỆM VỤ về mặt ngữ nghĩa.
Định dạng JSON object bắt buộc được quy định ở phần 8 bên dưới.

VÍ DỤ 1 (thuật ngữ trừu tượng cần định nghĩa):
Input: "Thu nhập từ chuyển nhượng Bitcoin có bị đánh thuế không?"
Nhiệm vụ: tìm quy định về tính hợp pháp và đối tượng chịu thuế đối với tài sản số; tìm thuế suất áp dụng đối với thu nhập từ chuyển nhượng tài sản số.

VÍ DỤ 2 (thuật ngữ phổ thông, 1 sự kiện → 1 bước):
Input: "Thuế suất thu nhập cá nhân từ trúng thưởng Vietlott 50 tỷ là bao nhiêu?"
Nhiệm vụ: tìm quy định thuế suất và cách tính thuế thu nhập cá nhân đối với thu nhập từ trúng thưởng xổ số.

VÍ DỤ 3 (tra cứu trực tiếp điều khoản):
Input: "Điều 7 quy định gì?"
Nhiệm vụ: tìm nội dung đầy đủ của Điều 7.

VÍ DỤ 4 (⚠️ NHIỀU SỰ KIỆN ĐỘC LẬP - phải tách dù mỗi sự kiện đã trực diện):
Input: "Vừa bán xe ô tô cũ lỗ 50 triệu, vừa trúng giải ba xổ số 5 triệu, tôi phải đóng thuế khoản nào?"
Nhiệm vụ: tìm quy định thuế thu nhập cá nhân đối với thu nhập từ bán xe ô tô; tìm riêng quy định thuế thu nhập cá nhân đối với thu nhập từ trúng thưởng xổ số.

VÍ DỤ 5 (1 sự kiện, nhiều biến số → KHÔNG tách theo biến số):
Input: "Lương 25 triệu/tháng, có 2 người phụ thuộc thì đóng thuế bao nhiêu?"
Nhiệm vụ: tìm quy định cách tính thuế thu nhập cá nhân từ tiền lương, tiền công và mức giảm trừ gia cảnh cho người phụ thuộc.

VÍ DỤ 6 (2 giai đoạn của cùng 1 tài sản, cần tách vì căn cứ pháp lý khác nhau):
Input: "Công ty thưởng Tết bằng cổ phiếu, sau đó vài tháng tôi bán đi thì tính thuế ra sao?"
Nhiệm vụ: tìm quy định thuế thu nhập cá nhân khi nhận cổ phiếu thưởng; tìm riêng quy định thuế thu nhập cá nhân khi bán cổ phiếu thưởng.

VÍ DỤ 7 (bảo toàn tình trạng cư trú và loại hoạt động):
Input: "Tôi là cá nhân cư trú bán hàng online, doanh thu 200 triệu đồng/năm, có phải nộp thuế không và tính thế nào?"
Nhiệm vụ: tìm mức doanh thu không phải nộp thuế thu nhập cá nhân cho trường hợp đã nêu; tìm căn cứ tính thuế và thuế suất nếu doanh thu vượt mức.
"""

        system_prompt += f"""

=====================================
8. KHÓA THỜI GIAN PHÁP LÝ
=====================================

Ngày hiện tại của hệ thống: {date.today().isoformat()}.

Mỗi bước BẮT BUỘC là một JSON object có task_id, task, query,
fact_requirements và temporal. Các ví dụ phía
trên chỉ minh họa cách chia nhiệm vụ; KHÔNG dùng output chuỗi kiểu cũ.

fact_requirements là danh sách 1-6 object mô tả thông tin pháp lý tối thiểu phải
có căn cứ để trả lời bước đó. Mỗi object có đúng bốn field:
- fact_id: ID duy nhất theo dạng T1_F1, T1_F2, T2_F1;
- task_id: phải trùng task_id của task chứa fact;
- type: một nhãn trong danh sách cho phép;
- description: mô tả trung tính về evidence cần truy xuất.

Danh sách type cho phép:
- applicability: đối tượng, phạm vi, điều kiện áp dụng hoặc phân loại pháp lý;
- tax_base: căn cứ tính thuế hoặc thu nhập chịu thuế;
- tax_rate: thuế suất hoặc tỷ lệ áp dụng;
- threshold_or_exemption: ngưỡng, miễn thuế hoặc trường hợp không phải nộp;
- calculation_method: công thức hoặc phương pháp tính;
- obligation: nghĩa vụ kê khai, nộp, khấu trừ, hóa đơn hoặc trách nhiệm khác;
- procedure: hồ sơ, biểu mẫu và trình tự thủ tục;
- deadline: thời hạn thực hiện;
- authority: cơ quan có thẩm quyền hoặc nơi nộp.

Không tạo type ngoài danh sách. Không thêm value, expected_value, answer, result,
amount, confidence hoặc provenance. Giá trị và provenance chỉ được bổ sung sau
khi Retriever cung cấp evidence.

Không tạo fact riêng để hỏi "quy định nào có hiệu lực tại ngày/tháng/năm được
hỏi". Trường temporal và Python filter đã khóa thời gian. Chỉ tạo fact về hiệu
lực hoặc ngày áp dụng khi chính người dùng hỏi trực tiếp về hiệu lực; trường hợp
đó dùng type=applicability.

KHÔNG ĐƯỢC dùng kiến thức nhớ sẵn để tự khẳng định trong task, query hoặc
fact_requirements rằng một ngưỡng, thuế suất, thời hạn hay phương pháp pháp lý là
bao nhiêu. Chỉ giữ con số nếu chính người dùng đã nêu nó như dữ kiện tình huống.
Nếu giá trị cần được pháp luật xác định, hãy viết trung tính:
- ĐÚNG: "ngưỡng doanh thu không phải nộp thuế áp dụng tại thời điểm được hỏi";
- SAI: "ngưỡng doanh thu 100 triệu đồng/năm" nếu người dùng không nêu 100 triệu;
- ĐÚNG: "căn cứ xác định thu nhập chịu thuế từ chuyển nhượng bất động sản";
- SAI: tự thêm "giá vốn và chi phí được trừ" khi câu hỏi không nêu căn cứ đó.

Đặc biệt với câu hỏi lịch sử hoặc câu hỏi chỉ đích danh Điều/Khoản:
- Không liệt kê trước các thành phần có thể từng tồn tại trong phiên bản khác.
- Nếu câu hỏi chỉ hỏi "được xác định theo căn cứ nào", fact phải là
  "căn cứ/phương pháp xác định ... áp dụng tại thời điểm được hỏi".
- Mọi danh từ pháp lý cụ thể trong fact_requirements như "giá vốn", "chi phí được
  trừ", "bảng giá đất", "thuế suất 2%" phải xuất hiện trong câu hỏi đầu vào;
  nếu không xuất hiện thì để Retriever tìm ra, không được Agent 2 tự thêm.

Không đổi tên khái niệm pháp lý. "Mức doanh thu được trừ khi tính thuế kinh
doanh" không phải là "giảm trừ gia cảnh". Khi chưa chắc tên pháp lý, dùng mô tả
trung tính bám sát nguyên văn câu hỏi.

Mỗi vế đầu ra mà người dùng yêu cầu phải xuất hiện trong fact_requirements. Ví dụ,
nếu hỏi đồng thời cách xử lý thuế đã nộp, thời hạn thông báo doanh thu và nghĩa
vụ hóa đơn điện tử thì fact_requirements phải bao phủ đủ cả ba nội dung.
Ví dụ:
- hỏi mức thuế phải nộp: đối tượng/loại thu nhập, căn cứ tính thuế, thuế suất,
  ngưỡng hoặc điều kiện miễn nếu có;
- hỏi thủ tục: đối tượng áp dụng, hồ sơ, nơi nộp, thời hạn;
- hỏi so sánh: cùng một fact ở từng mốc thời gian.
Không đưa "tìm văn bản liên quan" hoặc "đọc luật" thành fact_requirements.

temporal.type chỉ được là:
- current: người dùng không nêu thời gian hoặc hỏi hiện nay/hiện hành;
- exact_date: hỏi đúng một ngày hợp lệ, dùng date=YYYY-MM-DD;
- month: hỏi trong một tháng, dùng month=1..12 và year=YYYY;
- year: hỏi trong một năm hoặc kỳ tính thuế năm, dùng year=YYYY;
- date_range: hỏi một khoảng liên tục. Giữ đúng độ chi tiết của từng biên bằng
  from/to object {{"granularity":"year|month|day", "value":"..."}}; riêng to
  được phép dùng {{"granularity":"current", "value":"current"}}. Không dùng
  current cho from và không tự tính ngày đầu/cuối kỳ; Python sẽ mở rộng biên;
- comparison: chỉ dùng khi người dùng nêu rõ từ hai mốc/tháng/năm cần so sánh;
  dùng comparison_scopes. Python sẽ tách các scope thành những lượt retrieval
  độc lập.

Với comparison, task, query và fact_requirements phải mô tả NỘI DUNG PHÁP LÝ CHUNG
cần lấy tại mỗi mốc. Không ghi "nội dung tại mốc A" và "nội dung tại mốc B"
thành hai fact_requirements trong cùng object, vì Python sẽ sao object đó cho từng
mốc. Nhãn và temporal scope đã phân biệt các mốc.

Nếu người dùng hỏi "trong khoảng từ A đến B thay đổi thế nào", mặc định tạo một
task có type=date_range, from=A, to=B để lấy mọi version giao với cả khoảng.
Không tự đoán ngày thay đổi ở giữa khoảng và không đổi thành hai exact_date chỉ
lấy hai đầu mút. Chỉ tách thành nhiều task khi câu hỏi có nhiều nội dung pháp lý
độc lập; các task đó vẫn dùng cùng date_range A..B.

Không đổi một ngày cụ thể thành cả tháng. Không tạo ngày không tồn tại. query
chỉ chứa nội dung semantic cần tìm; không chèn valid_from, valid_to, is_latest
hay cú pháp Qdrant vì Python sẽ khóa filter.

OUTPUT BẮT BUỘC:
[
  {{
    "task_id": "T1",
    "task": "Mô tả nhiệm vụ đầy đủ và giữ nguyên dữ kiện người dùng",
    "query": "Câu semantic query cho Agent 3",
    "fact_requirements": [
      {{
        "fact_id": "T1_F1",
        "task_id": "T1",
        "type": "tax_base",
        "description": "căn cứ tính thuế áp dụng cho loại thu nhập được hỏi"
      }},
      {{
        "fact_id": "T1_F2",
        "task_id": "T1",
        "type": "tax_rate",
        "description": "thuế suất áp dụng cho loại thu nhập được hỏi"
      }}
    ],
    "temporal": {{
      "type": "current",
      "date": null,
      "month": null,
      "year": null,
      "from": null,
      "to": null,
      "comparison_scopes": []
    }}
  }}
]

Ví dụ hỏi lịch sử:
Input: "Trong năm 2010 mức giảm trừ gia cảnh là bao nhiêu?"
Output: [{{"task_id":"T1","task":"Tìm mức giảm trừ gia cảnh áp dụng trong năm 2010",
"query":"mức giảm trừ gia cảnh cho người nộp thuế và người phụ thuộc",
"fact_requirements":[
{{"fact_id":"T1_F1","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho người nộp thuế"}},
{{"fact_id":"T1_F2","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho mỗi người phụ thuộc"}}],
"temporal":{{"type":"year","year":2010}}}}]

Ví dụ so sánh:
Input: "So sánh mức giảm trừ gia cảnh trong năm 2010 và năm 2014."
Output: [{{"task_id":"T1","task":"Tìm mức giảm trừ gia cảnh tại từng năm để so sánh",
"query":"mức giảm trừ gia cảnh cho người nộp thuế và người phụ thuộc",
"fact_requirements":[
{{"fact_id":"T1_F1","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho người nộp thuế"}},
{{"fact_id":"T1_F2","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho mỗi người phụ thuộc"}}],
"temporal":{{"type":"comparison","comparison_scopes":[
{{"type":"year","year":2010,"label":"Năm 2010"}},
{{"type":"year","year":2014,"label":"Năm 2014"}}
]}}}}]

Ví dụ khoảng thời gian:
Input: "Trong khoảng từ 15/06/2013 đến 15/07/2013, mức giảm trừ thay đổi thế nào?"
Output: [{{"task_id":"T1","task":"Tìm mọi mức giảm trừ gia cảnh có hiệu lực trong khoảng được hỏi",
"query":"mức giảm trừ gia cảnh cho người nộp thuế và người phụ thuộc",
"fact_requirements":[
{{"fact_id":"T1_F1","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho người nộp thuế"}},
{{"fact_id":"T1_F2","task_id":"T1","type":"threshold_or_exemption","description":"mức giảm trừ cho mỗi người phụ thuộc"}}],
"temporal":{{"type":"date_range","from":{{"granularity":"day","value":"2013-06-15"}},"to":{{"granularity":"day","value":"2013-07-15"}}}}}}]

Chỉ trả JSON List object, không markdown và không giải thích.
"""
        
        prompt = f"{system_prompt}\n\nCâu hỏi: {user_query}\n\nOutput:"
        
        try:
            self.last_planner_raw_output = ""
            # Gọi Planner LLM
            response = self.planner_llm.complete_query(user_query)
            plan_response = str(response)
            self.last_planner_raw_output = plan_response
            
            print(f"\n📋 [PLANNER] Raw output:")
            print(f"   {plan_response[:300]}...")
            
            # Parse JSON
            return self._clean_and_parse_json(plan_response, user_query)
            
        except ValueError as e:
            raise RuntimeError(
                "Agent 2 returned an invalid plan; retrieval was not run: "
                f"{e}"
            ) from e
        except Exception as e:
            print(f"❌ [PLANNER] Lỗi khi gọi LLM: {e}")
            if not allow_fallback:
                raise RuntimeError(
                    "Agent 2 call failed; planner-only evaluation does not use fallback."
                ) from e
            print(f"   Fallback: Chuyển về 1 bước đơn giản")
            return self._fallback_plan(user_query)
    
    def _is_failed_response(self, response: str) -> bool:
        """
        Kiểm tra xem response có phải là lỗi không
        
        Args:
            response: Câu trả lời từ Search Agent
        
        Returns:
            True nếu bị lỗi (không tìm thấy, chưa có quy định)
        """
        response_lower = response.lower()
        
        # Các từ khóa báo hiệu lỗi
        error_keywords = [
            "không tìm thấy",
            "chưa có quy định",
            "không có quy định",
            "không có thông tin",
            "không tìm được",
            "chưa được quy định"
        ]
        
        return any(keyword in response_lower for keyword in error_keywords)
    
    def execute_workflow(
        self,
        user_query: str,
        verbose: bool = True,
        precomputed_plan: Optional[List[Dict]] = None,
        generate_final_answer: bool = True,
    ) -> Dict:
        """
        Hàm thực thi chính: Chạy tuần tự và gom evidences
        
        Args:
            user_query: Câu hỏi người dùng
            verbose: Có in log chi tiết không
        
        Returns:
            Dict chứa:
                - final_answer: Câu trả lời cuối cùng từ Agent 5
                - all_evidences: Tất cả evidences đã thu thập
                - plan: Kế hoạch đã thực thi
                - step_results: Chi tiết từng bước
                - failed_steps: Danh sách bước thất bại
        """
        # Khởi tạo state
        state = MultiAgentState(user_query)
        
        if verbose:
            print("\n" + "="*80)
            print("🧠 [PLANNER ORCHESTRATOR] BẮT ĐẦU WORKFLOW")
            print("="*80)
        
        # BƯỚC 1: Lập kế hoạch
        if verbose:
            print("\n📋 BƯỚC 1: LÊN KẾ HOẠCH...")
        
        if precomputed_plan is None:
            state.plan = self.generate_plan(user_query)
        else:
            state.plan = self._clean_and_parse_json(
                json.dumps(precomputed_plan, ensure_ascii=False),
                user_query,
            )
            self.last_planner_raw_output = json.dumps(
                state.plan,
                ensure_ascii=False,
                indent=2,
            )
            if verbose:
                print(
                    "   Sử dụng Frozen Plan đã lưu; bỏ qua lời gọi Agent 2."
                )
        
        # Cắt giảm nếu quá dài
        if len(state.plan) > self.max_steps:
            if verbose:
                print(f"⚠️  Kế hoạch quá dài ({len(state.plan)} bước), cắt xuống còn {self.max_steps} bước.")
            state.plan = state.plan[:self.max_steps]
        
        if verbose:
            print(f"\n✅ Kế hoạch chốt ({len(state.plan)} bước):")
            for i, step in enumerate(state.plan, 1):
                print(
                    f"   {i}. [{step.get('task_id', f'T{i}')}] {step['task']} "
                    f"[{describe_temporal_scope(step['temporal'])}]"
                )
        
        # BƯỚC 2: Thực thi tuần tự từng bước bằng Agent 3+4
        if verbose:
            print(f"\n{'='*80}")
            print("🚀 BƯỚC 2: THỰC THI TUẦN TỰ (AGENT 3 + AGENT 4)")
            print("="*80)
        
        for i, plan_step in enumerate(state.plan):
            step_task = str(plan_step["task"])
            semantic_query = str(plan_step.get("query") or step_task)
            fact_requirements = list(plan_step.get("fact_requirements") or [])
            required_facts = [
                str(fact)
                for fact in (plan_step.get("required_facts") or [step_task])
            ]
            temporal_scope = normalize_temporal_scope(plan_step["temporal"])
            scope_description = describe_temporal_scope(temporal_scope)
            scope_key = temporal_scope_key(temporal_scope)
            task_id = str(plan_step.get("task_id") or f"T{i + 1}")
            memory_key = f"{task_id}|{scope_key}"
            search_memory = state.search_memories.setdefault(
                memory_key,
                self.search_memory_class(),
            )
            if verbose:
                print(f"\n{'─'*80}")
                print(f"🔹 ĐANG THỰC THI BƯỚC {i+1}/{len(state.plan)}")
                print(
                    f"   Task: [{plan_step.get('task_id', f'T{i + 1}')}] "
                    f"{step_task}"
                )
                print(f"   Query: {semantic_query}")
                print(f"   Thời gian: {scope_description}")
                print("   Facts:")
                for fact in fact_requirements:
                    print(
                        f"     - [{fact.get('fact_id', '?')}] "
                        f"{fact.get('type', 'unspecified')}: "
                        f"{fact.get('description', '')}"
                    )
                print(f"{'─'*80}")
            
            try:
                # Gọi Agent 3+4 (search_with_retry)
                if verbose:
                    print(f"\n🔍 Đang gọi Agent 3 (Searcher) + Agent 4 (Evaluator)...")
                
                if plan_step.get("temporal_label"):
                    execution_query = (
                        f"Task: {step_task}\n"
                        f"Semantic query: {semantic_query}\n"
                        f"Temporal branch: {scope_description}\n"
                        "Constraint: chỉ tra cứu nhánh thời gian này; Python "
                        "xử lý các nhánh so sánh khác."
                    )
                else:
                    execution_query = (
                        f"Task: {step_task}\n"
                        f"Semantic query: {semantic_query}\n"
                        f"Original context: {user_query}\n"
                        f"Temporal scope: {scope_description}"
                    )

                search_kwargs = dict(
                    searcher=self.searcher,
                    evaluator=self.evaluator,
                    query=execution_query,
                    max_steps=5,  # Cho phép Agent 3+4 retry tối đa 5 lần
                    memory=search_memory,
                    temporal_scope=temporal_scope,
                    fact_requirements=fact_requirements,
                    required_facts=required_facts,
                )
                search_kwargs["verbose"] = verbose
                search_result = self.search_with_retry(**search_kwargs)
                candidate_items = []
                seen_candidate_keys = set()
                for candidate in search_memory.accumulated_chunks.values():
                    candidate_data = (
                        candidate.model_dump()
                        if hasattr(candidate, "model_dump")
                        else candidate.dict()
                    )
                    candidate_key = (
                        candidate_data.get("provision_key")
                        or candidate_data.get("chunk_id"),
                        candidate_data.get("version") or 1,
                    )
                    if candidate_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(candidate_key)
                    candidate_items.append(candidate_data)
                state.candidate_task_sets.append(
                    {
                        "task_id": memory_key,
                        "fact_requirements": [
                            {
                                **fact,
                                "fact_id": (
                                    f"{fact.get('fact_id')}@{scope_key}"
                                ),
                                "task_id": memory_key,
                                "temporal": temporal_scope,
                            }
                            for fact in fact_requirements
                        ],
                        "candidates": candidate_items,
                    }
                )
                
                # Lưu kết quả
                is_failed = (search_result['final_status'] == 'not_found')
                
                if is_failed:
                    state.failed_steps.append(i)
                    
                    if verbose:
                        print(f"❌ Bước {i+1} không tìm thấy dữ kiện pháp lý")
                else:
                    # Deduplicate within a temporal bucket, not across comparison dates.
                    for ev in search_result['all_evidences']:
                        chunk_id = getattr(ev, "chunk_id", None)
                        evidence_scope = (
                            getattr(ev, "temporal_scope", None) or temporal_scope
                        )
                        evidence_scope_key = temporal_scope_key(evidence_scope)
                        if chunk_id:
                            is_duplicate = any(
                                getattr(existing, "chunk_id", None) == chunk_id
                                and temporal_scope_key(
                                    getattr(existing, "temporal_scope", None)
                                    or temporal_scope
                                ) == evidence_scope_key
                                for existing in state.gathered_evidences
                            )
                        else:
                            # Compatibility with the legacy searcher, whose
                            # ChunkEvidence does not carry a Qdrant chunk ID.
                            is_duplicate = any(
                                getattr(existing, "chunk_id", None) is None
                                and existing.van_ban == ev.van_ban
                                and existing.dieu_khoan == ev.dieu_khoan
                                and existing.noi_dung_goc == ev.noi_dung_goc
                                and temporal_scope_key(
                                    getattr(existing, "temporal_scope", None)
                                    or temporal_scope
                                ) == evidence_scope_key
                                for existing in state.gathered_evidences
                            )
                        if not is_duplicate:
                            state.gathered_evidences.append(ev)
                    
                    if verbose:
                        print(f"✅ Hoàn thành Bước {i+1}")
                        print(f"   Evidences: {len(search_result['all_evidences'])}")
                        print(f"   Total evidences tích lũy: {len(state.gathered_evidences)}")
                
                # Tích lũy quality issues
                if 'quality_issues' in search_result:
                    state.quality_issues.extend(search_result['quality_issues'])
                if (
                    search_result.get('final_status') == 'partial'
                    and search_result.get('missing_info')
                ):
                    state.incomplete_searches.append(
                        f"Bước {i + 1} ({step_task}): "
                        f"{search_result['missing_info']}"
                    )
                
                # Lưu chi tiết
                state.step_results.append({
                        "step_number": i + 1,
                        "task_id": task_id,
                        "task": step_task,
                        "query": semantic_query,
                        "fact_requirements": fact_requirements,
                        "required_facts": required_facts,
                        "temporal": temporal_scope,
                    "final_status": search_result['final_status'],
                    "evidences_count": len(search_result['all_evidences']),
                    "total_searches": search_result['total_searches'],
                    "search_history": search_result.get("search_history", []),
                    "candidate_count": len(candidate_items),
                    "is_failed": is_failed
                })
                
            except Exception as e:
                if verbose:
                    print(f"❌ LỖI khi thực thi Bước {i+1}: {e}")
                
                state.failed_steps.append(i)
                
                state.step_results.append({
                    "step_number": i + 1,
                    "task": step_task,
                    "query": semantic_query,
                    "fact_requirements": fact_requirements,
                    "required_facts": required_facts,
                    "temporal": temporal_scope,
                    "final_status": "error",
                    "evidences_count": 0,
                    "is_failed": True,
                    "error": str(e)
                })
        
        # BƯỚC 3: Tổng hợp câu trả lời bằng Agent 5
        if verbose and generate_final_answer:
            print(f"\n{'='*80}")
            print("📝 BƯỚC 3: TỔNG HỢP CÂU TRẢ LỜI (AGENT 5)")
            print("="*80)
            print(f"Total evidences: {len(state.gathered_evidences)}")
        
        if generate_final_answer:
            final_answer = self._generate_final_answer(
                original_query=user_query,
                evidences=state.gathered_evidences,
                failed_steps=state.failed_steps,
                incomplete_searches=state.incomplete_searches,
                plan=state.plan,
                verbose=verbose
            )
        else:
            final_answer = ""
            if verbose:
                print(
                    "\n[Orchestrator] Da dong bang evidence; "
                    "bo qua Agent 5 trong pha nay."
                )
        
        # BƯỚC 4: Tổng kết
        if verbose:
            print(f"\n{'='*80}")
            print("⚖️  BƯỚC 4: KẾT THÚC WORKFLOW")
            print("="*80)
            print(f"✓ Đã thực thi {len(state.plan)} bước")
            print(f"✓ Bước thành công: {len(state.plan) - len(state.failed_steps)}")
            print(f"✓ Bước thất bại: {len(state.failed_steps)}")
            print(f"✓ Tổng evidences: {len(state.gathered_evidences)}")
            
            if state.failed_steps:
                print(f"\n⚠️  Các bước thất bại: {[i+1 for i in state.failed_steps]}")
            
            if state.quality_issues:
                print(f"\n⚠️  Quality issues: {len(state.quality_issues)}")
            
            print("="*80)
        
        # Trả về kết quả đầy đủ
        return {
            "final_answer": final_answer,
            "all_evidences": state.gathered_evidences,
            "candidate_task_sets": state.candidate_task_sets,
            "plan": state.plan,
            "step_results": state.step_results,
            "failed_steps": state.failed_steps,
            "quality_issues": state.quality_issues,
            "incomplete_searches": state.incomplete_searches,
            "original_query": state.original_query
        }
    
    def _generate_final_answer(
        self,
        original_query: str,
        evidences: List,
        failed_steps: List,
        incomplete_searches: List[str],
        plan: List,
        verbose: bool = True,
        raise_on_error: bool = False,
    ) -> str:
        """
        Agent 5: Final Generator - Tổng hợp evidences thành câu trả lời người dùng
        
        Args:
            original_query: Câu hỏi gốc từ user
            evidences: List evidences từ Agent 3+4
            failed_steps: Các bước thất bại
            plan: Kế hoạch đã thực thi
            verbose: In log
        
        Returns:
            Câu trả lời cuối cùng
        """
        if verbose:
            print(f"\n[Agent 5] Đang tổng hợp {len(evidences)} evidences...")

        # Nếu không có evidences nào
        if len(evidences) == 0:
            if verbose:
                print(f"[Agent 5] ⚠️ Không có evidences, trả lời không tìm thấy")
            
            return f"""Dựa trên tra cứu trong cơ sở dữ liệu pháp luật thuế thu nhập cá nhân Việt Nam hiện hành, tôi chưa tìm thấy quy định cụ thể để trả lời câu hỏi: "{original_query}".

Có thể:
- Quy định này chưa được ban hành hoặc đang trong quá trình soạn thảo
- Câu hỏi nằm ngoài phạm vi Luật thuế thu nhập cá nhân hiện hành
- Cần tra cứu thêm trong các văn bản hướng dẫn thi hành hoặc thông tư chi tiết

Khuyến nghị: Liên hệ cơ quan thuế địa phương hoặc tư vấn viên thuế để được hướng dẫn chính xác."""
        
        # Chuẩn bị evidences text
        evidences_text = ""
        for i, ev in enumerate(evidences, 1):
            evidences_text += f"\n[Evidence {i}]\n"
            evidences_text += f"Evidence ID: {ev.chunk_id}\n"
            evidences_text += f"Văn bản: {ev.van_ban}\n"
            evidences_text += f"Điều khoản: {ev.dieu_khoan}\n"
            evidences_text += f"Tình trạng: {ev.tinh_trang_hieu_luc}\n"
            evidences_text += (
                f"Phiên bản: {ev.version or 1}; "
                f"Khoảng hiệu lực: [{ev.valid_from or 'không rõ'}, "
                f"{ev.valid_to or 'chưa kết thúc'})\n"
            )
            if getattr(ev, "temporal_scope", None):
                evidences_text += (
                    "Phạm vi thời gian được yêu cầu: "
                    f"{describe_temporal_scope(ev.temporal_scope)}\n"
                )
            noi_dung_display = ev.noi_dung_goc if ev.noi_dung_goc else "(Không có nội dung)"
            evidences_text += f"Nội dung: {noi_dung_display}\n"
            if ev.amended_by_document_id:
                evidences_text += (
                    f"Được sửa bởi văn bản: {ev.amended_by_document_id}\n"
                    f"Vị trí sửa đổi: "
                    f"{ev.amended_by_source_location or 'Không rõ'}\n"
                    f"Loại thay đổi: {ev.change_type or 'Không rõ'}\n"
                )
            
            if ev.cac_bien_so_phap_ly:
                evidences_text += f"Biến số pháp lý:\n"
                bs = ev.cac_bien_so_phap_ly
                if bs.thue_suat:
                    evidences_text += f"  - Thuế suất: {bs.thue_suat}\n"
                if bs.nguong_mien_thue:
                    evidences_text += f"  - Ngưỡng miễn thuế: {bs.nguong_mien_thue}\n"
                if bs.cach_tinh:
                    evidences_text += f"  - Cách tính: {bs.cach_tinh}\n"
                if bs.giam_tru_gia_canh:
                    evidences_text += f"  - Giảm trừ gia cảnh: {bs.giam_tru_gia_canh}\n"
                if bs.thoi_han:
                    evidences_text += f"  - Thời hạn: {bs.thoi_han}\n"
                if bs.dieu_kien_ap_dung:
                    evidences_text += f"  - Điều kiện: {bs.dieu_kien_ap_dung}\n"
            evidences_text += "\n"

        incomplete_text = (
            "\n".join(f"- {item}" for item in incomplete_searches)
            if incomplete_searches
            else "(Không có; các bước tra cứu đã đủ evidence.)"
        )
        temporal_plan_text = "\n".join(
            f"- {step['task']}: {describe_temporal_scope(step['temporal'])}; "
            f"required_facts={json.dumps(step.get('required_facts') or [], ensure_ascii=False)}"
            for step in plan
        )

        verified_calculations = self._prepare_verified_calculations(
            original_query=original_query,
            evidences=evidences,
            evidences_text=evidences_text,
        )
        self.last_verified_calculations = verified_calculations
        calculation_text = (
            json.dumps(
                verified_calculations,
                ensure_ascii=False,
                indent=2,
            )
            if verified_calculations
            else "(Không có phép tính deterministic đã được xác minh.)"
        )
        
        # Prompt cho Agent 5
        prompt = f"""Bạn là Chuyên gia tư vấn pháp luật thuế Việt Nam. Nhiệm vụ của bạn là tổng hợp thông tin pháp lý thành câu trả lời dễ hiểu cho người dùng.

================================================================================
CÂU HỎI TỪ NGƯỜI DÙNG:
================================================================================
{original_query}

PHẠM VI THỜI GIAN CỦA CÁC NHIỆM VỤ:
{temporal_plan_text}

================================================================================
CĂN CỨ PHÁP LÝ ĐÃ TRA CỨU:
================================================================================
{evidences_text}

================================================================================
THÔNG TIN PHÁP LÝ CÒN THIẾU SAU KHI HẾT RETRY:
================================================================================
{incomplete_text}

================================================================================
KẾT QUẢ TỪ CALCULATION TOOL:
================================================================================
{calculation_text}

================================================================================
QUY TẮC TRẢ LỜI:
================================================================================

1. CẤU TRÚC CÂU TRẢ LỜI:
   
   a) MỞ ĐẦU - Câu trả lời trực tiếp (1-2 câu)
      - Trả lời NGAY câu hỏi của user
      - Nêu rõ con số (thuế suất, ngưỡng, thời hạn) nếu có
   
   b) CĂN CỨ PHÁP LÝ
      - Trích dẫn: Theo [Điều X Khoản Y] của [Văn bản]...
      - Trích nguyên văn nội dung QUAN TRỌNG (không cần trích hết)
   
   c) GIẢI THÍCH VÀ CÁCH TÍNH (nếu có)
      - Giải thích cách tính thuế
      - Chỉ nêu công thức tổng quát nếu evidences có căn cứ về cách tính
      - Chỉ thay số và nêu kết quả cuối cùng từ CALCULATION TOOL
      - KHÔNG tự tạo ví dụ minh họa, tình huống giả định hoặc số liệu giả định
   
   d) LƯU Ý (nếu cần)
      - Điều kiện áp dụng
      - Ngoại lệ
      - Khuyến nghị

2. NGUYÊN TẮC VÀNG:
   
   ✅ PHẢI:
   - Trả lời CHÍNH XÁC, dựa 100% vào evidences
   - Chỉ sử dụng thông tin có nguyên văn trong CÂU HỎI TỪ NGƯỜI DÙNG hoặc CĂN CỨ PHÁP LÝ ĐÃ TRA CỨU
   - Trích dẫn đầy đủ Điều/Khoản/Văn bản
   - Ngôn ngữ DỄ HIỂU, thân thiện
   - Chỉ nêu CON SỐ PHÁP LÝ hoặc DỮ KIỆN ĐẦU VÀO (thuế suất, ngưỡng, doanh thu, thời hạn...) nếu chính con số đó xuất hiện trong câu hỏi hoặc evidences
   - KẾT QUẢ SỐ HỌC không cần xuất hiện nguyên văn trong evidences, nhưng phải là kết quả đã được CALCULATION TOOL xác minh
   - Nếu mục THÔNG TIN PHÁP LÝ CÒN THIẾU không phải "Không có", PHẢI nói rõ giới hạn đó. KHÔNG được tính số tiền cuối cùng nếu phần còn thiếu ảnh hưởng đến căn cứ tính, ngưỡng, thuế suất hoặc phương pháp tính
   - Nội dung của evidence là nội dung thuộc đúng version trong khoảng hiệu lực đã ghi; không tự khôi phục nội dung từ version cũ
   - Với current/exact_date, chỉ dùng version bao phủ ngày đã khóa.
   - Với month/year/date_range, nếu cùng provision có nhiều version trong khoảng,
     phải trình bày rõ từng giai đoạn áp dụng; không gộp nội dung của các version.
   - Nếu evidence có "Được sửa bởi văn bản", dùng thông tin này để nêu nguồn hình thành version hiện hành khi cần
   - Trước khi tính số tiền cuối cùng, phải kiểm tra tất cả toán hạng của phép tính đều có trong câu hỏi hoặc evidences
   - Nếu CALCULATION TOOL có kết quả hợp lệ, phải dùng đúng kết quả đó và không tự tính lại bằng một biểu thức khác
   - Nếu CALCULATION TOOL báo lỗi hoặc không có kết quả, không được tự tạo toán hạng để bù vào
   - Phải ghép dữ kiện nằm ở nhiều nơi: dữ kiện thực tế từ câu hỏi, căn cứ tính từ evidence này, ngưỡng từ evidence/relation khác và thuế suất từ evidence khác
   - Nếu evidence quy định "phần vượt ngưỡng", câu hỏi cho doanh thu D, evidence cho ngưỡng N và thuế suất R, phải tính: doanh thu tính thuế = D - N; thuế phải nộp = (D - N) × R
   - Không được sửa biểu thức hoặc kết quả mà CALCULATION TOOL đã xác minh
   - Nếu thiếu ngưỡng, doanh thu, thuế suất, căn cứ tính thuế hoặc điều kiện áp dụng: chỉ nêu phần đã có, công thức tổng quát nếu có căn cứ, và nói rõ dữ kiện còn thiếu
   
   ❌ KHÔNG ĐƯỢC:
   - Tự bịa thông tin không có trong evidences
   - Suy diễn hoặc đoán mò
   - Dùng kiến thức nhớ sẵn, quy định cũ hoặc kiến thức bên ngoài evidences để điền phần còn thiếu
   - Tự đặt ngưỡng, doanh thu, số tiền, tỷ lệ, thời hạn hoặc giả định bằng các cụm như "giả sử", "ví dụ", "thông thường"
   - Tính ra số tiền cụ thể khi bất kỳ toán hạng nào không xuất hiện trong câu hỏi hoặc evidences
   - Tự tính lại hoặc suy ra một kết quả số học khác với CALCULATION TOOL
   - Vừa đưa ra một số tiền cụ thể, vừa nói rằng toán hạng dùng để tính số tiền đó còn thiếu
   - Khẳng định một phương pháp, quyền lựa chọn hoặc điều kiện áp dụng nếu evidence không nói rõ
   - Dùng ngôn ngữ pháp lý khô khan
   - Lặp lại toàn bộ evidences một cách máy móc
   - Dùng lại ngưỡng, số liệu hoặc nội dung cũ đã bị sửa đổi, thay thế hay bãi bỏ

3. XỬ LÝ TRƯỜNG HỢP ĐẶC BIỆT:
   
   - Nếu evidences KHÔNG ĐỦ để trả lời: Nói rõ "Các căn cứ đã tra cứu chưa cung cấp đủ thông tin về..."; không được kết luận pháp luật không có quy định
   - Nếu có NHIỀU evidences về cùng chủ đề: Tổng hợp thành 1 câu trả lời mạch lạc
   - Nếu evidences MÂU THUẪN trực tiếp cùng đối tượng, vấn đề, điều kiện và thời điểm áp dụng: ưu tiên theo thứ bậc LUẬT > NGHỊ ĐỊNH > THÔNG TƯ; không chỉ ưu tiên vì văn bản cấp dưới mới hơn
   - Không tự ghép lại hoặc sửa thêm nội dung evidence; Step 3 đã tái tạo đúng phạm vi Điều/Khoản/Điểm trước khi embedding

================================================================================
RÀ SOÁT BẮT BUỘC TRƯỚC KHI TRẢ LỜI:
================================================================================

- Mỗi con số trong câu trả lời có xuất hiện trong câu hỏi hoặc evidences không?
- Nếu là kết quả số học mới, nó có đúng bằng kết quả CALCULATION TOOL không?
- Mỗi phép tính được trình bày có đúng biểu thức và toán hạng đã xác minh không?
- Nếu CALCULATION TOOL có kết quả, đã thay số và trả kết quả cuối cùng chưa?
- Câu trả lời có vừa tính ra một số tiền vừa tuyên bố dữ kiện dùng để tính còn thiếu không?
- Mỗi kết luận về điều kiện, phương pháp hoặc quyền lựa chọn có được evidence nói rõ không?
- Nếu câu trả lời có dữ kiện không vượt qua một trong các kiểm tra trên, phải xóa dữ kiện đó và nêu rõ thông tin còn thiếu.

Hãy viết câu trả lời chuyên nghiệp, chính xác, dễ hiểu và hoàn toàn bị ràng buộc bởi evidences!"""
        
        try:
            import google.generativeai as genai

            response = self.generator_llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                )
            )
            final_answer = response.text.strip()
            
            if verbose:
                print(f"[Agent 5] ✓ Đã tạo câu trả lời ({len(final_answer)} ký tự)")
            
            return final_answer
            
        except Exception as e:
            if verbose:
                print(f"[Agent 5] ❌ Lỗi: {e}")

            if raise_on_error:
                raise
            
            # Fallback: Trả về evidences thô
            fallback = f"Căn cứ pháp lý tìm được:\n\n{evidences_text}\n\nLưu ý: Hệ thống gặp lỗi khi tổng hợp câu trả lời. Vui lòng tham khảo căn cứ pháp lý trên."
            return fallback

    @staticmethod
    def _needs_calculation(query: str) -> bool:
        text = str(query or "").casefold()
        return bool(
            re.search(
                r"\b(tính|bao nhiêu|số thuế|thuế phải nộp|"
                r"nghĩa vụ thuế|chênh lệch|tổng số)\b",
                text,
            )
            and re.search(r"\d", text)
        )

    def _prepare_verified_calculations(
        self,
        *,
        original_query: str,
        evidences: List,
        evidences_text: str,
    ) -> List[Dict]:
        if not self._needs_calculation(original_query):
            return []

        planning_prompt = f"""CÂU HỎI:
{original_query}

EVIDENCE:
{evidences_text}

Chỉ lập các phép tính thật sự cần để trả lời. Mỗi biến phải trích source_text
nguyên văn liên tục. source có thể là question, evidence_N hoặc calculation_N
đã được xác minh ở bước trước. Nếu source_text có nhiều số, dùng occurrence
(đếm từ 1) để chọn đúng số. Tỷ lệ phần trăm phải chuẩn hóa sang số thập phân,
ví dụ 1,5% thành 0.015. Sắp phép tính theo thứ tự phụ thuộc; không tham chiếu
phép tính hiện tại hoặc phép tính phía sau."""
        try:
            import google.generativeai as genai

            schema_prompt = planning_prompt + """

Trả JSON:
{"calculations":[{"label":"","expression":"","variables":[
{"name":"","value":"","source":"question|evidence_N|calculation_N","source_text":"","occurrence":1}
],"unit":""}]}
Không cần tính thì trả {"calculations":[]}."""
            response = self.generator_llm.generate_content(
                schema_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            plan = json.loads(response.text)

            self.last_calculation_plan = plan

            evidence_data = [
                (
                    evidence.model_dump()
                    if hasattr(evidence, "model_dump")
                    else evidence.dict()
                )
                for evidence in evidences
            ]
            return execute_calculation_plan(
                plan,
                question=original_query,
                evidences=evidence_data,
            )
        except Exception as exc:
            self.last_calculation_plan = {
                "calculations": [],
                "error": str(exc),
            }
            return [{"label": "Calculation tool", "error": str(exc)}]


def test_orchestrator():
    """Hàm test đơn giản cho Orchestrator"""
    print("\n" + "="*80)
    print("🧪 TEST PLANNER ORCHESTRATOR (Agent 3+4+5 Integrated)")
    print("="*80)
    
    # Tạo orchestrator (tự động khởi tạo Agent 3, 4, 5)
    print("\n[1/2] Khởi tạo Planner Orchestrator...")
    orchestrator = PlannerOrchestrator()
    
    # Test query
    test_query = "Thuế suất thu nhập từ chuyển nhượng tài sản số là bao nhiêu?"
    
    print(f"\n[2/2] Test với câu hỏi: {test_query}")
    print("="*80)
    
    # Chạy workflow
    result = orchestrator.execute_workflow(test_query, verbose=True)
    
    # In kết quả
    print(f"\n{'='*80}")
    print("📊 KẾT QUẢ CUỐI CÙNG")
    print("="*80)
    
    print(f"\n📋 Kế hoạch đã thực thi:")
    for i, step in enumerate(result['plan'], 1):
        print(f"   {i}. {step}")
    
    print(f"\n📦 Evidences thu thập:")
    print(f"   Total: {len(result['all_evidences'])}")
    for i, ev in enumerate(result['all_evidences'], 1):
        print(f"   [{i}] {ev.dieu_khoan} - {ev.van_ban}")
    
    print(f"\n💬 CÂU TRẢ LỜI CUỐI CÙNG:")
    print("─"*80)
    print(result['final_answer'])
    print("─"*80)
    
    print(f"\n✅ Test hoàn tất!")


if __name__ == "__main__":
    test_orchestrator()
