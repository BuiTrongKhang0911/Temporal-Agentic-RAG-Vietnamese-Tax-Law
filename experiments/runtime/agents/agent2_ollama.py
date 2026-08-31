"""Agent 2 for Ollama: 2A task/temporal planning, then 2B fact decomposition."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from ..retriever.temporal import normalize_temporal_scope


FACT_TYPES = [
    "applicability",
    "tax_base",
    "tax_rate",
    "threshold_or_exemption",
    "calculation_method",
    "obligation",
    "procedure",
    "deadline",
    "authority",
]

MAX_TASKS = 3
MAX_FACTS_PER_TASK = 6


# Retained temporarily as design history while the per-task temporal planner
# is evaluated against the existing benchmark.
_LEGACY_AGENT2_FUNCTION_CHOICE_PROMPT = """Bạn là Agent 2A của hệ thống tra cứu pháp luật thuế Việt Nam.

NHIỆM VỤ DUY NHẤT
1. Chọn đúng một function thể hiện phạm vi thời gian pháp lý người dùng muốn tra cứu.
2. Chia câu hỏi thành 1-3 nhiệm vụ tra cứu độc lập.
3. Không tạo fact_requirements, không trả lời câu hỏi và không tự điền giá trị pháp luật.

CHỌN FUNCTION THỜI GIAN
- submit_current_tasks: không có mốc tra cứu, hoặc hỏi hiện nay/hiện hành.
- submit_exact_date_tasks: tra cứu đúng một ngày được nêu rõ.
- submit_month_tasks: tra cứu một tháng cụ thể.
- submit_year_tasks: tra cứu một năm hoặc kỳ tính thuế năm.
- submit_date_range_tasks: tra cứu một khoảng liên tục từ A đến B.
- submit_comparison_tasks: chỉ khi so sánh pháp luật tại ít nhất hai mốc thời gian rời nhau.

BIÊN CỦA KHOẢNG THỜI GIAN
- Với submit_date_range_tasks, mỗi biên phải giữ đúng độ chi tiết người dùng đã nêu:
  * granularity="year", value="2023" nếu người dùng chỉ nêu năm 2023;
  * granularity="month", value="2023-08" nếu người dùng nêu tháng 8/2023;
  * granularity="day", value="2023-08-15" nếu người dùng nêu ngày cụ thể;
  * riêng biên to được phép dùng granularity="current", value="current" khi người dùng nói
    "đến nay", "đến hiện tại" hoặc "đến thời điểm hiện hành".
- Không được dùng current cho biên from. Nếu người dùng chỉ hỏi hiện nay, dùng submit_current_tasks.
- Không tự đổi year/month thành day và không tự tính ngày đầu hoặc ngày cuối kỳ. Python sẽ mở rộng:
  from-year thành ngày đầu năm, from-month thành ngày đầu tháng; to-year thành ngày cuối năm,
  to-month thành ngày cuối tháng; day được giữ nguyên và to-current thành ngày chạy hệ thống.
- Không nâng hoặc hạ độ chính xác của mốc thời gian. Ví dụ "từ 2023 đến tháng 6/2025"
  phải giữ from ở granularity year và to ở granularity month.

PHÂN BIỆT THỜI GIAN TRA CỨU VÀ DỮ KIỆN TÌNH HUỐNG
- Ngày giao dịch, ngày nộp hồ sơ, kỳ khai thuế, hạn cần kiểm tra và số ngày hiện diện
  chỉ là dữ kiện tình huống, trừ khi câu hỏi yêu cầu pháp luật có hiệu lực tại mốc đó.
- Tháng/quý/năm của khoản thuế hoặc kỳ khai thuế cần xử lý không tự động là temporal.
- Nếu người dùng yêu cầu tính hoặc xác định nghĩa vụ thuế phát sinh trong chính tháng,
  năm hay ngày đã nêu, mốc đó là temporal. Giữ nguyên độ chi tiết: tháng vẫn là month,
  không đổi thành ngày cuối tháng.
- Ngày đăng ký hộ kinh doanh, ngày bắt đầu làm việc hoặc ngày nộp hồ sơ chỉ mô tả hoàn
  cảnh; nếu câu hỏi hỏi quy định hiện hành thì vẫn dùng current.
- Mọi ngày, tháng, năm, from/to phải sao chép từ mốc tra cứu xuất hiện rõ trong câu hỏi.
- Không suy diễn ngày hiệu lực, không đổi ngày thành tháng/năm, không tạo ngày mới.
- Cụm "từ ngày D" dùng để hỏi quy định bắt đầu tại D là mốc tra cứu, không phải current.
- Nếu câu hỏi có cấu trúc "Trong năm Y, ... thay đổi thế nào từ ngày D", chọn
  submit_year_tasks(year=Y) để lấy được cả quy định trước và sau D trong năm đó.
- Nếu câu hỏi chỉ hỏi quy định áp dụng "từ ngày D" và không nêu ngày kết thúc, chọn
  submit_exact_date_tasks(date=D). Không tự tạo khoảng đến 2099 hoặc một ngày tương lai.
- Với comparison, mỗi comparison_scope phải sao chép đúng từng mốc trong câu hỏi.
  Không được thay scope thiếu dữ liệu bằng current và không tạo nhiều task trùng task_id.

CHIA TASK
- Trước khi chia task, tự liệt kê mọi đầu ra người dùng yêu cầu. Không được bỏ sót
  một vế chỉ vì câu hỏi kết thúc bằng một kết quả tổng hợp.
- Một kết quả trực diện chỉ là một task khi nó dùng cùng một nhánh căn cứ pháp lý.
- Tách task khi có nhiều giao dịch, loại thuế, loại thu nhập, đầu ra độc lập hoặc
  các thành phần của phép tính phải tra cứu theo những quy định khác nhau.
- Nếu câu hỏi yêu cầu tổng của nhiều khoản thuế/loại thu nhập, tách từng khoản cần
  tính thành task riêng; Agent 5 mới cộng và kết luận tổng cuối cùng.
- Nhiều điều kiện và số liệu phục vụ cùng một kết luận vẫn là một task.
- Mỗi task phải tự chứa đủ loại thuế, đối tượng, hoạt động/thu nhập, tình trạng cư trú,
  vai trò, địa điểm và số liệu liên quan từ câu hỏi gốc.
- Nếu câu hỏi nêu nhiều loại thuế độc lập, không được bỏ sót hoặc đổi tên loại thuế.
- Nếu hai loại thu nhập có cách tính khác nhau, tách thành hai task.
- Với nhiều nghĩa vụ hành chính khác nhau như xử lý thuế đã nộp, thông báo doanh thu
  và hóa đơn điện tử, ưu tiên task riêng cho từng đầu ra nếu không vượt quá 3 task.
- Không tạo task định nghĩa cho thuật ngữ phổ thông; đi thẳng vào kết quả pháp lý.
- Nếu chỉ đích danh Điều/Khoản/Điểm, mô tả trung tính nội dung cần tra cứu, không tự đoán
  đó là miễn thuế, nghĩa vụ hay thủ tục.
- query phải ngắn, bằng tiếng Việt, giữ nguyên thuật ngữ và không chứa filter Qdrant.
- Không thêm task về thủ tục, ngưỡng, miễn trừ hay hiệu lực nếu người dùng không hỏi.

BẢO TOÀN NGỮ NGHĨA
- task và query chỉ được diễn đạt lại nhu cầu tra cứu, tuyệt đối không dự đoán đáp án.
- Giữ nguyên tên loại thuế, loại thu nhập/hoạt động, phương pháp tính, đối tượng,
  số tiền, tỷ lệ và đơn vị xuất hiện trong câu hỏi gốc.
- Không thêm khái niệm pháp lý, điều kiện, ngưỡng, thuế suất hoặc con số từ trí nhớ.
- Không tính tổng, hiệu, tỷ lệ hoặc suy ra một giá trị mới trong task/query. Agent 5
  mới thực hiện phép tính sau khi có evidence.
- Cấm thay thế các khái niệm gần nghĩa nhưng khác chế định, đặc biệt:
  * "mức trừ doanh thu" không phải "giảm trừ gia cảnh";
  * "tiền chậm nộp" không phải "mức phạt vi phạm hành chính";
  * "doanh thu" không được đổi thành "doanh số";
  * "tính trên thu nhập" không được tự đổi thành "phương pháp kê khai";
  * "không chịu thuế" không được tự đổi thành "miễn thuế".
- Khi chưa biết tên chế định chính xác, dùng mô tả trung tính bám sát câu hỏi thay vì
  tự đặt tên pháp lý cho nó.
- Không được thêm tên gọi thay thế trong ngoặc. Ví dụ task có "phương pháp tính trên
  thu nhập" thì không được viết thêm "phương pháp kê khai"; có "doanh thu" thì cả
  task và query đều phải tiếp tục dùng "doanh thu".

TỰ KIỂM TRA TASK
- Function và tham số thời gian có đúng chính xác câu hỏi không?
- Mọi đầu ra độc lập người dùng yêu cầu đã có task bao phủ chưa?
- Có task trùng hoặc task thừa không?
- Mỗi task có đủ dữ kiện để Agent 3 tra cứu mà không cần đoán từ câu hỏi gốc không?
- Có đổi tên khái niệm pháp lý hoặc tự thêm kiến thức không?
- Mọi con số và đơn vị trong task/query có xuất hiện nguyên văn trong câu hỏi gốc không?
- Task/query có vô tình tính toán hoặc kết luận trước điều cần tra cứu không?

VÍ DỤ
- "Hiện nay mức giảm trừ gia cảnh là bao nhiêu?"
  -> submit_current_tasks, một task.
- "Ngày 15/06/2010 mức giảm trừ gia cảnh là bao nhiêu?"
  -> submit_exact_date_tasks(date="2010-06-15").
- "Trong tháng 06/2013 mức giảm trừ gia cảnh là bao nhiêu?"
  -> submit_month_tasks(month=6, year=2013).
- "Trong năm 2013 mức giảm trừ thay đổi thế nào?"
  -> submit_year_tasks(year=2013), không phải comparison.
- "Từ 15/06/2013 đến 15/07/2013 mức giảm trừ thay đổi thế nào?"
  -> submit_date_range_tasks(
       from={"granularity":"day","value":"2013-06-15"},
       to={"granularity":"day","value":"2013-07-15"}).
- "Từ năm 2023 đến năm 2025 quy định thay đổi thế nào?"
  -> submit_date_range_tasks(
       from={"granularity":"year","value":"2023"},
       to={"granularity":"year","value":"2025"}).
- "Từ tháng 8/2023 đến hiện tại quy định thế nào?"
  -> submit_date_range_tasks(
       from={"granularity":"month","value":"2023-08"},
       to={"granularity":"current","value":"current"}).
- "So sánh mức thuế ngày 31/12/2014 và ngày 01/01/2015."
  -> submit_comparison_tasks với hai exact_date.
- "Tôi hiện diện 190 ngày trong năm, hiện nay có là cá nhân cư trú không?"
  -> submit_current_tasks; 190 ngày là dữ kiện tình huống.
- "Thuế kỳ tháng 6/2026 được gia hạn đến ngày nào?"
  -> submit_current_tasks; tháng 6/2026 là kỳ thuế, deadline là thứ cần tìm.
- "Tháng 1/2025 tôi nhận lương và thưởng; thuế tháng đó phải nộp bao nhiêu?"
  -> submit_month_tasks(month=1, year=2025); đây là nghĩa vụ phát sinh trong tháng đó.
- "Đăng ký hộ kinh doanh từ tháng 2/2026; hiện nay hoạt động có chịu thuế không?"
  -> submit_current_tasks; tháng đăng ký chỉ là dữ kiện tình huống.
- "Bán hàng từ năm 2019; có bị truy thu không, còn mức thuế và thủ tục hiện nay thế nào?"
  -> tách task: task truy thu dùng date_range từ year 2019 đến current; task mức thuế
  và thủ tục hiện hành dùng current.
- "So sánh phương pháp doanh thu và phương pháp thu nhập hiện hành."
  -> submit_current_tasks; đây không phải so sánh theo thời gian.
- "Cá nhân phân phối hàng hóa, tính thuế thu nhập cá nhân theo tỷ lệ trên doanh thu
  và tính thuế giá trị gia tăng trực tiếp; hỏi tổng hai thuế."
  -> hai task: tra cứu riêng cách tính thuế thu nhập cá nhân và thuế giá trị gia tăng;
  không tự thêm thuế suất, ngưỡng hoặc đáp án.
- "Cá nhân không cư trú có tiền lương và thu nhập đầu tư vốn; hỏi tổng thuế."
  -> hai task theo hai loại thu nhập vì mỗi loại có căn cứ và thuế suất riêng.
- "Hỏi xử lý thuế đã nộp, thông báo doanh thu và hóa đơn điện tử."
  -> ba task tương ứng ba đầu ra; không đổi thành một task kê khai chung.
- "Phân bổ mức trừ 1 tỷ đồng giữa các hoạt động kinh doanh."
  -> giữ nguyên "mức trừ 1 tỷ đồng"; không đổi thành giảm trừ gia cảnh.
- "Tính tiền chậm nộp của khoản thuế 700 triệu đồng trong 20 ngày."
  -> task tra cứu cách tính tiền chậm nộp, giữ nguyên 700 triệu đồng và 20 ngày;
  không đổi thành xử phạt và không tự tính kết quả.

Chỉ gọi đúng một function. Không trả văn bản giải thích."""


AGENT2_FACT_PROMPT = """Bạn là Agent 2B, chuyên phân rã MỘT task tra cứu pháp luật thuế
thành các fact requirements tối thiểu cần evidence trực tiếp.

ĐẦU VÀO
- Câu hỏi gốc để bảo toàn ngữ nghĩa.
- Một task và semantic query đã được Agent 2A khóa.
- Temporal đã được Python khóa; bạn không được sửa hoặc đưa thời gian vào facts.

ĐẦU RA
Gọi submit_facts đúng một lần, tạo 1-6 facts. Mỗi fact chỉ có:
- type: applicability, tax_base, tax_rate, threshold_or_exemption,
  calculation_method, obligation, procedure, deadline hoặc authority.
- description: mô tả trung tính một dữ kiện pháp lý mà một provision có thể chứng minh.
Python sẽ tự gắn fact_id và task_id.

QUY TẮC
- Mỗi kết quả con task cần trả lời phải có fact tương ứng.
- Facts phải nguyên tử, không trùng nghĩa và không chỉ lặp lại toàn bộ task.
- Không trả lời, không điền value/amount/kết luận và không dùng kiến thức nhớ sẵn.
- Không thực hiện phép tính. Không cộng, trừ, nhân, chia hoặc biến nhiều số liệu đầu
  vào thành một số mới trong description.
- Số liệu giao dịch người dùng đã nêu được giữ trong task/query; không tạo fact chỉ
  để nhắc lại doanh thu, tiền lương hoặc số tiền đó.
- Chỉ giữ con số trong description khi chính con số ấy là đối tượng pháp lý cần
  phân loại hoặc tra cứu; không biến dữ kiện đầu vào thành một căn cứ pháp luật.
- Không đổi tên khái niệm pháp lý. Khi chưa chắc, bám sát nguyên văn câu hỏi.
- Bảo toàn nguyên văn mọi neo phân loại có thể quyết định provision cần tìm, gồm:
  loại thuế, loại thu nhập/hoạt động, đối tượng, phương pháp tính, mức tiền,
  tỷ lệ, ngưỡng và đơn vị. Ví dụ câu hỏi hỏi mức thu nhập tính thuế
  "25 triệu đồng/tháng" thì fact tax_rate phải giữ đúng cụm đó.
- Không suy diễn một thuật ngữ sang chế định khác dù nghe có vẻ gần nghĩa.
  "Mức trừ doanh thu", "ngưỡng doanh thu", "giảm trừ gia cảnh", "miễn thuế"
  và "không chịu thuế" là các khái niệm khác nhau; chỉ dùng đúng khái niệm
  xuất hiện trong câu hỏi/task.
- Luôn viết đầy đủ tên thuế. Cấm các chữ viết tắt VAT, GTGT, PIT, TNCN trong
  task facts và description.
- Không tạo fact về văn bản có hiệu lực; temporal đã được Python xử lý.
- Không đưa ngày, tháng, năm, "mốc A/B" hoặc "hiện hành" vào description.
- Không tự thêm thủ tục, ngưỡng, miễn trừ, công thức hoặc điều kiện phụ không cần thiết.
- Không viết kết luận như "là đối tượng chịu thuế". Nếu cần phân loại, viết trung
  tính: "phạm vi và điều kiện áp dụng ... đối với tình huống".
- Không tạo authority nếu người dùng không hỏi cơ quan có thẩm quyền hoặc nơi nộp.
- Mọi con số trong description phải xuất hiện nguyên văn trong câu hỏi gốc. Nếu giá
  trị pháp luật cần tìm chưa được nêu, chỉ gọi tên giá trị đó, không tự điền con số.
- Ngày giao dịch, kỳ tính thuế hoặc số ngày chậm nộp là dữ kiện nội dung thì được giữ
  khi cần xác định provision. Chỉ loại bỏ ngày dùng riêng làm temporal filter.

ÁNH XẠ NHU CẦU
- Hỏi một đối tượng/giao dịch có thuộc một nhóm pháp lý: applicability.
- applicability là cổng kiểm tra đối tượng, giao dịch, loại thu nhập và điều
  kiện phát sinh nghĩa vụ. Fact này có thể giúp Retriever tìm ra một ngưỡng
  hoặc ngoại lệ chưa được nêu trong câu hỏi; không tự khẳng định ngưỡng đó tồn tại.
- threshold_or_exemption là đầu ra riêng về ranh giới định lượng, điều kiện miễn,
  không chịu thuế hoặc không phải nộp. Chỉ tạo khi chính câu hỏi/task:
  * hỏi "từ bao nhiêu", "có vượt ngưỡng", "có được miễn/không phải nộp"; hoặc
  * đưa một giá trị để quyết định nhánh có/không phát sinh nghĩa vụ.
- Không tạo threshold_or_exemption chỉ vì pháp luật có thể tồn tại một ngưỡng
  mà câu hỏi không gợi ra. Trường hợp này dùng applicability trung tính nếu
  việc xác định phạm vi/nghĩa vụ là cần thiết.
- applicability và threshold_or_exemption không được trùng description:
  * applicability mô tả nhóm đối tượng, hoạt động và phạm vi nghĩa vụ;
  * threshold_or_exemption mô tả riêng ranh giới hoặc điều kiện miễn trừ cần tìm.
- Hỏi mức thuế phải nộp hoặc cách tính:
  * nếu câu hỏi chưa khẳng định tình huống chắc chắn phát sinh thuế, tạo
    applicability + tax_base + tax_rate + calculation_method;
  * nếu câu hỏi đã khẳng định thuộc diện chịu thuế hoặc đã khóa phương pháp
    áp dụng và chỉ yêu cầu tính, tạo tax_base + tax_rate + calculation_method;
  * nếu câu hỏi còn trực tiếp yêu cầu xét ngưỡng/miễn trừ, thêm
    threshold_or_exemption.
- Áp dụng ma trận trên RIÊNG cho từng khoản tiền, loại thuế hoặc loại thu nhập độc lập.
  Không để facts của khoản thứ nhất thay thế facts của khoản thứ hai.
- Khi người dùng hỏi "bao nhiêu", "tính số tiền" hoặc so sánh số tiền phải nộp,
  calculation_method là bắt buộc. tax_rate cũng bắt buộc nếu kết quả phụ thuộc vào
  thuế suất, tỷ lệ trên doanh thu, biểu thuế hoặc tỷ lệ tiền chậm nộp.
- tax_base phải bao phủ cách xác định căn cứ tính và các khoản loại trừ trực tiếp làm
  thay đổi căn cứ, nếu pháp luật có quy định. Cách viết này không đồng nghĩa với việc
  tự khẳng định có một ngưỡng hay miễn trừ.
- Với tiền chậm nộp, tax_base là số tiền làm căn cứ; tax_rate là tỷ lệ tính theo thời
  gian; calculation_method là công thức kết hợp căn cứ, tỷ lệ và số ngày.
- Với mỗi task tính một khoản thuế hoặc một loại thu nhập, tax_base và tax_rate phải
  nói rõ đúng khoản thuế/loại thu nhập đó, không gộp với nhánh khác.
- threshold_or_exemption chỉ mô tả trung tính điều kiện/ngưỡng cần kiểm tra; tuyệt đối
  không tự điền giá trị ngưỡng nếu người dùng không nêu.
- Hỏi riêng thuế suất: tax_rate; thêm applicability nếu phải phân loại đối tượng trước.
- Hỏi riêng thời hạn: deadline; thêm applicability chỉ khi cần xác định đúng đối tượng.
- Hỏi thủ tục: chỉ lấy đúng procedure, authority, deadline hoặc obligation được yêu cầu.
- procedure chỉ dùng khi cần trình tự, hồ sơ, cách khai/nộp hoặc các bước thực hiện.
  Điều kiện được lựa chọn một phương pháp là applicability; việc lựa chọn ràng buộc
  kỳ sau là obligation. Không gọi hai nội dung này là procedure.
- Hai loại thuế hoặc hai loại thu nhập có cách tính khác nhau không được gộp vào một
  fact tax_rate hoặc tax_base mơ hồ.
- Nếu nhiều hành vi/đối tượng trong cùng task cùng được kiểm tra theo đúng một quy
  tắc pháp lý, tạo một fact tổng hợp giữ đủ danh sách đó; không tách thành nhiều
  facts trùng type. Chỉ tách khi từng hành vi cần provision hoặc quy tắc khác nhau.
- Với thuế tính theo tỷ lệ phần trăm trên doanh thu, mô tả là "tỷ lệ thuế tính trên
  doanh thu", không tự đổi thành "thuế suất" nếu task không dùng thuật ngữ đó.
- Nếu task yêu cầu nhiều đầu ra như xử lý thuế đã nộp, thông báo doanh thu và hóa đơn
  điện tử, phải có facts riêng bao phủ đủ từng đầu ra.

VÍ DỤ ĐẦU RA
1. Task: "Cá nhân chuyển nhượng chứng khoán phải nộp bao nhiêu thuế?"
   -> facts:
   - applicability: "phạm vi áp dụng thuế đối với chuyển nhượng chứng khoán của cá nhân"
   - tax_base: "căn cứ tính thuế đối với chuyển nhượng chứng khoán của cá nhân"
   - tax_rate: "thuế suất hoặc tỷ lệ thuế đối với chuyển nhượng chứng khoán của cá nhân"
   - calculation_method: "cách tính số thuế đối với chuyển nhượng chứng khoán của cá nhân"
   Không thêm một ngưỡng hoặc con số nào.
2. Task: "Hộ kinh doanh có doanh thu 900 triệu đồng có phải nộp thuế không?"
   -> facts:
   - applicability: "phạm vi phát sinh nghĩa vụ thuế của hộ kinh doanh"
   - threshold_or_exemption: "ngưỡng doanh thu quyết định nghĩa vụ thuế của hộ kinh doanh có doanh thu 900 triệu đồng"
3. Task: "Tính thuế giá trị gia tăng trực tiếp đối với hoạt động phân phối hàng hóa."
   -> facts:
   - tax_base: "căn cứ tính thuế giá trị gia tăng trực tiếp đối với hoạt động phân phối hàng hóa"
   - tax_rate: "tỷ lệ thuế giá trị gia tăng tính trên doanh thu đối với hoạt động phân phối hàng hóa"
   - calculation_method: "cách tính số thuế giá trị gia tăng trực tiếp đối với hoạt động phân phối hàng hóa"
4. Task: "Tính tiền chậm nộp của khoản thuế 700 triệu đồng trong 20 ngày."
   -> facts:
   - tax_base: "số tiền làm căn cứ tính tiền chậm nộp đối với khoản thuế 700 triệu đồng"
   - tax_rate: "tỷ lệ tính tiền chậm nộp theo thời gian"
   - calculation_method: "cách tính tiền chậm nộp trong 20 ngày"
   Không tự tính 700 triệu đồng nhân với tỷ lệ hoặc 20 ngày.
5. Task: "Thuế suất lũy tiến cho phần thu nhập tính thuế 25 triệu đồng/tháng."
   -> một tax_rate giữ nguyên "25 triệu đồng/tháng"; không tự điền bậc hoặc thuế suất.
6. Task: "Phân bổ mức trừ 1 tỷ đồng giữa các hoạt động kinh doanh."
   -> giữ nguyên "mức trừ 1 tỷ đồng"; không tạo fact về giảm trừ gia cảnh.
7. Task: "Xác định hạn nộp hồ sơ khai thuế."
   -> một deadline; không tự thêm procedure hoặc authority.
8. Task đã khẳng định khoản thu nhập thuộc diện chịu thuế và chỉ yêu cầu tính.
   -> tax_base + tax_rate + calculation_method; không tự thêm applicability hay threshold.

TỰ KIỂM TRA FACTS
- Mọi kết quả con đã được bao phủ chưa?
- Có đúng 1-6 facts nguyên tử và không trùng không?
- Có fact thừa mà câu trả lời không cần không?
- applicability có thực sự cần để xác định phạm vi/nghĩa vụ, hay chỉ lặp lại task?
- threshold_or_exemption có được câu hỏi gợi ra như một ranh giới/miễn trừ
  cần tìm, hay đang bị tự suy diễn từ kiến thức nhớ sẵn?
- Nếu có cả applicability và threshold_or_exemption, hai description đã tách rõ
  phạm vi áp dụng và ranh giới định lượng chưa?
- Có vô tình trả lời hoặc tự thêm khái niệm/con số không?
- Mọi con số trong facts có xuất hiện nguyên văn trong câu hỏi gốc không?
- Có fact nào chứa kết quả cộng/trừ/nhân/chia do model tự tính không?
- Mỗi khoản tiền cần tính đã có đủ tax_base, tax_rate và calculation_method chưa?
- Mọi mức tiền, ngưỡng và đơn vị quyết định việc tra cứu đã được giữ nguyên chưa?
- Có đổi sai chế định, dùng chữ viết tắt thuế hoặc tách một quy tắc chung thành
  nhiều facts trùng nhau không?
- Mỗi fact có thể được một provision chứng minh trực tiếp không?

Chỉ gọi submit_facts. Không trả văn bản giải thích."""


AGENT2_BATCH_FACT_PROMPT = AGENT2_FACT_PROMPT + """

CHẾ ĐỘ BATCH CHO QWEN
- INPUT JSON chứa toàn bộ 1-3 tasks đã được Agent 2A khóa.
- Xử lý từng task độc lập. Fact của task nào chỉ được trả trong nhóm
  task_id đó.
- Không tạo task mới, không bỏ sót task, không gộp hoặc chuyển fact
  giữa các task.
- Mỗi task_id phải xuất hiện đúng một lần và có 1-6 facts.
- Cấm lấy một nghĩa vụ được yêu cầu ở task này làm fact cho task khác.

Trong chế độ này, bỏ qua chỉ dẫn submit_facts phía trên.
Chỉ gọi submit_task_facts đúng một lần. Không trả văn bản ngoài JSON."""


AGENT2_FUNCTION_CHOICE_PROMPT = """Bạn là Agent 2A của hệ thống tra cứu pháp luật thuế Việt Nam.

NHIỆM VỤ
1. Chia câu hỏi thành 1-3 nhiệm vụ tra cứu độc lập.
2. Gắn phạm vi thời gian pháp lý RIÊNG cho từng task.
3. Chỉ gọi submit_task_plan. Không tạo facts và không trả lời câu hỏi.

MỖI TASK GỒM
- task: đầu ra pháp lý cần xác định, tự chứa đủ đối tượng, hoạt động, loại thuế và số liệu liên quan.
- query: truy vấn ngắn cho Retriever, không chứa cú pháp filter.
- temporal: thời điểm mà quy định pháp luật cần được tra cứu cho chính task đó.

TEMPORAL HỢP LỆ
- current: hỏi hiện nay/hiện hành hoặc không nêu thời điểm pháp luật cần tra cứu.
- exact_date: tra đúng một ngày, dùng date=YYYY-MM-DD.
- month: tra toàn bộ một tháng, dùng month và year.
- year: tra toàn bộ một năm.
- date_range: khoảng liên tục với from/to giữ granularity year, month hoặc day; riêng to được phép current.
- comparison: so sánh quy định tại ít nhất hai mốc rời nhau trong comparison_scopes.

QUY TẮC THỜI GIAN
- Temporal là thời điểm áp dụng của QUY ĐỊNH cần tìm, không mặc định là mọi ngày xuất hiện trong tình huống.
- Ngày giao dịch, ngày nộp hồ sơ, kỳ khai thuế, số ngày hiện diện hoặc số ngày chậm nộp chỉ là dữ kiện
  nếu câu hỏi không yêu cầu tra pháp luật tại mốc đó.
- Nếu các đầu ra cần pháp luật ở các mốc khác nhau, tách task và gắn temporal riêng.
- Biên from giữ đúng độ chi tiết year/month/day người dùng nêu. Biên to còn được phép current.
- Không dùng current cho from. Không tự tính ngày đầu/cuối kỳ; Python sẽ mở rộng.
- Không suy diễn ngày hiệu lực và không tạo mốc không có trong câu hỏi hoặc câu trả lời làm rõ.

CHIA TASK
- Tách khi có loại thuế, loại thu nhập, giao dịch, đầu ra hoặc mốc pháp lý độc lập.
- Không tách nhiều điều kiện cùng phục vụ một kết luận.
- Nếu cần tính tổng nhiều khoản thuế, tách từng khoản; Agent 5 mới cộng.
- Không quá 3 task; không bỏ sót đầu ra và không tạo task thừa.
- Giữ nguyên thuật ngữ, chủ thể, số tiền, tỷ lệ và đơn vị. Không tính toán hay dự đoán đáp án.
- Không tự thêm ngưỡng, miễn trừ, thủ tục hoặc chế định mà người dùng không hỏi.

VÍ DỤ
1. "Hiện nay mức giảm trừ gia cảnh là bao nhiêu?"
   -> một task, temporal current.
2. "Trong năm 2013 mức giảm trừ thay đổi thế nào?"
   -> một task, temporal year 2013.
3. "Từ tháng 8/2023 đến hiện tại quy định thay đổi thế nào?"
   -> một task, date_range từ month 2023-08 đến current.
4. "So sánh mức thuế ngày 31/12/2014 và 01/01/2015."
   -> một task comparison với hai exact_date.
5. "Khoản thu nhập năm 2023 chưa quyết toán; hiện nay phải xử lý và có bị phạt không?"
   -> task về quy định áp dụng cho khoản thu nhập dùng year 2023 nếu câu hỏi thực sự hỏi luật của năm đó;
      task về cách xử lý hoặc chế tài hiện nay dùng current.
6. "Tôi hiện diện 190 ngày trong năm, hiện nay có là cá nhân cư trú không?"
   -> current; 190 ngày là dữ kiện, không phải temporal.
7. "Thuế kỳ tháng 6/2026 được gia hạn đến ngày nào?"
   -> current nếu hỏi quy định hiện hành xử lý kỳ thuế; tháng 6/2026 được giữ trong task/query.
8. "Quy định áp dụng cho đề tài trong giai đoạn 2026-2027 là gì?"
   -> date_range từ year 2026 đến year 2027.

TỰ KIỂM TRA
- Mỗi task đã có temporal đúng với chính đầu ra đó chưa?
- Có nhầm ngày của tình huống thành thời điểm tra cứu pháp luật không?
- Có task nào cần current trong khi task khác cần lịch sử hoặc khoảng thời gian không?
- Task/query có giữ nguyên mọi neo pháp lý và số liệu cần thiết không?

Chỉ gọi submit_task_plan đúng một lần. Không trả văn bản ngoài JSON."""


TASK_2A_SCHEMA = {
    "type": "object",
    "required": ["task", "query", "temporal"],
    "properties": {
        "task": {"type": "string"},
        "query": {"type": "string"},
        "temporal": {
            "type": "object",
            "required": ["type"],
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
                "date": {"type": ["string", "null"]},
                "month": {"type": ["integer", "null"]},
                "year": {"type": ["integer", "null"]},
                "from": {
                    "type": "object",
                    "required": ["granularity", "value"],
                    "properties": {
                        "granularity": {
                            "type": "string",
                            "enum": ["year", "month", "day"],
                        },
                        "value": {
                            "type": "string",
                            "description": "Use YYYY, YYYY-MM, or YYYY-MM-DD.",
                        },
                    },
                },
                "to": {
                    "type": "object",
                    "required": ["granularity", "value"],
                    "properties": {
                        "granularity": {
                            "type": "string",
                            "enum": ["year", "month", "day", "current"],
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "Use YYYY, YYYY-MM, YYYY-MM-DD, or current."
                            ),
                        },
                    },
                },
                "comparison_scopes": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["type", "label"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "exact_date",
                                    "month",
                                    "year",
                                    "date_range",
                                ],
                            },
                            "date": {"type": "string"},
                            "month": {"type": "integer"},
                            "year": {"type": "integer"},
                            "from": {
                                "type": "object",
                                "required": ["granularity", "value"],
                                "properties": {
                                    "granularity": {
                                        "type": "string",
                                        "enum": ["year", "month", "day"],
                                    },
                                    "value": {"type": "string"},
                                },
                            },
                            "to": {
                                "type": "object",
                                "required": ["granularity", "value"],
                                "properties": {
                                    "granularity": {
                                        "type": "string",
                                        "enum": [
                                            "year",
                                            "month",
                                            "day",
                                            "current",
                                        ],
                                    },
                                    "value": {"type": "string"},
                                },
                            },
                            "label": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

TASKS_2A_PARAMETER = {
    "type": "array",
    "description": "Danh sách 1-3 nhiệm vụ tra cứu độc lập.",
    "minItems": 1,
    "maxItems": MAX_TASKS,
    "items": TASK_2A_SCHEMA,
}

FACT_2B_SCHEMA = {
    "type": "object",
    "required": ["type", "description"],
    "properties": {
        "type": {"type": "string", "enum": FACT_TYPES},
        "description": {"type": "string"},
    },
}

DATE_RANGE_START_SCHEMA = {
    "type": "object",
    "required": ["granularity", "value"],
    "properties": {
        "granularity": {"type": "string", "enum": ["year", "month", "day"]},
        "value": {
            "type": "string",
            "description": "Keep source precision: YYYY, YYYY-MM, or YYYY-MM-DD.",
        },
    },
}

DATE_RANGE_END_SCHEMA = {
    "type": "object",
    "required": ["granularity", "value"],
    "properties": {
        "granularity": {
            "type": "string",
            "enum": ["year", "month", "day", "current"],
        },
        "value": {
            "type": "string",
            "description": "YYYY, YYYY-MM, YYYY-MM-DD, or current.",
        },
    },
}

COMPARISON_SCOPE_SCHEMA = {
    "type": "object",
    "required": ["type", "label"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["exact_date", "month", "year", "date_range"],
        },
        "date": {"type": ["string", "null"]},
        "month": {"type": ["integer", "null"]},
        "year": {"type": ["integer", "null"]},
        "from": DATE_RANGE_START_SCHEMA,
        "to": DATE_RANGE_END_SCHEMA,
        "label": {"type": "string"},
    },
}

def _task_parameters(
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {"tasks": TASKS_2A_PARAMETER}
    properties.update(extra_properties or {})
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def _function_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class OllamaFunctionChoicePlannerLLM:
    """Run Qwen Agent 2A, then one sequential Qwen Agent 2B batch."""

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str = "",
        model: str = "qwen3.5:27b-q4_K_M",
        num_ctx: int = 16384,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 600,
        keep_alive: str = "30m",
        fact_max_output_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.model = model
        self.num_ctx = num_ctx
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.fact_max_output_tokens = max(256, fact_max_output_tokens)
        self.endpoint = f"{self.base_url}/api/chat"

        self.stage_a_tools = [
            _function_tool(
                "submit_task_plan",
                (
                    "Return one to three legal retrieval tasks. Each task must "
                    "contain its own temporal scope."
                ),
                _task_parameters(),
            )
        ]
        self.last_stage_a: dict[str, Any] = {}
        self.last_stage_b: dict[str, dict[str, Any]] = {}

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _call_tool(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
        expected_names: set[str],
        output_tokens: int,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        response = requests.post(
            self.endpoint,
            headers=self._headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_predict": output_tokens,
                    "temperature": 0,
                },
            },
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            try:
                detail: Any = response.json()
            except ValueError:
                detail = response.text[:1000]
            raise RuntimeError(
                f"Ollama proxy HTTP {response.status_code}: {detail}"
            )

        payload = response.json()
        message = payload.get("message") or {}
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            content = str(message.get("content") or "")[:1000]
            raise RuntimeError(
                "Agent 2 must return exactly one function call; "
                f"received {len(calls)}. Content: {content!r}"
            )

        function = calls[0].get("function") or {}
        name = function.get("name")
        if name not in expected_names:
            raise RuntimeError(f"Unsupported Agent 2 function: {name!r}")

        raw_arguments = function.get("arguments") or {}
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Ollama returned invalid tool arguments: "
                    f"{raw_arguments[:1000]}"
                ) from exc
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise RuntimeError("Ollama returned unsupported tool arguments")

        return name, arguments, payload

    @staticmethod
    def _coerce_range_boundary(value: Any, *, side: str) -> Any:
        """Normalize only unambiguous alternate boundary representations."""

        if not isinstance(value, dict):
            return value

        boundary = dict(value)
        granularity = str(boundary.get("granularity") or "").strip().lower()
        raw_value = str(boundary.get("value") or "").strip().lower()

        if not raw_value:
            aliases = (
                ("date", "day"),
                ("month", "month"),
                ("year", "year"),
            )
            for key, inferred_granularity in aliases:
                alias_value = boundary.get(key)
                if alias_value not in (None, ""):
                    raw_value = str(alias_value).strip().lower()
                    if not granularity:
                        granularity = inferred_granularity
                    break

        if not granularity and raw_value:
            if raw_value == "current" and side == "to":
                granularity = "current"
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_value):
                granularity = "day"
            elif re.fullmatch(r"\d{4}-\d{2}", raw_value):
                granularity = "month"
            elif re.fullmatch(r"\d{4}", raw_value):
                granularity = "year"

        if granularity == "current" and side == "to":
            raw_value = "current"

        return {"granularity": granularity, "value": raw_value}

    @classmethod
    def _coerce_temporal_payload(
        cls,
        raw_temporal: dict[str, Any],
    ) -> dict[str, Any]:
        temporal = dict(raw_temporal)
        temporal_type = str(temporal.get("type") or "").strip().lower()
        alias_value = str(temporal.get("value") or "").strip()

        # Ollama occasionally puts a scalar temporal value in `value` even
        # though the tool schema names the destination field explicitly.
        # Accept only lossless ISO-shaped aliases; never infer a new date.
        if temporal_type == "year" and temporal.get("year") in (None, ""):
            if re.fullmatch(r"\d{4}", alias_value):
                temporal["year"] = int(alias_value)
        elif temporal_type == "month" and (
            temporal.get("month") in (None, "")
            or temporal.get("year") in (None, "")
        ):
            month_match = re.fullmatch(r"(\d{4})-(\d{1,2})", alias_value)
            if month_match:
                temporal["year"] = int(month_match.group(1))
                temporal["month"] = int(month_match.group(2))
        elif temporal_type == "exact_date" and not temporal.get("date"):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", alias_value):
                temporal["date"] = alias_value

        if temporal_type == "date_range":
            temporal["from"] = cls._coerce_range_boundary(
                temporal.get("from"), side="from"
            )
            temporal["to"] = cls._coerce_range_boundary(
                temporal.get("to"), side="to"
            )
        elif temporal_type == "comparison":
            scopes = temporal.get("comparison_scopes")
            if isinstance(scopes, list):
                temporal["comparison_scopes"] = [
                    cls._coerce_comparison_child(scope)
                    if isinstance(scope, dict)
                    else scope
                    for scope in scopes
                ]
        return temporal

    @classmethod
    def _coerce_comparison_child(
        cls,
        raw_scope: dict[str, Any],
    ) -> dict[str, Any]:
        """Recover an explicit scope when a tool call kept only its label."""

        scope = dict(raw_scope)
        temporal_type = str(scope.get("type") or "").strip().lower()
        label = str(scope.get("label") or "").strip()

        needs_value = (
            not temporal_type
            or temporal_type == "current" and bool(label)
            or temporal_type == "exact_date" and not scope.get("date")
            or temporal_type == "month"
            and (scope.get("month") in (None, "") or scope.get("year") in (None, ""))
            or temporal_type == "year" and scope.get("year") in (None, "")
        )
        if needs_value and label:
            numeric_date = re.search(
                r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)",
                label,
            )
            word_date = re.search(
                r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                label,
                flags=re.IGNORECASE,
            )
            date_match = numeric_date or word_date
            if date_match:
                day, month, year = map(int, date_match.groups())
                scope.update(
                    {
                        "type": "exact_date",
                        "date": f"{year:04d}-{month:02d}-{day:02d}",
                    }
                )
            else:
                month_match = re.search(
                    r"tháng\s+(\d{1,2})(?:\s+năm\s+|\s*/\s*)(\d{4})",
                    label,
                    flags=re.IGNORECASE,
                )
                if month_match:
                    month, year = map(int, month_match.groups())
                    scope.update({"type": "month", "month": month, "year": year})
                else:
                    year_match = re.fullmatch(
                        r"\s*(?:năm\s+)?(\d{4})\s*",
                        label,
                        flags=re.IGNORECASE,
                    )
                    if year_match:
                        scope.update(
                            {"type": "year", "year": int(year_match.group(1))}
                        )

        if not str(scope.get("type") or "").strip():
            raise ValueError(
                "comparison child is missing type and its label is not an "
                f"unambiguous date: {label!r}"
            )
        return cls._coerce_temporal_payload(scope)

    @staticmethod
    def _preserve_explicit_month_precision(
        raw_temporal: dict[str, Any],
        original_query: str,
    ) -> dict[str, Any]:
        """Undo a model-generated month-end date when the source says only month."""

        temporal = dict(raw_temporal)
        if str(temporal.get("type") or "").strip().lower() != "exact_date":
            return temporal
        target = str(temporal.get("date") or "").strip()
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", target)
        if not match:
            return temporal
        year, month, _ = map(int, match.groups())
        explicit_months = {
            (int(found_year), int(found_month))
            for found_month, found_year in re.findall(
                r"tháng\s+(\d{1,2})(?:\s+năm\s+|\s*/\s*)(\d{4})",
                original_query,
                flags=re.IGNORECASE,
            )
        }
        explicit_days = {
            (int(found_year), int(found_month))
            for _, found_month, found_year in re.findall(
                r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                original_query,
                flags=re.IGNORECASE,
            )
        }
        if (year, month) in explicit_months and (year, month) not in explicit_days:
            return {"type": "month", "month": month, "year": year}
        return temporal

    @classmethod
    def _normalize_tasks(
        cls,
        tasks: Any,
        *,
        original_query: str = "",
    ) -> list[dict[str, Any]]:
        if not isinstance(tasks, list) or not tasks:
            raise RuntimeError("Agent 2A returned no tasks")
        if len(tasks) > MAX_TASKS:
            raise RuntimeError(
                f"Agent 2A returned {len(tasks)} tasks; maximum is {MAX_TASKS}"
            )

        normalized = []
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                raise RuntimeError("Agent 2A returned an invalid task")
            task_text = str(task.get("task") or "").strip()
            query = str(task.get("query") or "").strip()
            if not task_text or not query:
                raise RuntimeError("Agent 2A task is missing task or query")
            raw_temporal = task.get("temporal")
            if not isinstance(raw_temporal, dict):
                raise RuntimeError(
                    f"Agent 2A task {index} is missing its temporal scope"
                )
            source_temporal = cls._preserve_explicit_month_precision(
                raw_temporal,
                original_query,
            )
            coerced_temporal = cls._coerce_temporal_payload(source_temporal)
            try:
                temporal = normalize_temporal_scope(coerced_temporal)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Agent 2A task {index} has invalid temporal: {exc}; "
                    "raw_temporal="
                    + json.dumps(raw_temporal, ensure_ascii=False)
                ) from exc
            normalized.append(
                {
                    "task_id": f"T{index}",
                    "task": task_text,
                    "query": query,
                    "temporal": temporal,
                }
            )
        return normalized

    def _facts_for_all_tasks(
        self,
        *,
        original_query: str,
        tasks: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
        task_ids = [task["task_id"] for task in tasks]
        declaration = {
            "name": "submit_task_facts",
            "description": (
                "Return facts grouped by task_id for every locked task."
            ),
            "parameters": {
                "type": "object",
                "required": ["task_facts"],
                "properties": {
                    "task_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["task_id", "facts"],
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                },
                                "facts": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": MAX_FACTS_PER_TASK,
                                    "items": FACT_2B_SCHEMA,
                                },
                            },
                        },
                    }
                },
            },
        }
        prompt = (
            f"{AGENT2_BATCH_FACT_PROMPT}\n\n"
            "INPUT JSON:\n"
            + json.dumps(
                {
                    "original_query": original_query,
                    "tasks": tasks,
                },
                ensure_ascii=False,
            )
        )
        _, arguments, payload = self._call_tool(
            prompt=prompt,
            tools=[
                _function_tool(
                    declaration["name"],
                    declaration["description"],
                    declaration["parameters"],
                )
            ],
            expected_names={"submit_task_facts"},
            output_tokens=self.fact_max_output_tokens,
        )
        raw_groups = arguments.get("task_facts")
        if not isinstance(raw_groups, list):
            raise RuntimeError("Agent 2B batch returned no task_facts")

        groups_by_id: dict[str, Any] = {}
        for group in raw_groups:
            if not isinstance(group, dict):
                raise RuntimeError("Agent 2B batch returned an invalid group")
            task_id = str(group.get("task_id") or "").strip()
            if task_id in groups_by_id:
                raise RuntimeError(
                    f"Agent 2B batch duplicated task_id {task_id!r}"
                )
            groups_by_id[task_id] = group.get("facts")

        expected_ids = set(task_ids)
        actual_ids = set(groups_by_id)
        if actual_ids != expected_ids:
            raise RuntimeError(
                "Agent 2B batch task IDs do not match Agent 2A: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"unexpected={sorted(actual_ids - expected_ids)}"
            )

        facts_by_task: dict[str, list[dict[str, str]]] = {}
        for task_id in task_ids:
            raw_facts = groups_by_id[task_id]
            if not isinstance(raw_facts, list) or not raw_facts:
                raise RuntimeError(
                    f"Agent 2B batch returned no facts for {task_id}"
                )
            facts: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for raw_fact in raw_facts:
                if not isinstance(raw_fact, dict):
                    continue
                fact_type = str(raw_fact.get("type") or "").strip()
                description = str(
                    raw_fact.get("description") or ""
                ).strip()
                if fact_type not in FACT_TYPES or not description:
                    continue
                dedupe_key = (
                    fact_type,
                    re.sub(r"\s+", " ", description).strip().casefold(),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
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
                    f"Agent 2B batch returned no valid facts for {task_id}"
                )
            facts_by_task[task_id] = facts
        return facts_by_task, payload

    def _generate_facts(
        self,
        *,
        original_query: str,
        tasks: list[dict[str, Any]],
    ) -> tuple[
        dict[str, list[dict[str, str]]],
        dict[str, dict[str, Any]],
    ]:
        facts_by_task, payload = self._facts_for_all_tasks(
            original_query=original_query,
            tasks=tasks,
        )
        model = (
            payload.get("model")
            or self.model
        )
        debug = {
            task["task_id"]: {
                "facts": facts_by_task[task["task_id"]],
                "model": model,
                "backend": "ollama_qwen",
                "batch": True,
            }
            for task in tasks
        }
        return facts_by_task, debug

    def complete_locked_plan(
        self,
        original_query: str,
        locked_plan: list[dict[str, Any]],
    ) -> str:
        """Run only Agent 2B while preserving tasks and temporal from a baseline."""
        if not locked_plan:
            raise ValueError("Locked Agent 2A plan must not be empty")

        tasks = self._normalize_tasks(
            [
                {
                    "task_id": step.get("task_id"),
                    "task": step.get("task"),
                    "query": step.get("query"),
                    "temporal": step.get("temporal"),
                }
                for step in locked_plan
            ],
            original_query=original_query,
        )

        facts_by_task, stage_b_debug = self._generate_facts(
            original_query=original_query,
            tasks=tasks,
        )
        self.last_stage_b = stage_b_debug

        plan = [
            {
                **task,
                "fact_requirements": facts_by_task[task["task_id"]],
                "_selected_temporal_function": "locked_from_baseline",
                "_ollama_model": self.model,
                "_agent2_architecture": (
                    "2A_frozen_task_temporal__2B_batch_facts"
                ),
            }
            for task in tasks
        ]
        return json.dumps(plan, ensure_ascii=False)

    def complete_query(self, original_query: str) -> str:
        """Create tasks/temporal, then enrich every task with its own facts."""
        stage_a_prompt = (
            f"{AGENT2_FUNCTION_CHOICE_PROMPT}\n\n"
            f"Câu hỏi: {original_query}"
        )
        function_name, arguments, stage_a_payload = self._call_tool(
            prompt=stage_a_prompt,
            tools=self.stage_a_tools,
            expected_names={"submit_task_plan"},
            output_tokens=self.max_output_tokens,
        )
        self.last_stage_a = {
            "function": function_name,
            "arguments": arguments,
            "model": stage_a_payload.get("model") or self.model,
        }
        tasks = self._normalize_tasks(
            arguments.get("tasks"),
            original_query=original_query,
        )

        facts_by_task, stage_b_debug = self._generate_facts(
            original_query=original_query,
            tasks=tasks,
        )
        self.last_stage_b = stage_b_debug

        plan = []
        for task in tasks:
            plan.append(
                {
                    **task,
                    "fact_requirements": facts_by_task[task["task_id"]],
                    "_selected_temporal_function": function_name,
                    "_ollama_model": stage_a_payload.get("model")
                    or self.model,
                    "_agent2_architecture": (
                        "2A_per_task_temporal__2B_batch_facts"
                    ),
                }
            )
        return json.dumps(plan, ensure_ascii=False)

    def complete(self, prompt: str) -> str:
        """Backward-compatible adapter for callers that still pass a full prompt."""
        marker = "Câu hỏi:"
        original_query = prompt.rsplit(marker, 1)[-1]
        original_query = original_query.rsplit("Output:", 1)[0].strip()
        return self.complete_query(original_query)
