const chat = document.querySelector("#chat");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const resetButton = document.querySelector("#resetButton");
const statusLabel = document.querySelector("#status");
const template = document.querySelector("#messageTemplate");

let sessionId = localStorage.getItem("legal-rag-session") || null;

function addMessage(role, text, trace = null) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".message-label").textContent = role === "user" ? "Bạn" : "Trợ lý";
  node.querySelector(".message-body").textContent = text;
  const details = node.querySelector(".trace");
  if (trace) {
    details.querySelector("pre").textContent = JSON.stringify(trace, null, 2);
  } else {
    details.remove();
  }
  chat.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function setBusy(busy) {
  input.disabled = busy;
  sendButton.disabled = busy;
  resetButton.disabled = busy;
  statusLabel.textContent = busy ? "Đang xử lý qua Agent 1–5" : "Sẵn sàng";
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    statusLabel.textContent = "Sẵn sàng";
  } catch {
    statusLabel.textContent = "Không kết nối được backend";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  setBusy(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Pipeline gặp lỗi.");

    sessionId = data.session_id;
    localStorage.setItem("legal-rag-session", sessionId);
    const trace = data.status === "completed" ? {
      normalized_query: data.normalized_query,
      plan: data.plan,
      evidence_count: data.evidence_count,
      failed_steps: data.failed_steps,
      duration_seconds: data.duration_seconds,
    } : {
      status: data.status,
      duration_seconds: data.duration_seconds,
    };
    addMessage("assistant", data.message || "Không có nội dung trả về.", trace);
  } catch (error) {
    addMessage("assistant", `Không thể hoàn thành yêu cầu: ${error.message}`);
  } finally {
    setBusy(false);
    input.focus();
  }
});

resetButton.addEventListener("click", async () => {
  if (sessionId) {
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => null);
  }
  sessionId = null;
  localStorage.removeItem("legal-rag-session");
  chat.innerHTML = "";
  addMessage("assistant", "Bạn cần tra cứu vấn đề thuế nào?");
  input.focus();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

checkHealth();

