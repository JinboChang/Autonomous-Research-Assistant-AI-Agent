const chat = document.getElementById("chat");
const form = document.getElementById("form");
const questionInput = document.getElementById("question");
const statusEl = document.getElementById("status");

function addMessage(text, role = "bot") {
  const div = document.createElement("div");
  div.className = "msg";
  const label = document.createElement("div");
  label.className = role === "user" ? "user" : "bot";
  label.textContent = role === "user" ? "You" : "Assistant";
  const pre = document.createElement("pre");
  pre.textContent = text;
  div.appendChild(label);
  div.appendChild(pre);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";
  statusEl.textContent = "Running...";
  form.querySelector("button").disabled = true;

  try {
    const resp = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    addMessage(data.report || "No response");
  } catch (err) {
    addMessage(`Error: ${err.message}`, "bot");
  } finally {
    statusEl.textContent = "";
    form.querySelector("button").disabled = false;
  }
});
