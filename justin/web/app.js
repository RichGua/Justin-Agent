const state = {
  currentSessionId: null,
  lastTrace: [],
};

const chatEl = document.getElementById("chat");
const sessionsEl = document.getElementById("sessions");
const candidatesEl = document.getElementById("candidates");
const memoriesEl = document.getElementById("memories");
const traceEl = document.getElementById("trace");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const newSessionBtn = document.getElementById("new-session-btn");
const memorySearchBtn = document.getElementById("memory-search-btn");
const memorySearchInput = document.getElementById("memory-search");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(payload.error || response.statusText);
  }
  return response.json();
}

function renderMessage(message) {
  const card = document.createElement("article");
  card.className = `message-card ${message.role}`;
  card.innerHTML = `
    <div class="card-label">${message.role}</div>
    <div>${escapeHtml(message.content).replace(/\n/g, "<br />")}</div>
  `;
  return card;
}

function renderSession(session) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = `session-card ${state.currentSessionId === session.id ? "active" : ""}`;
  card.innerHTML = `
    <p class="session-title">${escapeHtml(session.title)}</p>
    <div class="muted">${escapeHtml(session.updated_at)}</div>
  `;
  card.addEventListener("click", () => loadSession(session.id));
  return card;
}

function renderCandidate(candidate) {
  const card = document.createElement("article");
  card.className = "candidate-card";
  card.innerHTML = `
    <div class="card-label">${escapeHtml(candidate.kind)}</div>
    <div>${escapeHtml(candidate.content)}</div>
    <div class="muted">${escapeHtml(candidate.evidence)}</div>
    <span class="pill">confidence ${candidate.confidence.toFixed(2)}</span>
    <div class="candidate-actions">
      <button type="button" class="primary-button" data-action="confirm">Approve</button>
      <button type="button" class="ghost-button" data-action="reject">Reject</button>
    </div>
  `;
  card.querySelector('[data-action="confirm"]').addEventListener("click", () => approveCandidate(candidate.id));
  card.querySelector('[data-action="reject"]').addEventListener("click", () => rejectCandidate(candidate.id));
  return card;
}

function renderMemory(memory) {
  const card = document.createElement("article");
  card.className = "memory-card";
  card.innerHTML = `
    <div class="card-label">${escapeHtml(memory.kind)}</div>
    <div>${escapeHtml(memory.content)}</div>
    <span class="pill">${escapeHtml(memory.summary)}</span>
  `;
  return card;
}

function renderTrace(items) {
  traceEl.innerHTML = "";
  if (!items.length) {
    traceEl.textContent = "本次没有检索到已确认长期记忆。";
    traceEl.classList.add("muted");
    return;
  }
  traceEl.classList.remove("muted");
  for (const item of items) {
    const node = document.createElement("article");
    node.className = "memory-card";
    node.innerHTML = `
      <div class="card-label">${escapeHtml(item.kind)} · score ${item.score.toFixed(2)}</div>
      <div>${escapeHtml(item.content)}</div>
    `;
    traceEl.appendChild(node);
  }
}

async function refreshState() {
  const payload = await api("/api/state");
  renderSessions(payload.sessions);
  renderCandidates(payload.candidates);
  renderMemories(payload.memories);
}

function renderSessions(items) {
  sessionsEl.innerHTML = "";
  items.forEach((item) => sessionsEl.appendChild(renderSession(item)));
}

function renderCandidates(items) {
  candidatesEl.innerHTML = "";
  if (!items.length) {
    candidatesEl.innerHTML = '<div class="muted">当前没有待审核候选记忆。</div>';
    return;
  }
  items.forEach((item) => candidatesEl.appendChild(renderCandidate(item)));
}

function renderMemories(items) {
  memoriesEl.innerHTML = "";
  if (!items.length) {
    memoriesEl.innerHTML = '<div class="muted">还没有已确认长期记忆。</div>';
    return;
  }
  items.forEach((item) => memoriesEl.appendChild(renderMemory(item)));
}

async function loadSession(sessionId) {
  state.currentSessionId = sessionId;
  const payload = await api(`/api/sessions/${sessionId}`);
  chatEl.innerHTML = "";
  payload.messages.forEach((message) => chatEl.appendChild(renderMessage(message)));
  await refreshState();
}

async function sendMessage(event) {
  event.preventDefault();
  const content = messageInput.value.trim();
  if (!content) {
    return;
  }

  const result = await api("/api/messages", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.currentSessionId,
      content,
    }),
  });

  state.currentSessionId = result.session.id;
  messageInput.value = "";
  chatEl.appendChild(renderMessage({ role: "user", content }));
  chatEl.appendChild(renderMessage(result.assistant_message));
  chatEl.scrollTop = chatEl.scrollHeight;

  renderTrace(result.recalled_memories);
  await refreshState();
}

async function approveCandidate(candidateId) {
  await api(`/api/candidates/${candidateId}/confirm`, { method: "POST", body: "{}" });
  await refreshState();
}

async function rejectCandidate(candidateId) {
  await api(`/api/candidates/${candidateId}/reject`, { method: "POST", body: JSON.stringify({ note: "Rejected from dashboard" }) });
  await refreshState();
}

async function createSession() {
  const created = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: "New session" }),
  });
  state.currentSessionId = created.id;
  chatEl.innerHTML = "";
  renderTrace([]);
  await refreshState();
}

async function searchMemories() {
  const query = memorySearchInput.value.trim();
  if (!query) {
    const payload = await api("/api/memories");
    renderMemories(payload.items);
    return;
  }
  const payload = await api(`/api/search?q=${encodeURIComponent(query)}`);
  renderMemories(payload.items);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

chatForm.addEventListener("submit", sendMessage);
newSessionBtn.addEventListener("click", createSession);
memorySearchBtn.addEventListener("click", searchMemories);

refreshState().catch((error) => {
  traceEl.textContent = `Failed to load dashboard: ${error.message}`;
});
