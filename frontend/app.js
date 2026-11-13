/* RIO frontend: a minimal AG-UI client in plain JavaScript.
 *
 * Right panel : chat. Each message is sent as an AG-UI run (POST /agui) carrying the
 *               conversation and the current shared state.
 * Left panel  : rendered from the AG-UI shared state, which the server updates with
 *               STATE_SNAPSHOT / STATE_DELTA (JSON Patch) events while the agents work.
 *
 * Events handled: RUN_STARTED, RUN_FINISHED, RUN_ERROR, STEP_STARTED, STEP_FINISHED,
 * TEXT_MESSAGE_START / CONTENT / END, STATE_SNAPSHOT, STATE_DELTA.
 */
"use strict";

// ---------------------------------------------------------------- constants
const STAGES = {
  fundamental_analyst: { label: "Fundamental analyst", short: "FA", desc: "Valuation, profitability, growth, debt" },
  technical_analyst: { label: "Technical analyst", short: "TA", desc: "Trend, momentum, volatility, ML direction model" },
  sentiment_analyst: { label: "Sentiment analyst", short: "SA", desc: "News tone (VADER) and India VIX" },
  news_analyst: { label: "News analyst", short: "NA", desc: "Catalysts and risks in recent headlines" },
  trader: { label: "Trader", short: "TR", desc: "Weighs the reports into one proposal" },
  risk_manager: { label: "Risk manager", short: "RM", desc: "Volatility, VaR, drawdown, position cap" },
  finalize: { label: "Final decision", short: "FD", desc: "Proposal merged with the risk verdict (no LLM)" },
};
const ANALYSTS = ["fundamental_analyst", "technical_analyst", "sentiment_analyst", "news_analyst"];
const METRICS = [
  { key: "cumulative_return", label: "Cumulative return", fmt: (v) => pct(v), better: "higher" },
  { key: "sharpe_ratio", label: "Sharpe ratio", fmt: (v) => num(v, 2), better: "higher" },
  { key: "max_drawdown", label: "Max drawdown", fmt: (v) => pct(v), better: "closer-to-zero" },
  { key: "annualized_volatility", label: "Volatility (annual)", fmt: (v) => pct(v), better: "lower" },
];

// ---------------------------------------------------------------- app state
const app = {
  threadId: uuid(),
  messages: [],       // AG-UI messages sent with every run
  state: {},          // AG-UI shared state (drives the left panel)
  activeView: null,   // "analysis" | "backtest" (tab the user is looking at)
  stepTimes: {},      // stepName -> { start, end } from STEP_STARTED / STEP_FINISHED
  chats: [],          // saved chats for the Recent dropdown (from GET /api/chats)
  running: false,
  streamingId: null,  // assistant message currently being streamed
  controller: null,
};

const $ = (id) => document.getElementById(id);
const els = {
  board: $("board"),
  messages: $("messages"),
  suggestions: $("suggestions"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  runStatus: $("run-status"),
  marketChip: $("market-chip"),
  newChat: $("new-chat"),
  chatList: $("chat-list"),
  recentToggle: $("recent-toggle"),
  recentMenu: $("recent-menu"),
  theme: $("theme-toggle"),
};

// ---------------------------------------------------------------- helpers
function uuid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "id-" + Date.now().toString(36) + Math.random().toString(36).slice(2);
}
function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function pct(value, digits = 1) {
  return value == null || Number.isNaN(value) ? "—" : (value * 100).toFixed(digits) + "%";
}
function num(value, digits = 2) {
  return value == null || Number.isNaN(value) ? "—" : Number(value).toFixed(digits);
}
function seconds(ms) {
  return ms < 10000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms / 1000) + "s";
}
function tone(value) {
  return ({
    bullish: "pos", bearish: "neg", BUY: "pos", SELL: "neg", HOLD: "warn",
    APPROVE: "pos", ADJUST: "warn", REJECT: "neg", low: "pos", medium: "warn", high: "neg",
  })[value] || "";
}
function badge(text, extra = "") {
  return `<span class="badge ${tone(text)} ${extra}">${esc(text)}</span>`;
}
function runBadge(status) {
  return ({
    done: badge("Complete", "pos"), error: badge("Failed", "neg"), stopped: badge("Stopped"),
    interrupted: badge("Interrupted", "warn"),
  })[status] || badge("Running", "accent");
}

/** Tiny, safe Markdown: paragraphs, bullet / numbered lists, **bold**, *italic*, `code`. */
function markdown(source) {
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  let html = "", paragraph = [], list = null;
  const flushParagraph = () => { if (paragraph.length) { html += `<p>${inline(paragraph.join("<br>"))}</p>`; paragraph = []; } };
  const flushList = () => { if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`; list = null; } };
  for (const line of esc(source).split("\n")) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (bullet || ordered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((bullet || ordered)[1]);
    } else if (!line.trim()) {
      flushParagraph(); flushList();
    } else if (heading) {
      flushParagraph(); flushList();
      html += `<p><strong>${inline(heading[1])}</strong></p>`;
    } else {
      flushList(); paragraph.push(line);
    }
  }
  flushParagraph(); flushList();
  return html;
}

/** Applies RFC 6902 JSON Patch operations (add / replace / remove) in place. */
function applyPatch(doc, operations) {
  for (const op of operations) {
    const parts = op.path.split("/").slice(1).map((p) => p.replace(/~1/g, "/").replace(/~0/g, "~"));
    const last = parts.pop();
    let target = doc;
    for (const part of parts) {
      target = target[Array.isArray(target) ? Number(part) : part];
      if (target == null) throw new Error(`Invalid patch path ${op.path}`);
    }
    if (op.op === "add" || op.op === "replace") {
      if (Array.isArray(target)) {
        if (last === "-") target.push(op.value);
        else if (op.op === "add") target.splice(Number(last), 0, op.value);
        else target[Number(last)] = op.value;
      } else {
        target[last] = op.value;
      }
    } else if (op.op === "remove") {
      if (Array.isArray(target)) target.splice(Number(last), 1);
      else delete target[last];
    }
  }
}

// ---------------------------------------------------------------- AG-UI client
/** Sends one AG-UI run. `forwardedProps` carries commands such as {command: "resume", runId}. */
async function runAgent(text, forwardedProps = {}) {
  if (app.running || !text.trim()) return;
  const userMessage = { id: uuid(), role: "user", content: text.trim() };
  if (!app.messages.length) els.messages.innerHTML = ""; // drop the greeting in a new chat
  app.messages.push(userMessage);
  appendMessage(userMessage);
  setRunning(true);
  app.controller = new AbortController();
  history.replaceState(null, "", `#chat/${app.threadId}`);

  const body = {
    threadId: app.threadId,
    runId: uuid(),
    state: app.state,
    // Only the conversation goes to the agent; notices and errors are display-only.
    messages: app.messages.filter((m) => m.role === "user" || m.role === "assistant"),
    tools: [],
    context: [],
    forwardedProps,
  };
  try {
    const response = await fetch("/agui", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: app.controller.signal,
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}: ${await response.text()}`);
    await readEventStream(response.body, handleEvent);
  } catch (error) {
    if (error.name === "AbortError") {
      appendNotice("Stopped. The agents will not start any further work for this request.");
      markUnfinished("stopped");
    } else {
      appendError(error.message || String(error));
      markUnfinished("error");
    }
  } finally {
    app.controller = null;
    app.streamingId = null;
    setRunning(false);
    refreshChats();
  }
}

/** Reads a Server-Sent Events body and calls onEvent with each parsed AG-UI event. */
async function readEventStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).replace(/^ /, "")).join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "RUN_STARTED":
      setStatus("running", "Agents working");
      break;
    case "TEXT_MESSAGE_START": {
      const message = { id: event.messageId, role: "assistant", content: "" };
      app.messages.push(message);
      app.streamingId = message.id;
      appendMessage(message);
      break;
    }
    case "TEXT_MESSAGE_CONTENT": {
      const message = app.messages.find((m) => m.id === event.messageId);
      if (message) {
        message.content += event.delta;
        updateMessage(message);
      }
      break;
    }
    case "TEXT_MESSAGE_END":
      app.streamingId = null;
      break;
    case "STATE_SNAPSHOT":
      app.state = event.snapshot || {};
      app.activeView = app.state.view || app.activeView;
      renderBoard();
      break;
    case "STATE_DELTA":
      applyPatch(app.state, event.delta || []);
      renderBoard();
      break;
    case "STEP_STARTED":
      app.stepTimes[event.stepName] = { start: Date.now() };
      break;
    case "STEP_FINISHED":
      if (app.stepTimes[event.stepName]) app.stepTimes[event.stepName].end = Date.now();
      renderBoard();
      break;
    case "RUN_FINISHED":
      setStatus("idle", "Idle");
      break;
    case "RUN_ERROR":
      appendError(event.message);
      markUnfinished("error");
      setStatus("error", "Error");
      break;
  }
  updateTyping();
}

/** Marks whatever was still in progress as "error" (server failure) or "stopped" (user pressed Stop). */
function markUnfinished(status) {
  const analysis = app.state.analysis;
  if (analysis && analysis.status === "running") {
    analysis.status = status;
    for (const stage of Object.values(analysis.stages)) {
      if (stage.status === "running" || stage.status === "pending") stage.status = status;
    }
  }
  const backtest = app.state.backtest;
  if (backtest && backtest.status === "running") backtest.status = status;
  renderBoard();
}

// ---------------------------------------------------------------- chat UI
function appendMessage(message) {
  const el = document.createElement("div");
  el.className = `msg ${message.role}`;
  el.dataset.id = message.id;
  if (message.role === "assistant") el.innerHTML = markdown(message.content);
  else el.textContent = message.content; // user, error and notice messages are plain text
  els.messages.appendChild(el);
  scrollChat();
}
function updateMessage(message) {
  const el = els.messages.querySelector(`[data-id="${CSS.escape(message.id)}"]`);
  if (el) el.innerHTML = markdown(message.content);
  scrollChat();
}
function appendError(text) {
  appendNotice(text, "error");
}
function appendNotice(text, kind = "notice") {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  els.messages.appendChild(el);
  scrollChat();
}
function updateTyping() {
  const existing = els.messages.querySelector(".typing-msg");
  const show = app.running && !app.streamingId;
  if (show && !existing) {
    const el = document.createElement("div");
    el.className = "msg assistant typing-msg";
    el.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    els.messages.appendChild(el);
    scrollChat();
  } else if (!show && existing) {
    existing.remove();
  } else if (show && existing) {
    els.messages.appendChild(existing); // keep it last
  }
}
function scrollChat() {
  els.messages.scrollTop = els.messages.scrollHeight;
}
function greet() {
  els.messages.innerHTML = "";
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = markdown(
    "Hi! I run a team of AI agents on Indian stocks (NSE/BSE).\n\n" +
    "- **Analyze RELIANCE**: a decision for today\n" +
    "- **Analyze INFY as of 2025-06-30**: a decision on a past date\n" +
    "- **Backtest TCS from 2025-01-01 to 2025-06-30**: test the strategy\n\n" +
    "Once results are on the left, ask me about them."
  );
  els.messages.appendChild(el);
}
function renderSuggestions() {
  const items = [];
  const analysis = app.state.analysis;
  if (analysis && analysis.status === "done") {
    items.push("Why this decision?", "Where did the analysts disagree?", "What did the risk manager change?");
  } else if (app.state.backtest && app.state.backtest.status === "done") {
    items.push("How did the strategy compare to buy & hold?", "Analyze " + app.state.backtest.ticker.split(".")[0]);
  } else {
    items.push("Analyze RELIANCE", "Analyze INFY as of 2025-06-30", "Backtest TCS for H1 2025");
  }
  els.suggestions.innerHTML = items
    .map((text) => `<button type="button" class="suggestion" ${app.running ? "disabled" : ""}>${esc(text)}</button>`)
    .join("");
}
function setRunning(running) {
  app.running = running;
  els.send.classList.toggle("stop", running);
  els.send.setAttribute("aria-label", running ? "Stop" : "Send");
  els.newChat.disabled = running;
  if (!running && els.runStatus.dataset.state === "running") setStatus("idle", "Idle");
  updateTyping();
  renderSuggestions();
  renderChatList(); // chats cannot be switched while a run is streaming
  renderBoard();    // shows / hides the Resume button
}
function setStatus(state, label) {
  els.runStatus.dataset.state = state;
  els.runStatus.querySelector(".label").textContent = label;
}

// ---------------------------------------------------------------- board UI
function renderBoard() {
  const { analysis, backtest } = app.state;
  if (!analysis && !backtest) {
    els.board.innerHTML = emptyView();
    renderSuggestions();
    return;
  }
  let view = app.activeView || app.state.view || "analysis";
  if (view === "analysis" && !analysis) view = "backtest";
  if (view === "backtest" && !backtest) view = "analysis";
  const tabs = analysis && backtest
    ? `<div class="tabs" role="group">${["analysis", "backtest"].map((v) =>
        `<button type="button" data-view="${v}" aria-pressed="${v === view}">${v === "analysis" ? "Analysis" : "Backtest"}</button>`).join("")}</div>`
    : "";

  const scroll = els.board.scrollTop;
  els.board.innerHTML = view === "analysis" ? analysisView(analysis, tabs) : backtestView(backtest, tabs);
  els.board.scrollTop = scroll;
  if (view === "backtest" && backtest.curve) drawChart(els.board.querySelector(".chart"), backtest.curve);
  updateTimers();
  renderSuggestions();
}

function emptyView() {
  const box = (key) => `<div class="box">${STAGES[key].label}</div>`;
  return `
    <div class="empty">
      <h1>Your AI trading team, live</h1>
      <p>Ask in the chat. Four analysts study the stock in parallel, a trader proposes a trade, and a risk manager
         reviews it. Every stage's output appears here as soon as it is ready.</p>
      <div class="flow">
        <div class="stack">${ANALYSTS.map(box).join("")}</div>
        <span class="arrow">→</span>${box("trader")}<span class="arrow">→</span>${box("risk_manager")}
        <span class="arrow">→</span>${box("finalize")}
      </div>
      <div class="examples">
        ${["Analyze RELIANCE", "Analyze HDFCBANK", "Analyze INFY as of 2025-06-30", "Backtest TCS for H1 2025"]
          .map((t) => `<button type="button" class="suggestion">${esc(t)}</button>`).join("")}
      </div>
    </div>`;
}

function statusLabel(stepName, status) {
  const times = app.stepTimes[stepName];
  if (status === "running") return `Working${times ? ` <span class="mono" data-since="${times.start}"></span>` : ""}`;
  if (status === "done") return times && times.end ? `Done · ${seconds(times.end - times.start)}` : "Done";
  if (status === "error") return "Failed";
  if (status === "stopped") return "Stopped";
  if (status === "interrupted") return "Interrupted";
  return "Waiting";
}
function statusChip(stepName, status) {
  return `<span class="status" data-status="${status}"><span class="dot"></span>${statusLabel(stepName, status)}</span>`;
}
function updateTimers() {
  for (const el of els.board.querySelectorAll("[data-since]")) {
    el.textContent = seconds(Date.now() - Number(el.dataset.since));
  }
}
const skeleton = '<div class="skeleton"><span></span><span></span><span></span></div>';

function stageCard(name, stage, body) {
  const info = STAGES[name];
  let content;
  if (stage.status === "done" && stage.output) content = body(stage.output);
  else if (stage.status === "running") content = skeleton;
  else if (stage.status === "error") content = '<p class="placeholder">This stage did not finish. See the chat for the error.</p>';
  else if (stage.status === "stopped") content = '<p class="placeholder">Stopped before this stage finished.</p>';
  else if (stage.status === "interrupted") content = '<p class="placeholder">Interrupted by a server restart. Use Resume to continue.</p>';
  else content = '<p class="placeholder">Waiting for the previous stage.</p>';
  return `
    <article class="card" data-status="${stage.status}">
      <div class="card-head">
        <div class="card-title"><span class="avatar">${info.short}</span>
          <div><div>${info.label}</div><div class="card-meta">${info.desc}</div></div></div>
        ${statusChip(name, stage.status)}
      </div>
      ${content}
    </article>`;
}

function analystBody(report) {
  if (report.confidence === 0 && !report.key_points.length) {
    return `${badge("No data")}<p class="summary">${esc(report.summary)}</p>`;
  }
  return `
    <div style="display:flex;align-items:center;gap:10px">${badge(report.signal)}
      <div class="confidence" style="flex:1;margin:0">Confidence
        <div class="bar"><span style="width:${Math.round(report.confidence * 100)}%"></span></div>
        <span class="mono">${pct(report.confidence, 0)}</span></div></div>
    <p class="summary">${esc(report.summary)}</p>
    ${report.key_points.length ? `<ul class="points">${report.key_points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}`;
}

function traderBody(proposal) {
  return `
    <div style="display:flex;align-items:center;gap:10px">${badge(proposal.action, "lg")}
      <span class="mono" style="font-size:18px;font-weight:600">${pct(proposal.position_size_pct, 1)}</span>
      <span class="muted">of portfolio</span></div>
    <p class="summary">${esc(proposal.rationale)}</p>`;
}

function riskBody(assessment, metrics) {
  metrics = metrics || {};
  return `
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">${badge(assessment.decision, "lg")}
      <span class="muted">risk</span>${badge(assessment.risk_level)}</div>
    <dl class="kv">
      <div><dt>Approved size</dt><dd>${pct(assessment.adjusted_position_size_pct, 1)}</dd></div>
      <div><dt>Volatility (annual)</dt><dd>${pct(metrics.annualized_volatility)}</dd></div>
      <div><dt>Daily VaR 95%</dt><dd>${pct(metrics.daily_value_at_risk_95, 2)}</dd></div>
      <div><dt>Max drawdown</dt><dd>${pct(metrics.max_drawdown)}</dd></div>
    </dl>
    <p class="summary">${esc(assessment.comments)}</p>`;
}

function finalCard(analysis) {
  const stage = analysis.stages.finalize;
  const decision = stage.output;
  if (stage.status !== "done" || !decision) {
    return `
      <article class="card final" data-status="${stage.status}">
        <div class="card-head"><div class="card-title"><span class="avatar">FD</span>Final decision</div>${statusChip("finalize", stage.status)}</div>
        ${stage.status === "error" || stage.status === "stopped" ? '<p class="placeholder">The run did not reach a decision.</p>'
          : '<p class="placeholder">Appears after the risk review.</p>'}
      </article>`;
  }
  return `
    <article class="card final" data-status="done">
      <div class="card-head"><div class="card-title"><span class="avatar">FD</span>Final decision</div>
        <span class="card-meta">${esc(decision.ticker)} · ${esc(decision.as_of_date)}</span></div>
      <div class="final-main">${badge(decision.action, "lg")}
        <div class="final-size">${pct(decision.position_size_pct, 1)}<small>of portfolio</small></div>
        <span class="muted">risk</span>${badge(decision.risk_level)}</div>
      <h4>Rationale</h4><p>${esc(decision.rationale)}</p>
      <h4>Risk comments</h4><p>${esc(decision.risk_comments)}</p>
    </article>`;
}

function pipeline(analysis) {
  const stages = analysis.stages;
  const statuses = ANALYSTS.map((n) => stages[n] && stages[n].status);
  const doneCount = statuses.filter((s) => s === "done").length;
  const group = statuses.includes("error") ? "error"
    : statuses.includes("stopped") ? "stopped"
    : statuses.includes("interrupted") ? "interrupted"
    : statuses.includes("running") ? "running"
    : doneCount === ANALYSTS.length ? "done" : "pending";
  const steps = [
    { status: group, label: `Analysts ${doneCount}/${ANALYSTS.length}` },
    { status: stages.trader.status, label: "Trader" },
    { status: stages.risk_manager.status, label: "Risk review" },
    { status: stages.finalize.status, label: "Decision" },
  ];
  return `<div class="pipeline">${steps.map((s, i) => `
    ${i ? `<span class="pipe-line" data-done="${steps[i - 1].status === "done"}"></span>` : ""}
    <div class="pipe-step" data-status="${s.status}"><span class="node">${
      { done: "✓", error: "!", stopped: "■", interrupted: "■" }[s.status] || i + 1}</span>${s.label}</div>`).join("")}
  </div>`;
}

function analysisView(analysis, tabs) {
  const stages = analysis.stages;
  const runStatus = runBadge(analysis.status);
  // A stopped / interrupted / failed analysis continues from its last LangGraph checkpoint.
  const canResume = analysis.runId && ["stopped", "interrupted", "error"].includes(analysis.status) && !app.running;
  const resume = canResume
    ? `<button type="button" class="resume-btn" data-run-id="${esc(analysis.runId)}" data-ticker="${esc(analysis.ticker)}">
         Resume from checkpoint</button>`
    : "";
  return `
    <div class="board-head">
      <div class="board-title"><h1>${esc(analysis.ticker)}</h1><span class="sub">as of ${esc(analysis.asOfDate)}</span>${runStatus}${resume}</div>
      ${tabs}
    </div>
    ${pipeline(analysis)}
    <div class="section-label">Analysts <span class="count">· run in parallel</span></div>
    <div class="grid-2">${ANALYSTS.filter((n) => stages[n]).map((n) => stageCard(n, stages[n], analystBody)).join("")}</div>
    <div class="section-label">Trade proposal and risk review</div>
    <div class="grid-2">
      ${stageCard("trader", stages.trader, traderBody)}
      ${stageCard("risk_manager", stages.risk_manager, (a) => riskBody(a, analysis.riskMetrics))}
    </div>
    <div class="section-label">Outcome</div>
    ${finalCard(analysis)}`;
}

function backtestView(bt, tabs) {
  const done = bt.decisions.length;
  const total = bt.total || 0;
  const runStatus = runBadge(bt.status);
  const times = app.stepTimes.backtest;
  const elapsed = bt.status === "running" && times ? ` · <span class="mono" data-since="${times.start}"></span>`
    : times && times.end ? ` · ${seconds(times.end - times.start)}` : "";

  const metrics = bt.strategyMetrics ? `
    <div class="section-label">Performance</div>
    <div class="grid-2">
      <article class="card"><table class="metrics-table">
        <thead><tr><th>Metric</th><th>RIO strategy</th><th>Buy &amp; hold</th></tr></thead>
        <tbody>${METRICS.map((m) => {
          const s = bt.strategyMetrics[m.key], b = bt.buyAndHoldMetrics[m.key];
          const sBetter = m.better === "higher" ? s > b : m.better === "lower" ? s < b : Math.abs(s) < Math.abs(b);
          return `<tr><td>${m.label}</td><td class="${sBetter ? "better" : ""}">${m.fmt(s)}</td><td class="${!sBetter && s !== b ? "better" : ""}">${m.fmt(b)}</td></tr>`;
        }).join("")}</tbody></table></article>
      <article class="card">
        <div class="legend"><span><i style="background:var(--accent)"></i>RIO strategy</span><span><i style="background:var(--warn)"></i>Buy &amp; hold</span></div>
        <div class="chart"></div>
      </article>
    </div>` : "";

  return `
    <div class="board-head">
      <div class="board-title"><h1>${esc(bt.ticker)}</h1>
        <span class="sub">${esc(bt.startDate)} → ${esc(bt.endDate)} · decision every ${bt.rebalanceDays} trading days</span>${runStatus}</div>
      ${tabs}
    </div>
    <article class="card">
      <div class="card-head"><div class="card-title"><span class="avatar">BT</span>Walk-forward backtest</div>
        <span class="card-meta">${done}${total ? ` / ${total}` : ""} decisions${elapsed}</span></div>
      <div class="progress"><div class="bar"><span style="width:${total ? Math.round((done / total) * 100) : 0}%"></span></div></div>
      <p class="summary" style="font-size:12px">Each decision is a full agent run using only data up to that day.
        Fundamentals and news are switched off to avoid look-ahead.</p>
    </article>
    ${metrics}
    <div class="section-label">Decisions <span class="count">· ${done}</span></div>
    <article class="card">
      ${done ? `<div class="table-wrap"><table class="decisions-table">
        <thead><tr><th>Date</th><th>Action</th><th>Stock weight</th><th>Risk</th></tr></thead>
        <tbody>${bt.decisions.map((d) => `<tr><td class="mono">${esc(d.date)}</td><td>${badge(d.action)}</td>
          <td class="mono">${pct(d.weight, 0)}</td><td>${badge(d.risk_level)}</td></tr>`).join("")}</tbody></table></div>`
        : bt.status === "running" ? skeleton : '<p class="placeholder">No decisions.</p>'}
    </article>`;
}

/** Equity curves (growth of 1 unit) as an inline SVG with a hover read-out. */
function drawChart(container, curve) {
  if (!container || !curve.dates.length) return;
  const W = 640, H = 280, pad = { l: 60, r: 14, t: 14, b: 34 };
  const all = [...curve.strategy, ...curve.buyAndHold, 1];
  let min = Math.min(...all), max = Math.max(...all);
  const span = max - min || 0.01;
  min -= span * 0.05; max += span * 0.05;
  const n = curve.dates.length;
  const x = (i) => pad.l + (i / Math.max(n - 1, 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - min) / (max - min)) * (H - pad.t - pad.b);
  const path = (values) => values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const ticks = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);
  const xTicks = [0, Math.floor((n - 1) / 2), n - 1];
  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Equity curves of the RIO strategy and buy and hold">
      ${ticks.map((t) => `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}"/>
        <text class="axis" x="${pad.l - 8}" y="${y(t) + 4}" text-anchor="end">${pct(t - 1, 0)}</text>`).join("")}
      <line class="baseline" x1="${pad.l}" x2="${W - pad.r}" y1="${y(1)}" y2="${y(1)}"/>
      ${xTicks.map((i, k) => `<text class="axis" x="${x(i)}" y="${H - 8}" text-anchor="${["start", "middle", "end"][k]}">${curve.dates[i]}</text>`).join("")}
      <polyline class="line-bh" fill="none" stroke-width="2" points="${path(curve.buyAndHold)}"/>
      <polyline class="line-strategy" fill="none" stroke-width="2.25" points="${path(curve.strategy)}"/>
      <line class="hover-line" y1="${pad.t}" y2="${H - pad.b}" visibility="hidden"/>
      <rect x="${pad.l}" y="${pad.t}" width="${W - pad.l - pad.r}" height="${H - pad.t - pad.b}" fill="transparent"/>
    </svg>
    <div class="tooltip" hidden></div>`;

  const svg = container.querySelector("svg");
  const hoverLine = svg.querySelector(".hover-line");
  const tooltip = container.querySelector(".tooltip");
  svg.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((sx - pad.l) / (W - pad.l - pad.r)) * (n - 1))));
    hoverLine.setAttribute("x1", x(i)); hoverLine.setAttribute("x2", x(i));
    hoverLine.setAttribute("visibility", "visible");
    tooltip.hidden = false;
    tooltip.style.left = `${(x(i) / W) * rect.width}px`;
    tooltip.style.top = `${(y(Math.max(curve.strategy[i], curve.buyAndHold[i])) / H) * rect.height}px`;
    tooltip.innerHTML = `<div class="mono">${curve.dates[i]}</div>
      <div><span style="color:var(--accent)">RIO</span> <span class="mono">${pct(curve.strategy[i] - 1)}</span>
      · <span style="color:var(--warn)">B&amp;H</span> <span class="mono">${pct(curve.buyAndHold[i] - 1)}</span></div>`;
  });
  svg.addEventListener("mouseleave", () => { hoverLine.setAttribute("visibility", "hidden"); tooltip.hidden = true; });
}

// ---------------------------------------------------------------- wiring
els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (app.running) { app.controller && app.controller.abort(); return; }
  const text = els.input.value;
  els.input.value = "";
  autoGrow();
  runAgent(text);
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.composer.requestSubmit(); }
});
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 140) + "px";
}
els.input.addEventListener("input", autoGrow);

document.addEventListener("click", (e) => {
  const suggestion = e.target.closest(".suggestion");
  if (suggestion && !app.running) { runAgent(suggestion.textContent); return; }
  const tab = e.target.closest(".tabs button");
  if (tab) { app.activeView = tab.dataset.view; renderBoard(); return; }
  const resume = e.target.closest(".resume-btn");
  if (resume && !app.running) {
    runAgent(`Resume the analysis of ${resume.dataset.ticker}`, { command: "resume", runId: resume.dataset.runId });
  }
});

// ---------------------------------------------------------------- chat history (Recent dropdown)
function relativeTime(iso) {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.round(minutes / 60)} h ago`;
  return new Date(iso).toLocaleDateString();
}

async function refreshChats() {
  try {
    const response = await fetch("/api/chats");
    if (!response.ok) throw new Error(response.status);
    app.chats = await response.json();
  } catch (_) {
    app.chats = app.chats || [];
  }
  renderChatList();
}

function renderChatList() {
  const chats = app.chats || [];
  els.chatList.innerHTML = chats.length
    ? chats.map((chat) => `
        <li class="chat-item ${chat.id === app.threadId ? "active" : ""}" data-chat-id="${esc(chat.id)}">
          <button type="button" class="chat-open" ${app.running ? "disabled" : ""} title="${esc(chat.title)}">
            <span class="chat-title">${esc(chat.title)}</span>
            <span class="chat-time">${relativeTime(chat.updated_at)}</span>
          </button>
          <button type="button" class="chat-delete" ${app.running ? "disabled" : ""} aria-label="Delete chat" title="Delete chat">
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M9 3h6l1 2h4v2H4V5h4l1-2Zm-3 6h12l-1 12H7L6 9Z"/></svg>
          </button>
        </li>`).join("")
    : '<li class="chat-empty">No saved chats yet.</li>';
}

function startNewChat() {
  if (app.running) return;
  Object.assign(app, { threadId: uuid(), messages: [], state: {}, activeView: null, stepTimes: {} });
  history.replaceState(null, "", location.pathname);
  setStatus("idle", "Idle");
  greet();
  renderBoard();
  renderChatList();
  closeRecent();
  els.input.focus();
}

async function openChat(chatId) {
  if (app.running) return;
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}`);
    if (!response.ok) throw new Error(response.status === 404 ? "This chat no longer exists." : `Error ${response.status}`);
    const chat = await response.json();
    Object.assign(app, {
      threadId: chat.id,
      messages: chat.messages.map(({ id, role, content }) => ({ id, role, content })),
      state: chat.state || {},
      activeView: (chat.state || {}).view || null,
      stepTimes: {},
    });
    history.replaceState(null, "", `#chat/${chat.id}`);
    els.messages.innerHTML = "";
    if (app.messages.length) app.messages.forEach(appendMessage);
    else greet();
    setStatus("idle", "Idle");
    renderBoard();
    renderChatList();
    closeRecent();
  } catch (error) {
    startNewChat();
    appendError(error.message || String(error));
  }
}

els.chatList.addEventListener("click", async (e) => {
  const item = e.target.closest(".chat-item");
  if (!item || app.running) return;
  const chatId = item.dataset.chatId;
  if (e.target.closest(".chat-delete")) {
    if (!confirm("Delete this chat and its saved results?")) return;
    await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
    if (chatId === app.threadId) startNewChat();
    refreshChats();
  } else if (chatId !== app.threadId) {
    openChat(chatId);
  }
});

els.newChat.addEventListener("click", startNewChat);

// "Recent" dropdown in the chat header: opens the saved-chat list, closes on outside click / Escape.
function closeRecent() {
  els.recentMenu.hidden = true;
  els.recentToggle.setAttribute("aria-expanded", "false");
}
els.recentToggle.addEventListener("click", (e) => {
  e.stopPropagation();
  const opening = els.recentMenu.hidden;
  if (!opening) { closeRecent(); return; }
  els.recentMenu.hidden = false;
  els.recentToggle.setAttribute("aria-expanded", "true");
  refreshChats();
});
document.addEventListener("click", (e) => {
  if (!els.recentMenu.hidden && !e.target.closest("#recent-menu")) closeRecent();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRecent(); });

els.theme.addEventListener("click", () => {
  const root = document.documentElement;
  const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  root.dataset.theme = current === "dark" ? "light" : "dark";
  try { localStorage.setItem("rio-theme", root.dataset.theme); } catch (_) { /* storage unavailable */ }
});
try {
  const saved = localStorage.getItem("rio-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch (_) { /* storage unavailable */ }

setInterval(updateTimers, 500);

fetch("/api/info")
  .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
  .then((info) => { els.marketChip.textContent = `${info.market} · ${info.currency} · ${info.llm}`; })
  .catch(() => { els.marketChip.textContent = "Server unreachable"; });

// Reopen the chat in the address (#chat/<id>) after a reload; otherwise start a new one.
const chatInUrl = location.hash.match(/^#chat\/(.+)$/);
if (chatInUrl) openChat(decodeURIComponent(chatInUrl[1]));
else { greet(); renderBoard(); }
refreshChats();
els.input.focus();
