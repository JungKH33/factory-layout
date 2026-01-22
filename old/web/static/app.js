let sessionId = null;
let socket = null;
let currentState = null;
let policySeries = [];
let mctsSeries = [];
const maxPoints = 200;

const canvas = document.getElementById("layout-canvas");
const ctx = canvas.getContext("2d");
const chartCanvas = document.getElementById("value-chart");
const chartCtx = chartCanvas.getContext("2d");

function drawState(state) {
  currentState = state;
  if (state && typeof state.policy_v === "number" && typeof state.mcts_v === "number") {
    policySeries.push(state.policy_v);
    mctsSeries.push(state.mcts_v);
    if (policySeries.length > maxPoints) policySeries.shift();
    if (mctsSeries.length > maxPoints) mctsSeries.shift();
    drawChart();
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state) return;

  const gridW = state.grid.w;
  const gridH = state.grid.h;
  const scaleX = canvas.width / gridW;
  const scaleY = canvas.height / gridH;

  // placed rectangles
  ctx.fillStyle = "rgba(255,165,0,0.6)";
  ctx.strokeStyle = "#000";
  state.placed.forEach((item) => {
    const x = item.x * scaleX;
    const y = item.y * scaleY;
    const w = item.w * scaleX;
    const h = item.h * scaleY;
    ctx.beginPath();
    ctx.rect(x - w / 2, canvas.height - (y + h / 2) * scaleY, w, h);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.fillText(item.id, x - 6, canvas.height - y * scaleY);
    ctx.fillStyle = "rgba(255,165,0,0.6)";
  });

  // candidates
  state.candidates.forEach((cand) => {
    const cx = cand.x * scaleX;
    const cy = canvas.height - cand.y * scaleY;
    ctx.beginPath();
    ctx.fillStyle = cand.mask === 1 ? "green" : "red";
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  renderSidePanel(state);
}

function renderSidePanel(state) {
  const info = document.getElementById("state-info");
  info.innerHTML = `
    <div>Objective: ${state.obj.toFixed(3)}</div>
    <div>Remaining: ${state.remaining}</div>
    <div>Terminated: ${state.terminated}</div>
    <div>Policy V: ${state.policy_v.toFixed(3)}</div>
    <div>MCTS V: ${state.mcts_v.toFixed(3)}</div>
  `;

  const list = document.getElementById("candidate-list");
  list.innerHTML = "";
  state.candidates.forEach((cand, idx) => {
    const visits = state.visits && state.visits[idx] ? state.visits[idx] : 0;
    const el = document.createElement("div");
    el.className = "candidate-item";
    el.innerHTML = `
      <span>#${idx}</span>
      <span>(${cand.x.toFixed(1)}, ${cand.y.toFixed(1)})</span>
      <span>score ${cand.score.toFixed(2)}</span>
      <span>visits ${visits}</span>
      <button data-idx="${idx}">step</button>
    `;
    el.querySelector("button").addEventListener("click", () => step(idx));
    list.appendChild(el);
  });
}

function drawChart() {
  chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
  const all = policySeries.concat(mctsSeries);
  if (all.length === 0) return;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const pad = 10;
  const w = chartCanvas.width - pad * 2;
  const h = chartCanvas.height - pad * 2;

  const drawLine = (series, color) => {
    chartCtx.beginPath();
    series.forEach((v, i) => {
      const x = pad + (i / Math.max(series.length - 1, 1)) * w;
      const y = pad + h - ((v - min) / range) * h;
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 2;
    chartCtx.stroke();
  };

  drawLine(policySeries, "#ff7f0e");
  drawLine(mctsSeries, "#1f77b4");

  chartCtx.fillStyle = "#555";
  chartCtx.font = "10px Arial";
  chartCtx.fillText("Policy V", pad + 4, pad + 10);
  chartCtx.fillText("MCTS V", pad + 80, pad + 10);
}

async function startSession() {
  const model = document.getElementById("model-select").value;
  const useMcts = document.getElementById("use-mcts").checked;
  const res = await fetch("/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, use_mcts: useMcts }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  policySeries = [];
  mctsSeries = [];
  drawState(data.state);
  connectWebSocket();
}

async function step(action) {
  if (!sessionId) return;
  await fetch("/session/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, action }),
  });
}

async function runMcts() {
  if (!sessionId) return;
  const sims = parseInt(document.getElementById("sims-input").value, 10);
  await fetch("/session/mcts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, sims }),
  });
}

function connectWebSocket() {
  if (socket) {
    socket.close();
  }
  socket = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);
  socket.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.type === "state") {
      drawState(data.state);
    }
  };
}

document.getElementById("start-btn").addEventListener("click", startSession);
document.getElementById("mcts-btn").addEventListener("click", runMcts);

