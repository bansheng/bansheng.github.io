/* GTO 训练器 — 单文件前端，无构建步骤。
   刻意不用框架：这样同一份文件既能被 FastAPI 直接托管，也能原样丢到
   GitHub Pages 之类的静态托管上，不需要 npm build。 */

import { LocalBackend } from "./core/local-backend.js";

/* 后端探测：本地跑着 FastAPI 就用它（有 CFR solver + SQLite 全局手牌库），
   否则整套逻辑落到浏览器内的 LocalBackend（GitHub Pages 走这条）。 */
const REMOTE_CANDIDATES = [
  location.origin,
  "http://127.0.0.1:8848",
  "http://localhost:8848",
];
let remote = null;
let local = null;

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const SUITS = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED = new Set(["h", "d"]);
const GRID = "AKQJT98765432".split("");
const ACT_CN = { fold: "弃牌", check: "过牌", call: "跟注", bet: "下注", raise: "加注" };
const ACT_COLOR = { fold: "var(--fold)", check: "var(--dim)", call: "var(--call)", bet: "var(--raise)", raise: "var(--raise)" };

const state = {
  sessionId: null,
  hand: null,
  hero: 0,
  bots: {},
  drill: null,
  drillStats: { total: 0, hit: 0 },
  chart: null,
};

/* ---------------- helpers ---------------- */

async function probeRemote() {
  for (const base of REMOTE_CANDIDATES) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 1200);
      const res = await fetch(base + "/api/health", { signal: ctl.signal });
      clearTimeout(timer);
      if (res.ok) {
        const h = await res.json();
        if (h.ok && !h.local) return base;
      }
    } catch { /* 这个地址没有后端，试下一个 */ }
  }
  return null;
}

async function api(path, opts = {}) {
  if (remote) {
    const res = await fetch(remote + path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* body was not json */ }
      throw new Error(msg);
    }
    return res.json();
  }
  if (!local) local = new LocalBackend("./charts");
  return local.handle(path, opts);
}

function cardEl(txt, small = false) {
  const d = document.createElement("div");
  d.className = "pcard" + (small ? " small" : "");
  if (!txt) { d.classList.add("back"); d.textContent = "?"; return d; }
  const r = txt[0], s = txt[1];
  if (RED.has(s)) d.classList.add("red");
  d.innerHTML = `<span>${r}</span><span class="suit">${SUITS[s] || s}</span>`;
  return d;
}

function setStatus(text, cls = "") {
  const el = $("#conn");
  el.textContent = text;
  el.className = "status " + cls;
}

/* ---------------- tabs ---------------- */

$$(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.remove("active"));
    $$(".view").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#view-" + t.dataset.view).classList.add("active");
    if (t.dataset.view === "stats") loadStats();
    if (t.dataset.view === "review") loadHistory();
    if (t.dataset.view === "charts") loadCharts();
  })
);

/* ---------------- setup ---------------- */

function renderSetup() {
  const seats = +$("#cfg-seats").value;
  const heroSel = $("#cfg-hero");
  const prev = +heroSel.value || 0;
  heroSel.innerHTML = "";
  for (let i = 0; i < seats; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = `座位 ${i}`;
    heroSel.appendChild(o);
  }
  heroSel.value = Math.min(prev, seats - 1);

  const hero = +heroSel.value;
  const rows = $("#bot-rows");
  rows.innerHTML = "";
  for (let i = 0; i < seats; i++) {
    if (i === hero) continue;
    const row = document.createElement("div");
    row.className = "bot-row";
    row.innerHTML = `
      <span class="seat">座位 ${i}</span>
      <select data-seat="${i}">
        <option value="gto">GTO（按建议频率随机）</option>
        <option value="fish">鱼（松被动，练剥削）</option>
      </select>`;
    rows.appendChild(row);
  }
}
$("#cfg-seats").addEventListener("change", renderSetup);
$("#cfg-hero").addEventListener("change", renderSetup);
renderSetup();

$("#btn-start").addEventListener("click", async () => {
  const seats = +$("#cfg-seats").value;
  const hero = +$("#cfg-hero").value;
  const bots = $$("#bot-rows select").map((s) => ({
    seat: +s.dataset.seat, profile: s.value,
  }));
  try {
    const s = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        mode: "full", table_size: seats, stack_bb: +$("#cfg-stack").value,
        hero_seat: hero, bots,
      }),
    });
    state.sessionId = s.session_id;
    state.hero = hero;
    $("#setup").classList.add("hidden");
    $("#table-wrap").classList.remove("hidden");
    await deal();
  } catch (e) {
    setStatus("开局失败: " + e.message, "err");
  }
});

/* ---------------- play ---------------- */

async function deal() {
  $("#feedback").classList.add("hidden");
  const v = await api(`/api/sessions/${state.sessionId}/deal`, { method: "POST" });
  applyView(v);
}

function applyView(v) {
  state.hand = v.hand;
  state.bots = v.bots || {};
  $("#handno").textContent = v.hand_no;
  if (!v.hand) return;
  const h = v.hand;

  $("#pot").textContent = h.pot;
  $("#street").textContent = h.street;

  const board = $("#board");
  board.innerHTML = "";
  h.board.forEach((c) => board.appendChild(cardEl(c)));
  for (let i = h.board.length; i < 5; i++) {
    const ph = document.createElement("div");
    ph.className = "pcard back"; ph.style.opacity = ".18"; ph.textContent = "";
    board.appendChild(ph);
  }

  const seats = $("#seats");
  seats.innerHTML = "";
  h.seats.forEach((s) => {
    const d = document.createElement("div");
    d.className = "seat"
      + (s.index === h.hero_seat ? " hero" : "")
      + (s.index === h.actor ? " acting" : "")
      + (s.folded ? " folded" : "");
    const btn = s.index === h.button ? '<span class="btn-mark">D</span>' : "";
    const cards = (s.hole || [null, null])
      .map((c) => c).slice(0, 2);
    d.innerHTML = `
      <div class="nm"><span>${s.name} ${btn}</span><span class="pos">${s.position}</span></div>
      <div class="stk">筹码 ${s.stack}${s.all_in ? " · ALL-IN" : ""}</div>
      <div class="cards"></div>
      ${s.committed ? `<div class="bet">已投入 ${s.committed}</div>` : ""}`;
    const cw = d.querySelector(".cards");
    cards.forEach((c) => cw.appendChild(cardEl(c, true)));
    seats.appendChild(d);
  });

  renderActions(h);
  if (h.your_turn) loadAdvice();
  else if (h.finished) showResults(h);
  else $("#advice-bars").innerHTML = '<p class="hint">等待对手行动…</p>';
}

function renderActions(h) {
  const bar = $("#actionbar");
  bar.innerHTML = "";
  if (h.finished) {
    const b = document.createElement("button");
    b.className = "primary"; b.textContent = "下一手 →";
    b.onclick = deal;
    bar.appendChild(b);
    return;
  }
  if (!h.your_turn) {
    bar.innerHTML = '<span class="waiting">等待对手行动…</span>';
    return;
  }
  h.legal_actions.forEach((la) => {
    if (la.type === "bet" || la.type === "raise") {
      const wrap = document.createElement("div");
      wrap.className = "sizer";
      const btn = document.createElement("button");
      btn.className = "act raise";
      const range = document.createElement("input");
      range.type = "range"; range.min = la.min; range.max = la.max;
      range.value = Math.min(la.max, Math.max(la.min, Math.round(h.pot * 0.66)));
      const amt = document.createElement("span");
      amt.className = "amt";
      const sync = () => {
        amt.textContent = range.value;
        btn.textContent = `${ACT_CN[la.type]} 到 ${range.value}`;
      };
      range.oninput = sync; sync();
      btn.onclick = () => act(la.type, +range.value);
      wrap.append(btn, range, amt);
      bar.appendChild(wrap);
    } else {
      const b = document.createElement("button");
      b.className = "act " + la.type;
      b.textContent = ACT_CN[la.type] + (la.type === "call" ? ` ${la.amount}` : "");
      b.onclick = () => act(la.type, 0);
      bar.appendChild(b);
    }
  });
}

async function act(type, amount) {
  try {
    const v = await api(`/api/sessions/${state.sessionId}/act`, {
      method: "POST", body: JSON.stringify({ action: type, amount }),
    });
    if (v.last_decision) showFeedback(v.last_decision);
    applyView(v);
  } catch (e) {
    setStatus("动作失败: " + e.message, "err");
  }
}

function bars(actions, container, chosen) {
  container.innerHTML = "";
  actions.forEach((a) => {
    const row = document.createElement("div");
    row.className = "bar";
    const pct = (a.frequency * 100).toFixed(1);
    row.innerHTML = `
      <span class="lbl">${ACT_CN[a.action] || a.action}</span>
      <span class="track"><span class="fill" style="width:${pct}%;background:${ACT_COLOR[a.action] || "var(--accent)"}"></span></span>
      <span class="val">${pct}%</span>`;
    if (chosen && a.action === chosen) row.style.outline = "1px solid var(--accent)";
    container.appendChild(row);
  });
}

async function loadAdvice() {
  try {
    const a = await api(`/api/sessions/${state.sessionId}/advice`);
    const badge = $("#advice-source");
    badge.textContent = { exact: "精确解", approximate: "参考近似", heuristic: "启发式估算" }[a.confidence] || a.confidence;
    badge.className = "badge " + a.confidence;
    bars(a.actions, $("#advice-bars"));
    $("#advice-explain").textContent = a.explanation;
    $("#advice-caveat").textContent = a.caveat || "";
  } catch {
    $("#advice-bars").innerHTML = '<p class="hint">这个局面暂无建议。</p>';
  }
}

function showFeedback(d) {
  const fb = $("#feedback");
  const freq = (d.chosen_freq * 100).toFixed(1);
  const good = d.chosen_freq >= 0.2;
  fb.className = "feedback card " + (good ? "good" : "bad");
  fb.innerHTML = `
    <h3>${good ? "✓" : "✕"} 你选了「${ACT_CN[d.chosen] || d.chosen}」</h3>
    <p class="explain">${d.hand_label} @ ${d.position} · ${d.street}｜
       建议策略里这个动作占 <b>${freq}%</b>
       ${d.blunder ? "，属于基本不该打的动作" : ""}</p>`;
  fb.classList.remove("hidden");
}

function showResults(h) {
  if (!h.results) return;
  const net = h.results.net[h.hero_seat];
  const fb = $("#feedback");
  fb.className = "feedback card " + (net > 0 ? "good" : net < 0 ? "bad" : "");
  const rank = h.results.rankings[h.hero_seat];
  fb.innerHTML = `
    <h3>本手结束：${net > 0 ? "+" : ""}${net}</h3>
    <p class="explain">${h.results.showdown ? "摊牌" : "未摊牌"}
      ${rank ? "｜你的牌型：" + rank.category : ""}
      ｜公共牌 ${h.results.board || "—"}</p>`;
  fb.classList.remove("hidden");
  $("#advice-bars").innerHTML = "";
  $("#advice-explain").textContent = "";
  $("#advice-caveat").textContent = "";
}

/* ---------------- drill ---------------- */

$("#btn-drill").addEventListener("click", async () => {
  const q = await api("/api/drill", {
    method: "POST",
    body: JSON.stringify({
      spot_kind: $("#drill-kind").value,
      position: $("#drill-pos").value || null,
    }),
  });
  state.drill = q;
  $("#drill-q").classList.remove("hidden");
  $("#drill-answer").classList.add("hidden");
  $("#drill-spot").textContent = `${q.spot} — ${q.description}`;

  const hd = $("#drill-hand");
  hd.innerHTML = "";
  const [r1, r2] = [q.hand[0], q.hand[1]];
  const suited = q.hand[2] === "s";
  hd.appendChild(cardEl(r1 + "s"));
  hd.appendChild(cardEl(r2 + (suited ? "s" : "h")));

  const ac = $("#drill-actions");
  ac.innerHTML = "";
  ["fold", "call", "raise"].forEach((a) => {
    if (a === "call" && !("call" in q.answer) && !q.spot.includes("vs-")) return;
    const b = document.createElement("button");
    b.className = "act " + a;
    b.textContent = ACT_CN[a];
    b.onclick = () => answerDrill(a);
    ac.appendChild(b);
  });
});

function answerDrill(chosen) {
  const q = state.drill;
  const freq = q.answer[chosen] || 0;
  state.drillStats.total++;
  if (freq >= 0.2) state.drillStats.hit++;

  const ans = $("#drill-answer");
  ans.classList.remove("hidden");
  ans.innerHTML = `<h3>${q.hand} 在 ${q.spot} 的策略</h3><div id="drill-bars"></div>
    <p class="explain">你选了「${ACT_CN[chosen]}」，占 <b>${(freq * 100).toFixed(1)}%</b></p>
    <p class="caveat">${q.provenance.kind === "approximate-reference"
      ? "此 chart 为参考近似，范围宽度可信，单手牌混合频率不保证精确" : ""}</p>`;
  bars(
    Object.entries(q.answer).map(([action, frequency]) => ({ action, frequency })),
    $("#drill-bars"), chosen
  );
  const s = state.drillStats;
  $("#drill-score").textContent = `本轮：${s.hit}/${s.total} 命中（命中 = 选到建议频率 ≥20% 的动作）`;
}

/* ---------------- charts ---------------- */

async function loadCharts() {
  if (state.chartList) return;
  const data = await api("/api/charts");
  state.chartList = data.charts;
  const csel = $("#chart-name");
  csel.innerHTML = "";
  data.charts.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.name; o.textContent = c.label;
    csel.appendChild(o);
  });
  csel.onchange = () => { selectChart(csel.value); };
  selectChart(data.charts[0].name);
}

function selectChart(name) {
  state.chart = state.chartList.find((c) => c.name === name);
  const sel = $("#chart-spot");
  sel.innerHTML = "";
  state.chart.spots.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.key; o.textContent = `${s.key} — ${s.description}`;
    sel.appendChild(o);
  });
  $("#chart-prov").textContent = [
    state.chart.provenance.warning,
    state.chart.provenance.cross_check,
  ].filter(Boolean).join("  ");
  sel.onchange = drawMatrix;
  $("#chart-action").onchange = drawMatrix;
  drawMatrix();
}

async function drawMatrix() {
  const key = $("#chart-spot").value;
  const action = $("#chart-action").value;
  const spot = await api(
    `/api/charts/${state.chart.name}/spots/${encodeURIComponent(key)}`
  ).catch(() => null);
  const m = $("#matrix");
  m.innerHTML = "";
  if (!spot) { m.innerHTML = '<p class="empty">读取失败</p>'; return; }

  for (let r = 0; r < 13; r++) {
    for (let c = 0; c < 13; c++) {
      const hi = Math.min(r, c), lo = Math.max(r, c);
      const label = r === c ? GRID[r] + GRID[r]
        : GRID[hi] + GRID[lo] + (r < c ? "s" : "o");
      const st = spot.strategy[label] || { fold: 1 };
      const f = st[action] || 0;
      const cell = document.createElement("div");
      cell.className = "cell" + (f > 0.15 ? " on" : "");
      cell.title = `${label}  ` + Object.entries(st)
        .map(([k, v]) => `${ACT_CN[k]} ${(v * 100).toFixed(0)}%`).join("  ");
      const color = action === "raise" ? "--raise" : "--call";
      cell.innerHTML =
        `<span class="wfill" style="background:var(${color});opacity:${f.toFixed(2)}"></span>` +
        `<span class="lbltxt">${label}</span>`;
      m.appendChild(cell);
    }
  }
  $("#chart-pct").textContent = `${ACT_CN[action]}范围占比 ${spot.percent[action] ?? 0}%`;
}

/* ---------------- stats ---------------- */

async function loadStats() {
  if (!state.sessionId) {
    $("#kpis").innerHTML = '<p class="empty">先开一局。</p>';
    return;
  }
  const s = await api(`/api/sessions/${state.sessionId}/stats`);
  const k = s.summary;
  const kpi = (v, label, cls = "") =>
    `<div class="kpi ${cls}"><div class="v">${v}</div><div class="k">${label}</div></div>`;
  $("#kpis").innerHTML = [
    kpi(k.hands, "手数"),
    kpi(k.net_bb, "净收益 (bb)", k.net_bb > 0 ? "pos" : k.net_bb < 0 ? "neg" : ""),
    kpi(k.bb_per_100, "bb/100", k.bb_per_100 > 0 ? "pos" : k.bb_per_100 < 0 ? "neg" : ""),
    kpi(k.vpip_pct + "%", "VPIP"),
    kpi(k.pfr_pct + "%", "PFR"),
    kpi(k.wtsd_pct + "%", "摊牌率"),
    kpi(k.decisions, "决策数"),
    kpi(k.blunder_pct + "%", "严重偏差率", k.blunder_pct > 15 ? "neg" : ""),
  ].join("");

  const tbl = (rows, cols) =>
    rows.length
      ? `<table><tr>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr>` +
        rows.map((r) => `<tr class="${r.blunder_pct > 20 ? "warn" : ""}">` +
          cols.map((c) => `<td>${c[1](r)}</td>`).join("") + "</tr>").join("") +
        "</table>"
      : '<p class="empty">数据还不够。</p>';

  $("#leak-pos").innerHTML = tbl(s.by_position, [
    ["位置", (r) => r.position], ["街", (r) => r.street],
    ["决策数", (r) => r.decisions],
    ["平均偏差", (r) => (r.avg_freq_gap * 100).toFixed(1) + "%"],
    ["严重偏差", (r) => r.blunder_pct + "%"],
  ]);
  $("#leak-hand").innerHTML = tbl(s.by_hand, [
    ["起手牌", (r) => r.hand], ["次数", (r) => r.decisions],
    ["平均偏差", (r) => (r.avg_freq_gap * 100).toFixed(1) + "%"],
    ["严重偏差", (r) => r.blunders],
  ]);
}

/* ---------------- history / replay ---------------- */

async function loadHistory() {
  if (!state.sessionId) {
    $("#hand-list").innerHTML = '<p class="empty">先开一局。</p>';
    return;
  }
  const { hands } = await api(`/api/sessions/${state.sessionId}/hands`);
  const list = $("#hand-list");
  if (!hands.length) { list.innerHTML = '<p class="empty">还没有手牌。</p>'; return; }
  list.innerHTML = "";
  hands.forEach((h) => {
    const d = document.createElement("div");
    d.className = "hand-item";
    d.innerHTML = `
      <span>#${h.hand_no}</span>
      <span>${h.hero_position}</span>
      <span>${h.hero_cards}</span>
      <span style="color:var(--muted)">${h.board || "—"}</span>
      <span class="net ${h.hero_net > 0 ? "pos" : h.hero_net < 0 ? "neg" : ""}"
            style="margin-left:auto">${h.hero_net > 0 ? "+" : ""}${h.hero_net}</span>`;
    d.onclick = () => replay(h.id);
    list.appendChild(d);
  });
}

async function replay(handId) {
  const h = await api(`/api/hands/${handId}`);
  const box = $("#replay");
  box.classList.remove("hidden");
  const snap = h.snapshot;
  const acts = (snap.actions || []).map((a) => {
    const dec = h.decisions.find(
      (d) => d.street === a.street && d.chosen === a.type
    );
    const isHero = a.seat === h.hero_seat;
    const cls = isHero ? (dec && dec.blunder ? "blunder" : "hero") : "";
    const seatName = snap.seats[a.seat] ? snap.seats[a.seat].name : `座位${a.seat}`;
    const advised = dec
      ? `　建议：${Object.entries(dec.advised)
          .map(([k, v]) => `${ACT_CN[k] || k} ${(v * 100).toFixed(0)}%`).join(" / ")}`
      : "";
    return `<li class="${cls}">[${a.street}] ${seatName} ${ACT_CN[a.type] || a.type}` +
      `${a.amount ? " " + a.amount : ""}${advised}</li>`;
  }).join("");
  box.innerHTML = `
    <h3>第 ${h.hand_no} 手 · ${h.hero_position} · ${h.hero_cards}</h3>
    <p class="explain">公共牌 ${h.board || "—"}｜底池 ${h.pot}｜净收益
      <b style="color:${h.hero_net >= 0 ? "var(--call)" : "var(--raise)"}">
        ${h.hero_net > 0 ? "+" : ""}${h.hero_net}</b></p>
    <ul class="timeline">${acts || "<li>无动作记录</li>"}</ul>`;
}

/* ---------------- boot ---------------- */

(async () => {
  setStatus("探测后端…");
  remote = await probeRemote();
  try {
    const h = await api("/api/health");
    if (remote) {
      setStatus(`已连接后端 · ${h.charts.length} 套图表 · 支持 solver`, "ok");
    } else {
      setStatus(`浏览器本地模式 · ${h.charts.length} 套图表 · 无 solver`, "");
      $("#conn").title =
        "所有计算都在你的浏览器里跑，数据存在本机 localStorage。" +
        "想要 CFR 精确解和 SQLite 手牌库，在本机启动后端后刷新本页。";
    }
  } catch (e) {
    setStatus("初始化失败: " + e.message, "err");
  }
})();
