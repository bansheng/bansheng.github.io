/* GTO 训练器 — 单文件前端，无构建步骤。
   刻意不用框架：这样同一份文件既能被 FastAPI 直接托管，也能原样丢到
   GitHub Pages 之类的静态托管上，不需要 npm build。 */

import { LocalBackend } from "./core/local-backend.js";

/* 后端探测：本地跑着 FastAPI 就用它（有 CFR solver + SQLite 全局手牌库），
   否则整套逻辑落到浏览器内的 LocalBackend（GitHub Pages 走这条）。 */
const BACKEND_KEY = "gto-backend-url";
const savedBackend = () => { try { return localStorage.getItem(BACKEND_KEY) || ""; } catch { return ""; } };

/* 探测顺序：手动设的地址 → 当前站点 → 本机回环。
   注意一条浏览器的硬规则：**HTTPS 页面不能请求 HTTP 地址**（混合内容拦截），
   localhost / 127.0.0.1 是唯一的例外（浏览器把它当可信来源）。
   所以从 https://dingyadong.top/gto/ 连不上局域网里的 http 后端 ——
   要用后端就直接打开后端自己的地址，它同样托管这套界面。 */
const probeList = () => [savedBackend(), location.origin,
                         "http://127.0.0.1:8848", "http://localhost:8848"].filter(Boolean);
let remote = null;
let local = null;
let backendInfo = null;

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const SUITS = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED = new Set(["h", "d"]);
const GRID = "AKQJT98765432".split("");
const ACT_CN = { fold: "弃牌", check: "过牌", call: "跟注", bet: "下注", raise: "加注" };
const CONFIDENCE_CN = { exact: "精确解", approximate: "参考近似", heuristic: "启发式估算" };
const ACT_COLOR = { fold: "var(--fold)", check: "var(--dim)", call: "var(--call)", bet: "var(--raise)", raise: "var(--raise)" };

const state = {
  sessionId: null,
  hand: null,
  hero: 0,
  bots: {},
  drill: null,
  peeked: false,
  drillStats: { total: 0, hit: 0 },
  chart: null,
};

/* ---------------- helpers ---------------- */

async function probeRemote() {
  for (const base of probeList()) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 1200);
      const res = await fetch(base + "/api/health", { signal: ctl.signal });
      clearTimeout(timer);
      if (res.ok) {
        const h = await res.json();
        if (h.ok && !h.local) { backendInfo = h; return base; }
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

const bold = (t) => String(t || "").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

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
        chip_mode: $("#cfg-chipmode").value,
        buyin_budget: +$("#cfg-budget").value || 2000,
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
  clearTimeout(dealTimer);
  state.peeked = false;
  $("#feedback").classList.add("hidden");
  $("#analysis").classList.add("hidden");
  try {
    applyView(await api(`/api/sessions/${state.sessionId}/deal`, { method: "POST" }));
  } catch (e) {
    const fb = $("#feedback");
    fb.className = "feedback card bad";
    fb.innerHTML = `<h3>这一局结束</h3><p class="explain">${e.message}</p>`;
    fb.classList.remove("hidden");
    $("#actionbar").innerHTML = '<span class="waiting">刷新页面可以重新开一局。</span>';
  }
}

function applyView(v) {
  state.hand = v.hand;
  state.bots = v.bots || {};
  $("#handno").textContent = v.hand_no;
  if (v.chips) renderChips(v.chips);
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

  renderTable(h);

  renderActions(h);
  if (h.your_turn) hideAdvice();
  else if (h.finished) showResults(h);
  else $("#advice-bars").innerHTML = '<p class="hint">等待对手行动…</p>';
}

/* 训练的意义在于先自己判断。所以轮到你时建议是盖着的，
   出完动作才揭晓；实在想不出来可以主动点开，但那次会记为「偷看」。 */
function hideAdvice() {
  $("#analysis").classList.add("hidden");
  $("#advice-title").textContent = "GTO 建议";
  $("#advice-source").textContent = "已隐藏";
  $("#advice-source").className = "badge";
  $("#advice-bars").innerHTML =
    '<p class="hint">先自己判断 —— 出完动作才揭晓建议。</p>' +
    '<button class="peek" id="btn-peek">想不出来，先看答案</button>';
  $("#advice-explain").textContent = "";
  $("#advice-caveat").textContent = "";
  $("#btn-peek").onclick = () => { state.peeked = true; loadAdvice(); };
  renderSolveNow();
}

/* 座位按真实牌桌的位置摆：英雄永远在正下方，其余按顺时针绕桌一圈。
   坐标是每种人数写死的百分比 —— 用三角函数均分角度在 2 人和 9 人时都难看，
   写死反而能照顾每种人数的实际观感（比如 6 人桌希望正上方有人、正左右各一个）。 */
const SEAT_LAYOUT = {
  2: [[50, 90], [50, 8]],
  3: [[50, 90], [8, 32], [92, 32]],
  4: [[50, 90], [6, 50], [50, 6], [94, 50]],
  5: [[50, 90], [6, 60], [22, 12], [78, 12], [94, 60]],
  6: [[50, 90], [5, 64], [15, 16], [50, 6], [85, 16], [95, 64]],
  7: [[50, 90], [5, 68], [9, 28], [34, 8], [66, 8], [91, 28], [95, 68]],
  8: [[50, 90], [5, 72], [5, 38], [24, 10], [50, 5], [76, 10], [95, 38], [95, 72]],
  9: [[50, 90], [6, 76], [4, 46], [15, 16], [38, 6], [62, 6], [85, 16], [96, 46], [94, 76]],
};

/** 座位 i 在牌桌上的位置。英雄的相对序号是 0，其余顺时针排开。 */
function seatSpot(index, hero, n) {
  const layout = SEAT_LAYOUT[n] || SEAT_LAYOUT[6];
  const rel = ((index - hero) % n + n) % n;
  return layout[rel] || layout[0];
}

/* 下注筹码摆在座位和桌心之间，视觉上就是"推出去的那一份"。 */
function betSpot([x, y]) {
  return [x + (50 - x) * 0.42, y + (50 - y) * 0.42];
}

function renderTable(h) {
  const n = h.seats.length;
  const hero = h.hero_seat;
  const seatLayer = $("#seats");
  const betLayer = $("#bets");
  seatLayer.innerHTML = "";
  betLayer.innerHTML = "";

  h.seats.forEach((s) => {
    const [x, y] = seatSpot(s.index, hero, n);
    const box = document.createElement("div");
    box.className = "pseat"
      + (s.index === hero ? " me" : "")
      + (s.index === h.actor ? " acting" : "")
      + (s.folded ? " out" : "")
      + (s.all_in ? " allin" : "");
    box.style.left = x + "%";
    box.style.top = y + "%";

    const cards = document.createElement("div");
    cards.className = "pseat-cards";
    (s.hole || [null, null]).slice(0, 2).forEach((c) => cards.appendChild(cardEl(c, true)));

    const info = document.createElement("div");
    info.className = "pseat-info";
    info.innerHTML = `
      <div class="pseat-name">${s.name}<span class="pseat-pos">${s.position}</span></div>
      <div class="pseat-stack">${s.all_in ? "ALL-IN" : s.stack}</div>`;

    box.append(cards, info);
    seatLayer.appendChild(box);

    if (s.committed > 0) {
      const [bx, by] = betSpot([x, y]);
      const chip = document.createElement("div");
      chip.className = "bet-chip";
      chip.style.left = bx + "%";
      chip.style.top = by + "%";
      chip.innerHTML = `<span class="disc"></span>${s.committed}`;
      betLayer.appendChild(chip);
    }
  });

  // 按钮标记贴在按钮位座位的靠桌心一侧
  const d = $("#dealer");
  if (h.button != null && h.seats[h.button]) {
    const [bx, by] = seatSpot(h.button, hero, n);
    d.style.left = bx + (50 - bx) * 0.2 + "%";
    d.style.top = by + (50 - by) * 0.2 + "%";
    d.textContent = "D";
    d.classList.remove("hidden");
  } else {
    d.classList.add("hidden");
  }

  const board = $("#board");
  board.innerHTML = "";
  h.board.forEach((c) => board.appendChild(cardEl(c)));
  for (let i = h.board.length; i < 5; i++) {
    const ph = document.createElement("div");
    ph.className = "pcard slot";
    board.appendChild(ph);
  }
  $("#center-pot").innerHTML = h.pot
    ? `<span class="disc"></span>底池 <b>${h.pot}</b>`
    : "";
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

/* 后端一次就把「你的动作 + 所有 bot 的动作 + 新发的牌」全算完了。
   直接渲染的话，你还没看清反馈，转牌就已经拍在桌上了。
   所以只要公共牌变多了，就先给 DEAL_DELAY_MS 的时间看建议，再翻牌。 */
const DEAL_DELAY_MS = 3000;
let dealTimer = null;

async function act(type, amount) {
  const before = state.hand ? state.hand.board.length : 0;
  try {
    const v = await api(`/api/sessions/${state.sessionId}/act`, {
      method: "POST", body: JSON.stringify({ action: type, amount }),
    });
    // applyView 会为下一个决策把建议重新盖上，所以揭晓要排在它之后
    const settleView = () => {
      applyView(v);
      if (v.last_decision) showFeedback(v.last_decision);
    };

    const after = v.hand ? v.hand.board.length : 0;
    if (after > before) {
      // 先只把反馈和答案摆出来，牌桌停在旧街，等 3 秒再翻新牌
      if (v.last_decision) showFeedback(v.last_decision);
      renderActions({ ...v.hand, your_turn: false, finished: false });
      countdownThen(settleView);
    } else {
      settleView();
    }
  } catch (e) {
    setStatus("动作失败: " + e.message, "err");
  }
}

function countdownThen(fn) {
  clearTimeout(dealTimer);
  const bar = $("#actionbar");
  let left = Math.round(DEAL_DELAY_MS / 1000);
  const paint = () => {
    bar.innerHTML =
      `<span class="waiting">${left} 秒后发牌…</span>` +
      '<button class="act" id="btn-skip">立刻发牌</button>';
    $("#btn-skip").onclick = () => { clearInterval(tick); clearTimeout(dealTimer); fn(); };
  };
  paint();
  const tick = setInterval(() => { left -= 1; if (left > 0) paint(); }, 1000);
  dealTimer = setTimeout(() => { clearInterval(tick); fn(); }, DEAL_DELAY_MS);
}

function bars(actions, container, chosen) {
  container.innerHTML = "";
  // 你选的动作如果频率是 0，建议列表里根本没有它 —— 那更要画出来，
  // 一条 0% 的空槽比「找不到自己选的那项」清楚得多。
  if (chosen && !actions.some((a) => a.action === chosen)) {
    actions = actions.concat([{ action: chosen, frequency: 0 }]);
  }
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
    $("#advice-title").textContent = "GTO 建议（本步，已偷看）";
    const badge = $("#advice-source");
    badge.textContent = CONFIDENCE_CN[a.confidence] || a.confidence;
    badge.className = "badge " + a.confidence;
    bars(a.actions, $("#advice-bars"));
    $("#advice-explain").textContent = a.explanation;
    $("#advice-caveat").textContent = a.caveat || "";
    if (a.analysis) showAnalysis(a.analysis);
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
    <h3>${good ? "✓" : "✕"} 你选了「${ACT_CN[d.chosen] || d.chosen}」${state.peeked ? " （偷看过）" : ""}</h3>
    <p class="explain">${d.hand_label} @ ${d.position} · ${d.street}｜
       建议策略里这个动作占 <b>${freq}%</b>
       ${d.blunder ? "，属于基本不该打的动作" : ""}</p>`;
  fb.classList.remove("hidden");
  state.peeked = false;

  // 揭晓：把完整建议摊开，并高亮你实际选的那一项。
  // 这是**刚才那一步**的答案，不是当前面临的决策，标题要写明。
  $("#advice-title").textContent = `上一步的 GTO 建议（${d.street} · ${d.hand_label}）`;
  const a = d.advice;
  const badge = $("#advice-source");
  badge.textContent = CONFIDENCE_CN[a.confidence] || a.confidence;
  badge.className = "badge " + a.confidence;
  bars(a.actions, $("#advice-bars"), d.chosen);
  $("#advice-explain").textContent = a.explanation;
  $("#advice-caveat").textContent = a.caveat || "";

  if (d.analysis) showAnalysis(d.analysis);
  renderSolveNow();
}

/* 翻后要拿到真解，只有"这个局面已经解过"一条路 —— 现算要几分钟。
   所以这里给的是一个明确的动作：把当前局面丢进求解队列，算完自动把
   建议从红色「启发式估算」换成绿色「精确解」。 */
async function renderSolveNow() {
  const box = $("#solve-now");
  if (!box) return;
  box.innerHTML = "";
  const h = state.hand;
  if (!h || h.board.length < 3) return;
  if (!remote) {
    box.innerHTML =
      '<p class="caveat">翻后精确解需要后端（浏览器算不动 CFR）。' +
      '点右上角状态栏设置后端地址。</p>';
    return;
  }
  if (!backendInfo?.solver_ready) {
    box.innerHTML = '<p class="caveat">后端已连上，但求解器没编译：跑 <code>scripts/build_solver.sh</code>。</p>';
    return;
  }
  box.innerHTML =
    '<button class="peek" id="btn-solve-now">求解这个局面（真 CFR）</button>' +
    '<span class="hint" id="solve-now-msg" style="margin-left:8px"></span>';
  $("#btn-solve-now").onclick = solveCurrentSpot;
}

async function solveCurrentSpot() {
  const msg = $("#solve-now-msg");
  const btn = $("#btn-solve-now");
  btn.disabled = true;
  msg.textContent = "入队中…";
  let row;
  try {
    row = await api(`/api/sessions/${state.sessionId}/solve`,
      { method: "POST", body: JSON.stringify({ bet_sizes: [50], accuracy: 0.5 }) });
  } catch (e) {
    msg.textContent = "❌ " + e.message;
    btn.disabled = false;
    return;
  }
  const started = Date.now();
  const tick = async () => {
    let r;
    try { r = await api(`/api/solver/solve/${row.spot_key}`); } catch { return; }
    const secs = Math.round((Date.now() - started) / 1000);
    if (r.status === "done") {
      msg.textContent = `✅ 解完了（${r.seconds}s，可利用度 ${r.exploitability}），正在刷新建议…`;
      try {
        const a = await api(`/api/sessions/${state.sessionId}/advice`);
        showRevealedAdvice(a, null);
        msg.textContent = `✅ 已用精确解（${r.seconds}s，可利用度 ${r.exploitability}）`;
      } catch { msg.textContent = "✅ 解完了，但这手牌不在求解范围内，仍用启发式"; }
      return;
    }
    if (r.status === "failed") {
      msg.textContent = "❌ " + (r.error || "求解失败");
      btn.disabled = false;
      return;
    }
    msg.textContent = `求解中…（${r.status}，已等 ${secs}s；一个 flop 局面通常 1-4 分钟）`;
    setTimeout(tick, 4000);
  };
  tick();
}

/* 把一份 advice 直接画到建议面板上（求解完成后替换用） */
function showRevealedAdvice(a, chosen) {
  $("#advice-title").textContent = "GTO 建议";
  const badge = $("#advice-source");
  badge.textContent = CONFIDENCE_CN[a.confidence] || a.confidence;
  badge.className = "badge " + a.confidence;
  bars(a.actions, $("#advice-bars"), chosen);
  $("#advice-explain").textContent = a.explanation || "";
  $("#advice-caveat").textContent = a.caveat || "";
  if (a.analysis) showAnalysis(a.analysis);
}

/* 建议告诉你「打什么」，分析告诉你「为什么」。后者才是能带走的东西：
   频率记不住，但数组合、比价格和胜率，是可以练成习惯的。 */
function showAnalysis(an) {
  const box = $("#analysis");
  box.classList.remove("hidden");

  $("#ana-derive").textContent =
    `${an.villain.derivation}　→　${an.villain.combos} 个组合，占全部起手牌 ${an.villain.percent}%`;

  const g = $("#ana-grid");
  g.innerHTML = "";
  an.villain.grid.forEach((row, r) =>
    row.forEach((w, c) => {
      const hi = Math.min(r, c), lo = Math.max(r, c);
      const label = r === c ? GRID[r] + GRID[r] : GRID[hi] + GRID[lo] + (r < c ? "s" : "o");
      const cell = document.createElement("div");
      cell.className = "mcell";
      cell.title = `${label}  ${(w * 100).toFixed(0)}%`;
      cell.style.background = w > 0
        ? `color-mix(in srgb, var(--raise) ${Math.round(w * 100)}%, var(--fold))`
        : "var(--fold)";
      g.appendChild(cell);
    })
  );

  $("#ana-cats").innerHTML = an.villain.categories.length
    ? an.villain.categories.map((c) =>
        `<div class="catrow"><span>${c.name}</span>
         <span class="track"><span class="fill" style="width:${c.percent}%"></span></span>
         <span class="v">${c.percent}%</span></div>`).join("")
    : '<p class="hint">翻前没有公共牌，无法按牌型拆分。</p>';

  const m = an.matchup;
  const kv = (k, v, cls = "") => `<div class="kv"><span>${k}</span><b class="${cls}">${v}</b></div>`;
  $("#ana-matchup").innerHTML = [
    m.hero_hand ? kv("你的牌型", m.hero_hand) : "",
    m.equity != null ? kv("你的胜率", (m.equity * 100).toFixed(1) + "%",
      m.equity > 0.5 ? "good" : "bad") : "",
    m.ahead_pct != null ? kv("现在摊牌领先", `${m.ahead_pct}%（${m.ahead_combos} 个组合）`, "good") : "",
    m.behind_pct != null ? kv("现在摊牌落后", `${m.behind_pct}%（${m.behind_combos} 个组合）`, "bad") : "",
    m.tied_combos ? kv("平分", `${m.tied_combos} 个组合`) : "",
    m.note ? `<p class="hint" style="margin-top:8px">${m.note}</p>` : "",
  ].filter(Boolean).join("");

  const o = an.odds;
  $("#ana-odds").innerHTML = [
    kv("底池", `${o.pot_bb}bb`),
    o.to_call ? kv("需要跟", `${o.to_call_bb}bb`) : "",
    o.required_equity != null
      ? kv("跟注门槛胜率", (o.required_equity * 100).toFixed(1) + "%") : "",
    o.ev_call_bb != null
      ? kv("跟注 EV", `${o.ev_call_bb > 0 ? "+" : ""}${o.ev_call_bb}bb`,
           o.ev_call_bb > 0 ? "good" : "bad") : "",
    o.spr != null ? kv("SPR（有效筹码/底池）", o.spr) : "",
    kv("有效筹码", `${o.effective_stack_bb}bb`),
    o.bluff_breakeven?.length
      ? `<div class="kv"><span>诈唬打平所需弃牌率</span><b>${
          o.bluff_breakeven.map((b) => `${b.label} ${b.fold_pct_needed}%`).join("　")
        }</b></div>` : "",
  ].filter(Boolean).join("");

  $("#ana-summary").innerHTML = bold(an.villain.summary);
  $("#ana-read").innerHTML = bold(an.villain.read);

  const mine = an.hero_range || {};
  $("#ana-mine").innerHTML = bold(mine.summary || "翻前还没有公共牌，无法定位相对牌力。");
  $("#ana-mine-note").innerHTML = bold(mine.note || "");

  $("#ana-reasons").innerHTML = an.reasons.map((r) => `<li>${bold(r)}</li>`).join("");

  const lv = an.lead_vs_equity || {};
  $("#ana-lead").innerHTML = lv.trials ? `
    <div class="leadbox">
      <div class="leadrow">
        <div class="leadcell"><span>现在摊牌就赢</span><b>${(lv.lead_now * 100).toFixed(0)}%</b>
          <em>纯数组合，不含运气</em></div>
        <div class="leadarrow">→</div>
        <div class="leadcell"><span>发完剩下的牌后赢</span><b class="hi">${(lv.equity_after * 100).toFixed(0)}%</b>
          <em>这才是胜率(equity)</em></div>
      </div>
      <div class="leadsplit">
        <span class="s1">现在领先且守住 ${(lv.held * 100).toFixed(0)}%</span>
        <span class="s2">现在领先被反超 ${(lv.lost * 100).toFixed(0)}%</span>
        <span class="s3">现在落后但反超 ${(lv.caught * 100).toFixed(0)}%</span>
        <span class="s4">一直落后 ${(lv.never * 100).toFixed(0)}%</span>
      </div>
      <div class="fexpr">${bold(lv.formula)}</div>
      <div class="fapplied">${bold(lv.worked)}</div>
      <div class="fnote">${bold(lv.reading)}</div>
    </div>` : '<p class="hint">河牌上没有后续的牌，领先概率就等于胜率，两者是同一个数。</p>';

  const cm = an.combo_math || {};
  $("#ana-combomath").innerHTML = `
    <div class="formula">
      <div class="fname">第一步：一手牌有几种组合</div>
      ${(cm.basics || []).map((b) =>
        `<div class="fapplied">${b.kind} = <b>${b.full}</b> 个　<span class="fnote">${b.why}</span></div>`).join("")}
    </div>
    ${(cm.blockers || []).length ? `
    <div class="formula">
      <div class="fname">第二步：减掉已经露面的牌（最容易漏的一步）</div>
      ${cm.blockers.slice(0, 5).map((b) =>
        `<div class="fapplied"><b>${b.rank}</b> 已现 ${b.gone} 张 →
          ${b.pair}　${b.suited}　${b.offsuit}</div>`).join("")}
    </div>` : ""}
    ${(cm.beats_you || []).length ? `
    <div class="formula">
      <div class="fname">第三步：数出能打败你的组合</div>
      <div class="fapplied">${cm.beats_you.slice(0, 8).map((b) =>
        `${b.hand} <b>${b.combos}</b>`).join("　")}</div>
      <div class="fnote">合计 <b>${cm.beats_total}</b> 个，对手范围共 <b>${cm.range_total}</b> 个</div>
    </div>` : ""}
    ${(cm.steps || []).length ? `
    <div class="formula">
      <div class="fname">第四步：算成概率</div>
      ${cm.steps.map((s) => `<div class="fapplied">${bold(s)}</div>`).join("")}
    </div>` : ""}
    <p class="fnote">${bold(cm.note || "")}</p>`;

  $("#ana-shortcuts").innerHTML = (an.shortcuts || []).map((sc) => `
    <div class="formula">
      <div class="fname">${sc.name}</div>
      <div class="fexpr">${bold(sc.formula)}</div>
      <div class="fapplied">${bold(sc.applied)}</div>
      <div class="fnote">${bold(sc.note)}</div>
    </div>`).join("");
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

/* 筹码是连续的：一手打完带到下一手，只有下不起盲注才自动补。
   把还能打的筹码自动补满会让每手都是同一个深度，SPR、短码打法全都失去意义。 */
function renderChips(c) {
  state.chips = c;
  const el = $("#bankroll");
  const net = c.net >= 0 ? `+${c.net}` : `${c.net}`;
  el.innerHTML = c.chip_mode === "fixed"
    ? `<span class="dim">固定 ${c.buyin_bb}bb（练习模式，每手重置）</span>`
    : `你的筹码 <b>${c.hero_stack}</b>（${c.hero_stack_bb}bb）｜` +
      `已买入 ${c.hero_bought_in}/${c.buyin_budget}｜` +
      `<b class="${c.net >= 0 ? "pos" : "neg"}">${net}</b>` +
      (c.rebuys ? `｜补码 ${c.rebuys} 次` : "");
  const btn = $("#btn-rebuy");
  const canShow = c.can_rebuy && (!state.hand || state.hand.finished);
  btn.classList.toggle("hidden", !canShow);
  btn.textContent = `补码到 ${c.buyin}`;
  btn.onclick = async () => {
    try {
      renderChips(await api(`/api/sessions/${state.sessionId}/rebuy`, { method: "POST" }));
    } catch (e) { setStatus("补码失败: " + e.message, "err"); }
  };
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

  // 按钮从这个 spot 实际有哪些动作推出来，不要写死。
  // 推-弃表里 "raise" 的意思是全下，标成「加注」会看不懂。
  const isPush = q.spot.includes("push-") && !q.spot.includes("vs-push");
  // 用局面的动作集合，不用这手牌的 —— 只给一个"弃牌"按钮等于送分
  const available = new Set(q.spot_actions?.length
    ? q.spot_actions
    : ["fold", ...Object.keys(q.answer)]);
  const ac = $("#drill-actions");
  ac.innerHTML = "";
  ["fold", "call", "raise"].filter((a) => available.has(a)).forEach((a) => {
    const b = document.createElement("button");
    b.className = "act " + a;
    b.textContent = a === "raise" && isPush ? "全下" : ACT_CN[a];
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
      // 手机上格子里的字被 CSS 隐藏了，hover 也不存在，所以点一下把这手牌的
      // 完整策略显示在下方
      cell.onclick = () => {
        $("#chart-pct").textContent =
          `${label}：` + Object.entries(st)
            .map(([k, v]) => `${ACT_CN[k]} ${(v * 100).toFixed(0)}%`).join("　");
      };
      m.appendChild(cell);
    }
  }
  const isPush = key.includes("push-") && !key.includes("vs-push");
  const actionCn = action === "raise" && isPush ? "全下" : ACT_CN[action];
  $("#chart-pct").textContent = `${actionCn}范围占比 ${spot.percent[action] ?? 0}%`;
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

  // 你到底是被什么标准打的分 —— 全靠启发式打的分只是练手，不是 GTO 学习
  const src = k.by_source || {};
  $("#kpis").insertAdjacentHTML("afterend", `
    <p class="hint" id="src-note">本局评分依据：
      <b style="color:var(--call)">精确解 ${src.solver || 0}</b> ·
      <b style="color:var(--gold)">chart ${src.chart || 0}</b> ·
      <b style="color:var(--raise)">启发式 ${src.heuristic || 0}</b>
      ${k.solver_pct ? `（真解占比 ${k.solver_pct}%）`
        : "　—— 目前全部翻后决策都是启发式估算，不是 GTO。想要真解：连后端 + 点「求解这个局面」，或等预解库跑大。"}</p>`);
  const dup = document.querySelectorAll("#src-note");
  for (let i = 0; i < dup.length - 1; i++) dup[i].remove();

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
    <ul class="timeline">${acts || "<li>无动作记录</li>"}</ul>
    <div id="solve-box"></div>`;
  renderSolveBox(h);
}

/* 翻后精确解只有本机后端跑得动，而且一个局面要几十秒到几分钟。
   所以这里是「排队 + 轮询」，不是「点了就等」。 */
async function renderSolveBox(hand) {
  const box = $("#solve-box");
  if (!remote) {
    box.innerHTML =
      '<p class="caveat">精确解需要本机后端（浏览器算不动 CFR）。' +
      '在本机启动后端后刷新本页即可。</p>';
    return;
  }
  let st;
  try { st = await api("/api/solver/status"); } catch { return; }
  if (!st.available) {
    box.innerHTML =
      '<p class="caveat">求解器还没编译。跑 <code>scripts/build_solver.sh</code> 之后这里会出现「求解此局面」。</p>';
    return;
  }
  const board = (hand.board || "").slice(0, 6);
  if (board.length < 6) {
    box.innerHTML = '<p class="hint">这手牌没打到翻牌，没有可求解的翻后局面。</p>';
    return;
  }
  box.innerHTML =
    `<button class="primary" id="btn-solve">求解翻牌局面 ${board}（真 CFR）</button>` +
    '<p class="hint" id="solve-msg">一个 flop 局面大约 30 秒到几分钟，算完会存下来，' +
    '之后遇到同样的局面直接命中缓存。</p>';
  $("#btn-solve").onclick = () => enqueueSolve(hand, board);
}

async function enqueueSolve(hand, board) {
  const msg = $("#solve-msg");
  const snap = hand.snapshot || {};
  const bb = snap.big_blind || 2;
  try {
    const row = await api("/api/solver/solve", {
      method: "POST",
      body: JSON.stringify({
        board,
        // 用当前 chart 的默认范围当双方范围。真正贴合这手牌的范围需要
        // 从牌局动作反推，那是下一步的事 —— 现在如实标注用的是什么。
        oop_range: "22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,87s,76s,65s,A7o+,KTo+,QJo",
        ip_range: "22+,A2s+,K9s+,Q9s+,J8s+,T8s+,97s+,86s+,76s,65s,ATo+,KJo+,QJo",
        pot: Math.max(bb * 5, hand.pot || bb * 5),
        effective_stack: bb * 100,
      }),
    });
    msg.textContent = `已入队（key ${row.spot_key}），状态：${row.status}`;
    pollSolve(row.spot_key, msg);
  } catch (e) {
    msg.textContent = "入队失败：" + e.message;
  }
}

async function pollSolve(key, msg) {
  const started = Date.now();
  const tick = async () => {
    let row;
    try { row = await api(`/api/solver/solve/${key}`); } catch { return; }
    const secs = Math.round((Date.now() - started) / 1000);
    if (row.status === "done") {
      msg.innerHTML = `✅ 求解完成，用时 ${row.seconds}s` +
        (row.exploitability != null ? `，可利用度 ${row.exploitability}` : "") +
        `<br>结果已缓存：<code>${row.strategy_path}</code>`;
      return;
    }
    if (row.status === "failed") {
      msg.textContent = "❌ 求解失败：" + (row.error || "未知原因");
      return;
    }
    msg.textContent = `求解中…（${row.status}，已等 ${secs}s）`;
    setTimeout(tick, 3000);
  };
  tick();
}

/* ---------------- boot ---------------- */

/* 后端设置弹窗：公网版要连局域网后端，必须能手填地址 */
function openBackendModal() {
  const m = $("#backend-modal");
  $("#backend-url").value = savedBackend();
  const https = location.protocol === "https:";
  $("#backend-state").innerHTML = remote
    ? `当前：<b>已连接后端</b> ${remote}${backendInfo?.solver_ready
        ? `　求解器可用，库里 <b>${backendInfo.solved_spots}</b> 个已解局面`
        : "　求解器不可用"}`
    : "当前：<b>浏览器本地模式</b>（所有计算在本页跑，翻后只有启发式估算）";
  $("#backend-warn").innerHTML = https
    ? "⚠ 当前页面是 <b>HTTPS</b>，浏览器不允许它请求 <b>http://</b> 地址" +
      "（混合内容拦截），<b>localhost / 127.0.0.1 除外</b>。<br>" +
      "要用局域网后端，请直接打开后端自己的地址 —— 它同样托管这套界面。"
    : "当前页面是 HTTP，可以连任意 http 后端。";
  m.classList.remove("hidden");
}

$("#conn").onclick = openBackendModal;
$("#backend-close").onclick = () => $("#backend-modal").classList.add("hidden");
$("#backend-save").onclick = () => {
  try { localStorage.setItem(BACKEND_KEY, $("#backend-url").value.trim().replace(/\/$/, "")); } catch { /* 无痕模式 */ }
  location.reload();
};
$("#backend-clear").onclick = () => {
  try { localStorage.removeItem(BACKEND_KEY); } catch { /* 无痕模式 */ }
  location.reload();
};

(async () => {
  setStatus("探测后端…");
  remote = await probeRemote();
  try {
    const h = await api("/api/health");
    if (remote) {
      const solver = h.solver_ready
        ? `求解器就绪 · 库里 ${h.solved_spots} 个已解局面`
        : "求解器未编译";
      setStatus(`后端已连接 · ${solver}`, "ok");
      $("#conn").title = `${remote}\n点击可修改后端地址`;
    } else {
      setStatus(`浏览器模式 · 翻后只有启发式`, "warn");
      $("#conn").title =
        "所有计算都在你的浏览器里跑，翻后没有 CFR 精确解。\n点击设置后端地址。";
    }
  } catch (e) {
    setStatus("初始化失败: " + e.message, "err");
  }
})();
