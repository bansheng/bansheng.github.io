/* 纯前端后端 —— 实现和 FastAPI 完全相同的路由，用于 GitHub Pages 这类静态托管。
 *
 * app.js 不知道自己在跟谁说话：先探测本地 8848 有没有真后端，有就用真的
 * （能跑 CFR solver、SQLite 存全局手牌库），没有就落到这里。
 *
 * 和真后端的差别，UI 会如实显示：
 *   - 没有 solver：翻后只有启发式估算，拿不到精确解
 *   - 存储是 localStorage 而不是 SQLite：只留在这台机器的这个浏览器里
 */

import { Range, ALL_LABELS, cardsFromStr, holeLabel, handVsRange } from "./poker.js";
import { HandState } from "./engine.js";
import { advise, BotTable, freqGap, frequencyOf, isBlunder,
         inferVillainRangeDetailed } from "./brain.js";
import { analyse } from "./analysis.js";

const CHART_FILES = ["6max_100bb_cash", "rangeviewer_100bb", "hu_pushfold_nash"];
const STORE_KEY = "gto-trainer-v1";
const SUPPORTED_STACKS = [50, 100, 200];

/* ---------------- chart ---------------- */

function buildChart(raw) {
  const spots = {};
  for (const [key, spec] of Object.entries(raw.spots || {})) {
    const ranges = {};
    for (const [action, notation] of Object.entries(spec.actions || {})) {
      ranges[action] = Range.parse(notation);
    }
    const strategy = {};
    for (const label of ALL_LABELS) {
      const freqs = {};
      let used = 0;
      for (const [action, r] of Object.entries(ranges)) {
        const w = r.weight(label);
        if (w > 0) { freqs[action] = (freqs[action] || 0) + w; used += w; }
      }
      if (used > 1) { for (const k of Object.keys(freqs)) freqs[k] /= used; used = 1; }
      freqs.fold = (freqs.fold || 0) + Math.max(0, 1 - used);
      strategy[label] = freqs;
    }
    spots[key] = {
      key,
      description: spec.description || key,
      actions: spec.actions || {},
      ranges,
      strategy,
      percent: Object.fromEntries(
        Object.entries(ranges).map(([a, r]) => [a, Math.round(r.percent() * 100) / 100])
      ),
    };
  }
  return { ...raw, spots };
}

/* ---------------- 本地存储 ---------------- */

function loadStore() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY)) || { sessions: [], hands: [], decisions: [], nextId: 1 };
  } catch {
    return { sessions: [], hands: [], decisions: [], nextId: 1 };
  }
}
function saveStore(s) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(s)); } catch { /* 配额满了就不存 */ }
}

/* ---------------- 会话 ---------------- */

class LocalSession {
  constructor(id, config, chart, store) {
    this.id = id;
    this.config = config;
    this.chart = chart;
    this.store = store;
    this.handNo = 0;
    this.button = config.table_size - 1;
    this.stack = config.stack_bb * config.big_blind;
    this.hand = null;
    this.handRowId = null;
    this.pendingDecision = null;
    this.botTable = new BotTable(config.bots, chart);
  }

  get hero() { return this.config.hero_seat; }

  newHand() {
    if (this.hand && !this.hand.finished) throw new Error("finish the current hand first");
    this.handNo++;
    this.button = (this.button + 1) % this.config.table_size;
    this.hand = new HandState({
      numSeats: this.config.table_size,
      button: this.button,
      smallBlind: this.config.small_blind,
      bigBlind: this.config.big_blind,
      startingStacks: Array(this.config.table_size).fill(this.stack),
      names: Array.from({ length: this.config.table_size }, (_, i) =>
        i === this.hero ? "你" : (this.botTable.bots.get(i)?.config.name
          || `${this.botTable.bots.get(i)?.config.profile === "fish" ? "Fish" : "GTO"}-${i}`)),
    });
    this.handRowId = null;
    this.botTable.playUntilHuman(this.hand, [this.hero]);
    if (this.hand.finished) this._persistHand();
    return this.view();
  }

  advice() {
    const [a, report] = this.adviceAndAnalysis();
    return { ...a, analysis: report };
  }

  /* 建议和分析必须共用同一个对手范围估计 —— 分析里画的范围如果不是
     建议所依据的那个，那比不画还糟。 */
  adviceAndAnalysis() {
    if (!this.hand || this.hand.actor !== this.hero) throw new Error("not the hero's turn");
    const [villain, derivation] = inferVillainRangeDetailed(this.hand, this.hero, this.chart);
    const a = advise(this.hand, this.hero, this.chart, Math.random, villain);
    // chart 查表不产生胜率，但翻前的分析面板没有胜率就空了一半，
    // 所以这里补算一次蒙特卡洛。
    let equity = a.equity ?? null;
    if (equity == null) {
      try {
        equity = handVsRange(this.hand.seats[this.hero].hole, villain, this.hand.board, 4000).equity;
      } catch { equity = null; }
    }
    const report = analyse(this.hand, this.hero, villain, equity, derivation);
    return [a, report];
  }

  act(action, amount = 0) {
    if (!this.hand) throw new Error("no hand in progress");
    if (this.hand.finished) throw new Error("hand is already finished");
    if (this.hand.actor !== this.hero) throw new Error("not your turn");

    const [a, report] = this.adviceAndAnalysis();
    const street = this.hand.street;
    const position = this.hand.positionName(this.hero);
    const label = holeLabel(this.hand.seats[this.hero].hole);
    const potBb = this.hand.pot / this.hand.bigBlind;
    const facing = this.hand.currentBet - this.hand.seats[this.hero].committed > 0;

    const record = this.hand.act(action, amount);
    this.pendingDecision = {
      street, position, hand_label: label, spot: a.spot,
      facing_bet: facing, pot_bb: potBb,
      chosen: action, chosen_amount: record.amount,
      advice: a, analysis: report, chosen_freq: frequencyOf(a, action),
      freq_gap: freqGap(a, action), blunder: isBlunder(a, action),
    };

    // 先留一份引用：持久化会把 pendingDecision 清空，但调用方还要拿它显示反馈
    const decision = this.pendingDecision;

    if (!this.hand.finished) this.botTable.playUntilHuman(this.hand, [this.hero]);
    if (this.hand.finished) this._persistHand();
    else this._persistDecision();

    const out = this.view();
    out.last_decision = decision;
    return out;
  }

  _ensureHandRow() {
    if (this.handRowId === null) {
      this.handRowId = this.store.nextId++;
      this.store.hands.push({
        id: this.handRowId, session_id: this.id, hand_no: this.handNo,
        hero_position: this.hand.positionName(this.hero),
        hero_cards: this.hand.seats[this.hero].hole.map((c) =>
          "23456789TJQKA"[c >> 2] + "cdhs"[c & 3]).join(""),
        board: "", pot: this.hand.pot, hero_net: 0,
        went_to_showdown: 0, won: 0, snapshot: null,
      });
    }
    return this.handRowId;
  }

  _persistDecision() {
    const id = this._ensureHandRow();
    const d = this.pendingDecision;
    this.store.decisions.push({
      hand_id: id, session_id: this.id, street: d.street, position: d.position,
      hand_label: d.hand_label, spot: d.spot, chosen: d.chosen,
      advice_source: d.advice.source, confidence: d.advice.confidence,
      advised: Object.fromEntries(d.advice.actions.map((x) => [x.action, x.frequency])),
      chosen_freq: d.chosen_freq, freq_gap: d.freq_gap, blunder: d.blunder ? 1 : 0,
      equity: d.advice.equity ?? null,
    });
    this.pendingDecision = null;
    saveStore(this.store);
  }

  _persistHand() {
    if (this.pendingDecision) this._persistDecision();
    const id = this._ensureHandRow();
    const row = this.store.hands.find((h) => h.id === id);
    const net = this.hand.results.net[this.hero];
    Object.assign(row, {
      board: this.hand.results.board, pot: this.hand.buildPots().reduce((a, p) => a + p.amount, 0),
      hero_net: net, went_to_showdown: this.hand.results.showdown ? 1 : 0,
      won: net > 0 ? 1 : 0, snapshot: this.hand.snapshot(true, this.hero),
    });
    saveStore(this.store);
  }

  view() {
    if (!this.hand) return { session_id: this.id, config: this.config, hand: null, hand_no: this.handNo };
    const snap = this.hand.snapshot(this.hand.finished, this.hero);
    snap.hero_seat = this.hero;
    snap.your_turn = this.hand.actor === this.hero;
    snap.hand_label = holeLabel(this.hand.seats[this.hero].hole);
    return {
      session_id: this.id, hand_no: this.handNo, mode: this.config.mode,
      hand: snap, bots: this.botTable.asDict(),
    };
  }

  stats() {
    const hands = this.store.hands.filter((h) => h.session_id === this.id);
    const decs = this.store.decisions.filter((d) => d.session_id === this.id);
    const n = hands.length;
    const bb = this.config.big_blind;
    const net = hands.reduce((a, h) => a + h.hero_net, 0);
    const pf = decs.filter((d) => d.street === "preflop");
    const vp = pf.filter((d) => ["call", "bet", "raise"].includes(d.chosen)).length;
    const pfr = pf.filter((d) => ["bet", "raise"].includes(d.chosen)).length;
    const blunders = decs.filter((d) => d.blunder).length;
    const r2 = (x) => Math.round(x * 100) / 100;
    return {
      hands: n, net_chips: net, net_bb: r2(net / bb),
      bb_per_100: n ? r2((net / bb / n) * 100) : 0,
      wtsd_pct: n ? r2((100 * hands.filter((h) => h.went_to_showdown).length) / n) : 0,
      win_pct: n ? r2((100 * hands.filter((h) => h.won).length) / n) : 0,
      vpip_pct: pf.length ? r2((100 * vp) / pf.length) : 0,
      pfr_pct: pf.length ? r2((100 * pfr) / pf.length) : 0,
      decisions: decs.length,
      avg_freq_gap: decs.length ? r2(decs.reduce((a, d) => a + d.freq_gap, 0) / decs.length * 10000) / 10000 : 0,
      blunders,
      blunder_pct: decs.length ? r2((100 * blunders) / decs.length) : 0,
    };
  }
}

function groupLeaks(rows, keyFn, minCount = 1) {
  const g = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    if (!g.has(k)) g.set(k, []);
    g.get(k).push(r);
  }
  return [...g.entries()]
    .filter(([, v]) => v.length >= minCount)
    .map(([k, v]) => ({
      key: k, decisions: v.length,
      avg_freq_gap: Math.round((v.reduce((a, x) => a + x.freq_gap, 0) / v.length) * 10000) / 10000,
      blunders: v.filter((x) => x.blunder).length,
      blunder_pct: Math.round((100 * v.filter((x) => x.blunder).length) / v.length * 10) / 10,
    }))
    .sort((a, b) => b.avg_freq_gap - a.avg_freq_gap);
}

/* ---------------- 路由 ---------------- */

export class LocalBackend {
  constructor(chartBase = "./charts") {
    this.chartBase = chartBase;
    this.charts = null;
    this.sessions = new Map();
    this.nextSession = 1;
    this.store = loadStore();
  }

  async ready() {
    if (this.charts) return;
    this.charts = {};
    for (const name of CHART_FILES) {
      try {
        const raw = await (await fetch(`${this.chartBase}/${name}.json`)).json();
        this.charts[name] = buildChart(raw);
      } catch (e) {
        console.warn(`chart ${name} 未能加载:`, e.message);
      }
    }
  }

  async handle(path, opts = {}) {
    await this.ready();
    const body = opts.body ? JSON.parse(opts.body) : {};
    const [, , ...seg] = path.split("/");            // ["", "api", ...]

    if (seg[0] === "health")
      return { ok: true, local: true, supported_stacks: SUPPORTED_STACKS, charts: Object.keys(this.charts), live_sessions: this.sessions.size };

    if (seg[0] === "charts" && seg.length === 1)
      return { charts: Object.values(this.charts).map((c) => this._summary(c)) };

    if (seg[0] === "charts" && seg.length === 2)
      return this._summary(this.charts[seg[1]]);

    if (seg[0] === "charts" && seg[2] === "spots") {
      const chart = this.charts[seg[1]];
      const key = decodeURIComponent(seg.slice(3).join("/"));
      const spot = chart?.spots[key];
      if (!spot) throw new Error(`no spot ${key}`);
      return { ...spot, ranges: undefined, labels: ALL_LABELS, provenance: chart.provenance };
    }

    if (seg[0] === "sessions" && seg.length === 1) return this._createSession(body);

    if (seg[0] === "sessions" && seg.length >= 2) {
      const s = this.sessions.get(Number(seg[1]));
      if (!s) throw new Error(`no live session ${seg[1]}`);
      const sub = seg[2];
      if (!sub) return s.view();
      if (sub === "deal") return s.newHand();
      if (sub === "act") return s.act(body.action, body.amount || 0);
      if (sub === "advice") return s.advice();
      if (sub === "stats") return this._stats(s);
      if (sub === "hands") return { hands: this.store.hands.filter((h) => h.session_id === s.id).slice().reverse() };
    }

    if (seg[0] === "hands" && seg[1]) {
      const id = Number(seg[1]);
      const h = this.store.hands.find((x) => x.id === id);
      if (!h) throw new Error(`no hand ${id}`);
      return { ...h, decisions: this.store.decisions.filter((d) => d.hand_id === id) };
    }

    if (seg[0] === "drill") return this._drill(body);
    if (seg[0] === "equity") return this._equity(body);

    throw new Error(`本地版没有实现这个接口: ${path}`);
  }

  _summary(c) {
    if (!c) throw new Error("chart not found");
    return {
      name: c.name, label: c.label, table_size: c.table_size, stack_bb: c.stack_bb,
      open_size_bb: c.open_size_bb, provenance: c.provenance,
      spots: Object.values(c.spots).map((s) => ({
        key: s.key, description: s.description, percent: s.percent,
      })),
    };
  }

  _createSession(body) {
    const chartName = body.chart_name || CHART_FILES[0];
    const chart = this.charts[chartName];
    if (!chart) throw new Error(`chart ${chartName} 不可用`);
    const tableSize = body.table_size || 6;
    const heroSeat = body.hero_seat ?? 0;
    if (!SUPPORTED_STACKS.includes(body.stack_bb ?? 100))
      throw new Error(`stack_bb must be one of ${SUPPORTED_STACKS}`);
    const bots = (body.bots && body.bots.length)
      ? body.bots
      : Array.from({ length: tableSize }, (_, i) => i).filter((i) => i !== heroSeat)
          .map((seat) => ({ seat, profile: "gto" }));

    const config = {
      mode: body.mode || "full", table_size: tableSize,
      stack_bb: body.stack_bb ?? 100, small_blind: body.small_blind ?? 1,
      big_blind: body.big_blind ?? 2, hero_seat: heroSeat,
      bots, chart_name: chartName,
    };
    const id = this.nextSession++;
    const s = new LocalSession(id, config, chart, this.store);
    this.sessions.set(id, s);
    this.store.sessions.push({ id, ...config });
    saveStore(this.store);
    return s.view();
  }

  _stats(s) {
    const decs = this.store.decisions.filter((d) => d.session_id === s.id);
    return {
      summary: s.stats(),
      by_position: groupLeaks(decs, (d) => `${d.position}|${d.street}`).map((r) => ({
        position: r.key.split("|")[0], street: r.key.split("|")[1],
        decisions: r.decisions, avg_freq_gap: r.avg_freq_gap,
        blunders: r.blunders, blunder_pct: r.blunder_pct,
      })),
      by_hand: groupLeaks(decs, (d) => d.hand_label, 2)
        .slice(0, 30)
        .map((r) => ({ hand: r.key, decisions: r.decisions, avg_freq_gap: r.avg_freq_gap, blunders: r.blunders })),
    };
  }

  _drill(body) {
    const chart = this.charts[body.chart_name || CHART_FILES[0]];
    let keys = Object.keys(chart.spots);
    if (body.position) keys = keys.filter((k) => k.startsWith(`${body.position}|`));
    if (body.spot_kind === "rfi") keys = keys.filter((k) => k.endsWith("|rfi"));
    else if (body.spot_kind === "vs-open") keys = keys.filter((k) => k.includes("|vs-"));
    if (!keys.length) throw new Error("no spot matches that filter");

    const key = keys[Math.floor(Math.random() * keys.length)];
    const spot = chart.spots[key];
    const playable = ALL_LABELS.filter((lb) => 1 - (spot.strategy[lb].fold || 0) > 0);
    const label = playable.length && Math.random() < 0.75
      ? playable[Math.floor(Math.random() * playable.length)]
      : ALL_LABELS[Math.floor(Math.random() * ALL_LABELS.length)];
    return {
      spot: key, description: spot.description, hand: label,
      answer: spot.strategy[label], provenance: chart.provenance,
    };
  }

  _equity(body) {
    const board = body.board ? cardsFromStr(body.board) : [];
    const villain = Range.parse(body.villain);
    const heroCards = cardsFromStr(body.hero);
    if (heroCards.length !== 2) throw new Error("本地版的胜率计算目前只支持具体两张手牌 vs 范围");
    const r = handVsRange(heroCards, villain, board, Math.min(body.trials || 6000, 20000));
    return {
      hero: holeLabel(heroCards), villain: body.villain, board: body.board || "",
      equity: r.equity, win: r.win, tie: r.tie, lose: r.lose,
      trials: r.trials, exact: r.exact,
    };
  }
}
