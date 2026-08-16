/* 纯前端后端 —— 实现和 FastAPI 完全相同的路由，用于 GitHub Pages 这类静态托管。
 *
 * app.js 不知道自己在跟谁说话：先探测本地 8848 有没有真后端，有就用真的
 * （能跑 CFR solver、SQLite 存全局手牌库），没有就落到这里。
 *
 * 和真后端的差别，UI 会如实显示：
 *   - 没有 solver：翻后只有启发式估算，拿不到精确解
 *   - 存储是 localStorage 而不是 SQLite：只留在这台机器的这个浏览器里
 */

import { Range, ALL_LABELS, cardsFromStr, holeLabel, handVsRange } from "./poker.js?v=18362dd4ad";
import { HandState } from "./engine.js?v=18362dd4ad";
import { advise, BotTable, freqGap, frequencyOf, isBlunder,
         inferVillainRangeDetailed, inferHeroRange } from "./brain.js?v=18362dd4ad";
import { analyse } from "./analysis.js?v=18362dd4ad";
import { SolveLibrary } from "./solve-library.js?v=18362dd4ad";
import { rangeReport } from "./rangereport.js?v=18362dd4ad";

const CHART_FILES = ["6max_100bb_cash", "rangeviewer_100bb", "hu_pushfold_nash"];
const STORE_KEY = "gto-trainer-v1";
/* 浏览器模式下记录也按用户分开存，并且**逐手写回** ——
   刷新一次就全没了是原来最伤的问题：练了半小时的统计说没就没。 */
// 深筹码给常规打法；短的这几档存在是因为单挑推-弃 Nash 解只在这些深度有定义
const SUPPORTED_STACKS = [5, 8, 10, 12, 15, 20, 25, 50, 100, 200];
// 短深度是练习深度（推-弃 Nash 解就在那儿），默认每手重置；
// 深筹码默认按真实现金局走：筹码延续、破产才补。
const DRILL_DEPTHS = [5, 8, 10, 12, 15, 20, 25];
const DEFAULT_BUYIN_BUDGET = 2000;

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
  const empty = { sessions: [], hands: [], decisions: [], users: [], nextId: 1 };
  try {
    return { ...empty, ...(JSON.parse(localStorage.getItem(STORE_KEY)) || {}) };
  } catch {
    return empty;
  }
}
function saveStore(s) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(s)); } catch { /* 配额满了就不存 */ }
}

/* ---------------- 会话 ---------------- */

class LocalSession {
  constructor(id, config, chart, store, pushfoldChart = null) {
    this.id = id;
    this.config = config;
    this.chart = chart;
    this.store = store;
    this.pushfoldChart = pushfoldChart;
    this.handNo = 0;
    this.button = config.table_size - 1;
    this.buyin = config.stack_bb * config.big_blind;
    // 实时筹码，cash 模式下逐手延续
    this.stacks = Array(config.table_size).fill(this.buyin);
    this.heroBoughtIn = this.buyin;   // 只计量英雄的买入，bot 是练习道具
    this.rebuys = 0;
    this.busted = false;
    this.hand = null;
    this.handRowId = null;
    this.pendingDecision = null;
    this.botTable = new BotTable(config.bots, chart, Math.random, pushfoldChart);
  }

  get hero() { return this.config.hero_seat; }

  get minPlayable() { return this.config.big_blind; }

  /* 为下一手准备筹码。
     fixed 模式每手重置 —— 推-弃练习就该是固定深度。
     cash 模式筹码延续，只有连盲注都下不起的座位才补 —— 把还能打的筹码
     自动补满正是这次要修掉的 bug：那样每手都是 100bb，筹码深度形同虚设。 */
  _prepareStacks() {
    if (this.config.chip_mode === "fixed") {
      this.stacks = Array(this.config.table_size).fill(this.buyin);
      return;
    }
    for (let i = 0; i < this.config.table_size; i++) {
      if (this.stacks[i] >= this.minPlayable) continue;
      if (i === this.hero) {
        if (!this._chargeRebuy()) { this.busted = true; return; }
      } else {
        this.stacks[i] = this.buyin;       // bot 永远坐得回来
      }
    }
  }

  _chargeRebuy(target = null) {
    const goal = target === null ? this.buyin : target;
    const need = goal - this.stacks[this.hero];
    if (need <= 0) return true;
    const remaining = this.config.buyin_budget - this.heroBoughtIn;
    if (remaining <= 0) return false;
    const take = Math.min(need, remaining);
    this.stacks[this.hero] += take;
    this.heroBoughtIn += take;
    this.rebuys++;
    return true;
  }

  /* 手动补码，只能补到买入线 —— 现金局不让你坐得比桌上限还深，
     允许的话等于悄悄改了正在训练的那个游戏。 */
  rebuy() {
    if (this.config.chip_mode !== "cash") throw new Error("固定筹码模式下不需要补码");
    if (this.hand && !this.hand.finished) throw new Error("这手牌还没打完，不能补码");
    if (this.stacks[this.hero] >= this.buyin)
      throw new Error(`你现在有 ${this.stacks[this.hero]}，已经不低于买入线 ${this.buyin}，补不了`);
    if (!this._chargeRebuy()) throw new Error("买入额度已经用完");
    return this.chips();
  }

  chips() {
    const hs = this.stacks[this.hero];
    const bb = this.config.big_blind;
    const r2 = (x) => Math.round(x * 100) / 100;
    return {
      chip_mode: this.config.chip_mode,
      buyin: this.buyin,
      buyin_bb: this.config.stack_bb,
      stacks: this.stacks.slice(),
      hero_stack: hs,
      hero_stack_bb: Math.round((hs / bb) * 10) / 10,
      hero_bought_in: this.heroBoughtIn,
      buyin_budget: this.config.buyin_budget,
      budget_left: Math.max(0, this.config.buyin_budget - this.heroBoughtIn),
      rebuys: this.rebuys,
      net: hs - this.heroBoughtIn,
      net_bb: r2((hs - this.heroBoughtIn) / bb),
      can_rebuy: this.config.chip_mode === "cash" && hs < this.buyin
                 && this.heroBoughtIn < this.config.buyin_budget,
      busted: this.busted,
    };
  }

  newHand() {
    if (this.hand && !this.hand.finished) throw new Error("finish the current hand first");
    this._prepareStacks();
    if (this.busted)
      throw new Error(`买入额度用完了（已买入 ${this.heroBoughtIn}，上限 ${this.config.buyin_budget}）。这一局到此为止。`);
    this.handNo++;
    this.button = (this.button + 1) % this.config.table_size;
    this.hand = new HandState({
      numSeats: this.config.table_size,
      button: this.button,
      smallBlind: this.config.small_blind,
      bigBlind: this.config.big_blind,
      startingStacks: this.stacks.slice(),
      names: Array.from({ length: this.config.table_size }, (_, i) =>
        i === this.hero ? "你" : (this.botTable.bots.get(i)?.config.name
          || `${this.botTable.bots.get(i)?.config.profile === "fish" ? "Fish" : "GTO"}-${i}`)),
    });
    this.handRowId = null;
    this.botTable.playUntilHuman(this.hand, [this.hero]);
    if (this.hand.finished) this._persistHand();
    return this.view();
  }

  async advice() {
    const [a, report] = await this.adviceAndAnalysis();
    return { ...a, analysis: report };
  }

  /* 建议和分析必须共用同一个对手范围估计 —— 分析里画的范围如果不是
     建议所依据的那个，那比不画还糟。 */
  async adviceAndAnalysis() {
    if (!this.hand || this.hand.actor !== this.hero) throw new Error("not the hero's turn");
    const [villain, derivation] = inferVillainRangeDetailed(this.hand, this.hero, this.chart);
    const a = advise(this.hand, this.hero, this.chart, Math.random, villain,
                     this.pushfoldChart);
    // chart 查表不产生胜率，但翻前的分析面板没有胜率就空了一半，
    // 所以这里补算一次蒙特卡洛。
    let equity = a.equity ?? null;
    if (equity == null) {
      try {
        equity = handVsRange(this.hand.seats[this.hero].hole, villain, this.hand.board, 4000).equity;
      } catch { equity = null; }
    }
    const report = analyse(this.hand, this.hero, villain, equity, derivation,
                           inferHeroRange(this.hand, this.hero, this.chart));

    // 有导出的真解就用真解 —— 精确到你手上这两张具体的牌
    if (this.library && this.hand.board.length >= 3) {
      let read = null;
      try { read = await this.library.read(this.hand, this.hero); } catch { read = null; }
      if (read?.actions) {
        const merged = new Map();
        read.actions.forEach((label, i) => {
          const [verb, amt] = label.split(/\s+/);
          const kind = { FOLD: "fold", CHECK: "check", CALL: "call",
                         BET: "bet", RAISE: "raise" }[verb];
          if (!kind || read.frequencies[i] <= 0.0005) return;
          const row = merged.get(kind) || { action: kind, frequency: 0, sizes: [] };
          row.frequency += read.frequencies[i];
          if (amt !== undefined) row.sizes.push({ to: parseFloat(amt), frequency: read.frequencies[i] });
          merged.set(kind, row);
        });
        const actions = [...merged.values()].map((r) => ({
          ...r,
          frequency: Math.round(r.frequency * 1e4) / 1e4,
          note: r.sizes.length
            ? r.sizes.map((s) => `到 ${s.to}（${Math.round(s.frequency * 100)}%）`).join("／") : "",
        }));
        if (actions.length) {
          return [{
            source: "solver",
            confidence: "exact",
            spot: `solver:${read.entry.board}`,
            hand: read.combo,
            actions,
            equity: a.equity ?? equity,
            explanation: `翻后精确解 · 路径 ${read.path.join(" → ") || "（本街第一个决策）"}`,
            caveat: `这是**离线预解的真解**（可利用度 ${read.entry.exploitability}）。`
              + (read.permuted ? `本局面与已解的 ${read.entry.board} 花色同构，换个花色名称就是同一个博弈。` : "")
              + "转牌之后没有导出真解，会退回启发式。",
          }, report];
        }
      } else if (read?.unavailable) {
        a.caveat += `（这个牌面有预解，但用不上：${read.unavailable}）`;
      }
    }
    return [a, report];
  }

  async act(action, amount = 0) {
    if (!this.hand) throw new Error("no hand in progress");
    if (this.hand.finished) throw new Error("hand is already finished");
    if (this.hand.actor !== this.hero) throw new Error("not your turn");

    const [a, report] = await this.adviceAndAnalysis();
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

  restore(state) {
    if (!state) return;
    if (state.stacks?.length === this.config.table_size) this.stacks = state.stacks.slice();
    this.heroBoughtIn = state.hero_bought_in ?? this.heroBoughtIn;
    this.rebuys = state.rebuys ?? 0;
    this.handNo = state.hand_no ?? 0;
    this.button = state.button ?? this.button;
  }

  _persistHand() {
    this.stacks = this.hand.seats.map((s) => s.stack);   // 把筹码带回会话
    const sessionRow = (this.store.sessions || []).find((x) => x.id === this.id);
    if (sessionRow) {
      sessionRow.chip_state = {
        stacks: this.stacks.slice(), hero_bought_in: this.heroBoughtIn,
        rebuys: this.rebuys, hand_no: this.handNo, button: this.button,
      };
    }
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
    if (!this.hand) return { session_id: this.id, config: this.config, hand: null,
                             hand_no: this.handNo, chips: this.chips() };
    const snap = this.hand.snapshot(this.hand.finished, this.hero);
    snap.hero_seat = this.hero;
    snap.your_turn = this.hand.actor === this.hero;
    snap.hand_label = holeLabel(this.hand.seats[this.hero].hole);
    return {
      session_id: this.id, hand_no: this.handNo, mode: this.config.mode,
      hand: snap, bots: this.botTable.asDict(), chips: this.chips(),
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
  constructor(chartBase = "./charts", solveBase = "./solves") {
    this.chartBase = chartBase;
    // 公网版也能吃到真解：翻牌街的解已导出成静态文件，按牌面懒加载
    this.library = new SolveLibrary(solveBase);
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

    if (seg[0] === "users" && seg.length === 1 && !opts.method)
      return { users: this.store.users || [] };

    if (seg[0] === "users" && seg[1] === "login")
      return this._login(body.name);

    if (seg[0] === "users" && seg.length === 3 && seg[2] === "sessions")
      return this._login(decodeURIComponent(seg[1]), false);

    // 跨这个用户的所有对局：翻前漏洞是长期问题，只看当前一局每个位置都
    // 凑不够样本，报告只会一片"样本太小"。
    if (seg[0] === "users" && seg.length === 3 && seg[2] === "rangereport") {
      const name = decodeURIComponent(seg[1]);
      const u = (this.store.users || []).find((x) => x.name === name);
      if (!u) throw new Error(`没有这个用户: ${name}`);
      const ids = new Set((this.store.sessions || [])
        .filter((s) => s.user_id === u.id).map((s) => s.id));
      return rangeReport(this.store.decisions.filter((d) => ids.has(d.session_id)),
                         this.charts["6max_100bb_cash"]);
    }

    if (seg[0] === "health") {
      await this.library.ready();
      return {
        ok: true, local: true, supported_stacks: SUPPORTED_STACKS,
        charts: Object.keys(this.charts), live_sessions: this.sessions.size,
        solved_spots: this.library.size,
        solver_ready: false,          // 不能现算，只能查已导出的
      };
    }

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
      if (sub === "act") return await s.act(body.action, body.amount || 0);
      if (sub === "advice") return await s.advice();
      if (sub === "rebuy") return s.rebuy();
      if (sub === "chips") return s.chips();
      if (sub === "stats") return this._stats(s);
      if (sub === "rangereport")
        return rangeReport(this.store.decisions.filter((d) => d.session_id === s.id),
                           this.charts["6max_100bb_cash"]);
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

  /* 没有密码 —— 本地存储里的用户名只是记录的归属标签 */
  _login(name, touch = true) {
    name = String(name || "").trim();
    if (!name) throw new Error("用户名不能为空");
    this.store.users = this.store.users || [];
    let u = this.store.users.find((x) => x.name === name);
    if (!u) {
      u = { id: this.store.nextId++, name, created_at: new Date().toISOString() };
      this.store.users.push(u);
    }
    if (touch) { u.last_seen = new Date().toISOString(); saveStore(this.store); }
    const mySessions = (this.store.sessions || []).filter((s) => s.user_id === u.id);
    const ids = new Set(mySessions.map((s) => s.id));
    const myHands = this.store.hands.filter((h) => ids.has(h.session_id));
    const myDecs = this.store.decisions.filter((d) => ids.has(d.session_id));
    u.hands = myHands.length;
    u.sessions = mySessions.length;
    const pf = myDecs.filter((d) => d.street === "preflop");
    const r1 = (x) => Math.round(x * 10) / 10;
    const net = myHands.reduce((a, h) => a + h.hero_net, 0);
    return {
      user: u,
      stats: {
        sessions: mySessions.length, hands: myHands.length, net_chips: net,
        decisions: myDecs.length,
        blunder_pct: myDecs.length
          ? r1((100 * myDecs.filter((d) => d.blunder).length) / myDecs.length) : 0,
        solver_pct: myDecs.length
          ? r1((100 * myDecs.filter((d) => d.advice_source === "solver").length) / myDecs.length) : 0,
        vpip_pct: pf.length
          ? r1((100 * pf.filter((d) => ["call", "bet", "raise"].includes(d.chosen)).length) / pf.length) : 0,
        pfr_pct: pf.length
          ? r1((100 * pf.filter((d) => ["bet", "raise"].includes(d.chosen)).length) / pf.length) : 0,
        wtsd_pct: myHands.length
          ? r1((100 * myHands.filter((h) => h.went_to_showdown).length) / myHands.length) : 0,
        win_pct: myHands.length
          ? r1((100 * myHands.filter((h) => h.won).length) / myHands.length) : 0,
        avg_freq_gap: myDecs.length
          ? Math.round((myDecs.reduce((a, d) => a + d.freq_gap, 0) / myDecs.length) * 1e4) / 1e4 : 0,
        blunders: myDecs.filter((d) => d.blunder).length,
        solver_graded: myDecs.filter((d) => d.advice_source === "solver").length,
      },
      sessions: mySessions.slice().reverse().map((s) => ({
        ...s,
        hands: this.store.hands.filter((h) => h.session_id === s.id).length,
        net: this.store.hands.filter((h) => h.session_id === s.id)
          .reduce((a, h) => a + h.hero_net, 0),
      })),
    };
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

    const stackBb = body.stack_bb ?? 100;
    const config = {
      mode: body.mode || "full", table_size: tableSize,
      stack_bb: stackBb, small_blind: body.small_blind ?? 1,
      big_blind: body.big_blind ?? 2, hero_seat: heroSeat,
      bots, chart_name: chartName,
      chip_mode: body.chip_mode || (DRILL_DEPTHS.includes(stackBb) ? "fixed" : "cash"),
      buyin_budget: body.buyin_budget ?? DEFAULT_BUYIN_BUDGET,
    };
    const id = this.nextSession++;
    const s = new LocalSession(id, config, chart, this.store,
                               this.charts["hu_pushfold_nash"] || null);
    s.library = this.library;
    const user = body.user ? this._login(body.user).user : null;
    s.userId = user ? user.id : null;
    if (body.resume_from) {
      const prev = (this.store.sessions || []).find((x) => x.id === body.resume_from);
      if (prev?.chip_state) s.restore(prev.chip_state);
    }
    this.sessions.set(id, s);
    this.store.sessions.push({ id, user_id: s.userId, ...config });
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
      answer: spot.strategy[label],
      // 按钮取自这个**局面**有哪些动作，而不是这手牌恰好有哪些 ——
      // chart 里 100% 弃牌的手牌否则只会显示一个"弃牌"按钮，
      // 你不可能选错，也就什么都没练到。
      spot_actions: ["fold", ...Object.keys(spot.ranges).filter((a) => a !== "fold")],
      provenance: chart.provenance,
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
