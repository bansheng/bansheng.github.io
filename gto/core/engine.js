/* 无限注德州扑克牌局引擎 —— 与后端 gto/engine.py 逐条对齐。
 *
 * 两边规则必须一模一样，否则你在公网版打的牌和本地版结算不同。
 * tests/test_js_engine.py 用同一副牌 + 同一串动作跑两边，逐步比对
 * 底池 / 筹码 / 合法动作 / 结算，任何一步不同就报错。
 *
 * 容易写错、这里显式实现的规则：
 *  - 最小加注 = 上一次加注幅度
 *  - **不完整全下不重开加注权**：短码 all-in 的加注幅度不足一个完整加注时，
 *    已经行动过的玩家只能跟或弃，不能再加
 *  - 边池分层，奇数筹码给按钮左手第一位赢家
 *  - 单挑时按钮=小盲，且翻前先行动
 */

import { cardToStr, cardsToStr, holeLabel } from "./poker.js?v=36e7fc8dc6";
import { evaluate, categoryName } from "./evaluator.js?v=36e7fc8dc6";

export const STREETS = ["preflop", "flop", "turn", "river"];
export const BOARD_SIZE = { preflop: 0, flop: 3, turn: 4, river: 5 };

const POSITIONS = {
  2: ["BTN/SB", "BB"],
  3: ["BTN", "SB", "BB"],
  4: ["BTN", "SB", "BB", "CO"],
  5: ["BTN", "SB", "BB", "UTG", "CO"],
  6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
  7: ["BTN", "SB", "BB", "UTG", "UTG+1", "HJ", "CO"],
  8: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
  9: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "MP+1", "HJ", "CO"],
};

/* xorshift128+，用来让同一个种子在两次运行里发出同一副牌。
 * 不追求密码学强度，只要可复现。 */
export function makeRng(seed) {
  let s0 = (seed ^ 0x9e3779b9) >>> 0 || 1;
  let s1 = (seed * 0x85ebca6b + 0xc2b2ae35) >>> 0 || 2;
  return function next() {
    let x = s0, y = s1;
    s0 = y;
    x ^= x << 23; x >>>= 0;
    x ^= x >>> 17;
    x ^= y ^ (y >>> 26); x >>>= 0;
    s1 = x;
    return ((s0 + s1) >>> 0) / 4294967296;
  };
}

export function shuffledDeck(seed) {
  const rand = makeRng(seed >>> 0);
  const deck = Array.from({ length: 52 }, (_, i) => i);
  for (let i = 51; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = deck[i]; deck[i] = deck[j]; deck[j] = t;
  }
  return deck;
}

export class IllegalAction extends Error {}

export class HandState {
  constructor(opts) {
    const {
      numSeats = 6, button = 0, smallBlind = 1, bigBlind = 2,
      startingStacks = null, ante = 0, seed = null, deck = null, names = [],
    } = opts || {};
    if (numSeats < 2) throw new Error("need at least 2 seats");

    this.numSeats = numSeats;
    this.button = ((button % numSeats) + numSeats) % numSeats;
    this.smallBlind = smallBlind;
    this.bigBlind = bigBlind;
    this.ante = ante;
    this.startingStacks = startingStacks || Array(numSeats).fill(bigBlind * 100);
    if (this.startingStacks.length !== numSeats)
      throw new Error("startingStacks length must equal numSeats");

    this.seats = Array.from({ length: numSeats }, (_, i) => ({
      index: i, stack: this.startingStacks[i], hole: [],
      committed: 0, totalCommitted: 0, folded: false, allIn: false,
      name: names[i] || `P${i}`,
    }));
    this.board = [];
    this.street = "preflop";
    this.actor = null;
    this.currentBet = 0;
    this.lastRaiseSize = 0;
    this.actions = [];
    this.deadMoney = 0;
    this.finished = false;
    this.results = null;
    this.deck = deck ? deck.slice() : shuffledDeck(seed ?? Math.floor(Math.random() * 2 ** 30));
    this._acted = new Set();

    this._postBlinds();
    this._dealHole();
    this._beginRound(this._firstPreflopActor());
  }

  /* ---------- 座位 ---------- */

  positionName(seat) {
    const table = POSITIONS[this.numSeats];
    if (!table) return `seat${seat}`;
    return table[((seat - this.button) % this.numSeats + this.numSeats) % this.numSeats];
  }
  seatAtOffset(o) { return (this.button + o) % this.numSeats; }
  _next(seat) { return (seat + 1) % this.numSeats; }
  canAct(s) { return !s.folded && !s.allIn && s.stack > 0; }
  get liveSeats() { return this.seats.filter((s) => !s.folded).map((s) => s.index); }
  get pot() { return this.deadMoney + this.seats.reduce((a, s) => a + s.committed, 0); }

  /* ---------- 开局 ---------- */

  _commit(seat, amount) {
    amount = Math.min(amount, seat.stack);
    seat.stack -= amount;
    seat.committed += amount;
    seat.totalCommitted += amount;
    if (seat.stack === 0) seat.allIn = true;
  }

  _postBlinds() {
    if (this.ante) {
      for (const s of this.seats) this._commit(s, Math.min(this.ante, s.stack));
      this.deadMoney += this.seats.reduce((a, s) => a + s.committed, 0);
      for (const s of this.seats) s.committed = 0;
    }
    const hu = this.numSeats === 2;
    const sb = this.seats[hu ? this.button : this.seatAtOffset(1)];
    const bb = this.seats[hu ? this.seatAtOffset(1) : this.seatAtOffset(2)];
    this._commit(sb, Math.min(this.smallBlind, sb.stack));
    this._commit(bb, Math.min(this.bigBlind, bb.stack));
    this.currentBet = Math.max(sb.committed, bb.committed);
    this.lastRaiseSize = this.bigBlind;
  }

  _dealHole() {
    for (const s of this.seats) if (!s.hole.length) s.hole = [this.deck.pop(), this.deck.pop()];
  }

  _firstPreflopActor() {
    const hu = this.numSeats === 2;
    const start = hu ? this.seatAtOffset(1) : this.seatAtOffset(2);
    let seat = this._next(start);
    for (let i = 0; i < this.numSeats; i++) {
      if (this.canAct(this.seats[seat])) return seat;
      seat = this._next(seat);
    }
    return null;
  }

  _firstPostflopActor() {
    let seat = this._next(this.button);
    for (let i = 0; i < this.numSeats; i++) {
      if (this.canAct(this.seats[seat])) return seat;
      seat = this._next(seat);
    }
    return null;
  }

  _bettingPossible() {
    const actionable = this.seats.filter((s) => this.canAct(s));
    if (actionable.length >= 2) return true;
    return actionable.length === 1 && actionable[0].committed < this.currentBet;
  }

  _beginRound(first) {
    this._acted = new Set();
    this.actor = first;
    if (this.actor === null || !this._bettingPossible()) this._advanceStreet();
  }

  /* ---------- 动作 ---------- */

  legalActions() {
    if (this.finished || this.actor === null) return [];
    const seat = this.seats[this.actor];
    const toCall = this.currentBet - seat.committed;
    const out = [];
    if (toCall > 0) {
      out.push({ type: "fold" });
      out.push({ type: "call", amount: Math.min(toCall, seat.stack) });
    } else {
      out.push({ type: "check" });
    }
    const maxTo = seat.committed + seat.stack;
    if (maxTo > this.currentBet && !this._acted.has(seat.index)) {
      const kind = this.currentBet === 0 ? "bet" : "raise";
      let minTo = this.currentBet
        ? Math.max(this.bigBlind, this.currentBet + this.lastRaiseSize)
        : this.bigBlind;
      minTo = Math.min(minTo, maxTo);   // 短码可以低于最小加注全下
      out.push({ type: kind, min: minTo, max: maxTo });
    }
    return out;
  }

  act(type, amount = 0) {
    if (this.finished) throw new IllegalAction("hand is already finished");
    if (this.actor === null) throw new IllegalAction("no seat is to act");
    const seat = this.seats[this.actor];
    const legal = Object.fromEntries(this.legalActions().map((l) => [l.type, l]));
    if (!(type in legal))
      throw new IllegalAction(`${type} not legal; legal: ${Object.keys(legal).join(",")}`);

    const toCall = this.currentBet - seat.committed;
    if (type === "fold") seat.folded = true;
    else if (type === "check") { /* nothing */ }
    else if (type === "call") this._commit(seat, Math.min(toCall, seat.stack));
    else {
      const spec = legal[type];
      if (amount < spec.min || amount > spec.max)
        throw new IllegalAction(`${type} to ${amount} outside [${spec.min}, ${spec.max}]`);
      const raiseSize = amount - this.currentBet;
      this._commit(seat, amount - seat.committed);
      if (raiseSize >= this.lastRaiseSize) {
        this.lastRaiseSize = raiseSize;
        this._acted = new Set();          // 完整加注 → 所有人重新获得行动权
      }
      this.currentBet = amount;
    }

    const record = {
      seat: seat.index, type, amount: seat.committed,
      street: this.street, all_in: seat.allIn,
    };
    this.actions.push(record);
    this._acted.add(seat.index);
    this._advanceAfterAction();
    return record;
  }

  _advanceAfterAction() {
    const live = this.liveSeats;
    if (live.length === 1) { this._settle(live[0]); return; }

    const pending = this.seats.filter(
      (s) => this.canAct(s) && (!this._acted.has(s.index) || s.committed < this.currentBet)
    );
    if (!pending.length) { this._advanceStreet(); return; }

    const pendingSet = new Set(pending.map((s) => s.index));
    let seat = this._next(this.actor);
    for (let i = 0; i < this.numSeats; i++) {
      if (pendingSet.has(seat)) { this.actor = seat; return; }
      seat = this._next(seat);
    }
    this._advanceStreet();
  }

  _collectStreet() {
    this.deadMoney += this.seats.reduce((a, s) => a + s.committed, 0);
    for (const s of this.seats) s.committed = 0;
    this.currentBet = 0;
    this.lastRaiseSize = this.bigBlind;
  }

  _advanceStreet() {
    this._collectStreet();
    if (this.street === "river") { this._settle(null); return; }
    const next = STREETS[STREETS.indexOf(this.street) + 1];
    this.street = next;
    while (this.board.length < BOARD_SIZE[next]) this.board.push(this.deck.pop());

    if (this.seats.filter((s) => this.canAct(s)).length < 2) {
      while (this.board.length < 5) this.board.push(this.deck.pop());
      this.street = "river";
      this._settle(null);
      return;
    }
    this._beginRound(this._firstPostflopActor());
  }

  /* ---------- 结算 ---------- */

  buildPots() {
    const levels = [...new Set(this.seats.filter((s) => s.totalCommitted > 0)
      .map((s) => s.totalCommitted))].sort((a, b) => a - b);
    const pots = [];
    let prev = 0;
    for (const level of levels) {
      let amount = 0;
      const eligible = [];
      for (const s of this.seats) {
        const take = Math.min(s.totalCommitted, level) - Math.min(s.totalCommitted, prev);
        if (take > 0) { amount += take; if (!s.folded) eligible.push(s.index); }
      }
      if (amount > 0) pots.push({ amount, eligible });
      prev = level;
    }
    return pots;
  }

  _settle(uncontested) {
    this._collectStreet();
    const pots = this.buildPots();
    const payouts = Object.fromEntries(this.seats.map((s) => [s.index, 0]));
    let rankings = {};

    if (uncontested !== null && uncontested !== undefined) {
      for (const p of pots) payouts[uncontested] += p.amount;
    } else {
      while (this.board.length < 5) this.board.push(this.deck.pop());
      this.street = "showdown";
      for (const s of this.seats) if (!s.folded) rankings[s.index] = evaluate(s.hole.concat(this.board));
      for (const p of pots) {
        const contenders = p.eligible.filter((i) => i in rankings);
        if (!contenders.length) continue;
        const best = Math.max(...contenders.map((i) => rankings[i]));
        const winners = contenders.filter((i) => rankings[i] === best);
        const share = Math.floor(p.amount / winners.length);
        const remainder = p.amount - share * winners.length;
        for (const i of winners) payouts[i] += share;
        const order = winners.slice().sort(
          (a, b) => ((a - this.button - 1 + 2 * this.numSeats) % this.numSeats)
                  - ((b - this.button - 1 + 2 * this.numSeats) % this.numSeats)
        );
        for (let i = 0; i < remainder; i++) payouts[order[i]] += 1;
      }
    }

    for (const s of this.seats) s.stack += payouts[s.index];
    this.deadMoney = 0;
    this.finished = true;
    this.actor = null;
    this.results = {
      payouts,
      net: Object.fromEntries(this.seats.map((s) => [s.index, s.stack - this.startingStacks[s.index]])),
      pots,
      rankings: Object.fromEntries(
        Object.entries(rankings).map(([i, r]) =>
          [i, { rank: r, category: categoryName(r), name: categoryName(r) }])
      ),
      showdown: uncontested === null || uncontested === undefined,
      board: cardsToStr(this.board),
    };
  }

  /* ---------- 序列化 ---------- */

  snapshot(reveal = false, viewer = null) {
    return {
      street: this.street,
      board: this.board.map(cardToStr),
      pot: this.pot,
      current_bet: this.currentBet,
      min_raise_size: this.lastRaiseSize,
      button: this.button,
      actor: this.actor,
      finished: this.finished,
      big_blind: this.bigBlind,
      small_blind: this.smallBlind,
      seats: this.seats.map((s) => ({
        index: s.index, name: s.name, position: this.positionName(s.index),
        stack: s.stack, committed: s.committed, total_committed: s.totalCommitted,
        folded: s.folded, all_in: s.allIn,
        hole: reveal || (viewer !== null && s.index === viewer) ? s.hole.map(cardToStr) : null,
        // 只给看得见牌的座位：一眼读出牌型，比每条街重新看五张牌快得多；
        // 摊牌时也是靠它直接看出谁赢，而不用自己解读牌面。
        made: (reveal || (viewer !== null && s.index === viewer))
              && this.board.length >= 3 && !s.folded
          ? categoryName(evaluate(s.hole.concat(this.board))) : null,
      })),
      legal_actions: this.legalActions(),
      actions: this.actions,
      results: this.results,
      hand_label: holeLabel(this.seats[viewer ?? 0].hole),
    };
  }
}
