/* 建议引擎 + 对手 bot —— 与后端 gto/advisor.py、gto/bots.py 同策略。
 *
 * 三档建议来源，UI 上必须如实标注，不能混为一谈：
 *   chart      翻前查表。内置那套标了「参考近似」，就不能显示成精确解。
 *   heuristic  翻后的胜率 + 底池赔率模型。**不是 GTO**，只是能在 100ms 内给个答案。
 *   solver     真 CFR 解，只有本地后端跑得动，静态版拿不到。
 */

import { Range, handVsRange, holeLabel, cardMask } from "./poker.js";
import { evaluate } from "./evaluator.js";

export const DEFAULT_VILLAIN = Range.parse(
  "22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,87s,76s,65s,A7o+,KTo+,QJo"
);

const ACTION_CN = { fold: "弃牌", check: "过牌", call: "跟注", bet: "下注", raise: "加注" };

/* ---------------- 翻前查表 ---------------- */

function preflopContext(state) {
  const raises = state.actions.filter(
    (a) => a.street === "preflop" && (a.type === "bet" || a.type === "raise")
  );
  if (!raises.length) return { openerPos: null, raiseCount: 0, openerSeat: null };
  return {
    openerPos: state.positionName(raises[0].seat),
    raiseCount: raises.length,
    openerSeat: raises[0].seat,
  };
}

export function spotKey(heroPos, openerPos) {
  return openerPos === null ? `${heroPos}|rfi` : `${heroPos}|vs-${openerPos}-open`;
}

export function preflopAdvice(state, seat, chart) {
  const heroPos = state.positionName(seat);
  const { openerPos, raiseCount } = preflopContext(state);
  if (raiseCount > 1) return null;
  const key = spotKey(heroPos, openerPos);
  const spot = chart.spots[key];
  if (!spot) return null;

  const hand = holeLabel(state.seats[seat].hole);
  const freqs = spot.strategy[hand] || { fold: 1 };
  const approximate = chart.provenance?.kind !== "solver";

  return {
    source: "chart",
    confidence: approximate ? "approximate" : "exact",
    spot: key,
    hand,
    actions: Object.entries(freqs)
      .filter(([, f]) => f > 0)
      .map(([action, frequency]) => ({ action, frequency, note: "" })),
    explanation: `${heroPos} 位置，${spot.description}`,
    caveat: approximate
      ? "此 chart 标注为参考近似，范围宽度可信、单手牌混合频率不保证精确"
      : "",
  };
}

/* ---------------- 翻后启发式 ---------------- */

/** 按当前牌面的成手强度保留范围里最强的一部分。
 *  听牌会被低估（同花听牌算作高牌），所以这是「下注范围至少有多强」的下界。 */
export function narrowByStrength(base, board, dead, keepFraction) {
  const combos = base.combos(dead);
  if (!combos.length || board.length < 3) return base;
  const ranked = combos
    .map((c) => [c, evaluate([c[0], c[1]].concat(board))])
    .sort((a, b) => b[1] - a[1]);
  const keep = Math.max(1, Math.floor(ranked.length * keepFraction));
  const r = new Range({}, new Map());
  for (const [[a, b, w]] of ranked.slice(0, keep)) r.extra.set(`${a},${b}`, w);
  return r;
}

export function inferVillainRange(state, seat, chart) {
  const { openerPos, raiseCount } = preflopContext(state);
  const board = state.board;
  const dead = cardMask(board.concat(state.seats[seat].hole));

  let base;
  if (raiseCount >= 2) base = Range.parse("TT+,AJs+,KQs,AKo");
  else if (raiseCount === 1 && openerPos) {
    const spot = chart.spots[spotKey(openerPos, null)];
    base = spot?.actions?.raise ? Range.parse(spot.actions.raise) : DEFAULT_VILLAIN;
  } else base = DEFAULT_VILLAIN;

  let aggressor = null;
  for (let i = state.actions.length - 1; i >= 0; i--) {
    const a = state.actions[i];
    if (a.street === state.street && (a.type === "bet" || a.type === "raise")) { aggressor = a.seat; break; }
  }
  if (aggressor === null || aggressor === seat) return base;

  const bets = state.actions.filter(
    (a) => a.seat === aggressor && a.street !== "preflop" && (a.type === "bet" || a.type === "raise")
  ).length;
  if (bets <= 0) return base;
  const keep = bets === 1 ? 0.55 : bets === 2 ? 0.35 : 0.22;
  return narrowByStrength(base, board, dead, keep);
}

export function postflopAdvice(state, seat, villainRange, trials = 2500, rand = Math.random) {
  const hero = state.seats[seat];
  const board = state.board;
  const bb = state.bigBlind;
  const eq = handVsRange(hero.hole, villainRange || DEFAULT_VILLAIN, board, trials, rand).equity;
  const toCall = state.currentBet - hero.committed;
  const pot = state.pot;
  const required = toCall > 0 ? toCall / (pot + toCall) : 0;
  const legal = new Set(state.legalActions().map((l) => l.type));
  const actions = [];

  if (toCall > 0) {
    const evCall = (eq * pot - (1 - eq) * toCall) / bb;
    const margin = eq - required;
    let callF, raiseF, foldF;
    if (margin > 0.25) [callF, raiseF, foldF] = [0.62, 0.38, 0.0];
    else if (margin > 0.10) [callF, raiseF, foldF] = [0.86, 0.14, 0.0];
    else if (margin > 0.02) [callF, raiseF, foldF] = [0.92, 0.04, 0.04];
    else if (margin > -0.04) [callF, raiseF, foldF] = [0.45, 0.02, 0.53];
    else [callF, raiseF, foldF] = [0.02, eq < 0.15 ? 0.03 : 0.0, 0.95];

    actions.push({ action: "fold", frequency: foldF, ev_bb: 0, note: "弃牌 EV 归零，作为基准线" });
    actions.push({
      action: "call", frequency: callF, ev_bb: evCall,
      note: `需要 ${(required * 100).toFixed(1)}% 胜率才能跟注，你有 ${(eq * 100).toFixed(1)}%`,
    });
    if (legal.has("raise") && raiseF > 0)
      actions.push({ action: "raise", frequency: raiseF, note: "加注以获取价值或施压弃牌区" });
    else if (raiseF > 0) actions[1].frequency += raiseF;
  } else {
    let betF;
    if (eq > 0.72) betF = 0.82;
    else if (eq > 0.58) betF = 0.58;
    else if (eq > 0.45) betF = 0.28;
    else if (eq > 0.30) betF = 0.08;
    else betF = 0.10;
    const betKey = legal.has("bet") ? "bet" : "raise";
    actions.push({ action: "check", frequency: 1 - betF, ev_bb: 0, note: "过牌控池 / 保护过牌范围" });
    actions.push({
      action: betKey, frequency: betF,
      note: `对抗对手估计范围你有 ${(eq * 100).toFixed(1)}% 胜率`,
    });
  }

  const total = actions.reduce((a, x) => a + x.frequency, 0) || 1;
  for (const a of actions) a.frequency /= total;

  return {
    source: "heuristic",
    confidence: "heuristic",
    spot: `${state.street}|${state.positionName(seat)}`,
    hand: holeLabel(hero.hole),
    actions,
    equity: eq,
    pot_odds: pot && toCall ? toCall / pot : null,
    required_equity: toCall > 0 ? required : null,
    explanation: `${state.street} · 底池 ${(pot / bb).toFixed(1)}bb · 需跟 ${(toCall / bb).toFixed(1)}bb · 胜率 ${(eq * 100).toFixed(1)}%`,
    caveat: "这是基于胜率与底池赔率的启发式估算，**不是 GTO 解**。真实解需要 CFR 求解该局面，只有本地后端能跑。",
  };
}

export function advise(state, seat, chart, rand = Math.random) {
  seat = seat ?? state.actor;
  if (seat === null) throw new Error("no seat is to act");
  if (state.street === "preflop") {
    const a = preflopAdvice(state, seat, chart);
    if (a) return a;
    // chart 里没有这个局面（多次加注，或未收录的位置组合）。
    // 仍然给个按底池赔率的估算，但要说清楚是为什么。
    const fallback = postflopAdvice(state, seat, inferVillainRange(state, seat, chart), 2500, rand);
    fallback.caveat =
      "翻前的这个局面（多次加注或未收录的位置组合）不在 chart 里，" +
      "下面是按胜率和底池赔率算的估算，**不是 GTO 解**。";
    return fallback;
  }
  return postflopAdvice(state, seat, inferVillainRange(state, seat, chart), 2500, rand);
}

export function frequencyOf(advice, action) {
  const a = advice.actions.find((x) => x.action === action);
  return a ? a.frequency : 0;
}

export function freqGap(advice, chosen) {
  const best = advice.actions.reduce((a, b) => (b.frequency > a.frequency ? b : a), advice.actions[0]);
  if (!best) return 0;
  return Math.max(0, best.frequency - frequencyOf(advice, chosen));
}

export const isBlunder = (advice, chosen, threshold = 0.02) =>
  frequencyOf(advice, chosen) < threshold;

/* ---------------- Bot ---------------- */

const FISH_OPEN = Range.parse(
  "22+,A2s+,K2s+,Q4s+,J6s+,T6s+,95s+,85s+,74s+,64s+,54s,A2o+,K7o+,Q9o+,J9o+,T9o"
);
const FISH_CALL = Range.parse(
  "22+,A2s+,K2s+,Q2s+,J4s+,T5s+,95s+,85s+,74s+,63s+,53s+,43s,A2o+,K5o+,Q7o+,J8o+,T8o+,98o,87o"
);

export class Bot {
  constructor(config, chart, rand = Math.random) {
    this.config = { profile: "gto", aggression: 1.0, ...config };
    this.chart = chart;
    this.rand = rand;
  }

  decide(state) {
    return this.config.profile === "fish"
      ? this._fish(state, state.actor)
      : this._gto(state, state.actor);
  }

  _legal(state) {
    return Object.fromEntries(state.legalActions().map((l) => [l.type, l]));
  }

  _gto(state, seat) {
    const advice = advise(state, seat, this.chart, this.rand);
    const legal = this._legal(state);
    const weights = new Map();
    for (const item of advice.actions) {
      if (item.frequency <= 0) continue;
      let kind = item.action;
      if (!(kind in legal)) {
        kind = "call" in legal ? "call" : "check" in legal ? "check" : "fold";
      }
      const scale = kind === "bet" || kind === "raise" ? this.config.aggression : 1;
      weights.set(kind, (weights.get(kind) || 0) + item.frequency * scale);
    }
    if (!weights.size) return this._passive(legal);
    const kinds = [...weights.keys()];
    const total = kinds.reduce((a, k) => a + weights.get(k), 0);
    let x = this.rand() * total;
    let chosen = kinds[kinds.length - 1];
    for (const k of kinds) { x -= weights.get(k); if (x <= 0) { chosen = k; break; } }
    return [chosen, this._size(state, chosen, legal)];
  }

  _fish(state, seat) {
    const legal = this._legal(state);
    const hero = state.seats[seat];
    const toCall = state.currentBet - hero.committed;

    if (state.street === "preflop") {
      const label = holeLabel(hero.hole);
      if (toCall > 0) {
        if (["AA", "KK", "QQ", "AKs"].includes(label) && legal.raise)
          return ["raise", legal.raise.min];
        return FISH_CALL.weight(label) > 0 ? ["call", 0] : ["fold", 0];
      }
      if (FISH_OPEN.weight(label) > 0 && legal.bet) return ["bet", legal.bet.min];
      return legal.check ? ["check", 0] : ["fold", 0];
    }

    const eq = handVsRange(hero.hole, FISH_CALL, state.board, 800, this.rand).equity;
    if (toCall > 0) {
      if (state.street === "river" && eq < 0.45) return ["fold", 0];
      return eq > 0.30 ? ["call", 0] : ["fold", 0];
    }
    if (eq > 0.72 && legal.bet) return ["bet", this._size(state, "bet", legal, 0.5)];
    return legal.check ? ["check", 0] : this._passive(legal);
  }

  _passive(legal) {
    if (legal.check) return ["check", 0];
    if (legal.call) return ["call", 0];
    return ["fold", 0];
  }

  _size(state, kind, legal, fraction) {
    if (kind !== "bet" && kind !== "raise") return 0;
    const spec = legal[kind];
    let target;
    if (state.street === "preflop") {
      target = kind === "bet"
        ? Math.round(state.bigBlind * 2.5)
        : Math.round(state.currentBet * 3);
    } else {
      const fracs = [0.33, 0.5, 0.75];
      const frac = fraction ?? fracs[Math.floor(this.rand() * fracs.length)];
      target = state.currentBet + Math.round(state.pot * frac);
    }
    return Math.max(spec.min, Math.min(spec.max, target));
  }
}

export class BotTable {
  constructor(configs, chart, rand = Math.random) {
    this.bots = new Map(configs.map((c) => [c.seat, new Bot(c, chart, rand)]));
  }
  has(seat) { return this.bots.has(seat); }
  playUntilHuman(state, humanSeats, maxSteps = 200) {
    const humans = new Set(humanSeats);
    const taken = [];
    let steps = 0;
    while (!state.finished && state.actor !== null && !humans.has(state.actor)) {
      if (++steps > maxSteps) throw new Error("bot loop did not terminate");
      const bot = this.bots.get(state.actor);
      if (!bot) throw new Error(`seat ${state.actor} has neither a human nor a bot`);
      const [kind, amount] = bot.decide(state);
      taken.push(state.act(kind, amount));
    }
    return taken;
  }
  asDict() {
    return Object.fromEntries(
      [...this.bots].map(([seat, b]) => [seat, {
        profile: b.config.profile,
        name: b.config.name || `${b.config.profile === "gto" ? "GTO" : "Fish"}-${seat}`,
        aggression: b.config.aggression,
      }])
    );
  }
}

export { ACTION_CN };
