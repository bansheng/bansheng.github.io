/* 牌、范围记法、胜率 —— 与后端 gto/cards.py、gto/ranges.py、gto/equity.py 同语义。
 * 保持同语义是硬要求：同一个范围字符串，两边必须展开成同一组具体组合，
 * 否则本地版和后端版会给出不同的建议。tests/test_js_core.py 逐条对撞。 */

import { evaluate } from "./evaluator.js?v=2eea846f8e";

export const RANKS = "23456789TJQKA";
export const SUITS = "cdhs";
export const GRID_RANKS = "AKQJT98765432";

const RANK_INDEX = Object.fromEntries([...RANKS].map((r, i) => [r, i]));
const SUIT_INDEX = Object.fromEntries([...SUITS].map((s, i) => [s, i]));
const GRID_INDEX = Object.fromEntries([...GRID_RANKS].map((r, i) => [r, i]));

export const rankOf = (c) => c >> 2;
export const suitOf = (c) => c & 3;
export const makeCard = (r, s) => r * 4 + s;

export function cardFromStr(t) {
  const r = RANK_INDEX[t[0].toUpperCase()], s = SUIT_INDEX[t[1].toLowerCase()];
  if (r === undefined || s === undefined) throw new Error(`bad card ${t}`);
  return makeCard(r, s);
}
export const cardToStr = (c) => RANKS[rankOf(c)] + SUITS[suitOf(c)];

export function cardsFromStr(text) {
  const out = [];
  for (const blob of text.replace(/[,\-]/g, " ").split(/\s+/).filter(Boolean)) {
    if (blob.length % 2) throw new Error(`bad card sequence ${blob}`);
    for (let i = 0; i < blob.length; i += 2) out.push(cardFromStr(blob.slice(i, i + 2)));
  }
  return out;
}
export const cardsToStr = (cs) => cs.map(cardToStr).join("");

/* ---------------- 169 网格 ---------------- */

export function gridLabel(row, col) {
  if (row === col) return GRID_RANKS[row] + GRID_RANKS[row];
  const hi = Math.min(row, col), lo = Math.max(row, col);
  return GRID_RANKS[hi] + GRID_RANKS[lo] + (row < col ? "s" : "o");
}

export const ALL_LABELS = (() => {
  const out = [];
  for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) out.push(gridLabel(r, c));
  return out;
})();

export const comboCount = (label) =>
  label.length === 2 ? 6 : label[2] === "s" ? 4 : 12;

export function expandLabel(label) {
  const hi = RANK_INDEX[label[0].toUpperCase()];
  const lo = RANK_INDEX[label[1].toUpperCase()];
  const kind = (label[2] || "").toLowerCase();
  const out = [];
  const ord = (a, b) => (a > b ? [a, b] : [b, a]);
  if (hi === lo) {
    for (let s1 = 0; s1 < 4; s1++)
      for (let s2 = s1 + 1; s2 < 4; s2++) out.push([makeCard(hi, s2), makeCard(hi, s1)]);
  } else if (kind === "s") {
    for (let s = 0; s < 4; s++) out.push(ord(makeCard(hi, s), makeCard(lo, s)));
  } else {
    for (let s1 = 0; s1 < 4; s1++)
      for (let s2 = 0; s2 < 4; s2++) if (s1 !== s2) out.push(ord(makeCard(hi, s1), makeCard(lo, s2)));
  }
  return out;
}

export function holeLabel([a, b]) {
  const ra = a >> 2, rb = b >> 2;
  const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
  if (ra === rb) return RANKS[hi] + RANKS[hi];
  return RANKS[hi] + RANKS[lo] + ((a & 3) === (b & 3) ? "s" : "o");
}

/* ---------------- 范围记法解析 ---------------- */

const kindsOf = (k) => (k ? [k] : ["s", "o"]);

function normalise(body) {
  body = body.trim();
  if (body.length >= 3 && "soSO".includes(body[2]))
    return body.slice(0, 2).toUpperCase() + body[2].toLowerCase() + body.slice(3);
  return body.toUpperCase();
}

function labelsForPlus(base) {
  const hi0 = GRID_INDEX[base[0]], lo0 = GRID_INDEX[base[1]];
  if (hi0 === lo0) {
    const out = [];
    for (let i = hi0; i >= 0; i--) out.push(GRID_RANKS[i] + GRID_RANKS[i]);
    return out;
  }
  const hi = Math.min(hi0, lo0), lo = Math.max(hi0, lo0);
  const out = [];
  for (let k = lo; k > hi; k--)
    for (const kind of kindsOf(base[2] || "")) out.push(GRID_RANKS[hi] + GRID_RANKS[k] + kind);
  return out;
}

function labelsForSpan(start, end) {
  const sh = GRID_INDEX[start[0]], sl = GRID_INDEX[start[1]];
  const eh = GRID_INDEX[end[0]], el = GRID_INDEX[end[1]];
  const sk = start[2] || "", ek = end[2] || "";
  if ((sh === sl) !== (eh === el) || sk !== ek) throw new Error(`mismatched span ${start}-${end}`);
  const out = [];
  if (sh === sl) {
    const [a, b] = [Math.min(sh, eh), Math.max(sh, eh)];
    for (let i = a; i <= b; i++) out.push(GRID_RANKS[i] + GRID_RANKS[i]);
    return out;
  }
  if (Math.min(sh, sl) !== Math.min(eh, el)) throw new Error(`span ${start}-${end} must share a high card`);
  const hi = Math.min(sh, sl);
  const [a, b] = [Math.min(Math.max(sh, sl), Math.max(eh, el)), Math.max(Math.max(sh, sl), Math.max(eh, el))];
  for (let k = a; k <= b; k++)
    for (const kind of kindsOf(sk)) out.push(GRID_RANKS[hi] + GRID_RANKS[k] + kind);
  return out;
}

const CONCRETE = /^([2-9TJQKA][cdhs]){2}$/i;

export class Range {
  constructor(grid = {}, extra = new Map()) {
    this.grid = grid;
    this.extra = extra;
  }

  static parse(text) {
    const grid = {};
    const extra = new Map();
    for (const raw of text.trim().split(/[,\s]+/).filter(Boolean)) {
      const m = raw.match(/^([^:]+?)(?::([01](?:\.\d+)?|\.\d+))?$/);
      if (!m) throw new Error(`cannot parse token ${raw}`);
      let body = m[1];
      const weight = m[2] !== undefined ? parseFloat(m[2]) : 1.0;

      if (CONCRETE.test(body)) {
        const a = cardFromStr(body.slice(0, 2)), b = cardFromStr(body.slice(2));
        if (a === b) throw new Error(`duplicate card in ${raw}`);
        extra.set(a > b ? `${a},${b}` : `${b},${a}`, weight);
        continue;
      }
      body = normalise(body);
      let labels;
      if (body.includes("-")) {
        const [s, e] = body.split("-");
        labels = labelsForSpan(normalise(s), normalise(e));
      } else if (body.endsWith("+")) {
        labels = labelsForPlus(body.slice(0, -1));
      } else if (body.length === 2 && body[0] !== body[1]) {
        labels = [body + "s", body + "o"];
      } else {
        labels = [body];
      }
      for (const lb of labels) {
        if (GRID_INDEX[lb[0]] === undefined || GRID_INDEX[lb[1]] === undefined)
          throw new Error(`cannot parse token ${raw}`);
        grid[lb] = weight;
      }
    }
    return new Range(grid, extra);
  }

  weight(label) { return this.grid[label] || 0; }

  /** 展开成具体组合。dead 是 52 位死牌掩码（用 BigInt 避免 32 位溢出）。 */
  combos(dead = 0n) {
    const out = [];
    for (const [label, w] of Object.entries(this.grid)) {
      if (w <= 0) continue;
      for (const [a, b] of expandLabel(label)) {
        if (dead && ((dead >> BigInt(a)) & 1n || (dead >> BigInt(b)) & 1n)) continue;
        out.push([a, b, w]);
      }
    }
    for (const [key, w] of this.extra) {
      const [a, b] = key.split(",").map(Number);
      if (w <= 0) continue;
      if (dead && ((dead >> BigInt(a)) & 1n || (dead >> BigInt(b)) & 1n)) continue;
      out.push([a, b, w]);
    }
    return out;
  }

  comboWeight() {
    let t = 0;
    for (const [lb, w] of Object.entries(this.grid)) if (w > 0) t += w * comboCount(lb);
    for (const w of this.extra.values()) if (w > 0) t += w;
    return t;
  }

  percent() { return (100 * this.comboWeight()) / 1326; }

  toMatrix() {
    const m = [];
    for (let r = 0; r < 13; r++) {
      const row = [];
      for (let c = 0; c < 13; c++) row.push(this.grid[gridLabel(r, c)] || 0);
      m.push(row);
    }
    return m;
  }
}

export function cardMask(cards) {
  let m = 0n;
  for (const c of cards) m |= 1n << BigInt(c);
  return m;
}

/* ---------------- 胜率 ---------------- */

function shuffleTake(pool, n, rand) {
  // 部分 Fisher-Yates：只洗前 n 个位置，避免每次都洗整副牌
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(rand() * (pool.length - i));
    const t = pool[i]; pool[i] = pool[j]; pool[j] = t;
  }
  return pool.slice(0, n);
}

/** 具体两手牌对抗。board 不足 5 张时用蒙特卡洛；剩余 runout 少时穷举。 */
export function handVsHand(hero, villain, board = [], trials = 8000, rand = Math.random) {
  const known = [...hero, ...villain, ...board];
  if (new Set(known).size !== known.length) throw new Error("duplicate card");
  const dead = new Set(known);
  const pool = [];
  for (let c = 0; c < 52; c++) if (!dead.has(c)) pool.push(c);
  const need = 5 - board.length;

  if (need === 0 || (need === 1 && pool.length <= 46) || need === 2) {
    // 穷举：turn(44) 和 flop(C(45,2)=990) 都很便宜
    let w = 0, t = 0, l = 0;
    const walk = (start, chosen) => {
      if (chosen.length === need) {
        const full = board.concat(chosen);
        const h = evaluate(hero.concat(full)), v = evaluate(villain.concat(full));
        if (h > v) w++; else if (h === v) t++; else l++;
        return;
      }
      for (let i = start; i < pool.length; i++) walk(i + 1, chosen.concat(pool[i]));
    };
    walk(0, []);
    const n = w + t + l;
    return { win: w / n, tie: t / n, lose: l / n, equity: (w + t / 2) / n, trials: n, exact: true };
  }

  let w = 0, t = 0, l = 0;
  const work = pool.slice();
  for (let i = 0; i < trials; i++) {
    const full = board.concat(shuffleTake(work, need, rand));
    const h = evaluate(hero.concat(full)), v = evaluate(villain.concat(full));
    if (h > v) w++; else if (h === v) t++; else l++;
  }
  return { win: w / trials, tie: t / trials, lose: l / trials, equity: (w + t / 2) / trials, trials, exact: false };
}

/** 一手具体牌对一个范围。会先剔除被自己手牌/公共牌挡住的组合（blocker）。 */
export function handVsRange(hero, range, board = [], trials = 6000, rand = Math.random) {
  const dead = cardMask([...hero, ...board]);
  const combos = range.combos(dead);
  if (!combos.length) throw new Error("对手范围在这个牌面上是空的");

  const cum = [];
  let total = 0;
  for (const c of combos) { total += c[2]; cum.push(total); }
  const pick = () => {
    const x = rand() * total;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < x) lo = mid + 1; else hi = mid; }
    return combos[lo];
  };

  const need = 5 - board.length;
  const blocked = new Set([...hero, ...board]);

  if (need === 0) {
    let w = 0, t = 0, l = 0, tw = 0;
    const h = evaluate(hero.concat(board));
    for (const [a, b, wt] of combos) {
      const v = evaluate([a, b].concat(board));
      tw += wt;
      if (h > v) w += wt; else if (h === v) t += wt; else l += wt;
    }
    return { win: w / tw, tie: t / tw, lose: l / tw, equity: (w + t / 2) / tw, trials: combos.length, exact: true };
  }

  let w = 0, t = 0, l = 0;
  for (let i = 0; i < trials; i++) {
    const [va, vb] = pick();
    const pool = [];
    for (let c = 0; c < 52; c++) if (!blocked.has(c) && c !== va && c !== vb) pool.push(c);
    const full = board.concat(shuffleTake(pool, need, rand));
    const h = evaluate(hero.concat(full)), v = evaluate([va, vb].concat(full));
    if (h > v) w++; else if (h === v) t++; else l++;
  }
  return { win: w / trials, tie: t / trials, lose: l / trials, equity: (w + t / 2) / trials, trials, exact: false };
}

export function winners(holeBySeat, board) {
  const ranks = {};
  for (const [seat, hole] of Object.entries(holeBySeat)) ranks[seat] = evaluate(hole.concat(board));
  const best = Math.max(...Object.values(ranks));
  return { ranks, winners: Object.keys(ranks).filter((s) => ranks[s] === best).map(Number) };
}
