/* 读牌：对手可能拿什么，各占多少，哪些能打走。
 * 与后端 gto/handread.py 同口径。
 *
 * 胜率数字告诉你这次跟注赚不赚，但教不会你读牌。能带到真牌桌上的是这句话：
 * 「他这条线大概 60 个组合，15 个打过我，20 个是听牌，25 个是我能打走的空气。」
 * 所以这里把范围拆成人真正会思考的那几桶，数组合，并说明哪些弃、哪些跟。
 */

import { RANKS } from "./poker.js";
import { evaluate, categoryOf } from "./evaluator.js";

export const BUCKETS = ["超强牌", "两对", "顶对", "中/弱对", "强听牌", "弱听牌", "空气"];
const STRONG_MADE = ["超强牌", "两对", "顶对"];

function flushDraw(hole, board) {
  for (let s = 0; s < 4; s++) {
    let n = 0;
    for (const c of hole.concat(board)) if ((c & 3) === s) n++;
    if (n === 4 && hole.some((c) => (c & 3) === s)) return true;
  }
  return false;
}

function straightOuts(hole, board) {
  const ranks = new Set(hole.concat(board).map((c) => c >> 2));
  let outs = 0;
  for (let r = 0; r < 13; r++) {
    if (ranks.has(r)) continue;
    const withR = new Set(ranks); withR.add(r);
    if (withR.has(12)) withR.add(-1);            // A 也能当 1
    for (let start = -1; start <= 8; start++) {
      let ok = true;
      for (let i = 0; i < 5; i++) if (!withR.has(start + i)) { ok = false; break; }
      if (ok) { outs++; break; }
    }
  }
  return outs;
}

/* evaluator 的 category：8 同花顺 7 四条 6 葫芦 5 同花 4 顺子 3 三条 2 两对 1 一对 0 高牌 */
export function classify(hole, board) {
  if (board.length < 3) return "空气";
  const cat = categoryOf(evaluate(hole.concat(board)));
  if (cat >= 3) return "超强牌";
  if (cat === 2) return "两对";
  if (cat === 1) {
    const boardRanks = board.map((c) => c >> 2).sort((a, b) => b - a);
    const top = boardRanks[0];
    const mine = new Set(hole.map((c) => c >> 2));
    const paired = boardRanks.filter((r) => mine.has(r));
    if (paired.length && Math.max(...paired) === top) return "顶对";
    if ((hole[0] >> 2) === (hole[1] >> 2) && (hole[0] >> 2) > top) return "顶对";
    return "中/弱对";
  }
  if (flushDraw(hole, board)) return "强听牌";
  const o = straightOuts(hole, board);
  if (o >= 2) return "强听牌";
  if (o === 1) return "弱听牌";
  return "空气";
}

function labelOf([a, b]) {
  const ra = a >> 2, rb = b >> 2;
  const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
  return ra === rb ? RANKS[hi] + RANKS[hi]
    : RANKS[hi] + RANKS[lo] + ((a & 3) === (b & 3) ? "s" : "o");
}

export function readHands(state, seat, villainRange, examples = 6) {
  const hero = state.seats[seat];
  const board = state.board;
  if (board.length < 3) return {};
  let dead = 0n;
  for (const c of board.concat(hero.hole)) dead |= 1n << BigInt(c);
  const combos = villainRange.combos(dead);
  const total = combos.reduce((a, c) => a + c[2], 0);
  if (total <= 0) return {};

  const heroRank = evaluate(hero.hole.concat(board));
  const heroBucket = classify(hero.hole, board);
  const buckets = new Map(BUCKETS.map((b) => [b, { combos: 0, beats: 0, hands: new Map() }]));
  let ahead = 0, behind = 0, tie = 0;

  for (const [a, b, w] of combos) {
    if (w <= 0) continue;
    const bk = classify([a, b], board);
    const row = buckets.get(bk);
    row.combos += w;
    const lb = labelOf([a, b]);
    row.hands.set(lb, (row.hands.get(lb) || 0) + w);
    const r = evaluate([a, b].concat(board));
    if (r > heroRank) { row.beats += w; ahead += w; }
    else if (r === heroRank) tie += w;
    else behind += w;
  }

  const r1 = (x) => Math.round(x * 10) / 10;
  const table = BUCKETS.filter((b) => buckets.get(b).combos > 0).map((b) => {
    const row = buckets.get(b);
    return {
      name: b, combos: r1(row.combos), pct: r1((100 * row.combos) / total),
      beats_you: r1(row.beats),
      examples: [...row.hands.entries()].sort((x, y) => y[1] - x[1])
        .slice(0, examples).map(([h]) => h),
    };
  });
  const sum = (names) => names.reduce((a, n) => a + buckets.get(n).combos, 0);

  // 「能打走的」看对手手牌的**绝对强度**，不是他有没有打过你
  const foldable = sum(["空气", "弱听牌"]);
  const sticky = sum(["强听牌", "中/弱对"]);
  const never = sum(STRONG_MADE);

  return {
    total_combos: r1(total),
    hero_bucket: heroBucket,
    buckets: table,
    vs_hero: {
      you_beat: r1(behind), you_beat_pct: r1((100 * behind) / total),
      beats_you: r1(ahead), beats_you_pct: r1((100 * ahead) / total),
      tie: r1(tie),
      beaten_by: table.filter((b) => b.beats_you > 0.5).map((b) => b.name),
      you_beat_list: table.filter((b) => b.combos - b.beats_you > 0.5).map((b) => b.name),
    },
    fold_equity: {
      folds: r1(foldable), folds_pct: r1((100 * foldable) / total),
      sticky: r1(sticky), sticky_pct: r1((100 * sticky) / total),
      sticky_label: "中/弱对 + 强听牌：小注会跟，大注多半弃",
      never_folds: r1(never), never_folds_pct: r1((100 * never) / total),
      note: "「能打走的」看的是**对手手牌的绝对强度**，不是他有没有打过你 —— " +
            "A 高面对下注该弃就弃，不管它是不是压着你的 9 高。而这恰恰是诈唬赚钱的地方。",
    },
  };
}
