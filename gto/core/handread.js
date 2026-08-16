/* 读牌：对手可能拿什么，各占多少，哪些能打走。
 * 与后端 gto/handread.py 同口径。
 *
 * 胜率数字告诉你这次跟注赚不赚，但教不会你读牌。能带到真牌桌上的是这句话：
 * 「他这条线大概 60 个组合，15 个打过我，20 个是听牌，25 个是我能打走的空气。」
 * 所以这里把范围拆成人真正会思考的那几桶，数组合，并说明哪些弃、哪些跟。
 */

import { RANKS } from "./poker.js?v=2eea846f8e";
import { evaluate, categoryOf } from "./evaluator.js?v=2eea846f8e";

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

/** 一组点数里最好的顺子的最大牌，没有就返回 null。A 也能当 1。 */
function bestStraight(ranks) {
  const rs = new Set(ranks);
  if (rs.has(12)) rs.add(-1);
  let best = null;
  for (let start = -1; start <= 8; start++) {
    let ok = true;
    for (let i = 0; i < 5; i++) if (!rs.has(start + i)) { ok = false; break; }
    if (ok) best = start + 4;
  }
  return best;
}

/* 能成顺、且是**给你**成顺的点数。
 * 只有当成的顺用到了你的牌、并且比牌面自己配上同一张牌成的顺更大，才算一个 out。
 * 9-8-7-6 的牌面上人人都在"听"同一张 5 或 T —— 那不是谁能被打走的听牌，是平分。 */
function straightOuts(hole, board) {
  const hr = new Set(hole.map((c) => c >> 2));
  const br = new Set(board.map((c) => c >> 2));
  let outs = 0;
  for (let r = 0; r < 13; r++) {
    if (hr.has(r) || br.has(r)) continue;
    const mine = bestStraight([...hr, ...br, r]);
    const theirs = bestStraight([...br, r]);
    if (mine !== null && (theirs === null || mine > theirs)) outs++;
  }
  return outs;
}

/** 这手牌落在哪个桶 —— 只算**你**加了什么，不算七张牌合起来是什么。
 *
 * 关键的坑：evaluator 报的是最好的五张，于是在对子面上literally 每个人都
 * "有一对"。照抄那个结果，8-8-3 上的 KQ 就会被归成一对而不是空气；而一旦
 * 范围里没有任何空气，"能打走多少"在每个对子牌面上都是 0%。
 * 所以下面每一类都要问：是不是**你手里的牌**造成的。
 */
export function classify(hole, board) {
  if (board.length < 3) return "空气";
  const holeRanks = hole.map((c) => c >> 2);
  const boardRanks = board.map((c) => c >> 2);
  const topBoard = Math.max(...boardRanks);

  // 同花 —— 得用到你自己的牌才算你的
  for (let suit = 0; suit < 4; suit++) {
    const n = hole.concat(board).filter((c) => (c & 3) === suit).length;
    if (n >= 5) {
      if (hole.some((c) => (c & 3) === suit)) return "超强牌";
      break;   // 牌面同花而你没有那个花色：这里你什么都没有
    }
  }

  // 顺子 —— 得比牌面自己成的顺更大才算你的
  const mine = bestStraight(holeRanks.concat(boardRanks));
  const theirs = bestStraight(boardRanks);
  if (mine !== null && (theirs === null || mine > theirs)) return "超强牌";

  // 三条及以上（含葫芦、四条）：都表现为某个点数凑够 3 张，且你要出一张
  for (const r of new Set(holeRanks)) {
    const n = holeRanks.filter((x) => x === r).length
            + boardRanks.filter((x) => x === r).length;
    if (n >= 3) return "超强牌";
  }

  // 你自己的对子：手牌配上牌面，或者口袋对
  const paired = [...new Set(holeRanks.filter((r) => boardRanks.includes(r)))]
    .sort((a, b) => b - a);
  const pocket = holeRanks[0] === holeRanks[1] && !boardRanks.includes(holeRanks[0]);

  if (paired.length >= 2) return "两对";
  if (pocket && paired.length) return "两对";
  if (paired.length) return paired[0] === topBoard ? "顶对" : "中/弱对";
  if (pocket) return holeRanks[0] > topBoard ? "顶对" : "中/弱对";

  // 自己的牌什么都没成 —— 但听牌不是空气。
  // 河牌没有后续的牌了，所谓"听牌"就只是没成的牌：四张同花差一张，还有一张
  // 要发是听牌，一张都不发了就是空气。把它算成听牌会让这些组合进不了
  // "能打走"那一桶 —— 这是 "能打走 0%" 的另一半原因。
  if (board.length >= 5) return "空气";
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

  // 组合数是怎么来的 —— 面板不写清楚，"超强牌 24 个组合是算出来的还是估的"
  // 这个问题就会一直被问。所以把口径直接放进结果里。
  const strongCombos = sum(["超强牌"]);
  const removed = board.length + 2;

  // 这份范围到底是什么范围 —— 不写清楚，读者会把「还没人下注的完整范围」
  // 当成「下注范围」，然后得出"你的模型认为对手下注从不诈唬"的结论。
  let bettor = null;
  for (let i = state.actions.length - 1; i >= 0; i--) {
    const a = state.actions[i];
    if (a.street === state.street && (a.type === "bet" || a.type === "raise")) {
      bettor = a.seat; break;
    }
  }
  const rangeKind = (bettor === null || bettor === seat)
    ? { label: "对手的完整范围（本街还没人下注）", polarized: false,
        note: "这是他**所有**可能的牌，不是下注范围。完整范围里成手牌占比高很正常 —— "
            + "对子和大牌本来就容易成对。等他真的下注，范围会被极化成"
            + "「价值 + 诈唬」，空气占比会跳上去。" }
    : { label: "对手的下注范围（已极化）", polarized: true,
        note: "他下注了，所以这里是**下注范围**：中等牌多数被过滤掉（它们过牌），"
            + "剩下价值牌和按下注尺度配平的诈唬。注越大，里面的空气越多。" };

  return {
    total_combos: r1(total),
    range_kind: rangeKind,
    how: {
      you_beat: "把范围里每个组合和你的牌**摊牌比大小**，你赢的那些组合数相加",
      beats_you: "同上，他赢的那些相加。两者加起来 + 平局 = 总组合数",
      folds: "空气 + 弱听牌的组合数 —— 看的是他手牌的**绝对强度**，不是他有没有打过你",
      sticky: "中/弱对 + 强听牌：有摊牌价值或有听，小注会跟、大注多半弃",
      never: "顶对以上：这部分你下多大他都不弃，别指望打走",
    },
    method: {
      total: r1(total),
      removed,
      how: `先把对手可能的每一手**具体组合**列出来（AKo 有 12 个组合、AKs 有 4 个、`
         + `AA 有 6 个），再扣掉已经露面的 ${removed} 张牌`
         + `（${board.length} 张公共牌 + 你手上 2 张）——`
         + `这一步叫「阻断」，也是手算时最容易漏的一步。剩下 ${Math.round(total)} 个组合，`
         + `就是下面每一桶的分母。`,
      strong_how: `「超强牌」= 三条及以上（三条 / 顺 / 同花 / 葫芦 / 四条），`
         + `这里有 ${Math.round(strongCombos)} 个组合。判定只看**他手里那两张牌**做了什么：`
         + `牌面自己成的对子、同花、顺子不算他的 —— 8-8-3 上的 KQ 是空气，不是一对。`,
    },
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
