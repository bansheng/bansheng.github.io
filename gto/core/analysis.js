/* 决策后的局面分析 —— 与后端 gto/analysis.py 同口径。
 *
 * 建议面板回答「该怎么打」，这里回答「为什么」。后者才是能带到牌桌上的东西：
 * 你记不住每个局面的频率，但你能学会数组合、把价格和胜率比一比。
 */

import { RANKS, comboCount, gridLabel, cardToStr, makeCard } from "./poker.js?v=18362dd4ad";
import { evaluate, categoryOf, CATEGORY_CN } from "./evaluator.js?v=18362dd4ad";
import { readHands } from "./handread.js?v=18362dd4ad";

/* evaluator 的 category 序号是「越大越强」，这里按强到弱列出来 */
const ORDER = [8, 7, 6, 5, 4, 3, 2, 1, 0];

function rangeGrid(range, dead) {
  const byLabel = new Map();
  for (const [a, b, w] of range.combos(dead)) {
    const ra = a >> 2, rb = b >> 2;
    const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
    let label = RANKS[hi] + RANKS[lo];
    if (ra !== rb) label += (a & 3) === (b & 3) ? "s" : "o";
    byLabel.set(label, (byLabel.get(label) || 0) + w);
  }
  const grid = [];
  for (let r = 0; r < 13; r++) {
    const row = [];
    for (let c = 0; c < 13; c++) {
      const lb = gridLabel(r, c);
      row.push(Math.min(1, (byLabel.get(lb) || 0) / comboCount(lb)));
    }
    grid.push(row);
  }
  return grid;
}


/* ---------------- 文字解读 ---------------- */

function rangeWords(pct) {
  if (pct >= 45) return "非常宽（几乎什么都玩）";
  if (pct >= 28) return "偏宽";
  if (pct >= 18) return "中等";
  if (pct >= 10) return "偏紧";
  return "很紧（只有强牌）";
}

function labelOf(a, b) {
  const ra = a >> 2, rb = b >> 2;
  const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
  return ra === rb ? RANKS[hi] + RANKS[hi]
    : RANKS[hi] + RANKS[lo] + ((a & 3) === (b & 3) ? "s" : "o");
}

/* 只用来给范围里最强的几手命名，不是胜率模型 —— 能把 AA 排在 AKs 前面就够了 */
function preflopStrength([a, b]) {
  const ra = a >> 2, rb = b >> 2;
  const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
  if (hi === lo) return 100 + hi;
  return hi * 2 + lo - Math.min(hi - lo, 5) + ((a & 3) === (b & 3) ? 3 : 0);
}

function topHands(combos, board, limit = 8) {
  if (!combos.length) return [];
  const ranked = board.length < 3
    ? combos.map((c) => [c, preflopStrength(c)]).sort((x, y) => y[1] - x[1])
    : combos.map((c) => [c, evaluate([c[0], c[1]].concat(board))]).sort((x, y) => y[1] - x[1]);
  const seen = new Set(), out = [];
  for (const [[a, b]] of ranked) {
    const lb = labelOf(a, b);
    if (seen.has(lb)) continue;
    seen.add(lb); out.push(lb);
    if (out.length >= limit) break;
  }
  return out;
}

/** 对手这条街最后那个动作说明了什么。 */
function betRead(state, seat) {
  let last = null;
  for (let i = state.actions.length - 1; i >= 0; i--) {
    const a = state.actions[i];
    if (a.street === state.street && a.seat !== seat) { last = a; break; }
  }
  if (!last) return "这条街对手还没行动，没有额外信息。";
  const hero = state.seats[seat];
  const toCall = state.currentBet - hero.committed;
  if (last.type === "check")
    return "对手过牌。过牌范围通常是**去掉了最强和最弱两头**的中间段 —— " +
           "强牌想下注拿价值，最烂的牌想下注诈唬，剩下的才过牌。" +
           "所以你可以更放心地下注，但被过牌加注要当真。";
  if (last.type === "fold") return "对手弃牌了。";
  if (toCall <= 0) return "对手这条街没有下注给你。";

  const potBefore = state.pot - toCall;
  const ratio = potBefore > 0 ? toCall / potBefore : 0;
  const p = (x) => `${Math.round(x * 100)}%`;
  let shape;
  if (ratio <= 0.36)
    shape = `小注（约 ${p(ratio)} 底池）。小注通常是**融合型**范围 —— ` +
            "中等强度的牌想廉价拿价值，同时带一点便宜的诈唬。" +
            "这种尺度下对手范围更宽、更弱，你该防守得更宽。";
  else if (ratio <= 0.7)
    shape = `中注（约 ${p(ratio)} 底池）。这是最标准的尺度，` +
            "范围最平衡，价值和诈唬的比例接近理论值。";
  else
    shape = `大注（约 ${p(ratio)} 底池）。大注通常是**极化型**范围 —— ` +
            "要么很强要么在诈唬，中间强度的牌不会选这个尺度。" +
            "所以你的中等牌力在这里价值很低，要么加注要么弃牌。";
  if (last.type === "raise")
    shape += " 而且这是**加注**，不是主动下注 —— 加注范围通常比下注范围窄得多。";
  return shape;
}

/** 严格数 outs：这张牌来了能让你领先对手范围过半。 */
function countOuts(state, seat, combos, totalW, ahead, tied) {
  const board = state.board;
  if (board.length < 3 || board.length > 4 || totalW <= 0) return 0;
  const hero = state.seats[seat];
  const dead = new Set(board.concat(hero.hole));
  const nowAhead = (ahead + tied / 2) / totalW;
  let outs = 0;
  for (let card = 0; card < 52; card++) {
    if (dead.has(card)) continue;
    const nb = board.concat(card);
    const hr = evaluate(hero.hole.concat(nb));
    let better = 0, live = 0;
    for (const [a, b, w] of combos) {
      if (a === card || b === card) continue;
      live += w;
      const vr = evaluate([a, b].concat(nb));
      if (hr < vr) better += w;
      else if (hr === vr) better += w / 2;
    }
    if (live > 0 && better / live > 0.5 && nowAhead <= 0.5) outs++;
  }
  return outs;
}

/** 桌上能心算的公式，配上这个局面的真实数字做校准。 */
function buildShortcuts(state, board, equity, pot, toCall, ahead, tied, totalW, outs) {
  const out = [];
  const p0 = (x) => `${Math.round(x * 100)}%`;
  if (toCall > 0) {
    out.push({
      name: "底池赔率",
      formula: "跟注量 ÷ (当前底池 + 跟注量)",
      applied: `${toCall} ÷ (${pot} + ${toCall}) = ${(toCall / (pot + toCall) * 100).toFixed(1)}%`,
      note: "这是你**至少**要有的胜率。桌上心算就记：跟一份、赢几份。",
    });
    const potBefore = pot - toCall;
    out.push({
      name: "最小防守频率 MDF",
      formula: "下注前底池 ÷ (下注前底池 + 下注量)",
      applied: pot > 0 ? `${potBefore} ÷ (${potBefore} + ${toCall}) = ${p0(potBefore / pot)}` : "—",
      note: "**你整个范围**至少要防守这么高的比例，否则对手拿任意两张牌诈唬都稳赚。" +
            "注意这是对范围的要求，不是对你这一手牌的要求；防守 = 跟注 + 加注，不只是跟注。",
    });
  }
  if (board.length >= 3 && board.length <= 4) {
    const mult = board.length === 3 ? 4 : 2;
    const street = board.length === 3 ? "翻牌（还有两张牌）" : "转牌（还有一张牌）";
    const est = Math.min(0.95, (outs * mult) / 100);
    const now = totalW ? (ahead + tied / 2) / totalW : 0;
    const improve = equity != null ? Math.max(0, equity - now) : null;
    let applied = `${outs} 张 out × ${mult} = ${p0(est)}`;
    if (equity != null)
      applied += `｜真实拆开看：现在就领先 ${p0(now)} ＋ 改进后领先 ${p0(improve)} = 总胜率 ${p0(equity)}`;
    out.push({
      name: `2/4 法则 · ${street}`,
      formula: `outs × ${mult} ≈ **改进后领先**的概率（不是总胜率）`,
      applied,
      note: "最常见的误用就是拿它当总胜率。**总胜率 = 现在就领先的部分 ＋ 改进后领先的部分**，" +
            "2/4 只估后面那一半。这里的 outs 是按「这张牌来了能让你领先对手范围过半」" +
            "严格数出来的，不是拍脑袋 —— 所以它比你在桌上数的通常要保守。",
    });
  }
  if (toCall > 0) {
    for (const [label, mult] of [["2.5 倍", 2.5], ["3.5 倍", 3.5]]) {
      const size = Math.round(toCall * mult);
      const risk = size - toCall;
      if (risk > 0) out.push({
        name: `加注诈唬打平点 · 加到 ${label}`,
        formula: "多投入的部分 ÷ (加注后对手要面对的总底池)",
        applied: `${risk} ÷ (${pot} + ${risk}) = ${p0(risk / (pot + risk))}`,
        note: "对手弃牌超过这个比例，这个诈唬加注就不亏。",
      });
    }
  } else if (pot > 0) {
    for (const [label, frac] of [["半池", 0.5], ["满池", 1.0]]) {
      const size = Math.round(pot * frac);
      out.push({
        name: `诈唬打平点 · ${label}`,
        formula: "下注量 ÷ (底池 + 下注量)",
        applied: `${size} ÷ (${pot} + ${size}) = ${p0(size / (pot + size))}`,
        note: `下${label}诈唬，对手弃牌超过这个比例你就不亏 —— 这还没算被跟注后还能赢的部分。`,
      });
    }
  }
  out.push({
    name: "组合数速算",
    formula: "对子 6 个 · 同花 4 个 · 非同花 12 个",
    applied: "被公共牌占掉一张时：对子剩 3，同花剩 3，非同花剩 6",
    note: "数「对手范围里有几个组合能打败我」时用这个。" +
          "比如牌面有 K，对手 KK 只剩 3 个组合，AK 只剩 8 个。",
  });
  return out;
}

/* 教「怎么数组合」——用你这一手的真实数字走一遍。
   数组合是唯一一个在真牌桌上能徒手做的读牌技能：你算不了蒙特卡洛，
   但你能数对手有几种方式拿到一副 set。所以这里不是讲规则，是把当前局面
   的账一步步算给你看，包括最容易漏掉的「牌被占用后要减」。 */
export function comboMath(state, seat, combos, totalW, heroRank) {
  const hero = state.seats[seat];
  const board = state.board;
  const dead = new Set(board.concat(hero.hole));
  const deadRanks = new Map();
  for (const c of dead) deadRanks.set(c >> 2, (deadRanks.get(c >> 2) || 0) + 1);
  const gone = (r) => deadRanks.get(r) || 0;

  const basics = [
    { kind: "口袋对子（如 KK）", full: 6, why: "从 4 张同点数里挑 2 张：C(4,2) = 4×3÷2 = 6" },
    { kind: "同花两张（如 AKs）", full: 4, why: "花色必须一样，4 种花色各一种组合 = 4" },
    { kind: "非同花两张（如 AKo）", full: 12, why: "A 有 4 种花色 × K 有 4 种 = 16，减去 4 种同花 = 12" },
  ];

  const blockers = [];
  for (const rank of [...deadRanks.keys()].sort((a, b) => b - a)) {
    const g = gone(rank), left = 4 - g, r = RANKS[rank];
    if (g <= 0) continue;
    blockers.push({
      rank: r, gone: g,
      pair: `${r}${r}：6 → ${(left * (left - 1)) / 2}`,
      suited: `含 ${r} 的同花两张：4 → ${Math.max(0, left)}`,
      offsuit: `含 ${r} 的非同花两张：12 → ${left * 4 - left}`,
    });
  }

  const labelOfCombo = ([a, b]) => {
    const ra = a >> 2, rb = b >> 2;
    const hi = Math.max(ra, rb), lo = Math.min(ra, rb);
    return ra === rb ? RANKS[hi] + RANKS[hi]
      : RANKS[hi] + RANKS[lo] + ((a & 3) === (b & 3) ? "s" : "o");
  };

  let ties = 0, loses = 0;
  const byLabel = new Map();
  if (heroRank != null && board.length >= 3) {
    for (const [a, b, w] of combos) {
      const r = evaluate([a, b].concat(board));
      if (r > heroRank) {
        const lb = labelOfCombo([a, b]);
        byLabel.set(lb, (byLabel.get(lb) || 0) + w);
      } else if (r === heroRank) ties += w;
      else loses += w;
    }
  }
  const beats = [...byLabel.entries()].sort((x, y) => y[1] - x[1]).slice(0, 10)
    .map(([hand, w]) => ({ hand, combos: Math.round(w * 10) / 10 }));
  const allBeat = heroRank != null ? totalW - loses - ties : 0;

  const steps = [];
  if (heroRank != null && board.length >= 3 && totalW > 0) {
    steps.push(`1) 先定对手范围：${totalW.toFixed(0)} 个组合（这是分母）`);
    if (beats.length) {
      const top = beats.slice(0, 4).map((b) => `${b.hand} ${b.combos.toFixed(0)} 个`).join("、");
      steps.push(`2) 数出能打败你的：${top}…… 合计 ${allBeat.toFixed(0)} 个`);
    }
    steps.push(`3) 概率 = 能打败你的组合数 ÷ 范围总组合数 = ${allBeat.toFixed(0)} ÷ ` +
               `${totalW.toFixed(0)} = **${((100 * allBeat) / totalW).toFixed(0)}%**`);
    steps.push(`4) 反过来，你现在领先的概率 = ${(totalW - allBeat - ties).toFixed(0)} ÷ ` +
               `${totalW.toFixed(0)} = **${((100 * (totalW - allBeat - ties)) / totalW).toFixed(0)}%**` +
               (ties > 0.5 ? `（另有 ${ties.toFixed(0)} 个组合和你平分）` : ""));
  }

  return {
    basics, blockers, beats_you: beats,
    beats_total: Math.round(allBeat * 10) / 10,
    range_total: Math.round(totalW * 10) / 10,
    steps,
    note: "**这一步是能在牌桌上真做的**：你算不了蒙特卡洛，但你能数组合。" +
          "关键是别忘了减掉被公共牌和自己手牌占掉的那些 —— 牌面有一张 K，" +
          "对手的 KK 就只剩 3 个组合而不是 6 个，少算这一步会把对手的强牌概率高估将近一倍。",
  };
}

/* 把「现在领先吗」和「最终会赢吗」分开，并把两者的关系算出来。
   混淆这两个是读错局面最常见的方式：
     领先概率 = 此刻摊牌我赢的比例（纯数组合，不含运气）
     胜率     = 发完剩下的牌之后我赢的比例
   关系：胜率 = P(现在领先) × P(守住) + P(现在落后) × P(反超)
   河牌上两者按定义相等；在此之前，**差值就是后面那些牌的价值**。 */
export function leadVsEquity(state, seat, combos, heroRank, trials = 3000, rand = Math.random) {
  const hero = state.seats[seat];
  const board = state.board;
  if (heroRank == null || !combos.length || board.length >= 5) return {};

  const cum = []; let total = 0;
  for (const c of combos) { total += c[2]; cum.push(total); }
  const pick = () => {
    const x = rand() * total;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < x) lo = m + 1; else hi = m; }
    return combos[lo];
  };
  const need = 5 - board.length;
  const blocked = new Set(board.concat(hero.hole));

  let held = 0, lost = 0, caught = 0, never = 0;
  for (let i = 0; i < trials; i++) {
    const [va, vb] = pick();
    if (blocked.has(va) || blocked.has(vb)) continue;
    const nowAhead = heroRank > evaluate([va, vb].concat(board));
    const pool = [];
    for (let c = 0; c < 52; c++) if (!blocked.has(c) && c !== va && c !== vb) pool.push(c);
    for (let k = 0; k < need; k++) {
      const j = k + Math.floor(rand() * (pool.length - k));
      const tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
    }
    const full = board.concat(pool.slice(0, need));
    const endAhead = evaluate(hero.hole.concat(full)) > evaluate([va, vb].concat(full));
    if (nowAhead && endAhead) held++;
    else if (nowAhead) lost++;
    else if (endAhead) caught++;
    else never++;
  }
  const n = held + lost + caught + never;
  if (!n) return {};
  const leadNow = (held + lost) / n, eq = (held + caught) / n;
  const p = (x) => `${Math.round(x * 100)}%`;
  const holdRate = held + lost ? held / (held + lost) : 0;
  const catchRate = caught + never ? caught / (caught + never) : 0;
  const r4 = (x) => Math.round(x * 10000) / 10000;
  return {
    trials: n, lead_now: r4(leadNow), equity_after: r4(eq),
    held: r4(held / n), lost: r4(lost / n), caught: r4(caught / n), never: r4(never / n),
    hold_rate: r4(holdRate), catch_rate: r4(catchRate), gap: r4(eq - leadNow),
    formula: "胜率 = P(现在领先) × P(守住) + P(现在落后) × P(反超)",
    worked: `${p(leadNow)} × ${p(holdRate)} ＋ ${p(1 - leadNow)} × ${p(catchRate)} = **${p(eq)}**`,
    reading: Math.abs(eq - leadNow) < 0.03
      ? "两个数几乎一样，说明这手牌基本靠现在的牌力赢，后面的牌帮不上也害不了。"
      : eq > leadNow
      ? `胜率比领先概率高 ${Math.round((eq - leadNow) * 100)}pp —— 差出来的这部分**就是听牌的价值**，你现在落后但后面能反超。`
      : `胜率比领先概率低 ${Math.round((leadNow - eq) * 100)}pp —— 你现在领先但**很脆**，后面的牌更可能帮到对手，所以别把底池打太大。`,
  };
}

export function analyse(state, seat, villainRange, equity, derivation = "", heroRange = null) {
  const hero = state.seats[seat];
  const board = state.board;
  const bb = state.bigBlind;
  let dead = 0n;
  for (const c of board.concat(hero.hole)) dead |= 1n << BigInt(c);

  const combos = villainRange.combos(dead);
  const totalW = combos.reduce((a, c) => a + c[2], 0) || 1;
  const r1 = (x) => Math.round(x * 10) / 10;
  const r2 = (x) => Math.round(x * 100) / 100;

  const villainPct = (100 * totalW) / 1326;
  const villain = {
    combos: r1(totalW),
    percent: r1(villainPct),
    grid: rangeGrid(villainRange, dead),
    derivation,
    width_words: rangeWords(villainPct),
    top_hands: topHands(combos, board),
    read: betRead(state, seat),
    summary: "",
    categories: [],
  };

  let ahead = 0, tied = 0, behind = 0;
  const catW = new Map();
  let heroCategory = null;

  if (board.length >= 3) {
    const heroRank = evaluate(hero.hole.concat(board));
    heroCategory = CATEGORY_CN[categoryOf(heroRank)];
    for (const [a, b, w] of combos) {
      const r = evaluate([a, b].concat(board));
      const c = categoryOf(r);
      catW.set(c, (catW.get(c) || 0) + w);
      if (heroRank > r) ahead += w;
      else if (heroRank === r) tied += w;
      else behind += w;
    }
    villain.categories = ORDER.filter((c) => catW.get(c) > 0).map((c) => ({
      name: CATEGORY_CN[c],
      combos: r1(catW.get(c)),
      percent: r1((100 * catW.get(c)) / totalW),
    }));
    // 1 = 一对, 0 = 高牌；其余都算两对以上
    let strong = 0;
    for (const [c, w] of catW) if (c > 1) strong += w;
    villain.strong_pct = r1((100 * strong) / totalW);
  }

  {
    const top = villain.top_hands.slice(0, 5).join("、") || "—";
    const head = `对手范围${rangeWords(villainPct)}，约 ${Math.round(totalW)} 个组合` +
                 `（占全部起手牌 ${villainPct.toFixed(1)}%）。`;
    villain.summary = board.length >= 3
      ? head + `在 ${board.map(cardToStr).join("")} 这个牌面上，` +
        `其中 **${(villain.strong_pct ?? 0).toFixed(0)}% 是两对以上**，` +
        `剩下 ${(100 - (villain.strong_pct ?? 0)).toFixed(0)}% 只是一对或高牌。` +
        `最强的那几手是 ${top}。`
      : head + `里面最强的那几手是 ${top}。翻前还没有公共牌，无法拆牌型。`;
  }

  // 自己范围里的位置：绝对牌力没意义，相对位置才有
  const heroPos = state.positionName(seat);
  const heroBlock = {
    note: `你在 ${heroPos}。**判断自己这手牌好不好，要看它在你自己范围里的位置**，` +
          "不是绝对牌力 —— 顶对在一个很紧的范围里可能只是中游，在一个很宽的范围里就是顶端。",
  };
  if (heroRange && board.length >= 3) {
    let boardDead = 0n;
    for (const c of board) boardDead |= 1n << BigInt(c);
    const mine = heroRange.combos(boardDead);
    if (mine.length) {
      const heroRank = evaluate(hero.hole.concat(board));
      let worse = 0, myTotal = 0;
      for (const [a, b, w] of mine) {
        myTotal += w;
        if (evaluate([a, b].concat(board)) < heroRank) worse += w;
      }
      const pct = (100 * worse) / myTotal;
      Object.assign(heroBlock, {
        combos: r1(myTotal),
        percent: r1((100 * myTotal) / 1326),
        percentile: r1(pct),
        top_hands: topHands(mine, board),
        summary:
          `你的 **${heroCategory}** 在你自己的范围里打败了 **${pct.toFixed(0)}%** 的组合 —— ` +
          (pct >= 80 ? "属于范围顶端，可以放心拿价值。"
           : pct >= 60 ? "属于范围上游，通常够跟但不够加。"
           : pct >= 35 ? "在范围中游，这类牌最难打：跟注亏、弃牌可惜。"
           : "在范围底部，要么当诈唬打要么放掉。"),
      });
    }
  }

  const toCallNow = state.currentBet - hero.committed;
  const outs = countOuts(state, seat, combos, totalW, ahead, tied);

  const matchup = {
    hero_hand: heroCategory,
    outs,
    equity: equity == null ? null : Math.round(equity * 10000) / 10000,
    ahead_combos: r1(ahead), tied_combos: r1(tied), behind_combos: r1(behind),
    ahead_pct: board.length >= 3 ? r1((100 * ahead) / totalW) : null,
    behind_pct: board.length >= 3 ? r1((100 * behind) / totalW) : null,
    note: board.length === 3 || board.length === 4
      ? "「领先/落后」是**现在摊牌**的结果；胜率是算上后面还要发的牌之后的期望，" +
        "两者不同是正常的 —— 差得越多，说明这手牌的听牌价值越重"
      : "",
  };

  const pot = state.pot;
  const toCall = state.currentBet - hero.committed;
  const others = state.seats.filter((s) => !s.folded && s.index !== seat).map((s) => s.stack);
  const effective = Math.min(hero.stack, others.length ? Math.max(...others) : 0);
  const odds = {
    pot, pot_bb: r1(pot / bb), to_call: toCall, to_call_bb: r1(toCall / bb),
    pot_odds: toCall > 0 ? r2(toCall / (pot + toCall) * 100) / 100 : null,
    required_equity: toCall > 0 ? Math.round((toCall / (pot + toCall)) * 10000) / 10000 : null,
    ev_call_bb: toCall > 0 && equity != null
      ? r2((equity * pot - (1 - equity) * toCall) / bb) : null,
    spr: pot ? r2(effective / pot) : null,
    effective_stack_bb: r1(effective / bb),
  };

  const legal = new Set(state.legalActions().map((l) => l.type));
  if (legal.has("bet") || legal.has("raise")) {
    odds.bluff_breakeven = [["半池", 0.5], ["三分之二池", 0.66], ["满池", 1.0]]
      .map(([label, frac]) => {
        const size = Math.round(pot * frac);
        return size > 0 ? {
          label, size, size_bb: r1(size / bb),
          fold_pct_needed: r1((100 * size) / (pot + size)),
        } : null;
      })
      .filter(Boolean);
  }

  const reasons = [];
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  if (toCall > 0 && equity != null) {
    const req = toCall / (pot + toCall);
    reasons.push(
      `底池 ${(pot / bb).toFixed(1)}bb，需要跟 ${(toCall / bb).toFixed(1)}bb，` +
      `底池赔率 ${(pot / toCall).toFixed(1)}:1 —— 也就是至少要 **${pct(req)}** 胜率才不亏。`);
    const gap = equity - req;
    const ev = (equity * pot - (1 - equity) * toCall) / bb;
    reasons.push(gap > 0
      ? `你对抗这个范围有 **${pct(equity)}** 胜率，比门槛高 ${(gap * 100).toFixed(1)}pp，` +
        `跟注本身是 +EV（约 ${ev >= 0 ? "+" : ""}${ev.toFixed(2)}bb）。`
      : `你只有 **${pct(equity)}** 胜率，比门槛低 ${(-gap * 100).toFixed(1)}pp，` +
        `单纯跟注是 -EV（约 ${ev.toFixed(2)}bb）；要跟就得靠隐含赔率来补。`);
  } else if (equity != null) {
    reasons.push(`没有下注要面对，你对抗这个范围有 **${pct(equity)}** 胜率。`);
  }

  if (matchup.ahead_pct != null) {
    reasons.push(
      `按现在摊牌算，你的 **${heroCategory}** 领先对手范围里 ${matchup.ahead_pct.toFixed(0)}% ` +
      `的组合（${ahead.toFixed(0)} 个），落后 ${matchup.behind_pct.toFixed(0)}%（${behind.toFixed(0)} 个）。`);
    if (equity != null && board.length < 5) {
      const drift = equity - (ahead + tied / 2) / totalW;
      if (drift > 0.05)
        reasons.push(`胜率比「现在摊牌」高 ${(drift * 100).toFixed(0)}pp —— 你有听牌价值，后面的牌能帮你反超。`);
      else if (drift < -0.05)
        reasons.push(`胜率比「现在摊牌」低 ${(-drift * 100).toFixed(0)}pp —— 你现在领先但很脆，后面的牌更可能帮到对手。`);
    }
  }

  if (odds.spr != null) {
    const spr = odds.spr;
    reasons.push(spr < 1
      ? `SPR ${spr.toFixed(1)}（很浅）—— 底池已经承诺了大部分筹码，基本是成手就打光。`
      : spr < 4
      ? `SPR ${spr.toFixed(1)}（中等）—— 顶对以上就有全下的空间。`
      : `SPR ${spr.toFixed(1)}（很深）—— 后面还有多轮下注，别把中等强度的牌打成大底池。`);
  }

  if (odds.bluff_breakeven?.length && toCall === 0) {
    const b = odds.bluff_breakeven[0];
    reasons.push(
      `如果要诈唬：下 ${b.label}（${b.size_bb.toFixed(1)}bb）需要对手弃牌 ` +
      `**${b.fold_pct_needed.toFixed(0)}%** 才能打平。`);
  }

  const shortcuts = buildShortcuts(
    state, board, equity, pot, toCall, ahead, tied, totalW, outs);
  void toCallNow;

  const heroRankNow = board.length >= 3 ? evaluate(hero.hole.concat(board)) : null;
  return { villain, matchup, odds, reasons, hero_range: heroBlock, shortcuts,
           combo_math: comboMath(state, seat, combos, totalW, heroRankNow),
           lead_vs_equity: leadVsEquity(state, seat, combos, heroRankNow),
           hand_read: readHands(state, seat, villainRange) };
}
