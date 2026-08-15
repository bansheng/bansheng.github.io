/* 决策后的局面分析 —— 与后端 gto/analysis.py 同口径。
 *
 * 建议面板回答「该怎么打」，这里回答「为什么」。后者才是能带到牌桌上的东西：
 * 你记不住每个局面的频率，但你能学会数组合、把价格和胜率比一比。
 */

import { RANKS, comboCount, gridLabel } from "./poker.js";
import { evaluate, categoryOf, CATEGORY_CN } from "./evaluator.js";

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

export function analyse(state, seat, villainRange, equity, derivation = "") {
  const hero = state.seats[seat];
  const board = state.board;
  const bb = state.bigBlind;
  let dead = 0n;
  for (const c of board.concat(hero.hole)) dead |= 1n << BigInt(c);

  const combos = villainRange.combos(dead);
  const totalW = combos.reduce((a, c) => a + c[2], 0) || 1;
  const r1 = (x) => Math.round(x * 10) / 10;
  const r2 = (x) => Math.round(x * 100) / 100;

  const villain = {
    combos: r1(totalW),
    percent: r1((100 * totalW) / 1326),
    grid: rangeGrid(villainRange, dead),
    derivation,
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
  }

  const matchup = {
    hero_hand: heroCategory,
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

  return { villain, matchup, odds, reasons };
}
