/* 翻前入池范围 vs chart —— 与后端 gto/rangereport.py 同一套口径。
 *
 * 翻前范围漏了，是**系统性**亏损：那个位置以后每一手都在亏，翻后打得
 * 再好也补不回来。所以这份报告不给一个宽度数字了事，而是点名到具体牌：
 *
 *   打太宽 —— 你进池了、chart 会弃的牌
 *   打太紧 —— chart 会玩、你弃了的牌
 *
 * 只报"你 34%、chart 40%"几乎没用：它正好掩盖了你一边弃 AJo 一边跟 96s
 * ——两个漏洞在平均数里互相抵消掉了。
 *
 * 总量用 **VPIP** 和 chart 宽度比，不用"你玩过的牌有多宽"。后者会被样本量
 * 系统性压低：一个位置只打了 30 手，你根本不可能被**发到** 40% 的牌，
 * 于是不管你怎么打每个位置都会显示"太紧"。VPIP 没有这个偏差。
 */

import { comboCount, ALL_LABELS } from "./poker.js?v=36e7fc8dc6";

export const WINDOWS = [["all", null], ["last500", 500], ["last100", 100]];

/* 自愿投钱进池才算入池 —— 大盲免费看牌不算。 */
const ENTER = ["call", "bet", "raise"];

/** chart 在这个位置会玩的牌（非弃牌动作的并集）。 */
function chartRange(chart, position) {
  const spot = chart?.spots?.[`${position}|rfi`];
  if (!spot) return null;
  const played = {};
  for (const [action, r] of Object.entries(spot.ranges || {})) {
    if (action === "fold") continue;
    for (const label of ALL_LABELS) {
      const w = r.weight(label);
      if (w > 0) played[label] = Math.max(played[label] || 0, w);
    }
  }
  return played;
}

/** 一组牌型占全部起手牌的比例，按组合数加权。 */
function width(labels) {
  let total = 0;
  for (const [lb, w] of Object.entries(labels)) total += comboCount(lb) * w;
  return (100 * total) / 1326;
}

export function rangeReport(decisions, chart) {
  // 只看翻前，倒序（最近的在前）好切窗口
  // 只看无人加注的局面。对比的是 chart 的开池范围，而「跟 UTG 开池的牌」
  // 和「自己开池的牌」是两个范围（前者紧得多）。混在一起会让每个曾经
  // 面对过加注的位置无端显示成"太紧"。
  const rows = decisions
    .filter((d) => d.street === "preflop" && /\|rfi$/.test(d.spot || ""))
    .slice()
    .reverse();

  const out = { chart: chart?.name || "chart", windows: {} };
  for (const [name, limit] of WINDOWS) {
    const subset = limit ? rows.slice(0, limit) : rows;
    out.windows[name] = analyse(subset, chart);
    out.windows[name].hands = subset.length;
  }
  return out;
}

function analyse(rows, chart) {
  const byPos = new Map();
  for (const r of rows) {
    if (!byPos.has(r.position))
      byPos.set(r.position, { total: 0, played: {}, folded: new Set() });
    const b = byPos.get(r.position);
    b.total += 1;
    if (ENTER.includes(r.chosen)) b.played[r.hand_label] = (b.played[r.hand_label] || 0) + 1;
    else b.folded.add(r.hand_label);
  }

  const positions = [];
  for (const pos of [...byPos.keys()].sort()) {
    const b = byPos.get(pos);
    const entered = Object.values(b.played).reduce((a, x) => a + x, 0);
    const r1 = (x) => Math.round(x * 10) / 10;
    const entry = {
      position: pos,
      decisions: b.total,
      entered,
      your_vpip_pct: b.total ? r1((100 * entered) / b.total) : 0,
    };
    const cr = chartRange(chart, pos);
    if (!cr) {
      entry.chart_available = false;
      // 说清楚**为什么**没有，而不是渲染一行空白 —— 大盲是唯一一个
      // "VPIP 对比 RFI 范围"本身就没意义的位置，不是数据缺失。
      entry.note = pos === "BB"
        ? "大盲没有 RFI 范围可比：大盲是翻前最后行动的，无人加注时可以免费看牌，它的「入池率」和别的位置的开池范围不是一回事。"
        : "这个位置没有对应的 chart 条目";
      positions.push(entry);
      continue;
    }

    const yours = Object.keys(b.played);
    const chartSet = new Set(Object.keys(cr));
    const tooWide = yours.filter((lb) => !chartSet.has(lb)).sort();
    const yoursSet = new Set(yours);
    const tooTight = [...chartSet].filter((lb) => b.folded.has(lb) && !yoursSet.has(lb)).sort();
    // 样本太小时不给结论：30 手以下的 VPIP 噪声比信号大。
    const enough = b.total >= 30;
    Object.assign(entry, {
      chart_available: true,
      chart_range_pct: r1(width(cr)),
      seen_coverage_pct: r1(width(Object.fromEntries(yours.map((l) => [l, 1])))),
      too_wide: tooWide.slice(0, 24),
      too_wide_count: tooWide.length,
      too_tight: tooTight.slice(0, 24),
      too_tight_count: tooTight.length,
      enough_sample: enough,
      verdict: enough
        ? verdict(entry.your_vpip_pct, width(cr), tooWide.length, tooTight.length)
        : `只有 ${b.total} 个决策，样本太小，先别下结论`,
    });
    positions.push(entry);
  }
  return { positions };
}

function verdict(vpip, chart, wide, tight) {
  const gap = vpip - chart;
  if (Math.abs(gap) < 3 && wide + tight <= 3) return "和 chart 基本吻合";
  const parts = [];
  if (gap > 3) parts.push(`整体比 chart 宽 ${gap.toFixed(0)}pp`);
  else if (gap < -3) parts.push(`整体比 chart 紧 ${(-gap).toFixed(0)}pp`);
  if (wide) parts.push(`${wide} 手 chart 会弃你却在玩`);
  if (tight) parts.push(`${tight} 手 chart 会玩你却弃了`);
  // 两头同时偏是单看宽度必然漏掉的情形，所以显式点出来。
  if (wide && tight) parts.push("**两个方向同时偏** —— 只看宽度会以为你没问题");
  return parts.join("；");
}
