/* 7 张牌牌力评估 —— 浏览器端实现。
 *
 * 后端用的是 phevaluator（C 扩展，88 万次/秒）。浏览器里没有它，所以这里重写一份。
 * 关键是两边必须**排序完全一致**，否则同一手牌本地和后端会给出不同的赢家。
 * tests/test_js_evaluator.py 用 10 万手随机牌逐一对撞 phevaluator 验证这件事。
 *
 * 编码和后端一致：card = rank * 4 + suit，rank 0..12 = 2..A，suit 0..3 = c d h s。
 *
 * 返回值是一个可比较的整数，**越大越强**（注意：和 phevaluator 的「越小越强」相反，
 * 因为 JS 这边直接构造分数比取补更自然；比较时只看序，不看绝对值）。
 * 分数结构： category * 2^20 + 五张牌的 rank 依次降权排列。
 */

export const CATEGORY = {
  HIGH_CARD: 0, PAIR: 1, TWO_PAIR: 2, TRIPS: 3, STRAIGHT: 4,
  FLUSH: 5, FULL_HOUSE: 6, QUADS: 7, STRAIGHT_FLUSH: 8,
};

export const CATEGORY_CN = [
  "高牌", "一对", "两对", "三条", "顺子", "同花", "葫芦", "四条", "同花顺",
];

/* 把 5 个 rank 打包成一个 20 位整数，第一个权重最高 */
function kickers(a, b = 0, c = 0, d = 0, e = 0) {
  return (a << 16) | (b << 12) | (c << 8) | (d << 4) | e;
}

function score(category, a, b, c, d, e) {
  return category * 0x100000 + kickers(a, b, c, d, e);
}

/* A-2-3-4-5「轮子」：A 在这里当 1 用，顺子最大牌是 5 (rank 3)。 */
const WHEEL = (1 << 12) | (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0);

/** 从 rank 位掩码里找最高顺子，返回顺子最大牌的 rank；没有顺子返回 -1。 */
function straightHigh(mask) {
  for (let hi = 12; hi >= 4; hi--) {
    const need = (1 << hi) | (1 << (hi - 1)) | (1 << (hi - 2)) | (1 << (hi - 3)) | (1 << (hi - 4));
    if ((mask & need) === need) return hi;
  }
  return (mask & WHEEL) === WHEEL ? 3 : -1;
}

/** 评估 5–7 张牌，返回可比较分数（越大越强）。 */
export function evaluate(cards) {
  const rankCount = new Int8Array(13);
  const suitCount = new Int8Array(4);
  const suitRankMask = new Int32Array(4);
  let rankMask = 0;

  for (let i = 0; i < cards.length; i++) {
    const c = cards[i];
    const r = c >> 2, s = c & 3;
    rankCount[r]++;
    suitCount[s]++;
    suitRankMask[s] |= 1 << r;
    rankMask |= 1 << r;
  }

  /* 同花 / 同花顺 */
  for (let s = 0; s < 4; s++) {
    if (suitCount[s] >= 5) {
      const fm = suitRankMask[s];
      const sf = straightHigh(fm);
      if (sf >= 0) return score(CATEGORY.STRAIGHT_FLUSH, sf, 0, 0, 0, 0);
      const top = [];
      for (let r = 12; r >= 0 && top.length < 5; r--) if ((fm >>> r) & 1) top.push(r);
      return score(CATEGORY.FLUSH, top[0], top[1], top[2], top[3], top[4]);
    }
  }

  /* 按张数分组，同张数内按 rank 降序 */
  const quads = [], trips = [], pairs = [], singles = [];
  for (let r = 12; r >= 0; r--) {
    const n = rankCount[r];
    if (n === 4) quads.push(r);
    else if (n === 3) trips.push(r);
    else if (n === 2) pairs.push(r);
    else if (n === 1) singles.push(r);
  }

  if (quads.length) {
    // 踢脚是剩下最大的一张（可能来自 trips/pairs/singles）
    let kick = -1;
    for (let r = 12; r >= 0; r--) if (r !== quads[0] && rankCount[r] > 0) { kick = r; break; }
    return score(CATEGORY.QUADS, quads[0], kick, 0, 0, 0);
  }

  if (trips.length >= 2) {
    // 两组三条 → 取大的当三条，小的当对子
    return score(CATEGORY.FULL_HOUSE, trips[0], trips[1], 0, 0, 0);
  }
  if (trips.length === 1 && pairs.length) {
    return score(CATEGORY.FULL_HOUSE, trips[0], pairs[0], 0, 0, 0);
  }

  const st = straightHigh(rankMask);
  if (st >= 0) return score(CATEGORY.STRAIGHT, st, 0, 0, 0, 0);

  if (trips.length === 1) {
    const k = [];
    for (let r = 12; r >= 0 && k.length < 2; r--) if (rankCount[r] === 1) k.push(r);
    return score(CATEGORY.TRIPS, trips[0], k[0], k[1], 0, 0);
  }
  if (pairs.length >= 2) {
    let kick = -1;
    for (let r = 12; r >= 0; r--) {
      if (r !== pairs[0] && r !== pairs[1] && rankCount[r] > 0) { kick = r; break; }
    }
    return score(CATEGORY.TWO_PAIR, pairs[0], pairs[1], kick, 0, 0);
  }
  if (pairs.length === 1) {
    const k = [];
    for (let r = 12; r >= 0 && k.length < 3; r--) if (rankCount[r] === 1) k.push(r);
    return score(CATEGORY.PAIR, pairs[0], k[0], k[1], k[2], 0);
  }
  return score(CATEGORY.HIGH_CARD, singles[0], singles[1], singles[2], singles[3], singles[4]);
}

export function categoryOf(s) {
  return Math.floor(s / 0x100000);
}

export function categoryName(s) {
  return CATEGORY_CN[categoryOf(s)];
}
