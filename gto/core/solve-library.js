/* 静态站上的求解库：按牌面懒加载已导出的翻牌街真解。
 *
 * 公网版没有后端、没有求解器，但**解本身只是数据**。导出时只保留翻牌这一街
 * （转牌河牌的子树才是体积大头），再共享组合表 + 量化成百分数，
 * 一个牌面 69 KB，gzip 后 3 KB —— 浏览器按需拉一份完全无感。
 *
 * 查找按**花色同构**：一份解服务最多 24 个真实牌面。所以要先算出当前牌面的
 * 标准形去查索引，再把英雄手牌按同一个花色置换映射进解的坐标系。
 *
 * 天花板要说清楚：只有翻牌这一街有真解。转牌之后仍然是启发式估算，
 * 因为那些子树大 300 倍，静态站放不下。
 */

import { canonicalKeyString, findPermutation, applyPerm } from "./boards.js";
import { cardToStr } from "./poker.js";

export class SolveLibrary {
  constructor(base = "./solves") {
    this.base = base;
    this.index = null;
    this.files = new Map();
    this.missing = new Set();
  }

  async ready() {
    if (this.index !== null) return this.index;
    try {
      const res = await fetch(`${this.base}/index.json`);
      this.index = res.ok ? (await res.json()).spots || {} : {};
    } catch {
      this.index = {};
    }
    return this.index;
  }

  get size() { return this.index ? Object.keys(this.index).length : 0; }

  /** 这个翻牌有没有解（按同构算）。返回 {entry, perm} 或 null。 */
  async find(flop) {
    const index = await this.ready();
    const entry = index[canonicalKeyString(flop)];
    if (!entry) return null;
    const stored = entry.board.match(/../g).map((s) => {
      const r = "23456789TJQKA".indexOf(s[0]), su = "cdhs".indexOf(s[1]);
      return r * 4 + su;
    });
    const perm = findPermutation(flop, stored);
    return perm ? { entry, perm } : null;
  }

  async load(file) {
    if (this.files.has(file)) return this.files.get(file);
    if (this.missing.has(file)) return null;
    try {
      const res = await fetch(`${this.base}/${file}`);
      if (!res.ok) { this.missing.add(file); return null; }
      const data = await res.json();
      data._idx = new Map(data.combos.map((c, i) => [c, i]));
      this.files.set(file, data);
      return data;
    } catch {
      this.missing.add(file);
      return null;
    }
  }

  /** 沿着这手牌翻牌街的实际动作走到当前节点，读出这两张具体牌的频率。 */
  async read(state, seat) {
    if (state.board.length < 3) return null;
    const hit = await this.find(state.board.slice(0, 3));
    if (!hit) return null;
    // 转牌之后就走出了导出的范围 —— 如实返回 null，不拿翻牌的解冒充
    if (state.board.length > 3) {
      return { unavailable: "转牌之后没有导出真解（子树太大），仍用启发式" };
    }
    const data = await this.load(hit.entry.file);
    if (!data) return null;

    const hole = applyPerm(state.seats[seat].hole, hit.perm);
    const key = hole[0] > hole[1]
      ? cardToStr(hole[0]) + cardToStr(hole[1])
      : cardToStr(hole[1]) + cardToStr(hole[0]);
    const idx = data._idx.get(key);
    if (idx === undefined) {
      return { unavailable: `求解范围里没有 ${key}，这手牌用不上这个解` };
    }

    let node = data.tree;
    const path = [];
    for (const a of state.actions) {
      if (a.street !== "flop") continue;
      const label = matchLabel(node, a);
      if (!label) return { unavailable: "这条下注线不在求解树里" };
      node = node.c[label];
      path.push(label);
      if (!node || node.t === "deal") {
        return { unavailable: "走到发牌节点，转牌之后没有导出真解" };
      }
    }
    if (!node || !node.a) return null;

    const n = node.a.length;
    const freqs = node.f.slice(idx * n, idx * n + n).map((x) => x / 100);
    const total = freqs.reduce((a, b) => a + b, 0);
    if (total <= 0) {
      return { unavailable: `求解范围里没有 ${key}，这手牌用不上这个解` };
    }
    return {
      actions: node.a,
      frequencies: freqs.map((f) => f / total),
      player: node.p,
      combo: key,
      path,
      entry: hit.entry,
      permuted: hit.perm.join("") !== "0123",
    };
  }
}

/** 引擎的动作对上求解树的标签。金额是每条街的累计投入，两边同口径。 */
function matchLabel(node, action) {
  if (!node?.c) return null;
  const want = { fold: "FOLD", check: "CHECK", call: "CALL",
                 bet: "BET", raise: "RAISE" }[action.type];
  let best = null, bestGap = Infinity;
  for (const label of Object.keys(node.c)) {
    const [verb, amt] = label.split(/\s+/);
    if (verb !== want) {
      // 树里把它标成 BET 还是 RAISE 取决于上下文，玩家视角是同一个决定
      if (!((verb === "BET" && want === "RAISE") || (verb === "RAISE" && want === "BET"))) continue;
    }
    if (amt === undefined) return label;
    const gap = Math.abs(parseFloat(amt) - action.amount);
    if (gap < 0.51) return label;
    if (gap < bestGap) { bestGap = gap; best = label; }
  }
  return best;
}
