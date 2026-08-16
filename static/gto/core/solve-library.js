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

import { canonicalKeyString, findPermutation, applyPerm } from "./boards.js?v=2eea846f8e";
import { cardToStr } from "./poker.js?v=2eea846f8e";

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

  /** 这手牌是单加注池还是 3bet 池 —— 决定该查哪一份解。
   *  同一个牌面上这是两个**不同的博弈**：范围不一样、SPR 不一样，
   *  拿其中一个当另一个的答案，比没有解更糟（它顶着"精确解"的标签）。 */
  static familyOf(state) {
    const raises = (state.actions || []).filter(
      (a) => a.street === "preflop" && (a.type === "bet" || a.type === "raise")).length;
    return raises >= 2 ? "3bet" : "srp";
  }

  /** 这个翻牌有没有解（按同构算）。返回 {entry, perm} 或 null。 */
  async find(flop, family = "srp") {
    const index = await this.ready();
    const byFamily = index[canonicalKeyString(flop)];
    if (!byFamily) return null;
    const entry = byFamily[family];
    // 有这个牌面、但没有这个牌局类型的解：如实说，不拿另一种顶上
    if (!entry) {
      const have = Object.keys(byFamily);
      return have.length
        ? { missingFamily: family, have }
        : null;
    }
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
    // 预解库都是**单挑**解：两个范围、一个底池、一个 SPR。
    // 多人底池是另一个博弈（每家防守得更少，因为后面还有人），
    // 拿单挑解去答多人局，和拿 3bet 解去答单加注池是同一种错。
    const live = state.seats.filter((s) => !s.folded).length;
    if (live !== 2) {
      return { unavailable:
        `这手牌还有 ${live} 家在池，而预解库都是单挑解 —— 多人底池是另一个博弈，不能拿来当答案` };
    }
    const family = SolveLibrary.familyOf(state);
    const hit = await this.find(state.board.slice(0, 3), family);
    if (!hit) return null;
    if (hit.missingFamily) {
      const cn = { srp: "单加注池", "3bet": "3bet 池" };
      return { unavailable:
        `这个牌面解过 ${hit.have.map((f) => cn[f] || f).join("/")}，但没解过`
        + `${cn[family] || family} —— 两者范围和 SPR 都不同，不能拿来顶替` };
    }
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
