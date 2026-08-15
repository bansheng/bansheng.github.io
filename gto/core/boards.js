/* 翻牌的花色同构 —— 必须和后端 gto/boards.py 生成**同一个字符串**，
 * 因为那就是求解库索引的键。对不上就是全部查不到，而且是静默查不到。
 *
 * 22,100 个 flop 按花色重命名塌缩成 1,755 类：QsJh2h 和 QcJd2d 是同一个博弈，
 * 只是花色名字不同。所以一份解能服务最多 24 个真实牌面。
 */

/** 元组按逐元素比较，模仿 Python 的 tuple 比较语义。 */
function cmpTuple(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

/** 同构标准形。与 Python 的 canonical_key 一一对应。 */
export function canonicalKey(cards) {
  const ranks = cards.map((c) => c >> 2).sort((a, b) => b - a);
  const bySuit = new Map();
  for (const c of cards) {
    const s = c & 3;
    if (!bySuit.has(s)) bySuit.set(s, []);
    bySuit.get(s).push(c >> 2);
  }
  const groups = [...bySuit.values()]
    .map((v) => v.slice().sort((a, b) => b - a))
    .sort((a, b) => cmpTuple(b, a));
  return [ranks, groups];
}

/** 索引键：和 Python 侧 json.dumps(canonical_key, separators=(",",":")) 一致。 */
export function canonicalKeyString(cards) {
  return JSON.stringify(canonicalKey(cards));
}

/** 花色重命名：花色 s 变成 perm[s]，点数不动。 */
export function applyPerm(cards, perm) {
  return cards.map((c) => (c >> 2) * 4 + perm[c & 3]);
}

const PERMS = (() => {
  const out = [];
  const walk = (left, acc) => {
    if (!left.length) { out.push(acc); return; }
    for (let i = 0; i < left.length; i++) {
      walk(left.filter((_, j) => j !== i), acc.concat(left[i]));
    }
  };
  walk([0, 1, 2, 3], []);
  return out;
})();

/** 把 actual 变成 target 的花色置换，找不到返回 null。 */
export function findPermutation(actual, target) {
  const want = new Set(target);
  if (want.size !== new Set(actual).size) return null;
  for (const perm of PERMS) {
    const mapped = applyPerm(actual, perm);
    if (mapped.every((c) => want.has(c))) return perm;
  }
  return null;
}
