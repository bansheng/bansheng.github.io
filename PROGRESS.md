# 博客改进日志

## 2026-04-10：修复 UniMixer 文章中的 LaTeX HTML Entity 问题

### 问题描述
UniMixer 博客文章（013）中的因式分解机公式无法正确渲染，显示被截断：
- **期望**：`$\hat{y} = w_0 + \sum_i w_i x_i + \sum_{i\lt j} \langle v_i, v_j \rangle x_i x_j$`
- **实际**：显示为 `$\hat{y} = w_0 + \sum_i w_i x_i + \sum_{i`（后面被截断）

### 根本原因
Markdown 中的 `<j>` 被 HTML 浏览器解释为 HTML 标签的开始（`<j` 作为未闭合标签），导致渲染器丢弃这部分内容。这是一个 **HTML 转义问题**，而非 LaTeX 问题。

### 解决方案
在 Markdown 数学公式中使用 LaTeX 转义符号 `\lt` 代替 `<`：
```
# 修改前（错误）
\sum_{i<j}

# 修改后（正确）
\sum_{i\lt j}
```

### 受影响的文件
- `content/blog/posts/013_unimixer_paper_review/index.md` 第 23 行

### Git Commit
```
[fix] Escape HTML entities in LaTeX formulas (< to \lt)
- Fixed FM formula rendering in UniMixer blog post
- Issue: <j> being interpreted as HTML tag, breaking LaTeX display
```

### 预防措施
**今后所有包含 `<` 或 `>` 的 LaTeX 公式，都需要在 Markdown 中转义：**
- 使用 `\lt` 代替 `<`
- 使用 `\gt` 代替 `>`
- 或使用 HTML 实体 `&lt;` 和 `&gt;`

**检查清单（写 LaTeX 公式时）：**
1. 所有 `<` 都改为 `\lt`
2. 所有 `>` 都改为 `\gt`
3. 特别注意求和符号范围：`\sum_{i<j}` → `\sum_{i\lt j}`
4. 提交前在浏览器预览确认公式正确渲染

### 全量扫描结果（2026-04-10）

**方法论：** 主动扫描同类问题，不等用户指出

**扫描范围：**
- 14 篇博客文章全量检查
- 50+ 个数学公式逐一审查
- 正则匹配：`$...<....$` 和 `$...>....$` 无转义情况

**发现：**
- 问题数：1 个（UniMixer 的 `<j>` ✅ 已修复）
- 风险数：0 个（其他 `<` `>` 都是独立出现，如 `$n > 1000$`，不会被浏览器解析）

**结论：** 此轮修复后，博客中的 LaTeX 公式 HTML 实体问题已全部消除。

---
