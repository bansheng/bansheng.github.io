# 博客改进日志

## 2026-04-10：制作 NormFormer FFN Mid-LayerNorm 原理图（Task 1.4）

### 任务目标
为 NormFormer 论文综述制作 FFN Mid-LayerNorm 结构的原理图，展示三阶段流程。

### 输出成果
**文件名**：`ffn-mid-ln.png`
**目录**：`/content/blog/posts/014_normformer_paper_review/`

### 图表设计
主流程展示：
- Input (黄色) → LN_pre (绿色) → W1 (蓝色) → σ (粉红) → **LN_mid (新增，绿色)** → W2 (蓝色) → Output (浅黄)

附加信息：
1. **三阶段说明**
   - Stage 1: LN_pre → W1 → σ (扩展到高维)
   - Stage 2: LN_mid → W2 (新增的动态重新缩放)
   - Stage 3: Output + residual (投影回原维度)

2. **自适应 Gamma 机制**
   - 早期层：gamma < 1.0（梯度抑制）
   - 深层：gamma ≈ 1.0（保留信息）
   - 效果：稳定深层训练

3. **颜色图例**
   - 黄色：输入数据
   - 蓝色：线性变换 (W1, W2)
   - 粉红：激活函数 (σ)
   - 绿色：LayerNorm (新增 LN_mid)
   - 浅黄：输出

### 质量检查结果
✓ 分辨率：2774x1973 px (> 1000px)
✓ 文件大小：149.1 KB (在 100-300KB 范围内)
✓ bytes/pixel：0.0279 (< 0.25)
✓ 所有技术要求达成

### 技术实现
- 工具：Python matplotlib + PIL
- DPI：200 (高清输出)
- 格式：PNG (无损)
- 字体：英文标签（避免中文渲染问题）

### 关键设计要素
1. **清晰的流程箭头**：连接各阶段的黑色箭头
2. **新增标记**：LN_mid 上方加 "★ NEW" 红色标签（黄色背景）
3. **模块化布局**：方框 + 箭头的标准流程图风格
4. **信息分层**：主流程 + 详细说明 + 颜色图例

---

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
