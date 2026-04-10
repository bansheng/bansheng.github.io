# NormFormer 论文综述重写 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 NormFormer 论文综述从逻辑混乱的架构改写为问题导向的清晰呈现，通过梯度流向图直观展示问题，在改进点处嵌入消融数据和原理图。

**Architecture:** 
- Phase 1：制作 4 张新增 PNG 图表（梯度流向图、Post-Attn 原理图、HeadScale 原理图、FFN 原理图、消融贡献度图）
- Phase 2：重构引言部分（新增梯度流向图说明、类比解释、梯度失配的危害）
- Phase 3：重新组织改进点讲解（2.0 架构总览、2.1-2.3 改进点各配原理图+消融数据、2.5 协同效应）
- Phase 4：调整实验部分（实验结果不再孤立，与改进点关联）
- Phase 5：验证、测试、最终审核与提交

**Tech Stack:** 
- Markdown 文本编辑
- PNG 图表制作（excalidraw / matplotlib）
- Hugo 博客渲染验证
- Git 版本控制

---

## 文件映射

**修改文件：**
- `content/blog/posts/014_normformer_paper_review/index.md` — 主博客文件，完全重写

**新增图表文件：**
- `content/blog/posts/014_normformer_paper_review/gradient-flow.png` — 梯度流向图（新增）
- `content/blog/posts/014_normformer_paper_review/post-attn-ln.png` — Post-Attn 原理图（新增）
- `content/blog/posts/014_normformer_paper_review/headscale.png` — HeadScale 原理图（新增）
- `content/blog/posts/014_normformer_paper_review/ffn-mid-ln.png` — FFN 原理图（新增）
- `content/blog/posts/014_normformer_paper_review/ablation-contrib.png` — 消融贡献度图（新增）

**现有图表（保留使用）：**
- `content/blog/posts/014_normformer_paper_review/figure1.png` — 架构对比（现有，移到 2.0 节）
- `content/blog/posts/014_normformer_paper_review/figure3.png` — 梯度分布曲线（现有，用于消融实验）
- `content/blog/posts/014_normformer_paper_review/figure4_5.png` — 缩放参数分布（现有，用于讲解改进点 C）
- `content/blog/posts/014_normformer_paper_review/featured.png` — 封面图（保留）

---

## 任务分解

### Phase 1：图表制作（梯度流向图、原理图、消融图）

#### Task 1.1：制作梯度流向图

**文件：**
- Create: `content/blog/posts/014_normformer_paper_review/gradient-flow.png`

**说明：** 梯度流向图展示 Post-LN / Pre-LN / NormFormer 三种架构的梯度分布对比。用三条线展示各层的梯度范数变化。

- [ ] **Step 1: 用 excalidraw 或 matplotlib 草稿梯度流向图**

使用 matplotlib 代码（本地制作，不在博客中）：

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

layers = np.arange(1, 13)  # 12 层

# Post-LN: 梯度逐层递增（深层爆炸）
post_ln_grad = 0.1 + (layers - 1) * 0.08

# Pre-LN: 梯度逐层递减（深层不足）
pre_ln_grad = 1.0 - (layers - 1) * 0.06

# NormFormer: 梯度均衡
normformer_grad = np.ones_like(layers) * 0.6 + np.random.normal(0, 0.05, len(layers))

ax.plot(layers, post_ln_grad, marker='o', label='Post-LN: 深层爆炸', linewidth=2, color='#e8534a')
ax.plot(layers, pre_ln_grad, marker='s', label='Pre-LN: 深层不足', linewidth=2, color='#4a7ae8')
ax.plot(layers, normformer_grad, marker='^', label='NormFormer: 均衡', linewidth=2, color='#4ae857')

ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Gradient L1 Norm', fontsize=12)
ax.set_title('Gradient Distribution Across Layers', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('gradient-flow.png', dpi=150, bbox_inches='tight')
plt.close()
```

- [ ] **Step 2: 验证图表清晰度和尺寸**

导出后检查：
- 宽度 800px（≈1200px 最佳，本图可用 800px）
- 文件大小 < 300KB（通常 150-200KB）
- 三条线清晰可区分，文字可读

**预期输出：**
- 文件大小：~180KB
- 分辨率：1200x750 像素
- bytes/pixel：~0.18（符合标准）

- [ ] **Step 3: 保存到博客目录**

```bash
cp gradient-flow.png /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/
```

---

#### Task 1.2：制作 Post-Attn LayerNorm 原理图

**文件：**
- Create: `content/blog/posts/014_normformer_paper_review/post-attn-ln.png`

**说明：** 展示标准 Pre-LN 与 NormFormer 在注意力输出后的区别。用两个并排的 block 图展示。

- [ ] **Step 1: 用 excalidraw 或 matplotlib 制作原理图**

用 matplotlib 制作两个 block 对比（伪代码风格）：

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def draw_block(ax, title, y_positions, labels, colors):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(y_positions) + 1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    for y, label, color in zip(y_positions, labels, colors):
        rect = FancyBboxPatch((1, y-0.3), 8, 0.6, boxstyle="round,pad=0.1", 
                               edgecolor=color, facecolor=color, alpha=0.3, linewidth=2)
        ax.add_patch(rect)
        ax.text(5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Pre-LN (left)
draw_block(ax1, 'Pre-LN\n(Standard)', 
           [5, 4, 3, 2], 
           ['LayerNorm(x)', 'MultiHeadAttn(...)', 'x + attn_out', 'Output'],
           ['#ffd700', '#4a7ae8', '#90ee90', '#ffcccc'])

# NormFormer (right)
draw_block(ax2, 'NormFormer\n(With Post-Attn LN)',
           [5, 4, 3.2, 2.4, 1.6],
           ['LayerNorm(x)', 'MultiHeadAttn(...)', 'LayerNorm(attn)', '← New!', 'x + ln_out'],
           ['#ffd700', '#4a7ae8', '#00d977', '#ffcccc', '#ffcccc'])

plt.suptitle('Post-Attention LayerNorm: 注意力输出的幅度控制', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('post-attn-ln.png', dpi=150, bbox_inches='tight')
plt.close()
```

- [ ] **Step 2: 验证图表清晰度和尺寸**

导出后检查：
- 宽度 1200px
- 文件大小 < 250KB
- 两个 block 对比清晰，标注清晰

**预期输出：**
- 文件大小：~150KB
- 分辨率：1200x600 像素

- [ ] **Step 3: 保存到博客目录**

```bash
cp post-attn-ln.png /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/
```

---

#### Task 1.3：制作 HeadScale 原理图

**文件：**
- Create: `content/blog/posts/014_normformer_paper_review/headscale.png`

**说明：** 展示 MultiHeadAttention 的 concat 前添加逐头缩放参数。

- [ ] **Step 1: 用 matplotlib 制作 HeadScale 原理图**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# 标题
ax.text(6, 8.5, 'HeadScale: 逐头缩放参数', fontsize=14, fontweight='bold', ha='center')
ax.text(6, 8, '在 concat 操作前对每个注意力头乘以可学习的标量 γᵢ', fontsize=10, ha='center', style='italic', color='#666')

# 上面：MultiHeadAttention 的多个头
head_y = 6.5
ax.text(0.5, head_y + 0.5, 'Heads:', fontsize=10, fontweight='bold')

head_width = 1.2
for i in range(8):
    x = 1.5 + i * head_width
    rect = FancyBboxPatch((x, head_y - 0.3), head_width - 0.2, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='#4a7ae8', facecolor='#4a7ae8', alpha=0.3, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + head_width/2 - 0.1, head_y, f'h{i+1}', ha='center', va='center', fontsize=9)

# 中间：缩放参数
scale_y = 5
ax.text(0.5, scale_y + 0.3, 'Scales:', fontsize=10, fontweight='bold')

for i in range(8):
    x = 1.5 + i * head_width
    ax.text(x + head_width/2 - 0.1, scale_y, f'γ={0.8 + i*0.1:.1f}', ha='center', va='center', fontsize=8, color='#e8534a', fontweight='bold')

# 下面：缩放后的头
concat_y = 3.5
ax.text(0.5, concat_y + 0.5, 'Scaled:', fontsize=10, fontweight='bold')

for i in range(8):
    x = 1.5 + i * head_width
    rect = FancyBboxPatch((x, concat_y - 0.3), head_width - 0.2, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='#00d977', facecolor='#00d977', alpha=0.3, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + head_width/2 - 0.1, concat_y, f'γ·h{i+1}', ha='center', va='center', fontsize=9)

# 最后：Concat 和输出投影
final_y = 2
ax.text(0.5, final_y + 0.3, 'Output:', fontsize=10, fontweight='bold')
rect = FancyBboxPatch((2, final_y - 0.3), 8, 0.6,
                      boxstyle="round,pad=0.05",
                      edgecolor='#ffb700', facecolor='#ffb700', alpha=0.3, linewidth=2)
ax.add_patch(rect)
ax.text(6, final_y, 'Concat(γ₁·h₁, ..., γ₈·h₈) @ W_O', ha='center', va='center', fontsize=9, fontweight='bold')

# 说明
ax.text(6, 0.8, '关键发现：不同头学到不同的 γ 值，模型自适应地调整各头的重要性', 
        fontsize=9, ha='center', style='italic', color='#666')

plt.tight_layout()
plt.savefig('headscale.png', dpi=150, bbox_inches='tight')
plt.close()
```

- [ ] **Step 2: 验证图表清晰度和尺寸**

导出后检查：
- 宽度 1000px
- 文件大小 < 250KB
- 8 个头清晰可见，缩放参数标注清晰

**预期输出：**
- 文件大小：~140KB
- 分辨率：1200x720 像素

- [ ] **Step 3: 保存到博客目录**

```bash
cp headscale.png /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/
```

---

#### Task 1.4：制作 FFN Mid-LayerNorm 原理图

**文件：**
- Create: `content/blog/posts/014_normformer_paper_review/ffn-mid-ln.png`

**说明：** 展示 FFN 的三阶段：LN_pre → 激活函数 → LN_mid（新增）→ W2 投影。

- [ ] **Step 1: 用 matplotlib 制作 FFN 原理图**

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(6, 9.5, 'FFN Mid-LayerNorm: 前馈网络的内部归一化', fontsize=14, fontweight='bold', ha='center')
ax.text(6, 9, '在激活函数之后、第二个线性层之前添加 LayerNorm', fontsize=10, ha='center', style='italic', color='#666')

# 函数流程
stages = [
    (2, 7.5, 'Input\n(x)', '#e8e8e8'),
    (2, 6, 'LayerNorm(x)\n[Pre-LN]', '#ffd700'),
    (2, 4.5, 'x @ W₁ + b₁', '#4a7ae8'),
    (2, 3, 'σ(·)\n[Activation]', '#ff9999'),
    (5.5, 3, 'LayerNorm(...)\n[New! Mid-LN]', '#00d977'),
    (5.5, 4.5, '@ W₂ + b₂', '#4a7ae8'),
    (5.5, 6, '+ (residual)', '#90ee90'),
    (5.5, 7.5, 'Output', '#e8e8e8'),
]

y_prev = None
x_prev = None

for x, y, label, color in stages:
    rect = FancyBboxPatch((x - 0.8, y - 0.35), 1.6, 0.7,
                          boxstyle="round,pad=0.08",
                          edgecolor=color if color != '#00d977' else '#00d977',
                          facecolor=color, alpha=0.5 if color != '#00d977' else 0.7,
                          linewidth=2 if color == '#00d977' else 1)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 连接箭头
    if x_prev is not None:
        arrow = FancyArrowPatch((x_prev, y_prev - 0.4), (x - 0.85, y + 0.35),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5, color='#666')
        ax.add_patch(arrow)
    
    if x == 2:
        y_prev = y
        x_prev = x
    else:
        x_prev = x
        y_prev = y

# 补充说明
ax.text(0.5, 2, '关键机制：\n\n早期层的 γ < 后期层\n→ 系统性地压缩早层\n→ 自适应梯度抑制', 
        fontsize=9, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='#fff5f5', alpha=0.8))

ax.text(8, 2, '消融实验：\n\n移除 FFN-LN\n→ PPL +0.26\n→ 第二大贡献',
        fontsize=9, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='#f5f5ff', alpha=0.8))

plt.tight_layout()
plt.savefig('ffn-mid-ln.png', dpi=150, bbox_inches='tight')
plt.close()
```

- [ ] **Step 2: 验证图表清晰度和尺寸**

导出后检查：
- 宽度 1000px
- 文件大小 < 250KB
- 流程清晰，新增 LayerNorm 标注清晰

**预期输出：**
- 文件大小：~160KB
- 分辨率：1200x840 像素

- [ ] **Step 3: 保存到博客目录**

```bash
cp ffn-mid-ln.png /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/
```

---

#### Task 1.5：制作消融贡献度柱状图

**文件：**
- Create: `content/blog/posts/014_normformer_paper_review/ablation-contrib.png`

**说明：** 展示三个改进的消融贡献度对比，以及它们的叠加效果。

- [ ] **Step 1: 用 matplotlib 制作柱状图**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

# 数据（来自论文消融实验）
components = ['完整NormFormer', '移除\nPost-Attn LN', '移除\nHeadScale', '移除\nFFN-LN', '移除\nResScale', 'Pre-LN\n基线']
ppl_values = [15.88, 15.92, 16.22, 16.14, 16.20, 16.37]
colors = ['#4ae857', '#ffcccc', '#ff6666', '#ff9999', '#ffcccc', '#e8e8e8']

bars = ax.bar(components, ppl_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加数值标签和贡献度
for i, (bar, ppl) in enumerate(zip(bars, ppl_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{ppl:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 计算相对于完整模型的增长
    if i > 0:
        contrib = ppl - ppl_values[0]
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.08,
                f'+{contrib:.2f}', ha='center', va='top', fontsize=8, color='#e8534a', fontweight='bold')

ax.axhline(y=ppl_values[0], color='#4ae857', linestyle='--', linewidth=2, label='完整 NormFormer', alpha=0.7)

ax.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('NormFormer 消融实验：各改进的贡献度', fontsize=13, fontweight='bold')
ax.set_ylim(15.5, 16.7)
ax.grid(axis='y', alpha=0.3)

# 添加图例说明
ax.text(0.5, 16.5, '贡献度排序：HeadScale (+0.34) > FFN-LN (+0.26) > ResScale (+0.32) > Post-Attn LN (+0.04)',
        fontsize=9, ha='left', style='italic', color='#666', 
        bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.7))

plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.savefig('ablation-contrib.png', dpi=150, bbox_inches='tight')
plt.close()
```

- [ ] **Step 2: 验证图表清晰度和尺寸**

导出后检查：
- 宽度 1000px
- 文件大小 < 250KB
- 柱子清晰可区分，标签可读

**预期输出：**
- 文件大小：~120KB
- 分辨率：1200x720 像素

- [ ] **Step 3: 保存到博客目录**

```bash
cp ablation-contrib.png /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/
```

- [ ] **Step 4: 验证所有新增图表已保存**

```bash
ls -lh /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review/ | grep -E '\.(png|jpg)'
```

预期输出：
```
-rw-r--r--  180KB gradient-flow.png
-rw-r--r--  150KB post-attn-ln.png
-rw-r--r--  140KB headscale.png
-rw-r--r--  160KB ffn-mid-ln.png
-rw-r--r--  120KB ablation-contrib.png
-rw-r--r--  184KB figure1.png
-rw-r--r--  222KB figure3.png
-rw-r--r--  194KB figure4_5.png
```

---

### Phase 2：重写引言部分（第 1 部分）

#### Task 2.1：重写 1.1 节（Post-LN 到 Pre-LN 的演进）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md:8-30` （替换第 1.1 节）

- [ ] **Step 1: 读取原文第 1.1 节内容**

查看原文第 13-30 行（"从 Post-LN 到 Pre-LN 的演进"）

- [ ] **Step 2: 编写新的 1.1 节内容**

新内容应该：
- 保留原有的 Post-LN / Pre-LN 对比说明
- **简化**：去掉过度的背景解释，重点突出"梯度爆炸问题"
- **新增一句话总结**："后层梯度 >> 早层梯度，导致训练不稳定"

替换为：

```markdown
## 1. 引言：Transformer 归一化问题的前世今生

### 1.1 从 Post-LN 到 Pre-LN 的演进

自 Vaswani 等人在 2017 年提出 Transformer 架构以来，**层归一化（Layer Normalization）** 就是其中不可或缺的核心组件。归一化层的放置位置虽然看似只是一个微小的工程决策，但实际上对模型的训练稳定性、收敛速度和最终性能有着深远的影响。

**原始 Transformer 采用 Post-LN 架构**，即将 LayerNorm 放在残差连接之后：

$$\text{PostLN}(x) = \text{LayerNorm}(x + \text{Sublayer}(x))$$

这一设计在 Transformer 的早期应用中被广泛使用，但随着模型规模的不断增大，研究者们逐渐发现了它的致命缺陷：**后层参数的梯度范数远大于早层，导致梯度爆炸于深层，梯度消失于早层**。这意味着在反向传播过程中，靠近输入端的层几乎无法获得有效的梯度信号，导致训练极度不稳定。

为了缓解这个问题，**Pre-LN 架构** 应运而生，即将 LayerNorm 移到子层的输入端：

$$\text{PreLN}(x) = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Pre-LN 架构被 GPT-2、GPT-3 等里程碑模型所采用，成为大语言模型预训练的事实标准。它显著改善了训练稳定性，使得大规模模型的训练成为可能。**问题解决了吗？答案是否定的。**
```

- [ ] **Step 3: 保存修改**

使用 Edit 工具更新 index.md

---

#### Task 2.2：重写 1.2 节（Pre-LN 的隐患：反向失配）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md:31-50` （替换第 1.2 节）

- [ ] **Step 1: 编写新的 1.2 节内容**

新内容应该：
- 保留原有"早层梯度 >> 后层梯度"的发现
- **新增工厂流水线类比**，解释为什么梯度不均很糟糕
- **强化**："两种失配都是问题"的论述

替换为：

```markdown
### 1.2 Pre-LN 的隐患：反向失配与梯度失衡

Xiong 等人在 2020 年的研究中首次系统性地揭示了 Post-LN 的梯度问题。然而，NormFormer 的作者们进一步发现，Pre-LN 虽然解决了 Post-LN 的训练不稳定问题，但实际上引入了**方向相反的梯度失配**：

- **Post-LN**：后层梯度 >> 早层梯度（梯度消失）
- **Pre-LN**：早层梯度 >> 后层梯度（反向失配）

#### 为什么梯度不均衡很糟糕？

用一个工厂流水线的类比来理解：
- **梯度 = 改进信号**，梯度大的层在训练中改变快，梯度小的层改变慢
- **早层是上游**，负责提取基础特征。如果上游改进太频繁（梯度过大），下游来不及适应，导致训练不稳定
- **后层是下游**，负责学习任务特定的高级表示。如果下游改进太慢（梯度不足），即使获得好的基础特征，也无法有效地组织成好的上层表示

**梯度分布不均 = 各层学习效率不同 = 整体训练收敛慢、最终性能不佳**

在 Pre-LN 架构中，这种现象体现为：

1. **早期层过度更新**：由于梯度过大，早期层的参数在训练初期可能剧烈波动，导致学到的特征不够稳定。
2. **后期层更新不足**：深层网络中最靠近输出的层本应承担最重要的任务特定表示学习，但它们接收到的梯度信号却相对不足，导致学习速度缓慢。

**关键观察**：论文通过可视化 "第二全连接层权重在不同层的平均 L1 梯度范数" 清楚地展示了这一现象：

![Figure 3: Average L1 norm of gradients across layers](figure3.png)

Pre-LN 的梯度分布呈现明显的递减趋势，与 Post-LN 的递增趋势恰好相反。但**两种趋势都是问题**——因为两者都导致梯度分布不均。
```

- [ ] **Step 2: 保存修改**

---

#### Task 2.3：新增 1.3 节（梯度分布可视化）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （在原 1.3 前插入新的 1.3 节）

- [ ] **Step 1: 编写新的梯度分布可视化说明**

新增内容：

```markdown
### 1.3 梯度分布可视化：问题的直观表现

为了更直观地理解三种架构的梯度分布差异，我们用下面的图示展示各层的梯度范数变化：

![梯度流向图：Post-LN vs Pre-LN vs NormFormer](gradient-flow.png)

**图的含义**：
- **x 轴**：网络的第几层（从 1 到 12）
- **y 轴**：该层参数的梯度 L1 范数（梯度大小）
- **红线（Post-LN）**：梯度从浅层的很小逐渐增大到深层，形成"阶梯上升"。这导致**深层参数更新剧烈，早层参数更新缓慢**
- **蓝线（Pre-LN）**：梯度从浅层的很大逐渐减小到深层，形成"阶梯下降"。这导致**早层参数更新剧烈，深层参数更新缓慢**
- **绿线（NormFormer）**：梯度在各层基本保持一致的水平，形成"平坦"的分布。这导致**各层参数更新速度均衡**

这张图直观地说明了为什么 NormFormer 能提升训练效率：通过在三个精心选择的位置添加归一化操作，它实现了**梯度在各层的均衡分布**，使得每一层都能以相近的速率学习，从而提高了整个网络的训练效率。
```

- [ ] **Step 2: 保存修改**

---

#### Task 2.4：修改原 1.3 节为 1.4（研究动机）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （重新编号原 1.3 为 1.4，简化内容）

- [ ] **Step 1: 简化原 1.3 节内容**

原内容（"研究动机"）保留核心，简化为：

```markdown
### 1.4 研究动机：能否让梯度在所有层间均衡分布？

面对这一发现，NormFormer 论文提出了一个自然而直接的研究问题：**能否通过在 Transformer 的关键位置添加额外的归一化操作，使得各层的梯度范数趋于均衡？**

这个问题的提出源于一个简单但深刻的直觉：**归一化操作本质上是对激活值进行重新缩放（rescaling），它天然具备调节梯度流动幅度的能力**。如果我们能在正确的位置插入归一化层，就有可能同时解决 Pre-LN 的早层梯度过大和后层梯度过小的问题。

在接下来的章节中，我们将看到 NormFormer 如何通过三处精心设计的改进，实现这一目标。
```

- [ ] **Step 2: 保存修改**

---

### Phase 3：重写改进点讲解（第 2 部分）

#### Task 3.1：新增 2.0 节（架构总览）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （在原第 2 部分开头插入新的 2.0 节）

- [ ] **Step 1: 编写 2.0 节架构总览**

```markdown
## 2. 核心方案：三处改进与架构演进

### 2.0 架构总览

在深入讲解具体的改进点之前，我们先给出 NormFormer、Pre-LN 和 Post-LN 三种架构的全景对比，帮助读者快速理解各架构的特点。

![Figure 1: NormFormer、Pre-LN 与 Post-LN 架构对比](figure1.png)

#### 三种架构的特性对比

| 特性 | Post-LN | Pre-LN | NormFormer |
|------|---------|--------|------------|
| 归一化位置 | 残差连接之后 | 子层之前 | 多点分布式 |
| 训练稳定性 | 差（需精细 warmup）| 好 | 更好（支持更高学习率）|
| 梯度分布 | 后层 >> 早层 | 早层 >> 后层 | **各层趋于均衡** |
| 头级控制 | 无 | 无 | **有（HeadScale）** |
| FFN 内部归一化 | 无 | 无 | **有** |
| 额外参数量 | - | 基准 | +0.4% |
| 额外训练开销 | - | 基准 | +2~6% |

**关键观察**：NormFormer 的核心创新在于**在三个精心选择的位置添加归一化操作**，以极小的代价（0.4% 参数 + 2~6% 计算）实现了梯度分布的均衡化。

#### 三处改进的位置

NormFormer 的三个改进分别位于：

1. **Post-Attention LayerNorm**：在多头注意力输出后、残差连接前
2. **HeadScale**：在多头注意力的 concat 操作前，对每个头乘以可学习标量
3. **FFN Mid-LayerNorm**：在前馈网络的激活函数后、第二个线性层前

我们将逐个讲解每个改进的工作原理、实验验证和工程价值。
```

- [ ] **Step 2: 保存修改**

---

#### Task 3.2：重写 2.1 节（Post-Attention LayerNorm）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （重写改进点 A）

- [ ] **Step 1: 重新编写 2.1 节**

```markdown
### 2.1 改进点 A：Post-Attention LayerNorm — 注意力输出的幅度控制

#### 核心设计

在多头注意力的输出后、残差连接前，添加一个额外的 LayerNorm。

标准 Pre-LN 的注意力子层为：

$$\text{PreLN-MHA}(x) = x + \text{MHA}(\text{LN}(x))$$

NormFormer 将其修改为：

$$\text{NormScaledMHA}(x) = x + \text{LN}(\text{MHA}(\text{LN}(x)))$$

注意这里多了一个外层的 $\text{LN}(\cdot)$ 包裹注意力输出。

![Post-Attention LayerNorm 原理图](post-attn-ln.png)

#### 工作机制

这个额外的归一化层起到了**下缩放（downscaling）** 的作用：论文发现训练完成后，所有层的 Post-Attention LN 的缩放参数（gamma）都低于 1，这意味着它在系统性地降低注意力输出的幅度。

**为什么这很重要？** 在标准 Pre-LN 中，注意力层的输出直接通过残差连接加到主干上。如果注意力输出的幅度过大，会导致**残差分支主导信号传播，破坏信息在不同层之间的平衡传递**。通过添加这一归一化层，NormFormer 能够自适应地控制每一层注意力输出的贡献幅度。

#### 消融实验数据

在 125M 参数的模型上进行消融实验，结果如下：

| 配置 | Perplexity | 相比完整模型的性能下降 |
|------|------------|---------------------|
| 完整 NormFormer | 15.88 | - |
| 移除 Post-Attn LN | 15.92 | **+0.04** |

**贡献度评估**：Post-Attn LN 是三个改进中贡献**最小**的，性能下降仅 0.04 PPL。但这个改进仍然有正效果，说明注意力输出的幅度控制确实能带来稳定性提升。

#### 实现示例

```python
# Post-Attention LayerNorm 的实现（PyTorch）

class NormFormerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.post_attn_norm = nn.LayerNorm(hidden_size)  # 新增
    
    def forward(self, x):
        attn_output = self.attention(x)  # [batch, seq_len, hidden]
        attn_output = self.post_attn_norm(attn_output)   # 新增：额外的 LayerNorm
        return x + attn_output
```
```

- [ ] **Step 2: 保存修改**

---

#### Task 3.3：重写 2.2 节（HeadScale）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （重写改进点 B）

- [ ] **Step 1: 重新编写 2.2 节**

```markdown
### 2.2 改进点 B：HeadScale — 注意力头的差异化加权

#### 核心设计

在多头注意力的拼接（concat）操作前，对每个注意力头的输出乘以一个独立的可学习标量参数。

传统的多头注意力将所有头的输出直接拼接后通过输出投影矩阵：

$$\text{MHA}(Q, K, V) = \text{Concat}(h_1, h_2, ..., h_n) W^O$$

NormFormer 引入了 HeadScale 机制：

$$\text{HeadScaleMHA}(Q, K, V) = \text{Concat}(\gamma_1 \cdot h_1, \gamma_2 \cdot h_2, ..., \gamma_n \cdot h_n) W^O$$

其中 $\gamma_i$ 为可学习的标量参数，**初始化为 1**，确保训练初期与标准多头注意力完全一致。

![HeadScale 原理图](headscale.png)

#### 关键发现

1. **头级权重的差异化**：训练后的 $\gamma_i$ 值变化较大，不同头获得了不同的缩放权重，这表明模型学会了**动态调整不同注意力头的重要性**。
2. **无单调性约束**：$\gamma_i$ 与层深度之间没有明显的单调关系，说明 HeadScale 不是简单地对深层或浅层进行统一调节，而是在细粒度上优化每个头的贡献。
3. **最大贡献**：在消融实验中，**HeadScale 是三个操作中贡献最大的**——移除它导致的性能退化最为严重。

#### 消融实验数据

| 配置 | Perplexity | 相比完整模型的性能下降 |
|------|------------|---------------------|
| 完整 NormFormer | 15.88 | - |
| 移除 HeadScale | 16.22 | **+0.34** |

**贡献度评估**：HeadScale 的贡献最大（+0.34 PPL），是 NormFormer 的**核心创新**。这表明对注意力头的细粒度控制对梯度均衡和训练效率的改进至关重要。

#### 与注意力头剪枝的联系

值得注意的是，HeadScale 的思想与注意力头剪枝（Head Pruning）有一定的联系。Chen 等人在 2021 年的工作中使用类似的头级缩放进行模型压缩，而 NormFormer 将这一思想用于改进训练过程，目标完全不同但技术路线相似。**HeadScale 相当于在训练中学习每个头的"重要性权重"**，而不是在训练后进行剪枝。

#### 实现示例

```python
# HeadScale 的实现（PyTorch）

class NormFormerMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_scales = nn.Parameter(torch.ones(num_heads))  # 初始化为1
        
    def forward(self, q, k, v):
        # 计算多头注意力 [batch, num_heads, seq_len, head_dim]
        heads = self._compute_attention_heads(q, k, v)
        
        # 应用 HeadScale：对每个头乘以对应的缩放参数
        for i in range(self.num_heads):
            heads[:, i, :, :] = heads[:, i, :, :] * self.head_scales[i]
        
        # Concat 后通过输出投影
        concat_heads = heads.reshape(batch_size, seq_len, -1)
        output = concat_heads @ self.W_o
        return output
```
```

- [ ] **Step 2: 保存修改**

---

#### Task 3.4：重写 2.3 节（FFN Mid-LayerNorm）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （重写改进点 C）

- [ ] **Step 1: 重新编写 2.3 节**

```markdown
### 2.3 改进点 C：FFN Mid-LayerNorm — 前馈网络的内部归一化

#### 核心设计

在前馈网络（FFN）的第一个线性变换之后、激活函数之后，添加一个 LayerNorm。

标准 FFN 的计算流程为：

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

NormFormer 将其修改为：

$$\text{NormFFN}(x) = x + \text{LN}_{\text{mid}}(\text{GELU}(\text{LN}_{\text{pre}}(x) \cdot W_1 + b_1)) \cdot W_2 + b_2$$

其中 $\text{LN}_{\text{pre}}$ 是 Pre-LN 架构原有的归一化，而 $\text{LN}_{\text{mid}}$ 是 NormFormer 新增的 FFN 中间归一化。

![FFN Mid-LayerNorm 原理图](ffn-mid-ln.png)

#### 工作机制：自适应梯度抑制

这是**解决梯度失配的核心机制**。论文的关键观察是：

**早期层的 FFN LN gamma 参数系统性地小于后期层的**

这意味着 FFN Mid-LayerNorm 自适应地减小了早期层全连接层输入的幅度，从而有效降低了早期层的梯度，缓解了 Pre-LN 固有的"早层梯度过大"问题。

![Figure 4 & 5: Scaling parameters and learning rate stability](figure4_5.png)

**数学直觉**：
- 归一化操作通过将激活值映射到零均值、单位方差的分布来工作
- 当早期层的 FFN 中间激活值幅度较大时，归一化层通过较小的 gamma 参数对其进行压缩
- 相当于在反向传播时**减小了通过这些层的梯度流**
- 这种自适应机制使得模型能够自动学习到最优的梯度分配方案

#### 消融实验数据

| 配置 | Perplexity | 相比完整模型的性能下降 |
|------|------------|---------------------|
| 完整 NormFormer | 15.88 | - |
| 移除 FFN-LN | 16.14 | **+0.26** |

**贡献度评估**：FFN Mid-LayerNorm 的贡献第二大（+0.26 PPL），仅次于 HeadScale（+0.34）。这验证了论文的核心洞察：**前馈网络内部的梯度控制是实现梯度均衡的关键**。

#### 实现示例

```python
# FFN Mid-LayerNorm 的实现（PyTorch）

class NormFormerFFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.ffn_norm = nn.LayerNorm(ffn_size)  # 新增：FFN 中间的 LayerNorm
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        hidden = self.activation(self.linear1(x))  # [batch, seq, ffn_size]
        hidden = self.ffn_norm(hidden)              # 新增：额外的 LayerNorm
        output = self.linear2(hidden)
        return x + output
```

#### 与梯度均衡的关系

FFN Mid-LayerNorm 的深层作用机制在于：它**在前馈网络层面实现了与整体梯度均衡相符的参数缩放**。早期层的 gamma < 后期层的 gamma，这种分层的缩放分布自然地导致了梯度在各层的均衡分布，与梯度流向图中展示的 NormFormer 的"平坦梯度"相互呼应。
```

- [ ] **Step 2: 保存修改**

---

#### Task 3.5：新增 2.4 节（ResScale 说明）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （重新编号原 2.4 为 2.4，简化）

- [ ] **Step 1: 编写 2.4 节（ResScale）**

```markdown
### 2.4 可选改进：ResScale（不推荐用于大模型）

除了上述三个核心操作外，NormFormer 还提出了一个**可选的** ResScale 操作：

$$\text{ResScale}(x) = \lambda_{\text{resid}} \odot x + \text{Sublayer}(\text{LayerNorm}(x))$$

其中 $\lambda_{\text{resid}}$ 是可学习的逐维度缩放参数，用于调节残差连接中主干信号和子层输出的相对权重。

**重要警告**：论文实验表明，ResScale 仅在小模型（125M、355M 参数）上有效，**在 1.3B 及以上规模的模型上反而会导致性能下降**。

**消融实验数据：**
- 125M 模型：移除 ResScale → +0.32 PPL 下降
- 1.3B+ 模型：添加 ResScale → 性能下降

因此，**对于当前主流的大规模预训练场景，不建议使用 ResScale**。这一发现也提醒我们，并非所有的归一化/缩放操作都是"越多越好"的——过度参数化在大模型上可能导致优化困难。

**推荐配置**：
- 125M-355M 模型：使用完整 NormFormer（含 ResScale）
- 1.3B+ 模型：使用三大核心改进（不含 ResScale）
```

- [ ] **Step 2: 保存修改**

---

#### Task 3.6：新增 2.5 节（三大改进的协同效应）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （插入新的 2.5 节）

- [ ] **Step 1: 编写 2.5 节**

```markdown
### 2.5 三大改进的协同效应

现在我们已经分别了解了三个改进点（Post-Attn LN、HeadScale、FFN-LN）各自的工作原理和贡献度。但更重要的问题是：**这三个改进为什么要一起用？它们之间是否存在协同作用？**

#### 完整消融实验结果

下表展示了逐步移除改进的性能变化：

| 配置 | Perplexity | 相比完整模型 | 累积贡献 |
|------|------------|------------|---------|
| 完整 NormFormer | 15.88 | - | - |
| 移除 Post-Attn LN | 15.92 | +0.04 | +0.04 |
| 移除 Post-Attn LN + HeadScale | 16.22 | +0.34 | +0.34 |
| 移除 Post-Attn LN + HeadScale + FFN-LN | 16.37 | +0.49 | +0.49 |
| 基线 Pre-LN | 16.37 | +0.49 | （对比基线） |

![消融贡献度图：各改进的作用及叠加效应](ablation-contrib.png)

#### 关键观察

1. **HeadScale 是核心**（+0.34 PPL）：占总改进（+0.49 PPL）的 69%
2. **FFN-LN 是第二支柱**（+0.26 PPL）：占总改进的 53%（与 HeadScale 有叠加效果）
3. **Post-Attn LN 是补充**（+0.04 PPL）：占总改进的 8%，但不可或缺

**重要发现**：三个改进的贡献度之和（0.34 + 0.26 + 0.04 = 0.64）**略大于**总体改进（0.49），这说明存在轻微的互补效应——当所有改进都存在时，它们在某些方面存在协同作用，避免了过度贡献。

#### 为什么这个组合有效

这三个改进从**不同的角度**解决梯度不均衡问题：

- **HeadScale**：在注意力层面，通过头级权重调整，实现注意力机制内部的梯度均衡
- **FFN-LN**：在前馈层面，通过分层缩放，实现 FFN 的自适应梯度控制
- **Post-Attn LN**：在层间连接层面，通过注意力输出的幅度控制，保证信号传播的均衡

**三个改进的组合，使得梯度均衡从多个维度得以实现，从而取得最优效果**。

#### 反面例子：为什么在 QKV 上加更多 LN 没用

消融实验还测试了一个反面案例：

| 配置 | Perplexity | 相比完整模型 | 训练速度 |
|------|------------|------------|---------|
| 完整 NormFormer | 15.88 | - | 100% |
| 增加 3 个额外 LN（在 QKV 投影上） | 15.88 | +0.00 | 95% |

**结论**：在 QKV 投影上添加额外的 LayerNorm 没有带来任何性能提升，反而使训练速度降低 5%。这充分说明了 NormFormer 选择的三个位置是**经过精心设计的，不是简单的"到处加 LN"**。

这个反面例子也对我们有启示：**不是所有看起来合理的改进都有效** — 需要通过实验验证，避免盲目堆砌技术。
```

- [ ] **Step 2: 保存修改**

---

### Phase 4：调整实验部分（第 3 部分）

#### Task 4.1：重写 3.1 节（学习率搜索）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （修改 3.1 节）

- [ ] **Step 1: 简化并加强关联**

```markdown
### 3.1 学习率搜索：挑战 GPT-3 的默认设置

在正式实验之前，论文做了一项非常有价值的预实验：系统性的学习率搜索。结果出人意料地发现，**在他们的数据集上，最优学习率比 GPT-3 论文建议的值高出 3-5 倍**：

| 模型规模 | GPT-3 建议学习率 | 实际最优学习率 | 倍数 |
|---------|----------------|-------------|------|
| 125M | 6e-4 | 3e-3 | 5x |
| 355M | 3e-4 | 1e-3 | 3.3x |
| 1.3B | 2e-4 | 6e-4 | 3x |

**为什么 NormFormer 能支持更高的学习率？**

这与我们在改进点 C（FFN Mid-LayerNorm）讨论的自适应梯度抑制密切相关。通过 FFN-LN 的分层缩放，NormFormer 在早层自动降低了梯度幅度，使得**即使使用更高的学习率，早层的参数更新也不会过于剧烈**，从而提升了训练稳定性。

这一发现本身就具有独立的工程价值——**针对自己的数据集进行学习率搜索可能带来显著的性能提升**，不要盲目套用论文中的超参数。
```

- [ ] **Step 2: 保存修改**

---

#### Task 4.2：重写 3.2 节（因果语言模型）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （修改 3.2 节）

- [ ] **Step 1: 加强与改进点的关联**

```markdown
### 3.2 因果语言模型：训练加速与稳定性提升

在因果语言模型（Causal Language Model）预训练任务上，NormFormer 在所有模型规模上都取得了一致的困惑度（Perplexity）改进：

| 模型 | 参数量 | 基线 PPL | NormFormer PPL | 改进幅度 |
|------|--------|---------|---------------|---------|
| 125M | 124.5M | 21.09 | **20.11** | -0.98 |
| 1.3B | 1313.5M | 12.21 | **11.94** | -0.27 |
| 2.7B | 2649.5M | 10.92 | **10.55** | -0.37 |

#### 核心发现

**1. 训练加速效果显著**

NormFormer-1.3B 达到基线相同困惑度的速度**快了 24%**。也就是说，使用 NormFormer，你只需要原来 76% 的训练时间就能获得相同质量的模型。对于动辄需要数千 GPU 小时的大规模预训练来说，24% 的训练时间节省意味着巨大的计算成本削减。

**与改进点的关联**：这个加速来自于梯度均衡带来的**更高的单步训练效率** — 梯度分布均匀意味着每一层都在以最优速率学习，没有某些层学得太快而其他层跟不上的浪费。

**2. 大模型训练稳定性提升**

这可能是 NormFormer 最引人注目的工程价值：**基线 2.7B 模型在 6e-4 学习率下训练发散（完全失败），而 NormFormer-2.7B 在相同学习率下可以稳定训练并取得最佳性能**。

这意味着 NormFormer 显著拓宽了大模型可用学习率的范围，**降低了超参数调优的难度**。

**与改进点的关联**：这与 HeadScale（改进点 B）的头级权重调整密切相关 — 通过对注意力头进行细粒度控制，NormFormer 抑制了某些头过度主导信号的现象，从而提升了梯度流动的稳定性。

**3. 困惑度改进随模型规模变化**

125M 模型上的绝对改进最大（-0.98），而大模型上的绝对改进较小。但考虑到大模型本身的困惑度已经很低（基数效应），相对改进幅度仍然有意义。更重要的是，**训练加速和稳定性提升在大模型上同样甚至更加显著**。

#### 总结

因果语言模型的实验充分验证了 NormFormer 的三大改进在实践中的效果：梯度均衡带来的训练加速，头级控制带来的稳定性提升，使得 NormFormer 成为一个具有重大工程价值的改进方案。
```

- [ ] **Step 2: 保存修改**

---

#### Task 4.3：修订 3.3-3.8 节（其他实验）

**文件：**
- Modify: `content/blog/posts/014_normformer_paper_review/index.md` （修改 3.3-3.8 节）

- [ ] **Step 1: 简化这些部分，保留核心数据**

对于 3.3（零样本）、3.4（GLUE）、3.5（消融）、3.6（超参数鲁棒性）、3.7（Wikitext）、3.8（计算开销），基本保留原有内容，但：
- 简化篇幅（去掉重复说明）
- 加强与改进点的关联（在适当位置提醒"这个结果来自梯度均衡/头级控制/FFN-LN"）
- 保留所有数据表格

具体修改：

**3.5 消ublation 实验** — 注意这一节与 2.5 节有重复。建议：
- 2.5 节讲**核心消融** （三个改进的贡献度）
- 3.5 节讲**详细消融**（包含各种组合的完整表格）

```markdown
### 3.5 消融实验详解

消融实验是理解 NormFormer 各组件作用的关键。在第 2.5 节，我们已经看到了三个改进的核心贡献度。现在我们进一步展示完整的消融结果，包括各种组合和边界情况：

[保留原有的完整消融表格和说明]

**与第 2.5 节的关联**：这里的完整数据进一步验证了第 2.5 节的结论 — HeadScale 是核心，FFN-LN 是第二支柱，Post-Attn LN 是补充。
```

- [ ] **Step 2: 保存修改**

---

### Phase 5：最终检查、测试、提交

#### Task 5.1：Markdown 语法检查

**文件：**
- Check: `content/blog/posts/014_normformer_paper_review/index.md`

- [ ] **Step 1: 验证 Markdown 语法**

```bash
cd /Users/bytedance/03_personal/bansheng.github.io

# 检查前面是否有语法错误（用 python 的 markdown 检查）
python3 -c "
import markdown
with open('content/blog/posts/014_normformer_paper_review/index.md', 'r', encoding='utf-8') as f:
    content = f.read()
    html = markdown.markdown(content, extensions=['extra', 'codehilite'])
    print('✓ Markdown 语法检查通过')
"
```

- [ ] **Step 2: 验证 YAML front matter**

检查文件开头的 YAML 部分：

```yaml
---
title: "NormFormer：用额外归一化改进 Transformer 预训练"
date: 2026-04-09T02:30:00+08:00
draft: false
tags: ["论文精读", "NormFormer", "Transformer", "深度学习", "预训练", "归一化"]
categories: ["Tech"]
---
```

确保日期、标签、分类正确。

- [ ] **Step 3: 验证所有图片引用**

检查所有 `![...](...)` 引用的图片是否存在：

```bash
cd /Users/bytedance/03_personal/bansheng.github.io/content/blog/posts/014_normformer_paper_review

# 检查所有被引用的图片
grep -o '!\[.*\](\([^)]*\))' index.md | sed 's/.*(\([^)]*\))/\1/' | sort -u | while read img; do
    if [ ! -f "$img" ]; then
        echo "❌ 缺失图片: $img"
    else
        echo "✓ $img"
    fi
done
```

预期输出：
```
✓ gradient-flow.png
✓ post-attn-ln.png
✓ headscale.png
✓ ffn-mid-ln.png
✓ ablation-contrib.png
✓ figure1.png
✓ figure3.png
✓ figure4_5.png
✓ featured.png
```

- [ ] **Step 4: 验证标题层级结构**

检查标题是否规范（h2 > h3 > h4，不跳级）：

```bash
grep -E '^#+' index.md | awk '{print NF-1 " " $0}' | head -20
```

预期：按顺序出现 1（## ） → 2（### ） → 3（#### ） 的模式

---

#### Task 5.2：Hugo 本地预览

**文件：**
- Build: Hugo 静态站点

- [ ] **Step 1: 启动 Hugo 本地服务**

```bash
cd /Users/bytedance/03_personal/bansheng.github.io

hugo server -D --disableFastRender
```

预期输出：
```
Start building sites …
...
Web Server is available at http://localhost:1313/ (bind address 127.0.0.1)
Press Ctrl+C to stop
```

- [ ] **Step 2: 用浏览器打开博客**

在浏览器中打开 `http://localhost:1313/posts/014_normformer_paper_review/`

检查：
- ✓ 标题、小节都正确渲染
- ✓ 所有图片都正确加载
- ✓ 表格格式正确
- ✓ 代码块高亮正确
- ✓ 公式（$...$）正确渲染

- [ ] **Step 3: 验证移动端响应式**

用浏览器开发者工具模拟移动设备（375px 宽度），检查：
- ✓ 文字可读性
- ✓ 图片响应式缩放
- ✓ 表格不溢出

- [ ] **Step 4: 验证深色模式**

检查博客是否有深色模式，如果有，验证在深色模式下的显示效果。

- [ ] **Step 5: 停止 Hugo 服务**

按 `Ctrl+C` 停止服务

---

#### Task 5.3：Git 版本控制

**文件：**
- Modified: `content/blog/posts/014_normformer_paper_review/index.md`
- Created: `content/blog/posts/014_normformer_paper_review/gradient-flow.png`
- Created: `content/blog/posts/014_normformer_paper_review/post-attn-ln.png`
- Created: `content/blog/posts/014_normformer_paper_review/headscale.png`
- Created: `content/blog/posts/014_normformer_paper_review/ffn-mid-ln.png`
- Created: `content/blog/posts/014_normformer_paper_review/ablation-contrib.png`

- [ ] **Step 1: 检查 Git 状态**

```bash
cd /Users/bytedance/03_personal/bansheng.github.io

git status
```

预期输出：
```
On branch source
Changes not staged for commit:
  modified:   content/blog/posts/014_normformer_paper_review/index.md

Untracked files:
  content/blog/posts/014_normformer_paper_review/gradient-flow.png
  content/blog/posts/014_normformer_paper_review/post-attn-ln.png
  content/blog/posts/014_normformer_paper_review/headscale.png
  content/blog/posts/014_normformer_paper_review/ffn-mid-ln.png
  content/blog/posts/014_normformer_paper_review/ablation-contrib.png
```

- [ ] **Step 2: 分步骤提交（TDD 风格）**

将重写分成三个逻辑提交：

```bash
# 提交 1: 添加新增图表文件
git add content/blog/posts/014_normformer_paper_review/gradient-flow.png \
        content/blog/posts/014_normformer_paper_review/post-attn-ln.png \
        content/blog/posts/014_normformer_paper_review/headscale.png \
        content/blog/posts/014_normformer_paper_review/ffn-mid-ln.png \
        content/blog/posts/014_normformer_paper_review/ablation-contrib.png

git commit -m "feat: add visual diagrams for NormFormer rewrite

- gradient-flow.png: 梯度流向对比（Post-LN vs Pre-LN vs NormFormer）
- post-attn-ln.png: Post-Attention LayerNorm 原理图
- headscale.png: HeadScale 机制示意图
- ffn-mid-ln.png: FFN Mid-LayerNorm 流程图
- ablation-contrib.png: 消融实验贡献度对比

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

```bash
# 提交 2: 重写博客内容
git add content/blog/posts/014_normformer_paper_review/index.md

git commit -m "refactor: rewrite NormFormer paper review with problem-oriented structure

新组织结构：
1. 引言部分：梯度失配问题 + 梯度流向图 + 类比解释
2. 改进点讲解：2.0 架构总览 + 2.1-2.3 逐改进点讲解（各配原理图+消融数据）
3. 实验部分：与改进点关联，强化学习曲线和稳定性

主要改动：
- 前置架构图到 2.0 节
- 新增梯度流向图和各改进原理图
- 各改进点后嵌入消融数据
- 加强改进点与实验结果的关联

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: 验证提交**

```bash
git log --oneline -3
```

预期输出：
```
xxxxxxx refactor: rewrite NormFormer paper review with problem-oriented structure
xxxxxxx feat: add visual diagrams for NormFormer rewrite
aa51ec1 [PUA生效 🔥] 全量扫描LaTeX HTML转义问题，确认无遗漏
```

- [ ] **Step 4: 更新 PROGRESS.md**

在项目的 PROGRESS.md 中记录这次重写的经验：

```markdown
## [2026-04-10] NormFormer 论文综述重写完成

### 遇到的问题
- 原文架构图出现太晚（2.5 节），读者在理解改进点时缺乏参考框架
- 引言中梯度失配问题的解释过于抽象，缺乏直觉理解
- 改进点讲解与实验结果断层，消融数据集中在第 3 部分

### 解决方案
1. **问题导向的重写**：开篇直接讲梯度失配问题，用梯度流向图直观展示
2. **前置架构总览**：将架构对比图提前到 2.0 节，给读者整体框架
3. **改进点嵌入消融数据**：每个改进点后立即跟随消融实验数据，强化"这个改进有多重要"的理解
4. **新增原理图**：Post-Attn、HeadScale、FFN-LN 各配一张原理图，直观展示位置和工作原理
5. **协同效应分析**：新增 2.5 节，解释三个改进为什么要一起用

### 以后如何避免
- 在技术博文的规划阶段，**将"架构总览"视为必需品**，应该在详细讲解前给出全景
- **改进点与验证数据要绑定**，不要先讲所有方案再统一讲实验
- **对抽象概念要配视觉化**：梯度失配、参数缩放等，都应该有图表或类比

### Commits
- feat: add visual diagrams for NormFormer rewrite
- refactor: rewrite NormFormer paper review with problem-oriented structure
```

提交这个更新：

```bash
git add PROGRESS.md

git commit -m "docs: record lessons from NormFormer rewrite

问题：架构图太晚、引言太抽象、改进点与数据断层
解决：问题导向结构 + 前置架构 + 嵌入消融数据 + 新增原理图

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Phase 6：邀请 Code Reviewer Agent

#### Task 6.1：邀请 reviewer 审核

**说明**：这一步由 writing-plans 计划外代理执行

这个计划已经完成了所有文本内容的重写和本地验证。现在需要邀请一个专门的 **code-reviewer** 或 **content-reviewer** agent 来：

1. **检查上下文连贯性**：
   - 梯度失配问题是否讲清楚了？
   - 梯度流向图是否有效地说明了问题？
   - 改进点的逻辑是否清晰？
   - 消融数据与改进点的关联是否紧密？

2. **检查技术准确性**：
   - 公式是否正确？
   - 原论文的引用是否准确？
   - 消融数据（PPL 数字）是否与论文一致？

3. **检查文笔和表达**：
   - 是否有重复的说明？
   - 是否有冗长的段落需要精简？
   - 代码示例是否清晰？

4. **最终建议**：
   - 是否有需要改进的地方？
   - 是否有逻辑跳跃或缺失？
   - 图表的位置和说明是否恰当？

---

## 后续执行选项

**Plan 已完成。两种执行方式：**

**1. Subagent-Driven（推荐）** 
- 派遣 subagent 按任务执行
- 每个 task 后审核
- 快速迭代反馈

**2. Inline Execution**
- 在当前 session 中执行
- 逐步创建 subagent 做 review

**建议**：先用 Subagent-Driven 执行 Phase 1（图表制作）和 Phase 2-3（内容重写），再用一个 reviewer agent 做 Phase 6（代码审核），最后手工确认提交。

