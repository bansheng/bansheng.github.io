---
title: "TokenFormer：终结推荐系统的两个平行世界"
date: 2026-04-15T22:00:00+08:00
draft: false
featured: true
tags: ["推荐系统", "Transformer", "序列建模", "特征交互", "工业落地", "腾讯广告"]
categories: ["Tech"]
---

> 本文基于腾讯广告团队 2026 年 4 月最新发布的论文《TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds》（[arXiv:2604.13737](https://arxiv.org/abs/2604.13737)）撰写。论文提出了一种统一的推荐系统骨干架构，通过两项核心技术创新解决了长期困扰业界的"朴素统一导致序列坍缩传播"问题，并在微信视频号广告系统上取得了 **+4.03% GMV** 的在线收益。

---

## 0. 两个推荐世界的长期割裂

在过去十年里，工业推荐系统悄然形成了两套彼此独立的技术体系：

**第一套：多字段特征交互（Multi-Field Feature Interaction）**

这套体系的核心是处理异构稀疏特征——用户画像、商品属性、上下文信息等来自不同字段的类别特征。DIN 用 Attention 做目标感知的历史权重，DCN 用交叉网络显式建模高阶特征交叉，DeepFM 引入因式分解机……无数工作都在探索如何更好地捕捉这些静态特征之间的相关性。

**第二套：序列行为动态建模（Sequential Behavior Modeling）**

这套体系的核心是理解用户兴趣的时序演化——GRU4Rec 用 RNN 建模序列，SASRec 引入 Self-Attention，BERT4Rec 使用双向建模，后来又有 HSTU 在腾讯广告规模下证明了序列 Transformer 的价值……这套范式专注于从用户的行为轨迹中挖掘动态偏好。

两套体系共享相同的计算基元（Embedding、Attention），却长期平行演进，极少交融。现代工业推荐系统往往通过<strong>拼接</strong>的方式将两者整合：将各字段特征过一套交互模块，将序列特征过另一套序列模块，最后把两个模块的输出 concat 进入后续网络。

这种异构拼接的方式显然不够优雅。自然的问题是：**能否用一个统一的 Transformer 架构，直接端到端地处理所有输入？**

答案是可以的——但论文发现，朴素地统一这两类特征会触发一种此前未被识别的失效模式。

---

## 1. 发现问题：序列坍缩传播

### 1.1 朴素统一为什么会失败

最直观的统一方案是：将所有输入——多字段特征、序列行为、目标特征——展平为一条 token 流，然后喂给标准 Transformer。理论上，全注意力机制应该能够自行学习哪些 token 之间需要交互。

但实验告诉我们，这种方案会显著劣于精心设计的异构架构。论文通过仔细的表示分析找到了原因：<strong>序列坍缩传播（Sequential Collapse Propagation，SCP）</strong>。

**现象**：非序列字段（如用户画像、上下文特征）的嵌入维度通常较低，而序列行为的建模需要更高的表示维度来承载时序动态。当低维非序列 token 与高维序列 token 在全注意力中充分交互时，序列表示会发生<strong>维度坍缩</strong>——有效秩（effective rank）急剧下降，序列 token 的表示趋于同质化，丧失区分度。

可以用一个类比来理解：你在听一场 80 人的交响乐演奏时，如果强行让乐手们去迁就一位经验有限的独奏者的节奏，整体音乐的层次感反而会被拉低。

### 1.2 从谱分析看坍缩

论文通过分析 Transformer 各层的<strong>有效秩（erank）</strong>来量化这一现象。有效秩衡量的是一个矩阵在多少个奇异值方向上有实质性的"能量"——有效秩越高，表示越丰富，区分度越强。

在朴素统一的 Transformer 中，随着层数加深，序列 token 的表示矩阵谱衰减越来越陡峭：大量信息被压缩到少数几个主方向，模型失去了表达多样化序列模式的能力。

与此同时，论文还发现了另一个浪费：在深层网络中，序列 token 会<strong>反常地向非序列位置分配大量注意力权重</strong>（平均 40.0 vs 序列内部的权重），尽管这种跨域注意力在深层并没有实质性收益。

---

## 2. TokenFormer 架构设计

论文提出了 TokenFormer，通过两项互补的技术创新来解决上述问题。

### 2.1 统一令牌流

所有输入首先被组织为一条扁平化的令牌流：

$$\mathbf{S} = [\underbrace{f_1, f_2, \ldots, f_m}_{\text{非序列字段} \mathcal{F}}, \underbrace{t_1, t_2, \ldots, t_n}_{\text{序列行为} \mathcal{T}}, \underbrace{v_1, \ldots, v_k}_{\text{目标特征} \mathcal{V}}]$$

与其他统一方案不同，TokenFormer 使用 **RoPE（旋转位置编码）** 而非类型嵌入来区分不同段落。RoPE 通过位置感知索引方案，让模型在注意力计算阶段自然感知 token 的位置属性，而不需要额外引入分段标记。

### 2.2 BFTS：底部全注意力，顶部滑动窗口

这是 TokenFormer 的第一个核心创新：**分层注意力设计（Bottom Full-attention, Top Sliding-window，BFTS）**。

{{< mermaid >}}
graph TB
    subgraph "浅层（l ≤ lf）：全注意力"
        L1["Layer 1\n非序列 ↔ 序列\n全局特征融合"]
        L2["Layer 2\n跨域交互完成"]
    end
    subgraph "深层：收缩滑动窗口"
        L3["Layer 3\n窗口 w1\n序列局部建模"]
        L4["Layer 4\n窗口 w2 &lt; w1\n精细时序优化"]
        L5["Layer 5\n窗口 w3 &lt; w2\n近邻感知"]
    end
    L2 --> L3
    L3 --> L4
    L4 --> L5
    note["非序列 token\n在深层完全禁止\n关注序列位置"]
{{< /mermaid >}}

**设计逻辑如下：**

**浅层（$\ell \leq \ell_f$）使用全因果注意力**：在这个阶段，让所有 token 充分交互，完成跨域特征融合。非序列字段的静态信息需要在这里"注入"到序列表示中。

**深层使用收缩窗口滑动注意力（SWA）**：一旦全局交互完成，深层应该专注于序列内部的局部时序建模。窗口大小随层数递减（$w_1 \gt w_2 \gt \cdots \gt w_{L_s}$），让网络从粗粒度到细粒度地精炼序列表示。

**关键约束**：在深层，**完全禁止序列 token 关注非序列位置**。这解决了前面提到的"反常跨域注意力"浪费问题，让深层注意力专心处理时序动态。

消融实验清楚地验证了这一设计的必要性：

| 配置 | 相对 AUC 变化 |
|------|-------------|
| 全部使用全注意力（基线 Transformer） | 0 |
| 全部使用滑动窗口（4S） | **−36.35‰**（灾难性失败） |
| 仅 BFTS | +4.91‰ |
| 完整 TokenFormer | +8.15‰ |

全 SWA 配置的灾难性失败(-36.35‰)说明：**早期的全局特征融合是不可或缺的**。序列建模需要先"看见"上下文全貌，再聚焦局部。

### 2.3 NLIR：非线性交互表示

这是 TokenFormer 的第二个核心创新：**非线性交互表示（Non-Linear Interaction Representation，NLIR）**。

标准 Transformer 的注意力输出经过残差连接直接送入下一层：

$$\mathbf{X}^{(l+1)} = \mathbf{X}^{(l)} + \text{Attn}(\mathbf{X}^{(l)})$$

TokenFormer 在注意力输出处插入了一个门控机制：

$$\mathbf{G}^{(l)} = \mathbf{X}^{(l)} \mathbf{W}_g^{(l)} \quad \text{（门投影）}$$

$$\tilde{\mathbf{I}}^{(l)} = \sigma(\mathbf{G}^{(l)}) \odot \mathbf{A}^{(l)} \quad \text{（乘法调制）}$$

其中 $\sigma$ 为 Sigmoid 函数，$\mathbf{A}^{(l)}$ 是注意力输出，$\odot$ 是逐元素乘法。

**为什么这样设计？**

Sigmoid 门控引入了非线性变换，本质上是让注意力输出的每个维度通过"开关"进行动态选通。这有两个作用：

1. **恢复有效秩**：线性注意力本身是低秩操作，难以避免秩退化。Sigmoid 非线性打破了线性的秩约束，为序列表示注入了更丰富的维度多样性。

2. **自适应梯度调制**：门控参数在训练中自动学习，早期层的门控值趋向于更保守（保留更多原始信息），深层的门控值更积极（筛选关键模式）。这与 FFN Mid-LayerNorm 在 NormFormer 中发挥的作用类似——模型自动学习各层之间的信息流量分配。

论文通过<strong>互信息（Mutual Information）</strong>分析验证了 NLIR 的效果：在不同聚类数 K 下，BFTS+NLIR 的组合在各层一致提升了表示的区分度，单独使用任一模块也有显著收益。

---

## 3. 实验结果

### 3.1 离线基准对比

论文在 KuaiRand-27K 数据集上进行了全面的离线评估，与多个推荐系统 Baseline 对比：

**用户中心（User-Centric）设置**：

| 模型 | AUC 相对提升（vs Transformer 基线）|
|------|----------------------------------|
| OneTrans | −1.71‰ |
| HyFormer | +4.47‰ |
| **TokenFormer-S** | **+5.76‰** |
| **TokenFormer-L** | **+8.15‰** |

**新印象优化（New Impression Optimization）设置**：

| 模型 | AUC 相对提升（vs Transformer* 基线）|
|------|-----------------------------------|
| OneTrans* | +4.98‰ |
| HyFormer* | +0.98‰ |
| **TokenFormer-S*** | **+11.42‰** |

TokenFormer 在两种设置下均大幅领先此前的统一推荐架构，证明了 BFTS+NLIR 的有效性。

值得注意的是，HyFormer 在新印象优化设置下出现了明显退化，而 TokenFormer 在两种设置下都保持了稳健的提升——这反映了统一架构的泛化能力。

### 3.2 效率与效果的权衡

论文探索了 BFTS 配置（全注意力层数 + 滑动窗口层数）对效率的影响：

{{< mermaid >}}
graph LR
    subgraph "BFTS 配置探索"
        Config1["4F（全注意力）\n基线：AUC 0‰, GFLOPs 基准"]
        Config2["3F1S\n+0.21‰, −62.0‰ GFLOPs"]
        Config3["2F2S\n+0.85‰, −201.0‰ GFLOPs"]
        Config4["1F3S\n+0.05‰, −348.0‰ GFLOPs"]
    end
    Config1 --> Config2 --> Config3 --> Config4
{{< /mermaid >}}

**最优配置是 2F2S**：2 层全注意力 + 2 层滑动窗口，相比全注意力基线<strong>同时提升 AUC（+0.85‰）并大幅降低计算量（-201.0‰ GFLOPs）</strong>。这验证了 BFTS 的设计不只是为了精度，也为工业部署提供了显著的效率优化。

窗口大小的选择也有讲究：窗口 [32, 16] 优于均匀窗口和其他尺寸，收缩模式（从粗到细）优于均匀模式。

### 3.3 表示质量分析

论文通过两个维度量化了 TokenFormer 在表示质量上的改善：

**有效秩（Effective Rank）分析**：

在朴素 Transformer 中，序列 token 的表示矩阵谱衰减随层数加深而急剧恶化——大量奇异值趋近于零，表示实际上坍缩到极低维度空间。TokenFormer 引入 NLIR 后，各层有效秩显著高于基线，特别是在深层仍能维持丰富的表示维度。

**注意力模式分析**：

在浅层，TokenFormer 中静态（非序列）token 接收到的注意力权重（平均 52.7）高于 Vanilla Transformer（40.0），说明跨域融合更充分。在深层，TokenFormer 完全屏蔽了序列 token 对非序列位置的关注，而 Vanilla Transformer 仍在"浪费"注意力容量在无效的跨域交互上。

### 3.4 消融实验细节

| 变体 | AUC 相对基线 | 说明 |
|------|------------|------|
| Transformer（基线） | 0 | 朴素统一 |
| +NLIR 仅 | +4.87‰ | 非线性门控 |
| +BFTS 仅 | +4.91‰ | 分层注意力 |
| +NLIR +BFTS（TokenFormer） | **+8.15‰** | 完整方案 |
| 4S（全 SWA） | −36.35‰ | 无全注意力失败 |

两个组件各自贡献约 +4.9‰，合并后达到 +8.15‰，略有超加性效果，说明两种机制在解决 SCP 问题上具有互补性：NLIR 从表示维度出发恢复秩丰富度，BFTS 从注意力结构出发分离跨域融合与序列精炼。

### 3.5 在线 A/B 测试

论文报告了 TokenFormer 在**微信视频号广告系统**的上线结果：

- **测试时间**：2026 年 1 月至 2 月
- **流量曝露**：5% 流量
- **GMV 提升**：**+4.03%**（相对基线）

对于视频号广告这样体量的商业化系统，+4% 的 GMV 是非常显著的在线收益，这也是对 TokenFormer 在工业规模下有效性的最直接验证。

论文还报告了模型缩放（Scaling）实验：从 TokenFormer-T（Tiny）到 TokenFormer-L（Large），在公开数据集上性能持续提升；在腾讯广告平台的内部数据集上，更大规模的模型没有出现饱和迹象，暗示了 TokenFormer 在工业规模数据下的 Scaling 潜力。

---

## 4. 核心洞察与延伸思考

### 4.1 为什么"朴素统一"是个陷阱

TokenFormer 最有价值的贡献之一，是对"序列坍缩传播"现象的精确识别与命名。在此之前，业界普遍的经验是"多字段交互模型和序列模型各自调好再合"，而 TokenFormer 揭示了这背后隐藏的理论原因：这两类特征的维度分布天然不匹配，强行在全注意力下交互会导致高维空间向低维空间的单向坍缩。

这个分析框架对其他推荐系统设计场景也有启发意义。每当我们看到"多种输入的统一建模"设计时，都应该问：**不同模态/类型的输入之间，维度分布是否相容？它们的交互应该在何时、何处发生？**

### 4.2 BFTS 的架构直觉

BFTS 的设计体现了一种"先全局融合，后局部精炼"的计算哲学。这与 Cross-Mask Transformer（视频号序列建模）中"跨域交叉 vs 域内交叉"的思路有相似之处，也呼应了 NLP 领域 Longformer、BigBird 等高效注意力架构的设计经验——在不同层次使用不同粒度的注意力是有理论依据的。

从工程角度，BFTS 带来了双赢：精度上升（序列 token 不再浪费注意力于无效跨域交互），计算下降（滑动窗口将平方复杂度降为线性）。2F2S 配置的 −201.0‰ GFLOPs 对工业部署而言意义重大。

### 4.3 NLIR 与 SwiGLU 的关系

NLIR 的门控形式 $\sigma(G) \odot A$ 与近年 LLM 中广泛使用的 **SwiGLU/GLU** 机制有异曲同工之处：

$$\text{SwiGLU}(X) = \text{SiLU}(XW_1) \odot (XW_2)$$

两者都利用非线性门控来增强特征的表达能力和选择性。NLIR 将这一思想应用到注意力输出的后处理环节，而非 FFN 内部，针对性地解决了序列表示的维度坍缩问题。这种跨领域的技术迁移也反映了推荐系统与 LLM 研究之间越来越深的融合趋势。

### 4.4 统一 vs 专家系统

TokenFormer 的出现引出了一个更深层的架构选择问题：在工业推荐中，**统一骨干**（unified backbone）和**混合专家**（mixture of experts / heterogeneous modules）哪种路线更有前途？

统一骨干的优点是参数共享、端到端优化、结构简洁、易于迭代。专家系统的优点是可以为不同类型的输入设计最适合的归纳偏置。

TokenFormer 的实验结果表明，一个设计精良的统一骨干**可以超过异构专家组合**。但这需要精确识别并解决统一化过程中出现的失效模式（如 SCP），不能简单地"把所有东西扔进一个 Transformer"。

---

## 5. 总结

TokenFormer 是一篇解决了一个真实工程问题的论文。它的贡献链路清晰：

1. **识别问题**：朴素统一多字段特征与序列特征导致序列坍缩传播（SCP）
2. **分析根因**：低维非序列 token 与高维序列 token 的全注意力交互引发维度坍缩，深层注意力的反常跨域分配造成额外浪费
3. **提出方案**：BFTS（分层注意力，早期全局融合+深层局部精炼）+ NLIR（非线性门控，恢复有效秩）
4. **验证效果**：离线 +8.15‰ AUC，在线 +4.03% GMV

从更宏观的视角来看，TokenFormer 代表了推荐系统架构从"异构组合"向"统一骨干"演进的一步重要尝试。随着 LLM 技术在推荐领域的渗透加深，这条技术路线还有巨大的探索空间：更长的序列、更多的模态输入、更强的跨任务泛化……TokenFormer 展示的"精确识别失效模式 + 针对性机制设计"方法论，将在这些探索中持续发挥价值。

---

> **参考文献：**
> - Zhou, Y., et al. (2026). TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds. [arXiv:2604.13737](https://arxiv.org/abs/2604.13737)
> - Sun, F., et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM 2019.
> - Zhai, J., et al. (2024). Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
> - Wang, R., et al. (2021). DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems. WWW 2021.
