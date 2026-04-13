---
title: "视频号推荐超长序列技术演进：从端到端到 Cross-Mask Transformer"
date: 2026-04-13T22:00:00+08:00
draft: false
tags: ["推荐系统", "序列建模", "超长序列", "Transformer", "工业落地", "微信视频号"]
categories: ["Tech"]
---

> 本文基于微信视频号技术团队于 2026 年 3 月发布的技术分享《视频号推荐超长序列技术演进》整理而成。这是视频号推荐技术公众号的第一期内容，系统性地回顾了视频号从 2022 年至 2025 年在超长用户行为序列建模上的探索历程，覆盖工程优化、稀疏感知结构、多域暴力建模三大阶段。

---

## 0. 为什么超长序列是推荐系统最重要的 Scaling 方向

用户行为序列提供了对用户兴趣最直接、最明确的描述，是工业推荐模型中最重要的特征来源。**对序列长度的扩展，是推荐模型最清晰的 Scaling 方向之一。**

早期行业普遍采用以 **SIM / TWIN** 为代表的两阶段范式：先用轻量的 GSU（General Search Unit）从超长序列中检索出最相关的 Top-K，再用精细的 ESU（Exact Search Unit）做精确建模。这套范式在算力约束下取得了巨大成功。

然而，两阶段范式存在固有缺陷：
- **GSU 与 ESU 的表征一致性天然存在 gap**，两个模块的优化目标并不完全对齐
- **剪枝丢失信息**：GSU 的检索本质上是一次有损压缩，重要的长尾兴趣信号可能被过滤掉

视频号推荐技术团队从 **2022 年**开始选择另一条技术路线：**全序列端到端建模**，在这个过程中积累了大量宝贵经验。

### 四阶段演进脉络

{{< mermaid >}}
timeline
    title 视频号超长序列技术演进（2022-2025）
    2022 : SIM 两阶段建模
         : 行业验证的基础范式
    2022-2023 : 端到端全序列建模
              : 工程攻坚 + 联训策略
    2023-2024 : 轻量型序列结构
              : 稀疏感知 Efficient Transformer
    2024-2025 : 暴力型序列结构
              : Cross-Mask Transformer 多域感知
{{< /mermaid >}}

> 值得注意的是：文中涵盖的时间跨度非常长，早期方案是在 A100/H20 供应有限的算力约束下设计的折中方案。随着算力丰富，后期暴力方案逐步替代早期方案，但**在算力受限的场景下，早期方案仍具有重要参考价值**。

---

## 1. 端到端全序列建模：打通数据与计算链路

端到端长序列的核心思路是：将万级长度的完整用户行为序列直接引入计算图，以候选视频为 Query 对整个序列做 Target Attention（TA）。挑战主要来自**算力、显存、通信、数据链路**四个层面。

### 1.1 工程优化：让万级序列跑起来

#### 数据链路优化

万级序列使单条样本体积急剧膨胀。关键洞察：**同一用户相邻时刻的序列快照，绝大部分内容相同，仅头尾少量数据变化**。

基于此，设计了**内容哈希分块存储方案**：
- 按视频 ID 哈希值切分序列为若干块
- 相同内容的块跨请求复用
- 配合增量写入 + 按需读取，大幅降低存储和带宽开销

#### 卡间通信优化

在 GPU 同步训练架构下，每步都需跨机交换 embedding 参数，而万级序列使得每条样本包含数万条 feed，跨机通信量随序列长度线性增长。

两层优化策略：

**通用通信优化**：推荐场景下不同用户消费内容高度重叠，利用机内 NVLink 高带宽先完成机内特征去重，再跨机传输，并将相同维度 embedding 合并通信（减少碎片化小包）。

**针对 TA 结构的分布式 Attention**：TA 结构下，序列内部元素无需相互交互，各卡可独立计算局部 attention 后合并结果：

{{< mermaid >}}
graph LR
    Q["候选 Q\n(较小)"] -->|"分发到各卡"| GPU1 & GPU2 & GPU3
    GPU1 -->|"本地 KV 计算局部 attention"| Pool1["局部 Pooling"]
    GPU2 -->|"本地 KV 计算局部 attention"| Pool2["局部 Pooling"]
    GPU3 -->|"本地 KV 计算局部 attention"| Pool3["局部 Pooling"]
    Pool1 & Pool2 & Pool3 -->|"合并结果向量"| Output["最终输出"]
{{< /mermaid >}}

这种设计使得**跨机通信量不再由序列长度主导**，在万级序列长度下可大幅降低训练成本。

#### 显存优化

标准做法中，TA 计算需要 lookup 出序列中每个元素的 embedding，组装成 `[B, L, D]` 的 KV 矩阵，显存开销随序列长度线性增长。

解决方案：**低显存 TA 算子**，将 KV embedding 的 lookup 和 attention 计算合并为一个算子，"边查边算"，不再取出完整 KV 矩阵。

### 1.2 端到端与 SIM 联训：用 1K 逼近长序列收益

实验中发现了一个关键现象：**在端到端建模下，虽然序列输入很大，但实际激活的 key 数量非常少**（采用 element-wise ReLU 激活函数，activation score > 0 才被激活）。有相当比例的样本，激活数量甚至小于 SIM GSU 设置的检索数量。

这意味着：**端到端长序列的收益逻辑，并不主要来自序列更长带来的信息增量，而更可能来自端到端训练带来的表征质量提升。**

基于这一洞察，设计了联训方案：

{{< mermaid >}}
graph TB
    subgraph 共享层
        Emb["底层 Embedding 表征\n(共享参数)"]
    end
    subgraph SIM路径
        Emb --> GSU["GSU 检索 5K"]
        GSU --> ESU["ESU 精确建模"]
    end
    subgraph 端到端路径
        Emb --> Full["端到端全序列 TA\n(1K 长度)"]
    end
    ESU & Full --> Output["模型主体"]
{{< /mermaid >}}

**实验结论**：在 5K 长度 SIM 序列的基础上，引入 1K 长度的端到端全序列，即可取得绝大部分更长全序列的收益。这在算力资源受限时是 ROI 极高的方案。

### 1.3 User Level 样本组织

端到端长序列的计算成本主要集中在 user 侧（序列 IO、多特征 embedding 获取、复杂序列建模），自然想到对同一用户的多个样本进行 user-level 聚合以复用 user 侧计算。

然而，在充分优化的 pointwise 数据流基础上进行聚合，往往引入巨大的 UAUC 折损（完播目标 UAUC 折损 0.3%），这是一个巨大的障碍。后续在第 3.3.2 节会详细介绍最终攻克 user level 数据流的方案。

---

## 2. 轻量型序列结构：稀疏感知 Efficient Transformer

在端到端长序列的基础上，进一步尝试引入 Transformer 结构来捕捉推荐场景的高阶信息。然而，实验显示**在推荐场景直接上标准 Transformer 的 ROI 并不高**：平方计算开销高，相比简单 TA 增益有限。

问题在于：如何对长序列 Transformer 降本增效？

将推荐场景的 Transformer 结构分解为两个子问题：
1. **候选与序列间交互**（Candidate-Sequence Interaction）
2. **序列内部元素交互**（Intra-Sequence Element Interaction）

### 2.1 候选与序列间交互

探索了两种交叉模式：

| 模式 | 结构 | 实验结论 |
|------|------|----------|
| 双向交叉 | 序列→候选（attention）+ 候选→序列（concat + DNN）| 成本较高 |
| 单向交叉 | 仅序列→候选（attention） | 与双向效果相近，成本更低 |

**实验显示序列→候选的交叉重要性占主导地位**，最终采用单向感知结构（与 STCA 类似），在视频号多个场景均体现出良好的通用性。

### 2.2 序列内部元素交互：三种稀疏感知

通过分析短视频场景行为序列上的 Transformer attention 规律，发现每个序列元素有三种典型的感知倾向：

{{< mermaid >}}
graph LR
    subgraph "序列元素的感知模式"
        Item["序列元素 i"] -->|"感知相似 (2.2.1)"| Similar["embedding 相似的视频\n(同质内容)"]
        Item -->|"感知相关 (2.2.2)"| Related["聚类层面相关的视频\n(搭配兴趣)"]
        Item -->|"感知近邻 (2.2.3)"| Neighbor["时序上的前序邻居\n(即时上下文)"]
    end
{{< /mermaid >}}

#### 2.2.1 感知相似

**目标**：对于序列中每个元素，检索出底层 embedding 相似的视频，提取用户在相似视频上的高阶序列信息。

朴素方案（Top-k 相似视频做 Transformer）存在问题：由于 top-k 集合内视频相似度过高，attention 产生 **over-smooth** 问题，各 token 表征趋于一致，丧失区分度。

最终方案：**Top-k All-concat DNN + 余弦相似度压缩**

关键技巧：
- 利用高相似度序列的**低秩性质**，将各视频 embedding 映射到以候选视频为轴的一维子空间
- 用 `<cosine_similarity, side_info>` 序列逼近原始序列信息，大幅压缩 DNN 输入维度（信息损失很小）
- 按 `cos_sim` 对序列排序，使 DNN 输入结构化，提升训练效果

#### 2.2.2 感知相关

**相关 vs 相似**：相似是直接的 embedding 相近，而相关是**簇层面的相似**——两个视频属于不同兴趣簇，但这些簇之间经常共现（例如"搭配"关系）。

建模方案：**基于 attention 的隐式聚类（参考 Set Transformer）**

{{< mermaid >}}
graph LR
    Seq["原始序列"] -->|"全局 Query 做 Cross-Attention"| Cluster["兴趣聚类中心\n(隐式聚类)"]
    Cluster -->|"Self-Attention\n建模聚类中心间相似度"| ClusterSA["聚类间相关性"]
    ClusterSA -->|"以原序列为 Q\n做 Cross-Attention"| Output["还原到原序列空间"]
    Seq -->|"作为 Query"| Output
{{< /mermaid >}}

#### 2.2.3 感知近邻

**目标**：刻画用户和环境的即时状态，每个 item 感知其前序近邻上下文。

简单的 sliding window SA 有收益但提升不够大，需要强化近邻上下文的**捕捉方式**和**利用方式**。

**上下文捕捉**：在近邻窗口内显式聚合并构造多维 context feature，包含：
- 视频 id、作者、虚拟类目等基础特征
- 时间窗内的统计特征（平均长度、平均播放时长、完播次数、快划次数、时间差等）
- 多个不同宽度的窗口并行，捕捉多尺度上下文信息

**可解释性验证**：实验表明，用户在"当前视频和历史视频上的 label 一致率"随视频相似度和上下文相似度的共同提升而显著提升，证明 context feature 捕捉到了真实的兴趣信号。

**上下文利用**：在 TA 中显式引入视频间的 context 相似度来微调 attention score：

$$\text{TA score}' = \alpha \cdot \text{Sim}(q, k) + (1-\alpha) \cdot \text{ContextSim}(c_q, c_k)$$

**核心 insight**：用户对目标视频展示兴趣，不但依赖是否交互过类似视频，**还依赖两个视频的上下文是否一致**。只有视频内容和上下文环境同时匹配，才是高置信度的兴趣信号。

---

## 3. 暴力型序列结构：Cross-Mask Transformer 多域感知

轻量型结构取得了显著效果，但为什么**理论上能力完备的标准 Transformer 在推荐场景表现不及预期**，而各类先验设计的稀疏感知结构却相对有效？

### 3.1 推荐与文本任务的本质差异

通过深入分析，发现二者在两个维度存在显著差异：

**差异 1：输入形式**

| 任务 | 序列形式 | 特征类型 |
|------|---------|---------|
| NLP（文本） | 1D 序列 | 单一 token | 
| 推荐 | **2D 序列** | 多域异构特征（id、作者、类目、行为等） |

标准 Transformer 处理推荐序列时，通常在 attention 前对多特征做 pre-merge（拼接+投影），这个过程导致**信息混杂**，且序列特征间的高阶交叉结构缺失。

**差异 2：任务性质**

NLP 任务需要捕捉语法关系和语义逻辑，训练目标的**牵引能力强**，MHA 有足够的梯度信号自发分离出有意义的 head。

推荐任务需要捕捉**稀疏的高阶 pattern**（如"完播过同作者视频的用户对该作者的新视频也感兴趣"），目标的牵引能力弱，MHA 很难自动学到所需的细粒度 pattern。

> **实验验证**：显式添加一个 action mask head（只关注完播行为的子序列），比增加同等数量的标准 head 效果更好，说明标准 MHA 在弱信号下无法自动学到这类 pattern。

### 3.2 从 1D 到 2D：Cross-Mask Transformer

针对上述问题，放弃 pre-merge，设计了专为多域序列优化的 **Cross-Mask Transformer**，通过 **Masking** 和 **Crossing** 两种机制实现域内交叉和域间交叉。

#### 2D 序列的两种交叉类型

```
       特征域 1  特征域 2  特征域 3
item_1 [ id_1 ][ cat_1 ][ act_1 ]    ← 横向：域间交叉
item_2 [ id_2 ][ cat_2 ][ act_2 ]
item_3 [ id_3 ][ cat_3 ][ act_3 ]    ↕ 纵向：域内交叉
```

**域内交叉（Intra-Domain）**：单个域内序列元素之间的交叉，挖掘与该域绑定的稀疏高阶特征（如"只看完播视频中的同作者 pattern"）。

**域间交叉（Inter-Domain）**：序列版本的特征交叉，不只是单个元素特征间的交叉，还考虑序列的整体性。

#### Masking 机制

定义两种 mask：
- **Vertical Mask** $M_v^f$：在 attention map 的行方向上，筛选满足特征域 $f$ 条件的视频（如"action=完播"的行）
- **Horizontal Mask** $M_h^f$：在 embedding 维度上，从 concat 特征中提取特征域 $f$ 对应的 embedding

#### Crossing 机制

三步操作实现域内高阶交叉：

1. **QK 交叉**：对每个特征域 $f$，用 $M_h^f$ 提取出 Q 和 K，计算域内的 attention map $A^f$
2. **Attention Map Masking**：对 $A^f$ 施加 $M_v^f$，只保留满足条件的元素交叉
3. **V 交叉**：用 $M_h^f$ 提取出 V，与 masked $A^f$ 相乘得到域内交叉结果

最终实现了 $Q^f$、$K^f$、$V^f$ 之间的高阶交叉：

$$\text{Output}^f = \text{softmax}\left(\frac{Q^f (K^f)^T}{\sqrt{d}} \odot M_v^f\right) V^f$$

#### 整体结构

{{< mermaid >}}
graph TB
    subgraph 输入
        Seq["序列 embedding\n每行=一个视频的多特征concat"]
        Cand["候选 embedding"]
    end

    subgraph "Cross-Mask Transformer"
        subgraph "一阶交叉（TA）"
            TA["无参 Target Attention\n(候选→序列)"]
        end
        subgraph "域内交叉（SA）"
            SA1["Vertical Mask SA\n特征域1: 视频id (全mask)"]
            SA2["Vertical Mask SA\n特征域2: 完播action mask"]
            SA3["Multi-window context SA"]
        end
        subgraph "二阶交叉（MHA）"
            MHA["含参 MHA\n(TA结果 + SA结果)"]
        end
    end

    Seq & Cand --> TA
    Seq --> SA1 & SA2 & SA3
    TA & SA1 & SA2 & SA3 --> MHA
    MHA -->|"concat压缩"| Final["返回模型主体"]
{{< /mermaid >}}

### 3.3 工业落地：三层优化让暴力结构可行

#### 3.3.1 计算与显存优化

**计算共享**：多个 attention head 共享底层 attention map，对同一 attention map 施加不同的 mask，5 个 attention head 实际只需计算 2 套 attention map，大幅降低计算量。

**稀疏化交叉**：通过参数 $\lambda_{f_i, f_j}$ 标记哪些特征域间的交叉是重要的（不是所有两两交叉都保留）。实践中保留的关键交叉：视频id、multi-window context feature、完播行为、互动行为。

**零梯度外推**：Cross-Mask Transformer 在基线 TA 的基础上新增，二者的视频表征处于同一语义空间。因此让 TA 产生梯度训练底层 embedding，Cross-Mask Transformer 关闭对底层的梯度，转化为**纯前向过程**，消除多套大尺寸梯度矩阵。这个设计有很好的物理含义，且在效果几乎不折损的情况下显著降低训练开销。

**零显存计算**：参考 FlashAttention 的分块计算思路，通过分块方式减少 HBM 通信，将计算尽量放在访问速度极快的 SRAM 中完成。

#### 3.3.2 User Level 数据流

经过深入分析，样本聚合的折损主要来自三个因素：

| 折损来源 | 原因 | 视频号特殊性 |
|---------|------|-------------|
| **样本延迟增大** | listwise 数据流天然有更大的延迟 | 视频号一刷曝光量约 12 个视频，高于行业均值，延迟问题更严重 |
| **长尾样本丢弃** | 超长视频样本回流慢，通常被丢弃 | 导致学到的分布有偏，长视频被低估 |
| **训练更新次数减少** | user level 聚合导致 user 子图更新次数减少 | 折损最大的因素 |

**解决方案 1 - 多时间窗口**：将固定条数窗口改为 10 分钟时间窗口，更灵活地平衡聚合程度与实时性。这同时自然解决了长尾丢弃问题（先回流前序视频，等待长视频回流）。

**解决方案 2 - 随机延迟消除数据穿越**：10 分钟窗口导致播放时长 > 10min 的样本不会出现在同一刷中，回流顺序隐式泄露 label 信息。解决方法：针对 > 90s 的视频引入随机延迟：

$$t \sim \mathcal{U}(\min(\text{视频长度}, 10\text{min}), 20\text{min})$$

**解决方案 3 - ListCE Loss**：相比 pointwise loss，listwise loss 直接优化分类面，优化效率更高，且形式更匹配 user level 聚合样本：

$$\mathcal{L}_{\text{ListCE}} = -\sum_i y_i \log \frac{\exp(f_i)}{\sum_j \exp(f_j)}$$

值得一提：**ListCE 不但能提升 UAUC，还能同等程度提升 AUC**，说明并非 hack 了指标，而是真实地学到了更好的表征。

**解决方案 4 - Muon 优化器**：Muon 优化器对 dense 参数梯度做主成分均衡——对梯度 $G = U\Sigma V^T$ 做 SVD，新梯度为 $UV^T$，缺秩梯度强制变换为满秩状态，大幅提升 one-epoch 数据流下的单步迭代效率。

实践中遇到训练不稳定（NaN）问题：精排模型参数形态差异巨大，Muon 的梯度 RMS 与参数尺寸相关，导致大小参数更新幅度差距悬殊。解决方案：将原生缩放因子调整为与尺寸无关的常数，配合 Adam 热启 → Muon 接棒的训练策略。

#### 3.3.3 工程部署

- **混合精度**：bf16/fp32 混合精度降低推理开销，提升 QPM
- **多域算子**：实现了计算共享 + 分块运算的 cross-mask transformer 专用算子
- **Muon 算子优化**：将 Muon 的梯度计算分配到不同 GPU，避免多卡对 dense 梯度的重复计算

---

## 4. 实验收益总结

### 端到端序列

| 优化方向 | 主要收益 |
|---------|---------|
| 工程优化（数据链路 + 分布式 attention + 低显存算子） | 使万级序列端到端训练成为可能 |
| 端到端 + SIM 联训（1K 端到端 + 5K SIM） | 以极低成本逼近长序列效果 |
| 粗排引入端到端序列 Scaling | 持续取得收益（粗排无 SIM 基线） |

> 注：早期精排端到端序列 scaling 收益微弱，推测是在已有 SIM 的基础上，原有结构无法从更长序列中提取差异化信息。直到引入 Cross-Mask Transformer 后才打开局面。

### 轻量型结构

稀疏感知结构（相似感知、相关感知、近邻感知）在视频号多个场景均取得收益，其中相似感知和近邻感知各场景普遍有效，相关感知在部分场景有效。

### 暴力型结构

Cross-Mask Transformer 在**短视频场景和红点场景**取得显著收益，商业化场景效果不显著。

> 采用 Cross-Mask Transformer 替换原结构后，精排端到端序列的 Scaling 效率显著提升，证明**差异化的结构设计是打开序列 Scaling 空间的关键**。

---

## 5. 核心洞察与经验总结

**1. 端到端训练的收益逻辑不只是"序列更长"**

端到端建模的核心价值在于提升表征质量，而非单纯的信息输入增量。这解释了为什么 1K 端到端序列 + 5K SIM 联训就能逼近更长全序列的效果。

**2. 推荐场景的 Transformer 不能照搬 NLP 范式**

推荐序列是 2D 多域序列，目标牵引能力弱，需要专门设计域内交叉和域间交叉结构，不能期望标准 MHA 自动学到稀疏高阶 pattern。

**3. 上下文（Context）是兴趣信号的重要修正项**

用户的兴趣是情境依赖的。只有视频内容和上下文环境同时匹配，才是高置信度的兴趣信号。Multi-window context feature 提供了可解释且有效的上下文建模方式。

**4. 工业落地需要系统性优化**

暴力结构落地不只是"把 Transformer 堆上去"，需要从模型设计（零梯度外推）、算子（零显存计算、计算共享）、数据流（多时间窗口、随机延迟、ListCE）、优化器（Muon）各层面协同优化，才能在可接受的成本下发挥模型的理论能力。

**5. Scaling 的前提是差异化建模**

在已有 SIM 序列的基础上，简单拉长端到端序列并不能带来持续收益。需要设计与 SIM 差异化的建模方式（如 Cross-Mask Transformer 的多域感知），才能打开 Scaling 空间。

---

## 参考资料

- [视频号推荐超长序列技术演进（视频号技术团队，2026年3月）](https://mp.weixin.qq.com/s/xPidLQfNEF-fCCVksT9u_w)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)（HSTU，Meta AI）
- [UltraHSTU](https://arxiv.org/abs/2411.02645)（Meta AI）
- [STCA: Sequential Transformer with Causal Attention for Recommendation](https://arxiv.org/abs/2401.14930)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
