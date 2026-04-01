---
title: "TokenMixer-Large: 突破工业级推荐系统的大模型扩展瓶颈"
date: 2026-03-18T10:15:00+08:00
draft: false
tags: ["论文精读", "推荐系统", "深度学习", "字节跳动"]
categories: ["Tech"]
---

# 来源元数据 (Metadata)

- **原文标题**: TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
- **原文链接**: https://arxiv.org/pdf/2602.06563
- **来源**: Arxiv (ByteDance 团队)
- **作者**: Yuchen Jiang, Jie Zhu, Xintian Han, Hui Lu, Kunmin Bai, Mingyu Yang, Shikang Wu 等

---

# 核心摘要 (Executive Summary)

针对工业级推荐系统面临的大模型扩展瓶颈，本文提出了 **TokenMixer-Large** 架构，通过引入“Mixing & Reverting”操作、层间残差、辅助损失以及稀疏 Per-token MoE 等一系列创新，解决了深层网络中的梯度消失、MoE 稀疏化不足以及硬件利用率低等问题，在字节跳动的核心业务（电商、广告、直播）中成功扩展至百亿参数规模，并取得了显著的在线业务增长。

---

# 深度解读 (Deep Dive)

## 核心痛点

随着推荐系统大模型（DLRM）尝试向大规模参数扩展，现有的主流架构（如 RankMixer、Wukong、DHEN）在实际应用中暴露出多个严重瓶颈：

1. **次优的残差设计**: RankMixer 等架构通过 Mixing 操作改变了 Token 的维度和数量，导致前后残差连接时 Token 的语义无法对齐，限制了模型的表现上限。
2. **不纯粹的模型架构**: 由于历史迭代，推荐模型中通常保留了许多琐碎、访存密集型的底层算子（如 LHUC、DCNv2），导致整体模型的计算利用率 (MFU) 极低。
3. **深层网络梯度更新不足**: 传统的 TokenMixer 往往只有浅层配置（如 2 层），随着网络加深，梯度消失问题严重，难以保持训练稳定性。
4. **MoE 稀疏化不足**: 原有的 ReLU-MoE 设计局限于“稠密训练、稀疏推理”范式，并未降低训练成本，且动态激活机制对推理极不友好。
5. **扩展性受限**: 受限于上述原因，工业界之前的探索仅止步于 10亿（1B）参数级别。

## 方法论 (Methodology)

### 1. 整体设计思路

TokenMixer-Large 的设计哲学可以用一句话概括：**以”纯净架构”为基座，通过深度残差与稀疏化实现工业级大模型的高效扩展**。具体来说，团队遵循了以下三条核心设计原则：

- **架构纯净化 (Architecture Purification)**：移除所有历史遗留的碎片化算子，仅保留高计算密度的矩阵乘法操作，最大化 GPU 的 MFU（Model FLOPs Utilization）
- **残差对齐化 (Residual Alignment)**：通过 Mixing-Reverting 的对称设计，确保跨层残差连接的语义一致性，为深层网络训练铺平道路
- **稀疏高效化 (Sparse Efficiency)**：采用 Per-token MoE 实现真正的”稀疏训练+稀疏推理”，在保持模型容量的同时大幅降低计算开销

### 2. TokenMixer-Large 与初代 TokenMixer (RankMixer) 的核心区别

在理解 TokenMixer-Large 的创新之前，我们必须先看清它对初代架构做了哪些大刀阔斧的”革命”。以下是四个核心差异点：

### 2.1 从“维度错位”到“Mixing & Reverting 绝对对齐”
- **初代 TokenMixer 的痛点**：在进行 Token 混合（Mixing）时，将 $T$ 个 token 强行变为 $H$ 个。输入输出维度不匹配，导致直接加和时产生语义错位，无法实现有效的跨层残差连接。
- **Large 版本的解法**：设计了高度对称的“双层结构”：
  * **Mixing 层**：负责跨 Token 混合信息 ($T \rightarrow H$)。
  * **Reverting 层**：专门将混合后的 Token 维度完美恢复到原始状态 ($H \rightarrow T$)。
这种设计确保了输入和输出维度的绝对一致性，构建出平滑且语义对齐的深度残差通道。

```python
# 伪代码演示
# 输入 X: [T, D], T为Token数, D为维度

# 1. Mixing 阶段
H = Split_and_Concat(X) # 将 T 个 token 混合为 H 个, 维度变为 [H, T*D/H]
H_next = Norm(pSwiGLU(H) + H)

# 2. Reverting 阶段
X_revert = Split_and_Concat_Back(H_next) # 将 H 个 token 还原为 T 个, 维度恢复为 [T, D]
X_next = Norm(pSwiGLU(X_revert) + X)     # 语义严格对齐的残差连接
```

### 2.2 从”碎片化算子堆砌”到”纯净架构 (Pure Architecture)”
- **初代 TokenMixer 的痛点**：由于历史迭代，模型中通常堆砌了许多细碎、访存密集型的底层算子（如 LHUC、DCNv2），导致整体模型在 GPU 上的计算利用率（MFU）极低。
- **Large 版本的解法**：剥离所有底层低效交互算子，将 Post-LayerNorm 替换为 Pre-RMSNorm，使用 pSwiGLU 替换 pFFN。完全依靠堆叠纯净的 TokenMixer-Large Block 进行特征交叉，使得核心广告模型的 MFU 飙升至 60%。

关于组件替换的详细说明：
- **Pre-RMSNorm vs Post-LayerNorm**：Pre-RMSNorm 将归一化操作放在子层的输入端而非输出端，省去了均值计算步骤，减少了约 30% 的归一化开销。更重要的是，Pre-Norm 结构使得残差通道中的梯度流动更加顺畅，有利于深层网络的训练稳定性。
- **pSwiGLU vs pFFN**：pSwiGLU（Per-token SwiGLU）将传统的 ReLU 激活替换为 SiLU（Swish）门控线性单元，引入了乘法门控机制，增强了特征的非线性表达能力。其公式为 $\text{SwiGLU}(x) = (xW_1) \otimes \text{SiLU}(xW_2)$，相比传统 FFN 增加了约 50% 的参数量，但带来的效果提升远超参数增长。

### 2.3 从“浅层堆叠”到“深层跨层残差 (Inter-Layer Residuals)”
- **初代 TokenMixer 的痛点**：随着网络加深（如从浅层的 2 层扩展到深层），极易发生梯度消失现象。
- **Large 版本的解法**：采取了“组合拳”：
  * **跨层残差与辅助损失**：每隔 2-3 层引入跨层残差连接，并将底层输出与高层输出结合计算辅助损失（Auxiliary Loss）。
  * **Rezero 初始化**：将 SwiGLU 中最后一个投影矩阵的初始化方差缩小为 0.01，使模块在训练初期接近恒等映射，极大提升了模型收敛的稳定性。

### 2.4 稀疏 Per-token MoE (Sparse-Pertoken MoE) 的进化
- **初代 TokenMixer 的痛点**：原有的 ReLU-MoE 设计局限于“稠密训练、稀疏推理”范式，并未真正降低训练成本，且动态激活对线上推理极不友好。
- **Large 版本的解法**：采用**“先扩大，后稀疏” (First Enlarge, Then Sparse)** 的端到端策略：
  1. 将 Per-token SwiGLU 拆分为多个细粒度的专家（Expert）并进行稀疏激活。
  2. 引入 **门控值缩放 (Gate Value Scaling)** 来解决稀疏化带来的梯度更新不足问题。
  3. 加入 **共享专家 (Shared Expert)** 以稳定训练过程。

这使得模型能够真正实现”稀疏训练与稀疏推理”。在实验中，模型在仅激活一半参数（2.3B out of 4.6B）的情况下，FLOPs 下降近半，但取得了与稠密模型完全相同的业务增益（AUC +1.14%）。

## 与其他方法的对比分析

为了更全面地理解 TokenMixer-Large 的定位和优势，我们从架构设计、训练效率和扩展能力三个维度进行横向对比：

### 维度一：架构设计对比

| 对比项 | DLRM-MLP | Wukong | DHEN | RankMixer | **TokenMixer-Large** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 特征交叉方式 | MLP 隐式交叉 | 双塔交叉网络 | 层次化显式交叉 | Token Mixing | **Mixing & Reverting** |
| 残差连接 | 简单残差 | 跨塔残差 | 层内残差 | 维度错位残差 | **语义对齐残差** |
| 归一化方式 | BatchNorm | LayerNorm | LayerNorm | Post-LayerNorm | **Pre-RMSNorm** |
| 激活函数 | ReLU | ReLU | ReLU | FFN | **pSwiGLU** |
| 稀疏化支持 | 无 | 无 | 无 | ReLU-MoE | **Per-token MoE** |

### 维度二：训练效率对比

| 对比项 | DLRM-MLP | Wukong | RankMixer | **TokenMixer-Large** |
| :--- | :--- | :--- | :--- | :--- |
| GPU MFU | < 10% | ~15% | ~25% | **~60%** |
| 碎片化算子 | 大量 | 中等 | 中等 | **无** |
| 训练稳定性 | 浅层稳定 | 一般 | 深层退化 | **深层稳定** |
| 训练范式 | 稠密 | 稠密 | 稠密训练稀疏推理 | **稀疏训练+稀疏推理** |

### 维度三：扩展能力对比

| 对比项 | DLRM-MLP | Wukong | RankMixer | **TokenMixer-Large** |
| :--- | :--- | :--- | :--- | :--- |
| 已验证最大参数量 | ~500M | ~1B | ~1B | **15B（离线）/ 7B（在线）** |
| Scaling Law 表现 | 早期饱和 | 有限提升 | 中等 | **持续提升** |
| 多业务验证 | 单一场景 | 有限场景 | 有限场景 | **电商/广告/直播** |

## 实验结果详细分析

### Scaling Law 验证

TokenMixer-Large 在离线实验中展现出了清晰的 Scaling Law 特性：

- **500M -> 1B**：CTCVR AUC 持续提升，验证了架构设计的有效性
- **1B -> 4B**：引入 Sparse-Pertoken MoE 后，模型在激活参数仅为 2.3B 的情况下达到了与 4.6B 稠密模型相当的性能
- **4B -> 15B**：离线实验表明 AUC 仍在持续提升，未出现明显的饱和趋势

### 消融实验关键发现

论文中的消融实验揭示了几个重要结论：

- **Mixing & Reverting 的必要性**：去除 Reverting 层后，AUC 下降约 0.15%，证明语义对齐的残差连接对深层网络至关重要
- **跨层残差的贡献**：去除跨层残差后，深层模型（>6 层）出现明显的训练不稳定，AUC 波动加剧
- **Rezero 初始化的作用**：将初始化方差从标准值改为 0.01 后，训练初期的 loss 曲线更加平滑，最终收敛效果提升约 0.08% AUC
- **门控值缩放的影响**：在 MoE 稀疏化场景中，移除门控值缩放会导致约 0.12% 的 AUC 损失，验证了其对缓解稀疏梯度更新不足的有效性

## 流程图 (Flowchart)

{{< mermaid >}}
graph LR
    A[Raw Sparse Features] --> B(Embedding Layer)
    B --> C[Semantic Group-wise Tokenizer]
    C --> D[Global Token + Grouped Tokens X]
    
    subgraph TokenMixer-Large Block
        D --> E[Mixing: Split & Concat]
        E --> F[Pertoken SwiGLU + Norm]
        F --> G[Reverting: Split & Concat back to T]
        G --> H[Pertoken SwiGLU + Norm]
        
        D -.->|Semantic Aligned Residual: F_revert + X| H
    end
    
    H --> I[Deep Layers with Inter-Residual & Aux Loss]
    I --> J[Sparse-Pertoken MoE Layer]
    J --> K[Mean Pooling & Task Prediction]
{{< /mermaid >}}

## 优缺点分析

### 优势

1. **架构简洁高效**：通过彻底移除碎片化算子，TokenMixer-Large 将 GPU MFU 提升至 60%，这意味着同样的硬件资源可以训练更大的模型。这种”少即是多”的设计理念在工业界具有重要的参考价值。

2. **真正的稀疏训练+推理**：不同于 ReLU-MoE 的”稠密训练、稀疏推理”，Per-token MoE 实现了端到端的稀疏化，使得训练成本和推理成本同时降低。这对于大规模在线服务的部署预算控制至关重要。

3. **经过大规模工业验证**：该架构已在字节跳动电商、广告、直播三大核心业务线上线验证，覆盖了推荐系统的主要应用场景，证明了其普适性和鲁棒性。

4. **清晰的 Scaling Law**：实验证明了模型在 500M 到 15B 参数范围内持续受益于规模扩展，为后续进一步扩展提供了明确的方向。

### 不足

1. **Embedding 层优化不足**：论文主要聚焦于排序模型（Ranking Model）的上层架构，对 Embedding 层的优化讨论较少。而在实际工业系统中，Embedding 层往往占据了模型参数量的绝大部分（通常超过 90%），如何高效地扩展 Embedding 仍是一个开放问题。

2. **训练基础设施要求高**：扩展至 7B-15B 参数规模需要大量的 GPU 资源和分布式训练框架支持。论文对多机多卡的并行策略、通信优化等工程细节披露有限，其他团队复现的门槛较高。

3. **冷启动与长尾问题未涉及**：文章主要关注整体指标（AUC、GMV）的提升，未讨论大模型在推荐系统冷启动场景和长尾物品推荐上的表现，而这些恰恰是工业推荐系统的核心痛点。

4. **跨域泛化能力有待验证**：虽然在字节跳动内部三个业务线均有验证，但不同公司的推荐系统在数据分布、特征工程、业务目标上差异巨大，该架构的跨域迁移能力尚需更多外部验证。

## 工程实践启示

对于正在探索推荐系统大模型化的团队，TokenMixer-Large 提供了以下工程实践启示：

### 1. 先做架构”减法”，再做规模”加法”

- 在盲目扩大模型参数之前，优先审视现有架构中的碎片化算子
- 统计各算子的 FLOPs 占比和延迟占比，找出”高延迟、低计算”的瓶颈算子
- 逐步替换为高计算密度的标准化组件（如将各类特征交叉算子统一为矩阵乘法）

### 2. 渐进式扩展策略

- 不要一步跳到超大规模，建议按照 500M -> 1B -> 4B -> 10B 的节奏逐步扩展
- 每个阶段都需要充分的离线实验和在线 A/B 测试验证
- 关注 Scaling Law 曲线的拐点，当 AUC 提升开始饱和时及时调整策略

### 3. 稀疏化是大模型落地的关键

- 对于超过 1B 参数的在线推理模型，MoE 稀疏化几乎是必选项
- 建议同时评估训练和推理两端的稀疏化方案，优先选择能同时降低两端成本的方案
- 门控值缩放和共享专家等稳定化技巧在实践中非常重要，不可省略

### 4. MFU 是核心效率指标

- 将 MFU 纳入模型迭代的核心监控指标
- 目标至少达到 40% 以上（TokenMixer-Large 达到了 60%）
- 低 MFU 往往意味着存在大量的访存瓶颈或通信开销，需要针对性优化

## 结论 (Conclusion)

TokenMixer-Large 验证了在去除历史碎片化算子后，”纯净架构+大规模堆叠”在推荐领域的有效性。模型在离线实验中成功扩展至 **150亿 (15B)** 参数，在线部署达到了 **70亿 (7B)** 参数。在字节跳动核心业务取得巨大收益：

- **电商**: 订单量提升 1.66%，人均 GMV 提升 2.98%
- **广告**: ADSS 提升 2.0%
- **直播**: 收入增长 1.4%

---

# 关键代码/数据

**核心数据对比 (电商场景 500M 规模基线对比)**:

| 模型 | 参数量 | 训练 FLOPs/Batch | CTCVR AUC 提升 |
| :--- | ---: | ---: | ---: |
| DLRM-MLP | 499 M | 125.1 T | 基线 |
| Wukong | 513 M | 4.6 T | +0.76% |
| RankMixer | 567 M | 4.6 T | +0.84% |
| **TokenMixer-Large 500M** | 501 M | 4.2 T | **+0.94%** |
| **TokenMixer-Large 4B SP-MoE** | 2.3B 激活 | 15.1 T | **+1.14%** |

*注：Sparse-Pertoken MoE 在激活仅一半参数（2.3B in 4.6B）的情况下，不仅显著降低了 FLOPs，还达到了与稠密模型完全相同的业务增益，实现了极高的性价比 (ROI)。*

## 总结与展望

TokenMixer-Large 的成功为工业级推荐系统的大模型化树立了一个重要的里程碑。它证明了推荐系统同样遵循 Scaling Law，只要架构设计得当，参数规模的扩展能够持续带来业务增益。

未来值得关注的方向包括：

- **多模态融合**：将文本、图片等多模态特征纳入 TokenMixer 框架，进一步提升推荐质量
- **在线学习与实时更新**：探索大模型在在线学习场景下的高效更新策略
- **模型压缩与蒸馏**：研究如何将大模型的知识高效蒸馏到轻量级模型中，服务于延迟敏感的场景
- **跨业务迁移学习**：探索不同业务线之间的模型迁移和知识共享机制
