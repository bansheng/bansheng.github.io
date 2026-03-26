---
title: "TokenMixer-Large: 突破工业级推荐系统的大模型扩展瓶颈"
date: 2026-03-18T10:15:00+08:00
draft: false
tags: ["Paper Reading", "Recommendation System", "Deep Learning", "ByteDance"]
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

### 2. TokenMixer-Large 与初代 TokenMixer (RankMixer) 的核心区别

在理解 TokenMixer-Large 的创新之前，我们必须先看清它对初代架构做了哪些大刀阔斧的“革命”。以下是四个核心差异点：

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

### 2.2 从“碎片化算子堆砌”到“纯净架构 (Pure Architecture)”
- **初代 TokenMixer 的痛点**：由于历史迭代，模型中通常堆砌了许多细碎、访存密集型的底层算子（如 LHUC、DCNv2），导致整体模型在 GPU 上的计算利用率（MFU）极低。
- **Large 版本的解法**：剥离所有底层低效交互算子，将 Post-LayerNorm 替换为 Pre-RMSNorm，使用 pSwiGLU 替换 pFFN。完全依靠堆叠纯净的 TokenMixer-Large Block 进行特征交叉，使得核心广告模型的 MFU 飙升至 60%。

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

这使得模型能够真正实现“稀疏训练与稀疏推理”。在实验中，模型在仅激活一半参数（2.3B out of 4.6B）的情况下，FLOPs 下降近半，但取得了与稠密模型完全相同的业务增益（AUC +1.14%）。

## 流程图 (Flowchart)

{{< mermaid >}}
graph TD
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

## 结论 (Conclusion)

TokenMixer-Large 验证了在去除历史碎片化算子后，“纯净架构+大规模堆叠”在推荐领域的有效性。模型在离线实验中成功扩展至 **150亿 (15B)** 参数，在线部署达到了 **70亿 (7B)** 参数。在字节跳动核心业务取得巨大收益：

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
