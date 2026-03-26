---
title: "FuXi-Linear：释放线性注意力在超长序列推荐中的潜力"
date: 2026-03-19T21:00:00+08:00
draft: false
tags: ["Paper Reading", "Recommendation System", "Linear Attention", "Sequential Recommendation"]
categories: ["Tech"]
---

> 本文是关于最新论文《FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation》（[arXiv:2602.23671](http://arxiv.org/abs/2602.23671v1)）的阅读笔记。

在现代推荐系统中，基于 Transformer 的序列推荐模型已经成为主流。然而，传统的 Softmax 注意力机制具有二次复杂度 $\mathcal{O}(n^2)$，这导致在处理用户的超长行为序列（例如超过 $10^4$ 次交互）时，会面临严重的显存占用和推理吞吐量瓶颈。

虽然“线性注意力（Linear Attention）”架构在 NLP 领域（如 Mamba, RWKV, RetNet）展现了巨大的潜力，但将其直接生搬硬套到推荐系统中却面临着诸多水土不服。FuXi-Linear 就是为了解决这些问题而提出的一种全新架构。

## 1. 线性模型在推荐系统中的三大挑战

论文指出现有线性模型在推荐系统中的应用主要面临以下痛点：

1. **时间信号利用不佳**：现有的方法通常把“时间戳”当作一种特征，直接与物品的“语义特征”拼接或相加。这种强耦合会导致两种信号相互干扰，且无法显式地对用户行为的**周期性**（比如周末才打游戏）进行建模。
2. **位置信息缺失**：传统 Transformer 通常使用相对位置编码（RPE，如 T5, ALiBi）来提供精确的位置感知。但 RPE 的计算是二次复杂度的，无法兼容线性模型的递归（RNN）计算形式。而线性模型自带的自然衰减机制粒度又太粗。
3. **缺乏长序列扩展性**：现有的线性推荐模型大多局限在短序列（长度 $\le 100$）和浅层网络上测试，从未在真正的“超长序列”和“大规模参数”下验证过 Scaling Law。

## 2. FuXi-Linear 的核心架构创新

为了解决上述问题，FuXi-Linear 设计了三个独立的通道来分别处理不同的信号，最后再进行融合。

### 2.1 语义保留通道 (Retention Channel)

采用类似 RetNet 的 Retention 机制替代传统全注意力。通过一个指数衰减矩阵 $D$，模型可以支持“并行训练”和“递归推理”两种模式。
* 在训练时可以高效并行；
* 在推理时可以像 RNN 一样，复杂度降为 $\mathcal{O}(1)$。

$$ Retention(Q,K,V,D) = (QK^T \odot D)V $$
*(其中衰减矩阵 $D_{i,j} = \gamma^{i-j}$，$\gamma$ 为可学习参数)*

### 2.2 线性位置通道 (Linear Positional Channel)

这是本文非常精妙的一个设计。传统的相对位置编码需要两两计算 token 的相对距离，是 $\mathcal{O}(n^2)$ 的。作者通过引入可学习的核函数映射 $\mathbf{k}(x)$，将位置差函数 $f(x-y)$ 展开为内积形式：

$$ f(x-y) \approx g(x,y) = \mathbf{k}^T(x)\mathbf{k}(y) $$

这样一来，就可以在**维持线性递归特性**的同时，完美逼近了相对位置编码（RPE）的效果，让线性模型也拥有了精准的位置感知能力。

### 2.3 时序保留通道 (Temporal Retention Channel)

为了避免时间戳信号和语义信号打架，FuXi-Linear 专门开辟了一条通道。这条通道**完全利用时间戳数据**独立生成 Query 和 Key，专门用来捕获长序列中用户行为的“周期模式”，从而实现了特征解耦。

---

## 3. 架构流程图

{{< mermaid >}}
graph TD
    A[用户历史交互序列 + 时间戳] --> B[Embedding 层]
    B --> C[FuXi-Linear Block]
    
    C -->|输入 X| D1[Retention Channel<br>提取语义信息]
    C -->|时间戳| D2[Temporal Retention Channel<br>提取周期性时序信号]
    C -->|位置信息| D3[Linear Positional Channel<br>相对位置建模]
    
    D1 --> E[Concatenation & Gating<br>特征拼接与门控机制]
    D2 --> E
    D3 --> E
    
    E --> F[MFFN 多阶段前馈网络]
    F --> G[下一个 Item 预测]
{{< /mermaid >}}

## 4. 实验结论与收益

FuXi-Linear 在多个包含千级长度序列的真实数据集上进行了验证，得出了令人振奋的结论：

1. **极致的推理加速**：得益于线性架构，与表现最好的 Transformer 基线相比，FuXi-Linear 在 Prefill（预填充）阶段实现了高达 **10倍** 的加速，在 Decode（自回归解码）阶段实现了高达 **21倍** 的加速。
2. **卓越的推荐质量**：不仅速度快，推荐准确率也超越了现有的 SOTA 模型。
3. **验证 Scaling Law**：论文**首次**在推荐系统的线性架构中验证了稳健的幂律缩放特性（Power-law scaling property），这意味着该模型具备向“超大参数量”和“超长历史序列”扩展的巨大潜力。

**开源代码**：官方代码库已经开源在 GitHub: [USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear)。
