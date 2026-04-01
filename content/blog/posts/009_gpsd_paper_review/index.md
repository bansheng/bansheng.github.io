---
title: "GPSD：让判别式推荐模型也拥有 Scaling Law 的魔力"
date: 2026-03-24T10:00:00+08:00
draft: false
tags: ["论文精读", "推荐系统", "Transformer", "Scaling Law", "生成式预训练"]
categories: ["Tech"]
---

> 本文是关于最新论文《Scaling Transformers for Discriminative Recommendation via Generative Pretraining》（[arXiv:2506.03699](https://arxiv.org/pdf/2506.03699)）的阅读笔记。

在大语言模型（LLM）领域，增加参数量和数据量通常能带来性能的对数线性增长，即 **Scaling Law**。然而在推荐系统领域，特别是用于排序（Ranking）的**判别式模型**（如 CTR/CVR 预测），这一规律却迟迟没有出现。

Meta 团队最近提出的 **GPSD (Generative Pretraining for Scalable Discriminative Recommendation)** 框架，通过引入生成式预训练，成功打破了这一僵局，让判别式推荐模型也能随着参数规模的扩大而变强。

## 1. 为什么推荐大模型容易“掉点”？

在自然语言处理中，数据是密集的 token 流；但在推荐系统中，用户行为极其稀疏（百亿规模的物品 ID，大部分用户只消费过极小一部分）。

论文指出，直接在判别式任务（点击/转化预测）上训练大规模 Transformer，会遇到严重的**数据稀疏导致的过拟合（Overfitting）**：
* 稀疏参数（Embedding 表）极其庞大且难以充分训练。
* 随着模型层数（Dense 参数）增加，泛化误差（Generalization Gap）迅速扩大。
* 结果就是：模型参数越多，表现反而可能不如简单的小模型。

## 2. GPSD：生成式预训练的“桥接”艺术

GPSD 框架的核心思想是：**先用生成式任务训练稀疏参数，再在判别式任务中冻结它们。**

### 2.1 阶段一：生成式预训练 (Generative Pretraining)
作者发现，自回归的生成式任务（预测用户下一个交互物品）对过拟合更有韧性。因为生成式训练通过 Sampled Softmax 引入了广泛的随机负采样，能让庞大的 Embedding 表（稀疏参数）得到更稳健的更新。

### 2.2 阶段二：桥接与冻结 (Sparse Freeze Strategy)
这是 GPSD 最关键的创新点。在将模型迁移到判别式任务（如 CTR 预测）时，采取 **“冻结稀疏参数（Sparse Freeze）”** 的策略：
1. 继承预训练好的 Embedding 参数。
2. **在判别式微调阶段，固定住这些稀疏参数不更新**。
3. 仅更新稠密的 Transformer 层或 MLP Head。

这种策略完美避开了特征稀疏导致的过拟合陷阱，让模型能够专注于学习高阶的特征交叉。

## 3. 架构流程图

{{< mermaid >}}
graph LR
    subgraph Stage1 ["Stage 1: Generative Pretraining"]
        A[用户历史行为序列] --> B(Transformer Decoder)
        B --> C["预测下一个 Item (Sampled Softmax)"]
        C --> D{预训练模型}
    end

    subgraph Stage2 ["Stage 2: Bridging Strategy"]
        D -->|提取| E["稀疏参数: Embeddings"]
        D -->|提取| F["稠密参数: Transformer"]
    end

    subgraph Stage3 ["Stage 3: Discriminative Training (CTR/CVR)"]
        E -->|参数冻结| G(判别式 Transformer)
        F -->|参数迁移| G
        H["上下文特征 + 候选物品"] --> G
        G -->|仅更新稠密参数| I["预估点击率/转化率"]
    end
{{< /mermaid >}}

## 4. 实验结论与收益

GPSD 在多个维度上验证了其有效性：

1. **验证 Scaling Law**：在使用 GPSD 后，模型性能（如 AUC）随着稠密参数（从 13K 扩展到 0.3B）的增加，呈现出符合幂律（Power Laws）的持续增长。
2. **解决过拟合**：显著缩小了模型训练中的泛化差距（Generalization Gap），在大模型规模下依然保持卓越的泛化能力。
3. **线上业务收益**：在 Meta 的真实业务场景中进行了 A/B 测试，核心业务指标取得了显著的正向收益。

## 总结

GPSD 框架证明了推荐系统也可以像 LLM 一样通过 Scaling Up 变得更聪明。它通过生成式任务为庞大的 Embedding 表奠定了坚实的基础，并通过“参数冻结”策略让判别式模型摆脱了过拟合的泥潭。这一研究为未来推荐领域的“基础大模型（Foundation Models）”提供了关键的路径。

**开源代码**：[github.com/chqiwang/gpsd-rec](https://github.com/chqiwang/gpsd-rec)
