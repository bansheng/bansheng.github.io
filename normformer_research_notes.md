# NormFormer 论文研究笔记

> 论文全称：NormFormer: Improved Transformer Pretraining with Extra Normalization
> 作者：Sam Shleifer, Jason Weston, Myle Ott (Meta AI / Facebook AI Research)
> 发表时间：2021年10月 (arXiv: 2110.09456)
> 代码：https://github.com/facebookresearch/fairseq/tree/main/examples/normformer

---

## 一、核心问题：Pre-LN Transformer的梯度失衡

### 1.1 背景：Post-LN vs Pre-LN

Transformer架构中 LayerNorm 的放置位置一直是关键设计决策：

- **Post-LN（原始Transformer）**：LayerNorm 放在残差连接之后。问题：深层梯度远大于浅层（100倍以上），导致训练不稳定，需要精心的学习率warmup。
- **Pre-LN（GPT-2/3等现代架构）**：LayerNorm 放在残差连接之前。优势：训练更稳定，可以使用更大学习率、更短warmup。但存在**反向问题**：浅层梯度反而远大于深层。

### 1.2 NormFormer发现的核心问题

Pre-LN 虽然解决了 Post-LN 的训练不稳定问题，但引入了新的**梯度幅度不匹配**：
- 浅层（early layers）的梯度幅度远大于深层（later layers）
- 这意味着浅层参数更新过于激进，而深层更新不足
- 限制了模型的有效学习能力和最终收敛质量

---

## 二、算法创新：三重规范化改进

NormFormer 在标准 Pre-LN Transformer 的每一层中添加了**三个额外的规范化操作**，以平衡各层梯度幅度。

### 2.1 Head-wise Scaling（注意力头缩放）

**核心思想**：为每个注意力头学习一个独立的缩放系数。

```
HeadScaleMHA(Q, K, V) = Concat(γ₁h₁, γ₂h₂, ..., γₙhₙ) W^O
```

- γᵢ 是每个头的可学习标量参数，初始化为1
- 允许模型动态调整每个注意力头的重要程度
- 参数量增加极少（仅增加 n_heads 个标量参数）
- 消融实验中，移除HeadScale对性能影响最大（0.34 PPL退化）

**工程意义**：类似于一种轻量级的"注意力头剪枝"信号，模型可以自动学习抑制不重要的头。

### 2.2 Post-Attention LayerNorm（注意力后归一化）

**核心思想**：在多头注意力输出之后、残差连接之前，添加一层 LayerNorm。

```
标准Pre-LN:  x + MHA(LN(x))
NormFormer:  x + LN₂(MHA(LN₁(x)))
```

- 控制注意力输出的特征幅度
- 实验发现这些 LayerNorm 的 γ 参数在所有层都保持 < 1，起到下缩放作用
- 帮助抑制浅层的过大梯度

### 2.3 FFN Mid-LayerNorm（前馈网络中间归一化）

**核心思想**：在FFN的第一个全连接层的激活函数之后，添加 LayerNorm。

```
标准FFN:   W₂ · σ(W₁x + b₁) + b₂
NormFormer: W₂ · LN(σ(W₁x + b₁)) + b₂
```

- 管理FFN中间表示的尺度
- 浅层的 γ 参数较小 → 减小浅层梯度幅度
- 深层的 γ 参数较大 → 保持深层梯度幅度

### 2.4 可选：Residual Scaling（残差缩放）

```
ResScale: x · α + sublayer(x)
```

- α 是可学习标量，初始化为1
- 仅对小模型（125M、355M）有效，提升 0.04-0.35 PPL
- 大模型（1.3B+）反而有害，可能导致发散
- 因此在大模型配置中不推荐使用

---

## 三、完整架构对比

### 标准 Pre-LN Transformer Layer：
```
x → LN → MHA → + residual → LN → FFN → + residual
```

### NormFormer Layer：
```
x → LN → MHA → HeadScale → LN → + residual → LN → FC₁ → σ → LN → FC₂ → + residual
```

**参数开销**：仅增加 ~0.4% 参数量（几乎可忽略）
**内存开销**：增加 2-6%（取决于模型大小）

---

## 四、实验结果详解

### 4.1 预训练困惑度（Causal LM）

| 模型规模 | Baseline PPL | NormFormer PPL | 改进幅度 |
|---------|-------------|----------------|---------|
| 125M    | 21.09       | 20.11          | -0.98   |
| 355M    | 14.85       | 14.52          | -0.33   |
| 1.3B    | 12.21       | 11.94          | -0.27   |
| 2.7B    | 10.92       | 10.55          | -0.37   |

**关键发现**：在所有模型规模上都有一致的改进。

### 4.2 收敛速度

- **因果语言模型**：NormFormer 达到相同困惑度只需 ~60% 的计算量
- **掩码语言模型**：达到相同困惑度只需 ~57% 的计算量
- **1.3B模型**：达到等效困惑度快 **24%**，或在相同计算预算下困惑度降低 0.27

### 4.3 零样本下游任务（Zero-Shot）

测试任务：HellaSwag, PIQA, WinoGrande, StoryCloze, OpenBookQA

| 模型规模 | Baseline | NormFormer | 提升 |
|---------|----------|------------|-----|
| 1.3B   | 63.6%    | 64.7%      | +1.1% |
| 2.7B   | 66.3%    | 68.7%      | +2.4% |

**值得注意**：2.7B模型的提升幅度大于1.3B，暗示规模越大收益越明显。

### 4.4 GLUE微调结果（Masked LM → RoBERTa）

| 指标 | Baseline | NormFormer | 提升 |
|-----|----------|------------|-----|
| 平均 | 83.77%   | 85.69%     | +1.92% |

在CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2 等所有子任务上均有改进。

### 4.5 消融实验

| 配置 | Valid PPL |
|------|-----------|
| 完整NormFormer + ResScale | 15.88 |
| 去掉 Post-Attn LN | 15.92 (+0.04) |
| 去掉 FFN LN | 16.14 (+0.26) |
| 去掉 HeadScale | 16.22 (+0.34) |
| 去掉 ResScale | 16.20 (+0.32) |
| Baseline（全去掉）| 16.37 (+0.49) |

**关键结论**：
- 每个组件都有贡献，移除任何一个都会导致性能退化
- HeadScale 贡献最大（移除后退化 0.34 PPL）
- FFN LN 次之（移除后退化 0.26 PPL）
- Post-Attn LN 贡献相对最小但依然有正面影响

### 4.6 学习率稳定性

NormFormer 可以容忍比 Pre-LN baseline 高 **1.4-1.6倍** 的峰值学习率而不发散。这意味着更大的训练灵活性和调参空间。

---

## 五、梯度分析深入解读

### 5.1 三种架构的梯度分布特征

- **Post-LN**：深层梯度 >> 浅层梯度（100倍以上差距）→ 训练不稳定
- **Pre-LN**：浅层梯度 >> 深层梯度 → 训练稳定但学习不均衡
- **NormFormer**：各层梯度幅度**趋于平衡** → 既稳定又高效

### 5.2 NormFormer如何实现梯度平衡

1. **FFN LN 的 γ 参数**：浅层 γ 较小 → 抑制浅层梯度；深层 γ 较大 → 保持深层梯度
2. **Post-Attn LN 的 γ 参数**：全部层 γ < 1 → 全局下缩放注意力输出
3. **HeadScale 参数**：不依赖于层深度，独立调整每个头的重要性

这形成了一种**自适应的梯度流调节机制**，模型通过学习归一化参数来自动平衡梯度分布。

---

## 六、工程实践价值

### 6.1 实现简单性

在fairseq中仅需添加三个命令行参数即可启用：
```bash
--scale-attn    # 启用Post-Attention LayerNorm
--scale-fc      # 启用FFN Mid-LayerNorm  
--scale-heads   # 启用HeadScale
```

小模型可额外添加：`--scale-resids`（残差缩放）

### 6.2 计算成本极低

- 参数增加 < 0.07%（官方数据），约 0.4%（包含LN参数）
- 内存增加 2-6%
- 训练速度几乎无影响（每步增加可忽略不计）
- 但因收敛更快，**总训练成本实际降低 24-43%**

### 6.3 适用场景

- **预训练大语言模型**：直接适用，尤其是使用Pre-LN架构的模型
- **微调场景**：GLUE等下游任务显示一致改进
- **因果LM和掩码LM**：两种范式都受益
- **模型规模**：125M到2.7B均验证有效

### 6.4 局限性

- 仅在语言模型上验证，视觉Transformer等领域未探索
- 学习率配置与GPT-3参考值不同，可能影响可比性
- 未探索是否需要在每一层都添加所有三个操作
- 大模型中ResScale不适用

---

## 七、在Normalization技术演进中的位置

### 技术演进线索

1. **LayerNorm (2016)** → 基础规范化技术
2. **Post-LN (原始Transformer, 2017)** → 第一代架构
3. **Pre-LN (2019-2020)** → 解决训练稳定性
4. **NormFormer (2021)** → 解决Pre-LN的梯度失衡 ← **本文**
5. **RMSNorm (LLaMA, 2023)** → 简化计算，去掉均值中心化
6. **DeepNorm (2022)** → 支持1000层深度Transformer
7. **Peri-LN (2025)** → 继续探索最优归一化位置

### NormFormer的历史贡献

- 首次系统性地揭示了Pre-LN的梯度失衡问题
- 提出了**几乎零成本**的解决方案
- 证明了"归一化位置和数量"是Transformer架构设计的重要维度
- 影响了后续DeepNorm、Peri-LN等工作
- HeadScale思想影响了注意力头重要性评估研究

---

## 八、博客写作建议

### 推荐标题方向
- 《NormFormer：三个小改动让Transformer训练快24%》
- 《Transformer归一化的进化之路：从LayerNorm到NormFormer》

### 内容骨架建议

1. **引言**：为什么Normalization在Transformer中如此重要（从训练不稳定说起）
2. **问题揭示**：Post-LN vs Pre-LN的梯度分布问题（配图说明）
3. **NormFormer方案**：三个核心改进详解（配架构图）
4. **实验验证**：关键数据表格 + 性能分析
5. **梯度分析**：深入理解为什么有效
6. **工程实践**：如何使用、成本分析
7. **技术演进**：NormFormer在归一化技术线中的位置
8. **总结与展望**

### 可视化建议
- Pre-LN vs Post-LN vs NormFormer 的架构对比图
- 三种架构的梯度幅度对比图（概念图）
- 性能数据的柱状图或折线图
