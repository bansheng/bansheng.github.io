---
title: DARTS 可微分架构搜索
top: false
cover: false
oc: true
mathjax: true
date: 2020-11-01 13:43:04
password:
description: 可微分架构搜索DARTS
tags: NAS
categories: Paper
---

## 1. DARTS

 the computation procedure for an architecture (or a cell in it) is represented as a directed acyclic graph. 表示为有向图。

### 1.1 search space

寻找一个计算cell，作为最后架构的建造模块。学习出来的cell可以叠加起来组成cnn，或者递归连接起来组成rnn。

cell是由N个有序序列node组成的有向无环图。每一条edge都是一个计算。我们假设这个cell有两个input和一个output，对于cnn，它就是前面两个层的输出，对于rnn，它是上个step的state以及这个step的Input。cell的输出是通过对所有中间节点应用reduction得到的。

所有中间节点的计算依赖前置节点。
$$
x^{(j)} = \sum_{i<j}o^{(i,j)}(x^{(i)})
$$
注意zero operation也是可以被允许的edge类型。

### 1.2 continuous relaxation and optimization

找到每一个操作对应的权重矩阵$\alpha^{(i,j)}$，这样所有的权重矩阵集合为$\alpha$，我们将NAS的任务减小为学习一个连续变量的集合$\alpha$。

DARTS使用的是**GD**来优化validation loss。相似的有RL([Learning transferable architectures for scalable image recognition]())，EA([Hierarchical representations for efficient architecture search]())

NAS的目标是找到$\alpha^*$使得validation loss$L_{val}(w^*, \alpha^*)$最小，$w^*$是使得training loss$L_{train}(w, \alpha^*)$最小的w。
$$
min_{\alpha} L_{val}(w^*(\alpha), \alpha) \\
s.t. w^*(\alpha) = argmin_w L_{train}(w, \alpha)
$$
![](https://s2.loli.net/2022/02/08/QWc1fC3shg4q2tb.png)

### 1.3 approximate architecture gradient

$$
\begin{aligned} & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\ \approx & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w-\xi \nabla_{w} \mathcal{L}_{t r a i n}(w, \alpha), \alpha\right) \end{aligned}
$$

运用chain rule。将上式进一步处理。
$$
\triangledown_\alpha L_{val}(w', \alpha - \xi \triangledown^2_{\alpha, w} L_{train}(w, \alpha) \triangledown_{w'}L_{val}(w', \alpha))
$$
其中的$w' = w - \xi\triangledown_w L_{train}(w, \alpha)$指的就是one-step forward model。

使用the finite difference approximation(有限差分近似)可以减少复杂度。
$$
\epsilon 是极小量 \\
w^{\pm} = w \pm \epsilon \triangledown_{w'}L_{val}(w', \alpha) \\
\xi \triangledown^2_{\alpha, w} L_{train}(w, \alpha) \triangledown_{w'}L_{val}(w', \alpha)) \approx \frac{\triangledown_\alpha L_{train}(w^{+}, \alpha) - \triangledown_\alpha L_{train}(w^{-}, \alpha)}{2\xi}
$$
将$\xi = 0$作为一阶近似，将$\xi > 0$作为两阶近似。

### 1.4 deriving discrete architecture

在所有非0的候选operations保留top-k strongest operations，为了使得出的网络可以和现有网络比较，我们选择k=2 for cnn, k=1 for rnn。

为什么不使用zero operation呢？

1. 为了与现有模型进行公平的比较，我们需要每个节点恰好有k条非零的引入边
2. 因为增加零操作的logits只会影响结果节点表示的规模，由于BN处理的存在而不会而影响最终的分类结果

## 2. Experiments and results

### 2.1 architecture search

#### 2.1.1 search for convolutional cells on cifar-10

包含8种operation。 3 × 3 and 5 × 5 separable convolutions, 3 × 3 and 5 × 5 dilated separable convolutions, 3 × 3 max pooling, 3 × 3 average pooling, identity, and zero。

We use the ReLU-Conv-BN order for convolutional operations, and each separable convolution is always applied twice

在整个网络的1/3和2/3处，设立reduce cell。缩小空间分辨率。

#### 2.1.2 searching for recurrent cells for penn treebank

operation的种类：linear transformations followed by one of tanh, relu, sigmoid activations, as well as the identity mapping and the *zero* operation.

总共12个node，最初的intermediate node是由两个input node通过线性变换，求和，然后传过一个tanh激活函数得到的。

### 2.2 architecture evaluation

**寻找多次，避免初始化的影响** 。从cifar-10上迁移到imagenet上，从PTB上迁移到wikitext-2上。
