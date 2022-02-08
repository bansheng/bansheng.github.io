---
title: string_pattern
top: false
cover: false
oc: true
mathjax: true
date: 2020-07-15 14:08:05
password:
summary: 字符子串匹配算法
tags: string
categories: Algorithm
---

## 1.1 字符串匹配算法

| algorithm | $T_{best}$ | $T_{avg}$ | $T_{worst}$ |
| :- | :-: | :-: | :-: |
| 1. 朴素匹配算法 | O(nm) | O(nm) | O(nm) |
| 2. Robin-Karp 算法 | O(n) | O(nm) | O(nm) |
| 3. KMP算法 | O(n+m) | O(n+m) | O(n+m) |

### 1.1.1 朴素匹配算法

暴力求解

### 1.1.2 Robin-Karp算法

先是计算两个字符串的哈希值，然后通过比较这两个哈希值的大小来判断是否出现匹配。
选择一个合适的哈希函数很重要。假设文本串为t[0, n)，模式串为p[0, m)，其中 $0<m<n$，$Hash(t[i,j])$代表字符串t[i, j]的哈希值。

当 $Hash(t[0, m-1])!=Hash(p[0,m-1])$ 时，我们很自然的会把 $Hash(t[1, m])$ 拿过来继续比较。在这个过程中，若我们重新计算字符串t[1, m]的哈希值，还需要 O(n) 的时间复杂度，不划算。观察到字符串t[0, m-1]与t[1, m]中有 m-1 个字符是重合的，因此我们可以选用**滚动哈希函数**，那么重新计算的时间复杂度就降为 O(1)。

Rabin-Karp 算法选用的滚动哈希函数主要是利用$Rabin fingerprint$的思想，举个例子，计算字符串t[0, m - 1]的哈希值的公式如下，

$$Hash(t[0,m−1])=t[0]∗b_{m−1}+t[1]∗b_{m−2}+...+t[m−1]∗b_0$$

其中的 b_k 可以是一个常数，在 Rabin-Karp 算法中，我们一般取值为256，因为一个字符的最大值不超过255。上面的公式还有一个问题，哈希值如果过大可能会溢出，因此我们还需要对其取模，这个值应该尽可能大，且是质数，这样可以减小哈希碰撞的概率，在这里我们就取 101。

则计算字符串t[1, m]的哈希值公式如下，

$$Hash(t[1,m])=(Hash(t[0,m−1])−t[0]∗b_{m−1})∗b+t[m]∗b_0$$

### 1.1.3 KMP算法

KMP算法的精髓在于，对于一个不匹配的子串，我们将整个模式串向后面移动更多的位数而不是1，来加速子串的识别，这个移动步数跟只需要根据模式子串计算一次就可以得到。

[阮一峰KMP算法](http://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html)

```python
#KMP
def kmp_match(s, p):
    m = len(s); n = len(p)
    cur = 0#起始指针cur
    table = partial_table(p)
    while cur<=m-n:
        for i in range(n):
            if s[i+cur]!=p[i]:
                cur += max(i - table[i-1], 1)#有了部分匹配表,我们不只是单纯的1位1位往右移,可以一次移动多位
                break
        else:
            return True
    return False

#部分匹配表
def partial_table(p):
    '''partial_table("ABCDABD") -> [0, 0, 0, 0, 1, 2, 0]'''
    prefix = set()
    postfix = set()
    ret = [0]
    for i in range(1,len(p)):
        prefix.add(p[:i])
        postfix = {p[j:i+1] for j in range(1,i+1)}
        ret.append(len((prefix&postfix or {''}).pop()))
    return ret
```
