# 大致训练
:label:`sec_approx_train`

回想一下我们在 :numref:`sec_word2vec` 中的讨论。跳过图模型的主要思想是使用 softmax 运算来计算生成上下文单词 $w_o$ 的条件概率，基于 :eqref:`eq_skip-gram-softmax` 中给定的中心字 $w_c$，其相应的对数损失由 :eqref:`eq_skip-gram-log` 的相反值给出。 

由于 softmax 操作的性质，由于上下文词可能是字典 $\mathcal{V}$ 中的任何人，因此 :eqref:`eq_skip-gram-log` 的对面包含的项目总和与词汇的整个大小相同。因此，:eqref:`eq_skip-gram-grad` 中的跳过图模型的梯度计算和 :eqref:`eq_cbow-gradient` 中的连续字包模型的梯度计算都包含了总和。不幸的是，这种渐变的计算成本超过一个大字典（通常有成千上万或数百万个单词）的总和是巨大的！ 

为了降低上述计算复杂性，本节将介绍两种近似的训练方法：
*负采样 * 和 * 层次结构 softmax*。
由于跳跃图模型和连续包文字模型之间的相似性，我们只以跳跳图模型为例来描述这两种近似的训练方法。 

## 负面采样
:label:`subsec_negative-sampling`

负取样会修改原来的目标函数。鉴于中心字 $w_c$ 的上下文窗口，任何（上下文）单词 $w_o$ 来自此上下文窗口的事实都被视为一个事件，概率为 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

其中 $\sigma$ 使用了 sigmoid 激活函数的定义： 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

让我们首先最大限度地提高文本序列中所有此类事件的共同概率，以训练单词嵌入。具体来说，给定长度为 $T$ 的文本序列，用 $w^{(t)}$ 表示时间步长 $t$ 的单词，并且让上下文窗口大小为 $m$，考虑最大化联合概率 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

但是，:eqref:`eq-negative-sample-pos` 只考虑那些涉及积极例子的事件。因此，只有当所有单词矢量都等于无穷大时，:eqref:`eq-negative-sample-pos` 中的联合概率才最大化为 1。当然，这样的结果毫无意义。为了使客观功能更有意义,
*负面采样 *
添加了从预定义分布中采样的负面示例。 

用 $S$ 表示上下文字 $w_o$ 来自中心字 $w_c$ 的上下文窗口的事件。对于这个涉及 $w_o$ 的事件，来自预定义的分布 $P(w)$ 样本 $K$ * 噪音词 * 不在此上下文窗口中。用 $N_k$ 表示噪音词 $w_k$（$k=1, \ldots, K$）不是来自 $w_c$ 的上下文窗口的事件。假设这些涉及正面例子和负面例子 $S, N_1, \ldots, N_K$ 的事件是相互独立的。负抽样将 :eqref:`eq-negative-sample-pos` 中的联合概率（仅涉及正面示例）重写为 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

其中条件概率是通过事件 $S, N_1, \ldots, N_K$ 近似的： 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

分别用 $i_t$ 和 $h_k$ 表示文本序列时间步长 $t$ 的单词 $w^{(t)}$ 和噪声词 $w_k$ 的索引。:eqref:`eq-negative-sample-conditional-prob` 中的条件概率的对数损失是 

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

我们可以看到，现在每个训练步骤的渐变计算成本与字典大小无关，但线性取决于 $K$。将超参数 $K$ 设置为较小的值时，每个具有负采样的训练步骤中渐变的计算成本会降低。 

## 分层 Softmax

作为另一种近似训练方法,
*分层 softmax*
使用二叉树，一种在 :numref:`fig_hi_softmax` 中所示的数据结构，其中树的每个叶节点代表字典 $\mathcal{V}$ 中的一个单词。 

![Hierarchical softmax for approximate training, where each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

用 $L(w)$ 表示从根节点到叶节点的路径上的节点数（包括两端），代表二叉树中单词 $w$。让 $n(w,j)$ 成为这条路径上的 $j^\mathrm{th}$ 节点，其上下文词矢量为 $\mathbf{u}_{n(w, j)}$。例如，:numref:`fig_hi_softmax` 中的 $L(w_3) = 4$。层次 Softmax 近似 :eqref:`eq_skip-gram-softmax` 中的条件概率，如 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

其中函数 $\sigma$ 在 :eqref:`eq_sigma-f` 中定义，$\text{leftChild}(n)$ 是节点 $n$ 的左子节点：如果 $x$ 是真的，$ [\![x]\!]= 1$; otherwise $ [\![x]\!]= -1$。 

为了说明，让我们计算在 :numref:`fig_hi_softmax` 中给定单词 $w_c$ 生成单词 $w_3$ 的条件概率。这需要点积在 $w_c$ 的矢量 $\mathbf{v}_c$ 和路径上的非叶节点矢量（:numref:`fig_hi_softmax` 中的粗体路径）之间从根点到 $w_3$，后者是向左、右，然后向左遍历的 $w_3$： 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

自 $\sigma(x)+\sigma(-x) = 1$ 以来，它认为，基于任何单词 $w_c$ 在字典 $\mathcal{V}$ 中生成所有单词的条件概率总结为 1： 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

幸运的是，由于二叉树结构，$L(w_o)-1$ 约为 $\mathcal{O}(\text{log}_2|\mathcal{V}|)$，当字典大小 $\mathcal{V}$ 很大时，与没有近似训练的情况相比，使用分层 softmax 的每个训练步骤的计算成本大大降低。 

## 摘要

* 负抽样通过考虑涉及正面和负面例子的相互独立事件来构建损失函数。训练的计算成本线性取决于每个步骤的噪声词数。
* 分层 softmax 使用从根节点到二叉树中叶节点的路径构造损失函数。训练的计算成本取决于每个步骤中字典大小的对数。

## 练习

1. 我们如何在负面采样中对噪声词进行抽样？
1. 验证 :eqref:`eq_hi-softmax-sum-one` 是否持有。
1. 如何分别使用负取样和层次 Softmax 来训练连续文字包模型？

[Discussions](https://discuss.d2l.ai/t/382)
