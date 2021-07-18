# 字嵌入 (word2vec)
:label:`sec_word2vec`

自然语言是用来表达意义的复杂系统。在这个系统中，单词是含义的基本单位。顾名思义，
*单词矢量 * 是用来表示单词的矢量，
也可以被视为特征向量或单词的表示形式。将单词映射到实际矢量的技术称为 * 词嵌入 *。近年来，单词嵌入已逐渐成为自然语言处理的基本知识。 

## 一个热门矢量是不好的选择

我们在 :numref:`sec_rnn_scratch` 中使用了一个热点矢量来表示单词（字符是单词）。假设字典中不同单词的数量（字典大小）为 $N$，每个单词对应于从 $0$ 到 N-1 $ 不同的整数（索引）。为了获得索引为 $i$ 的任何单词的单热矢量表示形式，我们创建了一个包含全部 0 的长度 $N$ 矢量，并将位置 $i$ 的元素设置为 1。通过这种方式，每个单词都表示为长度为 $N$ 的矢量，神经网络可以直接使用。 

尽管一个热点词向量很容易构建，但它们通常不是一个好选择。一个主要原因是，一个热点词矢量无法准确表达不同单词之间的相似性，例如我们经常使用的 * 余弦相似性 *。对于向量 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，它们的余弦相似性是它们之间角度的余弦值： 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

由于任何两个不同单词的单热矢量之间的余弦相似性为 0，因此单热向量无法编码单词之间的相似之处。 

## 自我监督 word2vec

提议使用 [word2vec](https://code.google.com/archive/p/word2vec/) 工具来解决上述问题。它将每个单词映射到固定长度的矢量，这些向量可以更好地表达不同单词之间的相似性和类比关系。word2vec 工具包含两种模型，分别是 * 跳过克 * :cite:`Mikolov.Sutskever.Chen.ea.2013` 和 * 连续字包 * (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`。对于具有语义意义的表述，他们的训练依赖于条件概率，这可以被视为在语法中使用他们周围的一些单词来预测某些单词。由于监督来自没有标签的数据，因此跳过图和连续的文字包都是自我监督的模型。 

在下面，我们将介绍这两种模型及其训练方法。 

## 跳过格兰氏模型
:label:`subsec_skip-gram`

*skip-gram* 模型假定可以使用单词来生成文本序列中的周围单词。以文本序列 “该”、“男人”、“爱”、“他的”、“儿子” 为例。让我们选择 “爱” 作为 * 中心单词 * 并将上下文窗口大小设置为 2。如 :numref:`fig_skip_gram` 所示，给定中心单词 “爱”，跳过图模型考虑了生成 * 上下文词 * 的条件概率：“to”、“man”、“他的” 和 “儿子”，它们距离中心词不超过 2 个词： 

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

假设上下文词在给定中心单词（即条件独立性）的情况下是独立生成的。在这种情况下，上述条件概率可以重写为 

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg)
:label:`fig_skip_gram`

在跳过格模型中，每个单词都有两个用于计算条件概率的 $d$ 维矢量表示法。更具体地说，对于字典中索引为 $i$ 的任何单词，用 $\mathbf{v}_i\in\mathbb{R}^d$ 和 $\mathbf{u}_i\in\mathbb{R}^d$ 表示分别用作 * 中心 * 单词和 * 上下文 * 单词时，它的两个向量。根据中心单词 $w_c$（字典中的索引为 $c$），生成任何上下文单词 $w_o$（字典中有索引 $o$）的条件概率可以通过对矢量点积的 softmax 运算进行建模： 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

词汇指数设置了 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。给定长度为 $T$ 的文本序列，其中时间步长 $t$ 中的单词表示为 $w^{(t)}$。假设上下文单词是在任何中心词的情况下独立生成的。对于大小 $m$ 的上下文窗口，跳过图模型的可能性函数是给定任何中心单词生成所有上下文词的概率： 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

可以省略任何小于 $1$ 或大于 $T$ 的时间步长。 

### 训练

跳过格式模型参数是词汇中每个单词的中心单词矢量和上下文单词矢量。在训练中，我们通过最大化似然函数（即最大似然估计）来学习模型参数。这相当于最大限度地减少以下损失函数： 

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

当使用随机梯度下降来尽量减少损失时，在每次迭代中，我们都可以随机采样较短的子序列来计算此子序列的（随机）梯度，以更新模型参数。要计算这个（随机）梯度，我们需要获取与中心词矢量和上下文单词矢量相关的对数条件概率的渐变。一般来说，根据 :eqref:`eq_skip-gram-softmax`，涉及任何一对中心词 $w_c$ 和上下文词 $w_o$ 的对数条件概率是 

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

通过差异化，我们可以获得它相对于中心字矢量 $\mathbf{v}_c$ 的梯度 

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

请注意，:eqref:`eq_skip-gram-grad` 中的计算要求以 $w_c$ 为中心词的字典中所有单词的条件概率。另一个词向量的渐变也可以用同样的方式获取。 

训练后，对于字典中索引为 $i$ 的任何单词，我们都可以获得单词矢量 $\mathbf{v}_i$（作为中心词）和 $\mathbf{u}_i$（作为上下文词）。在自然语言处理应用程序中，跳过图模型的中心词矢量通常用作单词表示形式。 

## 连续的话包（CBO）模型

* 连续字包 * (CBOW) 模型类似于跳过格式模型。与跳过图模型的主要区别在于，连续包单词模型假设中心单词是根据文本序列中的周围上下文单词生成的。例如，在同一文本序列 “the”、“man”、“爱”、“他的” 和 “儿子” 中，“爱” 作为中心词，上下文窗口大小为 2，连续文字包模型根据上下文词 “the”、“man”、“他的” 和 “儿子” 考虑生成中心单词 “爱” 的条件概率“（如 :numref:`fig_cbow` 所示），这是 

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg)
:eqlabel:`fig_cbow`

由于连续单词包模型中有多个上下文单词，因此在计算条件概率时计算这些上下文词矢量。具体来说，对于字典中索引为 $i$ 的任何单词，分别用 $\mathbf{v}_i\in\mathbb{R}^d$ 和 $\mathbf{u}_i\in\mathbb{R}^d$ 表示它的两个向量作为 * 上下文 * 单词和 * 中心 * 词（含义在跳过图模型中切换）。鉴于其周围的上下文单词 $w_{o_1}, \ldots, w_{o_{2m}}$（字典中有索引 $o_1, \ldots, o_{2m}$）（字典中有索引 $c$）（字典中有索引 $o_1, \ldots, o_{2m}$）的条件概率可以通过以下方式建模： 

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

为了简洁起见，请让 $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ 和 $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$。然后 :eqref:`fig_cbow-full` 可以简化为 

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

给定长度为 $T$ 的文本序列，其中时间步长 $t$ 中的单词表示为 $w^{(t)}$。对于上下文窗口大小 $m$，连续文字包模型的可能性函数是根据上下文单词生成所有中心单词的概率： 

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### 训练

训练连续的文字模型几乎与训练跳跃图模型相同。对连续包文字模型的最大似然估计等同于最大限度地减少以下损失函数： 

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

请注意 

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

通过区分，我们可以获得它相对于任何上下文单词矢量 $\mathbf{v}_{o_i}$ ($i = 1, \ldots, 2m$) 的梯度，如 

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

另一个词向量的渐变也可以用同样的方式获取。与跳过格式模型不同，连续包单词模型通常使用上下文词矢量作为单词表示形式。 

## 摘要

* 词矢量是用于表示单词的矢量，也可以被视为特征向量或单词的表示。将单词映射到真实矢量的技术称为单词嵌入。
* word2vec 工具包含跳过图和连续的单词模型。
* 跳过图模型假设一个单词可用于在文本序列中生成其周围的单词；而连续的单词包模型假定基于其周围的上下文单词生成中心单词。

## 练习

1. 计算每个梯度的计算复杂性是多少？如果字典大小很大，可能会有什么问题？
1. 英语中的一些固定短语由多个单词组成，例如 “纽约”。如何训练他们的单词矢量？Hint: See section 4 in the word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`。
1. 让我们以跳跃图模型为例来反思 word2vec 设计。跳过图模型中两个单词矢量的点积与余弦相似性之间有什么关系？对于语义相似的一对单词，为什么它们的单词向量（由跳过图模型训练）的余弦相似性会很高？

[Discussions](https://discuss.d2l.ai/t/381)
