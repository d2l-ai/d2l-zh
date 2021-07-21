# 带全局向量的单词嵌入 (GLOVE)
:label:`sec_glove`

上下文窗口中的单词同时出现可能会带有丰富的语义信息。例如，在大型语料库中，“固体” 一词比 “蒸汽” 更有可能与 “冰” 共存，但是 “气” 一词可能与 “蒸汽” 共同出现的频率比 “冰” 更频繁。此外，可以预先计算此类同时出现的全球语料库统计数据：这可以提高培训效率。为了利用整个语料库中的统计信息进行单词嵌入，让我们首先重温 :numref:`subsec_skip-gram` 中的跳过图模型，但是使用全局语料库统计数（例如共生计数）来解释它。 

## 跳过 Gram 与全球语料库统计
:label:`subsec_skipgram-global`

以 $q_{ij}$ 表示 $w_j$ 字的条件概率 $P(w_j\mid w_i)$ 在跳过图模型中给出的单词 $w_i$，我们有 

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

其中，任何索引 $i$ 向量 $\mathbf{v}_i$ 和 $\mathbf{u}_i$ 分别表示单词 $w_i$ 作为中心词和上下文词，$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ 是词汇的索引集。 

考虑一下可能在语料库中多次出现的单词 $w_i$。在整个语料库中，所有上下文单词无论 $w_i$ 被视为中心词，都构成了 * 多集 * $\mathcal{C}_i$ 的单词索引，* 允许同一元素的多个实例 *。对于任何元素，它的实例数都称为其 * 多重性 *。举个例子来说明，假设单词 $w_i$ 在语料库中出现两次，在两个上下文窗口中以 $w_i$ 作为中心词的上下文词的索引是 $k, j, m, k$ 和 $k, l, k, j$。因此，多集 $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$，其中元素 $j, k, l, m$ 的多重性分别为 2、4、1、1。 

现在让我们将多集 $\mathcal{C}_i$ 中元素 $j$ 的多重性表示为 $x_{ij}$。这是整个语料库中同一上下文窗口中单词 $w_j$（作为上下文单词）和单词 $w_i$（作为中心词）的全局共生计数。使用这样的全局语料库统计数据，跳过图模型的损失函数等同于 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

我们进一步用 $x_i$ 表示上下文窗口中所有上下文单词的数量，其中 $w_i$ 作为中心词出现，相当于 $|\mathcal{C}_i|$。让 $p_{ij}$ 成为生成上下文单词 $w_j$ 的条件概率 $x_{ij}/x_i$，给定中心字 $w_i$，:eqref:`eq_skipgram-x_ij` 可以重写为 

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

在 :eqref:`eq_skipgram-p_ij` 中，$-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ 计算了全局语料库统计数据的条件分布 $p_{ij}$ 的交叉熵和模型预测的条件分布 $q_{ij}$。如上所述，这一损失也加权了 $x_i$。最大限度地减少 :eqref:`eq_skipgram-p_ij` 中的损失函数将允许预测的条件分布接近全球语料库统计数据中的条件分布。 

尽管通常用于测量概率分布之间的距离，但交叉熵损失函数可能不是一个很好的选择。一方面，正如我们在 :numref:`sec_approx_train` 中提到的那样，正确标准化 $q_{ij}$ 的成本导致了整个词汇的总和，这可能是计算昂贵的。另一方面，来自大量语料库的大量罕见事件通常以交叉熵损失为模型，而不能分配过多的权重。 

## Glove 模型

有鉴于此，*Glove* 模型基于平方损失 :cite:`Pennington.Socher.Manning.2014` 对跳跃图模型进行了三项更改： 

1. 使用变量 $p'_{ij}=x_{ij}$ 和 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ 
这不是概率分布，而是两者的对数，因此平方损失期限为 $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$。
2. 为每个单词 $w_i$ 添加两个标量模型参数：中心词偏差 $b_i$ 和上下文词偏差 $c_i$。
3. 用权重函数 $h(x_{ij})$ 替换每个损失期的权重，其中 $h(x)$ 在 $[0, 1]$ 的间隔内增加了 $h(x)$。

将所有事情放在一起，训练 GLOVE 是为了尽量减少以下损失功能： 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

对于权重函数，建议的选择是：$h(x) = (x/c) ^\alpha$（例如 $\alpha = 0.75$）如果是 $x < c$（例如 $c = 100$），否则为 $h(x) = 1$。在这种情况下，由于 $h(0)=0$，为了计算效率，可以省略任何 $x_{ij}=0$ 的平方损失期限。例如，当使用迷你批随机梯度下降进行训练时，在每次迭代中，我们都随机采样一个 * 非零 * $x_{ij}$ 的迷你匹配，以计算渐变并更新模型参数。请注意，这些非零 $x_{ij}$ 是预先计算的全局语料库统计数据；因此，该模型被称为 *Global Vector* 的 Glove。 

应该强调的是，如果单词 $w_i$ 出现在单词 $w_j$ 的上下文窗口中，那么 * 反之 *。因此，$x_{ij}=x_{ji}$。与适合不对称条件概率 $p_{ij}$ 的 word2vec 不同，Glove 适合对称 $\log \, x_{ij}$。因此，在 GLOVE 模型中，任何单词的中心单词矢量和上下文单词矢量在数学上是等同的。但是在实践中，由于初始值不同，训练后同一个词可能仍然会在这两个向量中得到不同的值：GloVE 将它们总结为输出矢量。 

## 从共发概率比例解释 Glove

我们也可以从另一个角度解释 GLOVE 模型。在 :numref:`subsec_skipgram-global` 中使用相同的符号，让 $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ 成为生成上下文单词 $w_j$ 的条件概率，给定 $w_i$ 作为语料库中的中心词。:numref:`tab_glove` 列出了 “冰” 和 “蒸汽” 两个词的几个共同出现概率及其基于大型语料库统计数据的比率。 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

我们可以从 :numref:`tab_glove` 观察到以下内容： 

* 对于与 “冰” 有关但与 “蒸汽” 无关的单词 $w_k$，例如 $w_k=\text{solid}$，我们预计共发概率比例更大，例如 8.9。
* 对于与 “蒸汽” 有关但与 “冰” 无关的单词 $w_k$，例如 $w_k=\text{gas}$，我们预计共发概率比例较小，例如 0.085。
* 对于与 “冰” 和 “蒸汽” 都有关的单词 $w_k$，例如 $w_k=\text{water}$，我们预计共同发生概率的比率接近 1，例如 1.36。
* 对于与 “冰” 和 “蒸汽” 无关的单词 $w_k$，例如 $w_k=\text{fashion}$，我们预计共同发生概率的比率接近 1，例如 0.96。

可以看出，共同发生概率的比率可以直观地表达单词之间的关系。因此，我们可以设计一个由三个词向量组成的函数来适应这个比例。对于共发概率的比例 ${p_{ij}}/{p_{ik}}$，其中 $w_i$ 是中心词，$w_j$ 和 $w_k$ 是上下文词，我们希望使用一些函数 $f$ 来调整这个比率： 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

在 $f$ 的许多可能设计中，我们只选择以下合理的选择。由于共生概率比率是标量，因此我们要求 $f$ 是标量函数，例如 $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$。在 :eqref:`eq_glove-f` 中切换字指数 $j$ 和 $k$，它必须持有 $f(x)f(-x)=1$，所以一种可能性是 $f(x)=\exp(x)$，即  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

现在让我们选择 $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$，其中 $\alpha$ 是常数。自 $p_{ij}=x_{ij}/x_i$ 起，在双方对数后，我们得到了 $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$。我们可能会使用其他偏见术语来适应 $- \log\, \alpha + \log\, x_i$，例如中心词偏差 $b_i$ 和上下文词偏差 $c_j$： 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

用重量测量 :eqref:`eq_glove-square` 的平方误差，获得了 :eqref:`eq_glove-loss` 中的 Glove 损失函数。 

## 摘要

* 跳过图模型可以使用全局语料库统计数据（例如单词共生计数）来解释。
* 交叉熵损失可能不是衡量两种概率分布差异的好选择，特别是对于大型语料库而言。GLOVE 使用平方损耗来适应预先计算的全局语料库统计数据。
* 中心单词矢量和上下文单词矢量在数学上对于 GloVE 中的任何单词来说都是等同的。
* GLOVE 可以从单词-词共生概率的比率来解释。

## 练习

1. 如果单词 $w_i$ 和 $w_j$ 同时出现在同一个上下文窗口中，我们怎样才能使用它们在文本序列中的距离来重新设计计算条件概率 $p_{ij}$ 的方法？Hint: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`。
1. 对于任何单词来说，它的中心词偏见和上下文词偏见在 Glove 中数学上是否等同？为什么？

[Discussions](https://discuss.d2l.ai/t/385)
