# 束搜索
:label:`sec_beam-search`

在:numref:`sec_seq2seq`中，我们逐个标记地预测输出序列令牌，直到预测出序列结束标记“&lt;eos&gt;”。在本节中，我们将首先对这种*贪心搜索*（greedy search）策略进行介绍，并探讨其存在的问题，然后将这种策略与其他替代策略进行比较：*穷举搜索*（exhaustive search）和*束搜索*（beam search）。

在正式介绍贪心搜索之前，让我们使用 :numref:`sec_seq2seq` 中相同的数学符号定义搜索问题。在任何时间步$t'$，解码器输出$y_{t'}$的概率取决于$t'$之前的输出子序列$y_1, \ldots, y_{t'-1}$和编码输入序列信息的上下文变量$\mathbf{c}$。为了量化计算成本，用$\mathcal{Y}$（它包含“&lt;eos&gt;”）表示输出词汇表。所以这个词汇集合的基数$\left|\mathcal{Y}\right|$就是词汇大小。我们还将输出序列的最大标记数指定为$T'$。因此，我们的目标是从所有$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$个可能的输出序列中寻找理想的输出。当然，对于所有这些输出序列，包括“&lt;eos&gt;”和之后的部分将在实际输出中丢弃。

## 贪心搜索

首先，让我们看看一个简单的策略：*贪心搜索*。该策略已用于:numref:`sec_seq2seq`的序列预测。在贪心搜索中，在输出序列的任何时间步$t'$，我们从$\mathcal{Y}$中搜索具有最高条件概率的标记，即：

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

一旦输出“&lt;eos&gt;”或输出序列达到其最大长度$T'$，输出序列即完成。

那么贪心搜索会出什么问题呢？实际上，*最优序列*（optimal sequence）应该是最大化$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$值的输出序列，这是基于输入序列生成输出序列的条件概率。不幸的是，不能保证通过贪心搜索得到最优序列。

![在每个时间步，贪心搜索选择具有最高条件概率的标记。](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

让我们用一个例子来说明这一点。假设输出中有四个标记“A”、“B”、“C”和“&lt;eos&gt;”。 在:numref:`fig_s2s-prob1` 中，每个时间步下的四个数字分别表示在该时间步生成“A”、“B”、“C”和“&lt;eos&gt;”的条件概率。在每个时间步，贪心搜索选择具有最高条件概率的令牌。因此，将在 :numref:`fig_s2s-prob1` 中预测输出序列“A”、“B”、“C”和“&lt;eos&gt;”。这个输出序列的条件概率是$0.5\times0.4\times0.4\times0.6 = 0.048$。

![每个时间步下的四个数字表示在该时间步生成“A”、“B”、“C”和“&lt;eos&gt;”的条件概率。在时间步2，选择具有第二高条件概率的令牌“C”。](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

接下来，让我们看看 :numref:`fig_s2s-prob2` 中的另一个例子。与 :numref:`fig_s2s-prob1` 不同，在时间步2中，我们选择 :numref:`fig_s2s-prob2` 中的令牌“C”，它具有第二高的条件概率。由于时间步3所基于的时间步1和2处的输出子序列已从 :numref:`fig_s2s-prob1` 中的“A”和“B”改变为 :numref:`fig_s2s-prob2` 中的“A”和“C”，因此时间步3处的每个标记的条件概率也在 :numref:`fig_s2s-prob2` 中改变。假设我们在时间步3选择令牌“B”。现在，时间步4以前三个时间步“A”、“C”和“B”的输出子序列为条件，这与 :numref:`fig_s2s-prob1` 中的“A”、“B”和“C”不同。因此，在 :numref:`fig_s2s-prob2` 中的时间步4生成每个标记的条件概率也不同于 :numref:`fig_s2s-prob1` 中的条件概率。结果，:numref:`fig_s2s-prob2`中的输出序列“A”、“C”、“B”和“&lt;eos&gt;”的条件概率为$0.5\times0.3 \times0.6\times0.6=0.054$，这大于:numref:`fig_s2s-prob1`中的贪心搜索的条件概率。在本例中，通过贪心搜索获得的输出序列“A”、“B”、“C”和“&lt;eos&gt;”不是最佳序列。

## 穷举搜索

如果目标是获得最优序列，我们可以考虑使用*穷举搜索*（exhaustive search）：穷举地枚举所有可能的输出序列及其条件概率，然后输出条件概率最高的一个。

虽然我们可以使用穷举搜索来获得最优序列，但其计算量$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$可能过高。例如，当$|\mathcal{Y}|=10000$和$T'=10$时，我们需要评估$10000^{10} = 10^{40}$序列。这几乎是不可能的。另一方面，贪心搜索的计算量是$\mathcal{O}(\left|\mathcal{Y}\right|T')$：它通常明显小于穷举搜索。例如，当$|\mathcal{Y}|=10000$和$T'=10$时，我们只需要评估$10000\times10=10^5$个序列。

## 束搜索

关于序列搜索策略的决定取决于一个范围，在任何一个极端都有问题。如果只有准确性才重要呢？显然，穷举搜索。如果计算成本很重要呢？显然，贪心搜索。实际应用介于这两个极端之间。

*束搜索*（beam search）是贪心搜索的改进版本。它有一个超参数，名为*束宽*（beam size）$k$。
在时间步1，我们选择具有最高条件概率的$k$个标记。它们中的每一个将分别是$k$个候选输出序列的第一个标记。在随后的每个时间步，基于上一时间步的$k$个候选输出序列，我们继续从$k\left|\mathcal{Y}\right|$个可能的选择中选择具有最高条件概率的$k$个候选输出序列。

![束搜索过程（束宽：2，输出序列的最大长度：3）。候选输出序列是$A$、$C$、$AB$、$CE$、$ABD$和$CED$。](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search`演示了束搜索的过程。假设输出词表只包含五个元素：$\mathcal{Y} = \{A, B, C, D, E\}$，其中一个是“&lt;eos&gt;”。让束宽为2，输出序列的最大长度为3。在时间步1，假设具有最高条件概率$P(y_1 \mid \mathbf{c})$的标记是$A$和$C$。在时间步2，我们计算所有$y_2 \in \mathcal{Y}$：

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

从这十个值中选择最大的两个，比如$P(A, B \mid \mathbf{c})$和$P(C, E \mid \mathbf{c})$。然后在时间步3，对于所有$y_3 \in \mathcal{Y}$，我们计算：

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

然后从这十个值中选择最大的两个，即$P(A, B, D \mid \mathbf{c})$和$P(C, E, D \mid  \mathbf{c}).$。结果，我们得到六个候选输出序列：（1）$A$；（2）$C$；（3）$B$；（4）$C$、$E$；（5）$A$、$B$、$D$以及（6）$C$、$D$。

最后，我们基于这六个序列（例如，包括“&lt;eos&gt;”和之后的丢弃部分）获得最终候选输出序列集合。然后我们选择以下得分最高的序列作为输出序列：

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

其中$L$是最终候选序列的长度，$\alpha$通常设置为0.75。因为一个较长的序列在:eqref:`eq_beam-search-score`的总和中有更多的对数项，分母中的$L^\alpha$惩罚长序列。

束搜索的计算量为$\mathcal{O}(k\left|\mathcal{Y}\right|T')$。这个结果介于贪心搜索和穷举搜索之间。实际上，贪心搜索可以看作是一种特殊类型的束搜索，束宽为1。通过灵活选择束宽，束搜索可以在精度和计算成本之间进行权衡。

## 小结

* 序列搜索策略包括贪心搜索、穷举搜索和束搜索。
* 束搜索通过灵活选择束宽，在精度和计算成本之间找到平衡。

## 练习

1. 我们能把穷举搜索看作一种特殊的束搜索吗？
1. 在 :numref:`sec_seq2seq` 机器翻译问题中应用束搜索。束宽如何影响结果和预测速度？
1. 在 :numref:`sec_rnn_scratch` 中，我们使用语言模型来生成用户提供前缀的文本。它使用了哪种搜索策略？你能改进一下吗？

[Discussions](https://discuss.d2l.ai/t/338)
