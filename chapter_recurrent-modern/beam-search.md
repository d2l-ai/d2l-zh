# 束搜索
:label:`sec_beam-search`

在 :numref:`sec_seq2seq` 中，我们通过令牌预测了输出序列令牌，直到特殊的序列结束 “<eos>” 令牌被预测为止。在本节中，我们将从正式确定这个 * 贪婪的搜索 * 策略并探索问题开始，然后将此策略与其他备选方案进行比较：
*详尽搜索 * 和 * 束搜索 *。

在正式介绍贪婪搜索之前，让我们使用 :numref:`sec_seq2seq` 的相同数学符号将搜索问题正式化。在任何时间步骤 $t'$，解码器输出 $y_{t'}$ 的概率取决于 $t'$ 之前的输出子序列 $y_1, \ldots, y_{t'-1}$ 和编码输入序列信息的上下文变量 $\mathbf{c}$。要量化计算成本，用 $\mathcal{Y}$（包含 <eos> “”）表示输出词汇表。因此，这套词汇集的基数 $\left|\mathcal{Y}\right|$ 就是词汇量大小。让我们还将输出序列的最大令牌数指定为 $T'$。因此，我们的目标是从所有 $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ 可能的输出序列中寻找理想的输出。当然，对于所有这些输出序列，包括 “<eos>” 之后的部分将被丢弃在实际输出中。

## 贪婪搜索

首先，让我们来看一个简单的策略：* 贪心搜索 *。该策略已被用来预测 :numref:`sec_seq2seq` 中的序列。在贪婪搜索中，在输出序列的任何时间步骤 $t'$，我们搜索条件概率从 $\mathcal{Y}$ 起最高的令牌，即

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

作为输出。一旦输 <eos> 出 “” 或输出序列达到最大长度 $T'$，输出序列就完成。

那么贪婪的搜索会出什么问题？事实上，* 最佳序列 * 应该是最大 $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$ 的输出序列，这是基于输入序列生成输出序列的条件概率。不幸的是，不能保证贪婪的搜索能够获得最佳顺序。

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

让我们用一个例子来说明它。假设 <eos> 输出字典中有四个标记 “A”、“B”、“C” 和 “”。在 :numref:`fig_s2s-prob1` 中，每个时间步长下的四个数字分别代表在 <eos> 该时间步长生成 “A”、“B”、“C” 和 “” 的条件概率。在每个时间步骤中，贪婪搜索都会选择条件概率最高的令牌。因此，输出序列 “A”、“B”、“C” 和 “<eos>” 将在 :numref:`fig_s2s-prob1` 中进行预测。此输出序列的条件概率为 $0.5\times0.4\times0.4\times0.6 = 0.048$。

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

接下来，让我们看一下 :numref:`fig_s2s-prob2` 中的另一个例子。与 :numref:`fig_s2s-prob1` 不同，我们在时间步骤 2 中选择 :numref:`fig_s2s-prob2` 中的令牌 “C”，该代币的条件概率为 * 秒 * 最高。由于时间步骤 1 和 2 的输出子序列（时间步骤 3 所基于的时间步骤 1 和 2）已从 :numref:`fig_s2s-prob1` 中的 “A” 和 “B” 变为 :numref:`fig_s2s-prob2` 中的 “A” 和 “C”，因此，时间步骤 3 中每个令牌的条件概率也在 :numref:`fig_s2s-prob2` 中发生了变化。假设我们在时间步骤 3 中选择令牌 “B”。现在，时间步长 4 取决于前三个时间步长 “A”、“C” 和 “B” 的输出子序列，这与 :numref:`fig_s2s-prob1` 中的 “A”、“B” 和 “C” 不同。因此，在 :numref:`fig_s2s-prob2` 的时间步骤 4 生成每个令牌的条件概率也与 :numref:`fig_s2s-prob1` 中的不同。因此，<eos> :numref:`fig_s2s-prob2` 中输出序列 “A”、“C”、“B” 和 “” 的条件概率为 $0.5\times0.3 \times0.6\times0.6=0.054$，比 :numref:`fig_s2s-prob1` 中贪婪搜索的概率大。在此示例中，<eos> 贪婪搜索获得的输出序列 “A”、“B”、“C” 和 “” 不是最佳序列。

## 详尽搜索

如果目标是获得最佳序列，我们可以考虑使用 * 详尽无遗的搜索 *：用条件概率详尽枚举所有可能的输出序列，然后输出条件概率最高的输出序列。

尽管我们可以使用详尽搜索来获得最佳序列，但其计算成本 $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ 可能会过高。例如，当 $|\mathcal{Y}|=10000$ 和 $T'=10$ 时，我们需要评估 $10000^{10} = 10^{40}$ 序列。这几乎是不可能的！另一方面，贪婪搜索的计算成本是 $\mathcal{O}(\left|\mathcal{Y}\right|T')$：它通常远低于详尽搜索。例如，当 $|\mathcal{Y}|=10000$ 和 $T'=10$ 时，我们只需要评估 $10000\times10=10^5$ 序列。

## 束搜索

关于序列搜索策略的决策取决于一个范围，在任何极端都很容易提出问题。如果只有准确性重要呢？显然，详尽的搜索。如果只有计算成本重要，该怎么办？显然，贪婪的搜索。真实世界的应用程序通常会提出一个复杂的问题，介于这两个极端之间。

*Beam search * 是贪婪搜索的改进版本。它有一个名为 * 束尺寸 * 的超参数，$k$。
在时间步骤 1，我们选择了条件概率最高的 $k$ 令牌。他们每个人都将分别成为 $k$ 个候选输出序列的第一个令牌。在后续的每个时间步中，根据上一个时间步的 $k$ 个候选输出序列，我们继续选择 $k$ 个候选输出序列，其条件概率为 $k\left|\mathcal{Y}\right|$ 个可能的选择。

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search` 以示例演示了光束搜索的过程。假设输出词汇表只包含五个元素：$\mathcal{Y} = \{A, B, C, D, E\}$，其中一个是 “<eos>”。让波束大小为 2，输出序列的最大长度为 3。在时间步骤 1，假设条件概率最高 $P(y_1 \mid \mathbf{c})$ 的令牌分别为 $A$ 和 $C$。在时间步骤 2，我们计算了所有 $y_2 \in \mathcal{Y},$

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

然后选择这十个值中最大的两个，比如 $P(A, B \mid \mathbf{c})$ 和 $P(C, E \mid \mathbf{c})$。然后在时间步骤 3，对于所有 $y_3 \in \mathcal{Y}$，我们计算

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

然后选择这十个值中最大的两个，比如 $P(A, B, D \mid \mathbf{c})$ 和 $P(C, E, D \mid  \mathbf{c}).$ 因此，我们得到了六个候选输出序列：(i) $A$; (ii) $C$; (iii) 73229293617; (iv) 73229293617; (iv) 73229293614; (v) 73229293618, $E$; (v) $A$ 17,$D$; 以及 (六) $C$、$D$、$D$。

最后，我们获得基于这六个序列的最终候选输出序列集（例如，丢弃包括 “<eos>” 之后的部分）。然后我们选择以下分数最高的序列作为输出序列：

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

其中 $L$ 是最终候选序列的长度，$\alpha$ 通常设置为 0.75。由于在 :eqref:`eq_beam-search-score` 的总和中，较长的序列具有更多的对数术语，分母中的术语 $L^\alpha$ 将处罚长序列。

光束搜索的计算成本为 $\mathcal{O}(k\left|\mathcal{Y}\right|T')$。这种结果介于贪婪搜索和详尽搜索的结果之间。事实上，贪婪搜索可以被视为波束大小为 1 的特殊类型的光束搜索。通过灵活选择光束尺寸，光束搜索可在准确性与计算成本之间进行权衡。

## 摘要

* 序列搜索策略包括贪婪搜索、详尽搜索和束搜索。
* 光束搜索通过灵活选择光束尺寸，在准确性与计算成本之间进行权衡。

## 练习

1. 我们可以将详尽搜索视为一种特殊类型的光束搜索吗？为什么或为什么不？
1. 在 :numref:`sec_seq2seq` 中的机器翻译问题中应用束搜索。光束大小如何影响翻译结果和预测速度？
1. 我们使用语言建模在 :numref:`sec_rnn_scratch` 中用户提供的前缀生成文本。它使用哪种搜索策略？你能改进吗？

[Discussions](https://discuss.d2l.ai/t/338)
