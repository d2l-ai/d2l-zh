# 束搜索

上一节介绍了如何训练输入输出均为不定长序列的编码器—解码器。在准备训练数据集时，我们通常会在样本的输入序列和输出序列后面分别附上一个特殊符号“&lt;eos&gt;”（end of sequence）表示序列的终止。在预测中，模型该如何输出不定长序列呢？

为了便于讨论，假设解码器的输出是一段文本序列。我们将在本节的讨论中沿用上一节的数学符号。


## 穷举搜索

设输出文本词典$\mathcal{Y}$（包含特殊符号“&lt;eos&gt;”）的大小为$|\mathcal{Y}|$，输出序列的最大长度为$T^\prime$。那么，所有可能的输出序列一共有$\mathcal{O}(|\mathcal{Y}|^{T^\prime})$种。这些输出序列中所有特殊符号“&lt;eos&gt;”及其后面的子序列将被舍弃。


我们在描述解码器时提到，输出序列基于输入序列的条件概率是$\prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c})$。为了搜索该概率最大的输出序列，一种方法是穷举所有可能序列的概率，并输出概率最大的序列。我们将该序列称为最优序列，并将这种搜索方法称为穷举搜索（exhaustive search）。很明显，穷举搜索的计算开销$\mathcal{O}(|\mathcal{Y}|^{T^\prime})$很容易过高而无法使用（例如，$10000^{10} = 1 \times 10^{40}$）。


## 贪婪搜索

我们还可以使用贪婪搜索（greedy search）。也就是说，对于输出序列任一时间步$t^\prime$，从$|\mathcal{Y}|$个词中搜索出输出词

$$y_{t^\prime} = \text{argmax}_{y_{t^\prime} \in \mathcal{Y}} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}),$$

且一旦搜索出“&lt;eos&gt;”符号即完成输出。


设输出文本词典$\mathcal{Y}$的大小为$|\mathcal{Y}|$，输出序列的最大长度为$T^\prime$。
贪婪搜索的计算开销是$\mathcal{O}(|\mathcal{Y}| \times {T^\prime})$。它比起穷举搜索的计算开销显著下降（例如，$10000 \times 10 = 1 \times 10^5$）。然而，贪婪搜索并不能保证输出是最优序列。


## 束搜索


束搜索（beam search）介于上面二者之间。我们通过一个具体例子描述它。

假设输出序列的词典中只包含五个元素：$\mathcal{Y} = \{A, B, C, D, E\}$，且其中一个为特殊符号“&lt;eos&gt;”。设束搜索的超参数束宽（beam width）等于2，输出序列最大长度为3。

在输出序列的时间步1时，假设条件概率$\mathbb{P}(y_{t^\prime} \mid \boldsymbol{c})$最大的两个词为$A$和$C$。我们在时间步2时将对所有的$y_2 \in \mathcal{Y}$都分别计算$\mathbb{P}(y_2 \mid A, \boldsymbol{c})$和$\mathbb{P}(y_2 \mid C, \boldsymbol{c})$，并从计算出的10个概率中取最大的两个：假设为$\mathbb{P}(B \mid A, \boldsymbol{c})$和$\mathbb{P}(E \mid C, \boldsymbol{c})$。那么，我们在时间步3时将对所有的$y_3 \in \mathcal{Y}$都分别计算$\mathbb{P}(y_3 \mid A, B, \boldsymbol{c})$和$\mathbb{P}(y_3 \mid C, E, \boldsymbol{c})$，并从计算出的10个概率中取最大的两个：假设为$\mathbb{P}(D \mid A, B, \boldsymbol{c})$和$\mathbb{P}(D \mid C, E, \boldsymbol{c})$。

接下来，我们可以在6个输出序列：$A$、$C$、$AB$、$CE$、$ABD$、$CED$中筛选出包含特殊符号“&lt;eos&gt;”的序列，并将它们中所有特殊符号“&lt;eos&gt;”及其后面的子序列舍弃，得到候选序列。在这些候选序列中，取以下分数最高的序列作为输出序列：

$$ \frac{1}{L^\alpha} \log \mathbb{P}(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t^\prime=1}^L \log \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}),$$

其中$L$为候选序列长度，$\alpha$一般可选为0.75。分母上的$L^\alpha$是为了惩罚较长序列在以上分数中较多的对数相加项。

穷举搜索和贪婪搜索也可看作是两种特殊束宽的束搜索。束搜索通过更灵活的束宽来权衡计算开销和搜索质量。


## 小结

* 预测不定长序列的方法包括穷举搜索、贪婪搜索和束搜索。
* 束搜索通过更灵活的束宽来权衡计算开销和搜索质量。


## 练习

* 在[“循环神经网络——从零开始”](../chapter_recurrent-neural-networks/rnn-scratch.md)一节中，我们使用语言模型创作歌词。它的输出属于哪种搜索？你能改进它吗？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6817)

![](../img/qr_beam-search.svg)
