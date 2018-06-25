# 注意力机制

在[“编码器—解码器（seq2seq）”](seq2seq.md)一节里的解码器设计中，输出序列的各个时间步使用了相同的背景变量。如果解码器的不同时间步可以使用不同的背景变量呢？这样做有什么好处？


## 动机

以英语-法语翻译为例，给定一对输入序列“They”、“are”、“watching”、“.”和输出序列“Ils”、“regardent”、“.”。解码器可以在输出序列的时间步1使用更多编码了“They”、“are”信息的背景变量来生成“Ils”，在时间步2使用更多编码了“watching”信息的背景变量来生成“regardent”，在时间步3使用更多编码了“.”信息的背景变量来生成“.”。这看上去就像是在解码器的每一时间步对输入序列中不同时间步编码的信息分配不同的注意力。这也是注意力机制的由来。它最早由Bahanau等提出 [1]。


## 设计

本节沿用[“编码器—解码器（seq2seq）”](seq2seq.md)一节里的数学符号。

我们对[“编码器—解码器（seq2seq）”](seq2seq.md)一节里的解码器稍作修改。在时间步$t^\prime$，设解码器的背景变量为$\boldsymbol{c}_{t^\prime}$，输出$y_{t^\prime}$的特征向量为$\boldsymbol{y}_{t^\prime}$。
和输入的特征向量一样，这里每个输出的特征向量也可能是模型参数。解码器在时间步$t^\prime$的隐藏状态

$$\boldsymbol{s}_{t^\prime} = g(\boldsymbol{y}_{t^\prime-1}, \boldsymbol{c}_{t^\prime}, \boldsymbol{s}_{t^\prime-1}).$$


令编码器在时间步$t$的隐藏状态为$\boldsymbol{h}_t$，且时间步数为$T$。解码器在时间步$t^\prime$的背景变量为

$$\boldsymbol{c}_{t^\prime} = \sum_{t=1}^T \alpha_{t^\prime t} \boldsymbol{h}_t,$$

其中$\alpha_{t^\prime t}$是权值。也就是说，给定解码器的当前时间步$t^\prime$，我们需要对编码器中不同时间步$t$的隐藏状态求加权平均。这里的权值也称注意力权重。它的计算公式是

$$\alpha_{t^\prime t} = \frac{\exp(e_{t^\prime t})}{ \sum_{k=1}^T \exp(e_{t^\prime k}) },$$

其中$e_{t^\prime t} \in \mathbb{R}$的计算为

$$e_{t^\prime t} = a(\boldsymbol{s}_{t^\prime - 1}, \boldsymbol{h}_t).$$

上式中的函数$a$有多种设计方法。Bahanau等使用了

$$e_{t^\prime t} = \boldsymbol{v}^\top \tanh(\boldsymbol{W}_s \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_h \boldsymbol{h}_t),$$

其中$\boldsymbol{v}$、$\boldsymbol{W}_s$、$\boldsymbol{W}_h$以及编码器与解码器中的各个权重和偏差都是模型参数 [1]。

Bahanau等在编码器和解码器中分别使用了门控循环单元 [1]。在解码器中，我们需要对门控循环单元的设计稍作修改。解码器在$t^\prime$时间步的隐藏状态为

$$\boldsymbol{s}_{t^\prime} = \boldsymbol{z}_{t^\prime} \odot \boldsymbol{s}_{t^\prime-1}  + (1 - \boldsymbol{z}_{t^\prime}) \odot \tilde{\boldsymbol{s}}_{t^\prime},$$

其中的重置门、更新门和候选隐含状态分别为


$$
\begin{aligned}
\boldsymbol{r}_{t^\prime} &= \sigma(\boldsymbol{W}_{yr} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{sr} \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_{cr} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_r),\\
\boldsymbol{z}_{t^\prime} &= \sigma(\boldsymbol{W}_{yz} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{sz} \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_{cz} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_z),\\
\tilde{\boldsymbol{s}}_{t^\prime} &= \text{tanh}(\boldsymbol{W}_{ys} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{ss} (\boldsymbol{s}_{t^\prime - 1} \odot \boldsymbol{r}_{t^\prime}) + \boldsymbol{W}_{cs} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_s).
\end{aligned}
$$


我们将在下一节中实现含注意力机制的编码器和解码器。


## 小结

* 我们可以在解码器的每个时间步使用不同的背景变量，并对输入序列中不同时间步编码的信息分配不同的注意力。

## 练习

* 不修改[“门控循环单元（GRU）——从零开始”](../chapter_recurrent-neural-networks/gru-scratch.md)一节中的`gru_rnn`函数，应如何用它实现本节介绍的解码器？

* 除了自然语言处理，注意力机制还可以应用在哪些地方？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6759)

![](../img/qr_attention.svg)


## 参考文献

[1] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
