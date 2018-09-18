# 注意力机制

在编码器—解码器（seq2seq）模型里，解码器依赖背景变量来获取输入序列信息。在[“编码器—解码器（seq2seq）”](seq2seq.md)一节里，我们介绍了如果使用了循环网络作为编码器，可以将其最后一个时间步的隐藏状态输出作为背景变量。但如果我们仔细观察那一节使用的“They are watching.”到“Ils regardent.”的翻译例子，可以看到解码器可以在输出序列的时间步1主要用到“They”、“are”的信息来生成“Ils”，在时间步2则需要编码了“watching”信息，最后则直接将“.”映射过来。也就是解码器在不同时间步中可能关注输入序列中的不同部分。使用固定的背景变量可能使得信息的使用不够高效。

注意力机制[1]的提出正是为了解决这个问题。仍然以循环神经网络为例，它通过对编码器所有时间步的隐藏状态输出做加权和来得到背景变量，我们通过调整这些权重，也称为注意力权重，来使得在不同的解码时间步中可以关注输入序列中的不同部分。本小节我们将讨论注意力机制是怎么工作的。

## 注意力机制的模型设计


在[“编码器—解码器（seq2seq）”](seq2seq.md)一节里我们介绍了解码时间步$t'$中的隐藏状态是通过$\boldsymbol{s}_{t'} = g(\boldsymbol{y}_{t'-1}, \boldsymbol{c}, \boldsymbol{s}_{t'-1})$来更新，这里$\boldsymbol{y}_{t'-1}$是输入词的$y_{t'-1}$的特征表示，且所有时间步里使用同样的背景变量$\boldsymbol{c}$。但注意力机制中，我们将使用可变的背景变量。记$\boldsymbol{c}_{t'}$是用于时间步$t'$的背景变量，那么解码器在时间步$t'$的隐藏状态为

$$\boldsymbol{s}_{t'} = g(\boldsymbol{y}_{t'-1}, \boldsymbol{c}_{t'}, \boldsymbol{s}_{t'-1}).$$

这里的关键步是如何计算背景变量$\boldsymbol{c}_{t'}$和如何利用其来更新隐藏状态。


## 计算背景变量

下图演示了注意力机制是如何为解码器的时间步2计算背景变量。首先通过函数$a$我们计算当前时间步隐藏状态输入和编码器所有时间里的隐藏状态输出计算分数，然后通过softmax将这些分数转换成注意力权重。最后对编码器三个时间步的隐藏状态进行加权和来得到背景变量。

![应用在seq2seq上的注意力机制。](../img/attention.svg)


具体来说，令编码器在时间步$t$的隐藏状态为$\boldsymbol{h}_t$，且总时间步数为$T$。那么解码器在时间步$t'$的背景变量为所有编码器隐藏状态的加权和：

$$\boldsymbol{c}_{t'} = \sum_{t=1}^T \alpha_{t' t} \boldsymbol{h}_t,$$

其中$\alpha_{t' t}$是权值。为了计算权重，我们首先通过函数$a$来衡量解码器隐藏状态$\boldsymbol{s}_{t' - 1}$和所有编码器隐藏状态的相似度，即对$t=1,\ldots,T$，

$$e_{t' t} = a(\boldsymbol{s}_{t' - 1}, \boldsymbol{h}_t).$$

这里$a$有多种选择，最简单的形式是计算内积$a(\boldsymbol{s}, \boldsymbol{h})=\boldsymbol{s}^\top \boldsymbol{h}$，[1]则将输入隐藏状态合并起来后输入到使用tanh激活函数的双层感知机：

$$a(\boldsymbol{s}, \boldsymbol{h}) = \boldsymbol{v}^\top \tanh(\boldsymbol{W}_s \boldsymbol{s} + \boldsymbol{W}_h \boldsymbol{h}),$$

其中$\boldsymbol{v}$、$\boldsymbol{W}_s$、$\boldsymbol{W}_h$是可以学习的模型参数。最后对所有分数作用softmax得到注意力权重。

$$\alpha_{t' t} = \frac{\exp(e_{t' t})}{ \sum_{k=1}^T \exp(e_{t' k}) },\quad t=1,\ldots,T.$$

## 更新隐藏状态

在seq2seq中我们并没有直接使用背景变量来参与隐藏状态的更新，而是将其当做解码器的初始隐藏状态。但在这里我们不能这样使用，这是因为背景变量在时间步之间是变化的。[1]中对门控循环单元的设计稍作修改。解码器在$t' $时间步的隐藏状态为

$$\boldsymbol{s}_{t'} = \boldsymbol{z}_{t'} \odot \boldsymbol{s}_{t'-1}  + (1 - \boldsymbol{z}_{t'}) \odot \tilde{\boldsymbol{s}}_{t'},$$

其中的重置门、更新门和候选隐含状态分别为


$$
\begin{aligned}
\boldsymbol{r}_{t'} &= \sigma(\boldsymbol{W}_{yr} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sr} \boldsymbol{s}_{t' - 1} + \boldsymbol{W}_{cr} \boldsymbol{c}_{t'} + \boldsymbol{b}_r),\\
\boldsymbol{z}_{t'} &= \sigma(\boldsymbol{W}_{yz} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sz} \boldsymbol{s}_{t' - 1} + \boldsymbol{W}_{cz} \boldsymbol{c}_{t'} + \boldsymbol{b}_z),\\
\tilde{\boldsymbol{s}}_{t'} &= \text{tanh}(\boldsymbol{W}_{ys} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{ss} (\boldsymbol{s}_{t' - 1} \odot \boldsymbol{r}_{t'}) + \boldsymbol{W}_{cs} \boldsymbol{c}_{t'} + \boldsymbol{b}_s).
\end{aligned}
$$

这里引入了新的模型参数$\boldsymbol{W}_{cr}$，$\boldsymbol{W}_{cz}$和$\boldsymbol{W}_{cs}$来将背景变量纳入到计算中。

## 小结

* 我们可以在解码器的每个时间步使用不同的背景变量，并对输入序列中不同时间步编码的信息分配不同的注意力。

## 练习

* 不修改[“门控循环单元（GRU）”](../chapter_recurrent-neural-networks/gru.md)一节中的`gru_rnn`函数，应如何用它实现本节介绍的解码器？

* 除了自然语言处理，注意力机制还可以应用在哪些地方？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6759)

![](../img/qr_attention.svg)


## 参考文献

[1] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
