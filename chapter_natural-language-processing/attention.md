# 注意力机制


在以上的解码器设计中，各个时刻使用了相同的背景向量。如果解码器的不同时刻可以使用不同的背景向量呢？

## 设计


以英语-法语翻译为例，给定一对输入序列“they are watching”和输出序列“Ils regardent”，解码器在时刻1可以使用更多编码了“they are”信息的背景向量来生成“Ils”，而在时刻2可以使用更多编码了“watching”信息的背景向量来生成“regardent”。这看上去就像是在解码器的每一时刻对输入序列中不同时刻分配不同的注意力。这也是注意力机制的由来。它最早[由Bahanau等在2015年提出](https://arxiv.org/abs/1409.0473)。

现在，对上面的解码器稍作修改。我们假设时刻$t^\prime$的背景向量为$\boldsymbol{c}_{t^\prime}$。那么解码器在$t^\prime$时刻的隐含层变量

$$\boldsymbol{s}_{t^\prime} = g(\boldsymbol{y}_{t^\prime-1}, \boldsymbol{c}_{t^\prime}, \boldsymbol{s}_{t^\prime-1})$$


令编码器在$t$时刻的隐含变量为$\boldsymbol{h}_t$，解码器在$t^\prime$时刻的背景向量为

$$\boldsymbol{c}_{t^\prime} = \sum_{t=1}^T \alpha_{t^\prime t} \boldsymbol{h}_t$$


也就是说，给定解码器的当前时刻$t^\prime$，我们需要对编码器中不同时刻$t$的隐含层变量求加权平均。而权值也称注意力权重。它的计算公式是

$$\alpha_{t^\prime t} = \frac{\exp(e_{t^\prime t})}{ \sum_{k=1}^T \exp(e_{t^\prime k}) } $$

而$e_{t^\prime t} \in \mathbb{R}$的计算为：

$$e_{t^\prime t} = a(\boldsymbol{s}_{t^\prime - 1}, \boldsymbol{h}_t)$$

其中函数$a$有多种设计方法。在[Bahanau的论文](https://arxiv.org/abs/1409.0473)中，

$$e_{t^\prime t} = \boldsymbol{v}^\top \tanh(\boldsymbol{W}_s \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_h \boldsymbol{h}_t)$$

其中的$\boldsymbol{v}$、$\boldsymbol{W}_s$、$\boldsymbol{W}_h$和编码器与解码器两个循环神经网络中的各个权重和偏移项以及嵌入层参数等都是需要同时学习的模型参数。在[Bahanau的论文](https://arxiv.org/abs/1409.0473)中，编码器和解码器分别使用了[门控循环单元（GRU）](../chapter_recurrent-neural-networks/gru-scratch.md)。


在解码器中，我们需要对GRU的设计稍作修改。
假设$\boldsymbol{y}_t$是单个输出$y_t$在嵌入层的结果，例如$y_t$对应的one-hot向量$\boldsymbol{o} \in \mathbb{R}^y$与嵌入层参数矩阵$\boldsymbol{B} \in \mathbb{R}^{y \times s}$的乘积$\boldsymbol{o}^\top \boldsymbol{B}$。
假设时刻$t^\prime$的背景向量为$\boldsymbol{c}_{t^\prime}$。那么解码器在$t^\prime$时刻的单个隐含层变量

$$\boldsymbol{s}_{t^\prime} = \boldsymbol{z}_{t^\prime} \odot \boldsymbol{s}_{t^\prime-1}  + (1 - \boldsymbol{z}_{t^\prime}) \odot \tilde{\boldsymbol{s}}_{t^\prime}$$

其中的重置门、更新门和候选隐含状态分别为


$$\boldsymbol{r}_{t^\prime} = \sigma(\boldsymbol{W}_{yr} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{sr} \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_{cr} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_r)$$

$$\boldsymbol{z}_{t^\prime} = \sigma(\boldsymbol{W}_{yz} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{sz} \boldsymbol{s}_{t^\prime - 1} + \boldsymbol{W}_{cz} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_z)$$

$$\tilde{\boldsymbol{s}}_{t^\prime} = \text{tanh}(\boldsymbol{W}_{ys} \boldsymbol{y}_{t^\prime-1} + \boldsymbol{W}_{ss} (\boldsymbol{s}_{t^\prime - 1} \odot \boldsymbol{r}_{t^\prime}) + \boldsymbol{W}_{cs} \boldsymbol{c}_{t^\prime} + \boldsymbol{b}_s)$$

## 小结

* 在解码器上应用注意力机制可以在解码器的每个时刻使用不同的背景向量。每个背景向量相当于对输入序列的不同部分分配了不同的注意力。


## 练习

* 了解其他的注意力机制设计。例如论文[Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)。

* 在[Bahanau的论文](https://arxiv.org/abs/1409.0473)中，我们是否需要重新实现解码器上的GRU？

* 除了机器翻译，你还能想到seq2seq的哪些应用？

* 除了自然语言处理，注意力机制还可以应用在哪些地方？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6759)

![](../img/qr_seq2seq-attention.svg)
