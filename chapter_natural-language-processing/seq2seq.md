# 编码器—解码器（seq2seq）


在基于词语的语言模型中，我们使用了[循环神经网络](../chapter_recurrent-neural-networks/rnn-gluon.md)。它的输入是一段不定长的序列，输出却是定长的，例如一个词语。然而，很多问题的输出也是不定长的序列。以机器翻译为例，输入是可以是英语的一段话，输出可以是法语的一段话，输入和输出皆不定长，例如

> 英语：They are watching.

> 法语：Ils regardent.

当输入输出都是不定长序列时，我们可以使用编码器—解码器（encoder-decoder）或者seq2seq。它们分别基于2014年的两个工作：

* Cho et al., [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://www.aclweb.org/anthology/D14-1179)
* Sutskever et al., [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

以上两个工作本质上都用到了两个循环神经网络，分别叫做编码器和解码器。编码器对应输入序列，解码器对应输出序列。下面我们来介绍编码器—解码器的设计。


## 编码器—解码器

编码器和解码器是分别对应输入序列和输出序列的两个循环神经网络。我们通常会在输入序列和输出序列后面分别附上一个特殊字符“&lt;eos&gt;”（end of sequence）表示序列的终止。在测试模型时，一旦输出“&lt;eos&gt;”就终止当前的输出序列。

### 编码器

编码器的作用是把一个不定长的输入序列转化成一个定长的背景向量$\boldsymbol{c}$。该背景向量包含了输入序列的信息。常用的编码器是循环神经网络。

我们回顾一下[循环神经网络](../chapter_recurrent-neural-networks/rnn-scratch.md)知识。假设循环神经网络单元为$f$，在$t$时刻的输入为$x_t, t=1, \ldots, T$。
假设$\boldsymbol{x}_t$是单个输出$x_t$在嵌入层的结果，例如$x_t$对应的one-hot向量$\boldsymbol{o} \in \mathbb{R}^x$与嵌入层参数矩阵$\boldsymbol{E} \in \mathbb{R}^{x \times h}$的乘积$\boldsymbol{o}^\top \boldsymbol{E}$。隐含层变量

$$\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}) $$

编码器的背景向量

$$\boldsymbol{c} =  q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T)$$

一个简单的背景向量是该网络最终时刻的隐含层变量$\boldsymbol{h}_T$。
我们将这里的循环神经网络叫做编码器。

#### 双向循环神经网络

编码器的输入既可以是正向传递，也可以是反向传递。如果输入序列是$x_1, x_2, \ldots, x_T$，在正向传递中，隐含层变量

$$\overrightarrow{\boldsymbol{h}}_t = f(\boldsymbol{x}_t, \overrightarrow{\boldsymbol{h}}_{t-1}) $$


而反向传递中，隐含层变量的计算变为

$$\overleftarrow{\boldsymbol{h}}_t = f(\boldsymbol{x}_t, \overleftarrow{\boldsymbol{h}}_{t+1}) $$




当我们希望编码器的输入既包含正向传递信息又包含反向传递信息时，我们可以使用双向循环神经网络。例如，给定输入序列$x_1, x_2, \ldots, x_T$，按正向传递，它们在循环神经网络的隐含层变量分别是$\overrightarrow{\boldsymbol{h}}_1, \overrightarrow{\boldsymbol{h}}_2, \ldots, \overrightarrow{\boldsymbol{h}}_T$；按反向传递，它们在循环神经网络的隐含层变量分别是$\overleftarrow{\boldsymbol{h}}_1, \overleftarrow{\boldsymbol{h}}_2, \ldots, \overleftarrow{\boldsymbol{h}}_T$。在双向循环神经网络中，时刻$i$的隐含层变量可以把$\overrightarrow{\boldsymbol{h}}_i$和$\overleftarrow{\boldsymbol{h}}_i$连结起来。

### 解码器

编码器最终输出了一个背景向量$\boldsymbol{c}$，该背景向量编码了输入序列$x_1, x_2, \ldots, x_T$的信息。

假设训练数据中的输出序列是$y_1, y_2, \ldots, y_{T^\prime}$，我们希望表示每个$t$时刻输出的既取决于之前的输出又取决于背景向量。之后，我们就可以最大化输出序列的联合概率

$$\mathbb{P}(y_1, \ldots, y_{T^\prime}) = \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c})$$


并得到该输出序列的损失函数

$$- \log\mathbb{P}(y_1, \ldots, y_{T^\prime})$$

为此，我们使用另一个循环神经网络作为解码器。解码器使用函数$p$来表示单个输出$y_{t^\prime}$的概率

$$\mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}) = p(y_{t^\prime-1}, \boldsymbol{s}_{t^\prime}, \boldsymbol{c})$$

其中的$\boldsymbol{s}_t$为$t^\prime$时刻的解码器的隐含层变量。该隐含层变量

$$\boldsymbol{s}_{t^\prime} = g(y_{t^\prime-1}, \boldsymbol{c}, \boldsymbol{s}_{t^\prime-1})$$

其中函数$g$是循环神经网络单元。

需要注意的是，编码器和解码器通常会使用[多层循环神经网络](../chapter_recurrent-neural-networks/rnn-gluon.md)。


## 小结

* 编码器-解码器（seq2seq）的输入和输出可以都是不定长序列。


## 练习

* 了解其他的注意力机制设计。例如论文[Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)。

* 在[Bahanau的论文](https://arxiv.org/abs/1409.0473)中，我们是否需要重新实现解码器上的GRU？

* 除了机器翻译，你还能想到seq2seq的哪些应用？

* 除了自然语言处理，注意力机制还可以应用在哪些地方？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4523)

![](../img/qr_seq2seq.svg)
