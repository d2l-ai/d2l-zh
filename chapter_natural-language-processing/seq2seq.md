# 编码器—解码器（seq2seq）

在很多应用中，输入和输出都可以是不定长序列。以机器翻译为例，输入可以是一段不定长的英语文本序列，输出可以是一段不定长的法语文本序列，例如

> 英语输入：“They”、“are”、“watching”、“.”

> 法语输出：“Ils”、“regardent”、“.”

当输入输出都是不定长序列时，我们可以使用编码器—解码器（encoder-decoder）[1] 或者seq2seq模型 [2]。这两个模型本质上都用到了两个循环神经网络，分别叫做编码器和解码器。编码器对应输入序列，解码器对应输出序列。

下图演示了如何使用基于循环神经网络的编码器—解码器来训练将上述英语句子翻译成法语句子。首先将英语句子的句号替换成结束符“<EOS>”（end of sentence)后输入到编码循环神经网络来得到结束符输出的隐藏状态。接着我们使用以它为初始隐藏状态的解码循环神经网络来进行翻译。我们依次将开始符“<BOS>”（begin of sentence）和对应的法语句子输入到解码器中，期望解码器能输出正确的输入单词的下一词。

![编码器—解码器。](../img/seq2seq.svg)

接下来我们介绍编码器和解码器的定义。

## 编码器

编码器的作用是把一个不定长的输入序列变换成一个定长的背景变量$\boldsymbol{c}$，并在该背景变量中编码输入序列信息。常用的编码器是循环神经网络，其隐藏状态输出则当做背景变量。

假设输入序列是$x_1,\ldots,x_T$，这里考虑批量大小为1的情况，例如$x_i$是输入句子中的第$i$个词。在时间步$t$中，循环神经网络将输入$x_t$的特征向量$\boldsymbol{x}_t$和上个时间步的隐藏状态$\boldsymbol{h}_{t-1}$变换为当前时间步的隐藏状态$\boldsymbol{h}_t$。我们可以用函数$f$表达循环神经网络隐藏层的变换：

$$\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}). $$

这里特征向量$\boldsymbol{x}_t$既可以是[“循环神经网络”](../chapter_recurrent-neural-networks/rnn.md)一节中介绍的one-hot表示，也可以是前面小节介绍的词嵌入。

接下来编码器通过自定义函数$q$将各个时间步的隐藏状态变换为背景变量

$$\boldsymbol{c} =  q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T).$$

例如，当选择$q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T) = \boldsymbol{h}_T$时，背景变量是输入序列最终时间步的隐藏状态$\boldsymbol{h}_T$。

以上描述的编码器是一个单向的循环神经网络，每个时间步的隐藏状态只取决于该时间步及之前的输入子序列。我们也可以使用双向循环神经网络构造编码器。这种情况下，编码器每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入），并编码了整个序列的信息。

## 解码器


刚刚已经介绍编码器输出的背景变量$\boldsymbol{c}$编码了整个输入序列$x_1, \ldots, x_T$的信息。给定训练样本中的输出序列$y_1, y_2, \ldots, y_{T'}$，对每个时间步$t'$，解码器输出$y_{t'}$基于之前输出序列$y_1,\ldots,y_{t'-1}$和背景变量$\boldsymbol{c}$的条件概率，即$\mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c})$。

如果我们也使用循环神经网络作为解码器。首先将其初始隐藏状态$\boldsymbol{s}_0$设为背景变量$\boldsymbol{c}$。假设$\boldsymbol{y}_{t'}$是$y_{t'}$的特征，那么对每个时间步$t'=1,\ldots,T'$，首先更新隐藏状态：

$$\boldsymbol{s}_{t'} = g(\boldsymbol{y}_{t'-1}, \boldsymbol{c}, \boldsymbol{s}_{t'-1}),$$

然后计算当前时间步输出

$$\boldsymbol{\hat y}_{t'} = u\left(\boldsymbol{y}_{t'-1}, \boldsymbol{c}, \boldsymbol{s}_{t'-1}\right),$$

再应用softmax后便可以得到条件概率输出，即

$$\mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}) = \mathrm{softmax}\left(\boldsymbol{\hat y}_{t'}\right).$$

## 模型训练

根据最大似然估计，我们可以知道输出序列基于输入序列的条件概率为


$$
\begin{aligned}
\mathbb{P}(y_1, \ldots, y_{T^\prime} \mid x_1, \ldots, x_T)
&= \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, x_1, \ldots, x_T)\\
&= \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}),
\end{aligned}
$$

设负对数最大似然估计为损失函数，即


$$- \log\mathbb{P}(y_1, \ldots, y_{T^\prime} \mid x_1, \ldots, x_T) = -\sum_{t^\prime=1}^{T^\prime} \log \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}),$$

在模型训练中，我们通过最小化这个损失函数来训练模型参数。


## 小结

* 编码器-解码器（seq2seq）可以输入并输出不定长的序列。
* 编码器—解码器使用了两个循环神经网络。
* 预测不定长序列的方法包括穷举搜索、贪婪搜索和束搜索。


（TODO @aston, 因为提到了[1]和[2], 最好解释下两边论文的区别。）

## 练习

* 除了机器翻译，你还能想到seq2seq的哪些应用？

* 有哪些方法可以设计解码器的输出层？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4523)

![](../img/qr_seq2seq.svg)

## 参考文献

[1] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

[2] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
