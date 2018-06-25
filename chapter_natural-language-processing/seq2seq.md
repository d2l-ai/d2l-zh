# 编码器—解码器（seq2seq）

在很多应用中，输入和输出都可以是不定长序列。以机器翻译为例，输入是可以是一段不定长的英语文本序列，输出可以是一段不定长的法语文本序列，例如

> 英语：“They”、“are”、“watching”、“.”

> 法语：“Ils”、“regardent”、“.”

当输入输出都是不定长序列时，我们可以使用编码器—解码器（encoder-decoder）[1] 或者seq2seq模型 [2]。这两个模型本质上都用到了两个循环神经网络，分别叫做编码器和解码器。编码器对应输入序列，解码器对应输出序列。下面我们来介绍编码器—解码器的设计。


## 编码器

编码器的把一个不定长的输入序列变换成一个定长的背景变量$\boldsymbol{c}$，并在该背景变量中编码输入序列信息。常用的编码器是循环神经网络。

让我们考虑批量大小为1的时序数据样本。
在时间步$t$，循环神经网络将输入$x_t$的特征向量$\boldsymbol{x}_t$和上个时间步的隐藏状态$\boldsymbol{h}_{t-1}$变换为当前时间步的隐藏状态$\boldsymbol{h}_t$。
其中，每个输入的特征向量可能是模型参数，例如[“循环神经网络——使用Gluon”](../chapter_recurrent-neural-networks/rnn-gluon.md)一节中需要学习的每个词向量。我们可以用函数$f$表达循环神经网络隐藏层的变换：

$$\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}). $$

假设输入序列的时间步数为$T$。编码器通过自定义函数$q$将各个时间步的隐藏状态变换为背景变量

$$\boldsymbol{c} =  q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T).$$

例如，当选择$q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T) = \boldsymbol{h}_T$时，背景变量是输入序列最终时间步的隐藏状态$\boldsymbol{h}_T$。
我们将这里的循环神经网络叫做编码器。

以上描述的编码器是一个单向的循环神经网络，每个时间步的隐藏状态只取决于该时间步及之前的输入子序列。我们也可以使用双向循环神经网络构造编码器。这种情况下，编码器每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入），并编码了整个序列的信息。



## 解码器

刚刚已经介绍，假设输入序列的时间步数为$T$，编码器输出的背景变量$\boldsymbol{c}$编码了整个输入序列$x_1, \ldots, x_T$的信息。给定训练样本中的输出序列$y_1, y_2, \ldots, y_{T^\prime}$。假设其中每个时间步$t^\prime$的输出同时取决于该时间步之前的输出序列和背景变量。那么，根据最大似然估计，我们可以最大化输出序列基于输入序列的条件概率

$$
\begin{aligned}
\mathbb{P}(y_1, \ldots, y_{T^\prime} \mid x_1, \ldots, x_T)
&= \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, x_1, \ldots, x_T)\\
&= \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c}),
\end{aligned}
$$


并得到该输出序列的损失

$$- \log\mathbb{P}(y_1, \ldots, y_{T^\prime}).$$

为此，我们可以使用另一个循环神经网络作为解码器。
在输出序列的时间步$t^\prime$，解码器将上一时间步的输出$y_{t^\prime-1}$以及背景变量$\boldsymbol{c}$作为输入，并将它们与上一时间步的隐藏状态$\boldsymbol{s}_{t^\prime-1}$变换为当前时间步的隐藏状态$\boldsymbol{s}_{t^\prime}$。因此，我们可以用函数$g$表达解码器隐藏层的变换：

$$\boldsymbol{s}_{t^\prime} = g(y_{t^\prime-1}, \boldsymbol{c}, \boldsymbol{s}_{t^\prime-1}).$$

有了解码器的隐藏状态后，我们可以自定义输出层来计算损失中的$\mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c})$，例如基于当前时间步的解码器隐藏状态 $\boldsymbol{s}_{t^\prime}$、上一时间步的输出$y_{t^\prime-1}$以及背景变量$\boldsymbol{c}$来计算当前时间步输出$y_{t^\prime}$的概率分布。

在实际中，我们常使用深度循环神经网络作为编码器和解码器。我们将在本章稍后的[“机器翻译”](nmt.md)一节中实现含深度循环神经网络的编码器和解码器。


## 小结

* 编码器-解码器（seq2seq）可以输入并输出不定长的序列。
* 编码器—解码器使用了两个循环神经网络。
* 预测不定长序列的方法包括穷举搜索、贪婪搜索和束搜索。


## 练习

* 除了机器翻译，你还能想到seq2seq的哪些应用？

* 有哪些方法可以设计解码器的输出层？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4523)

![](../img/qr_seq2seq.svg)


## 参考文献

[1] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

[2] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
