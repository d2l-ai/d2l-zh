# 隐藏状态

上一节介绍的$n$元语法中，时刻$t$的词$w_t$基于文本序列最近$n-1$个词的条件概率为

$$\mathbb{P}(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}).$$

需要注意的是，以上概率并没有考虑到比$t-(n-1)$更早时刻的词对$w_t$可能的影响。然而，考虑这些影响需要增大$n$的值，那么$n$元语法的模型参数的数量将随之呈指数级增长（可参考上一节的练习）。为了解决$n$元语法的局限性，我们可以在神经网络中引入隐藏状态。隐藏状态既可以捕捉时间序列的历史信息，且模型参数的数量不随着历史而增长。




## 不含隐藏状态的神经网络


让我们先回顾一下不含隐藏状态的神经网络，例如只有一个隐藏层的多层感知机。

给定样本数为$n$、输入个数（特征数或特征向量维度）为$x$的小批量数据样本$\boldsymbol{X} \in \mathbb{R}^{n \times x}$。设隐藏层的激活函数为$\phi$，那么隐藏层的输出$\boldsymbol{H} \in \mathbb{R}^{n \times h}$计算为

$$\boldsymbol{H} = \phi(\boldsymbol{X} \boldsymbol{W}_{xh} + \boldsymbol{b}_h),$$

其中权重参数$\boldsymbol{W}_{xh} \in \mathbb{R}^{x \times h}$，偏差参数 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，$h$为隐藏单元个数。我们之前也提到，上式的两项相加使用了广播机制。把隐藏变量$\boldsymbol{H}$作为输出层的输入，且设输出个数为$y$（例如分类问题中的类别数），输出层的输出

$$\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中输出变量$\boldsymbol{O} \in \mathbb{R}^{n \times y}$, 输出层权重参数$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times y}$, 输出层偏差参数$\boldsymbol{b}_y \in \mathbb{R}^{1 \times y}$。如果是[分类问题](../chapter_supervised-learning/classification.md)，我们可以使用$\text{softmax}(\boldsymbol{O})$来计算输出类别的概率分布。



## 含隐藏状态的循环神经网络


将上面网络改成循环神经网络，我们首先对输入输出加上时间戳$t$。假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times x}$是序列中的第$t$个批量输入（样本数为$n$，每个样本的特征向量维度为$x$），对应的隐含层输出是隐含状态$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$（隐含层长度为$h$），而对应的最终输出是$\hat{\boldsymbol{Y}}_t \in \mathbb{R}^{n \times y}$（每个样本对应的输出向量维度为$y$）。在计算隐含层的输出的时候，循环神经网络只需要在前馈神经网络基础上加上跟前一时间$t-1$输入隐含层$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$的加权和。为此，我们引入一个新的可学习的权重$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$：

$$\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h)$$

输出的计算跟前面一致：

$$\hat{\boldsymbol{Y}}_t = \text{softmax}(\boldsymbol{H}_t \boldsymbol{W}_{hy}  + \boldsymbol{b}_y)$$

一开始我们提到过，隐含状态可以认为是这个网络的记忆。该网络中，时刻$t$的隐含状态就是该时刻的隐含层变量$\boldsymbol{H}_t$。它存储前面时间里面的信息。我们的输出是只基于这个状态。最开始的隐含状态里的元素通常会被初始化为0。


## 小结

* 语言模型是自然语言处理的重要技术。
* $N$元语法是基于$n-1$阶马尔可夫链的概率语言模型。但它有一定的局限性。


## 练习

* 假设训练数据集中有十万个词，四元语法需要存储多少词频和多词相邻频率？
* 你还能想到哪些语言模型的应用？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6669)

![](../img/qr_hidden-state.svg)
