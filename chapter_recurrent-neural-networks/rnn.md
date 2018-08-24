# 循环神经网络

上一节介绍的$n$元语法中，时间步$t$的词$w_t$基于前面所有词的条件概率只考虑了最近$n-1$个词。如果要考虑比$t-(n-1)$更早时间步的词对$w_t$可能的影响，我们需要增大$n$，但这样模型参数的数量将随之呈指数级增长（可参考上一节的练习）。

本节我们介绍循环神经网络，它不是刚性地记住所有固定长度的序列，而是通过隐藏状态来储存前面时间的信息。首先我们回忆下前面介绍过的多层感知机，然后介绍如何加入隐藏状态来将它变成循环神经网络。

## 不含隐藏状态的神经网络

让我们考虑一个单隐藏层的多层感知机。给定样本数为$n$、输入个数（特征数或特征向量维度）为$d$的小批量数据样本$\boldsymbol{X} \in \mathbb{R}^{n \times d}$。设隐藏层的激活函数为$\phi$，那么隐藏层的输出$\boldsymbol{H} \in \mathbb{R}^{n \times h}$计算为

$$\boldsymbol{H} = \phi(\boldsymbol{X} \boldsymbol{W}_{xh} + \boldsymbol{b}_h),$$

其中权重参数$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$，偏差参数 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，$h$为隐藏单元个数。上式相加的两项形状不同，因此将按照广播机制相加（参见[“数据操作”](../chapter_prerequisite/ndarray.md)一节）。跟[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节不同在于我们在下标中加入了$x$。

将把隐藏变量$\boldsymbol{H}$作为输出层的输入，且设输出个数为$q$（例如分类问题中的类别数），输出层的输出为

$$\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中输出变量$\boldsymbol{O} \in \mathbb{R}^{n \times q}$, 输出层权重参数$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times q}$, 输出层偏差参数$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$。如果是分类问题，我们可以使用$\text{softmax}(\boldsymbol{O})$来计算输出类别的概率分布。


## 含隐藏状态的循环神经网络

现在我们考虑输入数据是有时间相关性的情况。假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$是序列中时间步$t$的小批量输入，$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$该时间步的隐藏层变量。跟多层感知机不同在于这里我们保存上一时间步的隐藏变量$\boldsymbol{H}_{t-1}$，并引入一个新的权重参数$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$，它用来描述在当前时间步如何使用上一时间步的隐藏变量。具体来说，当前隐藏变量的计算由当前输入和上一时间步的隐藏状态共同决定：

$$\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h),$$

对比多层感知机，这里我们引入了$\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$来包含来自前面时间的信息。这里隐藏变量捕捉了截至当前时间步的序列历史信息，就像是神经网络当前时间步的状态或记忆一样，因此也称之为隐藏状态。

在时间步$t$输出层输出和多层感知机中的计算类似：

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hy} + \boldsymbol{b}_y.$$

如果输入序列有$T$个时间步，我们会在计算开始前先将隐藏状态全部元素初始化为0，然后依次计算$\boldsymbol{H}_t$和$\boldsymbol{O}_t$，$t=1,\ldots,T$。因为神经网络下一时间步的隐藏状态的输出既取决于下一时间步的输入，又取决于当前时间步的隐藏状态我。我们将此类神经网络称作循环神经网络。

循环神经网络的参数包括隐藏层的权重$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times q}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$。值得一提的是，即便在不同时间步，循环神经网络始终使用这些模型参数。因此，循环神经网络模型参数的数量不随历史增长而增长。

图6.1展示了循环神经网络在三个时间步的计算逻辑。在时间步$t$，隐藏状态的计算可以看成是将输入$\boldsymbol{X}_t$和前一时间步隐藏状态$\boldsymbol{H}_{t-1}$合并后输入一个激活函数为$\phi$的全连接层。该全连接层的输出就是当前时间步的隐藏状态$\boldsymbol{H}_t$，且模型参数为$\boldsymbol{W}_{xh}$与$\boldsymbol{W}_{hh}$的合并，偏差为$\boldsymbol{b}_h$。当前时间步的隐藏状态将参与下一个时间步的隐藏状态的计算，并输入到当前时间步的全连接输出层。

![含隐藏状态的循环神经网络。](../img/rnn.svg)


## 基于字符级循环神经网络的语言模型

最后我们介绍如何使用循环神经网络来构建一个语言模型。设小批量中样本数为1，文本序列为“想”、“要”、“有”、“直”、“升”、“机”，图6.2演示了如何使用循环神经网络来给定当前字符预测下一个字符。在训练时，我们对每个时间步的输出作用Softmax，然后使用交叉熵损失函数来计算它与标签的误差。

![基于字符级循环神经网络的语言模型。输入序列和标签序列分别为“想”、“要”、“有”、“直”、“升”和“要”、“有”、“直”、“升”、“机”。](../img/rnn-train.svg)

因为每个输入词是一个字符，因此这个模型被称为字符级循环神经网络（character-level recurrent neural network）。因为不同字符的个数远小于不同词的个数（对于英文尤其如此），所以字符级循环神经网络通常计算更加简单。接下来的三节里我们将介绍它的具体实现。

## 小结

* 循环神经网络通过引入隐藏状态来捕捉时间序列的历史信息。
* 循环神经网络模型参数的数量不随历史增长而增长。
* 可以基于字符级循环神经网络来创建语言模型。

## 练习

* 如果我们使用循环神经网络来预测一段文本序列的下一个词，输出个数应该是多少？
* 为什么循环神经网络可以表达某时间步的词基于文本序列中所有过去的词的条件概率？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6669)

![](../img/qr_hidden-state.svg)
