# 双向循环神经网络

下面我们介绍双向循环神经网络的架构。

给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times x}$（样本数为$n$，输入个数为$x$）。在多隐藏层结构中，
设该时间步第$l$隐藏层的隐藏状态为$\boldsymbol{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$（隐藏单元个数为$h$），输出层变量为$\boldsymbol{O}_t \in \mathbb{R}^{n \times y}$（输出个数为$y$），隐藏层的激活函数为$\phi$。第一隐藏层的隐藏状态和之前的计算一样：

$$\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)}  + \boldsymbol{b}_h^{(1)}),$$


其中权重$\boldsymbol{W}_{xh}^{(1)} \in \mathbb{R}^{x \times h}, \boldsymbol{W}_{hh}^{(1)} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h^{(1)} \in \mathbb{R}^{1 \times h}$分别为第一隐藏层的模型参数。

假设隐藏层个数为$L$，当$1 < l \leq L$时，第$l$隐藏层的隐藏状态的表达式为

$$\boldsymbol{H}_t^{(l)} = \phi(\boldsymbol{H}_t^{(l-1)} \boldsymbol{W}_{xh}^{(l)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(l)}  + \boldsymbol{b}_h^{(l)}),$$


其中权重$\boldsymbol{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}, \boldsymbol{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$分别为第$l$隐藏层的模型参数。

最终，输出层的输出只需基于第$L$隐藏层的隐藏状态：

$$\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times y}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times y}$为输出层的模型参数。

双向循环神经网络结构如图6.4所示。我们将在“自然语言处理”篇章中实验双向循环神经网络。



![双向循环神经网络结构。这里省略了输出层。](../img/bi-rnn.svg)




## 小结




## 练习



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6732)

![](../img/qr_bi-rnn.svg)
