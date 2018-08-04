# 深度循环神经网络


本章到目前为止介绍的循环神经网络只有一个单向的隐藏层：隐藏状态里的信息沿着时间步从早到晚依次传递。在实际中，我们有时会用到其他架构的循环神经网络。本节和下一节将分别介绍多隐藏层和双向架构。它们分别称作深度循环神经网络和双向循环神经网络。


给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）。在深度循环神经网络中，
设该时间步第$l$隐藏层的隐藏状态为$\boldsymbol{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$（隐藏单元个数为$h$），输出层变量为$\boldsymbol{O}_t \in \mathbb{R}^{n \times q}$（输出个数为$q$），隐藏层的激活函数为$\phi$。第一隐藏层的隐藏状态和之前的计算一样：

$$\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)}  + \boldsymbol{b}_h^{(1)}),$$


其中权重$\boldsymbol{W}_{xh}^{(1)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh}^{(1)} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h^{(1)} \in \mathbb{R}^{1 \times h}$分别为第一隐藏层的模型参数。

假设隐藏层个数为$L$，当$1 < l \leq L$时，第$l$隐藏层的隐藏状态的表达式为

$$\boldsymbol{H}_t^{(l)} = \phi(\boldsymbol{H}_t^{(l-1)} \boldsymbol{W}_{xh}^{(l)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(l)}  + \boldsymbol{b}_h^{(l)}),$$


其中权重$\boldsymbol{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}, \boldsymbol{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$分别为第$l$隐藏层的模型参数。

最终，输出层的输出只需基于第$L$隐藏层的隐藏状态：

$$\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times q}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$为输出层的模型参数。

深度循环神经网络的架构如图6.11所示。隐藏状态的信息不断传递至当前层的下一时间步和当前时间步的下一层。

![深度循环神经网络的架构。](../img/deep-rnn.svg)


我们将在本章最后一节用Gluon实验深度循环神经网络。


## 小结

* 在深度循环神经网络中，隐藏状态的信息不断传递至当前层的下一时间步和当前时间步的下一层。


## 练习

* 将[“循环神经网络”](rnn.md)一节中的模型改为含有2个隐藏层的循环神经网络。观察并分析实验现象。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6730)

![](../img/qr_deep-rnn.svg)
