# 双向循环神经网络

下面我们介绍双向循环神经网络的架构。

给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和隐藏层激活函数为$\phi$。在双向架构中，
设该时间步正向隐藏状态为$\overrightarrow{\boldsymbol{H}}_t  \in \mathbb{R}^{n \times h}$（正向隐藏单元个数为$h$），
反向隐藏状态为$\overleftarrow{\boldsymbol{H}}_t  \in \mathbb{R}^{n \times h}$（反向隐藏单元个数为$h$）。我们可以分别计算正向和反向隐藏状态：

$$
\begin{aligned}
\overrightarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(f)} + \overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{hh}^{(f)}  + \boldsymbol{b}_h^{(f)}),\\
\overleftarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(b)} + \overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{hh}^{(b)}  + \boldsymbol{b}_h^{(b)}),
\end{aligned}
$$

其中权重$\boldsymbol{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \boldsymbol{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h^{(f)} \in \mathbb{R}^{1 \times h}, \boldsymbol{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$均为模型参数。

双向循环神经网络在时间步$t$的隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times 2h}$即连结两个方向的隐藏状态$\overrightarrow{\boldsymbol{H}}_t$和$\overleftarrow{\boldsymbol{H}}_t$的结果。输出层只需基于连结后的隐藏状态计算输出$\boldsymbol{O}_t \in \mathbb{R}^{n \times q}$（输出个数为$q$）：

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{2h \times q}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$为输出层的模型参数。

双向循环神经网络架构如图6.12所示。和前面介绍的单向循环神经网络不同，给定一段时间序列，双向循环神经网络在每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入），并编码了整个序列的信息。

![双向循环神经网络架构。](../img/birnn.svg)


我们将在“自然语言处理”篇章中应用并实验双向循环神经网络。


## 小结

* 双向循环神经网络在每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入）。


## 练习

* 参考图6.11和图6.12，设计含多个隐藏层的双向循环神经网络。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6732)

![](../img/qr_bi-rnn.svg)
