# 双向循环神经网络

之间介绍的循环神经网络模型都是假设当前时间步是由前面时间步的序列决定，因此它们都是将信息通过隐藏状态从前面往后面传递。有时候，当前时间步也同后面时间步相关。例如，当我们写下一个句子时，可能会根据句子后面的词来修改句子前面的用词。双向循环神经网络通过增加从后面往前面传递信息的隐藏层来更好地处理这类信息。

图6.12演示了将一个有单隐藏层的循环神经网络改造成双向循环神经网络，它增加一个新的隐藏层，它在时间步$t$的隐藏状态为$\overleftarrow{\boldsymbol{H}}_t$，它由当前输入$\boldsymbol{X}_t$和下一时间步的隐藏状态$\overleftarrow{\boldsymbol{H}}_{t+1}$计算得来。之后我们合并两个隐藏状态输入到输出层。

![双向循环神经网络架构。](../img/birnn.svg)

下面我们来介绍具体的定义。
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

然后我们合并两个方向的隐藏状态$\overrightarrow{\boldsymbol{H}}_t$和$\overleftarrow{\boldsymbol{H}}_t$来得到隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times 2h}$，并将其输入到输出层。输出层计算输出$\boldsymbol{O}_t \in \mathbb{R}^{n \times q}$（输出个数为$q$）：

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hy} + \boldsymbol{b}_y,$$

其中权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{2h \times q}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$为输出层的模型参数。

当然，不同方向上的隐藏单元数可以不同。同时可以仿照上一节来将其拓展成深度双向循环神经网络。

## 小结

* 双向循环神经网络在每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入）。


## 练习

* 如果不同方向上使用不同的的隐藏单元数，$\boldsymbol{H}_t$的形状会发生怎样的改变？
* 参考图6.11和图6.12，设计含多个隐藏层的双向循环神经网络。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6732)

![](../img/qr_bi-rnn.svg)
