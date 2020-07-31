# 正向传播、反向传播和计算图
:label:`sec_backprop`

到目前为止，我们已经对我们的模型进行了微型随机梯度下降的培训。但是，当我们实现算法时，我们只担心通过模型 * 向前传播 * 所涉及的计算。当计算梯度时，我们只是调用深度学习框架提供的反向传播函数。

梯度的自动计算（自动差异）极大地简化了深度学习算法的实现。在自动分化之前，即使对复杂模型的微小变化也需要手动重新计算复杂的衍生物。令人惊讶的是，学术论文不得不分配大量页面来推导更新规则。虽然我们必须继续依靠自动分化，以便我们可以专注于有趣的部分，但如果你想超越深度学习的浅层理解，你应该知道这些梯度是如何计算的。

在本节中，我们深入探讨 * 向后传播 *（通常称为 * 反向传播 *）的详细信息。为了传达对这些技术及其实现的一些洞察力，我们依靠一些基本的数学和计算图。首先，我们将重点放在一个具有权重衰减的单层 MLP（$L_2$ 正则化）上。

## 正向传播

*正向传播 *（或 * 转发通过 *）是指计算和存储
神经网络的中间变量（包括输出），按从输入图层到输出图层的顺序进行。我们现在逐步完成一个具有隐藏层的神经网络的机制。这可能看起来很乏味，但在放克演奏家詹姆斯·布朗永恒的话，你必须 “支付成本成为老板”。

为了简单起见，让我们假设输入样本是 $\mathbf{x}\in \mathbb{R}^d$，我们的隐藏层不包含偏差项。这里的中间变量是：

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

其中 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ 是隐藏层的权重参数。通过激活函数 $\phi$ 运行中间变量 $\mathbf{z}\in \mathbb{R}^h$ 之后，我们得到了长度为 $h$ 的隐藏激活向量，

$$\mathbf{h}= \phi (\mathbf{z}).$$

隐藏的变量 $\mathbf{h}$ 也是一个中间变量。假设输出图层的参数只具有 $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ 的权重，我们可以获得长度为 $q$ 的向量的输出图层变量：

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

假设损失函数为 $l$，样本标签为 $y$，然后我们可以计算单个数据样本的损失项，

$$L = l(\mathbf{o}, y).$$

根据 $L_2$ 正则化的定义, 给定超参数 $\lambda$, 正则化术语为

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

其中矩阵的 Frobenius 范数只是将矩阵平整为矢量后应用的 $L_2$ 范数。最后，模型在给定数据样本中的正则化损失是：

$$J = L + s.$$

在以下讨论中，我们将 $J$ 称为目标功能 *。

## 正向传播的计算图

绘制 * 计算图 * 有助于我们可视化计算中运算符和变量的依赖关系。:numref:`fig_forward` 包含与上述简单网络关联的图形，其中正方形表示变量，圆表示运算符。左下角表示输入，右上角表示输出。请注意，箭头的方向（说明数据流）主要是向右和向上。

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## 反向传播

*反向传播 * 是指计算方法
神经网络参数的梯度。简而言之，该方法根据演算中的 * 链规则 *，以相反的顺序遍历网络，从输出到输入图层。该算法存储计算梯度相对于某些参数计算梯度时所需的任何中间变量（部分导数）。假设我们有函数 $\mathsf{Y}=f(\mathsf{X})$ 和 $\mathsf{Z}=g(\mathsf{Y})$，其中输入和输出 $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ 是任意形状的张量。通过使用链条规则，我们可以通过

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

在这里，我们使用 $\text{prod}$ 运算符在执行必要的操作（如移位和交换输入位置）之后将其参数相乘。对于向量来说，这很简单：它只是矩阵矩阵乘法。对于较高维度张量，我们使用适当的对应部件。运算符 $\text{prod}$ 隐藏了所有符号开销。

回想一下，简单网络的参数与一个隐藏层，其计算图是 :numref:`fig_forward`，是 $\mathbf{W}^{(1)}$ 和 $\mathbf{W}^{(2)}$。反向传播的目的是计算梯度 $\partial J/\partial \mathbf{W}^{(1)}$ 和 $\partial J/\partial \mathbf{W}^{(2)}$.为此，我们应用链规则，然后计算每个中间变量和参数的梯度。计算的顺序相对于在正向传播中执行的顺序是逆转的，因为我们需要从计算图的结果开始，然后努力实现参数。第一步是计算目标函数 $J=L+s$ 相对于损失项 $L$ 和正则化项 $s$ 的梯度。

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

接下来，我们根据链规则计算目标函数相对于输出图层 $\mathbf{o}$ 变量的梯度：

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

接下来，我们计算两个参数的正则化项的渐变：

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

现在我们可以计算离输出图层最近的模型参数的梯度 $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$。使用链规则产生：

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

为了获得相对于 $\mathbf{W}^{(1)}$ 的梯度，我们需要继续沿输出图层向隐藏层图层反向传播。相对于隐藏图层输出 $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ 的梯度由

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

由于激活函数 $\phi$ 按元素适用，因此计算中间变量 $\mathbf{z}$ 的梯度 $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ 要求我们使用元素乘法运算符，我们用 $\odot$ 表示：

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

最后，我们可以获得最接近输入层的模型参数的梯度 $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$。根据链条规则，我们得到

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## 训练神经网络

训练神经网络时，正向和后向传播相互依赖。特别是，对于正向传播，我们沿依赖关系的方向遍历计算图，并计算其路径上的所有变量。然后将这些数据用于反向传播，其中图上的计算顺序反向传播。

以上述简单网络为样本说明。一方面，计算正向传播期间正则化项 :eqref:`eq_forward-s` 取决于模型参数 $\mathbf{W}^{(1)}$ 和 $\mathbf{W}^{(2)}$ 的电流值。它们是由优化算法根据最新迭代中的反向传播给出的。另一方面，反向传播期间参数 `eq_backprop-J-h` 的梯度计算取决于隐藏变量 $\mathbf{h}$ 的当前值，这是通过正向传播给出的。

因此，在训练神经网络时，在初始化模型参数之后，我们将向前传播与反向传播交替，使用反向传播给出的梯度来更新模型参数。请注意，反向传播重复使用正向传播中存储的中间值，以避免重复计算。其中一个后果是我们需要保留中间值，直到反向传播完成。这也是训练需要比普通预测显著更多的内存的原因之一。此外，此类中间值的大小与网络图层的数量和批量大致成比例。因此，使用较大的批量大小训练更深层次的网络会更容易导致 * 内存不足 * 错误。

## 摘要

* 正向传播按顺序计算和存储在由神经网络定义的计算图中的中间变量。它从输入到输出图层。
* 反向传播按顺序计算和存储神经网络中的中间变量和参数的梯度。
* 训练深度学习模型时，前向传播和后向传播是相互依赖的。
* 训练需要比预测更多的内存。

## 练习

1. 假设某些标量函数 $f$ 的输入是 $n \times m$ 矩阵。就 $\mathbf{X}$ 而言，$f$ 的梯度是什么维数？
1. 向本节所述模型的隐藏层添加偏差。
    * 绘制相应的计算图。
    * 推导正向和后向传播方程。
1. 在本节中描述的模型中计算训练和预测的内存占用量。
1. 假设你想计算二衍生物。计算图会发生什么？您预计计算需要多长时间？
1. 假设计算图对于 GPU 来说太大。
    * 你可以将它分区到多个 GPU 吗？
    * 在较小的微型批次上培训有什么优点和缺点？

[Discussions](https://discuss.d2l.ai/t/102)
