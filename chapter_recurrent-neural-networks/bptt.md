# 通过时间反向传播

在上一章[循环神经网络](rnn-scratch.md)的示例代码中，如果不使用梯度裁剪，模型将无法正常训练。为了深刻理解这一现象，并激发改进循环神经网络的灵感，本节我们将介绍循环神经网络中模型梯度的计算和存储，也即**通过时间反向传播**（back-propagation through time）。


我们在[正向传播和反向传播](../chapter_supervised-learning/backprop.md)中以$L_2$范数[正则化](../chapter_supervised-learning/reg-scratch.md)的[多层感知机](../chapter_supervised-learning/mlp-scratch.md)为例，介绍了深度学习模型梯度的计算和存储。事实上，所谓通过时间反向传播只是反向传播在循环神经网络的具体应用。我们只需将循环神经网络按时间展开，从而得到模型变量和参数之间的依赖关系，并依据链式法则应用反向传播计算梯度。

为了解释通过时间反向传播，我们以一个简单的循环神经网络为例。


### 模型定义

给定一个输入为$\mathbf{x}_t \in \mathbb{R}^x$（每个样本输入向量长度为$x$）和对应真实值为$y_t \in \mathbb{R}$的时序数据训练样本（$t = 1, 2, \ldots, T$为时刻），不考虑偏差项，我们可以得到隐含层变量的表达式

$$\mathbf{h}_t = \phi(\mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1})$$

其中$\mathbf{h}_t \in \mathbb{R}^h$是向量长度为$h$的隐含层变量，$\mathbf{W}_{hx} \in \mathbb{R}^{h \times x}$和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是隐含层模型参数。使用隐含层变量和输出层模型参数$\mathbf{W}_{yh} \in \mathbb{R}^{y \times h}$，我们可以得到相应时刻的输出层变量$\mathbf{o}_t \in \mathbb{R}^y$。不考虑偏差项，

$$\mathbf{o}_t = \mathbf{W}_{yh} \mathbf{h}_{t}$$

给定每个时刻损失函数计算公式$\ell$，长度为$T$的整个时序数据的损失函数$L$定义为

$$L = \frac{1}{T} \sum_{t=1}^T \ell (\mathbf{o}_t, y_t)$$

这也是模型最终需要被优化的目标函数。

## 计算图

为了可视化模型变量和参数之间在计算中的依赖关系，我们可以绘制计算图。我们以时序长度$T=3$为例。

![](../img/rnn-bptt.svg)

## 梯度的计算与存储

在上图中，模型的参数是$\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$和$\mathbf{W}_{yh}$。为了在模型训练中学习这三个参数，以随机梯度下降为例，假设学习率为$\eta$，我们可以通过

$$\mathbf{W}_{hx} = \mathbf{W}_{hx} - \eta \frac{\partial L}{\partial \mathbf{W}_{hx}}$$

$$\mathbf{W}_{hh} = \mathbf{W}_{hh} - \eta \frac{\partial L}{\partial \mathbf{W}_{hh}}$$

$$\mathbf{W}_{yh} = \mathbf{W}_{yh} - \eta \frac{\partial L}{\partial \mathbf{W}_{yh}}$$


来不断迭代模型参数的值。因此我们需要模型参数梯度$\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$和$\partial L/\partial \mathbf{W}_{yh}$。为此，我们可以按照反向传播的次序依次计算并存储梯度。

为了表述方便，对输入输出$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$为任意形状张量的函数$\mathsf{Y}=f(\mathsf{X})$和$\mathsf{Z}=g(\mathsf{Y})$，我们使用

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}})$$

来表达链式法则。以下依次计算得到的梯度将依次被存储。

首先，目标函数有关各时刻输出层变量的梯度$\partial L/\partial \mathbf{o}_t \in \mathbb{R}^y$可以很容易地计算

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial \ell (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} $$

事实上，这时我们已经可以计算目标函数有关模型参数$\mathbf{W}_{yh}$的梯度$\partial L/\partial \mathbf{W}_{yh} \in \mathbb{R}^{y \times h}$。需要注意的是，在计算图中，
$\mathbf{W}_{yh}$可以经过$\mathbf{o}_1, \ldots, \mathbf{o}_T$通向$L$，依据链式法则，

$$\frac{\partial L}{\partial \mathbf{W}_{yh}} 
= \sum_{t=1}^T \text{prod}(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{yh}}) 
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top
$$


其次，我们注意到隐含层变量之间也有依赖关系。
对于最终时刻$T$，
在计算图中，
隐含层变量$\mathbf{h}_T$只经过$\mathbf{o}_T$通向$L$。因此我们先计算目标函数有关最终时刻隐含层变量的梯度$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$。依据链式法则，我们得到

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} ) = \mathbf{W}_{yh}^\top \frac{\partial L}{\partial \mathbf{o}_T}
$$


为了简化计算，我们假设激活函数$\phi(x) = x$。
接下来，对于时刻$t < T$，
在计算图中，
由于$\mathbf{h}_t$可以经过$\mathbf{h}_{t+1}$和$\mathbf{o}_t$通向$L$，依据链式法则，
目标函数有关隐含层变量的梯度$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$需要按照时刻从晚到早依次计算：


$$\frac{\partial L}{\partial \mathbf{h}_t} 
= \text{prod}(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} ) 
+ \text{prod}(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} ) 
= \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{yh}^\top \frac{\partial L}{\partial \mathbf{o}_t}
$$

将递归公式展开，对任意$1 \leq t \leq T$，我们可以得到目标函数有关隐含层变量梯度的通项公式

$$\frac{\partial L}{\partial \mathbf{h}_t} 
= \sum_{i=t}^T {(\mathbf{W}_{hh}^\top)}^{T-i} \mathbf{W}_{yh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}
$$

由此可见，当每个时序训练数据样本的时序长度$T$较大或者时刻$t$较小，目标函数有关隐含层变量梯度较容易出现**衰减**（valishing）和**爆炸**（explosion）。想象一下$2^{30}$和$0.5^{30}$会有多大。


有了各时刻隐含层变量的梯度之后，我们可以计算隐含层中模型参数的梯度$\partial L/\partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times x}$和$\partial L/\partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$。在计算图中，它们都可以经过$\mathbf{h}_1, \ldots, \mathbf{h}_T$通向$L$。依据链式法则，我们有

$$\frac{\partial L}{\partial \mathbf{W}_{hx}} 
= \sum_{t=1}^T \text{prod}(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}) 
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top
$$

$$\frac{\partial L}{\partial \mathbf{W}_{hh}} 
= \sum_{t=1}^T \text{prod}(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}) 
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top
$$


在[正向传播和反向传播](../chapter_supervised-learning/backprop.md)中我们解释过，每次迭代中，上述各个依次计算出的梯度会被依次存储或更新。这是为了避免重复计算。例如，由于输出层变量梯度$\partial L/\partial \mathbf{h}_t$被计算存储，反向传播稍后的参数梯度$\partial L/\partial  \mathbf{W}_{hx}$和隐含层变量梯度$\partial L/\partial \mathbf{W}_{hh}$的计算可以直接读取输出层变量梯度的值，而无需重复计算。

还有需要注意的是，反向传播对于各层中变量和参数的梯度计算可能会依赖通过正向传播计算出的各层变量和参数的当前值。举例来说，参数梯度$\partial L/\partial \mathbf{W}_{hh}$的计算需要依赖隐含层变量在时刻$t = 1, \ldots, T-1$的当前值$\mathbf{h}_t$（$\mathbf{h}_0$是初始化得到的）。这个当前值是通过从输入层到输出层的正向传播计算并存储得到的。


## 总结

* 所谓通过时间反向传播只是反向传播在循环神经网络的具体应用。
* 当每个时序训练数据样本的时序长度$T$较大或者时刻$t$较小，目标函数有关隐含层变量梯度较容易出现衰减和爆炸。


## 练习

- 在循环神经网络中，梯度裁剪是否对梯度衰减和爆炸都有效？
- 你还能想到别的什么方法可以应对循环神经网络中的梯度衰减和爆炸现象？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/3711)
