# 多层感知机

我们已经介绍了包括线性回归和Softmax回归在内的单层神经网络。然而深度学习主要关注多层模型。本节中，我们将以多层感知机（multilayer perceptron，简称MLP）为例，介绍多层神经网络的概念。

## 隐藏层

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。图3.3展示了一个多层感知机的神经网络图。

![带有隐藏层的多层感知机。它含有一个隐藏层，该层中有5个隐藏单元。](../img/mlp.svg)

在图3.3的多层感知机中，输入和输出个数分别为4和3，中间的隐藏层中包含了5个隐藏单元（hidden unit）。由于输入层不涉及计算，图3.3中的多层感知机的层数为2。由图3.3可见，隐藏层中的神经元和输入层中各个输入完全连接，输出层中的神经元和隐藏层中的各个神经元也完全连接。因此，多层感知机中的隐藏层和输出层都是全连接层。


具体来说，给定一个小批量样本$\boldsymbol{X} \in \mathbb{R}^{n \times d}$，其批量大小为$n$，输入个数为$d$。假设多层感知机只有一个隐藏层，其中隐藏单元个数为$h$。记隐藏层的输出（也称为隐藏层变量或隐藏变量）为$\boldsymbol{H}$，我们有$\boldsymbol{H} \in \mathbb{R}^{n \times h}$。因为隐藏层和输出层均是全连接层，我们知道隐藏层的权重参数和偏差参数分别为$\boldsymbol{W}_h \in \mathbb{R}^{d \times h}$和 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times q}$和$\boldsymbol{b}_o \in \mathbb{R}^{1 \times q}$。

我们先来看一个简单的输出$\boldsymbol{O}$的计算方法：

$$
\begin{aligned}
\boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}
$$

也就是我们将这两个全连接层放置在一起，隐藏全连接层的输入直接进入输出全连接层。但如果我们将两个式子联立起来，就会发现

$$
\boldsymbol{O} = (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o
$$

这样等价与我们创建一个单层神经网络，它的输出层权重参数是$\boldsymbol{W}_h\boldsymbol{W}_o$，且偏差参数为$\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$。这样，不管使用多少隐藏层，其效果等同于只有输出层的单层神经网络。


## 激活函数

上述问题的根源在于全连接层只是对数据做线性变换（准确叫仿射变换（affine transformation））。但多个线性变换的叠加仍然是一个线性变化。解决问题的一个方法是引入非线性变换，例如对隐藏变量先作用一个按元素的非线性函数后再输入到后面的层中。这个非线性函数被叫做激活函数（activation function）。下面我们介绍几个常见的激活函数。

### ReLU函数

ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素$x$，该函数定义为

$$\text{relu}(x) = \max(x, 0).$$

可以看出，ReLU函数只保留正数元素，并将负数元素清零。为了直观地观察这一非线性变换，我们先定义一个绘图函数`xyplot`。

```{.python .input  n=6}
%matplotlib inline
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd

def xyplot(x_vals, y_vals, name):
    gb.set_figsize()
    gb.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    gb.plt.xlabel('x')
    gb.plt.ylabel(name+'(x)')
```

我们接下来通过NDArray提供的`relu`函数来绘制ReLU函数，可以看到它是一个两段线性函数。

```{.python .input  n=7}
x = nd.arange(-5.0, 5.0, 0.1)
xyplot(x, x.relu(), 'relu')
```

### Sigmoid函数

Sigmoid函数可以将元素的值变换到0和1之间：

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

早期神经网络使用较多，但目前渐渐被更简单的ReLU取代。在后面“循环神经网络”一章中我们会介绍如何利用它值域在0到1之间这一特性来控制信息在神经网络中的流动。下面绘制了sigmoid函数。当元素值接近0时，sigmoid函数接近线性变换。

```{.python .input  n=8}
xyplot(x, x.sigmoid(), 'sigmoid')
```

### Tanh函数

Tanh（双曲正切）函数可以将元素的值变换到-1和1之间：

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

我们接着绘制tanh函数。当元素值接近0时，tanh函数接近线性变换。值得一提的是，它的形状和sigmoid函数很像，但tanh函数在y轴上对称。

```{.python .input  n=9}
xyplot(x, x.tanh(), 'tanh')
```

## 多层感知机

多层感知机就是含有一或者多个隐藏层的由全连接层组成的网络，且在每个隐藏层的输出上作用激活函数。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。单隐藏层情况下我们如下计算输出：

$$
\begin{aligned}
\boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}
$$

这里$\phi$表示按元素的激活函数。在分类问题中，我们可以对输出$\boldsymbol{O}$做Softmax运算，并使用Softmax回归中的交叉熵损失函数。
在回归问题中，我们将输出层的输出个数设为1，并将输出$\boldsymbol{O}$直接提供给线性回归中使用的平方损失函数。



## 小结

* 多层感知机在输出层与输入层之间加入了一个或多个全连接隐藏层，且层之间加入非线性激活函数。
* 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。


## 练习

* 查一查还有哪些其他的激活函数（可以去维基百科查找）。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6447)

![](../img/qr_mlp.svg)
