# 多层神经网络

我们已经介绍了包括线性回归和Softmax回归在内的单层神经网络。本节中，我们将以多层感知机（multilayer perceptron，简称MLP）为例，介绍多层神经网络的概念。

多层感知机是最基础的深度学习模型。


## 隐藏层

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。图3.3展示了一个多层感知机的神经网络图。

![带有隐藏层的多层感知机。它含有一个隐藏层，该层中有5个隐藏单元](../img/mlp.svg)

在图3.3的多层感知机中，输入和输出个数分别为4和3，中间的隐藏层中包含了5个隐藏单元（hidden unit）。由于输入层不涉及计算，图3.3中的多层感知机的层数为2。由图3.3可见，隐藏层中的神经元和输入层中各个输入完全连接，输出层中的神经元和隐藏层中的各个神经元也完全连接。因此，多层感知机中的隐藏层和输出层都是全连接层。


## 线性转换

在描述隐藏层的计算之前，让我们看看多层感知机输出层是怎样计算的。它的计算和之前介绍的单层神经网络的输出层的计算类似：只是输出层的输入变成了隐藏层的输出。我们通常将隐藏层的输出称为隐藏层变量或隐藏变量。

给定一个小批量样本，其批量大小为$n$，输入个数为$x$，输出个数为$y$。假设多层感知机只有一个隐藏层，其中隐藏单元个数为$h$，隐藏变量$\boldsymbol{H} \in \mathbb{R}^{n \times h}$。假设输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times y}, \boldsymbol{b}_o \in \mathbb{R}^{1 \times y}$，多层感知机输出

$$
\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
$$

其中的加法运算使用了广播机制，$\boldsymbol{O}  \in \mathbb{R}^{n \times y}$。可见，多层感知机的输出$\boldsymbol{O}$是对上一层的输出$\boldsymbol{H}$的线性转换。


那么，如果隐藏层也对输入做线性转换会怎么样呢？为了便于描述这一问题，让我们暂时忽略每一层的偏差参数。设批量特征为$\boldsymbol{X} \in \mathbb{R}^{n \times x}$，隐藏层的权重参数$\boldsymbol{W}_h \in \mathbb{R}^{x \times h}$。假设$\boldsymbol{H} = \boldsymbol{X} \boldsymbol{W}_h$且$\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_o$，联立两式可得$\boldsymbol{O} = \boldsymbol{X} \boldsymbol{W}_h \boldsymbol{W}_o$：它等价于$\boldsymbol{O} = \boldsymbol{X} \boldsymbol{W}^\prime$，其中$\boldsymbol{W}^\prime = \boldsymbol{W}_h \boldsymbol{W}_o$。因此，使用线性转换的隐藏层使多层感知机与前面介绍的单层神经网络没什么区别。

## 激活函数

由上面的例子可以看出，我们必须在隐藏层中添加非线性转换，这样才能使多层感知机变得有意义。我们将这些非线性转换称为激活函数（activation function）。激活函数能对任意形状的输入按元素操作且不改变输入的形状。以下列举了三种常用的激活函数。

### ReLU函数

ReLU（rectified linear unit）函数提供了一个很简单的非线性转换。给定元素$x$，该函数的输出是

$$\text{relu}(x) = \max(x, 0).$$

ReLU函数只保留正数元素，并将负数元素清零。为了直观地观察这一非线性转换，让我们先导入一些包或模块。

```{.python .input}
%matplotlib inline
import sys
sys.path.append('..')
import gluonbook as gb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mxnet import nd
```

下面，让我们绘制ReLU函数。当元素值非负时，ReLU函数实际上在做线性转换。

```{.python .input}
gb.set_fig_size(mpl, (5, 2.5))

x = nd.arange(-5.0, 5.0, 0.1)
plt.plot(x.asnumpy(), x.relu().asnumpy())
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.show()
```

### Sigmoid函数

Sigmoid函数可以将元素的值转换到0和1之间：

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

我们会在后面“循环神经网络”一章中介绍如何利用sigmoid函数值域在0到1之间这一特性来控制信息在神经网络中的流动。

下面绘制了sigmoid函数。当元素值接近0时，sigmoid函数接近线性转换。

```{.python .input}
plt.plot(x.asnumpy(), x.sigmoid().asnumpy())
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()
```

### Tanh函数

Tanh（双曲正切）函数可以将元素的值转换到-1和1之间：

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

下面绘制了tanh函数。当元素值接近0时，tanh函数接近线性转换。值得一提的是，它的形状和sigmoid函数很像，且当元素在实数域上均匀分布时，tanh函数值的均值为0。

```{.python .input}
plt.plot(x.asnumpy(), x.tanh().asnumpy())
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.show()
```

下面，我们使用三种激活函数来转换输入。按元素操作后，输入和输出形状相同。

```{.python .input}
X = nd.array([[[0,1], [-2,3], [4,-5]], [[6,-7], [8,-9], [10,-11]]])
X.relu(), X.sigmoid(), X.tanh()
```

## 多层感知机

现在，我们可以给出多层感知机的矢量计算表达式了。

给定一个小批量样本$\boldsymbol{X} \in \mathbb{R}^{n \times x}$，其批量大小为$n$，输入个数为$x$，输出个数为$y$。
假设多层感知机只有一个隐藏层，其中隐藏单元个数为$h$，激活函数为$\phi$。假设
隐藏层的权重和偏差参数分别为$\boldsymbol{W}_h \in \mathbb{R}^{x \times h}, \boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，
输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times y}, \boldsymbol{b}_o \in \mathbb{R}^{1 \times y}$。
多层感知机的矢量计算表达式为

$$
\boldsymbol{H} = \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\
\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
$$

其中的加法运算使用了广播机制，$\boldsymbol{H} \in \mathbb{R}^{n \times h}, \boldsymbol{O}  \in \mathbb{R}^{n \times y}$。
在分类问题中，我们可以对输出$\boldsymbol{O}$做Softmax运算，并使用Softmax回归中的交叉熵损失函数。
在回归问题中，我们将输出层的输出个数设为1，并将输出$\boldsymbol{O}$直接提供给线性回归中使用的平方损失函数。定义了损失函数后，我们使用优化算法迭代模型参数从而不断降低损失函数的值。

我们可以添加更多的隐藏层来构造更深的模型。需要指出的是，多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。


## 随机初始化模型参数

在神经网络中，我们需要随机初始化模型参数。以图3.3为例，假设隐藏层使用相同的激活函数。为了便于描述，假设输出层只保留一个输出单元$o_1$（删去$o_2, o_3$和指向它们的箭头）。如果初始化后每个隐藏单元的参数都相同，那么在模型训练时每个隐藏单元将根据相同输入计算出相同的值。接下来输出层也将从每个隐藏单元拿到完全一样的值。在迭代每个隐藏单元的参数时，这些参数在每轮迭代的值都相同（我们将在后面章节中介绍迭代细节）。因此，我们需要通过随机初始化模型参数避免让每个隐藏单元做相同计算。

## 小结

* 多层感知机本质上是对输入做一系列线性和非线性的转换。
* 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。
* 我们需要随机初始化神经网络的模型参数。


## 练习

* 有人说随机初始化模型参数时为了“打破对称性”。这里的“对称”应如何理解？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6447)

![](../img/qr_multi-layer.svg)
