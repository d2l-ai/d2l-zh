# 线性回归——从零开始

在了解了线性回归的背景知识之后，现在我们可以动手实现它了。
尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，我们就会很难深入理解深度学习是如何工作的。因此，本节将介绍如何只利用NDArray和`autograd`来实现一个线性回归的训练。


## 线性回归

让我们先回忆一下上节中的内容。设数据样本数为$n$，特征数为$d$。给定批量数据样本的特征$\boldsymbol{X} \in \mathbb{R}^{n \times d}$和标签$\boldsymbol{y} \in \mathbb{R}^{n \times 1}$，线性回归的批量输出$\boldsymbol{\hat{y}} \in \mathbb{R}^{n \times 1}$的计算表达式为

$$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b,$$

其中$\boldsymbol{w} \in \mathbb{R}^{d \times 1}$和$b \in \mathbb{R}$分别为线性回归的模型参数：权重和偏差。为了学习权重和偏差，我们用预测值$\boldsymbol{\hat{y}}$和真实值$\boldsymbol{y}$之间的平方损失作为模型的损失函数。在模型训练过程中，我们使用小批量随机梯度下降不断迭代模型参数的值，以最小化损失函数。最终，在有限次迭代后，我们便学出了模型参数的值。

下面我们开始动手实现线性回归的训练。首先，导入本节中实验所需的包。

```{.python .input  n=1}
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, nd
import numpy as np
import random
```

在本书中，我们会将重复使用的函数定义在[utils.py](../utils.py)中。对于里面大部分较重要的函数，我们会在第一次使用时描述它是如何实现的，例如本节中的`sgd`函数。

```{.python .input}
import sys
sys.path.append('..')
import utils
```

## 生成数据集

我们在这里描述用来生成人工训练数据集的真实模型。

设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征$\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$，我们使用线性回归模型真实权重$\boldsymbol{w} = [2, -3.4]^\top$和偏差$b = 4.2$，以及一个随机噪音项$\epsilon$来生成标签

$$\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon,$$

其中噪音项$\epsilon$服从均值为0和标准差为0.01的正态分布。下面，让我们生成数据集。

```{.python .input  n=2}
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

注意到`features`的每一行是一个长度为2的向量，而`labels`的每一行是一个长度为1的向量（标量）。

```{.python .input  n=3}
print(features[0], labels[0])
```

通过生成第二个特征`features[:, 1]`和标签 `labels` 的散点图，我们可以更直观地观察两者间的线性关系。

```{.python .input  n=4}
utils.set_fig_size(mpl)
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()
```

## 读取数据

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回`batch_size`个随机样本的特征和标签。设批量大小（`batch_size`）为10。

```{.python .input  n=5}
batch_size = 10
def data_iter(): 
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
```

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为`（10, 2）`，分别对应批量大小`batch_size`和输入个数`num_inputs`；标签形状为`10`，也就是批量大小。

```{.python .input  n=6}
for X, y in data_iter():
    print(X, y)
    break
```

## 初始化模型参数

下面我们随机初始化模型参数。

```{.python .input  n=7}
w = nd.random.normal(scale=1, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w, b]
```

之后训练时我们需要对这些参数求梯度来迭代它们的值，以使损失函数不断减小。因此我们需要创建它们的梯度。

```{.python .input  n=8}
for param in params:
    param.attach_grad()
```

## 定义模型

下面是线性回归的矢量计算表达式的实现。我们使用`nd.dot`函数做矩阵乘法。

```{.python .input  n=9}
def net(X, w, b): 
    return nd.dot(X, w) + b 
```

## 定义损失函数

我们使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，我们需要把真实值`y`变形成预测值`y_hat`的形状。以下函数返回的结果也将和`y_hat`的形状相同。

```{.python .input  n=10}
def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

## 定义优化算法

以下的`sgd`函数实现了上一节中介绍的小批量随机梯度下降算法。这是我们最小化损失函数所需要的优化算法。

```{.python .input  n=11}
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

## 训练模型

现在我们可以开始训练模型了。在训练中，我们将有限次地迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征`features`和标签`label`），通过调用反向函数`backward`计算小批量随机梯度，并调用优化算法`sgd`迭代模型参数。在一个迭代周期（epoch）中，我们将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次。这里的迭代周期数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都是需要通过反复试错来不断调节。当迭代周期数设的越大时，虽然模型可能更有效，但是训练时间可能过长。而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。

```{.python .input  n=12}
lr = 0.03
num_epochs = 3
loss = squared_loss

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter():
        with autograd.record():
            y_hat = net(X, w, b)
            l = loss(y_hat, y)
        l.backward()
        sgd([w, b], lr, batch_size)
    print("epoch %d, loss: %f"
          % (epoch, loss(net(features, w, b), labels).mean().asnumpy()))
```

训练完成后，我们可以比较学到的参数和真实参数。它们应该很接近。

```{.python .input  n=13}
true_w, w
```

```{.python .input  n=14}
true_b, b
```

## 小结

* 我们现在看到，仅使用NDArray和`autograd`就可以很容易地实现一个模型。在接下来的章节中，我们会在此基础上描述更多深度学习模型，并介绍怎样使用更简洁的代码（例如下一节）实现它们。


## 练习

* 尝试用不同的学习率查看损失函数值的下降速度。

* 回顾[“自动求梯度”](../chapter_crashcourse/autograd.md)一节。本节代码中变量`l`并不是一个标量，运行`l.backward()`将如何对模型参数求梯度？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/743)

![](../img/qr_linear-regression-scratch.svg)
