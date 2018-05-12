# 线性回归——使用Gluon

随着深度学习框架的发展，开发深度学习应用变得越来越便利。实践中，我们通常可以用比上一节中更简洁的代码来实现相同模型。本节中，我们将介绍如何使用MXNet提供的Gluon接口更方便地实现线性回归的训练。

首先，导入本节中实验所需的一部分包。我们在之前的章节里使用过它们。

```{.python .input}
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
import mxnet as mx
from mxnet import autograd, nd
import numpy as np
```

## 生成数据集

我们生成与上一节中相同的数据集。其中`X`是训练数据特征，`y`是标签。

```{.python .input  n=2}
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

## 读取数据

这里，我们使用Gluon提供的`data`模块来读取数据。在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

和上一节一样，让我们读取并打印第一个小批量数据样本。

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## 定义模型

在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。当模型结构变得更复杂时，这些步骤将变得更加繁琐。其实，Gluon提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。下面将介绍如何使用Gluon更简洁地定义线性回归。

首先，导入`nn`模块。我们先定义一个模型变量`net`，它是一个Sequential实例。在Gluon中，Sequential实例可以看做是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。

```{.python .input  n=5}
from mxnet.gluon import nn

net = nn.Sequential()
```

回顾图3.1中线性回归在神经网络图中的表示。作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1。

```{.python .input  n=6}
net.add(nn.Dense(1))
```

值得一提的是，在Gluon中我们无需指定每一层输入的形状，例如线性回归的输入个数。当模型看见数据时，例如后面执行`net(X)`时，模型将自动推断出每一层的输入个数。我们将在之后“深度学习计算基础”一章详细介绍这个机制。Gluon的这一设计为模型开发带来便利。


## 初始化模型参数

在使用`net`前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。这里我们使用默认的随机初始化方法。

```{.python .input  n=7}
net.initialize()
```

## 定义损失函数

我们从Gluon中导入`loss`模块，并直接使用它所提供的平方损失作为模型的损失函数。

```{.python .input  n=8}
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()
```

## 定义优化算法

同样，我们也无需实现小批量随机梯度下降。在导入Gluon后，我们可以创建一个Trainer实例，并且将模型参数传递给它。下面定义了学习率为0.03的小批量随机梯度下降。

```{.python .input  n=9}
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## 训练模型

和上一节不同，我们通过调用`step`函数来迭代模型参数。由于变量`l`是`batch_size`维的NDArray，执行`l.backward()`等价于`l.sum().backward()`。按照小批量随机梯度下降的定义，我们在`step`函数中提供`batch_size`，以确保小批量随机梯度是该批量中每个样本梯度的平均。

```{.python .input  n=10}
num_epochs = 3
for epoch in range(1, num_epochs + 1): 
    for X, y in data_iter:
        with autograd.record():
            output = net(X)
            l = loss(output, y)
        l.backward()
        trainer.step(batch_size)
    print("epoch %d, loss: %f" 
          % (epoch, loss(net(features), labels).mean().asnumpy()))
```

下面我们分别比较学到的和真实的模型参数。我们从`net`获得需要的层，并访问其权重和位移。学到的和真实的参数很接近。

```{.python .input  n=12}
dense = net[0]
true_w, dense.weight.data()
```

```{.python .input  n=13}
true_b, dense.bias.data()
```

## 小结

* 使用Gluon可以更简洁地实现模型。


## 练习

* 如果将`l = loss(output, y)`替换成`l = loss(output, y).mean()`，我们需要将`trainer.step(batch_size)`相应地改成`trainer.step(1)`。这是为什么呢？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/742)

![](../img/qr_linear-regression-gluon.svg)
