# 线性回归的Gluon实现

随着深度学习框架的发展，开发深度学习应用变得越来越便利。实践中，我们通常可以用比上一节中更简洁的代码来实现相同模型。本节中，我们将介绍如何使用MXNet提供的Gluon接口更方便地实现线性回归的训练。

## 生成数据集

我们生成与上一节中相同的数据集。其中`features`是训练数据特征，`labels`是标签。

```{.python .input  n=2}
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

## 读取数据

Gluon提供了`data`模块来读取数据。由于`data`常用作变量名，我们将导入的`data`模块用添加了Gluon首字母的假名`gdata`代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels) # 将训练数据的特征和标签组合。
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True) # 随机小批量读取
```

`data_iter`的使用跟上一节一样，让我们读取并打印第一个小批量数据样本。

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## 定义模型

在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。当模型结构变得更复杂时，这些步骤将变得更加繁琐。其实，Gluon提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。下面将介绍如何使用Gluon更简洁地定义线性回归。

首先，导入`nn`模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。我们先定义一个模型变量`net`，它是一个Sequential实例。在Gluon中，Sequential实例可以看做是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。

```{.python .input  n=5}
from mxnet.gluon import nn

net = nn.Sequential()
```

回顾图3.1中线性回归在神经网络图中的表示。作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1。

```{.python .input  n=6}
net.add(nn.Dense(1))
```

值得一提的是，在Gluon中我们无需指定每一层输入的形状，例如线性回归的输入个数。当模型看见数据时，例如后面执行`net(X)`时，模型将自动推断出每一层的输入个数。我们将在之后“深度学习计算”一章详细介绍这个机制。Gluon的这一设计为模型开发带来便利。


## 初始化模型参数

在使用`net`前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。我们从MXNet导入`initializer`模块。该模块提供了模型参数初始化的各种方法。这里的`init`是`initializer`的缩写形式。我们通过`init.Normal(sigma=0.01)`指定权重参数每个元素将在初始化时随机采样于均值为0标准差为0.01的正态分布。偏差参数默认会初始化为零。

```{.python .input  n=7}
from mxnet import init

net.initialize(init.Normal(sigma=0.01))
```

## 定义损失函数

在Gluon中，`loss`模块定义了各种损失函数。我们用假名`gloss`代替导入的`loss`模块，并直接使用它所提供的平方损失作为模型的损失函数。

```{.python .input  n=8}
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss() # 平方损失又称 L2 norm 损失
```

## 定义优化算法

同样，我们也无需实现小批量随机梯度下降。在导入Gluon后，我们创建一个Trainer实例，并指定学习率为0.03的小批量随机梯度下降（`sgd`）为优化算法。该优化算法将用来迭代`net`实例所有通过`add`函数嵌套的层所包含的所有参数，其可以通过`collect_params`获取。

```{.python .input  n=9}
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## 训练模型

在使用Gluon训练模型时，我们通过调用`Trainer`实例的`step`函数来迭代模型参数。由于变量`l`是长度为`batch_size`的一维NDArray，执行`l.backward()`等价于`l.sum().backward()`。按照小批量随机梯度下降的定义，我们在`step`函数中指明批量大小，其跟上一节实现的`sgd`函数那样会对样本梯度做平均。

```{.python .input  n=10}
num_epochs = 3
for epoch in range(1, num_epochs + 1): 
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

下面我们分别比较学到的和真实的模型参数。我们从`net`获得需要的层，并访问其权重（`weight`）和位移（`bias`）。学到的和真实的参数很接近。

```{.python .input  n=12}
dense = net[0]
true_w, dense.weight.data()
```

```{.python .input  n=13}
true_b, dense.bias.data()
```

## 小结

* 使用Gluon可以更简洁地实现模型。
* 在Gluon中，`data`模块提供了有关数据处理的工具，`nn`模块定义了大量神经网络的层，`loss`模块定义了各种损失函数。
* MXNet的`initializer`模块提供了模型参数初始化的各种方法。


## 练习

* 如果将`l = loss(output, y)`替换成`l = loss(output, y).mean()`，我们需要将`trainer.step(batch_size)`相应地改成`trainer.step(1)`。这是为什么呢？
* 查看`gloss`和`init`里面提供了哪些其他的损失函数和初始方法。
* 如何访问`dense.weight`的梯度？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/742)

![](../img/qr_linear-regression-gluon.svg)
