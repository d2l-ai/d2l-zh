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
import sys
sys.path.append('..')
import utils
```

## 生成数据集

我们在这里描述生成人工训练数据集的真实模型。

设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征$\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$，我们使用线性回归模型真实权重$\boldsymbol{w} = [2, -3.4]^\top$和偏差$b = 4.2$，以及一个随机噪音项$\epsilon$来生成标签

$$\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon,$$

其中噪音项$\epsilon$服从均值为0和标准差为0.01的正态分布。下面，让我们生成数据集。


```{.python .input  n=2}
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += nd.random.normal(scale=0.01, shape=y.shape)
```

注意到`X`的每一行是一个长度为2的向量，而`y`的每一行是一个长度为1的向量（标量）。

```{.python .input  n=3}
print(X[0], y[0])
```

通过生成第二个特征`X[:, 1]`和标签 `y` 的散点图，我们可以更直观地观察两者间的线性关系。

```{.python .input  n=4}
utils.set_fig_size(mpl)
plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
plt.show()
```

## 遍历数据集

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回`batch_size`个随机样本的特征和标签。我们设批量大小为10。

```{.python .input  n=5}
batch_size = 10
def data_iter(): 
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield X.take(j), y.take(j)
```

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为`（10, 2）`，分别对应批量大小`batch_size`和输入个数`num_inputs`；标签形状为`10`，也就是批量大小。

```{.python .input  n=6}
for data, label in data_iter():
    print(data, label)
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

下面将线性回归的矢量计算表达式翻译成代码。

```{.python .input  n=9}
def net(X, w, b): 
    return nd.dot(X, w) + b 
```

## 损失函数

我们使用常见的平方误差来衡量预测目标和真实目标之间的差距。

```{.python .input  n=10}
def squared_loss(yhat, y): 
    # 注意这里我们把 y 变形成 yhat 的形状来避免矩阵形状的自动转换。
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2
```

## 优化

虽然线性回归有显式解，但绝大部分模型并没有。所以我们这里通过随机梯度下降来求解。每一步，我们将模型参数沿着梯度的反方向走特定距离，这个距离一般叫**学习率（learning rate）** `lr`。（我们会之后一直使用这个函数，我们将其保存在[utils.py](../utils.py)。）

```{.python .input  n=11}
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

## 训练

现在我们可以开始训练了。训练通常需要迭代数据数次，在这里使用`epochs`表示迭代总次数；一次迭代中，我们每次随机读取固定数个数据点，计算梯度并更新模型参数。


TODO(@astonzhang) 向量backward，epochs。

```{.python .input  n=12}
lr = 0.05
num_epochs = 3

for epoch in range(1, num_epochs + 1):
    for features, label in data_iter():
        with autograd.record():
            output = net(features, w, b)
            loss = squared_loss(output, label)
        loss.backward()
        sgd([w, b], lr, batch_size)
    print("epoch %d, loss: %f" 
          % (epoch, squared_loss(net(X, w, b), y).mean().asnumpy()))
```

训练完成后，我们可以比较学得的参数和真实参数

```{.python .input  n=13}
true_w, w
```

```{.python .input  n=14}
true_b, b
```

## 小结

* 我们现在看到，仅仅是使用NDArray和autograd就可以很容易实现的一个模型。在接下来的教程里，我们会在此基础上，介绍更多现代神经网络的知识，以及怎样使用少量的MXNet代码实现各种复杂的模型。

## 练习

* 尝试用不同的学习率查看误差下降速度（收敛率）


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/743)

![](../img/qr_linear-regression-scratch.svg)
