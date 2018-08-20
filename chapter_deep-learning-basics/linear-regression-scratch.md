# 线性回归的从零开始实现

在了解了线性回归的背景知识之后，现在我们可以动手实现它了。尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，会导致我们很难深入理解深度学习是如何工作的。因此，本节将介绍如何只利用NDArray和`autograd`来实现一个线性回归的训练。

首先，导入本节中实验所需的包或模块，其中的matplotlib包可用于作图，且设置成嵌入显示。

```{.python .input  n=1}
%matplotlib inline
import random
from matplotlib import pyplot as plt
from IPython import display
from mxnet import autograd, nd
```

## 生成数据集

我们构造一个简单的人工训练数据集，它可以使我们能够直观比较学到的参数和真实的模型参数的区别。设训练数据集样本数为1000，输入特征个数为2。给定随机生成的批量样本特征$\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$，我们使用线性回归模型真实权重$\boldsymbol{w} = [2, -3.4]^\top$和偏差$b = 4.2$，以及一个随机噪音项$\epsilon$来生成标签

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
features[0], labels[0]
```

通过生成第二个特征`features[:, 1]`和标签 `labels` 的散点图，我们可以更直观地观察两者间的线性关系。

```{.python .input  n=4}
def use_svg_display():
    # 用矢量图显示。
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    # 设置图的尺寸。
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize  # 设置

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1); 
```

我们将上面的`plt`作图函数以及`use_svg_display`和`set_figsize`函数定义在`gluonbook`包里。以后在作图时，我们将直接调用`gluonbook.plt`。由于`plt`在`gluonbook`包中是一个全局变量，我们在作图前只需要调用`gluonbook.set_figsize()`即可打印高清图并设置图的尺寸。


## 读取数据

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回`batch_size`（批量大小）个随机样本的特征和标签。

```{.python .input  n=5}
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的。
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])        
        yield features.take(j), labels.take(j) # take 函数根据索引返回对应元素。
```

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为（10， 2），分别对应批量大小和输入个数；标签形状为批量大小。

```{.python .input  n=6}
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

## 初始化模型参数

我们将权重初始化成均值0标准差为0.01的正态随机数，偏差则初始化成0。

```{.python .input  n=7}
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```

之后训练时我们需要对这些参数求梯度来迭代它们的值，因此我们需要创建它们的梯度。

```{.python .input  n=8}
w.attach_grad()
b.attach_grad()
```

## 定义模型

下面是线性回归的矢量计算表达式的实现。我们使用`dot`函数做矩阵乘法。

```{.python .input  n=9}
def linreg(X, w, b): 
    return nd.dot(X, w) + b 
```

## 定义损失函数

我们使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，我们需要把真实值`y`变形成预测值`y_hat`的形状。以下函数返回的结果也将和`y_hat`的形状相同。

```{.python .input  n=10}
def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

## 定义优化算法

以下的`sgd`函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断更新模型参数来优化损失函数。这里自动求导模块计算得来的梯度是一个批量样本的梯度和，我们除以批量大小来得到平均值。这样使用不同批量大小`batch_size`的时候对优化算法的影响将变小，例如学习率`lr`将不会过于敏感。

```{.python .input  n=11}
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

我们将`data_iter`、`linreg`、`squared_loss`和`sgd`函数定义在`gluonbook`包中供后面章节调用。


## 训练模型

现在我们可以开始训练模型了。在训练中，我们将有限次地迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征`features`和标签`label`），通过调用反向函数`backward`计算小批量随机梯度，并调用优化算法`sgd`迭代模型参数。在一个迭代周期（epoch）中，我们将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。当迭代周期数设的越大时，虽然模型可能更有效，但是训练时间可能过长。而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。

```{.python .input  n=12}
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs): # 训练模型一共需要 num_epochs 个迭代周期。
    # 在一个迭代周期中，使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X 和 y 分别是小批量样本的特征和标签。
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():            
            l = loss(net(X, w, b), y) # l 是有关小批量 X 和 y 的损失。        
        l.backward() # 小批量的损失对模型参数求导。        
        sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数。
    l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, l.mean().asnumpy()))
```

训练完成后，我们可以比较学到的参数和用来生成训练集的真实参数。它们应该很接近。

```{.python .input  n=13}
true_w, w
```

```{.python .input  n=14}
true_b, b
```

## 小结

我们现在看到，仅使用NDArray和`autograd`就可以很容易地实现一个模型。在接下来的章节中，我们会在此基础上描述更多深度学习模型，并介绍怎样使用更简洁的代码（例如下一节）实现它们。


## 练习

* 为什么`squared_loss`函数中需要使用`reshape`?
* 尝试用不同的学习率查看损失函数值的下降速度。
* 回顾[“自动求梯度”](../chapter_prerequisite/autograd.md)一节。本节代码中变量`l`并不是一个标量，运行`l.backward()`将如何对模型参数求梯度？
* 如果样本个数不能被批量大小整除时会怎么样？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/743)

![](../img/qr_linear-regression-scratch.svg)
