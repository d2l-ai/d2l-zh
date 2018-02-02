# 线性回归 --- 使用Gluon

[前一章](linear-regression-scratch.md)我们仅仅使用了`ndarray`和`autograd`来实现线性回归，这一章我们仍然实现同样的模型，但是使用高层抽象包`gluon`。

## 创建数据集

我们生成同样的数据集

```{.python .input  n=2}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
```

## 数据读取

但这里使用`data`模块来读取数据。

```{.python .input  n=3}
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
```

读取跟前面一致：

```{.python .input  n=5}
for data, label in data_iter:
    print(data, label)
    break
```

## 定义模型

之前一章中，当我们从0开始训练模型时，需要先声明模型参数，然后再使用它们来构建模型。但`gluon`提供大量预定义的层，我们只需要关注使用哪些层来构建模型。例如线性模型就是使用对应的`Dense`层；之所以称为dense层，是因为输入的所有节点都与后续的节点相连。在这个例子中仅有一个输出，但在大多数后续章节中，我们会用到具有多个输出的网络。

我们之后还会介绍如何构造任意结构的神经网络，但对于初学者来说，构建模型最简单的办法是利用`Sequential`来所有层串起来。输入数据之后，`Sequential`会依次执行每一层，并将前一层的输出，作为输入提供给后面的层。首先我们定义一个空的模型：

```{.python .input  n=5}
net = gluon.nn.Sequential()
```

然后我们加入一个`Dense`层，它唯一必须定义的参数就是输出节点的个数，在线性模型里面是1.

```{.python .input  n=6}
net.add(gluon.nn.Dense(1))
```

（注意这里我们并没有定义说这个层的输入节点是多少，这个在之后真正给数据的时候系统会自动赋值。我们之后会详细介绍这个特性是如何工作的。）

## 初始化模型参数

在使用前`net`我们必须要初始化模型权重，这里我们使用默认随机初始化方法（之后我们会介绍更多的初始化方法）。

```{.python .input  n=7}
net.initialize()
```

## 损失函数

`gluon`提供了平方误差函数：

```{.python .input  n=8}
square_loss = gluon.loss.L2Loss()
```

## 优化

同样我们无需手动实现随机梯度下降，我们可以创建一个`Trainer`的实例，并且将模型参数传递给它就行。

```{.python .input  n=9}
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练
使用`gluon`使模型训练过程更为简洁。我们不需要挨个定义相关参数、损失函数，也不需使用随机梯度下降。`gluon`的抽象和便利的优势将随着我们着手处理更多复杂模型的愈发显现。不过在完成初始设置后，训练过程本身和前面没有太多区别，唯一的不同在于我们不再是调用`SGD`，而是`trainer.step`来更新模型（此处一并省略之前绘制损失变化的折线图和散点图的过程，有兴趣的同学可以自行尝试）。

```{.python .input  n=10}
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
```

比较学到的和真实模型。我们先从`net`拿到需要的层，然后访问其权重和位移。

```{.python .input  n=12}
dense = net[0]
true_w, dense.weight.data()
```

```{.python .input  n=13}
true_b, dense.bias.data()
```

## 结论

可以看到`gluon`可以帮助我们更快更干净地实现模型。


## 练习

- 在训练的时候，为什么我们用了比前面要大10倍的学习率呢？（提示：可以尝试运行 `help(trainer.step)`来寻找答案。）
- 如何拿到`weight`的梯度呢？（提示：尝试 `help(dense.weight)`）

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/742)
