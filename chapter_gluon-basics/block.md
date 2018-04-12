# 模型构造

上一章介绍了简单的深度学习模型，例如多层感知机。为了引入深度学习计算的问题，我们以该模型为例，对输入数据做计算。

## 多层感知机的计算

在[“多层感知机——使用`Gluon`”](../chapter_supervised-learning/mlp-gluon.md)一节中，
我们通过在`nn.Sequential`里依次添加两个全连接层构造出多层感知机。其中第一层的输出大小为256，即隐藏层单元个数；第二层的输出大小为10，即输出层单元个数。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(10))
print(net)
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 256, Activation(relu))\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

由于此时`net`还没见到模型的输入数据，模型中每一层并未指明输入单元个数（以上打印结果中的`None`）。

接下来，让模型根据输入数据做一次计算。

```{.python .input  n=2}
net.initialize()
x = nd.random.uniform(shape=(4, 20))
y = net(x)
print(net)
print(y)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(20 -> 256, Activation(relu))\n  (1): Dense(256 -> 10, linear)\n)\n\n[[ 0.03126615  0.04562764  0.00039857 -0.08772386 -0.05355632  0.02904574\n   0.08102557 -0.01433946 -0.04224151  0.06047882]\n [ 0.02871901  0.03652265  0.00630051 -0.05650971 -0.07189322  0.08615957\n   0.05951559 -0.06045965 -0.0299026   0.05651001]\n [ 0.02147349  0.04818896  0.05321142 -0.12616856 -0.0685023   0.09096345\n   0.04064304 -0.05064794 -0.02200242  0.04859561]\n [ 0.03780478  0.0751239   0.03290457 -0.11641113 -0.03254967  0.0586529\n   0.02542157 -0.01697343 -0.00049652  0.05892839]]\n<NDArray 4x10 @cpu(0)>\n"
 }
]
```

在上面的例子中，`net`的输入数据`x`是4个样本，每个样本的特征向量长度为20（`shape=(4, 20)`）。在按照默认方式初始化好模型参数后，`net`计算出模型的输出并得到了一个$4 \times 10$的矩阵。其中4是数据样本个数，10是输出层单元个数。此外，当`net`见到模型的输入数据后（`y = net(x)`），模型中每一层的输入单元个数被自动求出（20和256）。

实际上，这个多层感知机计算的例子涉及到了深度学习计算的方方面面，例如模型的构造、模型参数的初始化、模型的层等。在本章中，我们将主要使用`Gluon`来介绍深度学习计算中的重要组成部分：模型构造、模型参数、自定义层、读写和GPU计算。通过本章的学习，读者将能够动手实现和训练更复杂的深度学习模型，例如之后章节里的一些模型。

本节中，我们将通过`Gluon`里的`nn.Block`来介绍如何构造深度学习模型。相信读者在学习完本节后，也会对上一章中使用的`nn.Sequential`有更深刻的认识。

## 使用 `nn.Block` 构造模型

使用`Gluon`提供的`nn.Block`，我们可以很方便地构造各种模型。例如，我们可以通过`nn.Block`来构造与上面相同的多层感知机。

```{.python .input  n=3}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)
    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))
```

可以看到`nn.Block`的使用是通过创建一个它子类的类，其中至少包含了两个函数。

- `__init__`：创建参数。上面例子我们使用了包含了参数的`dense`层
- `forward()`：定义网络的计算

我们所创建的类的使用跟前面`net`没有太多不一样。

```{.python .input  n=4}
net2 = MLP()
print(net2)
net2.initialize()
x = nd.random.uniform(shape=(4,20))
y = net2(x)
y
```

```{.python .input  n=5}
nn.Dense
```

如何定义创建和使用`nn.Dense`比较好理解。接下来我们仔细看下`MLP`里面用的其他命令：

- `super(MLP, self).__init__(**kwargs)`：这句话调用`nn.Block`的`__init__`函数，它提供了`prefix`（指定名字）和`params`（指定模型参数）两个参数。我们会之后详细解释如何使用。

- `self.name_scope()`：调用`nn.Block`提供的`name_scope()`函数。`nn.Dense`的定义放在这个`scope`里面。它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。默认自动会自动生成前缀，我们也可以在创建的时候手动指定。推荐在构建网络时，每个层至少在一个`name_scope()`里。

```{.python .input  n=17}


net3 = MLP(prefix='another_mlp_')
print('customized prefix:', net3.dense0.name)
```

大家会发现这里并没有定义如何求导，或者是`backward()`函数。事实上，系统会使用`autograd`对`forward()`自动生成对应的`backward()`函数。

## `nn.Block`到底是什么东西？

在`gluon`里，`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们可以（近似）无限地嵌套`nn.Block`来构建新的`nn.Block`。

`nn.Block`主要提供这个东西

1. 存储参数
2. 描述`forward`如何执行
3. 自动求导

## 那么现在可以解释`nn.Sequential`了吧

`nn.Sequential`是一个`nn.Block`容器，它通过`add`来添加`nn.Block`。它自动生成`forward()`函数，其就是把加进来的`nn.Block`逐一运行。

一个简单的实现是这样的：

```{.python .input  n=7}
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

可以跟`nn.Sequential`一样的使用这个自定义的类：

```{.python .input  n=8}
net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
y
```

可以看到，`nn.Sequential`的主要好处是定义网络起来更加简单。但`nn.Block`可以提供更加灵活的网络定义。考虑下面这个例子

```{.python .input  n=9}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))

    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x
```

看到这里我们直接手动创建和初始了权重`weight`，并重复用了`dense`的层。测试一下：

```{.python .input  n=10}
fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)
```

## `nn.Block`和`nn.Sequential`的嵌套使用

现在我们知道了`nn`下面的类基本都是`nn.Block`的子类，他们可以很方便地嵌套使用。

```{.python .input  n=11}
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)
```

## 小结

* 不知道你同不同意，通过`nn.Block`来定义神经网络跟玩积木很类似。

## 练习

* 如果把`RecMLP`改成`self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]`，`forward`就用for loop来实现，会有什么问题吗？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_block.svg)
