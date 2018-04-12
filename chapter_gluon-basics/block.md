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
```

接下来，让模型根据输入数据做一次计算。

```{.python .input  n=2}
net.initialize()
x = nd.random.uniform(shape=(4, 20))
y = net(x)
print(y)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.03126615  0.04562764  0.00039857 -0.08772386 -0.05355632  0.02904574\n   0.08102557 -0.01433946 -0.04224151  0.06047882]\n [ 0.02871901  0.03652265  0.00630051 -0.05650971 -0.07189322  0.08615957\n   0.05951559 -0.06045965 -0.0299026   0.05651001]\n [ 0.02147349  0.04818896  0.05321142 -0.12616856 -0.0685023   0.09096345\n   0.04064304 -0.05064794 -0.02200242  0.04859561]\n [ 0.03780478  0.0751239   0.03290457 -0.11641113 -0.03254967  0.0586529\n   0.02542157 -0.01697343 -0.00049652  0.05892839]]\n<NDArray 4x10 @cpu(0)>\n"
 }
]
```

在上面的例子中，`net`的输入数据`x`包含4个样本，每个样本的特征向量长度为20（`shape=(4, 20)`）。在按照默认方式初始化好模型参数后，`net`计算得到一个$4 \times 10$的矩阵作为模型的输出。其中4是数据样本个数，10是输出层单元个数。

实际上，这个多层感知机计算的例子涉及到了深度学习计算的方方面面，例如模型的构造、模型参数的初始化、模型的层等。在本章中，我们将主要使用`Gluon`来介绍深度学习计算中的重要组成部分：模型构造、模型参数、自定义层、读写和GPU计算。通过本章的学习，读者将能够动手实现和训练更复杂的深度学习模型，例如之后章节里的一些模型。

本节中，我们将通过`Gluon`里的`nn.Block`来介绍如何构造深度学习模型。相信读者在学习完本节后，也会对上一章中使用的`nn.Sequential`有更深刻的认识。

## 使用 `nn.Block` 构造模型

使用`Gluon`提供的`nn.Block`，我们可以很方便地构造各种模型。例如，我们可以通过`nn.Block`来构造与本节开头例子中相同的多层感知机。

```{.python .input  n=3}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(256)
            self.output = nn.Dense(10)
    def forward(self, x):
        return self.output(nd.relu(self.hidden(x)))
```

这里，我们通过创建`nn.Block`的子类构造模型。任意一个`nn.Block`的子类至少实现以下两个函数：

* `__init__`：创建模型的参数。在上面的例子里，模型的参数被包含在了两个`nn.Dense`层里。
* `forward`：定义模型的计算。

接下来我们解释一下`MLP`里面用的其他命令：

* `super(MLP, self).__init__(**kwargs)`：这句话调用`MLP`父类`nn.Block`的构造函数`__init__`。这样，我们在调用`MLP`的构造函数时还可以指定函数参数`prefix`（名字前缀）或`params`（模型参数，下一节会介绍）。这两个函数参数将通过`**kwargs`传递给`nn.Block`的构造函数。

* `with self.name_scope()`：本例中的两个`nn.Dense`层和其中模型参数的名字前面都将带有模型名前缀。该前缀可以通过构造函数参数`prefix`指定。若未指定，该前缀将自动生成。我们建议，在构造模型时将每个层至少放在一个`name_scope()`里。

我们可以实例化`MLP`类得到`net2`，并让`net2`根据输入数据`x`做一次计算。其中，`y = net2(x)`明确调用了`MLP`中的`__call__`函数。在`Gluon`中，这将进一步调用`MLP`中的`forward`函数从而完成一次模型计算。

```{.python .input  n=12}
net2 = MLP()
net2.initialize()
x = nd.random.uniform(shape=(4, 20))
y = net2(x)
print(y)
print('hidden layer name with default prefix:', net2.hidden.name)
print('output layer name with default prefix:', net2.output.name)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.03891213 -0.00315471  0.02346008  0.0115821  -0.0286346   0.04750714\n  -0.04022511  0.03334583 -0.02862417  0.00103271]\n [-0.02815309 -0.02004078  0.05494304 -0.01509079 -0.02990984  0.0201303\n  -0.06145176  0.00554352  0.00067628 -0.00712057]\n [-0.03600026 -0.01051114  0.03732241  0.00269452 -0.04287211  0.05638577\n  -0.04959137  0.03910026 -0.04694562  0.04687501]\n [-0.0008434  -0.01940108  0.06147967  0.00818795 -0.06385357 -0.00276085\n  -0.03296277  0.0426305  -0.01784454 -0.01684828]]\n<NDArray 4x10 @cpu(0)>\nhidden layer name with default prefix: mlp1_dense0\noutput layer name with default prefix: mlp1_dense1\n"
 }
]
```

在上面的例子中，隐藏层和输出层的名字前都加了默认前缀。接下来我们通过`prefix`指定它们的名字前缀。

```{.python .input  n=5}
net3 = MLP(prefix='my_mlp_')
print('hidden layer name with "my_mlp_" prefix:', net3.hidden.name)
print('output layer name with "my_mlp_" prefix:', net3.output.name)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "hidden layer name with \"my_mlp_\" prefix: my_mlp_dense0\noutput layer name with \"my_mlp_\" prefix: my_mlp_dense1\n"
 }
]
```

接下来，我们重新定义`MLP_NO_NAMESCOPE`类。它和`MLP`的区别就是不含`with self.name_scope():`。这是，隐藏层和输出层的名字前都不再含指定的前缀`prefix`。

```{.python .input  n=6}
class MLP_NO_NAMESCOPE(nn.Block):
    def __init__(self, **kwargs):
        super(MLP_NO_NAMESCOPE, self).__init__(**kwargs)
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)
    def forward(self, x):
        return self.output(nd.relu(self.hidden(x)))

net4 = MLP_NO_NAMESCOPE(prefix='my_mlp_')
print('hidden layer name without prefix:', net4.hidden.name)
print('output layer name without prefix:', net4.output.name)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "hidden layer name without prefix: dense0\noutput layer name without prefix: dense1\n"
 }
]
```

最后需要指出的是，在`gluon`里，`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们还可以反复嵌套`nn.Block`来构建新的`nn.Block`。

`nn.Block`类主要提供模型参数的存储、模型计算的定义和自动求导。读者也许已经发现了，以上`nn.Block`的子类中并没有定义如何求导，或者是`backward`函数。事实上，`MXNet`会使用`autograd`对`forward()`自动生成相应的`backward()`函数。


## `nn.Sequential`是特殊的`nn.Block`

在`Gluon`里，`nn.Sequential`是`nn.Block`的子类。它也可以被看作是一个`nn.Block`的容器：通过`add`函数来添加`nn.Block`。在`forward()`函数里，`nn.Sequential`把添加进来的`nn.Block`逐一运行。

一个简单的实现是这样的：

```{.python .input  n=17}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

它的使用和`nn.Sequential`类似：

```{.python .input  n=18}
net5 = MySequential()
with net5.name_scope():
    net5.add(nn.Dense(256, activation="relu"))
    net5.add(nn.Dense(10))
net5.initialize()
y = net5(x)
y
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[[ 9.5400095e-02  5.4852065e-02  1.7744238e-02 -1.3019045e-01\n  -6.2494464e-03 -1.1951900e-02  1.0515226e-02 -6.5923519e-02\n  -2.3991371e-02  1.1632015e-01]\n [ 9.6949443e-02  4.0487815e-02 -6.6828108e-03 -8.5242800e-02\n  -1.1140321e-02 -1.1530784e-02 -6.3379779e-03 -4.9934912e-02\n  -6.0178801e-02  4.9789295e-02]\n [ 1.1893168e-01  4.7171779e-02  1.1178985e-02 -1.2403722e-01\n  -2.7897144e-03 -1.9721132e-02 -1.2032428e-02 -2.1957448e-02\n  -2.9257955e-02  1.0800241e-01]\n [ 1.0659003e-01  7.6273575e-02 -4.2447657e-03 -1.0006615e-01\n   4.8290327e-05 -7.5394958e-03 -1.3146491e-02 -4.5292892e-02\n  -5.3848229e-02  8.6303681e-02]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到，`nn.Sequential`的主要好处是定义网络起来更加简单。但`nn.Block`可以提供更加灵活的网络定义。考虑下面这个例子

```{.python .input  n=21}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.weight = nd.random_uniform(shape=(256,20))
        with self.name_scope():
            self.dense = nn.Dense(256)

    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x
```

看到这里我们直接手动创建和初始了权重`weight`，并重复用了`dense`的层。测试一下：

```{.python .input  n=22}
fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)
```

```{.json .output n=22}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(4, 256)\n"
 }
]
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

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): RecMLP(\n    (net): Sequential(\n      (0): Dense(None -> 256, Activation(relu))\n      (1): Dense(None -> 128, Activation(relu))\n    )\n    (dense): Dense(None -> 64, linear)\n  )\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

## 小结

* 不知道你同不同意，通过`nn.Block`来定义神经网络跟玩积木很类似。

## 练习

* 如果把`RecMLP`改成`self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]`，`forward`就用for loop来实现，会有什么问题吗？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_block.svg)
