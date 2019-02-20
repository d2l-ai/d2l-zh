# 自定义层

深度学习的一个魅力在于神经网络中各式各样的层，例如全连接层和后面章节中将要介绍的卷积层、池化层与循环层。虽然Gluon提供了大量常用的层，但有时候我们依然希望自定义层。本节将介绍如何使用`NDArray`来自定义一个Gluon的层，从而可以被重复调用。


## 不含模型参数的自定义层

我们先介绍如何定义一个不含模型参数的自定义层。事实上，这和[“模型构造”](model-construction.md)一节中介绍的使用`Block`类构造模型类似。下面的`CenteredLayer`类通过继承`Block`类自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了`forward`函数里。这个层里不含模型参数。

```{.python .input  n=1}
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

我们可以实例化这个层，然后做前向计算。

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

我们也可以用它来构造更复杂的模型。

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer())
```

下面打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数。

```{.python .input  n=4}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()
```

## 含模型参数的自定义层

我们还可以自定义含模型参数的自定义层。其中的模型参数可以通过训练学出。

[“模型参数的访问、初始化和共享”](parameters.md)一节分别介绍了`Parameter`类和`ParameterDict`类。在自定义含模型参数的层时，我们可以利用`Block`类自带的`ParameterDict`类型的成员变量`params`。它是一个由字符串类型的参数名字映射到Parameter类型的模型参数的字典。我们可以通过`get`函数从`ParameterDict`创建`Parameter`实例。

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

现在我们尝试实现一个含权重参数和偏差参数的全连接层。它使用ReLU函数作为激活函数。其中`in_units`和`units`分别代表输入个数和输出个数。

```{.python .input  n=19}
class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

下面，我们实例化`MyDense`类并访问它的模型参数。

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

我们可以直接使用自定义层做前向计算。

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))
```

我们也可以使用自定义层构造模型。它和Gluon的其他层在使用上很类似。

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

## 小结

* 可以通过`Block`类自定义神经网络中的层，从而可以被重复调用。


## 练习

* 自定义一个层，使用它做一次前向计算。




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1256)

![](../img/qr_custom-layer.svg)
