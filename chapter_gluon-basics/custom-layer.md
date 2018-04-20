# 自定义层

深度学习的一个魅力之处在于神经网络中各式各样的层，例如全连接层和后面章节中将要介绍的卷积层、池化层与循环层。虽然Gluon提供了大量常用的层，但有时候我们依然希望自定义层。本节将介绍如何使用`NDArray`来自定义一个Gluon的层，从而以后可以被重复调用。


## 不含模型参数的自定义层

我们先介绍如何定义一个不含模型参数的自定义层。事实上，这和[“模型构造”](block.md)一节中介绍的使用Block构造模型类似。

首先，导入本节中实验需要的包。

```{.python .input}
from mxnet import nd, gluon
from mxnet.gluon import nn
```

下面通过继承Block自定义了一个将输入减掉均值的层`CenteredLayer`，并将层的计算放在`forward`函数里。这个层里不含模型参数。

```{.python .input  n=1}
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

我们可以实例化这个层用起来。

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

我们也可以用它来构造更复杂的模型。

```{.python .input  n=3}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

打印自定义层输出的均值。由于均值是浮点数，它的值是个很接近0的数。

```{.python .input  n=4}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean()
```

## 含模型参数的自定义层

我们还可以自定义含模型参数的自定义层。这样，自定义层里的模型参数就可以通过训练学出来了。我们在[“模型参数”](parameters.md)一节里介绍了`Parameter`类。其实，在自定义层的时候我们还可以使用Block自带的`ParameterDict`类型的成员变量`params`。顾名思义，这是一个由字符串类型的参数名字映射到`Parameter`类型的模型参数的字典。我们可以通过`get`从`ParameterDict`创建`Parameter`。

```{.python .input  n=7}
params = gluon.ParameterDict(prefix="block1_")
params.get("param2", shape=(2, 3))
params
```

现在我们看下如何实现一个含权重参数和偏差参数的全连接层。它使用ReLU作为激活函数。其中`in_units`和`units`分别是输入单元个数和输出单元个数。

```{.python .input  n=19}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))        

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

下面，我们实例化`MyDense`来看下它的模型参数。这里我们特意加了名字前缀`prefix`。在[“模型构造”](block.md)一节中介绍过，这是Block的构造函数自带的参数。

```{.python .input}
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
```

我们可以直接使用自定义层做计算。

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 10)))
```

我们也可以使用自定义层构造模型。它用起来和Gluon的其他层很类似。

```{.python .input  n=19}
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

## 小结

* 使用Block，我们可以方便地自定义层。


## 练习

* 如何修改自定义层里模型参数的默认初始化函数？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1256)

![](../img/qr_custom-layer.svg)
