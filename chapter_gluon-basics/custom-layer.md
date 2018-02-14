# 设计自定义层

神经网络的一个魅力是它有大量的层，例如全连接、卷积、循环、激活，和各式花样的连接方式。我们之前学到了如何使用Gluon提供的层来构建新的层(`nn.Block`)继而得到神经网络。虽然Gluon提供了大量的[层的定义](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#neural-network-layers)，但我们仍然会遇到现有层不够用的情况。

这时候的一个自然的想法是，我们不是学习了如何只使用基础数值运算包`NDArray`来实现各种的模型吗？它提供了大量的[底层计算函数](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html)足以实现即使不是100%那也是95%的神经网络吧。

但每次都从头写容易写到怀疑人生。实际上，即使在纯研究的领域里，我们也很少发现纯新的东西，大部分时候是在现有模型的基础上做一些改进。所以很可能大部分是可以沿用前面的而只有一部分是需要自己来实现。

这个教程我们将介绍如何使用底层的`NDArray`接口来实现一个`Gluon`的层，从而可以以后被重复调用。

## 定义一个简单的层

我们先来看如何定义一个简单层，它不需要维护模型参数。事实上这个跟前面介绍的如何使用nn.Block没什么区别。下面代码定义一个层将输入减掉均值。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
        
    def forward(self, x):
        return x - x.mean()
```

我们可以马上实例化这个层用起来。

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))
```

我们也可以用它来构造更复杂的神经网络：

```{.python .input  n=3}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

确认下输出的均值确实是0：

```{.python .input  n=4}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean()
```

当然大部分情况你可以看不到一个实实在在的0，而是一个很小的数。例如`5.82076609e-11`。这是因为MXNet默认使用32位float，会带来一定的浮点精度误差。

## 带模型参数的自定义层

虽然`CenteredLayer`可能会告诉实现自定义层大概是什么样子，但它缺少了重要的一块，就是它没有可以学习的模型参数。

记得我们之前访问`Dense`的权重的时候是通过`dense.weight.data()`，这里`weight`是一个`Parameter`的类型。我们可以显示的构建这样的一个参数。

```{.python .input  n=5}
from mxnet import gluon
my_param = gluon.Parameter("exciting_parameter_yay", shape=(3,3))
```

这里我们创建一个$3\times3$大小的参数并取名为"exciting_parameter_yay"。然后用默认方法初始化打印结果。

```{.python .input  n=6}
my_param.initialize()
(my_param.data(), my_param.grad())
```

通常自定义层的时候我们不会直接创建Parameter，而是用过Block自带的一个ParamterDict类型的成员变量`params`，顾名思义，这是一个由字符串名字映射到Parameter的字典。

```{.python .input  n=7}
pd = gluon.ParameterDict(prefix="block1_")
pd.get("exciting_parameter_yay", shape=(3,3))
pd
```

现在我们看下如果实现一个跟`Dense`一样功能的层，它概念跟前面的`CenteredLayer`的主要区别是我们在初始函数里通过`params`创建了参数：

```{.python .input  n=19}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))        

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

我们创建实例化一个对象来看下它的参数，这里我们特意加了前缀`prefix`，这是`nn.Block`初始化函数自带的参数。

```{.python .input}
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
```

它的使用跟前面没有什么不一致：

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2,10)))
```

我们构造的层跟Gluon提供的层用起来没太多区别：

```{.python .input  n=19}
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2,64)))
```

仔细的你可能还是注意到了，我们这里指定了输入的大小，而Gluon自带的`Dense`则无需如此。我们已经在前面节介绍过了这个延迟初始化如何使用。但如果实现一个这样的层我们将留到后面介绍了hybridize后。

## 总结

现在我们知道了如何把前面手写过的层全部包装了Gluon能用的Block，之后再用到的时候就可以飞起来了！

## 练习

1. 怎么修改自定义层里参数的默认初始化函数。
1. (这个比较难），在一个代码Cell里面输入`nn.Dense??`，看看它是怎么实现的。为什么它就可以支持延迟初始化了。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1256)
