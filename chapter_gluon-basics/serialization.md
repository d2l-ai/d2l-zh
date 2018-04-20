# 读取和存储

到目前为止，我们介绍了如何处理数据以及构建、训练和测试深度学习模型。然而在实际中，我们有时需要把训练好的模型部署到很多不同的设备。这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。


## 读写NDArrays

首先，导入本节中实验所需的包。

```{.python .input}
from mxnet import nd
from mxnet.gluon import nn
```

我们看看如何读写NDArray。我们可以直接使用`save`和`load`函数分别存储和读取NDArray。事实上，MXNet支持跨语言（例如R和Scala）的存储和读取。

下面是存储NDArray的例子。

```{.python .input  n=2}
x = nd.ones(3)
y = nd.zeros(4)
filename = "../data/test1.params"
nd.save(filename, [x, y])
```

读取并打印上面存储的NDArray。

```{.python .input  n=3}
a, b = nd.load(filename)
print(a, b)
```

我们也可以存储和读取含NDArray的词典。

```{.python .input  n=4}
mydict = {"x": x, "y": y}
filename = "../data/test2.params"
nd.save(filename, mydict)
c = nd.load(filename)
print(c)
```

## 读写Gluon模型的参数

在[“模型构造”](block.md)一节中，我们了解了Gluon模型通常是个Block。与NDArray类似，Block提供了`save_params`和`load_params`函数来读写模型参数。

下面，我们创建一个多层感知机。

```{.python .input  n=6}
def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(2))
    return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
print(net(x))
```

下面我们把该模型的参数存起来。

```{.python .input}
filename = "../data/mlp.params"
net.save_params(filename)
```

然后，我们构建一个同`net`一样的多层感知机`net2`。这一次，`net2`不像`net`那样随机初始化，而是直接读取`net`的模型参数。这样，给定同样的输入`x`，`net2`会输出同样的计算结果。

```{.python .input  n=8}
import mxnet as mx
net2 = get_net()
net2.load_params(filename, mx.cpu(0))
print(net2(x))
```

## 小结

* 通过`save`和`load`可以很方便地读写NDArray。
* 通过`load_params`和`save_params`可以很方便地读写Gluon的模型参数。

## 练习

* 即使无需把训练好的模型部署到不同的设备，存储模型参数在实际中还有哪些好处？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1255)

![](../img/qr_serialization.svg)
