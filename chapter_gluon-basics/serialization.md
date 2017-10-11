# 序列化 --- 读写模型

我们现在已经讲了很多，包括

- 如何处理数据
- 如何构建模型
- 如何在数据上训练模型
- 如何使用不同的损失函数来做分类和回归

但即使知道了所有这些，我们还没有完全准备好来构建一个真正的机器学习系统。这是因为我们还没有讲如何读和写模型。因为现实中，我们通常在一个地方训练好模型，然后部署到很多不同的地方。我们需要把内存中的训练好的模型存在硬盘上好下次使用。

## 读写NDArrays

作为开始，我们先看看如何读写NDArray。虽然我们可以使用Python的序列化包例如`Pickle`，不过我们更倾向直接`save`和`load`，通常这样更快，而且别的语言，例如R和Scala也能用到。

```{.python .input  n=2}
from mxnet import nd

x = nd.ones(3)
y = nd.zeros(4)
filename = "../data/test1.params"
nd.save(filename, [x, y])
```

读回来

```{.python .input  n=3}
a, b = nd.load(filename)
print(a, b)
```

不仅可以读写单个NDArray，NDArray list，dict也是可以的：

```{.python .input  n=4}
mydict = {"x": x, "y": y}
filename = "../data/test2.params"
nd.save(filename, mydict)
```

```{.python .input  n=5}
c = nd.load(filename)
print(c)
```

## 读写Gluon模型的参数

跟NDArray类似，Gluon的模型（就是`nn.Block`）提供便利的`save_params`和`load_params`函数来读写数据。我们同前一样创建一个简单的多层感知机

```{.python .input  n=6}
from mxnet.gluon import nn

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(2))
    return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2,10))
print(net(x))
```

下面我们把模型参数存起来

```{.python .input}
filename = "../data/mlp.params"
net.save_params(filename)
```

之后我们构建一个一样的多层感知机，但不像前面那样随机初始化，我们直接读取前面的模型参数。这样给定同样的输入，新的模型应该会输出同样的结果。

```{.python .input  n=8}
import mxnet as mx
net2 = get_net()
net2.load_params(filename, mx.cpu())  # FIXME, gluon will support default ctx later 
print(net2(x))
```

## 总结

通过`load_params`和`save_params`可以很方便的读写模型参数。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1255)
