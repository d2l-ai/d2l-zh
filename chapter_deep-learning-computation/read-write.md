# 读取和存储

到目前为止，我们介绍了如何处理数据以及构建、训练和测试深度学习模型。然而在实际中，我们有时需要把训练好的模型部署到很多不同的设备。这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。


## 读写NDArrays

我们首先看如何读写NDArray。我们可以直接使用`save`和`load`函数分别存储和读取NDArray。下面是例子我们创建`x`，并将其存在文件名同为`x`的文件里。

```{.python .input}
from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)
```

然后我们再将数据从文件读回内存。

```{.python .input}
x2 = nd.load('x')
x2
```

同样我们可以存储一列NDArray并读回内存。

```{.python .input  n=2}
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
(x2, y2)
```

或者是一个从字符串到NDArray的字典。

```{.python .input  n=4}
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
mydict2
```

## 读写Gluon模型的参数

Block类提供了`save_parameters`和`load_parameters`函数来读写模型参数。它实际做的事情就是将所有参数保存成一个名称到NDArray的字典到文件。读取的时候会根据参数名称找到对应的NDArray并赋值。下面的例子我们首先创建一个多层感知机，初始化后将模型参数保存到文件里。

下面，我们创建一个多层感知机。

```{.python .input  n=6}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()

# 由于延后初始化，我们需要先运行一次前向计算才能实际初始化模型参数。
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

下面我们把该模型的参数存起来。

```{.python .input}
filename = 'mlp.params'
net.save_parameters(filename)
```

然后，我们再实例化一次我们定义的多层感知机。但跟前面不一样是我们不是随机初始化模型参数，而是直接读取保存在文件里的参数。

```{.python .input  n=8}
net2 = MLP()
net2.load_parameters(filename)
```

因为这两个实例都有同样的参数，那么对同一个`x`的计算结果将会是一样。

```{.python .input}
y2 = net2(x)
y2 == y
```

## 小结

* 通过`save`和`load`可以很方便地读写NDArray。
* 通过`load_parameters`和`save_parameters`可以很方便地读写Gluon的模型参数。

## 练习

* 即使无需把训练好的模型部署到不同的设备，存储模型参数在实际中还有哪些好处？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1255)

![](../img/qr_read-write.svg)
