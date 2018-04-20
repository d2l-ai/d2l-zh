# 模型参数

为了引出本节的话题，让我们先构造一个多层感知机。首先，导入本节中实验所需的包。

```{.python .input  n=1}
from mxnet import init, gluon, nd
from mxnet.gluon import nn
import sys
```

下面定义多层感知机。

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(4)
            self.output = nn.Dense(2)

    def forward(self, x):
        return self.output(nd.relu(self.hidden(x)))
```

运行下面代码，系统抱怨说模型参数没有初始化。

```{.python .input  n=2}
x = nd.random.uniform(shape=(3, 5))
try:
    net = MLP()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

作如下修改之后，模型便计算成功。

```{.python .input  n=3}
net.initialize()
net(x)
```

这里添加的`net.initialize()`对模型参数做了初始化。模型参数是深度学习计算中的重要组成部分。本节中，我们将介绍如何访问、初始化和共享模型参数。

## 访问模型参数

在Gluon中，模型参数的类型是`Parameter`。下面让我们创建一个名字叫“good_param”、形状为$2 \times 3$的模型参数。在默认的初始化中，模型参数中的每一个元素是一个在`[-0.07, 0.07]`之间均匀分布的随机数。相应地，该模型参数还有一个形状为$2 \times 3$的梯度，初始值为0。

```{.python .input  n=4}
my_param = gluon.Parameter("good_param", shape=(2, 3))
my_param.initialize()
print('data: ', my_param.data(), '\ngrad: ', my_param.grad(),
      '\nname: ', my_param.name)
```

接下来，让我们访问本节开头定义的多层感知机`net`中隐藏层`hidden`的模型参数：权重`weight`和偏差`bias`。它们的类型也都是`Parameter`。我们可以看到它们的名字、形状和数据类型。

```{.python .input  n=5}
w = net.hidden.weight
b = net.hidden.bias
print('hidden layer name: ', net.hidden.name, '\nweight: ', w, '\nbias: ', b)
```

我们同样可以访问这两个参数的值和梯度。

```{.python .input  n=6}
print('weight:', w.data(), '\nweight grad:', w.grad(), '\nbias:', b.data(),
      '\nbias grad:', b.grad())
```

另外，我们也可以通过`collect_params`来访问Block里的所有参数（包括所有的子Block）。它会返回一个名字到对应`Parameter`的字典。在这个字典中，我们既可以用`[]`（需要指定前缀），又可以用`get()`（不需要指定前缀）来访问模型参数。

```{.python .input  n=7}
params = net.collect_params()
print(params)
print(params['mlp0_dense0_bias'].data())
print(params.get('dense0_bias').data())
```

## 初始化模型参数

在Gluon中，模型的偏差参数总是默认初始化为0。当我们对整个模型所有参数做初始化时，默认下权重参数的所有元素为[-0.07, 0.07]之间均匀分布的随机数。我们也可以使用其他初始化方法。以下例子使用了均值为0，标准差为0.02的正态分布来随机初始化模型中所有层的权重参数。

```{.python .input  n=8}
params = net.collect_params()
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print('hidden weight: ', net.hidden.weight.data(), '\nhidden bias: ',
      net.hidden.bias.data(), '\noutput weight: ', net.output.weight.data(),
      '\noutput bias: ',net.output.bias.data())
```

我们也可以把模型中任意层任意参数初始化，例如把上面模型中隐藏层的偏差参数初始化为1。

```{.python .input  n=9}
net.hidden.bias.initialize(init=init.One(), force_reinit=True)
print(net.hidden.bias.data())
```

### 自定义初始化方法

下面我们自定义一个初始化方法。它通过重载`_init_weight`来实现自定义的初始化方法。

```{.python .input  n=13}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状。
        nd.random.uniform(low=10, high=20, out=arr)

net = MLP()
net.initialize(MyInit())
net(x)
net.hidden.weight.data()
```

我们还可以通过`Parameter.set_data`来直接改写模型参数。

```{.python .input  n=14}
net = MLP()
net.initialize()
net(x)
print('output layer default weight:', net.output.weight.data())

w = net.output.weight
w.set_data(nd.ones(w.shape))
print('output layer modified weight:', net.output.weight.data())
```

## 延后的初始化

我们在本节开头定义的`MLP`模型的层`nn.Dense(4)`和`nn.Dense(2)`中无需指定它们的输入单元个数。定义`net = MLP()`和输入数据`x`。我们在[“模型构造”](block.md)一节中介绍过，执行`net(x)`将调用`net`的`forward`函数计算模型输出。在这次计算中，`net`也将从输入数据`x`的形状自动推断模型中每一层尚未指定的输入单元个数，得到模型中所有参数形状，并真正完成模型参数的初始化。因此，在上面两个例子中，我们总是在调用`net(x)`之后访问初始化的模型参数。

这种延后的初始化带来的一大便利是，我们在构造模型时无需指定每一层的输入单元个数。


下面，我们具体来看延后的初始化是怎么工作的。让我们新建一个网络并打印所有模型参数。这时，两个全连接层的权重的形状里都有0。它们代表尚未指定的输入单元个数。

```{.python .input}
net = MLP()
net.collect_params()
```

然后，调用`net.initialize()`并打印所有模型参数。这时模型参数依然没有被初始化。

```{.python .input}
net.initialize()
net.collect_params()
```

接下来，当模型见到输入数据`x`后（`shape=(3, 5)`），模型每一层参数的形状得以推断，参数的初始化最终完成。

```{.python .input  n=12}
print(x)
net(x)
net.collect_params()
```

## 共享模型参数

在有些情况下，我们希望模型的多个层之间共享模型参数。这时，我们可以通过Block的`params`来指定模型参数。在下面使用`Sequential`类构造的多层感知机中，模型的第二隐藏层（`net[1]`）和第三隐藏层（`net[2]`）共享模型参数。

```{.python .input  n=15}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu'))
    # 通过params指定需要共享的模型参数。
    net.add(nn.Dense(4, activation='relu', params=net[1].params))
    net.add(nn.Dense(2))

net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())
```

同样，我们也可以在使用Block构造的多层感知机中，让模型的第二隐藏层（`hidden2`）和第三隐藏层（`hidden3`）共享模型参数。

```{.python .input}
class MLP_SHARE(nn.Block):
    def __init__(self, **kwargs):
        super(MLP_SHARE, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden1 = nn.Dense(4, activation='relu')
            self.hidden2 = nn.Dense(4, activation='relu')
            # 通过params指定需要共享的模型参数。
            self.hidden3 = nn.Dense(4, activation='relu',
                                    params=self.hidden2.params)
            self.output = nn.Dense(2)

    def forward(self, x):
        return self.output(self.hidden3(self.hidden2(self.hidden1(x))))

net = MLP_SHARE()
net.initialize()
net(x)
print(net.hidden2.weight.data())
print(net.hidden3.weight.data())
```

## 小结

* 我们可以很方便地访问、自定义和共享模型参数。

## 练习

* 在本节任何一个例子中，`net.collect_params()`和`net.params`的返回有什么不同？
* 查阅[MXNet文档](https://mxnet.incubator.apache.org/api/python/model.html#initializer-api-reference)，了解不同的参数初始化方式。
* 构造一个含共享参数层的多层感知机并训练。观察每一层的模型参数。
* 如果两个层共用一个参数，求梯度的时候会发生什么？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
