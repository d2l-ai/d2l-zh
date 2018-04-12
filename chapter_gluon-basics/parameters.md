# 模型参数

为了引出本节的话题，让我们先使用`nn.Sequential`定义一个多层感知机。

```{.python .input  n=46}
import sys
from mxnet import init, gluon, nd
from mxnet.gluon import nn

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

```{.python .input  n=33}
x = nd.random.uniform(shape=(3, 5))
try:
    net = MLP()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

作如下修改之后，模型便计算成功。

```{.python .input  n=34}
net.initialize()
net(x)
```

这里添加的`net.initialize()`对模型参数做了初始化。模型参数是深度学习计算中的重要组成部分。本节中，我们将介绍如何访问、初始化和共享模型参数。

## 访问模型参数

在`Gluon`中，模型参数的类型是`Parameter`。下面让我们创建一个名字叫“good_param”、形状为$2 \times 3$的模型参数。在默认的初始化中，模型参数中的每一个元素是一个在`[-0.07, 0.07]`之间均匀分布的随机数。相应地，该模型参数还有一个形状为$2 \times 3$的梯度，初始值为0。

```{.python .input}
my_param = gluon.Parameter("good_param", shape=(2, 3))
my_param.initialize()
print('data: ', my_param.data(), '\ngrad: ', my_param.grad(),
      '\nname: ', my_param.name)
```

接下来，让我们访问本节开头定义的多层感知机`net`中隐藏层`hidden`的模型参数：权重`weight`和偏差`bias`。它们的类型也都是`Parameter`。我们可以看到它们的名字、形状和数据类型。

```{.python .input  n=35}
w = net.hidden.weight
b = net.hidden.bias
print('hidden layer name: ', net.hidden.name, '\nweight: ', w, '\nbias: ', b)
```

我们同样可以访问这两个参数的值和梯度。

```{.python .input  n=43}
print('weight:', w.data(), '\nweight grad:', w.grad(), '\nbias:', b.data(),
      '\nbias grad:', b.grad())
```

另外，我们也可以通过`collect_params`来访问`nn.Block`里的所有参数（包括所有的子`nn.Block`）。它会返回一个名字到对应`Parameter`的字典。在这个字典中，我们既可以用`[]`（需要前缀），又可以用`get()`（不需要前缀）来访问模型参数。

```{.python .input  n=7}
params = net.collect_params()
print(params)
print(params['mlp0_dense0_bias'].data())
print(params.get('dense0_bias').data())
```

## 使用不同的初始函数来初始化

我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布

```{.python .input}
params = net.collect_params()
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net.hidden.weight.data(), net.hidden.bias.data())
```

看得更加清楚点：

```{.python .input}
params.initialize(init=init.One(), force_reinit=True)
print(net.hidden.weight.data(), net.hidden.bias.data())
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input}
net = MLP()
net.collect_params()
```

然后我们初始化

```{.python .input}
net.initialize()
net.collect_params()
```

你会看到我们形状并没有发生变化，这是因为我们仍然不能确定权重形状。真正的初始化发生在我们看到数据时。

```{.python .input}
net(x)
net.collect_params()
```

这时候我们看到shape里面的0被填上正确的值了。



## 自定义初始化方法

下面我们自定义一个初始化方法。它通过重载`_init_weight`来实现不同的初始化方法。（注意到Gluon里面`bias`都是默认初始化成0）

```{.python .input}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)

net = MLP()
net.initialize(MyInit())
net(x)
net.hidden.weight.data()
```

当然我们也可以通过`Parameter.set_data`来直接改写权重。注意到由于有延后初始化，所以我们通常可以通过调用一次`net(x)`来确定权重的形状先。

```{.python .input}
net = MLP()
net.initialize()
net(x)

print('default weight:', net.output.weight.data())

w = net.output.weight
w.set_data(nd.ones(w.shape))

print('init to all 1s:', net.output.weight.data())
```

## 共享模型参数

在有些情况下，我们希望模型的多个层之间共享模型参数。这时，我们可以通过`nn.Block`的`params`来指定模型参数。举个例子：

```{.python .input}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[1].params))
    net.add(nn.Dense(2))

net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())
```

## 小结

* 我们可以很灵活地访问和修改模型参数。

## 练习

* 研究下`net.collect_params()`返回的是什么？`net.params`呢？
* 如何对每个层使用不同的初始化函数
* 如果两个层共用一个参数，那么求梯度的时候会发生什么？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
