# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=46}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=33}
import sys
try:
    net = get_net()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

正确的打开方式是这样

```{.python .input  n=34}
net.initialize()
net(x)
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：

```{.python .input  n=35}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=43}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

我们也可以通过`collect_params`来访问Block里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应Parameter的dict。既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=7}
params = net.collect_params()
print(params)
print(params['sequential0_dense0_bias'].data())
print(params.get('dense0_weight').data())
```

## 使用不同的初始函数来初始化

我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布

```{.python .input}
from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

看得更加清楚点：

```{.python .input}
params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input}
net = get_net()
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

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2))
```

初始化然后打印

```{.python .input}
net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())
```

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

net = get_net()
net.initialize(MyInit())
net(x)
net[0].weight.data()
```

当然我们也可以通过`Parameter.set_data`来直接改写权重。注意到由于有延后初始化，所以我们通常可以通过调用一次`net(x)`来确定权重的形状先。

```{.python .input}
net = get_net()
net.initialize()
net(x)

print('default weight:', net[1].weight.data())

w = net[1].weight
w.set_data(nd.ones(w.shape))

print('init to all 1s:', net[1].weight.data())
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)
