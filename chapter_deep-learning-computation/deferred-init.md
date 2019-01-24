# 模型参数的延后初始化

如果做了上一节练习，你会发现模型`net`在调用初始化函数`initialize`之后、在做前向计算`net(X)`之前时，权重参数的形状中出现了0。虽然直觉上`initialize`完成了所有参数初始化过程，然而这在Gluon中却是不一定的。我们在本节中详细讨论这个话题。


## 延后初始化

也许读者早就注意到了，在之前使用Gluon创建的全连接层都没有指定输入个数。例如，在上一节使用的多层感知机`net`里，我们创建的隐藏层仅仅指定了输出大小为256。当调用`initialize`函数时，由于隐藏层输入个数依然未知，系统也无法得知该层权重参数的形状。只有在当我们将形状是(2, 20)的输入`X`传进网络做前向计算`net(X)`时，系统才推断出该层的权重参数形状为(256, 20)。因此，这时候我们才能真正开始初始化参数。

让我们使用上一节中定义的`MyInit`类来演示这一过程。我们创建多层感知机，并使用`MyInit`实例来初始化模型参数。

```{.python .input  n=22}
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())
```

注意，虽然`MyInit`被调用时会打印模型参数的相关信息，但上面的`initialize`函数执行完并未打印任何信息。由此可见，调用`initialize`函数时并没有真正初始化参数。下面我们定义输入并执行一次前向计算。

```{.python .input  n=25}
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
```

这时候，有关模型参数的信息被打印出来。在根据输入`X`做前向计算时，系统能够根据输入的形状自动推断出所有层的权重参数的形状。系统在创建这些参数之后，调用`MyInit`实例对它们进行初始化，然后才进行前向计算。

当然，这个初始化只会在第一次前向计算时被调用。之后我们再运行前向计算`net(X)`时则不会重新初始化，因此不会再次产生`MyInit`实例的输出。

```{.python .input}
Y = net(X)
```

系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化（deferred initialization）。它可以让模型的创建更加简单：只需要定义每个层的输出大小，而不用人工推测它们的输入个数。这对于之后将介绍的定义多达数十甚至数百层的网络来说尤其方便。

然而，任何事物都有两面性。正如本节开头提到的那样，延后初始化也可能会带来一定的困惑。在第一次前向计算之前，我们无法直接操作模型参数，例如无法使用`data`函数和`set_data`函数来获取和修改参数。因此，我们经常会额外做一次前向计算来迫使参数被真正地初始化。

## 避免延后初始化

如果系统在调用`initialize`函数时能够知道所有参数的形状，那么延后初始化就不会发生。我们在这里分别介绍两种这样的情况。

第一种情况是我们要对已初始化的模型重新初始化时。因为参数形状不会发生变化，所以系统能够立即进行重新初始化。

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

第二种情况是我们在创建层的时候指定了它的输入个数，使系统不需要额外的信息来推测参数形状。下例中我们通过`in_units`来指定每个全连接层的输入个数，使初始化能够在`initialize`函数被调用时立即发生。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## 小结

* 系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化。
* 延后初始化的主要好处是让模型构造更加简单。例如，我们无须人工推测每个层的输入个数。
* 也可以避免延后初始化。


## 练习

* 如果在下一次前向计算`net(X)`前改变输入`X`的形状，包括批量大小和输入个数，会发生什么？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6320)

![](../img/qr_deferred-init.svg)
