# 模型参数的延后初始化

如果你注意到了上节练习，你会发现在`net.initialize()`后和`net(x)`前模型参数的形状都是空。直觉上`initialize`会完成了所有参数初始化过程，然而Gluon中这是不一定的。我们这里详细讨论这个话题。

## 延后的初始化

注意到前面使用Gluon的章节里，我们在创建全连接层时都没有指定输入大小。例如在一直使用的多层感知机例子里，我们创建了输出大小为256的隐藏层。但是当在调用`initialize`函数的时候，我们并不知道这个层的参数到底有多大，因为它的输入大小仍然是未知。只有在当我们将形状是`(2,20)`的`x`输入进网络时，我们这时候才知道这一层的参数大小应该是`(256,20)`。所以这个时候我们才能真正开始初始化参数。

让我们使用上节定义的MyInit类来清楚的演示这一个过程。下面我们创建多层感知机，然后使用MyInit实例来进行初始化。

```{.python .input  n=22}
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了。

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))

net.initialize(init=MyInit())
```

注意到MyInit在调用时会打印信息，但当前我们并没有看到相应的日志。下面我们执行前向计算。

```{.python .input  n=25}
x = nd.random.uniform(shape=(2,20))
y = net(x)
```

这时候系统根据输入`x`的形状自动推测数所有层参数形状，例如隐藏层大小是`(256，20)`，并创建参数。之后调用MyInit实例来进行初始方法，然后再进行前向计算。

当然，这个初始化只会在第一次执行被调用。之后我们再运行`net(x)`时则不会重新初始化，即我们不会再次看到MyInit实例的输出。

```{.python .input}
y = net(x)
```

我们将这个系统将真正的参数初始化延后到获得了足够信息到时候称之为延后初始化。它可以让模型创建更加简单，因为我们只需要定义每个层的输出大小，而不用去推测它们的的输入大小。这个对于之后将介绍的多达数十甚至数百层的网络尤其有用。

当然正如本节开头提到到那样，延后初始化也可能会造成一定的困解。在调用第一次前向计算之前我们无法直接操作模型参数。例如无法使用`data`和`set_data`函数来获取和改写参数。所以经常我们会额外调用一次`net(x)`来是的参数被真正的初始化。

## 避免延后初始化

当系统在调用`initialize`函数时能够知道所有参数形状，那么延后初始化就不会发生。我们这里给两个这样的情况。

第一个是模型已经被初始化过，而且我们要对模型进行重新初始化时。因为我们知道参数大小不会变，所以能够立即进行重新初始化。

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

第二种情况是我们在创建层到时候指定了每个层的输入大小，使得系统不需要额外的信息来推测参数形状。下例中我们通过`in_units`来指定每个全连接层的输入大小，使得初始化能够立即进行。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## 小结

* 在调用`initialize`函数时，系统可能将真正的初始化延后到后面，例如前向计算时，来执行。这样到主要好处是让模型定义可以更加简单。

## 练习

* 如果在下一次`net(x)`前改变`x`形状，包括批量大小和特征大小，会发生什么？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6320)

![](../img/qr_deferred-init.svg)
