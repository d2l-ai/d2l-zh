# 模型参数的访问、初始化和共享

在之前的小节里我们一直在使用默认的初始函数，`net.initialize()`，来初始化模型参数。我们也同时介绍过如何访问模型参数的简单方法。这一节我们将深入讲解模型参数的访问和初始化，以及如何在多个层之间共享同一份参数。

我们首先定义同前的多层感知机、初始化权重和计算前向结果。与之前不同的是，在这里我们从MXNet中导入了`init`这个包，它包含了多种模型初始化方法。

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2,20))
y = net(x)
```

## 访问模型参数

我们知道可以通过`[]`来访问Sequential类构造出来的网络的特定层。对于带有模型参数的层，我们可以通过Block类的`params`属性来得到它包含的所有参数。例如我们查看隐藏层的参数：

```{.python .input  n=2}
net[0].params
```

可以看到我们得到了一个由参数名称映射到参数实例的字典。第一个参数的名称为`dense0_weight`，它由`net[0]`的名称（`dense0_`）和自己的变量名（`weight`）组成。而且可以看到它参数的形状为`(256, 20)`，且数据类型为32位浮点数。

为了访问特定参数，我们既可以通过名字来访问字典里的元素，也可以直接使用它的变量名。下面两种方法是等价的，但通常后者的代码可读性更好。

```{.python .input  n=3}
net[0].params['dense0_weight'], net[0].weight
```

Gluon里参数类型为Parameter类，其包含参数权重和它对应的梯度，它们可以分别通过`data`和`grad`函数来访问。因为我们随机初始化了权重，所以它是一个由随机数组成的形状为`(256, 20)`的NDArray.

```{.python .input  n=4}
net[0].weight.data()
```

梯度的形状跟权重一样。但由于我们还没有进行反向传播计算，所以它的值全为0.

```{.python .input  n=5}
net[0].weight.grad()
```

类似我们可以访问其他的层的参数。例如输出层的偏差权重：

```{.python .input  n=6}
net[1].bias.data()
```

最后，我们可以`collect_params`函数来获取`net`实例所有嵌套（例如通过`add`函数嵌套）的层所包含的所有参数。它返回的同样是一个参数名称到参数实例的字典。

```{.python .input  n=11}
net.collect_params()
```

## 初始化模型参数

当使用默认的模型初始化，Gluon会将权重参数元素初始化为[-0.07, 0.07]之间均匀分布的随机数，偏差参数则全为0. 但经常我们需要使用其他的方法来初始话权重，MXNet的`init`模块里提供了多种预设的初始化方法。例如下面例子我们将权重参数初始化成均值为0，标准差为0.01的正态分布随机数。

```{.python .input  n=7}
# 非首次对模型初始化需要指定 force_reinit。
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

如果想只对某个特定参数进行初始化，我们可以调用`Paramter`类的`initialize`函数，它的使用跟Block类提供的一致。下例中我们对第一个隐藏层的权重使用Xavier初始化方法。

```{.python .input  n=8}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[0].weight.data()[0]
```

## 自定义初始化方法

有时候我们需要的初始化方法并没有在`init`模块中提供。这时，我们可以实现一个Initializer类的子类使得我们可以跟前面使用`init.Normal`那样使用它。通常，我们只需要实现`_init_weight`这个函数，将其传入的NDArray修改成需要的内容。下面例子里我们把权重初始化成`[-10,-5]`和`[5,10]`两个区间里均匀分布的随机数。

```{.python .input  n=9}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

此外，我们还可以通过`Parameter`类的`set_data`函数来直接改写模型参数。例如下例中我们将隐藏层参数在现有的基础上加1。

```{.python .input  n=10}
net[0].weight.set_data(net[0].weight.data() + 1)
net[0].weight.data()[0]
```

## 共享模型参数

在有些情况下，我们希望在多个层之间共享模型参数。我们在[“模型构造”](model-construction.md)这一节看到了如何在Block类里`forward`函数里多次调用同一个类来完成。这里将介绍另外一个方法，它在构造层的时候指定使用特定的参数。如果不同层使用同一份参数，那么它们不管是在前向计算还是反向传播时都会共享共同的参数。

在下面例子里，我们让模型的第二隐藏层和第三隐藏层共享模型参数。

```{.python .input}
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2,20))
net(x)

net[1].weight.data()[0] == net[2].weight.data()[0]
```

我们在构造第三隐藏层时通过`params`来指定它使用第二隐藏层的参数。由于模型参数里包含了梯度，所以在反向传播计算时，第二隐藏层和第三隐藏层的梯度都会被累加在`shared.params.grad()`里。


## 小结

* 我们有多种方法来访问、初始化和共享模型参数。

## 练习

* 查阅[MXNet文档](https://mxnet.incubator.apache.org/api/python/model.html#initializer-api-reference)，了解不同的参数初始化方式。
* 尝试在`net.initialize()`后和`net(x)`前访问模型参数，看看会发生什么。
* 构造一个含共享参数层的多层感知机并训练。观察每一层的模型参数和梯度计算。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
