# 模型参数的访问、初始化和共享

在[“线性回归的简洁实现”](../chapter_deep-learning-basics/linear-regression-gluon.md)一节中，我们通过`init`模块来初始化模型的全部参数。我们也介绍了访问模型参数的简单方法。本节将深入讲解如何访问和初始化模型参数，以及如何在多个层之间共享同一份模型参数。

我们先定义一个与上一节中相同的含单隐藏层的多层感知机。我们依然使用默认方式初始化它的参数，并做一次前向计算。与之前不同的是，在这里我们从MXNet中导入了`init`模块，它包含了多种模型初始化方法。

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算
```

## 访问模型参数

对于使用`Sequential`类构造的神经网络，我们可以通过方括号`[]`来访问网络的任一层。回忆一下上一节中提到的`Sequential`类与`Block`类的继承关系。对于`Sequential`实例中含模型参数的层，我们可以通过`Block`类的`params`属性来访问该层包含的所有参数。下面，访问多层感知机`net`中隐藏层的所有参数。索引0表示隐藏层为`Sequential`实例最先添加的层。

```{.python .input  n=2}
net[0].params, type(net[0].params)
```

可以看到，我们得到了一个由参数名称映射到参数实例的字典（类型为`ParameterDict`类）。其中权重参数的名称为`dense0_weight`，它由`net[0]`的名称（`dense0_`）和自己的变量名（`weight`）组成。而且可以看到，该参数的形状为(256, 20)，且数据类型为32位浮点数（`float32`）。为了访问特定参数，我们既可以通过名字来访问字典里的元素，也可以直接使用它的变量名。下面两种方法是等价的，但通常后者的代码可读性更好。

```{.python .input  n=3}
net[0].params['dense0_weight'], net[0].weight
```

Gluon里参数类型为`Parameter`类，它包含参数和梯度的数值，可以分别通过`data`函数和`grad`函数来访问。因为我们随机初始化了权重，所以权重参数是一个由随机数组成的形状为(256, 20)的`NDArray`。

```{.python .input  n=4}
net[0].weight.data()
```

权重梯度的形状和权重的形状一样。因为我们还没有进行反向传播计算，所以梯度的值全为0。

```{.python .input  n=5}
net[0].weight.grad()
```

类似地，我们可以访问其他层的参数，如输出层的偏差值。

```{.python .input  n=6}
net[1].bias.data()
```

最后，我们可以使用`collect_params`函数来获取`net`变量所有嵌套（例如通过`add`函数嵌套）的层所包含的所有参数。它返回的同样是一个由参数名称到参数实例的字典。

```{.python .input  n=7}
net.collect_params()
```

这个函数可以通过正则表达式来匹配参数名，从而筛选需要的参数。

```{.python .input  n=8}
net.collect_params('.*weight')
```

## 初始化模型参数

我们在[“数值稳定性和模型初始化”](../chapter_deep-learning-basics/numerical-stability-and-init.md)一节中描述了模型的默认初始化方法：权重参数元素为[-0.07, 0.07]之间均匀分布的随机数，偏差参数则全为0。但我们经常需要使用其他方法来初始化权重。MXNet的`init`模块里提供了多种预设的初始化方法。在下面的例子中，我们将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。

```{.python .input  n=9}
# 非首次对模型初始化需要指定force_reinit为真
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

下面使用常数来初始化权重参数。

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

如果只想对某个特定参数进行初始化，我们可以调用`Parameter`类的`initialize`函数，它与`Block`类提供的`initialize`函数的使用方法一致。下例中我们对隐藏层的权重使用Xavier随机初始化方法。

```{.python .input  n=11}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[0].weight.data()[0]
```

## 自定义初始化方法

有时候我们需要的初始化方法并没有在`init`模块中提供。这时，可以实现一个`Initializer`类的子类，从而能够像使用其他初始化方法那样使用它。通常，我们只需要实现`_init_weight`这个函数，并将其传入的`NDArray`修改成初始化的结果。在下面的例子里，我们令权重有一半概率初始化为0，有另一半概率初始化为$[-10,-5]$和$[5,10]$两个区间里均匀分布的随机数。

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

此外，我们还可以通过`Parameter`类的`set_data`函数来直接改写模型参数。例如，在下例中我们将隐藏层参数在现有的基础上加1。

```{.python .input  n=13}
net[0].weight.set_data(net[0].weight.data() + 1)
net[0].weight.data()[0]
```

## 共享模型参数

在有些情况下，我们希望在多个层之间共享模型参数。[“模型构造”](model-construction.md)一节介绍了如何在`Block`类的`forward`函数里多次调用同一个层来计算。这里再介绍另外一种方法，它在构造层的时候指定使用特定的参数。如果不同层使用同一份参数，那么它们在前向计算和反向传播时都会共享相同的参数。在下面的例子里，我们让模型的第二隐藏层（`shared`变量）和第三隐藏层共享模型参数。

```{.python .input  n=14}
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
net(X)

net[1].weight.data()[0] == net[2].weight.data()[0]
```

我们在构造第三隐藏层时通过`params`来指定它使用第二隐藏层的参数。因为模型参数里包含了梯度，所以在反向传播计算时，第二隐藏层和第三隐藏层的梯度都会被累加在`shared.params.grad()`里。


## 小结

* 有多种方法来访问、初始化和共享模型参数。
* 可以自定义初始化方法。


## 练习

* 查阅有关`init`模块的MXNet文档，了解不同的参数初始化方法。
* 尝试在`net.initialize()`后、`net(X)`前访问模型参数，观察模型参数的形状。
* 构造一个含共享参数层的多层感知机并训练。在训练过程中，观察每一层的模型参数和梯度。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
