# 模型构造

让我们回顾一下在[“多层感知机的Gluon实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中含单隐藏层的多层感知机的实现方法。我们首先构造Sequential实例，然后依次添加两个全连接层。其中第一层的输出大小为256，即隐藏层单元个数是256；第二层的输出大小为10，即输出层单元个数是10。我们在上一章的其他小节中也使用了Sequential类构造模型。这里我们介绍另外一种基于Block类的模型构造方法：它让模型构造更加灵活。


## 继承Block类来构造模型

Block类是`nn`模块里提供的一个模型构造类，我们可以继承它来定义我们想要的模型。下面继承Block类构造本节开头提到的多层感知机。这里定义的`MLP`类重载了Block类的`__init__`和`forward`函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # 声明带有模型参数的层，这里我们声明了两个全连接层。
    def __init__(self, **kwargs):
        # 调用 MLP 父类 Block 的构造函数来进行必要的初始化。这样在构造实例时还可以指定
        # 其他函数参数，例如后面章节将介绍的模型参数 params。
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层。
        self.output = nn.Dense(10)  # 输出层。

    # 定义模型的前向计算，即如何根据输入 x 计算返回所需要的模型输出。
    def forward(self, x):
        return self.output(self.hidden(x))
```

以上的`MLP`类中无需定义反向传播函数。系统将通过自动求梯度，从而自动生成反向传播所需要的`backward`函数。

我们可以实例化`MLP`类得到模型变量`net`。下面代码初始化`net`并传入输入数据`x`做一次前向计算。其中，`net(x)`会调用`MLP`继承自Block类的`__call__`函数，这个函数将调用`MLP`类定义的`forward`函数来完成前向计算。

```{.python .input  n=2}
x = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
net(x)
```

注意到我们并没有将Block类命名为层（Layer）或者模型（Model）之类的名字，这是因为该类是一个可供自由组建的部件。它的子类既可以是一个层（例如Gluon提供的`Dense`类），又可以是一个模型（例如这里定义的MLP类），或者是模型的一个部分。我们下面通过两个例子来展示它的灵活性。

## Sequential类继承自Block类

我们刚刚提到，Block类是一个通用的部件。事实上，Sequential类继承自Block类。当模型的前向计算为简单串联各个层的计算时，我们可以通过更加简单的方式定义模型。这正是Sequential类的目的：它提供`add`函数来逐一添加串联的Block子类实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。

下面我们实现一个跟Sequential类有相同功能的`MySequential`类。这或许可以帮助你更加清晰地理解Sequential类的工作机制。

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block 是一个 Block 子类实例，假设它有一个独一无二的名字。我们将它保存在 Block
        # 类的成员变量 _children 里，其类型是 OrderedDict。当 MySequential 实例调用
        # initialize 函数时，系统会自动对 _children 里所有成员初始化。
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict 保证会按照成员添加时的顺序遍历成员。
        for block in self._children.values():
            x = block(x)
        return x
```

我们用MySequential类来实现前面描述的`MLP`类，并使用随机初始化的模型做一次前向计算。

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

可以观察到这里`MySequential`类的使用跟[“多层感知机的Gluon实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中Sequential类的使用没什么区别。


## 构造复杂的模型

虽然Sequential类可以使得模型构造更加简单，且不需要定义`forward`函数，但直接继承Block类可以极大地拓展模型构造的灵活性。下面我们构造一个稍微复杂点的网络`FancyMLP`。在这个网络中，我们通过`get_constant`函数创建训练中不被迭代的参数，即常数参数。在前向计算中，除了使用创建的常数参数外，我们还使用NDArray的函数和Python的控制流，并多次调用相同的层。

```{.python .input  n=5}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用 get_constant 创建的随机权重参数不会在训练中被迭代（即常数参数）。
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及 NDArray 的 relu 和 dot 函数。
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 重用全连接层。等价于两个全连接层共享参数。
        x = self.dense(x)
        # 控制流，这里我们需要调用 asscalar 来返回标量进行比较。
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```

在这个`FancyMLP`模型中，我们使用了常数权重`rand_weight`（注意它不是模型参数）、做了矩阵乘法操作（`nd.dot`）并重复使用了相同的`Dense`层。下面我们来测试该模型的随机初始化和前向计算。

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

由于`FancyMLP`和Sequential类都是Block类的子类，我们可以嵌套调用它们。

```{.python .input  n=7}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())

net.initialize()
net(x)
```

## 小结

* 我们可以通过继承Block类来构造模型。
* Sequential类继承自Block类。
* 虽然Sequential类可以使得模型构造更加简单，但直接继承Block类可以极大地拓展模型构造的灵活性。


## 练习

* 如果不在`MLP`类的`__init__`函数里调用父类的`__init__`函数，会出现什么样的错误信息？
* 如果去掉`FancyMLP`类里面的`asscalar`函数，会有什么问题？
* 如果将`NestMLP`类中通过Sequential实例定义的`self.net`改为`self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]`，会有什么问题？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_model-construction.svg)
