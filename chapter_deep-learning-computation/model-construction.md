# 模型构造

回忆在[“多层感知机的Gluon实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中我们是如何实现一个单隐藏层感知机。我们首先构造Sequential实例，然后依次添加两个全连接层。其中第一层的输出大小为256，即隐藏层单元个数是256；第二层的输出大小为10，即输出层单元个数是10。这个简单例子已经包含了深度学习模型计算的方方面面，接下来的小节我们将围绕这个例子展开。

我们之前都是用了Sequential类来构造模型。这里我们另外一种基于Block类的模型构造方法，它让构造模型更加灵活，也将让你能更好的理解Sequential的运行机制。

## 继承Block类来构造模型

Block类是`gluon.nn`里提供的一个模型构造类，我们可以继承它来定义我们想要的模型。例如，我们在这里构造一个同前提到的相同的多层感知机。这里定义的MLP类重载了Block类的两个函数：`__init__`和`forward`.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # 声明带有模型参数的层，这里我们声明了两个全链接层。
    def __init__(self, **kwargs):
        # 调用 MLP 父类 Block 的构造函数来进行必要的初始化。这样在构造实例时还可以指定
        # 其他函数参数，例如下下一节将介绍的模型参数 params。
        super(MLP, self).__init__(**kwargs)
        # 隐藏层。
        self.hidden = nn.Dense(256, activation='relu')
        # 输出层。
        self.output = nn.Dense(10)
    # 定义模型的前向计算，即如何根据输出计算输出。
    def forward(self, x):
        return self.output(self.hidden(x))
```

我们可以实例化MLP类得到`net`，其使用跟[“多层感知机的Gluon实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中通过Sequential类构造的`net`一致。下面代码初始化`net`并传入输入数据`x`做一次前向计算。

```{.python .input  n=2}
x = nd.random.uniform(shape=(2,20))
net = MLP()
net.initialize()
net(x)
```

其中，`net(x)`会调用了MLP继承至Block的`__call__`函数，这个函数将调用MLP定义的`forward`函数来完成前向计算。

我们无需在这里定义反向传播函数，系统将通过自动求导，参考[“自动求梯度”](../chapter_prerequisite/autograd.md)一节，来自动生成`backward`函数。

注意到我们不是将Block叫做层或者模型之类的名字，这是因为它是一个可以自由组建的部件。它的子类既可以一个层，例如Gluon提供的Dense类，也可以是一个模型，我们定义的MLP类，或者是模型的一个部分，例如我们会在之后介绍的ResNet的残差块。我们下面通过两个例子说明它。

## Sequential类继承自Block类

当模型的前向计算就是简单串行计算模型里面各个层的时候，我们可以将模型定义变得更加简单，这个就是Sequential类的目的，它通过`add`函数来添加Block子类实例，前向计算时就是将添加的实例逐一运行。下面我们实现一个跟Sequential类有相同功能的类，这样你可以看的更加清楚它的运行机制。

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block 是一个 Block 子类实例，假设它有一个独一无二的名字。我们将它保存在
        # Block 类的成员变量 _children 里，其类型是 OrderedDict. 当调用
        # initialize 函数时，系统会自动对 _children 里面所有成员初始化。
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict 保证会按照插入时的顺序遍历元素。
        for block in self._children.values():
            x = block(x)
        return x
```

我们用MySequential类来实现前面的MLP类：

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

你会发现这里MySequential类的使用跟[“多层感知机的Gluon实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中Sequential类使用一致。

## 构造复杂的模型

虽然Sequential类可以使得模型构造更加简单，不需要定义`forward`函数，但直接继承Block类可以极大的拓展灵活性。下面我们构造一个稍微复杂点的网络。在这个网络中，我们通过`get_constant`函数创建训练中不被迭代的参数，即常数参数。在前向计算中，除了使用创建的常数参数外，我们还使用NDArray的函数和Python的控制流，并多次调用同一层。

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

在这个`FancyMLP`模型中，我们使用了常数权重`rand_weight`（注意它不是模型参数）、做了矩阵乘法操作（`nd.dot`）并重复使用了相同的`Dense`层。测试一下：

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

由于FancyMLP和Sequential都是Block的子类，我们可以嵌套调用他们。

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

* 我们可以通过继承Block类来构造复杂的模型。
* Sequential是Block的子类。


## 练习

* 在FancyMLP类里我们重用了`dense`，这样对输入形状有了一定要求，尝试改变下输入数据形状试试。
* 如果我们去掉FancyMLP里面的`asscalar`会有什么问题？
* 在NestMLP里假设我们改成 `self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]`，而不是用Sequential类来构造，会有什么问题？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_model-construction.svg)
