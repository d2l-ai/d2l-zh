# 层和块
:label:`sec_model_construction`

当我们第一次引入神经网络时，我们专注于具有单个输出的线性模型。在这里，整个模型只包含一个神经元。请注意，单个神经元 (i) 需要一组输入；(ii) 生成相应的标量输出；(iii) 具有一组相关参数，可以更新以优化某些目标函数。然后，一旦我们开始思考具有多个输出的网络，我们利用矢量化算术来表征整个神经元层。就像单个神经元一样，层 (i) 采用一组输入，(ii) 生成相应的输出，(iii) 由一组可调谐的参数描述。当我们处理 softmax 回归时，单个层本身就是模型。然而，即使我们随后引入了 MLP，我们仍然可以认为该模型保留了同样的基本结构。

有趣的是，对于 MLP 来说，整个模型及其组成层都共享这种结构。整个模型采用原始输入（特征），生成输出（预测），并拥有参数（来自所有组成层的组合参数）。同样，每个单独的层收集输入（由上一层提供）生成输出（输入到后续层），并具有一组可调谐参数，根据后续层向后流的信号进行更新。

虽然你可能会认为神经元、层和模型为我们提供了足够的抽象来开展业务，但事实证明，我们经常发现谈论比单个层大但小于整个模型的组件很方便。例如，在计算机视觉中广受欢迎的 Resnet-152 架构拥有数百层。这些图层由 * 层组 * 的重复模式组成。一次实施这样一个网络可能会变得乏味。这种担心不仅仅是假设的 — 这种设计模式在实践中很常见。上述 RESNet 架构赢得了 2015 年 ImageNet 和 COCO 计算机视觉比赛的识别和检测 :cite:`He.Zhang.Ren.ea.2016`，并且仍然是许多视觉任务的首选架构。在其他领域（包括自然语言处理和语音）中，图层按各种重复模式排列的类似架构现在已经普遍存在。

为了实现这些复杂的网络，我们介绍了神经网络 *block * 的概念。块可以描述单个层、由多个层组成的组件或整个模型本身！使用块抽象的一个好处是它们可以被组合成较大的工件，通常是递归的。这一点在 :numref:`fig_blocks` 中得到了说明。通过定义代码来根据需要生成任意复杂的块，我们可以编写出奇的紧凑代码，并且仍然实现复杂的神经网络。

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

从编程的角度来看，块由 * 类 * 表示。它的任何子类都必须定义一个将其输入转换为输出的正向传播函数，并且必须存储任何必要的参数。请注意，某些块根本不需要任何参数。最后，一个块必须具有反向传播函数，以便计算梯度。幸运的是，在定义我们自己的块时，由于自动分化（:numref:`sec_autograd` 引入）提供的一些幕后魔法，我们只需要担心参数和前向传播函数。

首先，我们重新讨论我们用来实现 MLP 的代码 (:numref:`sec_mlp_concise`)。下面的代码生成一个具有 256 个单元和 RelU 激活的完全连接隐藏层的网络，然后是具有 10 个单元的完全连接输出层（无激活功能）。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
在此示例中，我们通过实例化 `nn.Sequential` 来构建模型，将返回的对象分配给 `net` 变量。接下来，我们反复调用其 `add` 函数，按照应执行的顺序追加图层。简而言之，`nn.Sequential` 定义了一种特殊类型的 `Block`，这种类提出了一个块的葡萄糖。它保留了一份有序的组成部分 `Block` 的清单。`add` 函数简单地便于将每个连续的 `Block` 添加到列表中。请注意，每个图层都是 `Dense` 类的一个实例，该类本身就是 `Block` 的子类。正向传播 (`forward`) 函数也非常简单：它将列表中的每个 `Block` 链接在一起，将每个函数的输出作为输入传递给下一个。请注意，到目前为止，我们一直通过构造 `net(X)` 调用我们的模型来获取它们的输出。这实际上只是 `net.forward(X)` 的速记，这是通过 `Block` 类的 `__call__` 函数实现的一个光滑的 Python 技巧。
:end_tab:

:begin_tab:`pytorch`
在这个例子中，我们通过实例化 `nn.Sequential` 来构建模型，图层的执行顺序是作为参数传递的。总之，`nn.Sequential` 定义了一种特殊类型的 `Module`，这是在 Pytorch中呈现一个块的类。它维护了组成 `Module` 的有序列表。请注意，两个完全连接的图层中的每个图层都是 `Linear` 类的一个实例，该类本身就是 `Module` 的子类。正向传播 (`forward`) 函数也非常简单：它将列表中的每个块链接在一起，将每个块的输出作为输入传递给下一个。请注意，到目前为止，我们一直通过构造 `net(X)` 调用我们的模型来获取它们的输出。这实际上只是 `net.forward(X)` 的速记，这是通过块类的 `__call__` 函数实现的一个光滑的 Python 技巧。
:end_tab:

:begin_tab:`tensorflow`
在这个例子中，我们通过实例化 `keras.models.Sequential` 来构建模型，图层的执行顺序是作为参数传递的。总之，`Sequential` 定义了一种特殊类型的 `keras.Model`，这是在克拉斯提出一个块的类。它保留了一份有序的组成部分 `Model` 的清单。请注意，两个完全连接的图层中的每个图层都是 `Dense` 类的一个实例，该类本身就是 `Model` 的子类。正向传播 (`call`) 函数也非常简单：它将列表中的每个块链接在一起，将每个块的输出作为输入传递给下一个。请注意，到目前为止，我们一直通过构造 `net(X)` 调用我们的模型来获取它们的输出。这实际上只是 `net.call(X)` 的速记，这是通过块类的 `__call__` 函数实现的一个光滑的 Python 技巧。
:end_tab:

## 自定义块

也许最简单的方法来开发关于块如何工作的直觉是自己实现一个。在我们实现自己的自定义块之前，我们简要地总结了每个块必须提供的基本功能：

1. 将输入数据作为其正向传播函数的参数。
1. 通过让正向传播函数返回一个值来生成输出。请注意，输出的形状可能与输入不同。例如，上面模型中的第一个完全连接的图层会摄取任意尺寸的输入，但返回尺寸 256 的输出。
1. 计算其输出相对于其输入的梯度，可通过其反向传播函数访问该输入。通常，这会自动发生。
1. 存储并提供对执行正向传播计算所必需的参数的访问权限。
1. 根据需要初始化模型参数。

在下面的代码段中，我们从头开始编写一个块，对应于 MLP，其中包含 256 个隐藏单位的隐藏图层和 10 维输出图层。请注意，下面的 `MLP` 类继承代表块的类。我们将严重依赖父类的函数，只提供我们自己的构造函数（Python 中的 `__init__` 函数）和正向传播函数。

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

让我们首先关注正向传播函数。请注意，它将 `X` 作为输入，计算应用激活函数的隐藏表示，并输出其日志。在此 `MLP` 实现中，两个层都是实例变量。要了解为什么这是合理的，请想象一下实例化两个 MLP（`net1` 和 `net2`），然后根据不同的数据对它们进行培训。当然，我们希望它们代表两种不同的学习模式。

我们在构造函数中实例化 MLP 的图层，然后在每次调用前向传播函数时调用这些图层。请注意一些关键细节。首先，我们的自定义 `__init__` 函数通过 `super().__init__()` 调用父类的 `__init__` 函数，从而避免了重新陈述适用于大多数块的样板代码的痛苦。然后，我们实例化我们的两个完全连接的图层，将它们分配给 `self.hidden` 和 `self.out`。请注意，除非我们实现一个新的运算符，否则我们不必担心反向传播函数或参数初始化。系统将自动生成这些函数。让我们试试一下。

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

块抽象的一个关键优点是它的多功能性。我们可以对块进行子分类，以创建图层（如完全连接的图层类）、整个模型（如上面的 `MLP` 类）或中等复杂度的各种组件。我们在以下章节中利用了这种多功能性，例如在处理卷积神经网络时。

## 顺序块

我们现在可以仔细了解 `Sequential` 类的工作原理。回想一下，`Sequential` 的设计用于将其他块连接在一起。要构建我们自己的简化 `MySequential`，我们只需要定义两个关键函数：
1. 一个将块逐个附加到列表的函数。
2. 一个向前传播函数，用于通过块链传递输入，其顺序与其附加顺序相同。

以下 `MySequential` 类提供了与默认 `Sequential` 类相同的功能。

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # Here, `block` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` 函数为有序的字典 `_children` 添加了一个单一的块。你可能会想知道为什么每个 Gluon `Block` 都有一个 `_children` 属性，以及为什么我们使用它而不是自己定义一个 Python 列表。简而言之，`_children` 的主要优势是，在我们的块的参数初始化过程中，Gluon 知道查看 `_children` 字典内部以找到其参数也需要初始化的子块。
:end_tab:

:begin_tab:`pytorch`
在 `__init__` 方法中，我们将每个块逐个添加到有序字典 `_modules` 中。你可能会想知道为什么每个 `Module` 都拥有一个 `_modules` 属性，以及为什么我们使用它而不是自己定义一个 Python 列表。简而言之，`_modules` 的主要优势是在我们块的参数初始化过程中，系统知道在 `_modules` 字典中查找其参数也需要初始化的子块。
:end_tab:

当我们 `MySequential` 的正向传播函数被调用时，每个添加的块都按照添加的顺序执行。我们现在可以使用我们的 `MySequential` 类重新实现 MLP。

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

请注意，`MySequential` 的这种使用与我们之前为 `Sequential` 类编写的代码相同（如 :numref:`sec_mlp_concise` 中所述）。

## 在正向传播函数中执行代码

`Sequential` 类使模型构建变得简单，使我们能够组装新的架构，而无需定义自己的类。然而，并非所有的体系结构都是简单的菊花链。当需要更大的灵活性时，我们希望定义自己的块。例如，我们可能希望在正向传播函数中执行 Python 的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。

您可能已经注意到，到目前为止，我们网络中的所有操作都对我们网络的激活及其参数进行了操作。但是，有时候，我们可能希望包含既不是前面图层结果的术语，也不是可更新参数的结果。我们称这些 * 常量参数 *。例如，我们想要一个计算函数 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ 的图层，其中 $\mathbf{x}$ 是输入，$\mathbf{w}$ 是我们的参数，$c$ 是一些在优化过程中未更新的指定常量。所以我们实现了一个 `FixedHiddenMLP` 类，如下所示。

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

在这个 `FixedHiddenMLP` 模型中，我们实现了一个隐藏层，其权重 (`self.rand_weight`) 在实例化时随机初始化并随后保持不变。此权重不是模型参数，因此它永远不会被反向传播更新。然后，网络将此 “固定” 图层的输出通过一个完全连接的图层传递。

请注意，在返回输出之前，我们的模型做了一些不寻常的事情。我们运行了一个周期循环，测试其 $L_1$ 范数大于 $1$ 的条件，并将我们的输出向量除以 $2$，直到它满足条件。最后，我们返回了 `X` 中条目的总和。据我们所知，没有标准的神经网络执行此操作。请注意，此特定操作在任何真实世界的任务中可能无用。我们的目的只是向您展示如何将任意代码集成到神经网络计算流程中。

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

我们可以混合和匹配各种组装方式在一起。在下面的示例中，我们以一些创造性的方式嵌套块。

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## 汇编

:begin_tab:`mxnet, tensorflow`
狂热的读者可能会开始担心其中一些操作的效率。毕竟，我们有很多字典查找，代码执行和许多其他 Pythonic 事情发生在应该是一个高性能的深度学习库中。蟒蛇的 [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) 的问题是众所周知的。在深度学习的背景下，我们担心我们非常快的 GPU 可能不得不等到一个小的 CPU 运行 Python 代码之后才能得到另一个作业运行。加速 Python 的最佳方法是完全避免它。
:end_tab:

:begin_tab:`mxnet`
Gluon 做到这一点的一种方法是允许
*杂交 *，这将在稍后描述。
在这里，Python 解释器在第一次调用时执行一个块。Gluon 运行时记录正在发生的事情以及下一次周围的短路调用 Python。在某些情况下，这可以大大加速事情，但是当控制流（如上所示）导致不同的分支通过网络时，需要小心。我们建议感兴趣的读者检查杂交部分 (:numref:`sec_hybridize`)，以了解在完成本章之后编译。
:end_tab:

## 摘要

* 图层是块。
* 许多图层可以组成一个块。
* 许多块可以组成一个块。
* 块可以包含代码。
* 块负责大量的管理工作，包括参数初始化和反向传播。
* 图层和块的连续串联由 `Sequential` 块处理。

## 练习

1. 如果将 `MySequential` 更改为将块存储在 Python 列表中，会发生什么样的问题？
1. 实现一个以两个块作为参数的块，比如 `net1` 和 `net2`，并返回正向传播中两个网络的连接输出。这也称为并行块。
1. 假定您想要连接同一网络的多个实例。实施一个工厂函数，该函数生成同一块的多个实例，并从中构建更大的网络。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
