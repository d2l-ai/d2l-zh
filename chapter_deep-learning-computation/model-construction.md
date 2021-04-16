# 层和块
:label:`sec_model_construction`

当我们第一次介绍神经网络时，我们关注的是具有单一输出的线性模型。在这里，整个模型只由一个神经元组成。注意，单个神经元（1）接受一些输入；（2）生成相应的标量输出；（3）具有一组相关 *参数*（parameters），这些参数可以更新以优化某些感兴趣的目标函数。然后，当我们开始考虑具有多个输出的网络，我们就利用矢量化算法来描述整层神经元。像单个神经元一样，层（1）接受一组输入，（2）生成相应的输出，（3）由一组可调整参数描述。当我们使用softmax回归时，一个单层本身就是模型。然而，即使我们随后引入了多层感知机，我们仍然可以认为该模型保留了上面所说的基本结构。

有趣的是，对于多层感知机而言，整个模型及其组成层都是这种结构。整个模型接受原始输入（特征），生成输出（预测），并包含一些参数（所有组成层的参数集合）。同样，每个单独的层接收输入（由前一层提供）生成输出（到下一层的输入），并且具有一组可调参数，这些参数根据从下一层反向传播的信号进行更新。

虽然你可能认为神经元、层和模型为我们的业务提供了足够的抽象，但事实证明，我们经常发现谈论比单个层大但比整个模型小的组件更方便。例如，在计算机视觉中广泛流行的ResNet-152结构就有数百层。这些层是 由*层组*的重复模式组成。一次只实现一层这样的网络会变得很乏味。这种问题不是我们幻想出来的，这种设计模式在实践中很常见。上面提到的ResNet结构赢得了2015年ImageNet和COCO计算机视觉比赛的识别和检测任务 :cite:`He.Zhang.Ren.ea.2016`，目前ResNet结构仍然是许多视觉任务的首选结构。在其他的领域，如自然语言处理和语音，层以各种重复模式排列的类似结构现在也是普遍存在。

为了实现这些复杂的网络，我们引入了神经网络*块*的概念。块可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件，这一过程通常是递归的。这一点在 :numref:`fig_blocks` 中进行了说明。通过定义代码来按需生成任意复杂度的块，我们可以通过简洁的代码实现复杂的神经网络。

![多个层被组合成块，形成更大的模型。](../img/blocks.svg)
:label:`fig_blocks`

从编程的角度来看，块由*类*（class）表示。它的任何子类都必须定义一个将其输入转换为输出的正向传播函数，并且必须存储任何必需的参数。注意，有些块不需要任何参数。最后，为了计算梯度，块必须具有反向传播函数。幸运的是，在定义我们自己的块时，由于自动微分（在 :numref:`sec_autograd` 中引入）提供了一些后端实现，我们只需要考虑正向传播函数和必需的参数。

(**首先，我们回顾一下多层感知机**)（ :numref:`sec_mlp_concise` ）的代码。下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接的隐藏层，然后是一个具有10个隐藏单元且不带激活函数的全连接的输出层。

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
在这个例子中，我们通过实例化`nn.Sequential`来构建我们的模型，返回的对象赋给`net`变量。接下来，我们反复调用`net`变量的`add`函数，按照想要执行的顺序添加层。简而言之，`nn.Sequential`定义了一种特殊类型的`Block`，即在Gluon中表示块的类。它维护`Block`的有序列表。`add`函数方便将每个连续的`Block`添加到列表中。请注意，每层都是`Dense`类的一个实例，`Dense`类本身就是`Block`的子类。正向传播（`forward`）函数也非常简单：它将列表中的每个`Block`连接在一起，将每个`Block`的输出作为输入传递给下一层。注意，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。这实际上是`net.forward(X)`的简写，这是通过`Block`类的`__call__`函数实现的一个Python技巧。
:end_tab:

:begin_tab:`pytorch`
在这个例子中，我们通过实例化`nn.Sequential`来构建我们的模型，层的执行顺序是作为参数传递的。简而言之，(**`nn.Sequential`定义了一种特殊的`Module`**)，即在PyTorch中表示一个块的类。它维护了一个由`Module`组成的有序列表，注意，两个全连接层都是`Linear`类的实例，`Linear`类本身就是`Module`的子类。正向传播（`forward`）函数也非常简单：它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。注意，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。这实际上是`net.__call__(X)`的简写。
:end_tab:

:begin_tab:`tensorflow`
在这个例子中，我们通过实例化`keras.models.Sequential`来构建我们的模型，层的执行顺序是作为参数传递的。简而言之，`Sequential`定义了一种特殊的`keras.Model`，即在Keras中表示一个块的类。它维护了一个由`Model`组成的有序列表，注意两个全连接层都是`Model`类的实例，这个类本身就是`Model`的子类。正向传播（`call`）函数也非常简单：它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。注意，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。这实际上是`net.call(X)`的简写，这是通过Block类的`__call__`函数实现的一个Python技巧。
:end_tab:

## [**自定义块**]

要想直观地了解块是如何工作的，最简单的方法可能就是自己实现一个。在实现我们自定义块之前，我们简要总结一下每个块必须提供的基本功能：

1. 将输入数据作为其正向传播函数的参数。
1. 通过正向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收任意维的输入，但是返回一个维度256的输出。
1. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
1. 存储和访问正向传播计算所需的参数。
1. 根据需要初始化模型参数。

在下面的代码片段中，我们从零开始编写一个块。它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。注意，下面的`MLP`类继承了表示块的类。我们的实现将严重依赖父类，只需要提供我们自己的构造函数（Python中的`__init__`函数）和正向传播函数。

```{.python .input}
class MLP(nn.Block):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self, **kwargs):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.out = nn.Dense(10)  # 输出层

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def call(self, X):
        return self.out(self.hidden((X)))
```

让我们首先关注正向传播函数。注意，它以`X`作为输入，计算带有激活函数的隐藏表示，并输出其未归一化的输出值。在这个`MLP`实现中，两个层都是实例变量。要了解这为什么是合理的，可以想象实例化两个多层感知机（`net1`和`net2`），并根据不同的数据对它们进行训练。当然，我们希望它们学到两种不同的模型。

我们在构造函数中[**实例化多层感知机的层，然后在每次调用正向传播函数时调用这些层**]。注意一些关键细节。首先，我们定制的`__init__`函数通过`super().__init__()`调用父类的`__init__`函数，省去了重复编写适用于大多数块的模版代码的痛苦。然后我们实例化两个全连接层，分别为`self.hidden`和`self.out`。注意，除非我们实现一个新的运算符，否则我们不必担心反向传播函数或参数初始化，系统将自动生成这些。让我们试一下。

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

块抽象的一个主要优点是它的多功能性。我们可以子类化块以创建层（如全连接层的类）、整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。我们在接下来的章节中充分利用了这种多功能性，比如在处理卷积神经网络时。

## [**顺序块**]

现在我们可以更仔细地看看`Sequential`类是如何工作的。回想一下`Sequential`的设计是为了把其他模块串起来。为了构建我们自己的简化的`MySequential`，我们只需要定义两个关键函数：
1. 一种将块逐个追加到列表中的函数。
2. 一种正向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

下面的`MySequential`类提供了与默认`Sequential`类相同的功能。

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
    # 这里，`block`是`Block`子类的一个实例，我们假设它有一个唯一的名称。我们把它保存在'Block'类的成# 员变量`_children` 中。`block`的类型是OrderedDict。当`MySequential`实例调用`initialize`
    # 函数时，系统会自动初始化`_children`的所有成员
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
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
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_children` 中。`block`的类型是OrderedDict。
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
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
            # 这里，`block`是`tf.keras.layers.Layer`子类的一个实例。
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add`函数向有序字典`_children`添加一个块。你可能会想知道为什么每个Gluon中的`Block`都有一个`_children`属性，以及为什么我们使用它而不是自己定义一个Python列表。简而言之，`_children`的主要优点是，在块的参数初始化过程中，Gluon知道在`_children`字典中查找需要初始化参数的子块。
:end_tab:

:begin_tab:`pytorch`
在`__init__`方法中，我们将每个块逐个添加到有序字典`_modules`中。你可能会想知道为什么每个`Module`都有一个`_modules`属性，以及为什么我们使用它而不是自己定义一个Python列表。简而言之，`_modules`的主要优点是，在块的参数初始化过程中，系统知道在`_modules`字典中查找需要初始化参数的子块。
:end_tab:

当`MySequential`的正向传播函数被调用时，每个添加的块都按照它们被添加的顺序执行。现在可以使用我们的`MySequential`类重新实现多层感知机。

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

注意，`MySequential`的用法与之前为`Sequential`类编写的代码相同（如 :numref:`sec_mlp_concise` 中所述）。

## [**在正向传播函数中执行代码**]

`Sequential`类使模型构造变得简单，允许我们组合新的结构，而不必定义自己的类。然而，并不是所有的架构都是简单的顺序结构。当需要更大的灵活性时，我们需要定义自己的块。例如，我们可能希望在正向传播函数中执行Python的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。

你可能已经注意到，到目前为止，我们网络中的所有操作都对网络的激活值及网络的参数起作用。然而，有时我们可能希望合并既不是上一层的结果也不是可更新参数的项。我们称之为常数参数（constant parameters）。例如，我们需要一个计算函数$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的层，其中$\mathbf{x}$是输入，$\mathbf{w}$是我们的参数，$c$是某个在优化过程中没有更新的指定常量。因此我们实现了一个`FixedHiddenMLP`类，如下所示。

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用`get_constant`函数创建的随机权重参数在训练期间不会更新（即为常量参数）。
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.dense(X)
        # 控制流
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.linear(X)
        # 控制流
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
        # 使用`tf.constant`函数创建的随机权重参数在训练期间不会更新（即为常量参数）。
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # 使用创建的常量参数以及`relu`和`matmul`函数。
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.dense(X)
        # 控制流
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。这个权重不是一个模型参数，因此它永远不会被反向传播更新。然后，网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，我们的模型做了一些不寻常的事情。我们运行了一个while循环，在$L_1$范数大于$1$的条件下，将输出向量除以$2$，直到它满足条件为止。最后，我们返回了`X`中所有项的和。据我们所知，没有标准的神经网络执行这种操作。注意，此特定操作在任何实际任务中可能都没有用处。我们的重点只是向你展示如何将任意代码集成到神经网络计算的流程中。

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

我们可以[**混合搭配各种组合块的方法**]。在下面的例子中，我们以一些想到的方法嵌套块。

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

## 效率

:begin_tab:`mxnet`
热心的读者可能会开始担心其中一些操作的效率。毕竟，我们在一个应该是高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他的Python代码。Python的问题[全局解释器锁](https://wiki.python.org/moin/GlobalInterpreterLock)是众所周知的。在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。提高Python速度的最好方法是完全避免使用Python。

Gluon这样做的一个方法是允许*混合式编程*（hybridization），这将在后面描述。
Python解释器在第一次调用块时执行它。Gluon运行时记录正在发生的事情，以及下一次它将对Python调用加速。在某些情况下，这可以大大加快速度，但当控制流（如上所述）在不同的网络通路上引导不同的分支时，需要格外小心。我们建议感兴趣的读者在读完本章后，阅读混合式编程部分（ :numref:`sec_hybridize` ）来了解编译。
:end_tab:

:begin_tab:`pytorch`
热心的读者可能会开始担心其中一些操作的效率。毕竟，我们在一个应该是高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他的Python代码。Python的问题[全局解释器锁](https://wiki.python.org/moin/GlobalInterpreterLock)是众所周知的。
在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。
:end_tab:

:begin_tab:`tensorflow`
热心的读者可能会开始担心其中一些操作的效率。毕竟，我们在一个应该是高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他的Python代码。Python的问题[全局解释器锁](https://wiki.python.org/moin/GlobalInterpreterLock)是众所周知的。在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。提高Python速度的最好方法是完全避免使用Python。
:end_tab:

## 小结

* 层也是块。
* 一个块可以由许多层组成。
* 一个块可以由许多块组成。
* 块可以包含代码。
* 块负责大量的内部处理，包括参数初始化和反向传播。
* 层和块的顺序连接由`Sequential`块处理。

## 练习

1. 如果将`MySequential`中存储块的方式更改为Python列表，会出现什么样的问题？
1. 实现一个块，它以两个块为参数，例如`net1`和`net2`，并返回正向传播中两个网络的串联输出。这也被称为平行块。
1. 假设你想要连接同一网络的多个实例。实现一个工厂函数，该函数生成同一个块的多个实例，并在此基础上构建更大的网络。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1828)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1827)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1826)
:end_tab:
