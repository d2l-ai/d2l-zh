# 编译器和解释器
:label:`sec_hybridize`

目前为止，本书主要关注的是 **命令式编程**（imperative programming）。命令式编程使用诸如`print`、`+`和`if`之类的语句来更改程序的状态。考虑下面这段简单的命令式程序。

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python是一种 **解释型语言**（interpreted language）。因此，当对上面的 `fancy_func` 函数求值时，它按顺序执行函数体的操作。也就是说，它将通过对 `e = add(a, b)` 求值，并将结果存储为变量 `e`，从而更改程序的状态。接下来的两个语句 `f = add(c, d)` 和 `g = add(e, f)` 也将类似地操作，执行加法计算并将结果存储为变量。 :numref:`fig_compute_graph` 说明了数据流。

![命令式编程中的数据流。](../img/computegraph.svg)
:label:`fig_compute_graph`

尽管命令式编程很方便，但可能效率不高。一方面原因，Python 会单独执行这三个函数的调用，而没有考虑 `add` 函数在 `fancy_func` 中被重复调用。如果在一个 GPU（甚至多个 GPU）上执行这些命令，那么 Python 解释器产生的开销可能会非常大。此外，它需要保存 `e` 和 `f` 的变量值，直到 `fancy_func` 中的所有语句都执行完毕。这是因为程序不知道在执行语句 `e = add(a, b)` 和 `f = add(c, d)` 之后，其他部分是否会使用变量 `e `和 `f`。

## 符号式编程

考虑另一种选择 **符号式编程**（symbolic programming），通常代码只在完全定义了过程之后才执行计算。这个策略被多个深度学习框架使用，包括 Theano 和 TensorFlow（后者已经获得了命令式编程的扩展）。通常包括以下步骤：

1. 定义计算流程。
1. 将流程编译成可执行的程序。
1. 给定输入，调用编译好的程序执行。

这将允许进行大量的优化。首先，在大多数情况下，我们可以跳过 Python 解释器。从而消除因为多个更快的 GPU 与单个 CPU上 的单个 Python 线程搭配使用时产生的性能瓶颈。其次，编译器可以将上述代码优化和重写为`print((1 + 2) + (3 + 4))`甚至`print(10)`。因为编译器在将其转换为机器指令之前可以看到完整的代码，所以这种优化是可以实现的。例如，只要某个变量不再需要，编译器就可以释放内存（或者从不分配内存），或者将代码转换为一个完全等价的片段。下面，我们将通过对命令式编程的模拟来进一步了解它的概念。

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

命令式（解释型）编程和符号式编程的区别如下：

* 命令式编程更容易使用。在 Python 中，命令式编程的大部分代码都是简单易懂的。命令式编程也更容易调试，这是因为无论是获取和打印所有的中间变量值，或者使用 Python 的内置调试工具也更加简单。
* 符号式编程运行效率更高，更易于移植。符号式编程更容易在编译期间优化代码，同时还能够将程序移植到与 Python 无关的格式中，从而允许程序在非 Python 环境中运行，避免了与 Python 解释器相关的任何潜在的性能问题。

## 混合式编程

历史上，大部分深度学习框架都在命令式编程与符号式编程之间进行选择。例如，Theano、TensorFlow（灵感来自前者）、Keras 和 CNTK 采用了符号式编程。相反地，Chainer 和 PyTorch 采取了命令式编程。在后来的版本更新中，TensorFlow 2.0 和 Keras 增加了命令式编程。

:begin_tab:`mxnet`
开发人员在设计Gluon时思考了这个问题，有没有可能将两种编程模式的优点结合起来。于是得到了一个混合式编程模型，既允许用户使用纯命令式编程进行开发和调试，还能够将大多数程序转换为符号式程序，以便在需要产品级计算性能和部署时使用。

这意味着我们在实际开发中使用的是 `HybridBlock` 或 `HybridSequential` 类在构建模型。默认情况下，它们都与命令式编程中执行 `Block` 或 `Sequential` 类的方式相同。其中，`HybridSequential` 类是 `HybridBlock` 的子类（就如 `Sequential` 是 `Block` 的子类一样）。当 `hybridize` 函数被调用时，Gluon 将模型编译成符号式编程中使用的形式。这将允许在不牺牲模型实现方式的情况下优化计算密集型组件。下面，我们将重点放在`Sequential` 和 `Block` 上来详细描述其优点。
:end_tab:

:begin_tab:`pytorch`
如上所述，PyTorch 是基于命令式编程并且使用动态计算图。为了能够利用符号式编程的可移植性和效率，开发人员思考能否将两种编程模型的优点结合起来，于是就产生了torchscript。torchscript 允许用户使用纯命令式编程进行开发和调试，同时能够将大多数程序转换为符号式程序，以便在需要产品级计算性能和部署时使用。
:end_tab:

:begin_tab:`tensorflow`
命令式编程现在是 TensorFlow 2 的默认选择，对于那些刚接触该语言的人来说是一个很好的改变。不过，符号式编程技术和计算图仍然存在于 TensorFlow 中，并且可以通过易于使用的装饰器模式 `tf.function` 进行访问。这为 TensorFlow 带来了命令式编程范式，允许用户定义更加直观的函数，然后使用被 TensorFlow 团队称为 [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph) 的特性将它们封装，再自动编译成计算图。
:end_tab:

## `Sequential`的混合式编程

要了解混合式编程的工作原理，最简单的方法是考虑具有多层的深层网络。按照惯例，Python解释器将需要为所有层执行代码以生成指令，然后可以将该指令转发到CPU或GPU。对于单个（快速）计算设备，这不会导致任何重大问题。另一方面，如果我们使用高级的8-GPU服务器，比如AWS P3dn.24xlarge实例，Python将很难让所有GPU保持忙碌。单线程Python解释器成为这里的瓶颈。让我们看看如何通过用`HybridSequential`替换代码重要部分的`Sequential`来解决问题。我们首先定义一个简单的多层感知机。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# 生产网络的工厂
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# 生产网络的工厂
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 生产网络的工厂
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
通过调用`hybridize`函数，我们可以编译和优化多层感知机中的计算。模型的计算结果保持不变。
:end_tab:

:begin_tab:`pytorch`
通过使用`torch.jit.script`函数转换模型，我们可以编译和优化多层感知机的计算。模型的计算结果保持不变。
:end_tab:

:begin_tab:`tensorflow`
以前，tensorflow中构建的所有函数都是作为计算图构建的，因此默认情况下是JIT编译的。但是，随着TensorFlow2.X和Earge tensors的发布，这不再是默认行为。我们使用tf.function重新启用此功能。tf.function更常用作函数装饰器，但是可以直接将其作为普通python函数调用，如下所示。模型的计算结果保持不变。
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
只需将一个块指定为`HybridSequential`，编写与之前相同的代码并调用`hybridize`。一旦发生这种情况，网络将得到优化（我们将在下面对性能进行基准测试）。不幸的是，这并不是适用于每一层。也就是说，如果一个层从`Block`类而不是`HybridBlock`类继承，它将不会得到优化。
:end_tab:

:begin_tab:`pytorch`
编写与以前相同的代码，并使用`torch.jit.script`简单地转换模型。一旦发生这种情况，网络将得到优化（我们将在下面对性能进行基准测试）。
:end_tab:

:begin_tab:`tensorflow`
编写与以前相同的代码，并使用`tf.function`简单地转换模型。一旦发生这种情况，网络将以TensorFlow的MLIR中间表示形式构建为一个计算图，并在编译器级别进行大量优化，以实现快速执行（我们将在下面对性能进行基准测试）。显式地将`jit_compile = True`标志添加到`tf.function()`调用可以启用TensorFlow中的XLA（加速线性代数）功能。在某些情况下，XLA可以进一步优化JIT编译代码。。在没有这种显式定义的情况下，可以启用图形模式执行，但是，XLA可以使某些大型线性代数操作(与我们在深度学习应用程序中看到的操作类似)速度更快，特别是在GPU环境中。
:end_tab:

### 通过混合式编程加速

为了证明通过编译获得的性能改进，我们比较了混合编程前后执行`net(x)`所需的时间。让我们先定义一个函数来度量这个时间。当我们开始衡量（和改进）性能时，它在本章中将非常有用。

```{.python .input}
#@tab all
#@save
class Benchmark:
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
现在我们可以调用网络两次，一次启用混合式，一次没有启用混合式。
:end_tab:

:begin_tab:`pytorch`
现在我们可以调用网络两次，一次使用torchscript，一次不使用torchscript。
:end_tab:

:begin_tab:`tensorflow`
现在我们可以调用网络三次，一次使用eager执行，一次是以图模式执行，另一次是使用JIT编译的XLA。
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('无混合式'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('混合式'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('无torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('有torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager模式'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph模式'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
从上面的结果中可以看到，在`HybridSequential`实例调用`hybridize`函数之后，通过使用符号式编程提高了计算性能。
:end_tab:

:begin_tab:`pytorch`
从上面的结果中可以看到，使用`nn.Sequential`函数编写了`torch.jit.script`实例的脚本之后，通过使用符号式编程来提高计算性能。
:end_tab:

:begin_tab:`tensorflow`
从以上结果可以看出，在使用`tf.function`函数编写tf.keras `Sequential` 实例脚本之后，通过在tensorflow中通过图形模式执行使用符号式编程来提高计算性能。
:end_tab:

### 序列化

:begin_tab:`mxnet`
编译模型的好处之一是我们可以将模型及其参数序列化（保存）到磁盘。这允许我们以独立于所选前端语言的方式存储模型。这使我们能够将经过训练的模型部署到其他设备上，并方便地使用其他前端编程语言。同时，代码通常比命令式编程更快。让我们看看`export`的实际功能。
:end_tab:

:begin_tab:`pytorch`
编译模型的好处之一是我们可以将模型及其参数序列化（保存）到磁盘。这允许我们以独立于所选前端语言的方式存储模型。这使我们能够将经过训练的模型部署到其他设备上，并方便地使用其他前端编程语言。同时，代码通常比命令式编程更快。让我们看看`save`的实际功能。
:end_tab:

:begin_tab:`tensorflow`
编译模型的好处之一是我们可以将模型及其参数序列化（保存）到磁盘。这允许我们以独立于所选前端语言的方式存储模型。这使我们能够将经过训练的模型部署到其他设备上，并轻松地使用其他前端编程语言或在服务器上执行经过训练的模型。同时，代码通常比命令式编程更快。允许我们在tensorflow中保存的底层API是`tf.saved_model`。让我们来看看`saved_model`的运行情况。
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
模型被分解成一个大的二进制参数文件和一个执行模型计算所需的程序的JSON描述。这些文件可以由Python或MXNet支持的其他前端语言读取，例如C++、R、Scala和Perl。让我们看看模型描述中的前几行。
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
之前，我们演示了在调用`hybridize`函数之后，该模型能够实现优异的计算性能和可移植性。注意，混合式可能影响模型的灵活性，特别是在控制流方面。

此外，与`Block`实例需要使用`forward`函数不同，对于`HybridBlock`实例，我们需要使用`hybrid_forward`函数。
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
上面的代码实现了一个具有4个隐藏单元和2个输出的简单网络。`hybrid_forward`函数接受一个额外的参数`F`。这是必需的，因为根据代码是否已混合，它将使用稍微不同的库（`ndarray`或`symbol`）进行处理。这两个类执行非常相似的函数，MXNet自动确定参数。为了理解发生了什么，我们将参数作为函数调用的一部分打印出来。
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
重复前向计算将导致相同的输出（我们省略细节）。现在让我们看看如果调用`hybridize`函数会发生什么。
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
我们不再使用`ndarray`，而是使用`symbol`模块来表示`F`。此外，即使输入是`ndarray`类型，作为编译过程的一部分，经过网络的数据现在也转换为`symbol`类型。重复函数调用会产生令人惊讶的结果：
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet`
这与我们以前看到的情况大不相同。`hybrid_forward`中定义的所有打印语句都被省略。实际上，在执行`net(x)`之后，不再涉及Python解释器。这意味着任何Python代码（例如print语句）都会被省略，以利于更精简的执行和更好的性能。相反，MXNet直接调用C++后端。另外请注意，`symbol`模块中不支持某些功能（例如`asnumpy`），`a += b`和`a[:] = a + b`等操作必须重写为`a = a + b`。尽管如此，尽管如此，只要速度很重要，模型的编译都是值得的。根据模型的复杂性、CPU的速度以及GPU的速度和数量，优势可以从很小的百分比到两倍以上的速度不等。
:end_tab:

## 小结

* 命令式编程使设计新模型变得容易，因为它可以用控制流编写代码，并且能够使用大量Python软件生态。
* 符号式编程要求我们在执行程序之前指定并编译程序。其好处是提高了性能。

:begin_tab:`mxnet`
* MXNet能够根据需要结合这两种方法的优点。
* 由`HybridSequential`和`HybridBlock`类构造的模型能够通过调用`hybridize`函数将命令式程序转换为符号式程序。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 在本节中，在`HybridNet`类的`hybrid_forward`函数的第一行中添加`x.asnumpy()`。执行代码并观察遇到的错误。为什么会这样？
1. 如果我们在`hybrid_forward`函数中添加控制流，即Python语句`if`和`for`，会发生什么？
1. 回顾前几章中你感兴趣的模型。你能通过重新实现它们来提高它们的计算性能吗？
:end_tab:

:begin_tab:`pytorch,tensorflow`
1. 回顾前几章中你感兴趣的模型。你能通过重新实现它们来提高它们的计算性能吗？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2789)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2788)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2787)
:end_tab: