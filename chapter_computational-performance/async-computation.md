# 异步计算
:label:`sec_async`

今天的计算机是高度并行的系统，由多个 CPU 核心（通常是每个核心多个线程）、每个 GPU 的多个处理元素以及通常每台设备多个 GPU 组成。简而言之，我们可以同时处理许多不同的事物，通常是在不同的设备上。不幸的是，Python 不是编写并行和异步代码的好方法，至少没有一些额外的帮助。毕竟，Python 是单线程的，这在未来不太可能改变。MxNet 和 TensorFlow 等深度学习框架采用 * 异步编程 * 模型来提高性能，而 PyTorch 则使用 Python 自己的调度程序，从而实现不同的性能权衡。对于 PyTorch，默认情况下，GPU 操作是异步的。当您调用使用 GPU 的函数时，这些操作将入队到特定设备，但不一定要等到以后才执行。这使我们能够并行执行更多计算，包括 CPU 或其他 GPU 上的操作。 

因此，了解异步编程的工作原理有助于我们通过主动降低计算需求和相互依赖性来开发更高效的程序。这使我们能够减少内存开销并提高处理器利用率。

```{.python .input}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## 通过后端进行异步

:begin_tab:`mxnet`
对于热身，请考虑以下玩具问题：我们想生成一个随机矩阵并将其乘以。让我们在 NumPy 和 `mxnet.np` 中这样做来看看差异。
:end_tab:

:begin_tab:`pytorch`
对于热身，请考虑以下玩具问题：我们想生成一个随机矩阵并将其乘以。让我们在 NumPy 和 PyTorch 张量中这样做来看看差异。请注意，PyTorch `tensor` 是在 GPU 上定义的。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
通过 MxNet 的基准输出速度快了数量级。由于两者都在同一个处理器上执行，因此必须继续进行其他事情。强制 MxNet 在返回之前完成所有后端计算会显示以前发生的情况：计算由后端执行，而前端将控制权返回给 Python。
:end_tab:

:begin_tab:`pytorch`
通过 PyTorch 的基准输出速度快了数量级。NumPy 点积在 CPU 处理器上执行，而 PyTorch 矩阵乘法则在 GPU 上执行，因此后者的速度预计会快得多。但是，巨大的时差表明必须发生其他事情。默认情况下，PyTorch 中的 GPU 操作是异步的。强制 PyTorch 在返回之前完成所有计算会显示以前发生的情况：计算由后端执行，而前端则将控制权返回给 Python。
:end_tab:

```{.python .input}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
广义地说，MxNet 有一个用于与用户直接交互的前端（例如通过 Python）以及系统用于执行计算的后端。如 :numref:`fig_frontends` 所示，用户可以使用各种前端语言（如 Python、R、Scala 和 C++）编写 MxNet 程序。无论使用哪种前端编程语言，MxNet 程序的执行主要发生在 C ++ 实现的后端。前端语言发布的操作将传递到后端执行。后端管理自己的线程，这些线程持续收集和执行排队任务。请注意，为此，后端必须能够跟踪计算图中各个步骤之间的依赖关系。因此，不可能并行化彼此依赖的操作。
:end_tab:

:begin_tab:`pytorch`
广义地说，PyTorch 有一个用于与用户直接交互的前端（例如通过 Python）以及系统用于执行计算的后端。如 :numref:`fig_frontends` 所示，用户可以使用各种前端语言（如 Python 和 C ++）编写 PyTorch 程序。无论使用哪种前端编程语言，PyTorch 程序的执行主要发生在 C ++ 实现的后端。前端语言发布的操作将传递到后端执行。后端管理自己的线程，这些线程持续收集和执行排队任务。请注意，为此，后端必须能够跟踪计算图中各个步骤之间的依赖关系。因此，不可能并行化彼此依赖的操作。
:end_tab:

![Programming language frontends and deep learning framework backends.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

让我们看另一个玩具示例，以便更好地理解依赖关系图。

```{.python .input}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![The backend tracks dependencies between various steps in the computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

上面的代码片段也在 :numref:`fig_asyncgraph` 中进行了说明。每当 Python 前端线程执行前三个语句之一时，它只需将任务返回到后端队列。当最后一条语句的结果需要 * 打印 * 时，Python 前端线程将等待 C ++ 后端线程完成计算变量 `z` 的结果。这种设计的一个好处是 Python 前端线程不需要执行实际的计算。因此，无论 Python 的性能如何，对程序的整体性能都没有什么影响。:numref:`fig_threading` 说明了前端和后端的交互方式。 

![Interactions of the frontend and backend.](../img/threading.svg)
:label:`fig_threading`

:begin_tab:`mxnet`
## 障碍和阻滞剂

有许多操作会迫使 Python 等待完成：
* 最明显的是，无论计算指令何时发出，`npx.waitall()` 都会等到所有计算完成。实际上，除非绝对必要，否则使用此操作符是一个坏主意，因为它可能会导致性能不佳。
* 如果我们只想等到特定变量可用，我们可以调用 `z.wait_to_read()`。在这种情况下，MxNet 块返回到 Python，直到计算出变量 `z`。其他计算之后可能会继续进行。

让我们看看这在实践中是如何运作的。
:end_tab:

```{.python .input}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
两项操作需要大约相同的时间才能完成。除了显而易见的阻止操作之外，我们建议您知道 * 隐式 * 阻止程序。打印变量显然需要变量可用，因此它是阻止程序。最后，由于 NumPy 没有异步概念，通过 `z.asnumpy()` 转换为 NumPy 以及通过 `z.item()` 转换为标量的操作都受到阻碍。它需要像 `print` 函数一样访问这些值。  

经常将少量数据从 MxNet 的范围复制到 NumPy 然后会破坏本来有效的代码的性能，因为每个此类操作都需要计算图来评估获得相关术语所需的所有中间结果 * 之前 * 可以做的其他任何事情。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

:begin_tab:`mxnet`
## 改进计算在一个严重的多线程系统上（即使是普通笔记本电脑有 4 个或更多线程，在多插槽服务器上，这个数字可能超过 256 个），调度操作的开销可能会变得巨大。这就是为什么非常希望以异步和并行方式进行计算和调度。为了说明这样做的好处，让我们看看如果我们按顺序或异步方式多次增加一个变量，会发生什么情况。我们通过在每次添加之间插入 `wait_to_read` 障碍来模拟同步执行。
:end_tab:

```{.python .input}
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python 前端线程和 C ++ 后端线程之间稍微简化的交互可以总结如下：
1. 前端命令后端将计算任务 `y = x + 1` 插入队列。
1. 然后，后端接收队列中的计算任务并执行实际的计算。
1. 然后，后端将计算结果返回给前端。
假设这三个阶段的持续时间分别为 $t_1, t_2$ 和 $t_3$。如果我们不使用异步编程，则执行 10000 个计算所需的总时间约为 $10000 (t_1+ t_2 + t_3)$。如果使用异步编程，则执行 10000 个计算所花费的总时间可以减少到 $t_1 + 10000 t_2 + t_3$（假设为 $10000 t_2 > 9999t_1$），因为前端不必等后端返回每个循环的计算结果。 

## 摘要

* 深度学习框架可能会将 Python 前端与执行后端分离。这允许将命令快速异步插入到后端和相关的并行度。
* 异步导致前端响应相当灵敏。但是，请注意不要溢出任务队列，因为这可能会导致过多的内存消耗。建议对每个微型批次进行同步，以使前端和后端保持大致同步。
* 芯片供应商提供复杂的性能分析工具，以获得对深度学习效率的更精细的洞察。
:end_tab:

:begin_tab:`mxnet`
* 请注意，从 MxNet 的内存管理转换为 Python 将强制后端等到特定变量准备就绪。`print`、`asnumpy` 和 `item` 等函数都具有这样的效果。这可能是可取的，但粗心地使用同步可能会破坏性能。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 我们上面提到过，使用异步计算可以将执行 10000 次计算所需的总时间减少到 $t_1 + 10000 t_2 + t_3$。为什么我们必须在这里假设 $10000 t_2 > 9999 t_1$？
1. 衡量 `waitall` 和 `wait_to_read` 之间的差异。提示：执行许多指令并同步以获得中间结果。
:end_tab:

:begin_tab:`pytorch`
1. 在 CPU 上，在本节中对相同的矩阵乘法运算进行基准测试。你还能通过后端观察异步吗？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab:
