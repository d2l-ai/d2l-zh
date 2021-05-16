# 异步计算
:label:`sec_async`

今天的计算机是高度并行的系统，由多个CPU核（通常每个核有多个线程）、每个GPU有多个处理单元，每个设备通常有多个GPU组成。简而言之，我们可以同时处理许多不同的事情，且通常是在不同的设备上。不幸的是，Python不是编写并行和异步代码的好方法，至少在没有额外帮助的情况下不是好方法。毕竟，Python是单线程的，这在将来是不太可能改变。诸如MXNet和TensorFlow之类的深度学习框架采用了一种*异步编程*（asynchronous programming）模型来提高性能，而PyTorch则使用Python自己的调度器来实现不同的性能权衡。对于PyTorch，默认情况下，GPU操作是异步的。当你调用一个使用GPU的函数时，操作会排队到特定的设备上，但不一定要等到以后才执行。这允许我们并行执行更多的计算，包括在CPU或其他GPU上的操作。

因此，了解异步编程是如何工作的，通过主动减少计算需求和相互依赖，有助于我们开发更高效的程序。这使我们能够减少内存开销并提高处理器利用率。

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

## 通过后端异步处理

:begin_tab:`mxnet`
作为热身，考虑一个简单问题：我们要生成一个随机矩阵并将其相乘。让我们在NumPy和`mxnet.np`中都这样做，看看有什么不同。
:end_tab:

:begin_tab:`pytorch`
作为热身，考虑一个简单问题：我们要生成一个随机矩阵并将其相乘。让我们在NumPy和PyTorch张量中都这样做，看看它们的区别。请注意，PyTorch的 `tensor`是在GPU上定义的。
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
# GPU计算热身
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
通过MXNet的基准输出快了几个数量级。由于两者都在同一处理器上执行，因此一定有其他原因。强制MXNet在返回之前完成所有后端计算。这显示了之前发生的情况：计算由后端执行，而前端将控制权返回给Python。
:end_tab:

:begin_tab:`pytorch`
通过PyTorch的基准输出快了几个数量级。NumPy点积是在CPU上执行的，而PyTorch矩阵乘法是在GPU上执行的，后者的速度要快得多。但巨大的时差表明一定有其他原因。默认情况下，GPU操作在PyTorch中是异步的。强制PyTorch在返回之前完成所有计算。这显示了之前发生的情况：计算由后端执行，而前端将控制权返回给Python。
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
从广义上讲，MXNet有一个用于与用户直接交互的前端（例如通过Python），还有一个由系统用来执行计算的后端。如 :numref:`fig_frontends` 所示，用户可以用各种前端语言编写MXNet程序，如Python、R、Scala和C++。不管使用的前端编程语言是什么，MXNet程序的执行主要发生在C++实现的后端。由前端语言发出的操作被传递到后端执行。后端管理自己的线程，这些线程不断收集和执行排队的任务。请注意，要使其工作，后端必须能够跟踪计算图中各个步骤之间的依赖关系。因此，不可能并行化相互依赖的操作。
:end_tab:

:begin_tab:`pytorch`
广义地说，PyTorch有一个用于与用户直接交互的前端（例如通过Python），还有一个由系统用来执行计算的后端。如 :numref:`fig_frontends` 所示，用户可以用各种前端语言编写python程序，如Python和C++。不管使用的前端编程语言，PyTorch的执行主要发生在C++实现的后端。由前端语言发出的操作被传递到后端执行。后端管理自己的线程，这些线程不断收集和执行排队的任务。请注意，要使其工作，后端必须能够跟踪计算图中各个步骤之间的依赖关系。因此，不可能并行化相互依赖的操作。
:end_tab:

![编程语言前端和深度学习框架后端。](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

让我们看另一个简单例子，以便更好地理解依赖关系图。

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

![后端跟踪计算图中各个步骤之间的依赖关系。](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

上面的代码片段在 :numref:`fig_asyncgraph` 中进行了说明。每当Python前端线程执行前三条语句中的一条语句时，它只是将任务返回到后端队列。当最后一个语句的结果需要被打印出来时，Python前端线程将等待C++后端线程完成变量`z`的结果计算。这种设计的一个好处是Python前端线程不需要执行实际的计算。因此，不管Python的性能如何，对程序的整体性能几乎没有影响。 :numref:`fig_threading` 演示了前端和后端如何交互。

![前端和后端的交互。](../img/threading.svg)
:label:`fig_threading`

## 阻塞器（Blockers）

:begin_tab:`mxnet`
有许多操作将强制Python等待完成：

* 最明显的是，`npx.waitall()`等待直到所有计算完成，而不管计算指令是在什么时候发出的。在实践中，除非绝对必要，否则使用此运算符不是一个好主意，因为它可能会导致较差的性能。
* 如果我们只想等到一个特定的变量可用，我们可以调用`z.wait_to_read()`。在这种情况下，MXNet块返回Python，直到计算出变量`z`。其他的计算很可能在之后继续。

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
两个操作的完成时间大致相同。除了明显的阻塞操作之外，我们建议您注意*隐式*阻塞器。打印变量显然要求变量可用，因此是一个阻塞器。最后，通过`z.asnumpy()`到NumPy的转换和通过`z.item()`到标量的转换是阻塞的，因为NumPy没有异步的概念。它需要像`print`函数一样访问这些值。

频繁地将少量数据从MXNet的作用域复制到NumPy，可能会破坏原本高效代码的性能，因为每一个这样的操作都需要计算图来评估所有中间结果，以获得相关项，然后才能做其他事情。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 改进计算

:begin_tab:`mxnet`
在高度多线程的系统上（即使普通笔记本电脑也有4个或更多线程，在多插槽服务器上，这个数字可能超过256），调度操作的开销可能会变得非常大。这就是非常希望计算和调度异步并行进行的原因。为了说明这样做的好处，让我们看看如果我们按顺序或异步多次将变量递增1会发生什么情况。我们通过在每个加法之间插入`wait_to_read`阻塞来模拟同步执行。
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
Python前端线程和C++后端线程之间的简化交互可以概括如下：
1. 前端命令后端将计算任务`y = x + 1`插入队列。
1. 后端然后从队列接收计算任务并执行实际计算。
1. 后端然后将计算结果返回到前端。
假设这三个阶段的持续时间分别为$t_1, t_2$和$t_3$。如果不使用异步编程，执行10000次计算所需的总时间约为$10000 (t_1+ t_2 + t_3)$。如果使用异步编程，执行10000次计算所花费的总时间可以减少到$t_1 + 10000 t_2 + t_3$（假设$10000 t_2 > 9999t_1$），因为前端不必等待后端为每个循环返回计算结果。
:end_tab:

## 小结

* 深度学习框架可以将Python前端与执行后端解耦。这允许将命令快速异步插入后端。
* 异步导致了一个相当灵活的前端。但是，请注意不要过度填充任务队列，因为它可能会导致内存消耗过多。建议对每个小批量进行同步，以保持前端和后端大致同步。
* 芯片供应商提供了复杂的性能分析工具，以获得对深度学习效率更细粒度的洞察。

:begin_tab:`mxnet`
* 请注意，从MXNet管理的内存到Python的转换将迫使后端等待特定变量就绪。`print`、`asnumpy`和`item`等函数都具有此效果。这可能是需要的，但不小心使用同步会破坏性能。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 我们上面提到，使用异步计算可以将执行10000次计算所需的总时间减少到$t_1 + 10000 t_2 + t_3$。为什么我们要假设这里是$10000 t_2 > 9999 t_1$？
1. 测量`waitall`和`wait_to_read`之间的差值。提示：执行多条指令并同步以获得中间结果。
:end_tab:

:begin_tab:`pytorch`
1. 在CPU上，对本节中相同的矩阵乘法操作进行基准测试。你仍然可以通过后端观察异步吗？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2792)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2791)
:end_tab:
