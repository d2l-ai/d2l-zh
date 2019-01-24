# 自动并行计算

MXNet后端会自动构建计算图。通过计算图，系统可以知道所有计算的依赖关系，并可以选择将没有依赖关系的多个任务并行执行来获得计算性能的提升。例如[“异步计算”](async-computation.md)一节的第一个例子里依次执行了`a = nd.ones((1, 2))`和`b = nd.ones((1, 2))`。这两步计算之间并没有依赖关系，因此系统可以选择并行执行它们。

通常，一个运算符会用到所有CPU或单块GPU上全部的计算资源。例如，`dot`运算符会用到所有CPU（即使是一台机器上有多个CPU处理器）或单块GPU上所有的线程。如果每个运算符的计算量足够大，只在CPU上或者单块GPU上并行运行多个运算符时，每个运算符的运行只分到CPU或单块GPU上部分计算资源。即使这些计算可以并行，最终计算性能的提升可能也并不明显。本节中探讨的自动并行计算主要关注同时使用CPU和GPU的并行计算，以及计算和通信的并行。

首先导入本节中实验所需的包或模块。注意，需要至少一块GPU才能运行本节实验。

```{.python .input}
import d2lzh as d2l
import mxnet as mx
from mxnet import nd
```

## CPU和GPU的并行计算

我们先介绍CPU和GPU的并行计算，例如，程序中的计算既发生在CPU上，又发生在GPU上。先定义`run`函数，令它做10次矩阵乘法。

```{.python .input}
def run(x):
    return [nd.dot(x, x) for _ in range(10)]
```

接下来，分别在内存和显存上创建`NDArray`。

```{.python .input}
x_cpu = nd.random.uniform(shape=(2000, 2000))
x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))
```

然后，分别使用它们在CPU和GPU上运行`run`函数并打印运行所需时间。

```{.python .input}
run(x_cpu)  # 预热开始
run(x_gpu)
nd.waitall()  # 预热结束

with d2l.Benchmark('Run on CPU.'):
    run(x_cpu)
    nd.waitall()

with d2l.Benchmark('Then run on GPU.'):
    run(x_gpu)
    nd.waitall()
```

我们去掉`run(x_cpu)`和`run(x_gpu)`这两个计算任务之间的`waitall`同步函数，并希望系统能自动并行这两个任务。

```{.python .input}
with d2l.Benchmark('Run on both CPU and GPU in parallel.'):
    run(x_cpu)
    run(x_gpu)
    nd.waitall()
```

可以看到，当两个计算任务一起执行时，执行总时间小于它们分开执行的总和。这表明，MXNet能有效地在CPU和GPU上自动并行计算。


## 计算和通信的并行计算

在同时使用CPU和GPU的计算中，经常需要在内存和显存之间复制数据，造成数据的通信。在下面的例子中，我们在GPU上计算，然后将结果复制回CPU使用的内存。我们分别打印GPU上计算时间和显存到内存的通信时间。

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU.'):
    y = run(x_gpu)
    nd.waitall()

with d2l.Benchmark('Then copy to CPU.'):
    copy_to_cpu(y)
    nd.waitall()
```

我们去掉计算和通信之间的`waitall`同步函数，打印这两个任务完成的总时间。

```{.python .input}
with d2l.Benchmark('Run and copy in parallel.'):
    y = run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()
```

可以看到，执行计算和通信的总时间小于两者分别执行的耗时之和。需要注意的是，这个计算并通信的任务不同于本节之前介绍的同时使用CPU和GPU并行计算的任务。这里的运行和通信之间有依赖关系：`y[i]`必须先在GPU上计算好才能复制到CPU使用的内存。所幸的是，在计算`y[i]`的时候系统可以复制`y[i-1]`，从而减少计算和通信的总运行时间。

## 小结

* MXNet能够通过自动并行计算提升计算性能，例如CPU和GPU的并行计算以及计算和通信的并行。


## 练习

* 本节中定义的`run`函数里做了10次运算。它们之间也没有依赖关系。设计实验，看看MXNet有没有自动并行执行它们。
* 设计包含更加复杂的数据依赖的计算任务，通过实验观察MXNet能否得到正确的结果并提升计算性能。
* 当运算符的计算量足够小时，仅在CPU或单块GPU上并行计算也可能提升计算性能。设计实验来验证这一点。




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1883)

![](../img/qr_auto-parallelism.svg)
