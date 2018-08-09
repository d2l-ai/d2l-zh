# 自动并行计算

在[“异步计算”](async-computation.md)一节里我们提到MXNet后端会自动构建计算图。通过计算图，系统可以知道所有计算的依赖关系，并可以选择将没有依赖关系的多个任务并行执行来获得性能的提升。以[“异步计算”](async-computation.md)一节中的计算图（图8.1）为例。其中`a = nd.ones((1, 2))`和`b = nd.ones((1, 2))`这两步计算之间并没有依赖关系。因此，系统可以选择并行执行它们。

通常一个运算符会用到所有CPU或单个GPU上全部的计算资源。例如，`dot`操作符会用到所有CPU（即使是一台机器上有多个CPU处理器）或单个GPU上所有线程。因此单纯在CPU或者GPU上并行运行多个运算符可能效果并不明显。本节中探讨的自动并行计算主要关注同时使用CPU和GPU的并行计算，以及计算和通讯的并行。

首先导入本节中实验所需的包或模块。注意，我们需要至少一个GPU才能运行本节实验。

```{.python .input}
import mxnet as mx
from mxnet import nd
from time import time
```

## CPU和GPU的并行计算

我们先介绍CPU和GPU的并行计算，例如程序中的计算既发生在CPU，又发生在GPU之上。

先定义一个函数，令它做10次矩阵乘法。

```{.python .input}
def run(x):
    return [nd.dot(x, x) for _ in range(10)]
```

接下来，分别在CPU和GPU上创建NDArray。

```{.python .input}
x_cpu = nd.random.uniform(shape=(2000, 2000))
x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))
```

然后，分别使用它们在CPU和GPU上运行`run`函数并打印所需时间。

```{.python .input}
run(x_cpu)  # 预热开始。
run(x_gpu)
nd.waitall()  # 预热结束。

start = time()
run(x_cpu)
nd.waitall()
print('run on CPU: %f sec' % (time() - start))

start = time()
run(x_gpu)
nd.waitall()
print('run on GPU: %f sec' % (time() - start))
```

我们去掉`run(x_cpu)`和`run(x_gpu)`两个计算任务之间的`nd.waitall()`，希望系统能自动并行这两个任务。

```{.python .input}
start = time()
run(x_cpu)
run(x_gpu)
nd.waitall()
print('run on both CPU and GPU: %f sec' % (time() - start))
```

可以看到，当两个计算任务一起执行时，执行总时间小于它们分开执行的总和。这表示，MXNet能有效地在CPU和GPU上自动并行计算。


## 计算和通讯的并行计算

在同时使用CPU和GPU的计算中，我们经常需要在CPU和GPU之间复制数据，造成数据的通讯。举个例子，在下面例子中，我们在GPU上计算，然后将结果复制回CPU。我们分别打印GPU上计算时间和GPU到CPU的通讯时间。

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

start = time()
y = run(x_gpu)
nd.waitall()
print('run on GPU: %f sec' % (time() - start))

start = time()
copy_to_cpu(y)
nd.waitall()
print('copy to CPU: %f sec' % (time() - start))
```

我们去掉计算和通讯之间的`waitall`函数，打印这两个任务完成的总时间。

```{.python .input}
start = time()
y = run(x_gpu)
copy_to_cpu(y)
nd.waitall()
t = time() - start
print('run on GPU then copy to CPU: %f sec' % (time() - start))
```

可以看到，执行计算和通讯的总时间小于两者分别执行的耗时之和。需要注意的是，这个计算并通讯的任务不同于本节之前介绍的同时使用CPU和GPU并行计算的任务。这里的运行和通讯之间有依赖关系：`y[i]`必须先计算好才能复制到CPU。所幸的是，在计算`y[i]`的时候系统可以复制`y[i-1]`，从而减少计算和通讯的总运行时间。

## 小结

* MXNet能够通过自动并行计算提升计算性能，例如CPU和GPU的并行以及计算和通讯的并行。


## 练习

* 本节中定义的`run`函数里做了10次运算。它们之间也没有依赖关系。看看MXNet有没有自动并行执行它们。

* 试试包含更加复杂的数据依赖的计算任务。MXNet能不能得到正确结果并提升计算性能？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1883)

![](../img/qr_auto-parallelism.svg)
