# 自动并行
:label:`sec_auto_para`

深度学习框架（例如，MxNet 和 PyTorch）会在后端自动构建计算图。使用计算图，系统可以了解所有依赖关系，并且可以选择性地并行执行多个不相互依赖的任务以提高速度。例如，:numref:`sec_async` 中的 :numref:`fig_asyncgraph` 独立初始化两个变量。因此，系统可以选择并行执行它们。 

通常，单个操作员将使用所有 CPU 或单个 GPU 上的所有计算资源。例如，`dot` 操作员将使用所有 CPU 上的所有内核（和线程），即使一台计算机上有多个 CPU 处理器也是如此。同样适用于单个 GPU。因此，并行化对于单设备计算机来说并不是那么有用。对于多台设备，事情更重要。虽然并行化通常在多个 GPU 之间最相关，但添加本地 CPU 将略微提高性能。例如，请参阅 :cite:`Hadjis.Zhang.Mitliagkas.ea.2016`，其中重点介绍了结合 GPU 和 CPU 的计算机视觉模型的训练。借助自动并行化框架的便利性，我们可以通过几行 Python 代码实现相同的目标。更广泛地说，我们对自动并行计算的讨论侧重于使用 CPU 和 GPU 的并行计算，以及计算和通信的并行化。 

请注意，我们至少需要两个 GPU 来运行本节中的实验。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU 上的并行计算

让我们首先定义要测试的参考工作负载：下面的 `run` 函数使用分配到两个变量的数据在我们选择的设备上执行 10 个矩阵-矩阵乘法：`x_gpu1` 和 `x_gpu2`。

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
现在我们将函数应用于数据。为了确保缓存不会在结果中发挥作用，我们在测量之前通过对其中任何一个设备执行一次传递来预热设备。
:end_tab:

:begin_tab:`pytorch`
现在我们将函数应用于数据。为了确保缓存不会在结果中发挥作用，我们在测量之前通过对其中任何一个设备执行一次传递来预热设备。`torch.cuda.synchronize()` 等待 CUDA 设备上所有流中的所有内核完成。它采用 `device` 参数，我们需要同步的设备。如果设备参数为 `None`（默认值），则它使用 `current_device()` 给出的当前设备。
:end_tab:

```{.python .input}
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
如果我们删除两个任务之间的 `waitall` 语句，系统就可以自由地在两个设备上自动并行计算。
:end_tab:

:begin_tab:`pytorch`
如果我们删除两个任务之间的 `synchronize` 语句，系统就可以自由地在两个设备上自动并行计算。
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

在上述情况下，总执行时间少于各部分的总和，因为深度学习框架会自动安排两台 GPU 设备上的计算，而无需代表用户编写复杂的代码。 

## 并行计算和通信

在许多情况下，我们需要在不同设备之间移动数据，比如在 CPU 和 GPU 之间，或在不同的 GPU 之间移动数据。例如，当我们想要执行分布式优化时，我们需要在多个加速器卡上聚合渐变时，就会发生这种情况。让我们通过在 GPU 上进行计算，然后将结果复制回 CPU 来模拟此操作。

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
这有点效率低下。请注意，我们已经可以开始将 `y` 的部分复制到 CPU，而列表的其余部分仍在计算中。例如，当我们计算微型批次的梯度时，就会发生这种情况。其中一些参数的梯度将比其他参数的梯度更早提供。因此，在 GPU 仍在运行的同时开始使用 PCI-Express 总线带宽对我们有利。在两个部分之间删除 `waitall` 使我们能够模拟这种情况。
:end_tab:

:begin_tab:`pytorch`
这有点效率低下。请注意，我们已经可以开始将 `y` 的部分复制到 CPU，而列表的其余部分仍在计算中。例如，当我们计算微型批次上的（backprop）渐变时，就会发生这种情况。其中一些参数的梯度将比其他参数的梯度更早提供。因此，在 GPU 仍在运行的同时开始使用 PCI-Express 总线带宽对我们有利。在 PyTorch 中，`to()` 和 `copy_()` 等多个函数都承认了一个明确的 `non_blocking` 参数，在不必要的情况下，调用者可以绕过同步。设置 `non_blocking=True` 允许我们模拟此场景。
:end_tab:

```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

两项操作所需的总时间（如预期的那样）都少于其各部分的总和。请注意，此任务与并行计算不同，因为它使用不同的资源：CPU 和 GPU 之间的总线。事实上，我们可以同时在两台设备上进行计算并进行通信。如上所述，计算和通信之间存在依赖关系：必须先计算 `y[i]`，然后才能将其复制到 CPU。幸运的是，该系统可以在计算 `y[i]` 的同时拷贝 `y[i-1]` 以减少总运行时间。 

如 :numref:`fig_twogpu` 中所述，我们最后说明了在 CPU 和两个 GPU 上训练时，计算图及其对简单的双层 MLP 的依赖关系。手动安排由此产生的并行程序将非常痛苦。在这里，拥有基于图形的计算后端进行优化是有利的。 

![The computational graph and its dependencies of a two-layer MLP on a CPU and two GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`

## 摘要

* 现代系统具有各种设备，例如多个 GPU 和 CPU。它们可以并行、异步使用。 
* 现代系统还有各种通信资源，例如 PCI Express、存储（通常是固态硬盘或通过网络）和网络带宽。它们可以并行使用以实现峰值效率。 
* 后端可以通过自动并行计算和通信来提高性能。 

## 练习

1. 在本节定义的 `run` 函数中执行了八个操作。它们之间没有依赖关系。设计一个实验，看看深度学习框架是否会自动并行执行它们。
1. 当单个操作员的工作负载足够小时，即使在单个 CPU 或 GPU 上，并行化也可以提供帮助。设计一个实验来验证这一点。 
1. 设计一个在 CPU、GPU 上使用并行计算以及两个设备之间的通信的实验。
1. 使用 NVIDIA [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) 之类的调试器来验证您的代码是否有效。 
1. 设计包含更复杂的数据依赖关系的计算任务，并运行实验以查看是否可以在提高性能的同时获得正确的结果。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
