# Automatic Parallelism
:label:`sec_auto_para`


Deep learning frameworks (e.g., MXNet and PyTorch) automatically construct computational graphs at the backend. Using a
computational graph, the system is aware of all the dependencies,
and can selectively execute multiple non-interdependent tasks in parallel to
improve speed. For instance, :numref:`fig_asyncgraph` in :numref:`sec_async` initializes two variables independently. Consequently the system can choose to execute them in parallel.


Typically, a single operator will use all the computational resources on all CPUs or on a single GPU. For example, the `dot` operator will use all cores (and threads) on all CPUs, even if there are multiple CPU processors on a single machine. The same applies to a single GPU. Hence parallelization is not quite so useful for single-device computers. With multiple devices things matter more. While parallelization is typically most relevant between multiple GPUs, adding the local CPU will increase performance slightly. For example, see :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` that focuses on training computer vision models combining a GPU and a CPU. With the convenience of an automatically parallelizing framework we can accomplish the same goal in a few lines of Python code. More broadly, our discussion of automatic parallel computation focuses on parallel computation using both CPUs and GPUs, as well as the parallelization of computation and communication.

Note that we need at least two GPUs to run the experiments in this section.

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

## Parallel Computation on GPUs

Let us start by defining a reference workload to test: the `run` function below performs 10 matrix-matrix multiplications on the device of our choice using data allocated into two variables: `x_gpu1` and `x_gpu2`.

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
Now we apply the function to the data. To ensure that caching does not play a role in the results we warm up the devices by performing a single pass on either of them prior to measuring.
:end_tab:

:begin_tab:`pytorch`
Now we apply the function to the data. To ensure that caching does not play a role in the results we warm up the devices by performing a single pass on either of them prior to measuring. `torch.cuda.synchronize()` waits for all kernels in all streams on a CUDA device to complete. It takes in a `device` argument, the device for which we need to synchronize. It uses the current device, given by `current_device()`, if the device argument is `None` (default).
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
If we remove the `waitall` statement between both tasks the system is free to parallelize computation on both devices automatically.
:end_tab:

:begin_tab:`pytorch`
If we remove the `synchronize` statement between both tasks the system is free to parallelize computation on both devices automatically.
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

In the above case the total execution time is less than the sum of its parts, since the deep learning framework automatically schedules computation on both GPU devices without the need for sophisticated code on behalf of the user.



## Parallel Computation and Communication

In many cases we need to move data between different devices, say between the CPU and GPU, or between different GPUs. 
For instance,
this occurs when we want to perform distributed optimization where we need to aggregate the gradients over multiple accelerator cards. Let us simulate this by computing on the GPU and then copying the results back to the CPU.

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
This is somewhat inefficient. Note that we could already start copying parts of `y` to the CPU while the remainder of the list is still being computed. This situation occurs, e.g., when we compute the gradient on a minibatch. The gradients of some of the parameters will be available earlier than that of others. Hence it works to our advantage to start using PCI-Express bus bandwidth while the GPU is still running. Removing `waitall` between both parts allows us to simulate this scenario.
:end_tab:

:begin_tab:`pytorch`
This is somewhat inefficient. Note that we could already start copying parts of `y` to the CPU while the remainder of the list is still being computed. This situation occurs, e.g., when we compute the (backprop) gradient on a minibatch. The gradients of some of the parameters will be available earlier than that of others. Hence it works to our advantage to start using PCI-Express bus bandwidth while the GPU is still running. In PyTorch, several functions such as `to()` and `copy_()` admit an explicit `non_blocking` argument, which lets the caller bypass synchronization when it is unnecessary. Setting `non_blocking=True` allows us to simulate this scenario.
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

The total time required for both operations is (as expected) less than the sum of their parts.
Note that this task is different from parallel computation as it uses a different resource: the bus between the CPU and GPUs. In fact, we could compute on both devices and communicate, all at the same time. As noted above, there is a dependency between computation and communication: `y[i]` must be computed before it can be copied to the CPU. Fortunately, the system can copy `y[i-1]` while computing `y[i]` to reduce the total running time.

We conclude with an illustration of the computational graph and its dependencies for a simple two-layer MLP when training on a CPU and two GPUs, as depicted in :numref:`fig_twogpu`. It would be quite painful to schedule the parallel program resulting from this manually. This is where it is advantageous to have a graph-based computing backend for optimization.

![The computational graph and its dependencies of a two-layer MLP on a CPU and two GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`


## Summary

* Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously. 
* Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically solid-state drives or via networks), and network bandwidth. They can be used in parallel for peak efficiency. 
* The backend can improve performance through automatic parallel computation and communication. 

## Exercises

1. Eight operations were performed in the `run` function defined in this section. There are no dependencies between them. Design an experiment to see if the deep learning framework will automatically execute them in parallel.
1. When the workload of an individual operator is sufficiently small, parallelization can help even on a single CPU or GPU. Design an experiment to verify this. 
1. Design an experiment that uses parallel computation on CPUs, GPUs, and communication between both devices.
1. Use a debugger such as NVIDIA's [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) to verify that your code is efficient. 
1. Designing computation tasks that include more complex data dependencies, and run experiments to see if you can obtain the correct results while improving performance.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
