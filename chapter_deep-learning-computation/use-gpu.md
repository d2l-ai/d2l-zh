# GPU计算

目前为止我们一直在使用CPU计算。对于复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效。本节中，我们将介绍如何使用单块NVIDIA GPU来计算。首先，需要确保已经安装好了至少一块NVIDIA GPU。然后，下载CUDA并按照提示设置好相应的路径 [1]。这些准备工作都完成后，下面就可以通过`nvidia-smi`命令来查看显卡信息了。

```{.python .input  n=1}
!nvidia-smi
```

接下来，我们需要确认安装了MXNet的GPU版本。如果装了MXNet的CPU版本，我们需要先卸载它，例如使用`pip uninstall mxnet`命令，然后根据CUDA的版本安装相应的MXNet版本。假设你安装了CUDA 9.0，可以通过`pip install mxnet-cu90`来安装支持CUDA 9.0的MXNet版本。运行本节中的程序需要至少两块GPU。

## 计算设备

MXNet可以指定用来存储和计算的设备，例如使用内存的CPU或者使用显存的GPU。默认情况下，MXNet会将数据创建在内存，然后利用CPU来计算。在MXNet中，`mx.cpu()`（或者在括号里填任意整数）表示所有的物理CPU和内存。这意味着MXNet的计算会尽量使用所有的CPU核。但`mx.gpu()`只代表一块GPU和相应的显存。如果有多块GPU，我们用`mx.gpu(i)`来表示第$i$块GPU及相应的显存（$i$从0开始）且`mx.gpu(0)`和`mx.gpu()`等价。

```{.python .input}
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

mx.cpu(), mx.gpu(), mx.gpu(1)
```

## `NDArray`的GPU计算

默认情况下，`NDArray`存在内存上。因此，之前我们每次打印`NDArray`的时候都会看到`@cpu(0)`这个标识。

```{.python .input  n=4}
x = nd.array([1, 2, 3])
x
```

我们可以通过`NDArray`的`context`属性来查看该`NDArray`所在的设备。

```{.python .input}
x.context
```

### GPU上的存储

我们有多种方法将`NDArray`存储在显存上。例如我们可以在创建`NDArray`的时候通过`ctx`参数指定存储设备。下面我们将`NDArray`变量`a`创建在`gpu(0)`上。注意到在打印`a`时，设备信息变成了`@gpu(0)`。创建在显存上的`NDArray`只消耗同一块显卡的显存。我们可以通过`nvidia-smi`命令查看显存的使用情况。通常，我们需要确保不创建超过显存上限的数据。

```{.python .input  n=5}
a = nd.array([1, 2, 3], ctx=mx.gpu())
a
```

假设你至少有两块GPU，下面代码将会在`gpu(1)`上创建随机数组。

```{.python .input}
B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
B
```

除了在创建时指定，我们也可以通过`copyto`和`as_in_context`函数在设备之间传输数据。下面我们将内存上的`NDArray`变量`x`复制到`gpu(0)`上。

```{.python .input  n=7}
y = x.copyto(mx.gpu())
y
```

```{.python .input}
z = x.as_in_context(mx.gpu())
z
```

需要区分的是，如果源变量和目标变量的`context`一致，`as_in_context`函数使目标变量和源变量共享源变量的内存或显存。

```{.python .input  n=8}
y.as_in_context(mx.gpu()) is y
```

而`copyto`函数总是为目标变量开新的内存或显存。

```{.python .input}
y.copyto(mx.gpu()) is y
```

### GPU上的计算

MXNet的计算会在数据的`context`所指定的设备上执行。为了使用GPU计算，我们只需要事先将数据存储在显存上。计算结果会自动保存在同一块显卡的显存上。

```{.python .input  n=9}
(z + 2).exp() * y
```

注意，MXNet要求计算的所有输入数据都在内存或同一块显卡的显存上。这样设计的原因是CPU和不同的GPU之间的数据交互通常比较耗时。因此，MXNet希望用户确切地指明计算的输入数据都在内存或同一块显卡的显存上。例如，如果将内存上的`NDArray`变量`x`和显存上的`NDArray`变量`y`做运算，会出现错误信息。当我们打印`NDArray`或将`NDArray`转换成NumPy格式时，如果数据不在内存，MXNet会将它先复制到内存，从而造成额外的传输开销。

## Gluon的GPU计算

同`NDArray`类似，Gluon的模型可以在初始化时通过`ctx`参数指定设备。下面代码将模型参数初始化在显存上。

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
```

当输入是显存上的`NDArray`时，Gluon会在同一块显卡的显存上计算结果。

```{.python .input  n=13}
net(y)
```

下面我们确认一下模型参数存储在同一块显卡的显存上。

```{.python .input  n=14}
net[0].weight.data()
```

## 小结

* MXNet可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。默认情况下，MXNet会将数据创建在内存，然后利用CPU来计算。
* MXNet要求计算的所有输入数据都在内存或同一块显卡的显存上。

## 练习

* 试试大一点的计算任务，例如大矩阵的乘法，看看使用CPU和GPU的速度区别。如果是计算量很小的任务呢？
* GPU上应如何读写模型参数？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/988)

![](../img/qr_use-gpu.svg)


## 参考文献

[1] CUDA下载地址。 https://developer.nvidia.com/cuda-downloads
