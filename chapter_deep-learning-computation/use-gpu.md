# GPU计算

目前为止我们一直在使用CPU计算。对于复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效。本节中，我们将介绍如何使用单块Nvidia GPU来计算。首先，需要确保至少有一块Nvidia GPU已经安装好了。然后，下载[CUDA](https://developer.nvidia.com/cuda-downloads)并按照提示设置好相应的路径。这些准备工作都完成后，下面就可以通过`nvidia-smi`来查看显卡信息了。

```{.python .input  n=1}
!nvidia-smi
```

接下来，我们需要确认安装了MXNet的GPU版本。如果装了MXNet的CPU版本，我们需要先卸载它。例如我们可以使用`pip uninstall mxnet`。然后根据CUDA的版本安装对应的MXNet版本。假设你安装了CUDA 9.1，那么我们可以通过`pip install --pre mxnet-cu91`来安装支持CUDA 9.1的MXNet版本。

## 计算设备

MXNet使用`context`来指定用来存储和计算的设备，例如可以是CPU或者GPU。默认情况下，MXNet会将数据创建在主内存，然后利用CPU来计算。在MXNet中，CPU和GPU可分别由`cpu()`和`gpu()`来表示。需要注意的是，`mx.cpu()`（或者在括号里填任意整数）表示所有的物理CPU和内存。这意味着计算上会尽量使用所有的CPU核。但`mx.gpu()`只代表一块显卡和相应的显卡内存。如果有多块GPU，我们用`mx.gpu(i)`来表示第$i$块GPU（$i$从0开始）且`mx.gpu(0)`和`mx.gpu()`等价。

```{.python .input}
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

[mx.cpu(), mx.gpu(), mx.gpu(1)]
```

## NDArray的GPU计算

默认情况下，NDArray存在CPU上。因此，之前我们每次打印NDArray的时候都会看到`@cpu(0)`这个标识。

```{.python .input  n=4}
x = nd.array([1,2,3])
x
```

我们可以通过NDArray的`context`属性来查看其所在的设备。

```{.python .input}
x.context
```

### GPU上的存储

我们有多种方法将NDArray放置在GPU上。例如我们可以在创建NDArray的时候通过`ctx`指定存储设备。下面我们将`a`创建在GPU 0上。注意到在打印`a`时，设备信息变成了`@gpu(0)`。创建在GPU上时我们会只用GPU内存，你可以通过`nvidia-smi`查看GPU内存使用情况。通常你需要确保不要创建超过GPU内存上限的数据。

```{.python .input  n=5}
a = nd.array([1, 2, 3], ctx=mx.gpu())
a
```

假设你至少有两块GPU，下面代码将会在GPU 1上创建随机数组

```{.python .input}
b = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1)) 
b
```

除了在创建时指定，我们也可以通过`copyto`和`as_in_context`函数在设备之间传输数据。下面我们将CPU上的`x`复制到GPU 0上。

```{.python .input  n=7}
y = x.copyto(mx.gpu())
y
```

```{.python .input}
z = x.as_in_context(mx.gpu())
z
```

需要区分的是，如果源变量和目标变量的`context`一致，`as_in_context`使目标变量和源变量共享源变量的内存，

```{.python .input  n=8}
y.as_in_context(mx.gpu()) is y
```

而`copyto`总是为目标变量新创建内存。

```{.python .input}
y.copyto(mx.gpu()) is y
```

### GPU上的计算

MXNet的计算会在数据的`context`上执行。为了使用GPU计算，我们只需要事先将数据放在GPU上面。而计算结果会自动保存在相同的GPU上。

```{.python .input  n=9}
(z + 2).exp() * y
```

注意，MXNet要求计算的所有输入数据都在同一个CPU/GPU上。这个设计的原因是不同CPU/GPU之间的数据交互通常比较耗时。因此，MXNet希望用户确切地指明计算的输入数据都在同一个CPU/GPU上。例如，如果将CPU上的`x`和GPU上的`y`做运算，会出现错误信息。

当我们打印NDArray或将NDArray转换成NumPy格式时，如果数据不在主内存里，MXNet会自动将其先复制到主内存，从而带来隐形的传输开销。

## Gluon的GPU计算

同NDArray类似，Gluon的模型可以在初始化时通过`ctx`指定设备。下面代码将模型参数初始化在GPU上。

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
```

当输入是GPU上的NDArray时，Gluon会在相同的GPU上计算结果。

```{.python .input  n=13}
net(y)
```

确认一下模型参数存储在相同的GPU上。

```{.python .input  n=14}
net[0].weight.data()
```

## 小结

* 通过`context`，我们可以在不同的CPU/GPU上存储数据和计算。

## 练习

* 试试大一点的计算任务，例如大矩阵的乘法，看看CPU和GPU的速度区别。如果是计算量很小的任务呢？
* GPU上应如何读写模型参数？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/988)

![](../img/qr_use-gpu.svg)
