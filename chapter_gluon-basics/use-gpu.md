# GPU计算

目前为止，我们一直在使用CPU计算。的确，绝大部分的计算设备都有CPU。然而，CPU的设计目的是处理通用的计算。对于复杂的神经网络和大规模的数据来说，使用单块CPU计算可能不够高效。

本节中，我们将介绍如何使用单块Nvidia GPU来计算。

首先，需要确保至少有一块Nvidia显卡已经安装好了。然后，下载安装显卡驱动和[CUDA](https://developer.nvidia.com/cuda-downloads)（推荐下载8.0，CUDA自带了驱动）。Windows用户还需要设一下PATH：

> `set PATH=C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%`

这些准备工作都完成后，下面就可以通过`nvidia-smi`来查看显卡信息了。

```{.python .input  n=1}
!nvidia-smi
```

接下来，我们需要确认安装了MXNet的GPU版本。如果装了MXNet的CPU版本，我们需要卸载它。例如

> `pip uninstall mxnet`

为了使用MXNet的GPU版本，我们需要根据CUDA版本安装`mxnet-cu75`、`mxnet-cu80`或者`mxnet-cu90`。例如

> `pip install --pre mxnet-cu80`

## 处理器

使用MXNet的GPU版本和之前没什么不同。下面导入本节中实验所需的包。

```{.python .input}
import mxnet as mx
from mxnet import gluon, nd
import sys
```

MXNet使用`context`来指定用来存储和计算的设备。默认情况下，MXNet会将数据开在主内存，然后利用CPU来计算。在MXNet中，CPU和GPU可分别由`mx.cpu()`和`mx.gpu()`来表示。需要注意的是，`mx.cpu()`表示所有的物理CPU和内存。这意味着计算上会尽量使用所有的CPU核。但`mx.gpu()`只代表一块显卡和相应的显卡内存。如果有多块GPU，我们用`mx.gpu(i)`来表示第$i$块GPU（$i$从0开始）。

```{.python .input  n=3}
[mx.cpu(), mx.gpu(), mx.gpu(1)]
```

## NDArray的GPU计算

每个NDArray都有一个`context`属性来表示它存在哪个设备上。默认情况下，NDArray存在CPU上。因此，之前我们每次打印NDArray的时候都会看到`@cpu(0)`这个标识。

```{.python .input  n=4}
x = nd.array([1,2,3])
print('x: ', x, '\ncontext of x: ', x.context)
```

### GPU上的存储

我们可以在创建NDArray的时候通过`ctx`指定存储设备。

```{.python .input  n=5}
a = nd.array([1, 2, 3], ctx=mx.gpu())
b = nd.zeros((3, 2), ctx=mx.gpu())
# 假设至少存在2块GPU。如果不存在则会报错。
c = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1)) 
print('a: ', a, '\nb: ', b, '\nc: ', c)
```

我们可以通过`copyto`和`as_in_context`函数在设备之间传输数据。

```{.python .input  n=7}
y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())
print('x: ', x, '\ny: ', y, '\nz: ', z)
```

需要区分的是，如果源变量和目标变量的`context`一致，`as_in_context`使目标变量和源变量共享源变量的内存，而`copyto`总是为目标变量新创建内存。

```{.python .input  n=8}
y_target = y.as_in_context(mx.gpu())
z_target = z.copyto(mx.gpu())
print('y: ', y, '\ny_target: ', y_target)
print('z: ', z, '\nz_target: ', z_target)
print('y_target and y share memory? ', y_target is y)
print('z_target and z share memory? ', z_target is z)
```

### GPU上的计算

MXNet的计算会在数据的`context`上执行。为了使用GPU计算，我们只需要事先将数据放在GPU上面。而计算结果会自动保存在相同的设备上。

```{.python .input  n=9}
nd.exp(z + 2) * y
```

注意，MXNet要求计算的所有输入数据都在同一个设备上。这个设计的原因是设备之间的数据交互通常比较耗时。因此，MXNet希望用户确切地指明计算的输入数据都在同一个设备上。例如，如果将CPU上的`x`和GPU上的`y`做运算，会出现错误信息。

### 其他复制到主内存的操作

当我们打印NDArray或将NDArray转换成NumPy格式时，MXNet会自动将数据复制到主内存。

```{.python .input  n=11}
print(y)
print(y.asnumpy())
print(y.sum().asscalar())
```

## Gluon的GPU计算

同NDArray类似，Gluon的大部分函数可以通过`ctx`指定设备。下面代码将模型参数初始化在GPU上。

```{.python .input  n=12}
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize(ctx=mx.gpu())
```

当输入是GPU上的NDArray时，Gluon会在相同的GPU上计算结果。

```{.python .input  n=13}
data = nd.random.uniform(shape=[3, 2], ctx=mx.gpu())
net(data)
```

确认一下模型参数存储在相同的GPU上。

```{.python .input  n=14}
net[0].weight.data()
```

## 小结

* 通过`context`，我们可以在不同的设备上存储数据和计算。

## 练习

* 试试大一点的计算任务，例如大矩阵的乘法，看看CPU和GPU的速度区别。如果是计算量很小的任务呢？
* GPU上应如何读写模型参数？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/988)

![](../img/qr_use-gpu.svg)
