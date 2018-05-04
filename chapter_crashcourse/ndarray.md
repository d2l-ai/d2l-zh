# 数据操作

在深度学习中，我们通常会频繁地对数据进行操作。作为动手学深度学习的基础，本节将介绍如何对内存中的数据进行操作。

在MXNet中，NDArray是存储和转换数据的主要工具。如果你之前用过NumPy，你会发现NDArray和NumPy的多维数组非常类似。然而，NDArray提供更多的功能，例如CPU和GPU的异步计算，以及自动求导。这些都使得NDArray更加适合深度学习。


## 创建NDArray

我们先介绍NDArray的最基本功能。如果你对我们用到的数学操作不是很熟悉，可以参阅[“数学基础”](../chapter_appendix/math.md)一节。

首先从MXNet导入NDArray。

```{.python .input  n=1}
from mxnet import nd
```

然后我们用NDArray创建一个行向量。

```{.python .input  n=2}
x = nd.arange(12)
x
```

以上创建的NDArray一共包含12个元素（element），分别为`arange(12)`所指定的从0开始的12个连续整数。可以看到，打印的`x`中还标注了属性`<NDArray 12 @cpu(0)>`。其中，12指的是NDArray的形状，即向量的长度；而@cpu(0)说明默认情况下NDArray被创建在CPU上。

下面使用`reshape`函数把向量`x`的形状改为(3, 4)，也就是一个3行4列的矩阵。除了形状改变之外，`x`中的元素保持不变。

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

接下来，我们创建一个各元素为0，形状为(2, 3, 4)的张量。实际上，之前创建的向量和矩阵都是特殊的张量。

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

类似地，我们可以创建各元素为1的张量。

```{.python .input  n=5}
nd.ones((3, 4))
```

我们也可以通过Python的列表（list）指定需要创建的NDArray中每个元素的值。

```{.python .input  n=6}
y = nd.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

有些情况下，我们需要随机生成NDArray中每个元素的值。下面我们创建一个形状为(3, 4)的NDArray。它的每个元素都随机采样于均值为0标准差为1的正态分布。

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

每个NDArray的形状可以通过`shape`属性来获取。

```{.python .input  n=8}
y.shape
```

一个NDArray的大小（size）即其元素的总数。

```{.python .input  n=9}
y.size
```

## 运算

NDArray支持大量的运算符（operator）。例如，我们可以对之前创建的两个形状为(3, 4)的NDArray做按元素加法。所得结果形状不变。

```{.python .input  n=10}
x + y
```

以下是按元素乘法。

```{.python .input  n=11}
x * y
```

以下是按元素除法。

```{.python .input}
x / y
```

以下是按元素做指数运算。

```{.python .input  n=12}
nd.exp(y)
```

接下来，我们对矩阵`y`做转置，并做矩阵乘法操作。由于`x`是3行4列的矩阵，`y`转置为4行3列的矩阵，两个矩阵相乘得到3行3列的矩阵。

```{.python .input  n=13}
nd.dot(x, y.T)
```

下面，我们对NDArray中的元素求和。结果虽然是个标量，却依然保留了NDArray格式。

```{.python .input}
x = nd.array([3, 4])
x.sum()
```

其实，我们可以把为标量的NDArray通过`asscalar`函数直接变换为Python中的数。下面例子中`x`的$L_2$范数不再是一个NDArray。

```{.python .input}
x.norm().asscalar()
```

## 广播机制

正如我们所见，我们可以对两个形状相同的NDArray做按元素操作。然而，当我们对两个形状不同的NDArray做按元素操作时，可能会触发广播（broadcasting）机制：先令这两个NDArray形状相同再按元素操作。

让我们先看个例子。

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
print('a:', a)
print('b:', b)
print('a + b:', a + b)
```

由于`a`和`b`分别是3行1列和1行2列的矩阵，为了使它们可以按元素相加，计算时`a`中第一列的三个元素被广播（复制）到了第二列，而`b`中第一行的两个元素被广播（复制）到了第二行和第三行。如此，我们就可以对两个3行2列的矩阵按元素相加，得到上面的结果。

## 运算的内存开销

在前面的例子中，我们为每个操作新开内存来存储它的结果。举个例子，假设`x`和`y`都是NDArray，在执行`y = x + y`操作后, `y`所对应的内存地址将变成为存储`x + y`计算结果而新开内存的地址。为了展示这一点，我们可以使用Python自带的`id`函数：如果两个实例的ID一致，它们所对应的内存地址相同；反之则不同。

```{.python .input  n=15}
x = nd.ones((3, 4))
y = nd.ones((3, 4))
before = id(y)
y = y + x
id(y) == before
```

在下面的例子中，我们先通过`nd.zeros_like(x)`创建和`y`形状相同且元素为0的NDArray，记为`z`。接下来，我们把`x + y`的结果通过`[:]`写进`z`所对应的内存中。

```{.python .input  n=16}
z = nd.zeros_like(y)
before = id(z)
z[:] = x + y
id(z) == before
```

然而，这里我们还是为`x + y`创建了临时内存来存储计算结果，再复制到`z`所对应的内存。为了避免这个内存开销，我们可以使用运算符的全名函数中的`out`参数。

```{.python .input  n=17}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

如果现有的NDArray的值在之后的程序中不会复用，我们也可以用 `x[:] = x + y` 或者 `x += y` 来减少运算的内存开销。

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## 索引

在NDArray中，索引（index）代表了元素的位置。NDArray的索引从0开始逐一递增。例如一个3行2列的矩阵的行索引分别为0、1和2，列索引分别为0和1。

在下面的例子中，我们指定了NDArray的行索引截取范围[1:3]。依据左闭右开指定范围的惯例，它截取了矩阵`x`中行索引为1和2的两行。

```{.python .input  n=19}
x = nd.arange(9).reshape((3, 3))
print('x:', x)
x[1:3]
```

我们可以指定NDArray中需要访问的单个元素的位置，例如矩阵中行和列的索引，并重设该元素的值。

```{.python .input  n=20}
x[1, 2] = 9
x
```

当然，我们也可以截取一部分元素，并重设它们的值。

```{.python .input  n=21}
x[1:2, 1:3] = 10
x
```

## NDArray和NumPy相互转换

我们可以通过`array`和`asnumpy`函数令数据在NDArray和Numpy格式之间相互转换。以下是一个例子。

```{.python .input  n=22}
import numpy as np
x = np.ones((2, 3))
y = nd.array(x)  # NumPy转换成NDArray。
z = y.asnumpy()  # NDArray转换成NumPy。
print([z, y])
```

## 小结

* NDArray是MXNet中存储和转换数据的主要工具。
* 我们可以轻松地对NDArray进行创建、运算、指定索引和与NumPy之间的相互转换。


## 练习

* 运行本节代码。将广播机制中按元素操作的两个NDArray替换成其他形状，结果是否和预期一样？
* 查阅MXNet官方网站上的[文档](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html)，了解NDArray支持的其他操作。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/745)

![](../img/qr_ndarray.svg)
