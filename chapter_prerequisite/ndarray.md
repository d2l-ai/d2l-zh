# 数据操作

在深度学习中，我们通常会频繁地对数据进行操作。作为动手学深度学习的基础，本节将介绍如何对内存中的数据进行操作。

在MXNet中，NDArray是存储和变换数据的主要工具。如果你之前用过NumPy，你会发现NDArray和NumPy的多维数组非常类似。然而，NDArray提供诸如GPU计算和自动求导在内的更多功能，这些使得NDArray更加适合深度学习。


## 创建NDArray

我们先介绍NDArray的最基本功能。如果你对我们用到的数学操作不是很熟悉，可以参阅附录中[“数学基础”](../chapter_appendix/math.md)一节。

首先从MXNet导入`ndarray`模块。这里的`nd`是`ndarray`的缩写形式。

```{.python .input  n=1}
from mxnet import nd
```

然后我们用`arange`函数创建一个行向量。

```{.python .input  n=2}
x = nd.arange(12)
x
```

其返回一个NDArray实例，里面一共包含从0开始的12个连续整数。从打印`x`时显示的属性`<NDArray 12 @cpu(0)>`可以看到，它是长度为12的一维数组，且被创建在CPU主内存上，CPU里面的0没有特别的意义，并不代表特定的核。

我们可以通过`shape`属性来获取NDArray实例形状。

```{.python .input  n=8}
x.shape
```

我们也能够通过`size`属性得到NDArray实例中元素（element）的总数。

```{.python .input  n=9}
x.size
```

下面使用`reshape`函数把向量`x`的形状改为（3，4），也就是一个3行4列的矩阵。除了形状改变之外，`x`中的元素保持不变。

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

注意`x`属性中的形状发生了变化。上面`x.reshape((3, 4))`也可写成`x.reshape((-1, 4))`或`x.reshape((3, -1))`。由于`x`的元素个数是已知的，这里的`-1`是能够通过元素个数和其他维度的大小推断出来的。

接下来，我们创建一个各元素为0，形状为（2，3，4）的张量。实际上，之前创建的向量和矩阵都是特殊的张量。

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

类似地，我们可以创建各元素为1的张量。

```{.python .input  n=5}
nd.ones((3, 4))
```

我们也可以通过Python的列表（list）指定需要创建的NDArray中每个元素的值。

```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

有些情况下，我们需要随机生成NDArray中每个元素的值。下面我们创建一个形状为（3，4）的NDArray。它的每个元素都随机采样于均值为0标准差为1的正态分布。

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

## 运算

NDArray支持大量的运算符（operator）。例如，我们可以对之前创建的两个形状为(3, 4)的NDArray做按元素加法。所得结果形状不变。

```{.python .input  n=10}
x + y
```

按元素乘法：

```{.python .input  n=11}
x * y
```

按元素除法：

```{.python .input}
x / y
```

按元素做指数运算：

```{.python .input  n=12}
y.exp()
```

除去按元素计算外，我们可以做矩阵运算。下面将`x`与`y`的转置做矩阵乘法。由于`x`是3行4列的矩阵，`y`转置为4行3列的矩阵，两个矩阵相乘得到3行3列的矩阵。

```{.python .input  n=13}
nd.dot(x, y.T)
```

我们也可以将多个NDArray合并。下面分别在行上（维度0，即形状中的最左边元素）和列上（维度1，即形状中左起第二个元素）连结两个矩阵。

```{.python .input}
nd.concat(x, y, dim=0), nd.concat(x, y, dim=1)
```

使用条件判断式可以得到元素为0或1的新的NDArray。以`x == y`为例，如果`x`和`y`在相同位置的条件判断为真（值相等），那么新NDArray在相同位置的值为1；反之为0。

```{.python .input}
x == y
```

对NDArray中的所有元素求和得到只有一个元素的NDArray。

```{.python .input}
x.sum()
```

我们可以通过`asscalar`函数将结果变换为Python中的标量。下面例子中`x`的$L_2$范数结果同上一样是单元素NDArray，但最后结果是Python中标量。

```{.python .input}
x.norm().asscalar()
```

我们也可以把`y.exp()`、`x.sum()`、`x.norm()`等分别改写为`nd.exp(y)`、`nd.sum(x)`、`nd.norm(x)`等。

## 广播机制

前面我们看到如何对两个形状相同的NDArray做按元素操作。当对两个形状不同的NDArray做同样操作时，可能会触发广播（broadcasting）机制：先适当复制元素使得这两个NDArray形状相同后再按元素操作。

定义两个NDArray：

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

由于`a`和`b`分别是3行1列和1行2列的矩阵，如果要计算`a+b`，那么`a`中第一列的三个元素被广播（复制）到了第二列，而`b`中第一行的两个元素被广播（复制）到了第二行和第三行。如此，我们就可以对两个3行2列的矩阵按元素相加。

```{.python .input}
a + b
```

## 索引

在NDArray中，索引（index）代表了元素的位置。NDArray的索引从0开始逐一递增。例如一个3行2列的矩阵的行索引分别为0、1和2，列索引分别为0和1。

在下面的例子中，我们指定了NDArray的行索引截取范围`[1:3]`。依据左闭右开指定范围的惯例，它截取了矩阵`x`中行索引为1和2的两行。

```{.python .input  n=19}
x[1:3]
```

我们可以指定NDArray中需要访问的单个元素的位置，例如矩阵中行和列的索引，并重设该元素的值。

```{.python .input  n=20}
x[1, 2] = 9
x
```

当然，我们也可以截取一部分元素，并重设它们的值。

```{.python .input  n=21}
x[1:2, 1:3] = 12
x
```

## 运算的内存开销

前面例子里我们对每个操作新开内存来储存它的结果。例如即使是`y = x + y`我们也会新创建内存，然后再将`y`指向新内存。为了展示这一点，我们可以使用Python自带的`id`函数：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

如果我们想指定结果到特定内存，我们可以使用前面介绍的索引来进替换操作。在下面的例子中，我们先通过`zeros_like`创建和`y`形状相同且元素为0的NDArray，记为`z`。接下来，我们把`x + y`的结果通过`[:]`写进`z`所对应的内存中。

```{.python .input  n=16}
z = y.zeros_like()
before = id(z)
z[:] = x + y
id(z) == before
```

注意到这里我们还是为`x + y`创建了临时内存来存储计算结果，再复制到`z`所对应的内存。如果想避免这个内存开销，我们可以使用运算符的全名函数中的`out`参数。

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

## NDArray和NumPy相互变换

我们可以通过`array`和`asnumpy`函数令数据在NDArray和NumPy格式之间相互转换。下面将NumPy实例变换成NDArray实例。

```{.python .input  n=22}
import numpy as np

p = np.ones((2, 3))
d = nd.array(p)
d
```

再将NDArray实例变换成NumPy实例。

```{.python .input}
d.asnumpy()
```

## 小结

* NDArray是MXNet中存储和变换数据的主要工具。
* 我们可以轻松地对NDArray进行创建、运算、指定索引和与NumPy之间的相互变换。


## 练习

* 运行本节代码。将本节中条件判断式`x == y`改为`x < y`或`x > y`，看看能够得到什么样的NDArray。
* 将广播机制中按元素操作的两个NDArray替换成其他形状，结果是否和预期一样？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/745)

![](../img/qr_ndarray.svg)
