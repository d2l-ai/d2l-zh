# 使用NDArray来处理数据

处理数据是万事的开头。它包含两个部分，数据读取，和当数据以及在内存里时如何处理。本章将关注后者。我们首先介绍`NDArray`，这是MXNet储存和变化数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点是的`NDArray`能更好的支持机器学习。

## 让我们开始

我们先介绍最基本的功能。不用担心如果你不懂我们用到的数学操作，例如按元素加法，或者正态分布，我们会在之后的两章分别详细介绍。

我们首先从`mxnet`导入`ndarray`这个包

```{.python .input}
from mxnet import ndarray as nd
```

然后我们创建一个有3行和2列的2D数组（通常也叫矩阵），并且把每个元素初始化成0

```{.python .input}
nd.zeros((3, 4))
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input}
x = nd.ones((3, 4))
x
```

我们经常需要创建随机数组，就是说每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。下面创建数组，它的元素服从均值0方差1的正太分布。

```{.python .input}
y = nd.random_normal(0, 1, shape=(3, 4))
y
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input}
y.shape
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input}
y.size
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input}
x + y
```

乘法：

```{.python .input}
x * y
```

指数运算：

```{.python .input}
nd.exp(y)
```

也可以转秩一个矩阵然后计算矩阵乘法：

```{.python .input}
nd.dot(x, y.T)
```

我们会在下一章[线性代数](P01-C03-linear-algebra.md)里详细介绍更多操作，这里我们主要关注NDArray的基本工作机制。

## 广播

## 替换操作

在前面的样例中，我们为每个操作新开内存来存储它的结果。例如，如果我们写`y = x + y`, 我们会把`y`从现在指向的实例转到新建的实例上去。我们可以用Python的`id()`函数来看这个是怎么执行的：

```{.python .input}
before = id(y)
y = y + x
id(y) == before
```

我们可以吧结果通过`[:]`写到一个之前开好的数组里：

```{.python .input}
z = nd.zeros_like(x)
before = id(z)
z[:] = x + y
id(z) == before
```

但是这里我们还是为`x+y`创建了临时空间，然后在复制到`z`. 需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

如果可以现有的数组之后不会再用，我们也可以用复制操作符达到这个目的
```{.python .input}
before = id(x)
x += y
id(x) == before
```

## 跟NumPy的转换

