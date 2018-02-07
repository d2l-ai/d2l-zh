# 使用NDArray来处理数据

对于机器学习来说，处理数据往往是万事之开头。它包含两个部分：(i)数据读取，(ii)数据已经在内存中时如何处理。本章将关注后者。

我们首先介绍`NDArray`，这是MXNet储存和变换数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点使得`NDArray`能更好地支持机器学习。

## 让我们开始

我们先介绍最基本的功能。如果你不懂我们用到的数学操作也不用担心，例如按元素加法、正态分布；我们会在之后的章节分别从数学和代码编写的角度详细介绍。

我们首先从`mxnet`导入`ndarray`这个包

```{.python .input  n=1}
from mxnet import ndarray as nd
```

然后我们创建一个3行和4列的2D数组（通常也叫**矩阵**），并且把每个元素初始化成0

```{.python .input  n=2}
nd.zeros((3, 4))
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input  n=3}
x = nd.ones((3, 4))
x
```

或者从python的数组直接构造

```{.python .input  n=4}
nd.array([[1,2],[2,3]])
```

我们经常需要创建随机数组，即每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。以下代码创建数组，它的元素服从均值0标准差1的正态分布。

```{.python .input  n=5}
y = nd.random_normal(0, 1, shape=(3, 4))
y
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input  n=6}
y.shape
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input  n=7}
y.size
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input  n=8}
x + y
```

乘法：

```{.python .input  n=9}
x * y
```

指数运算：

```{.python .input  n=10}
nd.exp(y)
```

也可以转置一个矩阵然后计算矩阵乘法：

```{.python .input  n=11}
nd.dot(x, y.T)
```

我们会在之后的线性代数一章讲解这些运算符。

## 广播（Broadcasting）

当二元操作符左右两边ndarray形状不一样时，系统会尝试将其复制到一个共同的形状。例如`a`的第0维是3, `b`的第0维是1，那么`a+b`时会将`b`沿着第0维复制3遍：

```{.python .input  n=23}
a = nd.arange(3).reshape((3,1))
b = nd.arange(2).reshape((1,2))
print('a:', a)
print('b:', b)
print('a+b:', a+b)

```

## 跟NumPy的转换

ndarray可以很方便地同numpy进行转换

```{.python .input  n=26}
import numpy as np
x = np.ones((2,3))
y = nd.array(x)  # numpy -> mxnet
z = y.asnumpy()  # mxnet -> numpy
print([z, y])
```

## 替换操作

在前面的样例中，我们为每个操作新开内存来存储它的结果。例如，如果我们写`y = x + y`, 我们会把`y`从现在指向的实例转到新建的实例上去。我们可以用Python的`id()`函数来看这个是怎么执行的：

```{.python .input}
x = nd.ones((3, 4))
y = nd.ones((3, 4))

before = id(y)
y = y + x
id(y) == before
```

我们可以把结果通过`[:]`写到一个之前开好的数组里：

```{.python .input}
z = nd.zeros_like(x)
before = id(z)
z[:] = x + y
id(z) == before
```

但是这里我们还是为`x+y`创建了临时空间，然后再复制到`z`。需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

如果现有的数组不会复用，我们也可以用 `x[:] = x + y` ，或者 `x += y` 达到这个目的：

```{.python .input  n=16}
before = id(x)
x += y
id(x) == before
```

## 截取（Slicing）

MXNet NDArray 提供了各种截取方法。截取 x 的 index 为 1、2 的行：

```{.python .input}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[1:3]
```

以及直接写入指定位置：

```{.python .input}
x[1,2] = 9.0
x
```

多维截取：

```{.python .input}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[1:2,1:3]
```

多维写入：

```{.python .input}
x[1:2,1:3] = 9.0
x
```

## 总结

ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见[NDArray API文档](https://mxnet.incubator.apache.org/api/python/ndarray.html)。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/745)
