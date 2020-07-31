# 查阅文档

由于这本书篇幅的限制，我们不可能介绍每一个 MxNet 函数和类（你可能不希望我们这样做）。API 文档和其他教程和示例提供了大量超出书籍的文档。在本节中，我们为您提供了一些有关浏览 MxNet API 的指导。

## 查找模块中的所有函数和类

为了知道可以在模块中调用哪些函数和类，我们调用 `dir` 函数。实例，我们可以查询模块中的所有属性来生成随机数：

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

通常，我们可以忽略以 `__` 开始和结束的函数（Python 中的特殊对象）或以单个 `_` 开始的函数（通常是内部函数）。根据剩余的函数或属性名称，我们可能会猜测这个模块提供了各种生成随机数的方法，包括从均匀分布 (`uniform`)、正态分布 (`normal`) 和多项分布 (`multinomial`) 中取样。

## 查找特定函数和类的用法

有关如何使用给定函数或类的更具体说明，我们可以调用 `help` 函数。作为一个样本，我们来探讨天量 `ones` 函数的使用说明。

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

从文档中，我们可以看到 `ones` 函数创建一个具有指定形状的新张量，并将所有元素设置为 1 的值。在可能的情况下，您应该运行一个快速测试来确认您的解释：

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

在木星笔记本中，我们可以使用 `？`在另一个窗口中显示文档。样本，`列表？`将创建与 `help(list)` 几乎相同的内容，并在新的浏览器窗口中显示它。此外，如果我们使用两个问号，例如 `列表？？`，也将显示实现该函数的 Python 代码。

## 摘要

* 官方文档提供了大量本书以外的描述和示例。
* 我们可以通过调用 `dir` 和 `help` 函数或 `?是什么？`在木星笔记本电脑。

## 练习

1. 在深度学习框架中查找任何函数或类的文档。你也可以在框架的官方网站上找到文档吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
