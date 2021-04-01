# 查阅文档
:begin_tab:`mxnet`
由于这本书篇幅的限制，我们不可能介绍每一个 MXNet 函数和类（你可能也不希望我们这样做）。API文档、其他教程和示例提供了本书之外的大量文档。在本节中，我们为你提供了一些查看MXNet API 的指导。
:end_tab:

:begin_tab:`pytorch`
由于本书篇幅的限制，我们不可能介绍每一个PyTorch函数和类（你可能也不希望我们这样做）。API文档、其他教程和示例提供了本书之外的大量文档。在本节中，我们为你提供了一些查看PyTorch API的指导。
:end_tab:

:begin_tab:`tensorflow`
由于本书篇幅的限制，我们不可能介绍每一个TensorFlow函数和类（你可能也不希望我们这样做）。API文档、其他教程和示例提供了本书之外的大量文档。在本节中，我们为你提供了一些查TensorFlow API的指导。
:end_tab:

## 查找模块中的所有函数和类

为了知道模块中可以调用哪些函数和类，我们调用 `dir` 函数。例如，我们可以(**查询随机数生成模块中的所有属性：**)

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

通常，我们可以忽略以 `__` 开始和结束的函数（Python 中的特殊对象）或以单个 `_` 开始的函数（通常是内部函数）。根据剩余的函数名或属性名，我们可能会猜测这个模块提供了各种生成随机数的方法，包括从均匀分布（`uniform`）、正态分布 （`normal`）和多项分布（`multinomial`）中采样。

## 查找特定函数和类的用法

有关如何使用给定函数或类的更具体说明，我们可以调用 `help` 函数。例如，我们来[**查看张量 `ones` 函数的用法。**]

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

从文档中，我们可以看到 `ones` 函数创建一个具有指定形状的新张量，并将所有元素值设置为 1。让我们来[**运行一个快速测试**]来确认这一解释：

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

在Jupyter记事本中，我们可以使用 `?` 在另一个窗口中显示文档。例如，`list?`将创建与`help(list)` 几乎相同的内容，并在新的浏览器窗口中显示它。此外，如果我们使用两个问号，如 `list??`，将显示实现该函数的 Python 代码。

## 小结

* 官方文档提供了本书之外的大量描述和示例。
* 我们可以通过调用 `dir` 和 `help` 函数或在Jupyter记事本中使用`?` 和 `??`查看API的用法文档。

## 练习

1. 在深度学习框架中查找任何函数或类的文档。你能在这个框架的官方网站上找到文档吗?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1764)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1765)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1763)
:end_tab:
