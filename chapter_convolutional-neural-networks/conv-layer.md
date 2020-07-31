# 图像的卷积
:label:`sec_conv_layer`

现在我们了解卷积层在理论上是如何工作的，我们已经准备好了解它们在实践中是如何工作的。基于我们将卷积神经网络作为探索图像数据结构的高效架构的动机，我们坚持以图像作为我们的运行示例。

## 互相关操作

回想一下，严格来说，卷积图层是一个错误的用词，因为它们表达的操作被更准确地描述为交叉关系。根据我们在 :numref:`sec_why-conv` 中对卷积层的描述，在这样的层中，将输入张量和核张量结合起来，通过交叉相关运算产生输出张量。

现在让我们忽略通道，看看这是如何处理二维数据和隐藏表示的。在 :numref:`fig_correlation` 中，输入是一个高度为 3，宽度为 3 的二维张量。我们将张量的形状标记为 $3 \times 3$ 或（$3$，$3$）。内核的高度和宽度都是 2。* 内核窗口 *（或 * 卷积窗口 *）的形状由内核的高度和宽度（这里是 $2 \times 2$）给出。

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

在二维交叉相关运算中，我们从位于输入张量左上角的卷积窗口开始，然后从左到右和从上到下滑过输入张量。当卷积窗口滑动到某个位置时，该窗口中包含的输入子张量和内核张量将按元素方式乘以，生成的张量总和得出一个标量值。此结果给出了相应位置的输出张量值。在这里，输出张量的高度为 2，宽度为 2，四个元素从二维互相关运算中得出：

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

请注意，沿每个轴，输出大小略小于输入大小。由于内核的宽度和高度大于 1，我们只能正确计算内核完全适合图像的位置的互相关性，输出大小由输入大小 $n_h \times n_w$ 减去卷积内核的大小 $k_h \times k_w$ 通过

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

这是因为我们需要足够的空间来跨图像 “移动” 卷积内核。稍后我们将看到如何通过在图像边界上填充零来保持大小不变，以便有足够的空间来移动内核。接下来，我们在 `corr2d` 函数中实现这个过程，该函数接受一个输入张量 `X` 和一个核张量 `K`，并返回一个输出张量 `Y`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

我们可以从 :numref:`fig_correlation` 构造输入张量 `X` 和核张量 `K`, 以验证上述二维互相关运算实现的输出.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 卷积层

卷积层将输入和内核交叉关联，并添加标量偏置以生成输出。卷积层的两个参数是核和标量偏差。当基于卷积图层训练模型时，我们通常会随机初始化内核，就像我们使用完全连接的图层一样。

我们现在已经准备好实现基于上述 `corr2d` 函数的二维卷积层。在 `__init__` 构造函数中，我们声明 `weight` 和 `bias` 为两个模型参数。正向传播函数调用 `corr2d` 函数并添加偏置。

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

在卷积内核或卷积内核中，卷积内核的高度和宽度分别为 $h$ 和 $h \times w$。我们也将卷积内核的卷积层简单地称为 $h \times w$ 卷积层。

## 图像中的对象边缘检测

让我们花点时间来解析卷积图层的简单应用程序：通过查找像素变化的位置来检测图像中对象的边缘。首先，我们构建了一个 $6\times 8$ 像素的 “图像”。中间四列为黑色 (0)，其余为白色 (1)。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

接下来，我们构建一个高度为 1，宽度为 2 的内核 `K`。当我们与输入执行交叉相关运算时，如果水平相邻的元素相同，则输出为 0。否则，输出为非零。

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

我们准备用参数 `X`（我们的输入）和 `K`（我们的内核）执行交叉相关操作。正如你所看到的，我们检测 1 对于从白到黑的边缘，-1 对于从黑到白的边缘。所有其他输出均采用值 0。

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

我们现在可以将内核应用于转置的图像。正如预期的那样，它消失了。内核 `K` 只检测垂直边缘。

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## 学习内核

如果我们知道这正是我们正在寻找的东西，通过有限的差异设计边缘检测器 `[1, -1]` 是整洁的。但是，当我们查看较大的内核并考虑连续的卷积层时，可能无法精确指定每个过滤器应手动执行的操作。

现在让我们看看我们是否可以通过仅查看输入-输出对来学习从 `X` 生成 `Y` 的内核。我们首先构建卷积层并将其内核初始化为随机张量。接下来，在每次迭代中，我们将使用平方误差将 `Y` 与卷积图层的输出进行比较。然后，我们可以计算渐变来更新内核。为了简单起见，下面我们将内置类用于二维卷积图层并忽略偏差。

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i+1}, loss {tf.reduce_sum(l):.3f}')
```

请注意，在 10 次迭代之后，错误已降至较小的值。现在我们来看看我们学到的内核张量。

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

事实上，学习的内核张量非常接近我们之前定义的内核张量 `K`。

## 互相关和卷积

回想一下我们从 :numref:`sec_why-conv` 中观察到的交叉相关和卷积运算之间的对应关系。在这里，我们继续考虑二维卷积层。如果这些图层执行 :eqref:`eq_2d-conv-discrete` 中定义的严格卷积操作而不是交叉关联，该怎么办？为了获得严格的 * 卷积 * 运算的输出，我们只需要水平和垂直翻转二维核张量，然后使用输入张量执行 * 交叉关联 * 运算。

值得注意的是，由于内核是从深度学习中的数据中学习的，因此卷积图层的输出都不会受到影响，无论此类图层执行严格的卷积操作还是交叉相关操作。

为了说明这一点，假设卷积层执行 * 交叉关联 *，并在 :numref:`fig_correlation` 中学习内核，在此处表示为矩阵 $\mathbf{K}$。假设其他条件保持不变，当此层执行严格的 * 卷积 * 时，学习的内核 $\mathbf{K}'$ 将与 $\mathbf{K}$ 相同，在水平和垂直翻转 $\mathbf{K}'$ 之后。也就是说，当卷积层对 :numref:`fig_correlation` 和 $\mathbf{K}'$ 中的输入执行严格的 * 卷积 * 时，将得到 :numref:`fig_correlation` 中的相同输出（输入和 $\mathbf{K}$ 的互相关性）。

根据深度学习文献的标准术语，我们将继续将交叉相关运算称为卷积，尽管严格地说，它略有不同。此外，我们使用术语 * element* 来指表示层表示或卷积内核的任何张量的条目（或组件）。

## 特征地图和接受场

如 :numref:`subsec_why-conv-channels` 所述，:numref:`fig_correlation` 中的卷积图层输出有时称为 * 要素地图 *，因为它可以被视为后续图层的空间维度（例如，宽度和高度）中的学习制图表达（要素）。在 CNN 中，对于某个层的任何元素 $x$，其 * 接受场 * 是指可能影响正向传播期间计算 $x$ 的所有元素（来自以前所有层）。请注意，接受场可能大于输入的实际大小。

让我们继续使用 :numref:`fig_correlation` 来解释接受场。给定 $2 \times 2$ 卷积内核，阴影输出元素的接受场（值 $19$）是输入阴影部分中的四个元素。现在让我们将 $2 \times 2$ 输出表示为 $\mathbf{Y}$，并考虑一个更深的 CNN，另外还有一个卷积层，以 $2 \times 2$ 作为其输入，输出一个单元 $z$。在这种情况下，$\mathbf{Y}$ 上 $z$ 的接受场包括 $\mathbf{Y}$ 的所有四个元素，而输入上的接受场包括所有九个输入元素。因此，当要素地图中的任何元素需要更大的接受场来检测更广泛区域的输入要素时，我们可以构建更深的网络。

## 摘要

* 二维卷积层的核心计算是二维互相关运算。以最简单的形式，这会对二维输入数据和内核执行交叉相关操作，然后添加偏置。
* 我们可以设计一个内核来检测图像中的边缘。
* 我们可以从数据中学习内核的参数。
* 通过从数据中获取的内核，卷积图层的输出保持不受影响，无论此类图层的执行操作如何（严格卷积或互相关）。
* 当要素地图中的任何元素需要更大的接受场来检测输入中更广泛的要素时，可以考虑更深的网络。

## 练习

1. 构建一个带有对角线边缘的图像 `X`。
    1. 如果您将本节中的内核 `K` 应用于它，会发生什么情况？
    1. 如果你转置 `X` 会发生什么？
    1. 如果你转置 `K` 会发生什么？
1. 当您尝试自动查找我们创建的 `Conv2D` 类的渐变时，您会看到什么样的错误消息？
1. 如何通过更改输入和内核张量来将交叉相关运算表示为矩阵乘法？
1. 手动设计一些内核。
    1. 第二个导数的内核形式是什么？
    1. 什么是积分的内核？
    1. 获得 $d$ 度导数的内核的最小尺寸是多少？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
