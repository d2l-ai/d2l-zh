# 共享
:label:`sec_pooling`

通常，当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率，聚合信息，使我们在网络中的越高，接受场（在输入中）越大，每个隐藏节点是敏感的。

我们的最终任务通常会问一些关于图像的全球性问题，例如 * 它是否包含猫？* 因此，通常我们最终层的单位应该对整个输入敏感。通过逐渐聚合信息，生成粗糙和粗糙的地图，我们实现了这一目标，即最终学习全局表示，同时将卷积图层的所有优势保留在处理的中间层。

此外，在检测较低级别的要素时，如边（如 :numref:`sec_conv_layer` 中所讨论的那样），我们通常希望我们的制图表达与平移有些不变。例如，如果我们拍摄图像 `X`，并在黑白之间划分清晰，然后将整个图像向右移动一个像素，即 `Z[i, j] = X[i, j + 1]`，那么新图像 `Z` 的输出可能会有很大的不同。边将移动一个像素。在现实中，对象几乎没有发生完全在同一个地方。事实上，即使是一个三脚架和一个固定的物体，由于快门的移动而产生的相机振动可能会将所有内容都移动一个像素左右（高端摄像机装有特殊功能来解决这个问题）。

本节介绍了 * 池化图层 *，它具有双重目的，可降低卷积图层对位置的敏感性和空间缩减采样制图表达的敏感性。

## 最大池和平均池

与卷积图层一样，*pooling* 运算符包含一个固定形状的窗口，该窗口根据其步幅滑动到输入中的所有区域，为固定形状窗口（有时称为 * 池化窗口 *）遍历的每个位置计算单个输出。但是，与卷积层中输入和内核的互相关计算不同，池层不包含任何参数（没有 * 内核 *）。相反，池运算符是确定性的，通常计算池窗口中元素的最大值或平均值。这些操作分别称为 * 最大池数 *（* 最大池数 * 表示短）和 * 平均池数 *。

在这两种情况下，与交叉相关运算符一样，我们可以将池窗口视为从输入张量的左上角开始，然后从左到右，从上到下滑动输入张量。在池窗口命中的每个位置，它根据使用最大池还是平均池，计算窗口中输入子张量的最大值或平均值。

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` 中的输出张量的高度为 2，宽度为 2。这四个元素是从每个池窗口中的最大值派生的：

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

池窗口形状为 $p \times q$ 的池层称为 $p \times q$ 池层。池化操作称为 $p \times q$ 池化操作。

让我们回到本节开头提到的对象边缘检测示例。现在我们将使用卷积图层的输出作为 $2\times 2$ 最大池的输入。将卷积层输入设置为 `X`，将池层输出设置为 `Y`。无论 `X[i, j]` 和 `X[i, j + 1]` 的值是否不同，或者 `X[i, j + 1]` 和 `X[i, j + 2]` 的值是否不同，池层总是输出 `Y[i, j] = 1`。也就是说，使用 $2\times 2$ 最大池层，我们仍然可以检测卷积层识别的模式是否移动高度或宽度不超过一个元素。

在下面的代码中，我们在 `pool2d` 函数中实现了池层的正向传播。此函数与 :numref:`sec_conv_layer` 中的函数类似。但是，在这里我们没有内核，将输出计算为输入中每个区域的最大值或平均值。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

我们可以在 :numref:`fig_pooling` 中构建输入张量 `X`，以验证二维最大池层的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

此外，我们还试验了平均池层。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## 填充和步幅

与卷积图层一样，池化图层也可以更改输出形状。和以前一样，我们可以通过填充输入并调整步长来改变操作以达到所需的输出形状。我们可以通过深度学习框架中的内置二维最大池层演示填充和步幅在池图层中的使用。我们首先构建一个输入张量 `X`，其形状有四个维度，其中示例数和通道数都是 1。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

默认情况下，框架内置类中的实例中的步幅和池窗口具有相同的形状。下面，我们使用形状为 `(3, 3)` 的池窗口，因此默认情况下，我们得到的步幅形状为 `(3, 3)`。

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

可以手动指定步长和填充。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same',
                                   strides=2)
pool2d(X)
```

当然，我们可以指定一个任意的矩形池窗口，并分别指定高度和宽度的填充和步长。

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='same',
                                   strides=(2, 3))
pool2d(X)
```

## 多通道

处理多通道输入数据时，池化图层分别将每个输入通道集合起来，而不是像卷积图层中那样对通道上的输入进行汇总。这意味着池图层的输出通道数与输入通道数相同。下面，我们将在通道维度上连接张量 `X` 和 `X + 1`，以构建一个带有 2 个通道的输入。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
```

正如我们所看到的，池化后输出通道的数量仍然是 2。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
pool2d(X)
```

## 摘要

* 采用池化窗口中的输入元素，最大池化操作将最大值指定为输出，平均池化操作将平均值指定为输出。
* 集合层的主要优点之一是减轻卷积层对位置的过度敏感性。
* 我们可以为池图层指定填充和步长。
* 最大池和大于 1 的步长可用于减少空间尺寸（例如，宽度和高度）。
* 池图层的输出通道数与输入通道数相同。

## 练习

1. 你可以将平均池作为卷积层的特殊情况来实现吗？如果是这样，那么做。
1. 你可以实现最大池作为卷积层的特殊情况吗？如果是这样，那么做。
1. 池层的计算成本是多少？假设池层的输入大小为 $c\times h\times w$，池窗口的形状为 $p_h\times p_w$，填充为 $(p_h, p_w)$，步幅为 $(s_h, s_w)$。
1. 为什么您期望最大池和平均池的工作方式有所不同？
1. 我们是否需要单独的最小池层？你可以用另一个操作来替换它吗？
1. 平均和最大池之间是否有另一个操作可以考虑（提示：召回 softmax）？为什么它可能不那么受欢迎？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
