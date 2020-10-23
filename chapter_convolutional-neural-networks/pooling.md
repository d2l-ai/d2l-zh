# 池化
:label:`sec_pooling`

通常，当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率，聚合信息，以便我们在网络中走得越高，每个隐藏节点对其敏感的接受字段（输入中）就越大。

通常我们的最终任务会询问一些关于图像的全球性问题，例如，* 它是否包含猫？* 因此，通常我们的最终图层的单位应该对整个输入敏感。通过逐步聚合信息，生成更粗糙和更粗糙的地图，我们实现了最终学习全局表示的目标，同时将卷积图层的所有优势保留在处理的中间层。

此外，在检测较低级别的要素（例如边（如 :numref:`sec_conv_layer` 中所述）时，我们通常希望我们的制图表达在转换中有些不变。例如，如果我们使用黑白之间的清晰划分图像 `X`，并将整个图像向右移动一个像素，即 `Z[i, j] = X[i, j + 1]`，那么新图像 `Z` 的输出可能会有很大的不同。边缘将移动一个像素。在现实中，对象几乎没有完全发生在同一个地方。事实上，即使有一个三脚架和一个固定的物体，由于快门移动而导致的相机振动也可能会将所有内容移动一个像素左右（高端摄像机加载了特殊功能来解决这个问题）。

本节介绍 * 合并图层 *，它有双重目的，即减少卷积图层对位置的敏感度和空间缩减采样表示的敏感度。

## 最大池和平均池

与卷积图层一样，*pooling* 运算符由一个固定形状的窗口组成，该窗口根据其步幅滑动在输入中的所有区域上，计算由固定形状窗口（有时称为 * 池化窗口 *）遍历的每个位置的单个输出。但是，与卷积层中输入和内核的相互相关计算不同，池层不包含任何参数（没有 *kernel*）。相反，池运算符是确定性的，通常计算池窗口中元素的最大值或平均值。这些操作分别称为 * 最大池 *（短期 * 最大池 *）和 * 平均池 *。

在这两种情况下，与互相关运算符一样，我们可以将池窗口视为从输入张量的左上角开始，从左到右，从上到下滑动在输入张量之间。在池窗口点击的每个位置，它都会计算窗口中输入子张量的最大值或平均值，具体取决于是使用了最大池还是平均池。

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` 中的输出张量的高度为 2，宽度为 2。这四个元素是从每个池窗口中的最大值派生的：

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

池窗口形状为 $p \times q$ 的池合图层称为 $p \times q$ 池合图层。池化操作称为 $p \times q$ 池。

让我们回到本节开头提到的对象边缘检测示例。现在我们将使用卷积层的输出作为 $2\times 2$ 最大池的输入。将卷积图层输入设置为 `X`，将池图层输出设置为 `Y`。无论这两个值是否不同，还是 `X[i, j + 1]` 和 `X[i, j + 1]` 的值是不同的，池层总是会输出 `Y[i, j] = 1`。也就是说，使用 $2\times 2$ 最大池层，我们仍然可以检测卷积层识别的模式是否在高度或宽度上移动不超过一个元素。

在下面的代码中，我们在 `pool2d` 函数中实现了池层的正向传播。此函数与 :numref:`sec_conv_layer` 中的函数类似。但是，这里我们没有内核，将输出计算为输入中每个区域的最大值或平均值。

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

我们可以在 :numref:`fig_pooling` 中构建输入张量 `X`，验证二维最大池层的输出。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

此外，我们还试验平均池图层。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## 填充和步幅

与卷积图层一样，合并图层也可以更改输出形状。和以前一样，我们可以通过填充输入并调整步幅来改变操作以达到所需的输出形状。我们可以通过深度学习框架内置的二维最大池层来演示在池中使用填充和步伐。我们首先构建一个输入张量 `X`，其形状有四个维度，其中示例数量和通道数量均为 1。

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

默认情况下，框架内置类中的实例中的步长和池窗口具有相同的形状。下面，我们使用形状为 `(3, 3)` 的池合窗口，因此默认情况下，我们得到的步幅形状为 `(3, 3)`。

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

步幅和填充可以手动指定。

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

## 多个通道

在处理多通道输入数据时，池合图层将每个输入通道单独合并，而不是像卷积层一样在通道上对输入进行汇总。这意味着池层的输出通道数与输入通道数相同。下面，我们将在通道尺寸上连接张量 `X` 和 `X + 1`，以构建具有 2 个通道的输入。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
```

正如我们所看到的，池后输出通道的数量仍然是 2。

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

* 使用池化窗口中的输入元素，最大池化操作会将最大值指定为输出，平均池操作将平均值指定为输出。
* 池合层的主要优点之一是减轻卷积层对位置的过度敏感度。
* 我们可以指定池合图层的填充和步长。
* 最大池化，加上大于 1 的步幅可用于减少空间维度（例如，宽度和高度）。
* 池层的输出通道数与输入通道数相同。

## 练习

1. 你能否将平均池作为卷积层的特殊情况实现？如果是这样，请做到这一点。
1. 你可以实现最大池作为卷积层的特殊情况吗？如果是这样，请做到这一点。
1. 池图层的计算成本是多少？假设池图层的输入大小为 $c\times h\times w$，则池窗口的形状为 $p_h\times p_w$，填充为 $(p_h, p_w)$，步幅为 $(s_h, s_w)$。
1. 为什么您期望最大池和平均池的工作方式不同？
1. 我们是否需要一个单独的最小池层？您可以用另一个操作替换它吗？
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
