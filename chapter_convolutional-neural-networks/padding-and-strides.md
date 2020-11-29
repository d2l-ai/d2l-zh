# 填充和步幅
:label:`sec_padding`

在前面的示例 :numref:`fig_correlation` 中，输入的高度和宽度都为$3$，卷积内核的高度和宽度都为 $2$ ，生成的输出表示的维数为 $2\times2$。
正如我们在 :numref:`sec_conv_layer` 中所概括的那样，假设输入形状为 $n_h\times n_w$，卷积内核形状为 $k_h\times k_w$，那么输出形状将是 $(n_h-k_h+1) \times (n_w-k_w+1)$。
因此，卷积的输出形状取决于输入形状和卷积内核的形状。

还有什么技术会影响输出的大小呢？本节我们将介绍*填充*和*步幅*。假设以下情景：
- 有时，在应用了连续的卷积之后，我们最终得到的输出远小于输入大小。这是由于卷积内核的宽度和高度通常大于 $1$ 所导致的。比如，一个 $240 \times 240$ 像素的图像，经过 $10$ 层 $5 \times 5$ 的卷积后，将减少到 $200 \times 200$ 像素。如此一来，原始图像的边界丢失了许多有趣信息。 而*填充*（Padding） 是解决此问题最有效的方法。
- 有时，我们可能希望大幅降低图像的宽度和高度。例如，如果我们发现原始的输入分辨率十分冗余。 *步幅*则可以在这类情况下提供帮助。



## 填充

如上所述，在应用多层卷积时，我们常常丢失边缘像素。
解决这个问题的简单方法即为*填充*（padding）：在输入图像的长和宽填充元素（通常填充元素是 $0$ ）。
例如，在   :numref:`img_conv_pad`  中，我们将 $3 \times 3$ 输入填充到 $5 \times 5$，那么它的输出就增加为 $4 \times 4$。着色部分是第一个输出元素以及用于输出计算的输入和核张量元素：
$0\times0+0\times1+0\times2+0\times3=0$。

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

通常，如果我们添加 $p_h$ 行填充（大约一半在顶部，一半在底部）和 $p_w$ 列填充（左侧大约一半，右侧半），则输出形状将为

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)。$$

这意味着输出的高度和宽度将分别增加 $p_h$ 和 $p_w$。

在许多情况下，我们需要设置 $p_h=k_h-1$ 和 $p_w=k_w-1$，使输入和输出具有相同的高度和宽度。
这样可以在构建网络时更容易地预测每个图层的输出形状。假设 $k_h$ 是奇数，我们将在高度的两侧填充 $p_h/2$ 行。
如果 $k_h$ 是偶数，则一种可能性是在输入顶部填充 $\lceil p_h/2\rceil$ 行，在底部填充 $\lfloor p_h/2\rfloor$ 行。同理，我们填充宽度的两侧。

CNN 卷积内核的高度和宽度通常为奇数，例如 1、3、5 或 7。
选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。

此外，使用奇数核和填充也提供了文书上的便利。对于任何二维张量 `X`，当满足：
1. 内核的大小是奇数；
2. 所有侧面的填充行和列数相同；
3. 输出与输入具有相同高度和宽度
则可以得出：输出 `Y[i, j]` 是通过以输入 `X[i, j]` 为中心，和卷积核进行互相关计算。

比如，在下面的示例中，我们创建一个高度和宽度为 $3$ 的二维卷积图层，并在所有侧面填充 $1$ 个像素。给定高度和宽度为 $8$ 的输入，则输出的高度和宽度也是 $8$。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

当卷积内核的高度和宽度不同时，我们可以通过为高度和宽度设置不同的填充数字来使输出和输入具有相同的高度和宽度。

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='valid')
comp_conv2d(conv2d, X).shape
```

## 步幅

在计算互相关时，我们从输入张量左上角的卷积窗口开始，然后向下和向右滑动所有位置。在前面的示例中，我们默认每次滑动一个元素。但是，有时候，无论是为了计算效率还是因为我们希望缩减采样，我们一次移动窗口多个元素，跳过中间位置。

我们将每张幻灯片遍历的行数和列数称为 * stride*。到目前为止，我们已经使用了 1 的步幅，无论是高度还是宽度。有时候，我们可能需要使用较大的步幅。:numref:`img_conv_stride` 显示了二维交叉相关运算，步幅为 3，水平为 2。阴影部分是输出元素以及用于输出计算的输入和内核张量元素：$0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$。我们可以看到，当输出第一列的第二个元素时，卷积窗口向下滑动三行。当输出第一行的第二个元素时，卷积窗口会向右滑动两列。当卷积窗口继续向输入的右侧滑动两列时，没有输出，因为输入元素无法填充窗口（除非我们添加另一列填充）。

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

通常，当高度的步幅为 $s_h$ 且宽度的步幅为 $s_w$ 时，输出形状为

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

如果我们设置了 $p_h=k_h-1$ 和 $p_w=k_w-1$，则输出形状将简化为 $\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。更进一步，如果输入高度和宽度可以被高度和宽度的步幅整除，则输出形状将为 $(n_h/s_h) \times (n_w/s_w)$。

下面，我们将高度和宽度的步幅设置为 2，从而将输入高度和宽度减半。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

接下来，我们将看一个稍微复杂的例子。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

为了简洁起见，当输入高度和宽度两侧的填充数量分别为 $p_h$ 和 $p_w$ 时，我们称之为填充 $(p_h, p_w)$。具体来说，当 $p_h = p_w = p$ 时，填充是 $p$。当高度和宽度上的步幅分别为 $s_h$ 和 $s_w$ 时，我们称之为步幅 $(s_h, s_w)$。具体而言，当时的步幅为 $s_h = s_w = s$ 时，步幅为 $s$。默认情况下，填充为 0，步幅为 1。在实践中，我们很少使用不均匀的步幅或填充，也就是说，我们通常有 $p_h = p_w$ 和 $s_h = s_w$。

## 摘要

* 填充可以增加输出的高度和宽度。这通常用于为输出提供与输入相同的高度和宽度。
* 步幅可以降低输出的分辨率，例如，将输出的高度和宽度降低到输入高度和宽度的 $1/n$（$n$ 是一个大于 $1$ 的整数）。
* 填充和步幅可用于有效地调整数据的维度。

## 练习

1. 对于本节中的最后一个示例，使用数学计算输出形状，以查看它是否与实验结果一致。
1. 在本节中的实验中尝试其他填充和步幅组合。
1. 对于音频信号，步幅 2 对应什么？
1. 步幅大于 1 的计算优势是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:
