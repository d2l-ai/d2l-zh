# 密集连接的网络 (DenSet)

ResNet 显著改变了如何在深层网络中参数化功能的观点。* 密度 *（密集卷积网络）在某种程度上是此 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` 的逻辑扩展。要了解如何到达它，让我们采取一个小绕道数学。

## 从资源信息网到密森网

回想一下功能的泰勒扩展。对于点 $x = 0$，它可以写为

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

关键点在于它将函数分解为越来越高阶项。同样，ResNet 将函数分解为

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

也就是说，RENet 将 $f$ 分解为一个简单的线性项和一个更复杂的非线性项。如果我们想捕获（不一定添加）超出两个术语的信息，该怎么办？其中一个解决方案是密森网 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

如 :numref:`fig_densenet_block` 所示，信息网和电子信息网之间的主要区别在于，在后一种情况下，输出是 * 连接 *（用 $[,]$ 表示）而不是添加的。因此，在应用越来越复杂的函数序列之后，我们将执行从 $\mathbf{x}$ 到其值的映射：

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

最后，所有这些函数都在 MLP 中进行组合，以便再次减少特征的数量。在实现方面，这很简单：我们不是添加术语，而是连接它们。这个名字 DenSenet 源于变量之间的依赖关系图变得相当密集的事实。这样一个链的最后一层密集连接到所有以前的层。密集连接如 :numref:`fig_densenet` 所示。

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`

构成 DensenNet 的主要组件是 * 密集块 * 和 * 过渡层 *。前者定义输入和输出的串联方式，而后者控制通道的数量，以避免过大。

## 密集块

DenSet 使用经过修改的 RENet “批量规范化、激活和卷积” 结构（参见 :numref:`sec_resnet` 中的练习）。首先，我们实现这个卷积块结构。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

* 密集块 * 由多个卷积块组成，每个块使用相同数量的输出通道。然而，在正向传播中，我们在通道维度上连接每个卷积块的输入和输出。

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

在下面的示例中，我们定义了一个 `DenseBlock` 实例，该实例具有 2 个卷积块，包含 10 个输出通道。当使用带有 3 个通道的输入时，我们将获得一个带有 $3+2\times 10=23$ 通道的输出。卷积块通道的数量控制输出通道数量相对于输入通道数量的增长。这也称为 * 增长率 *。

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## 过渡层

由于每个密集块都会增加通道的数量，因此添加过多的信道将导致模型过于复杂。* 过渡层 * 用于控制模型的复杂性。它通过使用 $1\times 1$ 卷积层减少了通道数，并将平均池层的高度和宽度减半，步幅为 2，进一步降低了模型的复杂性。

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

将具有 10 个通道的过渡层应用于上一个示例中的密集块的输出。这将输出通道的数量减少到 10，并将高度和宽度降低一半。

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## 密森网模型

接下来，我们将构建一个 DenSet 模型。DenSet 首先使用与 ResNet 中相同的单卷积层和最大池池层。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

然后，与 ResNet 使用的残余块组成的四个模块类似，DensenNet 使用四个密集块。与 ResNet 类似，我们可以设置每个密集块中使用的卷积层数。在这里，我们将其设置为 4，与 :numref:`sec_resnet` 中的瑞士网 18 型号保持一致。此外，我们将密集块中卷积层的通道数量（即增长速率）设置为 32，因此每个密集块将添加 128 个通道。

在 ResNet 中，每个模块之间的高度和宽度会减少一个残余块，步幅为 2。在这里，我们使用过渡层将高度和宽度减半，并将通道数量减半。

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that haves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that haves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

与 ResNet 类似，全局池层和完全连接的层在末端连接以生成输出。

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## 培训

由于我们在这里使用更深层次的网络，因此在本节中，我们将将输入高度和宽度从 224 减少到 96 以简化计算。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* 在跨层连接方面，与 RENet 不同，将输入和输出相加在一起，DenSENet 在通道尺寸上连接输入和输出。
* 构成 DenSENet 的主要组件是密集块和过渡层。
* 在合成网络时，我们需要通过添加过渡层，再次缩小通道数量，从而保持对维度的控制。

## 练习

1. 为什么我们在过渡层中使用平均池而不是最大池？
1. DensenNet 文件中提到的一个优点是，其模型参数比 ResNet 的参数小。为什么会出现这种情况？
1. DenSenet 受到批评的一个问题是其高内存消耗。
    1. 这真的是这样吗？尝试将输入形状更改为 $224\times 224$ 以查看实际 GPU 内存消耗量。
    1. 你能想到一种减少内存消耗的替代方法吗？你需要如何改变框架？
1. 实施电子邮件 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` 号文件表 1 中提供的各种密码网版本。
1. 通过应用 Densenet 想法设计一个基于 MLP 的模型。将其应用于住房价格预测任务在 :numref:`sec_kaggle_house`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:
