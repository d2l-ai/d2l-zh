# 具有并行连接的网络（GooGlenet）
:label:`sec_googlenet`

2014 年，*Googlenet * 赢得了 Imagenet 挑战赛，提出了一个结合 Nin 的优势和重复区块 :cite:`Szegedy.Liu.Jia.ea.2015` 的范例的结构。本文的重点之一是解决哪些大小卷积内核最适合的问题。毕竟，以前流行的网络采用的选择小至 $1 \times 1$，大到 $11 \times 11$。本文中的一个见解是，有时使用不同大小的内核组合是有利的。在本节中，我们将介绍 GooGlenet，介绍原始模型的稍微简化版本：我们省略了一些特别功能，这些功能是为了稳定训练而添加的，但现在没有必要使用更好的训练算法。

## 初始模块

GooGlenet 中的基本卷积块被称为 * 初始块 *，可能是由于电影 * 潜入 *（“我们需要更深入”）引用的引用而命名的（“我们需要更深入”），它推出了一种病毒模因。

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

如 :numref:`fig_inception` 所示，初始模块由四条并行路径组成。前三条路径使用窗口大小为 $1\times 1$、$3\times 3$ 和 $5\times 5$ 的卷积图层从不同空间大小中提取信息。中间的两条路径在输入上执行 $1\times 1$ 卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用 $3\times 3$ 最大池层，然后是一个 $1\times 1$ 卷积层来更改通道数。四个路径都使用适当的填充来为输入和输出提供相同的高度和宽度。最后，沿着每条路径的输出沿通道尺寸串联，并组成块的输出。通常调整的 Uperd 模块的超参数是每个图层的输出通道数。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

为了获得一些直觉，了解为什么这个网络运行如此良好，请考虑一下滤波器的组合。他们探索各种滤镜尺寸的图像。这意味着不同大小的过滤器可以有效地识别不同范围的细节。同时，我们可以为不同的过滤器分配不同数量的参数。

## Googlenet 模型

如 :numref:`fig_inception_full` 所示，GooGlenet 使用总共 9 个启动块和全球平均池来生成其估计值。启动模块之间的最大池可降低维度。第一个模块类似于 AlexNet 和 Lenet。块堆栈是从 VGG 继承的，而全局平均池最终避免了一堆完全连接的层。

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

我们现在可以一块一块地实现 Googlenet。第一个模块使用 64 通道 $7\times 7$ 卷积层。

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第二个模块使用两个卷积层：第一个是 64 通道 $1\times 1$ 卷积层，然后是将通道数量翻三倍的 $3\times 3$ 卷积层。这对应于 “启动” 块中的第二条路径。

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第三个模块连接两个完整的启动模块串联。第一个启动模块的输出通道数为 $64+128+32+32=256$，四个路径之间的输出通道数量比为 $64:128:32:32=2:4:1:1$。第二个和第三个路径首先将输入通道的数量分别减少到 $96/192=1/2$ 和 $16/192=1/12$，然后连接第二个卷积层。第二个启动模块的输出通道数增加到 $128+192+96+64=480$，四个路径之间的输出通道数量比为 $128:192:96:64 = 4:6:3:2$。第二条和第三条路径首先会将输入通道的数量分别减少到 $128/256=1/2$ 和 $32/256=1/8$。

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第四个模块比较复杂。它可以串联连接五个启动模块，并分别具有 $192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$ 和 $256+320+128+128=832$ 输出通道。分配给这些路径的通道数与第三个模块中的信道数相似：带有 $3\times 3$ 卷积层的第二条路径输出最大数量的通道，其次是仅有 $1\times 1$ 卷积层的第一条路径，第三条路径为 $5\times 5$ 卷积层，第四条路径，其中包含 $3\times 3$ 最大池层。第二个和第三个路径将首先根据比例减少通道数。这些比率在不同的 “启动” 模块中略有不同。

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

第五个模块具有两个启动模块，具有 $256+320+128+128=832$ 和 $384+384+128+128=1024$ 输出通道。分配给每个路径的通道数与第三个和第四个模块中的通道数相同，但在特定值上有所不同。应该注意的是，第五个块后跟输出层。此模块使用全局平均池图层将每个通道的高度和宽度更改为 1，就像在 NIn 中一样。最后，我们将输出转换为二维数组，然后是一个完全连接的图层，其输出数是标注分类的数量。

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveMaxPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

GooGlenet 模型在计算上很复杂，因此修改通道数量并不像 VGG 中那么容易。为了有一个合理的训练时间在时尚 MNist, 我们将输入高度和宽度从 224 减少到 96.这简化了计算。各种模块之间输出形状的变化如下所示。

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## 培训

和以前一样，我们使用时尚 MNist 数据集训练我们的模型。在调用训练过程之前，我们将其转换为 $96 \times 96$ 像素分辨率。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* “启动” 块等效于具有四条路径的子网。它通过不同窗口形状的卷积层和最大池化层并行提取信息。$1 \times 1$ 卷积可降低每像素级别的通道维数。最大池会降低分辨率。
* Googlenet 将多个精心设计的初始模块与其他层串联。通过在 iMagenet 数据集上进行大量实验，可以获得 “启动” 模块中分配的通道数量的比率。
* GooGlenet 及其后续版本是 iMagenet 上最有效的模型之一，它提供了类似的测试精度和较低的计算复杂性。

## 练习

1. 有几个迭代的 GooGlenet。尝试实现和运行它们。其中一些措施包括以下内容：
    * 添加批量归一化图层 :cite:`Ioffe.Szegedy.2015`，如下文所述。
    * 对 “启动” 模块进行调整。
    * 使用标签平滑进行模型正则化 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`。
    * 将其包含在剩余连接 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017` 中，如下文所述。
1. Googlenet 工作的最小图像大小是多少？
1. 将 AlexNet、VGG 和 NIN 的模型参数大小与 Googlenet 进行比较。后两个网络架构如何显著减少模型参数大小？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
