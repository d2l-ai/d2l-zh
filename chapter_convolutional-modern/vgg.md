# 使用块的网络 (VGG)
:label:`sec_vgg`

虽然 AlexNet 提供了经验证据，表明深层有线电视网络可以取得良好的结果，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。在下面的章节中，我们将介绍一些常用于设计深层网络的启发式概念。

这一领域的进展反映了在芯片设计中，工程师从放置晶体管到逻辑元件，再到逻辑模块。同样，神经网络架构的设计也逐渐变得更加抽象，研究人员从单个神经元的思维转向整个层次，现在转变为块，重复层的模式。

使用块的想法首先出现在牛津大学的 [视觉几何组](http://www.robots.ox.ac.uk/~vgg/) (VGG)，在其同名的 *VGG* 网络中。通过使用循环和子例程，在任何现代深度学习框架的代码中轻松实现这些重复结构。

## VGG 块

经典 CNN 的基本构建块是以下顺序的序列：(i) 带填充以保持分辨率的卷积层；(ii) 非线性度，如 RELU，(iii) 池层，如最大池合层。一个 VGG 块由一系列卷积图层组成，后面是用于空间缩减采样的最大池化图层。在最初的 VGG 纸 :cite:`Simonyan.Zisserman.2014` 中，作者使用了带有 $3\times3$ 内核的卷积，填充为 1（保持高度和宽度）和 $2 \times 2$ 最大池步幅为 2（每个块后的分辨率减半）。在下面的代码中，我们定义了一个名为 `vgg_block` 的函数来实现一个 VGG 块。该函数采用两个参数对应于卷积层的数量 `num_convs` 和输出通道的数量 `num_channels`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## VGG 网络

与 AlexNet 和 Lenet 一样，VGG 网络可以分为两部分：第一部分主要由卷积层和池化层组成，第二部分由完全连接的层组成。这是在 :numref:`fig_vgg` 中描述的。

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

网络的卷积部分连续连接 :numref:`fig_vgg`（也在 `vgg_block` 函数中定义）的几个 VGG 块。下面的变量 `conv_arch` 由一个元组列表（每个块一个）组成，其中每个元组包含两个值：卷积层数和输出通道数，这正是调用 `vgg_block` 函数所需的参数。VGG 网络的完全连接部分与 AlexNet 中覆盖的部分相同。

原始 VGG 网络有 5 个卷积块，其中前两个区块各有一个卷积层，后三个区块各包含两个卷积层。第一个模块有 64 个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到 512。由于该网络使用 8 个卷积层和 3 个完全连接的层，因此它通常被称为 VGG-11。

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

下面的代码实现了 VGG-11。这是在 `conv_arch` 上执行 for 循环的一个简单问题。

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    # The convolutional part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

接下来，我们将构建一个高度和宽度为 224 的单通道数据示例，以观察每个图层的输出形状。

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

正如您所看到的，我们在每个区块的高度和宽度减半，最终达到高度和宽度 7，然后再拼合表示法以供网络完全连接的部分处理。

## 培训

由于 VGG-11 比 AlexNet 更具计算重量，因此我们建立了一个具有较少数量信道的网络。这是足够的时尚 MNist 培训.

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

除了使用略高的学习率外，模型培训流程与 :numref:`sec_alexnet` 中的 AlexNet 类似。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* VGG-11 使用可重复使用的卷积块构建一个网络。不同的 VGG 模型可通过每个模块中卷积层数和输出通道数量的差异来定义。
* 使用块会导致网络定义的非常紧凑的表示形式。它允许高效地设计复杂网络。
* 在他们的 VGG 论文中，西蒙扬和齐塞尔曼尝试了各种架构。特别是，他们发现，几层深卷积和窄卷积（即 $3 \times 3$）比较宽卷积较少的层更有效。

## 练习

1. 打印图层的尺寸时，我们只看到 8 个结果，而不是 11 个结果。剩余的 3 层信息去哪里？
1. 与 AlexNet 相比，VGG 在计算方面要慢得多，而且它还需要更多的 GPU 内存。分析出现这种情况的原因。
1. 尝试将图像的高度和宽度的时尚 MNist 从 224 改为 96。这对实验有什么影响？
1. 请参阅 VGG 纸张 :cite:`Simonyan.Zisserman.2014` 中的表 1，以构建其他常见模型，例如 VGG-16 或 VGG-19。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
