# 残余网络
:label:`sec_resnet`

随着我们设计越来越深入的网络，必须了解添加图层如何增加网络的复杂性和表现力。更重要的是设计网络的能力，其中添加图层可使网络更具表现力，而不仅仅是不同。为了取得一些进展，我们需要一些数学。

## 函数类

考虑 $\mathcal{F}$，即特定网络架构（以及学习速率和其他超参数设置）可以达到的功能类别。也就是说，对于所有 $f \in \mathcal{F}$，都存在一些参数（例如，权重和偏差），可以通过对合适的数据集进行培训来获得。让我们假设 $f^*$ 是我们真正想找到的 “真相” 功能。如果它是在 $\mathcal{F}$, 我们是在良好的状态，但通常我们不会很幸运.相反，我们会尝试找到一些 $f^*_\mathcal{F}$，这是我们在 $\mathcal{F}$ 内最好的选择。例如，给定一个包含要素 $\mathbf{X}$ 和标签 $\mathbf{y}$ 的数据集，我们可能会尝试通过解决以下优化问题来找到它：

$$f^*_\mathcal{F} := \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

只有合理的假设是，如果我们设计一个不同和更强大的架构 $\mathcal{F}'$，我们应该得到更好的结果。换句话说，我们预计 $f^*_{\mathcal{F}'}$ 比 $f^*_{\mathcal{F}}$ “更好”。但是，如果 $\mathcal{F} \not\subseteq \mathcal{F}'$ 没有保证，甚至会发生这种情况。事实上，$f^*_{\mathcal{F}'}$ 可能更糟糕。如 :numref:`fig_functionclasses` 所示，对于非嵌套函数类，较大的函数类并不总是靠近 “真实” 函数 $f^*$。例如，在 :numref:`fig_functionclasses` 的左边，虽然 $\mathcal{F}_3$ 比 $f^*$ 接近 $f^*$，但是 $\mathcal{F}_6$ 会移动，并且不能保证进一步增加复杂性可以减少距离，从 $f^*$。使用嵌套函数类，其中 $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$ 位于 :numref:`fig_functionclasses` 右侧，我们可以避免非嵌套函数类中的上述问题。

![For non-nested function classes, a larger (indicated by area) function class does not guarantee to get closer to the "truth" function ($f^*$). This does not happen in nested function classes.](../img/functionclasses.svg)
:label:`fig_functionclasses`

因此，只有当较大的函数类包含较小的函数类时，我们才能保证增加它们会严格增加网络的表达能力。对于深度神经网络，如果我们可以将新添加的层训练成一个身份函数 $f(\mathbf{x}) = \mathbf{x}$，新模型将与原始模型一样有效。由于新模型可能会得到更好的解决方案来拟合训练数据集，因此添加的图层可能会更轻松地减少训练错误。

这是他等人的问题. 在工作非常深的计算机视觉模型 :cite:`He.Zhang.Ren.ea.2016` 时考虑.他们提出的 * 残留网络 * (*Resnet*) 的核心是，每个额外的图层都应更容易地包含身份函数作为其元素之一。这些考虑相当深刻，但它们导致了一个令人惊讶的简单解决方案，一个 * 残余块 *。凭借它，RESnet 在 2015 年赢得了 iMageNet 大规模视觉识别挑战赛。该设计对如何构建深度神经网络具有深远的影响。

## 残余块

让我们专注于神经网络的局部部分，如 :numref:`fig_residual_block` 所示。用 $\mathbf{x}$ 表示输入。我们假设我们希望通过学习获得的所需底层映射是 $f(\mathbf{x})$，用作顶部激活函数的输入。在 :numref:`fig_residual_block` 的左侧，虚线框中的部分必须直接了解映射 $f(\mathbf{x})$。在右侧，点线框中的部分需要了解 * 残差映射 * $f(\mathbf{x}) - \mathbf{x}$，这是残余块如何派生其名称。如果身份映射 $f(\mathbf{x}) = \mathbf{x}$ 是所需的底层映射，残差映射更容易学习：我们只需要将虚线框中的上层权重层（例如，完全连接的层和卷积层）的权重和偏差推到零。:numref:`fig_residual_block` 中的右图显示了 ResNet 的 * 残余块 *，其中将层输入 $\mathbf{x}$ 传送到加法运算符的实线称为 * 残差连接 *（或 * 快捷连接 *）。使用残余块时，输入可以通过跨层的残余连接更快地传播。

![A regular block (left) and a residual block (right).](../img/residual-block.svg)
:label:`fig_residual_block`

RENet 采用 VGG 的完整卷积层设计。残余模块有两个 $3\times 3$ 卷积层，具有相同数量的输出通道。每个卷积层后跟一个批量归一化层和一个 RELU 激活函数。然后，我们跳过这两个卷积操作，直接在最终的 RELU 激活函数之前添加输入。这种设计要求两个卷积图层的输出必须与输入的形状相同，以便它们可以相加在一起。如果我们想要更改通道数量，我们需要引入额外的 $1\times 1$ 卷积层，以便将输入转换为加法操作所需的形状。让我们看看下面的代码。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

此代码生成两种类型的网络：一种是在 `use_1x1conv=False` 应用 RELU 非线性度之前将输入添加到输出中，另一种是我们在添加前通过 $1 \times 1$ 卷积调整通道和分辨率。:numref:`fig_resnet_block` 说明了这一点：

![ResNet block with and without $1 \times 1$ convolution.](../img/resnet-block.svg)
:label:`fig_resnet_block`

现在让我们看一下输入和输出形状相同的情况。

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

我们还可以选择将输出高度和宽度减半，同时增加输出通道的数量。

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

## 资源网模型

RENet 的前两层与我们前面描述的 Googlenet 相同：$7\times 7$ 卷积层有 64 个输出通道，步幅为 2 的最大池层后面是 $3\times 3$，步幅为 2。不同之处在于在 ResNet 中每个卷积图层之后添加的批量归一化图层。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Googlenet 使用四个模块组成的模块。然而，ResNet 使用由残余块组成的四个模块，每个模块都使用具有相同数量输出通道的几个残余块。第一个模块中的通道数与输入通道数相同。由于已经使用了步幅为 2 的最大池层，因此没有必要减少高度和宽度。在每个后续模块的第一个残余块中，通道数量与前一个模块相比增加了一倍，并且高度和宽度减半。

现在，我们实现这个模块。请注意，在第一个模块上执行了特殊处理。

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

然后，我们将所有模块添加到 ResNet 中。在这里，每个模块使用两个残余块。

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

最后，就像 Googlenet 一样，我们添加了一个全局平均池图层，然后是完全连接的图层输出。

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that we define this as a function so we can reuse later and run it
# within `tf.distribute.MirroredStrategy`'s scope to utilize various
# computational resources, e.g. GPUs. Also note that even though we have
# created b1, b2, b3, b4, b5 but we will recreate them inside this function's
# scope instead
def net():
    return tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

每个模块中有 4 个卷积层（不包括 $1\times 1$ 卷积层）。再加上第一个 $7\times 7$ 卷积层和最后一个完全连接的层，共有 18 个层。因此，这种模式通常被称为 RENET-18。通过在模块中配置不同数量的通道和残余块，我们可以创建不同的 ResNet 模型，例如更深的 152 层 Resnet-152。虽然 ResNet 的主要体系结构与 Googlenet 的体系结构相似，但 ResNet 的结构更简单、更容易修改。所有这些因素都导致人们迅速和广泛地使用了资源信息网。

![The ResNet-18 architecture.](../img/resnet18.svg)
:label:`fig_resnet18`

在培训 ResNet 之前，让我们观察在 ResNet 中不同模块之间输入形状如何变化。与之前的所有体系结构一样，分辨率随着通道数量的增加而降低，直到全局平均池图层聚合所有要素为止。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## 培训

我们训练 Resnet 上的时尚 MNist 数据集, 就像以前一样.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* 嵌套函数类是可取的。在深度神经网络中学习一个额外的层作为身份函数（尽管这是一个极端情况）应该很容易。
* 残差映射可以更容易地学习身份函数，例如将权重层中的参数推到零。
* 我们可以通过具有残余块来训练一个有效的深度神经网络。输入可以通过跨层的残余连接更快地向前传播。
* ResNet 对后续深度神经网络的设计具有重大影响，无论是卷积性还是顺序性质。

## 练习

1. :numref:`fig_inception` 中的 “启动” 块与剩余块之间的主要区别是什么？删除 “启动” 块中的某些路径后，它们彼此之间的关系如何？
1. 请参阅 RENet 文件 :cite:`He.Zhang.Ren.ea.2016` 中的表 1，以实现不同的变体。
1. 对于更深层次的网络，ResNet 引入了一种 “瓶颈” 体系结构，以降低模型复杂性。尝试实现它。
1. 在后续版本的 ResNet 中，作者将 “卷积、批量规范化和激活” 结构更改为 “批量规范化、激活和卷积” 结构。自己做这一改进。有关详细信息，请参阅 :cite:`He.Zhang.Ren.ea.2016*1` 中的图 1。
1. 为什么我们不能在没有绑定的情况下增加函数的复杂性，即使函数类是嵌套的？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/333)
:end_tab:
