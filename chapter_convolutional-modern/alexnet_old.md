# 深度卷积神经网络
:label:`sec_alexnet`

尽管在推出 LenNet 之后，有线电视网络在计算机视觉和机器学习社区中是众所周知的，但它们并没有立即占据该领域的主导地位。尽管 LenNet 在早期的小数据集上取得了良好的成果，但是在更大、更现实的数据集上培训有线电视网络的性能和可行性尚待确定。事实上，在 1990 年代初到 2012 年分水岭结果之间的大部分时间里，神经网络往往被其他机器学习方法（如支持向量机）所超越。

对于计算机视觉来说，这种比较可能是不公平的。也就是说，尽管卷积网络的输入由原始或轻度处理（例如，通过居中）像素值组成，但从业人员永远不会将原始像素馈入传统模型。相反，典型的计算机视觉管道由手动设计要素提取管道组成。这些功能不是 * 学习功能 *，而是 * 精心制作的 *。大部分进步都来自于对功能有更多聪明的想法，而学习算法往往被降级为事后想法。

虽然 1990 年代有一些神经网络加速器，但它们还不够强大，无法制作具有大量参数的深度多通道多层 CNN。此外，数据集仍然相对较小。除了这些障碍之外，训练神经网络的关键技巧包括参数初始化启发式算法、随机梯度下降的智能变体、非压缩激活函数以及有效正则化技术仍然缺少。

因此，经典管道不是训练 * 端到端 *（像素到分类）系统，而是看起来更像这样：

1. 获取一个有趣的数据集。早期，这些数据集需要昂贵的传感器（当时，1 百万像素的图像是最先进的）。
2. 根据对光学、几何、其他分析工具的一些知识，以及偶尔基于幸运研究生的偶然发现，使用手工制作的要素对数据集进行预处理。
3. 通过一组标准的要素提取器提供数据，例如 SIFT（比例不变要素转换）:cite:`Lowe.2004`、SURF（加速鲁棒特征）:cite:`Bay.Tuytelaars.Van-Gool.2006` 或任意数量的其他手动调谐管道。
4. 将生成的表示转储到您最喜欢的分类器（可能是线性模型或内核方法），以训练分类器。

如果你和机器学习研究人员交谈，他们认为机器学习既重要又美观。优雅的理论证明了各种分类器的特性。机器学习领域蓬勃发展、严格且非常有用。但是，如果你和计算机视觉研究员交谈，你会听到一个非常不同的故事。他们会告诉你，图像识别的肮脏真相是特征，而不是学习算法，推动了进步。计算机视觉研究人员有理由认为，稍大或更干净的数据集或稍微改进的要素提取管道对于最终准确性而言比任何学习算法更重要。

## 学习表示

另一种解决事态状况的方法是，管道中最重要的部分是代表性。直到 2012 年，表示是以机械方式计算的。事实上，设计一组新的特征函数，改进结果，并编写该方法是一个突出的文章类型。SIFT :cite:`Lowe.2004`、SUG（方向梯度的直方图）、:cite:`Dalal.Triggs.2005`、[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) 和类似的特征提取器统治了根源。

另一组研究人员，包括延恩·勒恩，杰夫·欣顿，约书亚·本吉奥，安德鲁·吴顺一，阿马里和尤尔根·施密杜伯，有不同的计划。他们认为，特征本身应该被学习。此外，他们认为，为了相当复杂，这些特征应该由多个共同学习层组成，每个层都有可学习的参数。对于图像，最低层可能会检测边缘、颜色和纹理。事实上，亚历克斯·克里热夫斯基，伊利亚·苏茨凯维尔和杰夫·欣顿提出了一个有线电视新闻网的新变体，
*阿列克尼 *,
，在 2012 年国际网站挑战赛中取得了出色的表现。AlexNet 被命名为亚历克斯·克里热夫斯基，突破性的 ImageNet 分类纸 :cite:`Krizhevsky.Sutskever.Hinton.2012` 的第一作者。

有趣的是，在网络的最低层，该模型学习了类似于一些传统过滤器的特征提取器。:numref:`fig_filters` 是从 AlexNet 纸 :cite:`Krizhevsky.Sutskever.Hinton.2012` 重现的，并描述了较低级别的图像描述符。

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

网络中较高的图层可能基于这些制图表达来表示较大的结构，如眼睛、鼻子、草叶等。更高的层可能代表整个对象，如人、飞机、狗或飞盘。最终，最终的隐藏状态会学习图像的紧凑表示形式，汇总其内容，从而可以轻松分离属于不同类别的数据。

尽管多层 CNN 的最终突破在 2012 年出现，但一批核心研究人员致力于这一想法，多年来试图学习视觉数据的层次表示。2012 年的最终突破可归因于两个关键因素。

### 缺少成分：数据

具有多层的深度模型需要大量数据才能进入系统，其性能显著优于基于凸优化的传统方法（例如线性方法和核方法）。然而，鉴于计算机存储容量有限、传感器的相对开支以及 1990 年代研究预算相对较紧，大多数研究依赖于微小的数据集。许多论文涉及 UCI 数据集集，其中许多仅包含数百或（少数）数千张图像，分辨率低的非自然环境中捕获的图像。

2009 年，IMageNet 数据集发布，挑战研究人员从 100 万个示例中学习模型，每个 1000 个不同类别的对象中学习模型。由李飞飞率领的研究人员介绍了此数据集，他利用 Google 图像搜索为每个类别预先筛选大型候选集，并利用亚马逊机械土耳其人众包管道确认每张图片是否属于相关类别。这种规模是前所未有的。被称为 iMagenet 挑战赛的相关竞赛推动了计算机视觉和机器学习研究的前进，挑战研究人员确定哪些模型在比学者之前考虑的更大规模上表现最佳。

### 缺少成分：五金

深度学习模型是计算周期的贪婪消费者。训练可能需要数百个时代，每次迭代都需要通过多层计算昂贵的线性代数操作传递数据。这也是在 20 世纪 90 年代和 21 世纪初期，基于更高效优化凸目标的简单算法的主要原因之一。

*图形处理单元 * (GPU) 被证明是游戏改变者
使深度学习变得可行。这些芯片早已开发用于加速图形处理，从而使计算机游戏受益。特别是，它们针对大量计算机图形任务所需的高吞吐量 $4 \times 4$ 矩阵矢量产品进行了优化。幸运的是，这种数学与计算卷积层所需的数学非常相似。在那个时候，NVIDIA 和 ATI 已经开始针对通用计算操作优化 GPU，甚至将它们作为 * 通用 GPU * (GPGPU) 进行市场推广。

为了提供一些直觉，请考虑现代微处理器 (CPU) 的核心。每个核心都相当强大，以高时钟频率运行，并运动大型缓存（高达几兆字节 L3）。每个核心都非常适合执行各种指令，包括分支预测变量、深度管道以及其他铃声和口哨，使其能够运行各种程序。然而，这种明显的强度也是其跟腱的脚跟：通用核心的建造成本非常昂贵。它们需要大量的芯片面积、复杂的支持结构（内存接口、内核之间的缓存逻辑、高速互连等），并且在任何单个任务中都相对较差。现代笔记本电脑有多达 4 个内核，即使是高端服务器也很少超过 64 个内核，只是因为它不具有成本效益。

相比之下，GPU 由 $100 \sim 1000$ 小型处理元件组成（NVIDIA、ATI、ARM 和其他芯片供应商之间的细节略有不同），通常被分组为较大的组（NVIDIA 称之为翘曲）。虽然每个内核都相对较弱，有时甚至在 1GHz 以下的时钟频率下运行，但正是这些内核的总数使得 GPU 的数量级比 CPU 快。例如，NVIDIA 最近一代沃尔特为专用指令提供高达 120 个 TFlop 的每个芯片（对于更多通用的指令，最多可提供 24 个 TFlop），而目前 CPU 的浮点性能并未超过 1 TFlop。为什么这样做是可能的原因实际上很简单：首先，功耗趋于随着时钟频率增长 * 二次 *。因此，对于运行速度快 4 倍的 CPU 内核（典型数字）的功耗预算，您可以使用 16 个 GPU 内核，速度为 $1/4$，从而产生的性能是 $16 \times 1/4 = 4$ 倍。此外，GPU 内核更简单（事实上，很长一段时间内它们甚至没有 * 能够执行通用代码），这使得它们更加节能。最后，深度学习中的许多操作都需要高内存带宽。同样，GPU 在这里闪耀着至少是 CPU 数量的 10 倍的总线。

回到二零一二年。当亚历克斯·克里热夫斯基和伊利亚 Sutskever 实施了可以在 GPU 硬件上运行的深层 CNN 时，一个重大的突破就来了。他们意识到，CNN 中的计算瓶颈、卷积和矩阵乘法都是可以在硬件中并行化的操作。它们使用两台 NVIDIA GTX 580 和 3GB 内存，实现了快速卷积。代码 [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) 足够好，几年来它是行业标准，并为深度学习热潮的头几年提供了动力。

## AlexNet

AlexNet 采用 8 层有线电视新闻网，赢得了 ImageNet 2012 大规模视觉识别挑战赛的优势。这个网络首次表明，通过学习获得的功能可以超越人工设计的功能，打破了以前的计算机视觉模式。

AlexNet 和 Lenet 的架构非常相似，如 :numref:`fig_alexnet` 所示。请注意，我们提供了一个稍微简化的 AlexNet 版本，去除了 2012 年使模型适合两个小 GPU 所需的一些设计怪癖。

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet 和 Lenet 的设计理念非常相似，但也有显著的差异。首先，AlexNet 比相对较小的 Lenet5 要深得多。AlexNet 由八个层组成：五个卷积层、两个完全连接的隐藏层和一个完全连接的输出层。其次，AlexNet 使用 RELU 而不是西格体作为激活函数。让我们深入研究下面的细节。

### 建筑

在 AlexNet 的第一层中，卷积窗口形状为 $11\times11$。由于 iMagenet 中的大多数图像比 MNIST 图像高出 10 倍以上，因此 iMagenet 数据中的对象往往占用更多像素。因此，需要一个较大的卷积窗口来捕获对象。第二层中的卷积窗口形状会减少到 $5\times5$，然后是 $3\times3$。此外，在第一个、第二个和第五个卷积图层之后，网络添加了窗口形状为 $3\times3$ 且步幅为 2 的最大池层。此外，AlexNet 的卷积通道比 Lenet 多十倍。

在最后一个卷积层之后，有两个具有 4096 输出的完全连接层。这两个巨大的完全连接层产生了近 1 GB 的模型参数。由于早期 GPU 中的内存有限，原始 AlexNet 使用了双数据流设计，因此两个 GPU 中的每个 GPU 都可以仅负责存储和计算模型的一半。幸运的是，GPU 内存现在相对丰富，所以现在我们很少需要跨 GPU 分解模型（我们的 AlexNet 模型版本在这方面与原始论文偏离）。

### 激活函数

此外，AlexNet 还将符号激活函数改为更简单的 RELU 激活函数。一方面，RELU 激活函数的计算更简单。例如，它没有在 sigmoid 激活函数中找到的指数运算。另一方面，在使用不同的参数初始化方法时，RELU 激活函数使模型训练变得更加容易。这是因为，当 sigmoid 激活函数的输出非常接近 0 或 1 时，这些区域的梯度几乎为 0，因此反向传播无法继续更新某些模型参数。相比之下，在正时间间隔内，RELU 激活函数的梯度始终为 1。因此，如果模型参数未正确初始化，sigmoid 函数可能会在正区间内获得几乎 0 的梯度，因此无法对模型进行有效训练。

### 容量控制和预处理

AlexNet 通过压差 (:numref:`sec_dropout`) 控制完全连接层的模型复杂度，而 Lenet 仅使用权重衰减。为了进一步增加数据，AlexNet 的训练循环添加了大量图像增强功能，例如翻转、剪切和颜色更改。这使得模型更加坚固，样本数量越大，有效地减少了过度拟合。我们将在 :numref:`sec_image_augmentation` 中更详细地讨论数据增强问题。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

我们构建了一个高度和宽度均为 224 的单通道数据示例，以观察每个图层的输出形状。它与 :numref:`fig_alexnet` 中的亚历克斯网络架构相匹配。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)
```

## 读取数据集

虽然 AlexNet 是在纸上进行的 Imagenet 训练，但我们在这里使用时尚 MNist，因为训练 IMagenet 模型以融合可能需要几个小时或几天，甚至在现代 GPU 上。其中一个应用 AlexNet 直接在时尚 MNist 上的问题是，它的图像具有较低的分辨率（$28 \times 28$ 像素）比 iMagenet 图像。为了使事情发挥作用，我们将它们升级到 $224 \times 224$（通常不是一个聪明的实践，但我们在这里这样做是为了忠实于 AlexNet 体系结构）。我们使用 `d2l.load_data_fashion_mnist` 函数中的 `resize` 参数执行此调整大小。

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 培训

现在，我们可以开始训练 AlexNet。与 :numref:`sec_lenet` 的 Lenet 相比，这里的主要变化是，由于网络更深、更宽、更高的图像分辨率和更昂贵的卷积，使用更低的学习率和更慢的训练速度。

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* AlexNet 的结构与 Lenet 的结构相似，但使用更多卷积层和更大的参数空间来适应大规模 iMagenet 数据集。
* 今天 AlexNet 已经被更有效的体系结构所超越，但它是从浅层网络到深层网络的关键一步，现在使用这些网络。
* 尽管 AlexNet 的实现似乎只有几条线路比在 LenNet 中更多，但学术界花了许多年时间才接受这一概念变化，并利用其出色的实验结果。这也是由于缺乏有效的计算工具。
* 在计算机视觉任务中实现卓越性能的另一个关键步骤是辍学、RELU 和预处理。

## 练习

1. 尝试增加周期的数量。与 LenNet 相比，结果有什么不同？为什么？
1. AlexNet 可能是时尚 MNist 数据集过于复杂.
    1. 尝试简化模型以加快训练速度，同时确保精度不会显著下降。
    1. 设计一个更好的模型，直接适用于 $28 \times 28$ 图像。
1. 修改批处理大小，并观察精度和 GPU 内存的变化。
1. 分析 AlexNet 的计算性能。
    1. AlexNet 内存占用空间的主要部分是什么？
    1. 在 AlexNet 中计算的主要部分是什么？
    1. 计算结果时的内存带宽如何？
1. 将辍学和 RELU 应用于 Lenet-5。它是否有所改善？预处理怎么样？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
