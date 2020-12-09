# 深度卷积神经网络之AlexNet
:label:`sec_alexnet`

尽管卷积神经网络在LeNet的引入后在计算机视觉和机器学习领域中很有名，但它们并没有立即主导这个领域。虽然LeNet在早期的小数据集上取得了很好的效果，但是在更大、更真实的数据集上训练卷积神经网络的性能和可行性还有待于确定。事实上，在上世纪90年代初到2012年分水岭结果之间的大部分时间里，神经网络往往被其他机器学习方法所超越，比如支持向量机。

对于计算机视觉来说，这种比较也许不公平。也就是说，尽管卷积网络的输入由原始像素值或经过轻微处理（例如通过居中）的像素值组成，但从业者永远不会将原始像素输入到传统模型中。相反，典型的计算机视觉管道由人工工程特征提取管道组成。这些功能不是*学习功能*而是*精心设计的*。大部分的进步来自于对特性有了更聪明的想法，学习算法常常被后置之脑后。

虽然上世纪90年代就有了一些神经网络加速器，但它们还不足以制造出具有大量参数的深层多通道多层卷积神经网络。此外，数据集仍然相对较小。除了这些障碍，训练神经网络的关键技巧，包括参数初始化启发式、随机梯度下降的巧妙变体、非压缩激活函数和有效的正则化技术仍然缺失。

因此，与训练*端到端*（像素到分类）系统不同，经典管道看起来更像这样：

1. 获取一个有趣的数据集。在早期，这些数据集需要昂贵的传感器（当时，100万像素的图像是最先进的）。
2. 根据光学、几何学和其他分析工具的一些知识，以及偶尔对幸运的研究生的偶然发现，用手工制作的特征对数据集进行预处理。
3. 通过一组标准的特征提取程序（如SIFT（标度不变特征变换）:cite:`Lowe.2004`、SURF（加速鲁棒特征）:cite:`Bay.Tuytelaars.Van-Gool.2006`或任何数量的其他手动调节的管道来输入数据。
4. 将结果表示转储到您最喜欢的分类器中，例如线性模型或内核方法，以训练分类器。

如果你和机器学习研究人员交谈，他们相信机器学习既重要又美丽。优雅的理论证明了各种量词的性质。机器学习是一个蓬勃发展、严谨且非常有用的领域。然而，如果你和计算机视觉研究人员交谈，你会听到一个完全不同的故事。他们会告诉你，图像识别的肮脏事实是，推动进步的是特征，而不是学习算法。计算机视觉研究人员有理由相信，稍微大一点或更干净一点的数据集或稍微改进的特征提取管道比任何学习算法对最终精度的影响要大得多。

## 学习表征

另一种预测事态发展的方法是，管道最重要的部分是代表性。直到2012年，这种代表性都是机械计算出来的。事实上，设计一套新的特征函数，改进结果，并编写方法是一种突出的论文体裁。SIFT :cite:`Lowe.2004`、SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`、HOG（定向梯度直方图）:cite:`Dalal.Triggs.2005`、[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)和类似的特征提取程序占据了主导地位。

另一组研究人员，包括Yann LeCun、Geoff Hinton、Yoshua Bengio、Andrew Ng、Shun ichi Amari和Juergen Schmidhuber，有不同的计划。他们认为特征本身应该被学习。此外，他们还认为，为了合理地复杂化，特征应该由多个共同学习的层组成，每个层都有可学习的参数。在图像的情况下，最低层可能检测边缘、颜色和纹理。事实上，亚历克斯·克里兹夫斯基、伊利亚·萨茨克弗和杰夫·辛顿提出了一种新的卷积神经网络变体，
*亚历克内特*，
在2012年ImageNet挑战赛中取得了优异的表现。AlexNet以Alex Krizhevsky的名字命名，他是ImageNet分类论文:cite:`Krizhevsky.Sutskever.Hinton.2012`的第一作者。

有趣的是，在网络的最底层，模型学习了一些类似于传统滤波器的特征抽取器。:numref:`fig_filters`是从AlexNet论文:cite:`Krizhevsky.Sutskever.Hinton.2012`复制的，描述了低级图像描述符。

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

网络中的更高层可能建立在这些表示的基础上，以表示更大的结构，如眼睛、鼻子、草叶等等。更高的层次可能代表整个物体，如人、飞机、狗或飞盘。最终，最终的隐藏状态学习图像的紧凑表示，该图像概括了其内容，从而使属于不同类别的数据易于分离。

虽然多层次卷积神经网络的最终突破出现在2012年，但一组核心研究人员致力于这一想法，多年来一直试图学习视觉数据的分层表示。2012年的最终突破可归因于两个关键因素。

### 缺失成分：数据

具有多层的深层模型需要大量的数据才能进入这样一种状态：它们显著优于基于凸优化的传统方法（如线性和核方法）。然而，考虑到计算机的存储容量有限，传感器的相对开销，以及90年代相对紧张的研究预算，大多数研究都依赖于微小的数据集。许多论文涉及UCI收集的数据集，其中许多只包含数百或（少数）数千幅在非自然环境下以低分辨率拍摄的图像。

2009年，ImageNet数据集发布，向研究人员提出了挑战，要求他们从100万个样本中学习模型，其中1000个样本来自1000个不同类别的对象。由李飞飞（Fei-Fei-Li）领导的研究人员介绍了这一数据集，利用谷歌图像搜索（Google Image Search）对每一类图像进行预筛选，并利用Amazon-Mechanical-Turk众包管道来确认每张图片是否属于相关类别。这种规模是前所未有的。这项被称为ImageNet挑战赛的相关竞赛推动了计算机视觉和机器学习研究的发展，挑战研究人员确定哪些模型在更大的范围内表现最好，而不是学者们之前所认为的。

### 缺少的成分：硬件

深度学习模型是计算周期的贪婪消费者。训练可能需要数百个时期，每次迭代都需要通过计算代价高昂的线性代数操作的许多层传递数据。这也是为什么在20世纪90年代和21世纪初，基于更有效优化凸目标的简单算法成为首选的主要原因之一。

*图形处理单元*（GPU）被证明是一个游戏规则的改变者
使深度学习成为可能。这些芯片早就被开发用来加速图形处理，从而使电脑游戏受益。特别是，它们被优化为高吞吐量$4 \times 4$矩阵向量产品，这是许多计算机图形任务所需要的。幸运的是，这个数学与计算卷积层所需的数学惊人地相似。大约在那个时候，NVIDIA和ATI已经开始为通用计算操作优化gpu，甚至把它们作为通用gpu*来销售。

为了提供一些直觉，考虑一下现代微处理器（CPU）的核心。每一个核心都是相当强大的运行在一个高时钟频率和运动大型缓存（高达数兆字节的L3）。每个内核都非常适合执行各种指令，具有分支预测器、深管道和其他使其能够运行各种程序的各种各样的功能。然而，这种明显的优势也是它的致命弱点：通用核心的制造成本非常高。它们需要大量的芯片面积、复杂的支持结构（内存接口、内核之间的缓存逻辑、高速互连等等），而且它们在任何单个任务上都相对较差。现代笔记本电脑最多有4核，即使是高端服务器也很少超过64核，仅仅是因为它的性价比不高。

相比之下，gpu由$100 \sim 1000$个小的处理元素组成（NVIDIA、ATI、ARM和其他芯片供应商之间的细节有所不同），通常被分成更大的组（NVIDIA称之为翘曲）。虽然每个内核都相对较弱，有时甚至以低于1GHz的时钟频率运行，但正是这些内核的总数使GPU比CPU快几个数量级。例如，NVIDIA最近一代的Volta为每个芯片提供了高达120 TFlop的专用指令（对于更通用的指令，最高可达24 TFlop），而cpu的浮点性能到目前为止还没有超过1 TFlop。之所以可以这样做，原因其实很简单：首先，功耗往往会随时钟频率呈二次方增长。因此，对于一个运行速度快4倍的CPU内核（一个典型的数字），您可以使用16个GPU内核，其速度是$1/4$，其性能是$16 \times 1/4 = 4$倍。此外，GPU内核要简单得多（事实上，在很长一段时间内，它们甚至不能执行通用代码），这使得它们更节能。最后，深度学习中的许多操作需要高内存带宽。再说一次，gpu在这里闪耀着至少是cpu宽度10倍的总线。

回到2012年。当亚历克斯·克里兹夫斯基（Alex Krizhevsky）和伊利亚·萨茨克弗（Ilya Sutskever）实现了一个可以在GPU硬件上运行的深度卷积神经网络时，一个重大突破出现了。他们意识到卷积神经网络中的计算瓶颈，卷积和矩阵乘法，都是可以在硬件上并行化的操作。使用两个nvidiagtx580和3GB内存，他们实现了快速卷积。代码[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)已经足够好了，几年来它一直是行业标准，并推动了深度学习热潮的头几年。

## 亚历克斯内特

采用8层卷积神经网络的AlexNet以惊人的优势赢得了2012年ImageNet大规模视觉识别挑战赛。该网络首次表明，通过学习获得的特征可以超越人工设计的特征，打破了以往计算机视觉的范式。

AlexNet和LeNet的架构非常相似，如:numref:`fig_alexnet`所示。请注意，我们提供了一个稍微精简的版本，消除了2012年需要的一些设计怪癖，使模型适合两个小型gpu。

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet和LeNet的设计理念非常相似，但也存在显著差异。首先，AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层，两个完全连接的隐藏层和一个完全连接的输出层。其次，AlexNet使用ReLU而不是sigmoid作为其激活函数。让我们深入研究下面的细节。

### 建筑

在AlexNet的第一层，卷积窗口的形状是$11\times11$。由于ImageNet中的大多数图像比MNIST图像高10倍以上，因此ImageNet数据中的对象往往占据更多的像素。因此，需要一个更大的卷积窗口来捕获目标。第二层中的卷积窗形状被缩减为$5\times5$，然后是$3\times3$。此外，在第一层、第二层和第五层之后，网络增加了最大的池层，窗口形状为$3\times3$，步长为2。此外，AlexNet的卷积通道是LeNet的10倍。

在最后一个卷积层之后有两个完全连接的层，有4096个输出。这两个巨大的完全连接层产生了将近1GB的模型参数。由于早期gpu内存有限，原有的AlexNet采用了双数据流设计，使得每个gpu只负责存储和计算模型的一半。幸运的是，现在GPU内存相对充裕，所以我们现在很少需要跨GPU分解模型（我们版本的AlexNet模型在这方面偏离了原始论文）。

### 激活函数

此外，AlexNet将sigmoid激活函数改为更简单的ReLU激活函数。一方面，ReLU激活函数的计算更简单。例如，它没有在sigmoid激活函数中找到的求幂运算。另一方面，当使用不同的参数初始化方法时，ReLU激活函数使模型训练更加容易。这是因为，当sigmoid激活函数的输出非常接近于0或1时，这些区域的梯度几乎为0，因此反向传播无法继续更新一些模型参数。相反，ReLU激活函数在正区间的梯度总是1。因此，如果模型参数没有正确初始化，sigmoid函数可能在正区间内得到几乎为0的梯度，从而使模型无法得到有效的训练。

### 容量控制和预处理

AlexNet通过dropout（:numref:`sec_dropout`）控制全连接层的模型复杂度，而LeNet只使用权重衰减。为了进一步扩充数据，AlexNet的训练循环增加了大量的图像增强，如翻转、剪切和颜色变化。这使得模型更健壮，更大的样本量有效地减少了过度拟合。我们将在:numref:`sec_image_augmentation`中更详细地讨论数据扩充。

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

我们构造了一个高度和宽度都为224的单通道数据实例来观察每一层的输出形状。它与:numref:`fig_alexnet`中的AlexNet架构相匹配。

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

尽管本文中AlexNet是在ImageNet上进行训练的，但我们在这里使用的是时尚MNIST，因为即使在现代GPU上，训练ImageNet模型以使其收敛可能需要数小时或数天的时间。将AlexNet直接应用于Fashion MNIST的一个问题是，它的图像分辨率（$28 \times 28$像素）低于ImageNet图像。为了使工作正常，我们将它们增加到$224 \times 224$（通常不是一个明智的做法，但我们在这里这样做是为了忠实于AlexNet架构）。我们使用`d2l.load_data_fashion_mnist`函数中的`resize`参数执行此调整。

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 培训

现在，我们可以开始训练亚历克内特了。与:numref:`sec_lenet`中的LeNet相比，这里的主要变化是使用更小的学习速率和更慢的训练，这是因为网络更深更广，图像分辨率更高，卷积更昂贵。

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* AlexNet的结构与LeNet相似，但使用了更多的卷积层和更大的参数空间来拟合大规模的ImageNet数据集。
* 今天，AlexNet已经被更有效的体系结构所超越，但它是当今从浅层到深层网络的关键一步。
* 尽管AlexNet的实现似乎只比LeNet多出几行，但学术界花了很多年才接受这一概念转变，并利用其出色的实验结果。这也是由于缺乏有效的计算工具。
* Dropout、ReLU和预处理是实现计算机视觉任务出色性能的其他关键步骤。

## 练习

1. 试着增加纪元的数量。与LeNet相比，结果有什么不同？为什么？
1. AlexNet对于时尚MNIST数据集来说可能太复杂了。
    1. 尝试简化模型以加快训练速度，同时确保准确性不会显著下降。
    1. 设计一个更好的模型，直接在$28 \times 28$图像上工作。
1. 修改批处理大小，并观察精度和GPU内存的变化。
1. 分析了AlexNet的计算性能。
    1. AlexNet的内存占用占主导地位的是什么？
    1. 在AlexNet中计算的主导部分是什么？
    1. 计算结果时内存带宽如何？
1. 将dropout和ReLU应用于LeNet-5。改善了吗？预处理怎么样？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
