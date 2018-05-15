# 深度卷积神经网络：AlexNet

LeNet提出后的将近二十年里，神经网络曾一度被许多其他方法超越。虽然LeNet可以在MNIST上得到好的成绩，在更大的真实世界的数据集上，神经网络表现并不佳。一方面神经网络计算慢，虽然90年代也有过一些针对神经网络的加速硬件，但并没有像目前GPU那样普及。因此训练一个多通道、多层和大量参数的在当年很难实现。另一方面，当时候研究者还没有大量深入研究权重的初始化和优化算法等话题，导致复杂的神经网络很难收敛。

跟神经网络从原始像素直接到最终标签，或者通常被称为端到端（end-to-end），不同。很长一段时间里流行的时研究者们通过勤劳、智慧和黑魔法生成了许多手工特征。通常的模式是

1. 找个数据集；
2. 用一堆已有的特征提取函数生成特征；’
3. 把这些特征表示放进一个简单的线性模型（当时认为的机器学习部分仅限这一步）。

这样的局面一直维持到2012年。如果那时候如果你跟机器学习研究者们交谈，他们会认为机器学习既重要又优美。优雅的定理证明了许多分类器的性质。机器学习领域生机勃勃，严谨，而且极其有用。
然而如果你跟一个计算机视觉研究者交谈，则是另外一幅景象。这人会告诉你图像识别里"不可告人"的现实是，计算机视觉里的机器学习流程中真正重要的是数据和特征。稍微干净点的数据集，或者略微好些的手调特征对最终准确度意味着天壤之别。反而分类器的选择对表现的区别影响不大。说到底，把特征扔进逻辑回归、支持向量机、或者其他任何分类器，表现都差不多。


## 学习特征表示

简单来说，给定一个数据集，当时流程里最重要的是特征表示这步。并且直到2012年，特征表示这步都是基于硬拼出来的直觉和机械化手工地生成的。事实上，做出一组特征，改进结果，并把方法写出来是计算机视觉论文里的一个重要流派。

另一些研究者则持异议。他们认为特征本身也应该由学习得来。他们还相信，为了表征足够复杂的输入，特征本身应该阶级式地组合起来。持这一想法的研究者们相信通过把许多神经网络层组合起来训练，他们可能可以让网络学得阶级式的数据表征。

例如在图片中，靠近数据的神经层可以表示边、色彩和纹理这一些底层的图片特征。中间的神经层可能可以基于这些表示来表征更大的结构，如眼睛、鼻子、草叶和其他特征。更靠近输出的神经层可能可以表征整个物体，如人、飞机、狗和飞盘。最终，在分类器层前的隐含层可能会表征经过汇总的内容，其中不同的类别将会是线性可分的。然而许多年来，研究者们由于种种原因并不能实现这一愿景。

### 缺失要素一：数据

尽管这群执着的研究者不断钻研，试图学习深度的视觉数据表征，很长的一段时间里这些野心都未能实现，这其中有诸多因素。第一，包含许多表征的深度模型需要大量的有标签的数据才能表现得比其他经典方法更好，虽然这些当时还不为人知。限于当时计算机有限的存储和相对囊中羞涩的90年代研究预算，大部分研究基于小的公开数据集。比如，大部分可信的研究论文是基于UCI提供的若干个数据集，其中许多只有几百至几千张图片。

这一状况在2009年李飞飞团队贡献了ImageNet数据库后得以焕然一新。它包含了1000类，每类有1000张不同的图片，这一规模是当时其他公开数据集不可相提并论的。这个数据集同时推动了计算机视觉和机器学习研究进入新的阶段，使得之前的最佳方法不再有优势。

### 缺失要素二：硬件

深度学习对计算资源要求很高。这也是为什么上世纪90年代左右基于凸优化的算法更被青睐的原因。毕竟凸优化方法里能很快收敛，并可以找到全局最小值和高效的算法。

GPU的到来改变了格局。很久以来，GPU都是为了图像处理和计算机游戏而生的，尤其是为了大吞吐量的4x4矩阵和向量乘法，用于基本的图形转换。值得庆幸的是，这其中的数学与深度网络中的卷积层非常类似。通用计算GPU（GPGPU）这个概念在2001年开始兴起，涌现诸如OpenCL和CUDA的编程框架。而且GPU也在2010年前后开始被机器学习社区开始使用。

## AlexNet

2012年AlexNet [1]，名字来源于论文一作名字Alex Krizhevsky，横空出世，它使用8层卷积神经网络以很大的优势赢得了ImageNet 2012图像识别挑战。它与LeNet的设计理念非常相似。但也有非常显著的特征。

第一，与相对较小的LeNet相比，AlexNet包含8层变换，其中有五层卷积和两层全连接隐含层，以及一个输出层。

第一层中的卷积窗口是$11\times11$，接着第二层中的是$5\times5$，之后都是$3\times3$。此外，第一，第二和第五个卷积层之后都跟了有重叠的窗口为$3\times3$，步幅为$2\times2$的最大池化层。

紧接着卷积层，AlexNet有每层大小为4096个节点的全连接层们。这两个巨大的全连接层带来将近1GB的模型大小。由于早期GPU显存的限制，最早的AlexNet包括了双数据流的设计，以让网络中一半的节点能存入一个GPU。这两个数据流，也就是说两个GPU只在一部分层进行通信，这样达到限制GPU同步时的额外开销的效果。有幸的是，GPU在过去几年得到了长足的发展，除了一些特殊的结构外，我们也就不再需要这样的特别设计了。

第二，将sigmoid激活函数改成了更加简单的relu函数$f(x)=\max(x,0)$。它计算上更简单，同时在不同的参数初始化方法下收敛更加稳定。

第三，通过丢弃法（参见[“丢弃法”](../chapter_supervised-learning/dropout-scratch.md)这一小节）来控制全连接层的模型复杂度。

第四，引入了大量的图片增广，例如翻转、裁剪和颜色变化，来进一步扩大数据集来减小过拟合。我们将在后面的[“图片增广”](chapter_computer-vision/image-augmentation.md)的小节来详细讨论。

下面我们实现（稍微简化过的）Alexnet：

```{.python .input}
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import nn

net = nn.Sequential()
net.add(
    # 因为输入图片尺寸（224 x 224）比LeNet（28 x 28）大很多，
    # 因此使用较大的卷积窗口来捕获物体。同时使用步幅4来较大减小输出高宽。
    # 这里使用的输入通道数比 LeNet 也要大很多。
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 减小卷积窗口，使用填充为2来使得输入输出高宽一致。且增大输出通道数。
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 连续三个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，
    # 进一步增大了输出通道数。前两个卷积层后不使用池化层来减小输入的高宽。
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 使用比 LeNet 输出大数倍了全连接层。其使用丢弃层来控制复杂度。
    nn.Dense(4096, activation="relu"),
    nn.Dropout(.5),
    nn.Dense(4096, activation="relu"),
    nn.Dropout(.5),
    # 输出层。我们这里使用 FashionMNIST，所以用 10，而不是论文中的 1000。
    nn.Dense(10)
)
```

我们构造一个高宽均为224的单通道数据点来观察每一层的输出大小。

```{.python .input}
X = nd.random.uniform(shape=(1,1,224,224))

net.initialize()

for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## 读取数据

虽然论文中Alexnet使用Imagenet数据，它因为Imagenet数据训练时间较长，我们仍用前面的FashionMNIST来演示。读取数据的时候我们额外做了一步将图片高宽扩大到原版Alexnet使用的224。

```{.python .input}
train_data, test_data = gb.load_data_fashion_mnist(batch_size=128, resize=224)
```

## 训练

这时候我们可以开始训练。相对于上节的LeNet，这里的主要改动是使用了更小的学习率。

```{.python .input}
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=3)
```

## 小结

AlexNet跟LeNet类似，但使用了更多的卷积层和更大的参数空间来拟合大规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线。虽然看上去AlexNet的实现比LeNet也就就多了几行而已。但这个观念上的转变和真正跑出好实验结果，学术界整整花了20年。

## 练习

- 多迭代几轮看看？跟LeNet比有什么区别？为什么？
- AlexNet对于FashionMNIST过于复杂，试着简化模型来使得训练更快，同时精度不明显下降。
- 修改批量大小，观察性能和GPU内存的变化。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1258)

![](../img/qr_alexnet-gluon.svg)

## 参考文献 

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
