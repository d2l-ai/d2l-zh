# 深度卷积神经网络（AlexNet）

在LeNet提出后的将近二十年里，神经网络一度被其他方法（例如支持向量机）超越。虽然LeNet可以在MNIST上取得到好的成绩，但是在更大的真实数据集上神经网络表现并不佳。一方面神经网络计算复杂，虽然90年代也有过一些针对神经网络的加速硬件，但并没有大量普及。因此训练一个多通道、多层和有大量参数的卷积神经网络在当年很难完成。另一方面，当年研究者还没有大量深入研究参数初始化和非凸优化算法等诸多领域，导致复杂的神经网络收敛通常很困难。

即使神经网络可以从原始像素直接预测标签，这种称为端到端（end-to-end）的途径节省了很多中间步奏。但在很长一段时间里更流行的是研究者们通过勤劳智慧和黑魔法生成的手工特征。通常的模式是

1. 找个数据集；
2. 用一堆已有的特征提取函数生成特征；
3. 把这些特征表示放进一个简单的线性模型。

当时认为的机器学习部分仅限最后这一步。如果那时候你跟机器学习研究者们交谈，他们会认为机器学习既重要又优美。优雅的定理证明了许多分类器的性质。机器学习领域生机勃勃、严谨、而且极其有用。然而如果你跟一个计算机视觉研究者交谈，则是另外一幅景象。他们会告诉你图像识别里“不可告人”的现实是，计算机视觉流程中真正重要的是数据和特征。是否有稍微干净点的数据集，或者略微好些的手调特征的差距对最终准确度意味着天壤之别。反观分类器的选择对最终表现的区别影响不大。说到底，把特征扔进逻辑回归、支持向量机、或者其他任何分类器，表现都差不多。

## 学习特征表示

在相当长的时间里特征表示这步都是基于硬拼出来的直觉和机械化手工地生成的。事实上，做出一组特征、改进结果、并把方法写出来是计算机视觉论文里的一个重要流派。

另一些研究者则持异议。他们认为特征本身也应该由学习得来。他们还相信，为了表征足够复杂的输入，特征本身应该阶级式地组合起来。持这一想法的研究者们相信通过把许多神经网络层组合起来训练，他们可能可以让网络学得阶级式的数据表征。

例如在图片中，靠近数据的神经层可以表示边、色彩和纹理这一些底层的图片特征。中间的神经层可能可以基于这些表示来表征更大的结构，如眼睛、鼻子、草叶和其他特征。更靠近输出的神经层也许可以表征整个物体，如人、飞机、狗和飞盘。最终，在分类器层前的隐含层可能可以表征经过汇总的内容，其中不同的类别将会是线性可分的。尽管这群执着的研究者不断钻研，试图学习深度的视觉数据表征，很长的一段时间里这些野心都未能实现，这其中有诸多因素值得我们一一分析。

### 缺失要素一：数据

包含许多特征的深度模型需要大量的有标签的数据才能表现得比其他经典方法更好。虽然通过大量累加数据很快就达到了线性模型的上限并在工业界应用，例如广告点击预测里已经被广泛认同，但学术界直到很久以后才普遍认识到这个问题。限于当时计算机有限的存储和相对囊中羞涩的90年代研究预算，大部分研究基于小的公开数据集。比如，大部分可信的研究论文是基于UCI提供的若干个数据集，其中许多数据集只有几百至几千张图片。

这一状况在2010前后兴起的大数据浪潮里得到改善。尤其是2009年李飞飞团队贡献了ImageNet数据集。它包含了1000大类物体，每类有多达数千张不同的图片，这一规模是当时其他公开数据集无法相提并论的。这个数据集同时推动了计算机视觉和机器学习研究进入新的阶段，使得之前的最佳方法不再有优势。

### 缺失要素二：硬件

深度学习对计算资源要求很高。这也是为什么上世纪90年代左右基于凸优化的算法更被青睐的原因。毕竟凸优化方法是能很快收敛，并可以找到全局最小值的高效的算法。

GPU的到来改变了格局。很久以来，GPU都是为了图像处理和计算机游戏而设计，尤其是针对大吞吐量的矩阵和向量乘法来用于基本的图形转换。值得庆幸的是，这其中的数学表达与深度网络中的卷积层非常类似。通用计算GPU（GPGPU）这个概念在2001年开始兴起，涌现出诸如OpenCL和CUDA之类的编程框架。使得GPU也在2010年前后开始被机器学习社区使用。

## AlexNet

2012年AlexNet横空出世。它的名字来源于论文一作姓名Alex Krizhevsky [1]。它使用8层卷积神经网络以很大的优势赢得了ImageNet 2012图像识别挑战赛。它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。

AlextNet与LeNet的设计理念非常相似。但也有非常显著的区别。

1. 与相对较小的LeNet相比，AlexNet包含8层变换，其中有五层卷积和两层全连接隐含层，以及一个输出层。

   第一层中的卷积窗口是$11\times11$。因为ImageNet图片高宽均比MNIST大十倍以上，对应图片的物体占用更多的像素，所以需要使用更大的窗口来捕获物体。第二层减少到$5\times5$，之后全采用$3\times3$。此外，第一，第二和第五个卷积层之后都使用了窗口为$3\times3$步幅为2的最大池化层。另外，AlexNet使用的卷积通道数也数十倍大于LeNet。

   紧接着卷积层的是两个输出大小为4096的全连接层们。这两个巨大的全连接层带来将近1GB的模型大小。由于早期GPU显存的限制，最早的AlexNet使用双数据流的设计使得一个GPU只需要处理一半模型。幸运的是GPU内存在过去几年得到了长足的发展，除了一些特殊的结构外，我们也就不再需要这样的特别设计了。

2. 将sigmoid激活函数改成了更加简单的relu函数$f(x)=\max(x,0)$。它计算上更简单，同时在不同的参数初始化方法下收敛更加稳定。

3. 通过丢弃法（参见[“丢弃法”](../chapter_deep-learning-basics/dropout.md)这一小节）来控制全连接层的模型复杂度。

4. 引入了大量的图片增广，例如翻转、裁剪和颜色变化，进一步扩大数据集来减小过拟合。我们将在后面的[“图片增广”](chapter_computer-vision/image-augmentation.md)的小节来详细讨论。

下面我们实现（稍微简化过的）AlexNet：

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import os
import sys

net = nn.Sequential()
net.add(
    # 使用较大的 11 x 11 窗口来捕获物体。同时使用步幅 4 来较大减小输出高宽。
    # 这里使用的输入通道数比 LeNet 也要大很多。
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 减小卷积窗口，使用填充为2来使得输入输出高宽一致。且增大输出通道数。
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 连续三个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
    # 前两个卷积层后不使用池化层来减小输入的高宽。
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 使用比 LeNet 输出大数倍了全连接层。其使用丢弃层来控制复杂度。
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    # 输出层。我们这里使用 Fashion-MNIST，所以用 10，而不是论文中的 1000。
    nn.Dense(10)
)
```

我们构造一个高和宽均为224像素的单通道数据点来观察每一层的输出大小。

```{.python .input  n=2}
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## 读取数据

虽然论文中AlexNet使用ImageNet数据，但因为ImageNet数据训练时间较长，我们仍用前面的Fashion-MNIST来演示。读取数据的时候我们额外做了一步将图片高宽扩大到原版AlexNet使用的224，这个可以通过`Resize`来实现。即我们在`ToTenor`前使用`Resize`，然后使用`Compose`来将这两个变化合并成一个来方便调用。数据读取的其他部分跟前面一致。

```{.python .input  n=3}
def load_data_fashion_mnist(batch_size, resize=None,
                            root=os.path.join('~', '.mxnet', 'datasets',
                                              'fashion-mnist')):
    root = os.path.expanduser(root)  # 展开用户路径 '~'。
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size,
                                                resize=224)
```

我们将`load_data_fashion_mnist`函数定义在`gluonbook`包中供后面章节调用。

## 训练

这时候我们可以开始训练。相对于上节的LeNet，这里的主要改动是使用了更小的学习率。

```{.python .input  n=5}
lr = 0.01
num_epochs = 5
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
```

## 小结

* AlexNet跟LeNet结构类似，但使用了更多的卷积层和更大的参数空间来拟合大规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线。虽然看上去AlexNet的实现比LeNet也就多了几行而已。但这个观念上的转变和真正优秀实验结果的产生，学术界整整花了20年。

## 练习

- 多迭代几轮训练看看？跟LeNet比有什么区别？为什么？
- AlexNet对于Fashion-MNIST过于复杂，试着简化模型来使得训练更快，同时保证精度不明显下降。
- 修改批量大小，观察性能和GPU内存的变化。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1258)

![](../img/qr_alexnet-gluon.svg)

## 参考文献

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
