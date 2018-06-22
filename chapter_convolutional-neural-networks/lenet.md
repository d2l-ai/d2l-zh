# 卷积神经网络

在[“多层感知机——从零开始”](../chapter_supervised-learning/mlp-scratch.md)一节里我们构造了一个两层感知机模型来对FashionMNIST里图片进行分类。每张图片高宽均是28，我们将其展开成长为784的向量输入到模型里。这样的做法虽然简单，但也有局限性：

1. 垂直方向接近的像素在这个向量的图片表示里可能相距很远，它们组成的模式难被模型识别。
2. 对于大尺寸的输入图片，我们会得到过大的模型。假设输入是高宽为1000的彩色照片，即使隐藏层输出仍是256，这一层的模型形状是$3,000,000\times 256$，其占用将近3GB的内存，这带来过复杂的模型和过高的存储开销。

卷积层尝试解决这两个问题：它保留输入形状，使得可以有效的发掘水平和垂直两个方向上的数据关联。它通过滑动窗口将卷积核重复作用在输入上，而得到更紧凑的模型参数表示。

卷积神经网络就是主要由卷积层组成的网络，本小节里我们将介绍一个早期用来识别手写数字图片的卷积神经网络：LeNet [1]，其名字来源于论文一作Yann LeCun。LeNet证明了通过梯度下降训练卷积神经网络可以达到手写数字识别的最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。

## LeNet模型

LeNet分为卷积层块和全连接层块两个部分。卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图片里的空间模式，例如线条和物体局部，之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基础块构成，即两个卷积层和两个最大池化层。每个卷积层都使用$5\times 5$的窗口，且在输出上作用sigmoid激活函数$f(x)=\frac{1}{1+e^{-x}}$来将输出非线性变换到$(0,1)$区间。第一个卷积层输出通道为6，第二个则增加到16，这是因为其输入高宽比之前卷积层要小，所以增加输出通道来保持相似的模型复杂度。两个最大池化层的窗口均为$2\times 2$，且步幅为2。这意味着每个作用的池化窗口都是不重叠的。

卷积层块对每个样本输出被拉升成向量输入到全连接层块中。全连接层块由两个输出大小分别为120和84的全连接层，然后接上输出大小为10（因为标号类数为10）的输出层构成。下面我们用过Sequential类来实现LeNet。

```{.python .input}
import sys
sys.path.append('..')
import gluonbook as gb
import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import nn

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense 会默认将（批量大小，通道，高，宽）形状的输入转换成
    #（批量大小，通道 x 高 x 宽）形状的输入。
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)
```

接下来我们构造一个高宽均为28的单通道数据点，并逐层进行前向计算来查看每个层的输出大小。

```{.python .input}
X = nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

可以看到在卷积层块中图片的高宽在逐层减小，卷积层由于没有使用填充从而将高宽减4，池化层则减半，但通道数则从1增加到16。全连接层则进一步减小输出大小直到变成10。

## 获取数据和训练


我们仍然使用FashionMNIST作为训练数据。

```{.python .input}
train_data, test_data = gb.load_data_fashion_mnist(batch_size=256)
```

因为卷积神经网络计算比多层感知机要复杂，因此我们使用GPU来加速计算。我们尝试在GPU 0上创建NDArray，如果成功则使用GPU 0，否则则使用CPU。（下面代码将保存在GluonBook的`try_gpu`函数里来方便重复使用）。

```{.python .input}
try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx
```

我们重新将模型参数初始化到`ctx`，且使用Xavier [2]（使用论文一作姓氏命名）来进行随机初始化。Xavier根据每个层的输入输出大小来选择随机数的上下区间，来使得每一层输出有相似的方差，从而使得训练时数值更加稳定。损失函数和训练算法则使用跟之前一样的交叉熵损失函数和小批量随机梯度下降。

```{.python .input}
lr = 1
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)
```

## 小结

* LeNet交替使用卷积层和最大池化层后接全连接层来进行图片分类。

## 练习

- LeNet的设计是针对MNIST，但在我们这里使用的FashionMNIST复杂度更高。尝试基于LeNet构造更复杂的网络来改善精度。例如可以考虑调整卷积窗口大小、输出层大小、激活函数和全连接层输出大小。在优化方面，可以尝试使用不同学习率、初始化方法和多使用一些迭代周期。
- 找出Xavier的具体初始化方法。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/737)

![](../img/qr_cnn-gluon.svg)

## 参考文献

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

[2] Glorot, X., & Bengio, Y. (2010, March). Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256).
