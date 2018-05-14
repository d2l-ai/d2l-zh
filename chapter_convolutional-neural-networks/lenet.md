# 卷积神经网络

在[“多层感知机——从零开始”](../chapter_supervised-learning/mlp-scratch.md)这一节里我们构造了一个两层感知机模型来对FashionMNIST这个图片数据集进行分类。这个数据集的图片形状是$28\times 28$。我们将二维图片展开成一个长为784的向量输入到模型里。这样的做法虽然简单，但有两个重要的局限性。

1. 垂直方向接近的像素在这个向量的图片表示里可能相距很远，从而很难被模型察觉。
2. 对于大图片输入模型可能会过大。例如对于输入是$1000 \times 1000\times3$的彩色照片，即使隐藏层输出仍为256，这一层的模型形状是$3,000,000\times 256$，其占用将近3GB的内存，这带来过于复杂的模型和过高的存储开销。

卷积层尝试解决这两个问题：它保留输入形状，使得有效的发掘水平和垂直两个方向上的数据关联。且通过滑动窗口将核参数重复作用在输入上，而得到更紧凑的参数表示。卷积神经网络就是主要由卷积层组成的网络，这一小节我们介绍一个著名的早期用来识别手写数字图片的卷积神经网络：LeNet [1]。它证明了通过梯度下降训练卷积神经网络可以达到手写数字识别的最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。

## 定义模型

LeNet分为卷积层块和全连接层块两个部分。卷积层块里的基础单位是卷积层后接最大池化层：卷积层用来识别图片里的空间模式，例如笔画，之后的最大池化层则用来减低卷积层对位置的敏感性。卷积层块由两个这样的基础块构成。每一块里的卷积层都使用$5\times 5$窗口，且在输出上作用sigmoid激活函数$f(x)=\frac{1}{1+e^{-x}}$来将输入非线性变换到$(0,1)$区间。不同点在于第一个卷积层输出通道为5，第二个则增加到16。两个最大池化层的窗口大小均为$2\times 2$，且步幅为2。这意味着每个池化窗口都是不重叠的。卷积层块对每个样本输出被拉升成向量输入到全连接层块中。全连接层块由两个输出大小分别为120和84的全连接层，然后接上输出大小为10，对应10类数字，的输出层构成。

下面我们用过Sequential类来实现LeNet。

```{.python .input}
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

因为卷积神经网络计算比多层感知机要复杂，因此我们使用GPU来加速计算。我们尝试在GPU 0上创建NDArray，如果成功则使用GPU 0，否则则使用CPU。（我们将下面这段代码保存在GluonBook的`try_gpu`函数里方便下次重复使用）。

```{.python .input}
try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx
```

我们重新将模型参数初始化到`ctx`，且使用Xavier [2]来进行随机初始化。Xavier根据每个层的输入输出大小来选择随机数区间，从而使得输入输出的方差相似来使得网络优化更加稳定。损失函数和训练算法则使用跟之前一样的交叉熵损失函数和小批量随机梯度下降。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1})

gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)
```

## 小节

LeNet使用交替使用卷积层和最大池化层，后接全连接层来进行图片分类。

## 练习

LeNet是针对MNIST提出，但在我们这里的FashionMNIST上效果不是特别好。一个原因可能是FashionMNIST数据集更加复杂，可能需要更复杂的网络。尝试修改它来提升精度。可以考虑调整卷积窗口大小，输出层大小，激活函数，全连接层输出大小。当然在优化方面，可以尝试使用不同学习率，初始化方法和多使用一些迭代周期。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/737)

![](../img/qr_cnn-gluon.svg)

## 参考文献

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

[2] Glorot, X., & Bengio, Y. (2010, March). Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256).
