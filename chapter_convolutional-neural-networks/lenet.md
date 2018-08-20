# 卷积神经网络（LeNet）

在[“多层感知机的从零开始实现”](../chapter_deep-learning-basics/mlp-scratch.md)一节里我们构造了一个两层感知机模型来对Fashion-MNIST里图片进行分类。每张图片高和宽均是28像素。我们将其展开成长为784的向量输入到模型里。这样的做法虽然简单，但也有局限性：

1. 垂直方向接近的像素在这个向量的图片表示里可能相距很远，它们组成的模式难被模型识别。
2. 对于大尺寸的输入图片，我们会得到过大的模型。假设输入是高和宽均为1000像素的彩色照片，即使隐藏层输出仍是256，这一层的模型形状是$3,000,000\times 256$，其占用将近3GB的内存，这带来过复杂的模型和过高的存储开销。

卷积层尝试解决这两个问题：它保留输入形状，使得可以有效的发掘水平和垂直两个方向上的数据关联。它通过滑动窗口将卷积核重复作用在输入上，从而得到更紧凑的模型参数表示。

卷积神经网络就是主要由卷积层组成的网络，本小节里我们将介绍一个早期用来识别手写数字图片的卷积神经网络：LeNet [1]。这个名字来源于论文第一作者Yann LeCun。LeNet证明了通过梯度下降训练卷积神经网络可以达到手写数字识别的最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。

## LeNet模型

LeNet分为卷积层块和全连接层块两个部分。卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图片里的空间模式，例如线条和物体局部，之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基础块重复堆叠构成，即拥有两个卷积层和两个最大池化层。每个卷积层都使用$5\times 5$的窗口，且在输出上使用sigmoid激活函数$f(x)=\frac{1}{1+e^{-x}}$来将输出非线性变换到$(0,1)$区间。第一个卷积层输出通道为6，第二个则增加到16，这是因为其输入高宽比之前卷积层要小，所以增加输出通道来保持相似的模型复杂度。两个最大池化层的窗口均为$2\times 2$，且步幅为2。这意味着每个池化窗口的作用范围都是不重叠的。

卷积层块把每个样本输出拉升成向量输入到全连接层块中。全连接层块由两个输出大小分别为120和84的全连接层，然后接上输出大小为10（因为数字的类别一共为10）的输出层构成。下面我们通过Sequential类来实现LeNet。

```{.python .input}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import loss as gloss, nn
from time import time

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense 会默认将（批量大小，通道，高，宽）形状的输入转换成
    #（批量大小，通道 * 高 * 宽）形状的输入。
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)
```

接下来我们构造一个高宽均为28的单通道数据点，并逐层进行前向计算来查看每个层的输出大小。

```{.python .input}
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

可以看到在卷积层块中图片的高宽在逐层减小，卷积层由于没有使用填充从而将高宽减少4，池化层则减半高宽，但通道数则从1增加到16。全连接层则进一步减小输出大小直到变成10。

## 获取数据和训练


我们仍然使用Fashion-MNIST作为训练数据。

```{.python .input}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size)
```

因为卷积神经网络计算比多层感知机要复杂，因此我们使用GPU来加速计算。我们尝试在GPU 0上创建NDArray，如果成功则使用GPU 0，否则则使用CPU。

```{.python .input}
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
ctx
```

相应地，我们对[“Softmax回归的从零开始实现”](../chapter_deep-learning-basics/softmax-regression-scratch.md)一节中描述的`evaluate_accuracy`函数略作修改。由于数据刚开始存在CPU的内存上，当`ctx`为GPU时，我们通过[“GPU计算”](../chapter_deep-learning-computation/use-gpu.md)一节中介绍的`as_in_context`函数将数据复制到GPU上（例如GPU 0）。

```{.python .input}
def evaluate_accuracy(data_iter, net, ctx):
    acc = nd.array([0], ctx=ctx)
    for X, y in data_iter:
        # 如果 ctx 是 GPU，将数据复制到 GPU 上。
        X = X.as_in_context(ctx)
        y = y.as_in_context(ctx)
        acc += gb.accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)
```

我们同样对[“Softmax回归的从零开始实现”](../chapter_deep-learning-basics/softmax-regression-scratch.md)一节中定义的`train_ch3`函数略作修改，确保计算使用的数据和模型同在CPU或GPU的内存上。

```{.python .input}
def train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        start = time()
        for X, y in train_iter:
            # 如果 ctx 是 GPU，将数据复制到 GPU 上。
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc, time() - start))
```

我们重新将模型参数初始化到`ctx`，并使用[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节里介绍过Xavier随机初始化。损失函数和训练算法则使用跟之前一样的交叉熵损失函数和小批量随机梯度下降。

```{.python .input}
lr = 0.8
num_epochs = 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
          num_epochs)
```

本节中的`try_gpu`、`evaluate_accuracy`和`train_ch5`函数被定义在`gluonbook`包中供后面章节调用。其中的`evaluate_accuracy`函数会被进一步改进：它的完整实现将在[“图片增广”](../chapter_computer-vision/image-augmentation.md)一节中描述。


## 小结

* LeNet交替使用卷积层和最大池化层后接全连接层来进行图片分类。

## 练习

- LeNet的设计是针对MNIST，但在我们这里使用的Fashion-MNIST复杂度更高。尝试基于LeNet构造更复杂的网络来改善精度。例如可以考虑调整卷积窗口大小、输出层大小、激活函数和全连接层输出大小。在优化方面，可以尝试使用不同学习率、初始化方法和多使用一些迭代周期。
- 找出Xavier的具体初始化方法。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/737)

![](../img/qr_cnn-gluon.svg)

## 参考文献

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
