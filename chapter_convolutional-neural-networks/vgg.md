# 使用重复元素的网络：VGG

AlexNet在LeNet的基础上增加了三个卷积层。但作者对它们的卷积窗口、通道数和构造顺序均做了大量的调整。虽然AlexNet指明了深度卷积神经网络可以取得很高的结果，但并没有提供简单的规则来告诉后来的研究者如何设计新的网络。我们将在接下来数个小节里介绍几种不同的网络设计思路。

本节我们介绍VGG [1]，它名字来源于论文作者所在实验室Visual Geometry Group。VGG提出了可以通过重复使用简单的基础块来构建深层模型。

## VGG块

VGG模型的基础组成单位是连续数个相同的使用填充1的$3\times 3$卷积层后接上一个步幅为2的$2\times 2$最大池化层。卷积层保持输入高宽，而池化层则对其减半。我们使用`vgg_block`函数来实现这个基础块，它可以指定使用多少卷积层和其输出通道数。

```{.python .input  n=1}
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import nn

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(
            num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## VGG模型

VGG网络同AlexNet和LeNet一样由卷积层模块后接全连接层模块构成。卷积层模块串联数个`vgg_block`，其超参数由`conv_arch`定义，其指定每个块里卷积层个数和输出通道。全连接模块则跟AlexNet一样。

现在我们构造一个VGG网络。它有5个卷积块，前三块使用单卷积层，而后两块使用双卷基层。第一块的输出通道是64，之后每次对输出通道数翻倍。因为这个网络使用了8个卷基层和3个全连接层，所以经常被称之为VGG 11。

```{.python .input  n=3}
conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))
```

下面我们根据架构实现VGG 11。

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(nn.Dense(4096, activation="relu"), nn.Dropout(.5),
            nn.Dense(4096, activation="relu"), nn.Dropout(.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

然后我们打印每个卷积块的输出变化。

```{.python .input}
net.initialize()
X = nd.random.uniform(shape=(1,1,224,224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

可以看到每次我们将长宽减半，最后高宽变成7后进入全连接层。同时输出通道数每次都翻倍。因为每个卷积层的窗口大小一样，所以每层的模型参数大小和计算复杂度跟 高$\times$宽$\times$输入通道数$\times$输出通道数 成正比。VGG这种高宽减半和通道翻倍的设计使得每个卷基层都有相同的模型参数大小和计算复杂度。

## 模型训练

因为VGG 11计算上比AlexNet更加复杂，我们构造一个通道数更小，或者说更窄的，的网络来训练FashionMNIST。

```{.python .input}
ratio = 4
small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

模型训练跟上一节的AlexNet类似，除了使用使用了稍大些的学习率。

```{.python .input}
lr = 0.05
ctx = gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_data, test_data = gb.load_data_fashion_mnist(batch_size=128, resize=224)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=3)
```

## 小结

VGG通过5个可以重复使用的卷积块来构造网络。根据每块里卷积层个数和输出通道不同可以定义出不同的VGG模型。

## 练习

- VGG的计算比AlexNet慢很多，也需要很多的GPU内存。分析下原因。
- 尝试将FashionMNIST的高宽由224改成96，实验其带来的影响。
- 参考[1]里的表1来构造VGG其他常用模型，例如VGG16和VGG19。
 
## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1277)

![](../img/qr_vgg-gluon.svg)

## 参考文献 

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
