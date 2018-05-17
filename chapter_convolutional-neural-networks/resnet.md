# 残差网络：ResNet

上一小节介绍的批量归一化层对网络中间层的输出做归一化，使得训练时数值更加稳定和收敛更容易。但对于深层网络来说，还有一个问题困扰训练。在进行梯度反传计算时，我们从误差函数（顶部）开始，朝着输入数据方向（底部）逐层计算梯度。当我们将层串联在一起的时候，根据链式法则我们将每层的梯度乘在一起，这样经常导致梯度大小指数衰减。从而在靠近底部的层只得到很小的梯度，随之权重的更新量也变小，使得他们的收敛缓慢。

ResNet [1] 成功增加跨层的数据线路来允许梯度可以快速的到达底部层，来有效避免这一情况。这一节我们将介绍ResNet的工作原理。


## 残差块

ResNet的基础块叫做残差块。如下图所示，它将层A的输出在输入给层B的同时跨过B，并和B的输出相加作为下面层的输入。它可以看成是两个网络相加，一个网络只有层A，一个则有层A和B。这里层A在两个网络之间共享参数。在求梯度的时候，来自层B上层的梯度既可以通过层B也可以直接到达层A，从而使得层A可以更容易获取足够大的梯度来进行模型更新。

![残差快（左）和它的分解（右）](../img/resnet.svg)


ResNet沿用了VGG全$3\times 3$卷积层设计。残差块输入首先被连续作用两次同样输入通道的$3\times 3$卷积层，每个卷积层后跟一个批量归一化层，然后是ReLU激活层。然后我们将输入跳过这两个卷积层后直接加在最后的ReLU激活层前。这样我们要求这两个卷积层的输出都保持跟输入形状一样来保证可以之后与输入相加。

如果我们想改变输入大小，意味着卷积层使用跟输入不一样的通道大小，同时我们让第一个卷积层使用步幅2来减半输入高宽。为了保证的相加操作还能进行，我们引入一个额外的$1\times 1$卷积层来将输入变换成需要的形状后再相加。

残差块的实现见下。它可以设定输出通道数，和是否保持输出形状和输入一致。

```{.python .input  n=1}
import sys
sys.path.append('..')
import gluonbook as gb
import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import nn

class Residual(nn.Block):
    def __init__(self, num_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        if same_shape:
            self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
            self.conv3 = None
        else:
            self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, 
                                   strides=2)
            self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)            
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=2)

        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
```

查看输入输出形状一致的情况：

```{.python .input  n=2}
blk = Residual(3)
blk.initialize()

x = nd.random.uniform(shape=(4, 3, 6, 6))
blk(x).shape
```

否则我们改变输出形状的同时减半输出高宽：

```{.python .input  n=3}
blk2 = Residual(6, same_shape=False)
blk2.initialize()
blk2(x).shape
```

## ResNet模型

ResNet主体是由多个残差块构成。它的构建模式是首先一个减半高宽的残差块，然后接数个保持输入形状的残差块，然后再接一个通道翻倍但高宽减半的残差块。如此重复4次。我们先定义一个这样的模式，它在一个减半高宽的残差块后加数个保持形状的残差块：

```{.python .input  n=4}
def resnet_block(num_channels, num_residuals):
    blk = nn.Sequential()
    for i in range(num_residuals):
        blk.add(Residual(num_channels, same_shape=(i is not 0)))
    return blk
```

下面我们构造一个ResNet。前面两层跟前面介绍的GoogLeNet一样，在输出通道为64、步幅为2的$7\times 7$卷积层后接步幅为2的$3\times 3$的最大池化层。不同于GoogLeNet在后面接4个有Inception块组成的模块，这里我们使用输出通道数从64开始，每次翻倍的由2个残差块组成的模块。最后跟GoogLeNet一样使用全局平均池化层和全连接层来输出。

因为这里每个模块里有4个卷积层（$1\times 1$卷积层不算），加上最开始的卷积层和最后的全连接层，一共有18层。这个模型也通常被称之为ResNet 18。通过配置不同的通道数和模块里的残差块数我们可以得到不同的ResNet模型。

```{.python .input  n=5}
net = nn.Sequential()
net.add(
    nn.Conv2D(64, kernel_size=7, strides=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    resnet_block(64, 2),
    resnet_block(128, 2),
    resnet_block(256, 2),
    resnet_block(512, 2),
    nn.GlobalAvgPool2D(),
    nn.Dense(10),
)
```

主要到每个残差块里我们都将输入直接或者通过简单的$1\times 1$卷积层加在输出上，所以即使层数很多，损失函数的梯度也能很快的传递到靠近输入的层那里。这使得即使是很深的ResNet（例如ResNet 152）在收敛速度上也同浅的ResNet（例如这里实现的ResNet 18）类似。同时虽然它的主体架构上跟GoogLeNet类似，但ResNet结构更加简单，修改也更加方便。这些因素都导致了ResNet迅速的被广泛使用。

最后我们考察输入在ResNet不同模块之间的变化。

```{.python .input}
X = nd.random.uniform(shape=(1,1,96,96))

net.initialize()

for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## 获取数据并训练

使用跟GoogLeNet一样的超参数。

```{.python .input  n=7}
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_data, test_data = gb.load_data_fashion_mnist(batch_size=256, resize=96)
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)
```

## 小结

残差块通过将输入加在卷积层作用过的输出上来引入跨层通道。这使得即使非常深的网络也能很容易训练。

## 练习

- 参考 [1] 的表1来实现不同的ResNet版本。
- 在对于比较深的网络，[1]介绍了一个“bottleneck”架构来降低模型复杂度。尝试实现它。
- 在ResNet的后续版本里 [2]，作者将残差块里的“卷积、批量归一化和激活”结构改成了“批量归一化、激活和卷积”（参考[2]中的图1），实现这个改进。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1663)

![](../img/qr_resnet-gluon.svg)

## 参考文献

[1] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.

[2] He, Kaiming, et al. "Identity mappings in deep residual networks." ECCV, 2016.
