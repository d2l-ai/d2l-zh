# 稠密连接网络（DenseNet）

ResNet中的跨层连接设计引申出了数个后续工作。这一节我们介绍其中的一个：稠密连接网络（DenseNet） [1]。 它与ResNet的主要区别如图5.10演示。

![ResNet（左）对比DenseNet（右）。](../img/densenet.svg)

主要区别在于，DenseNet里层B的输出不是像ResNet那样和层A的输出相加，而是在通道维上合并，这样层A的输出可以不受影响的进入上面的神经层。这个设计里，层A直接跟上面的所有层连接在了一起，这也是它被称为“稠密连接“的原因。

DenseNet的主要构建模块是稠密块和过渡块，前者定义了输入和输出是如何合并的，后者则用来控制通道数不要过大。

## 稠密块

DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构（参见上一节习题），我们首先在`conv_block`函数里实现这个结构。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

稠密块由多个`conv_block`组成，每块使用相同的输出通道数。但在正向传播时，我们将每块的输出在通道维上同其输出合并进入下一个块。

```{.python .input  n=2}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 在通道维上将输入和输出合并。
            X = nd.concat(X, Y, dim=1)
        return Y
```

下面例子中我们定义一个有两个输出通道数为10的卷积块，使用通道数为3的输入时，我们会得到通道数为$3+2\times 10=23$的输出。卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）。

```{.python .input  n=8}
blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

## 过渡块

由于每个稠密块都会带来通道数的增加。使用过多则会导致过于复杂的模型。过渡块（transition block）则用来控制模型复杂度。它通过$1\times1$卷积层来减小通道数，同时使用步幅为2的平均池化层来将高宽减半来进一步降低复杂度。

```{.python .input  n=3}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

我们对前面的稠密块的输出使用通道数为10的过渡块。

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

## DenseNet模型

DenseNet首先使用跟ResNet一样的单卷积层和最大池化层：

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

类似于ResNet接下来使用的四个基于残差块，DenseNet使用的是四个稠密块。同ResNet一样我们可以设置每个稠密块使用多少个卷积层，这里我们设成4，跟上一节的ResNet 18保持一致。稠密块里的卷积层通道数（既增长率）设成32，所以每个稠密块将增加128通道。

ResNet里通过步幅为2的残差块来在每个模块之间减小高宽，这里我们则是使用过渡块来减半高宽，并且减半输入通道数。

```{.python .input  n=5}
# 当前的数据通道数。
num_channels = 64
growth_rate = 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # 上一个稠密的输出通道数。
    num_channels += num_convs * growth_rate
    # 在稠密块之间加入通道数减半的过渡块。
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add(transition_block(num_channels // 2))
```

最后同ResNet一样我们接上全局池化层和全连接层来输出。

```{.python .input}
net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

## 获取数据并训练

因为这里我们使用了比较深的网络，所以我们进一步把输入减少到$32\times 32$来训练。

```{.python .input}
lr = 0.1
num_epochs = 5
batch_size = 256
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,
                                                   resize=96)
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
```

## 小结

* 不同于ResNet中将输入加在输出上完成跨层连接，DenseNet在通道维上合并输入和输出来使得底部神经层能跟其上面所有层连接起来。

## 练习

- DenseNet论文中提交的一个优点是其模型参数比ResNet更小，这是为什么？
- DenseNet被人诟病的一个问题是内存消耗过多。真的会这样吗？可以把输入换成$224\times 224$，来看看实际（GPU）内存消耗。
- 实现DenseNet论文中的表1提出的各个DenseNet版本 [1]。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1664)

![](../img/qr_densenet-gluon.svg)

## 参考文献

[1] Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (Vol. 1, No. 2).
