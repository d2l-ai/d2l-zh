# 稠密连接的网络：DenseNet

ResNet的跨层连接思想影响了接下来的众多工作。这里我们介绍其中的一个：[DenseNet](https://arxiv.org/pdf/1608.06993.pdf)。下图展示了这两个的主要区别：

![](../img/densenet.svg)

可以看到DenseNet里来自跳层的输出不是通过加法（`+`）而是拼接（`concat`）来跟目前层的输出合并。因为是拼接，所以底层的输出会保留的进入上面所有层。这是为什么叫“稠密连接”的原因

## 稠密块（Dense Block）

我们先来定义一个稠密连接块。DenseNet的卷积块使用ResNet改进版本的`BN->Relu->Conv`。每个卷积的输出通道数被称之为`growth_rate`，这是因为假设输出为`in_channels`，而且有`layers`层，那么输出的通道数就是`in_channels+growth_rate*layers`。

```{.python .input}
from mxnet import nd
from mxnet.gluon import nn

def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out

class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x
```

我们验证下输出通道数是不是符合预期。

```{.python .input}
dblk = DenseBlock(2, 10)
dblk.initialize()

x = nd.random.uniform(shape=(4,3,8,8))
dblk(x).shape
```

## 过渡块（Transition Block）
因为使用拼接的缘故，每经过一次拼接输出通道数可能会激增。为了控制模型复杂度，这里引入一个过渡块，它不仅把输入的长宽减半，同时也使用$1\times1$卷积来改变通道数。

```{.python .input}
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out
```

验证一下结果：

```{.python .input}
tblk = transition_block(10)
tblk.initialize()

tblk(x).shape
```

## DenseNet

DenseNet的主体就是交替串联稠密块和过渡块。它使用全局的`growth_rate`使得配置更加简单。过渡层每次都将通道数减半。下面定义一个121层的DenseNet。

```{.python .input}
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10

def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net

```

## 获取数据并训练

因为这里我们使用了比较深的网络，所以我们进一步把输入减少到$32\times 32$来训练。

```{.python .input}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=32)

ctx = utils.try_gpu()
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

## 小结

* Desnet通过将ResNet里的`+`替换成`concat`从而获得更稠密的连接。

## 练习

- DesNet论文中提交的一个优点是其模型参数比ResNet更小，想想为什么？
- DesNet被人诟病的一个问题是内存消耗过多。真的会这样吗？可以把输入换成$224\times 224$（需要改最后的`AvgPool2D`大小），来看看实际（GPU）内存消耗。
- 这里的FashionMNIST有必要用100+层的网络吗？尝试将其改简单看看效果。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1664)

![](../img/qr_densenet-gluon.svg)
