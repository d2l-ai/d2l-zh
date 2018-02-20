# 网络中的网络

Alexnet之后一个重要的工作是[Network in Network（NiN）](https://arxiv.org/abs/1312.4400)，其提出的两个想法影响了后面的网络设计。

首先一点是注意到卷积神经网络一般分成两块，一块主要由卷积层构成，另一块主要是全连接层。在Alexnet里我们看到如何把卷积层块和全连接层分别加深加宽从而得到深度网络。另外一个自然的想法是，我们可以串联数个卷积层块和全连接层块来构建深度网络。

![](../img/nin.svg)

不过这里的一个难题是，卷积的输入输出是4D矩阵，然而全连接是2D。同时在[卷积神经网络](./cnn-scratch.md)里我们提到如果把4D矩阵转成2D做全连接，这个会导致全连接层有过多的参数。NiN提出只对通道层做全连接并且像素之间共享权重来解决上述两个问题。就是说，我们使用kernel大小是$1 \times 1$的卷积。

下面代码定义一个这样的块，它由一个正常的卷积层接上两个kernel是$1 \times 1$的卷积层构成。后面两个充当两个全连接层的角色。

```{.python .input  n=2}
from mxnet.gluon import nn

def mlpconv(channels, kernel_size, padding,
            strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out
```

测试一下：

```{.python .input  n=8}
from mxnet import nd

blk = mlpconv(64, 3, 0)
blk.initialize()

x = nd.random.uniform(shape=(32, 3, 16, 16))
y = blk(x)
y.shape
```

NiN的卷积层的参数跟Alexnet类似，使用三组不同的设定

- kernel: $11\times 11$, channels: 96
- kernel: $5\times 5$, channels: 256
- kernel: $3\times 3$, channels: 384

除了使用了$1\times 1$卷积外，NiN在最后不是使用全连接，而是使用通道数为输出类别个数的`mlpconv`，外接一个平均池化层来将每个通道里的数值平均成一个标量。

```{.python .input  n=9}
net = nn.Sequential()
# add name_scope on the outer most Sequential
with net.name_scope():
    net.add(
        mlpconv(96, 11, 0, strides=4),
        mlpconv(256, 5, 2),
        mlpconv(384, 3, 1),
        nn.Dropout(.5),
        # 目标类为10类
        mlpconv(10, 3, 1, max_pooling=False),
        # 输入为 batch_size x 10 x 5 x 5, 通过AvgPool2D转成
        # batch_size x 10 x 1 x 1。
        # 我们可以使用 nn.AvgPool2D(pool_size=5), 
        # 但更方便是使用全局池化，可以避免估算pool_size大小
        nn.GlobalAvgPool2D(),
        # 转成 batch_size x 10
        nn.Flatten()
    )

```

## 获取数据并训练

跟Alexnet类似，但使用了更大的学习率。

```{.python .input  n=4}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=224)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

## 结论

这种“一卷卷到底”最后加一个平均池化层的做法也成为了深度卷积神经网络的常用设计。

## 练习

- 为什么mlpconv里面要有两个$1\times 1$卷积？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1661)
