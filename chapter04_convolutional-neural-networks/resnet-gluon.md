# ResNet：深度残差网络

当大家还在惊叹GoogLeNet用结构化的连接纳入了大量卷积层的时候，微软亚洲研究院的研究员已经在设计更深但结构更简单的网络[ResNet](https://arxiv.org/abs/1512.03385)。他们凭借这个网络在2015年的Imagenet竞赛中大获全胜。

ResNet有效的解决了深度卷积神经网络难训练的问题。这是因为在误差反传的过程中，梯度通常变得越来越小，从而权重的更新量也变小。这个导致远离损失函数的层训练缓慢，随着层数的增加这个现象更加明显。之前有两种常用方案来尝试解决这个问题：

1. 按层训练。先训练靠近数据的层，然后慢慢的增加后面的层。但效果不是特别好，而且比较麻烦。
2. 使用更宽的层（增加输出通道）而不是更深来增加模型复杂度。但更宽的模型经常不如更深的效果好。

ResNet通过增加跨层的连接来解决梯度逐层回传时变小的问题。虽然这个想法之前就提出过了，但ResNet真正的把效果做好了。

下图演示了一个跨层的连接。

![](../img/residual.svg)


最底下那层的输入不仅仅是输出给了中间层，而且其与中间层结果相加进入最上层。这样在梯度反传时，最上层梯度可以直接跳过中间层传到最下层，从而避免最下层梯度过小情况。

为什么叫做残差网络呢？我们可以将上面示意图里的结构拆成两个网络的和，一个一层，一个两层，最下面层是共享的。

![](../img/residual2.svg)

在训练过程中，左边的网络因为更简单所以更容易训练。这个小网络没有拟合到的部分，或者说残差，则被右边的网络抓取住。所以直观上来说，即使加深网络，跨层连接仍然可以使得底层网络可以充分的训练，从而不会让训练更难。

## Residual块

ResNet沿用了VGG的那种全用$3\times 3$卷积，但在卷积和池化层之间加入了批量归一层来加速训练。每次跨层连接跨过两层卷积。这里我们定义一个这样的残差块。


```{.python .input  n=22}
from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = nd.relu(self.bn2(self.conv2(x)))
        if not self.same_shape:
            x = self.conv3(x)
        return out + x
```

测试1

```{.python .input  n=23}
blk = Residual(3)
blk.initialize()

x = nd.random.uniform(shape=(4, 3, 6, 6))
blk(x).shape
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "(4, 3, 6, 6)"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

测试2

```{.python .input  n=24}
blk2 = Residual(8, same_shape=False)
blk2.initialize()
blk2(x).shape
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "(4, 8, 6, 6)"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 构建ResNet

未完成
