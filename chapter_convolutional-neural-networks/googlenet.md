# 含并行连结的网络（GoogLeNet）

在2014年的Imagenet竞赛中，一个名叫GoogLeNet的网络结构大放光彩 [1]。它虽然在名字上是向LeNet致敬，但在网络结构上已经很难看到LeNet的影子。GoogLeNet吸收了NiN的网络嵌套网络的想法，并在此基础上做了很大的改进。在随后的几年里研究人员对它进行了数次改进，本小节将介绍这个模型系列的第一个版本。

## Inception 块

GoogLeNet中的基础卷积块叫做Inception，得名于同名电影《盗梦空间》（Inception），寓意梦中嵌套梦。比较上一节介绍的NiN，这个基础块在结构上更加复杂。

![Inception块。](../img/inception.svg)

由图5.8可以看出，Inception里有四个并行的线路。前三个线路里使用窗口大小分别是$1\times 1$、$3\times 3$和$5\times 5$的卷积层来抽取不同空间尺寸下的信息。其中中间两个线路会对输入先作用$1\times 1$卷积来减小输入通道数，以此降低模型复杂度。第四条线路则是使用$3\times 3$最大池化层，后接$1\times 1$卷积层来变换通道。四条线路都使用了合适的填充来使得输入输出高宽一致。最后我们将每条线路的输出在通道维上合并，输入到接下来的层中去。

Inception块中可以自定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import loss as gloss, nn

class Inception(nn.Block):
    # c1 - c4 为每条线路里的层的输出通道数。
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路 1，单 1 x 1 卷积层。
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 线路 2，1 x 1 卷积层后接 3 x 3 卷积层。
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # 线路 3，1 x 1 卷积层后接 5 x 5 卷积层。
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # 线路 4，3 x 3 最大池化层后接 1 x 1 卷积层。
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维上合并输出。
        return nd.concat(p1, p2, p3, p4, dim=1)
```

## GoogLeNet模型

GoogLeNet跟VGG一样，在主体卷积部分中使用五个模块，每个模块之间使用步幅为2的$3\times 3$最大池化层来减小输出高宽。第一模块使用一个64通道的$7\times 7$卷积层。

```{.python .input  n=2}
b1 = nn.Sequential()
b1.add(
    nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)
```

第二模块使用两个卷积层，首先是64通道的$1\times 1$卷积层，然后是将通道增大3倍的$3\times 3$卷积层。它对应Inception块中的第二线路。

```{.python .input  n=3}
b2 = nn.Sequential()
b2.add(
    nn.Conv2D(64, kernel_size=1),
    nn.Conv2D(192, kernel_size=3, padding=1),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)
```

第三模块串联两个完整的Inception块。第一个Inception块的输出通道数为256,其中四个线路的输出通道比例为2：4：1：1。且第二、三线路先分别将输入通道减小2倍和12倍后再进入第二层卷积层。第二个Inception块输出通道数增至480，每个线路的通道比例为4：6：3：2。且第二、三线路先分别减少2倍和8倍通道数。

```{.python .input  n=4}
b3 = nn.Sequential()
b3.add(
    Inception(64, (96, 128), (16, 32), 32),
    Inception(128, (128, 192), (32, 96), 64),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)
```

第四模块更加复杂，它串联了五个Inception块，其输出通道分别是512、512、512、528和832。其线路的通道分配类似之前，$3\times 3$卷积层线路输出最多通道，其次是$1\times 1$卷积层线路，之后是$5\times 5$卷积层和$3\times 3$最大池化层线路。其中前两个线路都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。

```{.python .input  n=5}
b4 = nn.Sequential()
b4.add(
    Inception(192, (96, 208), (16, 48), 64),
    Inception(160, (112, 224), (24, 64), 64),
    Inception(128, (128, 256), (24, 64), 64),
    Inception(112, (144, 288), (32, 64), 64),
    Inception(256, (160, 320), (32, 128), 128),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)
```

第五模块有输出通道数为832和1024的两个Inception块，每个线路的通道分配使用同前的原则，但具体数字又是不同。因为这个模块后面紧跟输出层，所以它同NiN一样使用全局平均池化层来将每个通道高宽变成1。最后我们将输出变成二维数组后加上一个输出大小为标签类数的全连接层作为输出。

```{.python .input  n=6}
b5 = nn.Sequential()
b5.add(
    Inception(256, (160, 320), (32, 128), 128),
    Inception(384, (192, 384), (48, 128), 128),
    nn.GlobalAvgPool2D()
)

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

因为这个模型相计算复杂，而且修改通道数不如VGG那样简单。本节里我们将输入高宽从224降到96来加速计算。下面演示各个模块之间的输出形状变化。

```{.python .input  n=7}
X = nd.random.uniform(shape=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## 获取数据并训练

我们使用高宽为96的数据来训练。

```{.python .input  n=8}
lr = 0.1
num_epochs = 5
batch_size = 128
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,
                                                   resize=96)
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
```

## 小结

* Inception块相当于一个有四条线路的子网络，它通过不同窗口大小的卷积层和最大池化层来并行抽取信息，并使用$1\times 1$卷积层减低通道数来减少模型复杂度。GoogLeNet将多个精细设计的Inception块和其他层串联起来。其通道分配比例是在ImageNet数据集上通过大量的实验得来。GoogLeNet和它的后继者一度是ImageNet上最高效的模型之一，即在给定同样的测试精度下计算复杂度更低。

## 练习

* GoogLeNet有数个后续版本，尝试实现他们并运行看看有什么不一样。这些后续版本包括加入批量归一化层（后一小节将介绍）[2]、对Inception块做调整 [3] 和加入残差连接（后面小节将介绍）[4]。

* 对比AlexNet、VGG和NiN、GoogLeNet的模型参数大小。分析为什么后两个网络可以显著减小模型大小。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1662)

![](../img/qr_googlenet-gluon.svg)

## 参考文献

[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., & Anguelov, D. & Rabinovich, A.(2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).

[2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.

[3] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826).

[4] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 4, p. 12).
