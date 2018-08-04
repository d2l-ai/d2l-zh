# 多GPU计算的Gluon实现

在Gluon中，我们可以很方便地使用数据并行进行多GPU计算。比方说，我们并不需要自己实现[“多GPU计算”](multiple-gpus.md)一节里介绍的多GPU之间同步数据的辅助函数。

先导入本节实验需要的包或模块。同上一节，运行本节中的程序需要至少两块GPU。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
from time import time
```

## 多GPU上初始化模型参数

我们使用ResNet-18来作为本节的样例模型。由于本节的输入图片使用原尺寸（未放大），这里的模型构造与[“残差网络（ResNet）”](../chapter_convolutional-neural-networks/resnet.md)一节中的ResNet-18构造稍有不同。这里的模型在一开始使用了较小的卷积核、步幅和填充，并去掉了最大池化层。我们将`resnet18`函数定义在`gluonbook`包中供后面章节调用。

```{.python .input  n=2}
def resnet18(num_classes):
    net = nn.Sequential()
    # 这里使用了较小的卷积核、步幅和填充，并去掉了最大池化层。
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))                 

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(gb.Residual(num_channels, use_1x1conv=True,
                                    strides=2))
            else:
                blk.add(gb.Residual(num_channels))
        return blk 

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2), 
            resnet_block(256, 2), 
            resnet_block(512, 2)) 
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net 

net = resnet18(10)
```

之前我们介绍了如何使用`initialize`函数的`ctx`参数在CPU或单个GPU上初始化模型参数。事实上，`ctx`可以接受一系列的CPU/GPU，从而使初始化好的模型参数复制到`ctx`里所有的CPU/GPU上。

```{.python .input  n=3}
ctx = [mx.gpu(0), mx.gpu(1)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
```

Gluon提供了上一节中实现的`split_and_load`函数。它可以划分一个小批量的数据样本并复制到各个CPU/GPU上。之后，根据输入数据所在的CPU/GPU，模型计算会发生在相同的CPU/GPU上。

```{.python .input  n=4}
x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
net(gpu_x[0]), net(gpu_x[1])
```

回忆一下[“模型参数的延后初始化”](../chapter_deep-learning-computation/deferred-init.md)一节中介绍的延后的初始化。现在，我们可以通过`data`访问初始化好的模型参数值了。需要注意的是，默认下`weight.data()`会返回CPU上的参数值。由于我们指定了2个GPU来初始化模型参数，我们需要指定GPU访问。我们看到，相同参数在不同的GPU上的值一样。

```{.python .input  n=5}
weight = net[0].params.get('weight')
try:
    weight.data()
except:
    print('not initialized on', mx.cpu())
weight.data(ctx[0])[0], weight.data(ctx[1])[0]
```

## 多GPU训练模型

我们先定义交叉熵损失函数。

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

当我们使用多个GPU来训练模型时，`gluon.Trainer`会自动做数据并行，例如划分小批量数据样本并复制到各个GPU上，对各个GPU上的梯度求和再广播到所有GPU上。这样，我们就可以很方便地实现训练函数了。

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(1, 6):
        start = time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y)
                      for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        print('epoch %d, training time: %.1f sec' % (epoch, time() - start))
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('validation accuracy: %.4f' % (test_acc))
```

我们在2个GPU上训练模型。

```{.python .input  n=10}
train(num_gpus=2, batch_size=512, lr=0.3)
```

## 小结

* 在Gluon中，我们可以很方便地进行多GPU计算，例如在多GPU上初始化模型参数和训练模型。

## 练习

* 本节使用了ResNet-18。试试不同的迭代周期、批量大小和学习率。如果条件允许，使用更多GPU计算。
* 有时候，不同的CPU/GPU的计算能力不一样，例如同时使用CPU和GPU，或者GPU之间型号不一样。这时候应该怎么办？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1885)

![](../img/qr_multiple-gpus-gluon.svg)
