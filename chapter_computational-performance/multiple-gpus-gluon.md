# 多GPU计算的简洁实现

在Gluon中，我们可以很方便地使用数据并行进行多GPU计算。例如，我们并不需要自己实现[“多GPU计算”](multiple-gpus.md)一节里介绍的多GPU之间同步数据的辅助函数。

首先导入本节实验所需的包或模块。运行本节中的程序需要至少2块GPU。

```{.python .input  n=1}
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
import time
```

## 多GPU上初始化模型参数

我们使用ResNet-18作为本节的样例模型。由于本节的输入图像使用原尺寸（未放大），这里的模型构造与[“残差网络（ResNet）”](../chapter_convolutional-neural-networks/resnet.md)一节中的ResNet-18构造稍有不同。这里的模型在一开始使用了较小的卷积核、步幅和填充，并去掉了最大池化层。

```{.python .input  n=2}
def resnet18(num_classes):  # 本函数已保存在d2lzh包中方便以后使用
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # 这里使用了较小的卷积核、步幅和填充，并去掉了最大池化层
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net

net = resnet18(10)
```

之前我们介绍了如何使用`initialize`函数的`ctx`参数在内存或单块显卡的显存上初始化模型参数。事实上，`ctx`可以接受一系列的CPU及内存和GPU及相应的显存，从而使初始化好的模型参数复制到`ctx`里所有的内存和显存上。

```{.python .input  n=3}
ctx = [mx.gpu(0), mx.gpu(1)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
```

Gluon提供了上一节中实现的`split_and_load`函数。它可以划分一个小批量的数据样本并复制到各个内存或显存上。之后，根据输入数据所在的内存或显存，模型计算会相应地使用CPU或相同显卡上的GPU。

```{.python .input  n=4}
x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
net(gpu_x[0]), net(gpu_x[1])
```

现在，我们可以访问已初始化好的模型参数值了。需要注意的是，默认情况下`weight.data()`会返回内存上的参数值。因为我们指定了2块GPU来初始化模型参数，所以需要指定显存来访问参数值。我们看到，相同参数在不同显卡的显存上的值一样。

```{.python .input  n=5}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on', mx.cpu())
weight.data(ctx[0])[0], weight.data(ctx[1])[0]
```

## 多GPU训练模型

当使用多块GPU来训练模型时，`Trainer`实例会自动做数据并行，例如，划分小批量数据样本并复制到各块显卡的显存上，以及对各块显卡的显存上的梯度求和再广播到所有显存上。这样，我们就可以很方便地实现训练函数了。

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
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
        train_time = time.time() - start
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test acc %.2f' % (
            epoch + 1, train_time, test_acc))
```

首先在单GPU上训练模型。

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

然后尝试在2块GPU上训练模型。与上一节使用的LeNet相比，ResNet-18的计算更加复杂，通信时间比计算时间更短，因此ResNet-18的并行计算所获得的性能提升更佳。

```{.python .input  n=10}
train(num_gpus=2, batch_size=512, lr=0.2)
```

## 小结

* 在Gluon中，可以很方便地进行多GPU计算，例如，在多GPU及相应的显存上初始化模型参数和训练模型。


## 练习

* 本节使用了ResNet-18模型。试试不同的迭代周期、批量大小和学习率。如果条件允许，使用更多GPU来计算。
* 有时候，不同设备的计算能力不一样，例如，同时使用CPU和GPU，或者不同GPU之间型号不一样。这时候，应该如何将小批量划分到内存或不同显卡的显存？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1885)

![](../img/qr_multiple-gpus-gluon.svg)
