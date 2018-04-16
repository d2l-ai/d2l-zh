# 多GPU训练模型——使用Gluon


在Gluon里可以很容易的使用数据并行。在[多GPU来训练 --- 从0开始](./multiple-gpus-scratch.md)里我们手动实现了几个数据同步函数来使用数据并行，Gluon里实现了同样的功能。


## 多设备上的初始化

之前我们介绍了如果使用`initialize()`里的`ctx`在CPU或者特定GPU上初始化模型。事实上，`ctx`可以接受一系列的设备，它会将初始好的参数复制所有的设备上。

这里我们使用之前介绍Resnet18来作为演示。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
from mxnet import gpu
from mxnet import cpu

net = utils.resnet18(10)
ctx = [gpu(0), gpu(1)]
net.initialize(ctx=ctx)
```

记得前面提到的[延迟初始化](../chapter_gluon-basics/parameters.md)，这里参数还没有被初始化。我们需要先给定数据跑一次。

Gluon提供了之前我们实现的`split_and_load`函数，它将数据分割并返回各个设备上的复制。然后根据输入的设备，计算也会在相应的数据上执行。

```{.python .input}
from mxnet import nd
from mxnet import gluon

x = nd.random.uniform(shape=(4, 1, 28, 28))
x_list = gluon.utils.split_and_load(x, ctx)
print(net(x_list[0]))
print(net(x_list[1]))
```

这时候我们可以来看初始的过程发生了什么了。记得我们可以通过`data`来访问参数值，它默认会返回CPU上值。但这里我们只在两个GPU上初始化了，在访问的对应设备的值的时候，我们需要指定设备。

```{.python .input}
weight = net[1].params.get('weight')
print(weight.data(ctx[0])[0])
print(weight.data(ctx[1])[0])
try:
    weight.data(cpu())
except:
    print('Not initialized on', cpu())
```

上一章我们提到过如何在多GPU之间复制梯度求和并广播，这个在`gluon.Trainer`里面会被默认执行。这样我们可以实现完整的训练函数了。

## 训练

```{.python .input  n=7}
from mxnet import gluon
from mxnet import autograd
from time import time
from mxnet import init

def train(num_gpus, batch_size, lr):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on', ctx)

    net = utils.resnet18(10)
    net.initialize(init=init.Xavier(), ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(
        net.collect_params(),'sgd', {'learning_rate': lr})

    for epoch in range(5):
        start = time()
        total_loss = 0
        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx)
            label_list = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                losses = [loss(net(X), y) for X, y in zip(
                    data_list, label_list)]
            for l in losses:
                l.backward()
            total_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)

        nd.waitall()
        print('Epoch %d, training time = %.1f sec'%(
            epoch, time()-start))

        test_acc = utils.evaluate_accuracy(test_data, net, ctx[0])
        print('         validation accuracy = %.4f'%(test_acc))
```

尝试在单GPU上执行。

```{.python .input}
train(1, 256, .1)
```

同样的参数，但使用两个GPU。

```{.python .input}
train(2, 256, .1)
```

增大批量值和学习率

```{.python .input}
train(2, 512, .2)
```

## 小结

* Gluon的参数初始化和Trainer都支持多设备，从单设备到多设备非常容易。

## 练习

* 跟[多GPU来训练 --- 从0开始](./multiple-gpus-scratch.md)不一样，这里我们使用了更现代些的ResNet。看看不同的批量大小和学习率对不同GPU个数上的不一样。
* 有时候各个设备计算能力不一样，例如同时使用CPU和GPU，或者GPU之间型号不一样，这时候应该怎么办？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1885)

![](../img/qr_multiple-gpus-gluon.svg)
