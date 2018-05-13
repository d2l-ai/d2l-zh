# 多GPU计算——使用Gluon

在Gluon中，我们可以很方便地使用数据并行进行多GPU计算。比方说，我们并不需要自己实现[“多GPU计算——从零开始”](./multiple-gpus-scratch.md)一节里介绍的多GPU之间同步数据的辅助函数。

先导入本节实验需要的包。同上一节，运行本节中的程序需要至少两块GPU。

```{.python .input}
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, utils as gutils
import sys
from time import time
sys.path.append('..')
import utils
```

## 多GPU上初始化模型参数

我们使用ResNet-18来作为本节的样例模型。

```{.python .input  n=1}
net = utils.resnet18(10)
```

之前我们介绍了如何使用`initialize`函数的`ctx`参数在CPU或单个GPU上初始化模型参数。事实上，`ctx`可以接受一系列的CPU/GPU，从而使初始化好的模型参数复制到`ctx`里所有的CPU/GPU上。

```{.python .input}
ctx = [mx.gpu(0), mx.gpu(1)]
net.initialize(ctx=ctx)
```

Gluon提供了上一节中实现的`split_and_load`函数。它可以划分一个小批量的数据样本并复制到各个CPU/GPU上。之后，根据输入数据所在的CPU/GPU，模型计算会发生在相同的CPU/GPU上。

```{.python .input}
x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
print(net(gpu_x[0]))
print(net(gpu_x[1]))
```

回忆一下[“模型参数的延后初始化”](../chapter_gluon-basics/deferred-init.md)一节中介绍的延后的初始化。现在，我们可以通过`data`访问初始化好的模型参数值了。需要注意的是，默认下`weight.data()`会返回CPU上的参数值。由于我们指定了2个GPU来初始化模型参数，我们需要指定GPU访问。我们看到，相同参数在不同的GPU上的值一样。

```{.python .input}
weight = net[1].params.get('weight')
try:
    weight.data()
except:
    print('not initialized on', mx.cpu())
print(weight.data(ctx[0])[0])
print(weight.data(ctx[1])[0])
```

## 多GPU训练模型

我们先定义交叉熵损失函数。

```{.python .input}
loss = gloss.SoftmaxCrossEntropyLoss()
```

当我们使用多个GPU来训练模型时，`gluon.Trainer`会自动做数据并行，例如划分小批量数据样本并复制到各个GPU上，对各个GPU上的梯度求和再广播到所有GPU上。这样，我们就可以很方便地实现训练函数了。

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(1, 6):
        start = time()
        for X, y in train_data:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y) for gpu_X, gpu_y in zip(
                    gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        print('epoch %d, training time: %.1f sec'%(epoch, time() - start))
        test_acc = utils.evaluate_accuracy(test_data, net, ctx[0])
        print('validation accuracy: %.4f'%(test_acc))
```

我们在2个GPU上训练模型。

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.3)
```

## 小结

* 在Gluon中，我们可以很方便地进行多GPU计算，例如在多GPU上初始化模型参数和训练模型。

## 练习

* 本节使用了ResNet-18。试试不同的迭代周期、批量大小和学习率。如果条件允许，使用更多GPU计算。
* 有时候，不同的CPU/GPU的计算能力不一样，例如同时使用CPU和GPU，或者GPU之间型号不一样。这时候应该怎么办？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1885)

![](../img/qr_multiple-gpus-gluon.svg)
