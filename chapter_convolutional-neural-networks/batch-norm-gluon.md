# 批量归一化的Gluon实现

相比于前小节定义的BatchNorm类，`nn`模块定义的BatchNorm使用更加简单。它不需要指定输出数据的维度和特征维的大小，这些都将通过延后初始化来获取。我们实现同前小节一样的批量归一化的LeNet。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)
```

使用同样的超参数进行训练。

```{.python .input  n=3}
lr = 1.0
num_epochs = 5
batch_size = 256
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
```

## 小结

* Gluon提供的BatchNorm在使用上更加简单。

## 练习

* 查看BatchNorm文档来了解更多使用方法，例如如何在训练时使用全局平均的均值和方差。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1254)

![](../img/qr_batch-norm-gluon.svg)
