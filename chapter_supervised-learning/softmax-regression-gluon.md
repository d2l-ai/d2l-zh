# Softmax回归——使用Gluon

现在让我们使用gluon来更快速地实现一个多类逻辑回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../utils.py](../utils.py)这样这里不用复制一遍。

```{.python .input  n=1}
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, loss as gloss
```

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size` x `?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input  n=3}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Flatten())
    net.add(nn.Dense(10))
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练

```{.python .input  n=6}
num_epochs = 5

gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             trainer)
```

## 小结

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

- 再尝试调大下学习率看看？
- 为什么参数都差不多，但gluon版本比从0开始的版本精度更高？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/740)

![](../img/qr_softmax-regression-gluon.svg)
