# 多层感知机的Gluon实现

下面我们使用Gluon来实现上一节中的多层感知机。首先我们导入所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## 定义模型

和Softmax回归唯一的不同在于，我们多加了一个全连接层作为隐藏层。它的隐藏单元个数为256，并使用ReLU作为激活函数。

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## 读取数据并训练模型

我们使用和训练Softmax回归几乎相同的步骤来读取数据并训练模型。

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
```

## 小结

通过Gluon我们可以更方便地构造多层感知机。

## 练习

- 尝试多加入几个隐藏层，对比上节中从零开始的实现。
- 使用其他的激活函数，看看对结果的影响。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/738)

![](../img/qr_mlp-gluon.svg)
