# 多层感知机的简洁实现

下面我们使用Gluon来实现上一节中的多层感知机。首先导入所需的包或模块。

```{.python .input}
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## 定义模型

和softmax回归唯一的不同在于，我们多加了一个全连接层作为隐藏层。它的隐藏单元个数为256，并使用ReLU函数作为激活函数。

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## 读取数据并训练模型

我们使用与[“softmax回归的简洁实现”](softmax-regression-gluon.md)一节中训练softmax回归几乎相同的步骤来读取数据并训练模型。

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

## 小结

* 通过Gluon可以更简洁地实现多层感知机。

## 练习

* 尝试多加入几个隐藏层，对比上一节中从零开始的实现。
* 使用其他的激活函数，看看对结果的影响。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/738)

![](../img/qr_mlp-gluon.svg)
