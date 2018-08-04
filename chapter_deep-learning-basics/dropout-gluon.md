# 丢弃法的Gluon实现

本节中，我们将上一节的实验代码用Gluon实现一遍。你会发现代码将精简很多。


## 定义模型并添加丢弃层

在多层感知机中Gluon实现的基础上，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时，Dropout层并不发挥作用。

```{.python .input  n=5}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

drop_prob1 = 0.2
drop_prob2 = 0.5

net = nn.Sequential()
net.add(nn.Flatten())
net.add(nn.Dense(256, activation="relu"))
# 在第一个全连接层后添加丢弃层。
net.add(nn.Dropout(drop_prob1))
net.add(nn.Dense(256, activation="relu"))
# 在第二个全连接层后添加丢弃层。
net.add(nn.Dropout(drop_prob2))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## 训练和测试模型

这部分依然和多层感知机中的训练和测试没有多少区别。

```{.python .input  n=6}
num_epochs = 5
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
```

## 小结

* 使用Gluon，我们可以更方便地构造多层神经网络并使用丢弃法。

## 练习

* 尝试不同丢弃概率超参数组合，观察并分析结果。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1279)

![](../img/qr_dropout-gluon.svg)
