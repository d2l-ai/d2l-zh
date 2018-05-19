# 丢弃法（Dropout）——使用Gluon

本章介绍如何使用``Gluon``在训练和测试深度学习模型中使用丢弃法 (Dropout)。


## 定义模型并添加丢弃层

有了`Gluon`，我们模型的定义工作变得简单了许多。我们只需要在全连接层后添加`gluon.nn.Dropout`层并指定元素丢弃概率。一般情况下，我们推荐把
更靠近输入层的元素丢弃概率设的更小一点。这个试验中，我们把第一层全连接后的元素丢弃概率设为0.2，把第二层全连接后的元素丢弃概率设为0.5。

```{.python .input}
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
```

```{.python .input  n=5}
drop_prob1 = 0.2
drop_prob2 = 0.5

net = nn.Sequential()
net.add(nn.Flatten())
# 第一层全连接。
net.add(nn.Dense(256, activation="relu"))
# 在第一层全连接后添加丢弃层。
net.add(nn.Dropout(drop_prob1))
# 第二层全连接。
net.add(nn.Dense(256, activation="relu"))
# 在第二层全连接后添加丢弃层。
net.add(nn.Dropout(drop_prob2))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## 读取数据并训练

这跟之前没什么不同。

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 
                        'sgd', {'learning_rate': 0.5})
num_epochs = 5
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
```

## 小结

通过`Gluon`我们可以更方便地构造多层神经网络并使用丢弃法。

## 练习

* 尝试不同元素丢弃概率参数组合，看看结果有什么不同。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1279)

![](../img/qr_dropout-gluon.svg)
