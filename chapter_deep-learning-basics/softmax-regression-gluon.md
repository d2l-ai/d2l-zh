# Softmax回归的Gluon实现

我们在[“线性回归的Gluon实现”](linear-regression-gluon.md)一节中已经了解了使用Gluon实现模型的便利。下面，让我们使用Gluon来实现一个Softmax回归模型。首先导入本节实现所需的包或模块。

```{.python .input  n=1}
%matplotlib inline
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## 获取和读取数据

我们仍然使用Fashion-MNIST数据集和上一节中相同的批量大小。

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## 定义和初始化模型

在[“Softmax回归”](softmax-regression.md)一节中，我们提到Softmax回归的输出层是一个全连接层。因此，我们添加一个输出个数为10的全连接层。我们使用均值为0标准差为0.01的正态分布随机初始化模型的权重参数。

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## Softmax和交叉熵损失函数

如果你做了上一节的练习，那么你可能意识到了分开定义Softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，Gluon提供了一个包括Softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 定义优化算法

我们使用学习率为0.1的小批量随机梯度下降作为优化算法。

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练模型

接下来，我们使用上一节中定义的训练函数来训练模型。

```{.python .input  n=6}
num_epochs = 5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             None, trainer)
```

## 小结

* Gluon提供的函数往往具有更好的数值稳定性。
* 我们可以使用Gluon更简洁地实现Softmax回归。

## 练习

* 尝试调一调超参数，例如批量大小、迭代周期和学习率，看看结果会怎样。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/740)

![](../img/qr_softmax-regression-gluon.svg)
