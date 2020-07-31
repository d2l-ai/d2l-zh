# 多层感知器的简洁实现
:label:`sec_mlp_concise`

正如您所期望的那样，通过依靠高级 API，我们可以更简洁地实现 MLP。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 模型

与我们简明的 softmax 回归实现（:numref:`sec_softmax_concise`）相比，唯一的区别是我们添加
*两个 * 完全连接的层
（以前，我们添加了 * 一个 *）。第一个是我们的隐藏层，其中包含 256 个隐藏单位，并应用 RelU 激活函数。第二个是我们的输出图层。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

训练循环与我们实施 softmax 回归时完全相同。这种模块化使我们能够将有关模型架构的事项与正交考虑分开。

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 摘要

* 使用高级 API，我们可以更简洁地实现 MLP。
* 对于同一分类问题，除了带有激活函数的附加隐藏图层外，MLP 的实现与 softmax 回归的实现相同。

## 练习

1. 尝试添加不同数量的隐藏图层（您也可以修改学习率）。什么设置最适合？
1. 尝试不同的激活功能。哪一个最适合？
1. 尝试初始化权重的不同方案。什么方法最适合？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
