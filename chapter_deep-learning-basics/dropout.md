# 丢弃法

除了前一节介绍的权重衰减以外，深度学习模型常常使用丢弃法（dropout）[1] 来应对过拟合问题。丢弃法有一些不同的变体。本节中提到的丢弃法特指倒置丢弃法（inverted dropout）。

## 方法和原理

回忆在[“多层感知机”](mlp.md)一节的图3.3描述了一个单隐藏层的多层感知机，其中输入个数为4，隐藏单元个数为5，且隐藏单元$h_i$（$i=1, \ldots, 5$）的计算表达式为

$$h_i = \phi\left(x_1 w_1^{(i)} + x_2 w_2^{(i)} + x_3 w_3^{(i)} + x_4 w_4^{(i)} + b^{(i)}\right),$$

这里$\phi$是激活函数，$x_1, \ldots, x_4$是输入，$w_1^{(i)}, \ldots, w_4^{(i)}$是隐藏单元$i$对应的权重参数，$b^{(i)}$是相应的偏差参数。

当我们对改隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为$p$，有$p$的概率我们将$h_i$设成0，有$1-p$的概率将它除以$1-p$做拉伸。丢弃概率是丢弃法的超参数。具体来说，设随机变量$\xi_i$有$p$概率为0，有$1-p$概率为1。使用丢弃法时我们计算新的隐藏单元$h_i'$

$$h_i' = \frac{\xi_i}{1-p} h_i,$$

由于$\mathbb{E}(\xi_i) = 1-p$，因此

$$\mathbb{E}(h_i') = \frac{\mathbb{E}(\xi_i)}{1-p}h_i = h_i.$$

即丢弃法不改变其输入的期望值。让我们对图3.3中的隐藏层使用丢弃法，一种可能的结果如图3.5所示，其中$h_2$和$h_5$被置为了0。

![隐藏层使用了丢弃法的多层感知机。](../img/dropout.svg)

这时输出值的计算不再依赖$h_2$和$h_5$，在反向传播时，与这两个隐藏单元相关的权重的梯度均为0。因为在训练中我们随机的丢弃神经元，即$h_i$（$i=1, \ldots, 5$）都有可能为0。这样输出层的计算都无法过度依赖$h_1, \ldots, h_5$中的任一个，从而起到正则化的作用，并可以用来应对过拟合。

可以看到丢弃法主要是在训练中起到正则化效果。在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法。

## 丢弃法从零开始实现

根据丢弃法的定义，我们可以很容易地实现它。下面的`dropout`函数将以`drop_prob`的概率丢弃NDArray输入`X`中的元素。

```{.python .input}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import nn, loss as gloss

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃。
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob
```

我们运行几个例子来验证一下`dropout`函数。

```{.python .input}
X = nd.arange(16).reshape((2, 8))
dropout(X, 0)
```

```{.python .input}
dropout(X, 0.5)
```

```{.python .input}
dropout(X, 1)
```

### 定义模型参数

实验中，我们依然使用[“Softmax回归——从零开始”](softmax-regression-scratch.md)一节中介绍的Fashion-MNIST数据集。我们将定义一个包含两个隐藏层的多层感知机。其中两个隐藏层的输出个数都是256。

```{.python .input}
num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

### 定义模型

我们的模型就是将全连接层和激活函数ReLU串起来，并对激活函数的输出使用丢弃法。我们可以分别设置各个层的丢弃概率。通常，建议把靠近输入层的丢弃概率设的小一点。在这个实验中，我们把第一个隐藏层的丢弃概率设为0.2，把第二个隐藏层的丢弃概率设为0.5。我们只需在训练模型时使用丢弃法。

```{.python .input}
drop_prob1 = 0.2
drop_prob2 = 0.5

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    # 只在训练模型时使用丢弃法。
    if autograd.is_training():
        # 在第一层全连接后添加丢弃层。
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        # 在第二层全连接后添加丢弃层。
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3
```

### 训练和测试模型

这部分和之前多层感知机的训练与测试类似。

```{.python .input}
num_epochs = 5
lr = 0.5
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params,
             lr)
```

## 丢弃法的Gluon实现

在Gluon中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时，Dropout层并不发挥作用。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # 在第一个全连接层后添加丢弃层。
        nn.Dropout(drop_prob1),
        nn.Dense(256, activation="relu"),
        # 在第二个全连接层后添加丢弃层。
        nn.Dropout(drop_prob2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

然后同样的训练。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
```

## 小结

* 我们可以通过使用丢弃法应对过拟合。
* 丢弃法只在在训练模型时使用。

## 练习

- 尝试不使用丢弃法，看看这个包含两个隐藏层的多层感知机可以得到什么结果。
- 如果把本节中的两个丢弃概率超参数对调，会有什么结果？
- 多迭代一些周期，来比较使用丢弃法与不使用丢弃法的区别。
- 如果将模型改得更加复杂，例如增加隐藏层单元，是不是丢弃法效果更加明显？
- 比较丢弃法与权重衰减的效果区别。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1278)

![](../img/qr_dropout.svg)

## 参考文献

[1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR
