# Adagrad


在我们之前介绍过的优化算法中，无论是梯度下降、随机梯度下降、小批量随机梯度下降还是使用动量法，目标函数自变量的每一个元素在相同时刻都使用同一个学习率来自我迭代。举个例子，假设目标函数为$f$，自变量为一个多维向量$[x_1, x_2]^\top$，该向量中每一个元素在更新时都使用相同的学习率。例如在学习率为$\eta$的梯度下降中，元素$x_1$和$x_2$都使用相同的学习率$\eta$来自我迭代：

$$
x_1 \leftarrow x_1 - \eta \frac{\partial{f}}{\partial{x_1}}, \quad
x_2 \leftarrow x_2 - \eta \frac{\partial{f}}{\partial{x_2}}.
$$

在[“动量法”](./momentum.md)一节里我们看到当$x_1$和$x_2$的梯度值有较大差别时，我们需要选择足够小的学习率使得自变量在梯度值较大的维度上不发散。但这样会导致自变量在梯度值较小的维度上迭代过慢。动量法依赖指数加权移动平均使得自变量的更新方向更加一致，从而降低发散的可能。这一节我们介绍Adagrad算法，它根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。


## Adagrad算法

Adagrad的算法会使用一个小批量随机梯度按元素平方的累加变量$\boldsymbol{s}$，其形状与自变量形状相同。开始时将变量$\boldsymbol{s}$中每个元素初始化为0。在每次迭代中，首先计算小批量随机梯度$\boldsymbol{g}$，然后将该梯度按元素平方后累加到变量$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \boldsymbol{s} + \boldsymbol{g} \odot \boldsymbol{g},$$

其中$\odot$是按元素相乘（请参见[“数学基础”](../chapter_appendix/math.md)一节）。接着，我们将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：

$$\boldsymbol{g}' \leftarrow \frac{\eta}{\sqrt{\boldsymbol{s} + \epsilon}} \odot \boldsymbol{g},$$

其中$\eta$是初始学习率且$\eta > 0$，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-7}$。我们需要注意，其中开方、除法和乘法的运算都是按元素进行的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

最后，自变量的迭代步骤与小批量随机梯度下降类似。只是这里梯度前的学习率已经被调整过了：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}'.$$




## Adagrad的特点

需要强调的是，小批量随机梯度按元素平方的累加变量$\boldsymbol{s}$出现在含调整后学习率的梯度$\boldsymbol{g}'$的分母项。因此，如果目标函数有关自变量中某个元素的偏导数一直都较大，那么就让该元素的学习率下降快一点；反之，如果目标函数有关自变量中某个元素的偏导数一直都较小，那么就让该元素的学习率下降慢一点。然而，由于$\boldsymbol{s}$一直在累加按元素平方的梯度，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad在迭代后期由于学习率过小，可能较难找到一个有用的解。



## 实验

首先，导入本节中实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, nd
import numpy as np
import math
```

实验中，我们以之前介绍过的线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。

我们把梯度按元素平方的累加变量初始化为和模型参数形状相同的零张量。

```{.python .input  n=2}
# 生成数据集。
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 初始化模型参数。
def init_params():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    sqrs = []
    for param in params:
        param.attach_grad()
        # 把梯度按元素平方的累加变量初始化为和参数形状相同的零张量。
        sqrs.append(param.zeros_like())
    return params, sqrs
```

接下来基于NDArray实现Adagrad算法。

```{.python .input  n=1}
def adagrad(params, sqrs, lr, batch_size):
    eps_stable = 1e-7
    for param, sqr in zip(params, sqrs):
        g = param.grad / batch_size
        sqr[:] += g.square()
        param[:] -= lr * g / (sqr + eps_stable).sqrt()
```

优化函数`optimize`与[“梯度下降和随机梯度下降”](gd-sgd.md)一节中的类似。需要指出的是，这里的初始学习率`lr`无需自我衰减。

```{.python .input  n=3}
net = gb.linreg
loss = gb.squared_loss

def optimize(batch_size, lr, num_epochs, log_interval):
    [w, b], sqrs = init_params()
    ls = [loss(net(features, w, b), labels).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        for batch_i, (X, y) in enumerate(
            gb.data_iter(batch_size, features, labels)):
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()
            adagrad([w, b], sqrs, lr, batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features, w, b), labels).mean().asnumpy())
    print('w[0]=%.2f, w[1]=%.2f, b=%.2f'
          % (w[0].asscalar(), w[1].asscalar(), b.asscalar()))
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    gb.semilogy(es, ls, 'epoch', 'loss')
```

最终，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=4}
optimize(batch_size=10, lr=0.9, num_epochs=3, log_interval=10)
```

## 小结

* Adagrad在迭代过程中不断调整学习率，并让目标函数自变量中每个元素都分别拥有自己的学习率。
* 使用Adagrad时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。


## 练习

* 在介绍Adagrad的特点时，我们提到了它可能存在的问题。你能想到什么办法来应对这个问题？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2273)

![](../img/qr_adagrad.svg)


## 参考文献

[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.
