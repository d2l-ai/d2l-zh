# Adagrad


在我们之前介绍过的优化算法中，无论是梯度下降、随机梯度下降、小批量随机梯度下降还是使用动量法，目标函数自变量的每一个元素在相同时刻都使用同一个学习率来自我迭代。举个例子，假设目标函数为$f$，自变量为一个多维向量$[x_1, x_2]^\top$，该向量中每一个元素在更新时都使用相同的学习率。例如在学习率为$\eta$的梯度下降中，元素$x_1$和$x_2$都使用相同的学习率$\eta$来自我迭代：

$$
x_1 \leftarrow x_1 - \eta \frac{\partial{f}}{\partial{x_1}}, \quad
x_2 \leftarrow x_2 - \eta \frac{\partial{f}}{\partial{x_2}}.
$$

在[“动量法”](./momentum.md)一节里我们看到当$x_1$和$x_2$的梯度值有较大差别时（我们使用的例子是20倍不同），我们需要选择足够小的学习率使得梯度较大的维度不发散，但这样会导致梯度较小的维度收敛缓慢。动量依赖指数加权移动平均来使得自变量的更新方向更加一致来降低发散可能。这一节我们介绍Adagrad算法，它根据每个维度的数值大小来自动调整学习率，从而避免统一的学习率难以适应所有维度的问题。


## Adagrad算法

Adagrad的算法对每个维度维护其所有时间步里梯度的平法累加。在算法开始前我们定义累加变量$\boldsymbol{s}$，其元素个数等于自变量的个数，并将其中每个元素初始化为0。在每次迭代中，假设小批量随机梯度为$\boldsymbol{g}$，我们将该梯度按元素平方后累加到变量$\boldsymbol{s}$里：

$$\boldsymbol{s} \leftarrow \boldsymbol{s} + \boldsymbol{g} \odot \boldsymbol{g}, $$

这里$\odot$是按元素相乘（请参见[“数学基础”](../chapter_appendix/math.md)一节）。

在自变量更新前，我们将梯度中的每个元素除以累加变量中对应元素的平方根，这样使得每个元素数值在同样的尺度下，然后再乘以学习率后更新：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \frac{\eta}{\sqrt{\boldsymbol{s} + \epsilon}} \odot \boldsymbol{g},$$

这里开方、除法和乘法的运算都是按元素进行的，$\epsilon$是为了使得除数不为0来而添加的正常数，例如$10^{-7}$。

## Adagrad特性

为了更好的理解累加变量是如何将每个自变量的更新变化到同样尺度，我们来看时间步1时的更新，这时候$\boldsymbol{s} = \boldsymbol{g} \odot \boldsymbol{g}$，忽略掉$\epsilon$的话，这时候的更新是$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta\cdot\textrm{sign}(\boldsymbol{g})$，这里$\textrm{sign}$是按元素的取符号。就是说，不管梯度具体值是多少，此时的每个自变量的更新量只是$\eta$，$-\eta$或0。

从另一个角度来看，如果自变量中某个元素取值总是另外一个元素的数倍，例如$x_2=20x_1$，那么其梯度也是20被的关系，那么在Adagrad里这两个元素的更新量总是一样，而不是20倍的关系。

此外，由于累加变量里我们总是累加，所以其会变得越来越大，等效于我们一直减低学习率。例如如果每个时间步的梯度都是常数$c$，那么时间步$t$的学习率就是$\frac{\eta}{c\sqrt{t}}$，其以平方根的速度依次递减。



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

我们先实现一个简单的针对二维目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2$的Adagrad来查看其自变量更新轨迹。

```{.python .input}
f = lambda x1, x2: 0.1*x1**2 + 2*x2**2
f_grad = lambda x1, x2: (0.2*x1, 2*x2)

def adagrad(eta):    
    x1, x2 = -5, -2
    sx1, sx2 = 0, 0
    eps = 1e-7
    res = [(x1, x2)]
    for i in range(15):
        gx1, gx2 = f_grad(x1, x2)
        sx1 += gx1 ** 2
        sx2 += gx2 ** 2        
        x1 -= eta / math.sqrt(sx1 + eps) * gx1
        x2 -= eta / math.sqrt(sx2 + eps) * gx2
        res.append((x1, x2))
    return res

def show(res):
    x1, x2 = zip(*res)
    gb.set_figsize((3.5, 2.5))
    gb.plt.plot(x1, x2, '-o')
    
    x1 = np.arange(-5.0, 1.0, .1)
    x2 = np.arange(min(-3.0, min(x2)-1), max(3.0, max(x2)+1), .1)
    x1, x2 = np.meshgrid(x1, x2)
    gb.plt.contour(x1, x2, f(x1, x2), colors='g')
    
    gb.plt.xlabel('x1')
    gb.plt.ylabel('x2')
    
show(adagrad(.9))
```

可以看到使用$\eta=0.9$，Adagrad的更新轨迹非常平滑。但由于其自有的降低学习率特性，我们看到在后期收敛比较缓慢。这个特性同样也使得我们可以在Adagrad中使用更大的学习率。

```{.python .input}
show(adagrad(2))
```

接下来我们以之前介绍过的线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。

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

接下来基于NDArray来实现Adagrad。

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
    print('w:', w, '\nb:', b, '\n')
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
