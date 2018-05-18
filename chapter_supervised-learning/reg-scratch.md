# 正则化——从零开始

本章从0开始介绍如何的正则化来应对[过拟合](underfit-overfit.md)问题。

## 高维线性回归

我们使用高维线性回归为例来引入一个过拟合问题。


具体来说我们使用如下的线性函数来生成每一个数据样本

$$y = 0.05 + \sum_{i = 1}^p 0.01x_i +  \text{noise}$$

这里噪音服从均值0和标准差为0.01的正态分布。

需要注意的是，我们用以上相同的数据生成函数来生成训练数据集和测试数据集。为了观察过拟合，我们特意把训练数据样本数设低，例如$n=20$，同时把维度升高，例如$p=200$。

```{.python .input}
%matplotlib inline
import sys
sys.path.append('..')
import gluonbook as gb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mxnet import autograd, gluon, nd

gb.set_fig_size(mpl)
```

```{.python .input  n=1}
n_train = 20
n_test = 100
num_inputs = 200
```

## 生成数据集


这里定义模型真实参数。

```{.python .input  n=2}
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05
```

我们接着生成训练和测试数据集。

```{.python .input  n=3}
features = nd.random.normal(shape=(n_train+n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

features_train, features_test = features[:n_train, :], features[n_train:, :]
labels_train, labels_test = labels[:n_train], labels[n_train:]
```

当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回`batch_size`个随机的样本和对应的目标。我们通过python的`yield`来构造一个迭代器。

```{.python .input  n=4}
batch_size = 1
```

## 初始化模型参数

下面我们随机初始化模型参数。之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。

```{.python .input  n=5}
def init_params():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params
```

## $L_2$范数正则化

这里我们引入$L_2$范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，我们在训练时其实在最小化

$$\text{loss} + \lambda \sum_{p \in \textrm{params}}\|p\|_2^2。$$

直观上，$L_2$范数正则化试图惩罚较大绝对值的参数值。下面我们定义L2正则化。注意有些时候大家对偏移加罚，有时候不加罚。通常结果上两者区别不大。这里我们演示对偏移也加罚的情况：

```{.python .input  n=6}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
num_epochs = 10
lr = 0.003
```

## 定义训练和测试

下面我们定义剩下的所需要的函数。这个跟之前的教程大致一样，主要是区别在于计算`loss`的时候我们加上了L2正则化，以及我们将训练和测试损失都画了出来。

```{.python .input  n=7}
net = gb.linreg
loss = gb.squared_loss

def fit_and_plot(lambd):
    w, b = params = init_params()
    train_ls = []
    test_ls = []
    for _ in range(num_epochs):        
        for X, y in gb.data_iter(batch_size, n_train, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            gb.sgd(params, lr, batch_size)
        train_ls.append(loss(net(features_train, w, b),
                             labels_train).mean().asscalar())
        test_ls.append(loss(net(features_test, w, b),
                            labels_test).mean().asscalar())
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.semilogy(range(1, num_epochs+1), train_ls)
    plt.semilogy(range(1, num_epochs+1), test_ls)
    plt.legend(['train','test'])
    plt.show()
    return 'w[:10]:', w[:10].T, 'b:', b
```

## 观察过拟合

接下来我们训练并测试我们的高维线性回归模型。注意这时我们并未使用正则化。

```{.python .input  n=8}
fit_and_plot(0)
```

即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。


## 使用正则化

下面我们重新初始化模型参数并设置一个正则化参数。

```{.python .input  n=9}
fit_and_plot(5)
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 小结

* 我们可以使用正则化来应对过拟合问题。

## 练习

* 除了正则化、增大训练量、以及使用合适的模型，你觉得还有哪些办法可以应对过拟合现象？
* 如果你了解贝叶斯统计，你觉得$L_2$范数正则化对应贝叶斯统计里的哪个重要概念？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/984)

![](../img/qr_reg-scratch.svg)
