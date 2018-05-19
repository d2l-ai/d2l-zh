# 正则化——使用Gluon

本章介绍如何使用``Gluon``的正则化来应对[过拟合](underfit-overfit.md)问题。

## 高维线性回归数据集

我们使用与[上一节](reg-scratch.md)相同的高维线性回归为例来引入一个过拟合问题。

```{.python .input  n=1}
%matplotlib inline
import sys
sys.path.append('..')
import gluonbook as gb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
```

```{.python .input  n=2}
n_train = 20
n_test = 100
num_inputs = 200
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

features = nd.random.normal(shape=(n_train+n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

num_epochs = 10
learning_rate = 0.003
batch_size = 1
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)
loss = gloss.L2Loss()
```

## 定义训练和测试

跟前一样定义训练模块。你也许发现了主要区别，`Trainer`有一个新参数`wd`。我们通过优化算法的``wd``参数 (weight decay)实现对模型的正则化。这相当于$L_2$范数正则化。

```{.python .input  n=3}
gb.set_fig_size(mpl)

def fit_and_plot(weight_decay):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    # 注意到这里 'wd'
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {
        'learning_rate': learning_rate})
    train_ls = []
    test_ls = []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.semilogy(range(1, num_epochs+1), train_ls)
    plt.semilogy(range(1, num_epochs+1), test_ls)
    plt.legend(['train','test'])
    plt.show()
    return 'w[:10]:', net[0].weight.data()[:,:10], 'b:', net[0].bias.data()
```

### 训练模型并观察过拟合

接下来我们训练并测试我们的高维线性回归模型。

```{.python .input  n=4}
fit_and_plot(0)
```

即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。

## 使用``Gluon``的正则化

下面我们重新初始化模型参数并在`Trainer`里设置一个`wd`参数。

```{.python .input  n=5}
fit_and_plot(5)
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。
但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 小结

* 使用``Gluon``的`weight decay`参数可以很容易地使用正则化来应对过拟合问题。

## 练习

* 如何从字面正确理解`weight decay`的含义？它为何相当于$L_2$范式正则化？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/985)

![](../img/qr_reg-gluon.svg)
