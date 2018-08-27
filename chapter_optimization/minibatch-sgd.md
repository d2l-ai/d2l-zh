# 小批量随机梯度下降

在每一轮参数迭代里，梯度下降使用整个训练数据集来计算梯度，通常也称为批量梯度下降（batch gradient descent），而随机梯度下降则只随机采样一个样本。深度学习中的常用算法在这两个极端之间：每次随机采样多个样本来组成一个小批量（mini-batch），然后对它计算梯度。这个算法被称为小批量随机梯度下降（mini-batch stochastic gradient descent）。

具体来说，在每一轮迭代里，小批量随机梯度下降随机均匀采样一个由训练数据样本索引所组成的小批量（mini-batch）$\mathcal{B}$。我们可以通过重复采样（sampling with replacement）或者不重复采样（sampling without replacement）得到同一个小批量中的各个样本。前者允许同一个小批量中出现重复的样本，后者则不允许如此，且更常见。对于这两者间的任一种方式，我们可以使用

$$\nabla f_\mathcal{B}(\boldsymbol{x}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\nabla f_i(\boldsymbol{x})$$

来计算当前小批量上的梯度。这里$|\mathcal{B}|$代表样本批量大小，是一个超参数。同随机梯度一样，小批量随机梯度$\nabla f_\mathcal{B}(\boldsymbol{x})$也是对梯度$\nabla f(\boldsymbol{x})$的无偏估计。给定学习率$\eta$（取正数），在每次迭代时，小批量随机梯度下降对$\boldsymbol{x}$的迭代如下：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_\mathcal{B}(\boldsymbol{x}).$$

小批量随机梯度下降中每次迭代的计算开销为$\mathcal{O}(|\mathcal{B}|)$。当批量大小为1时，该算法即随机梯度下降；当批量大小等于训练数据样本数，该算法即梯度下降。当批量较小时，每次迭代中使用的样本少，这会导致并行处理和内存使用效率变低。这使得在计算同样数目样本的情况下比使用更大批量时所花时间更多。当批量较大时，每个小批量梯度里可能含有更多的冗余信息。为了得到较好的解，批量较大时比批量较小时可能需要计算更多数目的样本，例如增大迭代周期数。

## 读取数据

首先导入本节需要的包和模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import time
import random
import numpy as np
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss

```

这一章里我们将使用一个来自NASA的测试不同[飞机机翼噪音](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)的数据集。这个数据集有1503个样本和4个特征，然后我们使用标准化对它进行预处理。

```{.python .input}
def get_data_ch7():  # 将保存在 GluonBook 中方便之后使用。
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-2]), nd.array(data[:1500,-1])  # 取整

features, labels = get_data_ch7()
features.shape
```

## 从零开始实现

小批量随机梯度下降算法在[“线性回归的从零开始实现”](../chapter_deep-learning-basics/linear-regression-scratch.md)一节中实现过。我们这里加入了一个状态`states`输入和将超参数放在一个字典里，这样本章后面介绍的其他优化算法也可以使用同样的输入。此外，我们将使用平均损失，所以这里梯度不需要除以批量大小。

```{.python .input  n=3}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

下面实现一个同样的训练的函数，它初始化一个线性回归模型，然后可以使用小批量随机梯度下降以及后续小节介绍的其它算法来训练模型。

```{.python .input  n=4}
# 将保存在 GluonBook 中方便之后使用。
def train_ch7(trainer_fn, states, hyperparams,
              features, labels, batch_size=10, num_epochs=2):
    # 初始化模型。
    net, loss = gb.linreg, gb.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()
    # 纪录训练误差。
    eval_loss = lambda : loss(net(features, w, b), labels).mean().asscalar()
    ls = [eval_loss()]
    # 读取数据。
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for epoch in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()  # 使用平均损失。
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # 模型更新。
            if (batch_i+1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图。
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

当批量大小为样本总数的1500时，优化使用的是梯度下降。梯度下降的1个迭代周期对模型参数只迭代1次。可以看到仅仅6次迭代训练损失就趋向平稳。

```{.python .input  n=13}
def train_sgd(lr, batch_size, epoch_size=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, epoch_size)

train_sgd(1, 1500, 6)
```

当批量大小为1时，优化使用的是随机梯度下降。这时候由于梯度中有大量噪音，我们一般使用较小的学习率。由于每处理一个样本会更新一次权重，一个周期里面会对权重进行1500次更新。所以可以看到训练损失下降非常迅速，而且一个周期后训练损失就很平缓。

虽然随机梯度下降和梯度下降在一个周期里面有同样的运算复杂度，因为它们都处理了1500个样本。但实际上随机梯度下降的一个周期耗时要多，这是因为有更多的参数更新，以及但样本梯度计算难以有效平行计算。

```{.python .input  n=5}
train_sgd(.005, 1)
```

当批量大小为10时，优化使用的是小批量随机梯度下降。它跟随机梯度下降一样能很快的降低训练损失，但每一个周期的计算要更加迅速。

```{.python .input  n=10}
train_sgd(.05, 10)
```

## 使用Gluon的实现

在Gluon里我们可以通过`Trainer`类来调用预实现好的优化算法。下面实现一个通用的训练函数，它通过训练器的名字`trainer_name`和超参数`trainer_hyperparams`来构建Trainer类实例。

```{.python .input  n=8}
# 将保存在 GluonBook 中方便之后使用。
def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型。
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    # 纪录训练误差。
    eval_loss = lambda : loss(net(features), labels).mean().asscalar()
    ls = [eval_loss()]
    # 读取数据。
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    # 使用 Trainer 来更新权重。
    trainer = gluon.Trainer(
        net.collect_params(), trainer_name, trainer_hyperparams)
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i+1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图。
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

以下重复上一个实验：

```{.python .input  n=14}
train_gluon_ch7('sgd', {'learning_rate': .05}, features, labels, 10)
```

## 小结

* 小批量随机梯度每次随机均匀采样一个小批量训练样本来计算梯度。
* 可以通过调整的批量大小来权衡计算效率和训练误差下降速度。


## 练习

* 试着修改批量大小和学习率，观察训练误差下降速度和计算性能。
* 通常我们会在训练中逐渐减小学习率，例如每$k$个（这是一个超参数）周期将学习率减小到1/10（另外一个超参数）。试着实现这个逻辑，并查看实际效果（Trainer类的`set_learning_rate`函数可以在迭代过程中调整学习率）。
