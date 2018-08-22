# 梯度下降和随机梯度下降

本节中，我们将介绍梯度下降（gradient descent）的工作原理，并随后引出（小批量）随机梯度下降（(mini-batch) stochastic gradient descent）这一深度学习中最常用的优化算法。


## 一维梯度下降

我们先以简单的一维梯度下降为例，解释梯度下降算法可以降低目标函数值的原因。假设连续可导的函数$f: \mathbb{R} \rightarrow \mathbb{R}$的输入和输出都是标量。给定绝对值足够小的数$\epsilon$，根据泰勒展开公式（参见[“数学基础”](../chapter_appendix/math.md)一节），我们得到以下的近似

$$f(x + \epsilon) \approx f(x) + \epsilon f'(x) .$$

这里$f'(x)$是函数$f$在$x$处的梯度。一维函数的梯度是一个标量，也称导数。

接下来我们找一个常数$\eta > 0$，使得$\left|\eta f'(x)\right|$足够小，那么可以将$\epsilon$替换为$-\eta f'(x)$得到

$$f(x - \eta f'(x)) \approx f(x) -  \eta f'(x)^2.$$

如果导数$f'(x) \neq 0$，那么$\eta f'(x)^2>0$，所以

$$f(x - \eta f'(x)) \lesssim f(x).$$

这意味着，如果我们通过以下规则来更新$x$：

$$x \leftarrow x - \eta f'(x),$$

函数$f(x)$的值可能被降低。一般来说，我们选取一个初始值$x$和常数$\eta > 0$，然后不断的通过上式来迭代$x$，直到达到停止条件，例如$f'(x)^2$的值已经足够小。

下面我们以目标函数$f(x)=x^2$为例来看一看梯度下降是如何执行的。虽然我们知道最小化$f(x)$的解为$x=0$，这里我们依然使用这个简单函数来观察$x$是如何被迭代的。首先，导入本节实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import random
```

接下来我们使用$x=10$作为初始值，设$\eta=0.2$。使用梯度下降对$x$迭代10次，可见最后$x$的值较接近最优解。

```{.python .input  n=2}
f = lambda x: x ** 2
f_grad = lambda x: 2 * x

def gd(eta):    
    x = 10    
    res = [x]
    for i in range(10):
        x = x - eta * f_grad(x)
        res.append(x)        
    return res

res = gd(0.2)
res[-1]
```

下面将绘制出$x$的迭代过程。

```{.python .input  n=3}
def plot_iterate(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    gb.set_figsize()
    gb.plt.plot(f_line, [f(x) for x in f_line])
    gb.plt.plot(res, [f(x) for x in res], '-o')
    gb.plt.xlabel('x')
    gb.plt.ylabel('f(x)')

plot_iterate(res)
```

## 学习率

上述梯度下降算法中的正数$\eta$通常叫做学习率。这是一个超参数，需要人工设定。如果使用过小的学习率，会导致$x$更新缓慢从而使得需要更多的迭代才能得到较好的解。下面展示了使用$\eta=0.05$时$x$的迭代过程。

```{.python .input  n=4}
plot_iterate(gd(0.05))
```

如果使用过大的学习率，$\left|\eta f'(x)\right|$可能会过大从而使前面提到的一阶泰勒展开公式不再成立：这时我们无法保证迭代$x$会
降低$f(x)$的值。举个例子，当我们设$\eta=1.1$时，可以看到$x$不断越过（overshoot）最优解$x=0$并逐渐发散。

```{.python .input  n=5}
plot_iterate(gd(1.1))
```

## 多维梯度下降

接下来考虑一个更广义的情况：目标函数的输入为向量，输出为标量。假设目标函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的输入是一个$d$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$。目标函数$f(\boldsymbol{x})$有关$\boldsymbol{x}$的梯度是一个由$d$个偏导数组成的向量：

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_d}\bigg]^\top.$$


为表示简洁，我们用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。梯度中每个偏导数元素$\partial f(\boldsymbol{x})/\partial x_i$代表着$f$在$\boldsymbol{x}$有关输入$x_i$的变化率。为了测量$f$沿着单位向量$\boldsymbol{u}$（即$\|\boldsymbol{u}\|=1$）方向上的变化率，在多元微积分中，我们定义$f$在$\boldsymbol{x}$上沿着$\boldsymbol{u}$方向的方向导数为

$$D_{\boldsymbol{u}} f(\boldsymbol{x}) = \lim_{h \rightarrow 0}  \frac{f(\boldsymbol{x} + h \boldsymbol{u}) - f(\boldsymbol{x})}{h}.$$

依据方向导数性质 \[1，14.6节定理三\]，该方向导数可以改写为

$$D_{\boldsymbol{u}} f(\boldsymbol{x}) = \nabla f(\boldsymbol{x}) \cdot \boldsymbol{u}.$$

方向导数$D_{\boldsymbol{u}} f(\boldsymbol{x})$给出了$f$在$\boldsymbol{x}$上沿着所有可能方向的变化率。为了最小化$f$，我们希望找到$f$能被降低最快的方向。因此，我们可以通过单位向量$\boldsymbol{u}$来最小化方向导数$D_{\boldsymbol{u}} f(\boldsymbol{x})$。

由于$D_{\boldsymbol{u}} f(\boldsymbol{x}) = \|\nabla f(\boldsymbol{x})\| \cdot \|\boldsymbol{u}\|  \cdot \text{cos} (\theta) = \|\nabla f(\boldsymbol{x})\|  \cdot \text{cos} (\theta)$，
其中$\theta$为梯度$\nabla f(\boldsymbol{x})$和单位向量$\boldsymbol{u}$之间的夹角，当$\theta = \pi$，$\text{cos}(\theta)$取得最小值-1。因此，当$\boldsymbol{u}$在梯度方向$\nabla f(\boldsymbol{x})$的相反方向时，方向导数$D_{\boldsymbol{u}} f(\boldsymbol{x})$被最小化。所以，我们可能通过下面的梯度下降算法来不断降低目标函数$f$的值：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f(\boldsymbol{x}).$$

相同地，其中$\eta$（取正数）称作学习率。

## 随机梯度下降

然而，当训练数据集很大时，梯度下降算法可能会难以使用。为了解释这个问题，考虑目标函数

$$f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\boldsymbol{x}),$$

其中$f_i(\boldsymbol{x})$是有关索引为$i$的训练数据样本的损失函数，$n$是训练数据样本数。那么在$\boldsymbol{x}$处的梯度计算为

$$\nabla f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}).$$

可以看到，梯度下降每次迭代的计算开销随着$n$线性增长。因此，当训练数据样本数很大时，梯度下降每次迭代的计算开销很高。给定学习率$\eta$（取正数），在每次迭代时，随机梯度下降算法随机均匀采样$i$并计算$\nabla f_i(\boldsymbol{x})$来迭代$\boldsymbol{x}$：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_i(\boldsymbol{x}).$$


事实上，随机梯度$\nabla f_i(\boldsymbol{x})$是对梯度$\nabla f(\boldsymbol{x})$的无偏估计：

$$\mathbb{E}_i \nabla f_i(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}) = \nabla f(\boldsymbol{x}).$$


## 小批量随机梯度下降

梯度下降中每次更新使用所有样本来计算梯度，而随机梯度下降则随机选取一个样本来计算梯度。深度学习中真正常用的是小批量随机梯度下降。它每次随机均匀采样一个由训练数据样本索引所组成的小批量（mini-batch）$\mathcal{B}$来计算梯度。我们可以通过重复采样（sampling with replacement）或者不重复采样（sampling without replacement）得到同一个小批量中的各个样本。前者允许同一个小批量中出现重复的样本，后者则不允许如此，且更常见。对于这两者间的任一种方式，我们可以使用

$$\nabla f_\mathcal{B}(\boldsymbol{x}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\nabla f_i(\boldsymbol{x})$$ 

来计算当前小批量上的梯度。这里$|\mathcal{B}|$代表样本批量大小，是一个超参数。同随机梯度一样，小批量随机梯度$\nabla f_\mathcal{B}(\boldsymbol{x})$也是对梯度$\nabla f(\boldsymbol{x})$的无偏估计。给定学习率$\eta$（取正数），在每次迭代时，小批量随机梯度下降对$\boldsymbol{x}$的迭代如下：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_\mathcal{B}(\boldsymbol{x}).$$

小批量随机梯度下降中每次迭代的计算开销为$\mathcal{O}(|\mathcal{B}|)$。当批量大小为1时，该算法即随机梯度下降；当批量大小等于训练数据样本数，该算法即梯度下降。当批量较小时，每次迭代中使用的样本少，这会导致并行处理和内存使用效率变低。这使得在计算同样数目样本的情况下比使用更大批量时所花时间更多。当批量较大时，每个小批量梯度里可能含有更多的冗余信息。为了得到较好的解，批量较大时比批量较小时可能需要计算更多数目的样本，例如增大迭代周期数。


## 实验

接下来我们构造一个数据集来实验小批量随机梯度下降。我们以之前介绍过的线性回归为例。下面直接调用`gluonbook`中的线性回归模型和平方损失函数。它们已在[“线性回归的从零开始实现”](../chapter_deep-learning-basics/linear-regression-scratch.md)一节中实现过了。

```{.python .input  n=6}
net = gb.linreg
loss = gb.squared_loss
```

设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。所需学习的模型在整个数据集上的平方损失函数即我们需要优化的目标函数。模型参数即目标函数自变量。

```{.python .input  n=7}
# 生成数据集。
def get_data_ch7():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    return num_inputs, num_examples, true_w, true_b, features, labels

num_inputs, num_examples, true_w, true_b, features, labels = get_data_ch7()

# 初始化模型参数。
def init_params_vars():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return [params]
```

小批量随机梯度下降算法同样在[“线性回归的从零开始实现”](../chapter_deep-learning-basics/linear-regression-scratch.md)一节中实现过。为了阅读方便，在这里我们重复实现一次。

```{.python .input  n=8}
def sgd(params_vars, hyperparams, batch_size):
    lr = hyperparams['lr']
    [w, b] = params_vars[0]
    for param in [w, b]:
        param[:] = param - lr * param.grad / batch_size
```

由于随机梯度的方差在迭代过程中无法减小，（小批量）随机梯度下降的学习率通常会采用自我衰减的方式。如此一来，学习率和随机梯度乘积的方差会衰减。实验中，当迭代周期（`epoch`）大于2时，（小批量）随机梯度下降的学习率在每个迭代周期开始时自乘0.1作自我衰减。而梯度下降在迭代过程中一直使用目标函数的真实梯度，无需自我衰减学习率。在迭代过程中，每当`log_interval`个样本被采样过后，模型当前的损失函数值（`loss`）被记录下并用于作图。例如，当`batch_size`和`log_interval`都为10时，每次迭代后的损失函数值都被用来作图。

```{.python .input  n=9}
def optimize(optimizer_fn, params_vars, hyperparams, features, labels, 
             decay_epoch=None, batch_size=10, log_interval=10, num_epochs=3, 
             is_adam=False):
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    w, b = params_vars[0]
    net = gb.linreg
    loss = gb.squared_loss                                                                                                 
    ls = [loss(net(features, w, b), labels).mean().asnumpy()]
    # 当优化算法为 Adam 时才会用到（后面章节会介绍），本节可以忽略。
    if is_adam:
        t = 0 
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if decay_epoch and decay_epoch and epoch > decay_epoch:
            hyperparams['lr'] *= 0.1 
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y)
            # 先对变量 l 中元素求和，得到小批量损失之和，然后求参数的梯度。
            l.backward()
            # 当优化算法为 Adam 时才会用到（后面章节会介绍），本节可以忽略。
            if is_adam:
                t += 1
                optimizer_fn(params_vars, hyperparams, batch_size, t)
            else:
                optimizer_fn(params_vars, hyperparams, batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features, w, b), labels).mean().asnumpy())
    print('w[0]=%.2f, w[1]=%.2f, b=%.2f'
          % (w[0].asscalar(), w[1].asscalar(), b.asscalar()))
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    gb.semilogy(es, ls, 'epoch', 'loss')
```

当批量大小为1时，优化使用的是随机梯度下降。在当前学习率下，损失函数值在早期快速下降后略有波动。这是由于随机梯度的方差在迭代过程中无法减小。当迭代周期大于2，学习率自我衰减后，损失函数值下降后较平稳。最终，优化所得的模型参数值`w`和`b`与它们的真实值[2, -3.4]和4.2较接近。

```{.python .input  n=10}
optimize(optimizer_fn=sgd, params_vars=init_params_vars(),
         hyperparams={'lr': 0.2}, features=features, labels=labels,
         decay_epoch=2, batch_size=1)
```

当批量大小为1000时，由于数据样本总数也是1000，优化使用的是梯度下降。梯度下降无需自我衰减学习率（`decay_epoch=None`）。最终，优化所得的模型参数值与它们的真实值较接近。需要注意的是，梯度下降的1个迭代周期对模型参数只迭代1次。而随机梯度下降的批量大小为1，它在1个迭代周期对模型参数迭代了1000次。我们观察到，1个迭代周期后，梯度下降所得的损失函数值比随机梯度下降所得的损失函数值略大。而在3个迭代周期后，梯度下降和随机梯度下降得到的损失函数值较接近。

```{.python .input  n=11}
optimize(optimizer_fn=sgd, params_vars=init_params_vars(),
         hyperparams={'lr': 0.999}, features=features, labels=labels,
         decay_epoch=None, batch_size=1000, log_interval=1000)
```

当批量大小为10时，由于数据样本总数也是1000，优化使用的是小批量随机梯度下降。最终，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=12}
optimize(optimizer_fn=sgd, params_vars=init_params_vars(),
         hyperparams={'lr': 0.2}, features=features, labels=labels,
         decay_epoch=2, batch_size=10)
```

同样是批量大小为10，我们把学习率改大。这时损失函数值不断增大，直到出现“nan”（not a number，非数）。
这是因为，过大的学习率造成了模型参数越过最优解并发散。最终学到的模型参数也是“nan”。

```{.python .input  n=13}
optimize(optimizer_fn=sgd, params_vars=init_params_vars(),
         hyperparams={'lr': 5}, features=features, labels=labels,
         decay_epoch=2, batch_size=10)
```

同样是批量大小为10，我们把学习率改小。这时我们观察到损失函数值下降较慢，直到3个迭代周期模型参数也没能接近它们的真实值。

```{.python .input  n=14}
optimize(optimizer_fn=sgd, params_vars=init_params_vars(),
         hyperparams={'lr': 0.002}, features=features, labels=labels,
         decay_epoch=2, batch_size=10)
```

## 使用Gluon的实现

在Gluon里，使用小批量随机梯度下降很方便，我们无需重新实现该算法。特别地，当批量大小等于数据集样本数时，该算法即为梯度下降；批量大小为1即为随机梯度下降。为了使学习率能够自我衰减，我们需要访问`gluon.Trainer`的`learning_rate`属性并使用`set_learning_rate`函数。

```{.python .input}
# 线性回归模型。
net = nn.Sequential()
net.add(nn.Dense(1))

def optimize_with_trainer(trainer, features, labels, net, decay_epoch=None,
                          batch_size=10, log_interval=10, num_epochs=3):
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    loss = gloss.L2Loss()
    ls = [loss(net(features), labels).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if decay_epoch and epoch > decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features), labels).mean().asnumpy())
    print('w[0]=%.2f, w[1]=%.2f, b=%.2f' 
          % (net[0].weight.data()[0][0].asscalar(),
             net[0].weight.data()[0][1].asscalar(),
             net[0].bias.data().asscalar()))
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    gb.semilogy(es, ls, 'epoch', 'loss')
```

以下使用Gluon分别实验了随机梯度下降、梯度下降和小批量随机梯度下降（批量大小为10）。这几组实验分别重现了本节中使用NDArray实现优化算法的前三组实验结果。这些结果有一定的随机性。

```{.python .input}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize_with_trainer(trainer=trainer, features=features, labels=labels, 
                      net=net, decay_epoch=2, batch_size=1)
```

```{.python .input}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.999})
optimize_with_trainer(trainer=trainer, features=features, labels=labels, 
                      net=net, decay_epoch=None, batch_size=1000,
                      log_interval=1000)
```

```{.python .input}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize_with_trainer(trainer=trainer, features=features, labels=labels, 
                      net=net, decay_epoch=2, batch_size=10)
```

本节使用的`get_data_ch7`、`optimize`和`optimize_with_trainer`函数被定义在`gluonbook`包中供后面章节调用。


## 小结

* 当训练数据较大，梯度下降每次迭代计算开销较大，因而（小批量）随机梯度下降更受青睐。
* 学习率过大过小都有问题。一个合适的学习率通常是需要通过多次实验找到的。
* 使用Gluon的`Trainer`可以方便地使用小批量随机梯度下降。
* 访问`gluon.Trainer`的`learning_rate`属性并使用`set_learning_rate`函数可以在迭代过程中调整学习率。


## 练习

* 运行本节中实验代码。比较一下随机梯度下降和梯度下降的运行时间。
* 梯度下降和随机梯度下降虽然看上去有效，但可能会有哪些问题？
* 查阅网络或书本资料，了解学习率自我衰减的其他方法。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1877)

![](../img/qr_gd-sgd.svg)


## 参考文献

[1] Stewart, J. (2010). Calculus: Early Transcendentals (7th Edition). Brooks Cole.
