# 梯度下降和随机梯度下降

本节中，我们将介绍梯度下降（gradient descent）的工作原理，并随后引出（小批量）随机梯度下降（(mini-batch) stochastic gradient descent）这一深度学习中最常用的优化算法。


## 一维梯度下降

我们先以简单的一维梯度下降为例，解释梯度下降算法可以降低目标函数值的原因。假设函数$f: \mathbb{R} \rightarrow \mathbb{R}$的输入和输出都是标量。给定绝对值足够小的数$\epsilon$，根据泰勒展开公式（参见[“数学基础”](../chapter_appendix/math.md)一节），我们得到以下的近似

$$f(x + \epsilon) \approx f(x) + \epsilon f'(x) .$$

这里$f'(x)$是函数$f$在$x$处的梯度。一维函数的梯度是一个标量，也称导数。

接下来我们找一个正常数$\eta$，使得$|\eta f'(x)|$足够小，那么可以将$\epsilon$替换为$-\eta f'(x)$得到

$$f(x - \eta f'(x)) \approx f(x) -  \eta f'(x)^2.$$

如果导数$f'(x) \neq 0$，那么$\eta f'(x)^2>0$，所以

$$f(x - \eta f'(x)) \lesssim f(x).$$

这样意味这如果我们通过下面规则来更新$x$，

$$x \leftarrow x - \eta f'(x),$$

我们可以降低$f(x)$的值。一般来说，我们选取一个初始值$x$和正常数$\eta$，然后不断的通过上式来更新$x$，直到达到停止条件，例如$f'(x)^2$的值已经足够小。

下面我们以$f(x)=x^2$为例来看梯度下降是如何执行的。虽然我们知道最小化$f(x)$的解为$x=0$，这里我们使用这个简单函数来观察$x$是如何被更新的。首先导入本小节需要的包。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, nd
import numpy as np
import random
```

接下来我们使用$x=10$作为初始值，固定$\eta=0.2$，并对$x$更新10次，可以看到最后$x$的值已经很接近了最优解。

```{.python .input}
f = lambda x: x**2
f_grad = lambda x: 2*x
    
def gd(eta):    
    x = 10    
    res = []
    for i in range(10):
        x = x - eta * f_grad(x)
        res.append(x)        
    return res

res = gd(0.2)
res[-1]
```

下面我们画出$x$是如何被更新的。

```{.python .input}
def show(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, .1)
    gb.set_figsize((3.5, 2.5))
    gb.plt.plot(f_line, [f(x) for x in f_line])
    gb.plt.plot(res, [f(x) for x in res], '-o')
    gb.plt.xlabel('x')
    gb.plt.ylabel('f(x)')
    
show(res)
```

## 学习率

上述梯度下降算法中的正数$\eta$通常叫做学习率。这是一个超参数，需要人工设定。如果使用过小的学习率，会导致$x$更新缓慢从而使得需要更多的迭代才能得到需要的解。例如下面展示了使用$\eta=0.05$的情况。

```{.python .input}
show(gd(.05))
```

如果使用过大的学习率，可能会使得$|\eta f'(x)|$过大从而前面提到的一阶泰勒展开公式不再成立，这时我们不能保证每次更新还能降低$f(x)$的值。例如如果使用$\eta=0.9$，可以看到$x$不断的越过（overshoot）最优解$x=0$。

```{.python .input}
show(gd(.9))
```

而使用$\eta=1.1$则使得更新不再收敛。

```{.python .input}
show(gd(1.1))
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

在深度学习中，我们一般不知道$f$的具体定义，而是多个训练数据的平均来近似它。例如

$$f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\boldsymbol{x}),$$

其中$f_i(\boldsymbol{x})$是有关索引为$i$的训练数据样本的目标函数，$n$是训练数据样本数。那么在$\boldsymbol{x}$处的导数计算为

$$\nabla f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}).$$

可以看到，梯度下降每次迭代的计算开销随着$n$线性增长。因此，当训练数据样本数很大时，梯度下降每次迭代的计算开销很高。一个降低计算复杂度的方法是每次使用更少的样本来估计$f$。如果我们假设训练数据样本是随机采样而来，那么我们这里是通过$n$个样本的平均来近似$f$在整个样本空间上的期望，记为$\mathbb{E} f$。通常我们希望使用的更大的训练数据集，因为增加$n$可以使得近似方差变小。但即使使用很小的$n$，只要样本是随机采样而来，我们仍然可以得到无偏的估计。


随机梯度下降正是使用了上述观察。每次更新中，我们随机均匀采样样本$i$，并用$f_i(\boldsymbol{x})$来近似目标函数。如果训练集包含了样本空间内所有样本，那么我们有$\mathbb{E}_i(\boldsymbol{x}) = \mathbb{E} f(\boldsymbol{x})$，即这是一个无偏估计。当然，实际中我们一般只有有限个样本，即使我们每次是从$n$个样本中随机选取，我们仍然有$\mathbb{E}_i f_i(\boldsymbol{x}) = f(\boldsymbol{x})$，这是对$f$的一个无偏估计。同样知道梯度的估计也是无偏的，即$\mathbb{E}_i \nabla f_i(\boldsymbol{x}) = \nabla f(\boldsymbol{x})$。这时候，随机梯度下降中的更新为：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_i(\boldsymbol{x}).$$


## 小批量随机梯度下降

梯度下降中每次更新使用所有样本来计算梯度，而随机梯度下降则随机选取一个样本来计算梯度。深度学习中真正常用的是小批量（mini-batch）随机梯度下降，其每次随机均匀采样一个由训练数据样本索引所组成的小批量（mini-batch）$\mathcal{B}$来计算梯度。我们可以通过重复采样（sampling with replacement）或者不重复采样（sampling without replacement）得到同一个小批量中的各个样本。前者允许同一个小批量中出现重复的样本，后者则不允许如此，且更常见。对于这两者间的任一种方式，我们可以使用

$$\nabla f_\mathcal{B}(\boldsymbol{x}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\nabla f_i(\boldsymbol{x})$$ 

来计算当前小批量上的梯度。这里$|\mathcal{B}|$代表样本批量大小，是一个超参数。容易看出小批量随机梯度$\nabla f_\mathcal{B}(\boldsymbol{x})$也是对梯度$\nabla f(\boldsymbol{x})$的无偏估计。对$\boldsymbol{x}$的更新如下：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_\mathcal{B}(\boldsymbol{x}).$$

小批量随机梯度下降中每次迭代的计算开销为$\mathcal{O}(|\mathcal{B}|)$。当批量大小为1时，该算法即随机梯度下降；当批量大小等于训练数据样本数，该算法即梯度下降。批量大小的选取通常是计算效率和收敛快慢的权衡。当批量较小时，每次迭代中使用的样本少，这会导致并行处理和内存使用效率变低。从而使得在计算同样多样本的情况下比使用更大批量大小时所花时间更多。但另一方面边，当批量大小较大时，每次梯度里含有更多的冗余信息从而使得更新效率变低，从而收敛（即处理的样本数对比目标函数值）变慢。



## 小批量随机梯度下降的收敛实验

接下来我们构造一个数据集来观察批量大小和学习率对收敛的影响。我们以之前介绍过的线性回归为例。我们直接调用`gluonbook`中的线性回归模型和平方损失函数。它们已在[“线性回归的从零开始实现”](../chapter_deep-learning-basics/linear-regression-scratch.md)一节中实现过了。

```{.python .input}
net = gb.linreg
loss = gb.squared_loss
```

设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。

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
    for param in params:
        param.attach_grad()
    return params
```

模型参数更新同样在[“线性回归的从零开始实现”](../chapter_deep-learning-basics/linear-regression-scratch.md)一节中介绍过，但这里为了阅读方便我们重复实现一次。

```{.python .input}
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

由于随机梯度的方差在迭代过程中无法减小，（小批量）随机梯度下降的学习率通常会采用自我衰减的方式。如此一来，学习率和随机梯度乘积的方差会衰减来帮助收敛。在下面定义的优化函数`optimize`中，每`decay_epoch`次迭代周期将学习率减少10倍。在迭代过程中，每当`log_interval`个样本被采样过后，模型当前的损失函数值（`loss`）被记录下并用于作图。例如，当`batch_size`和`log_interval`都为10时，每次迭代后的损失函数值都被用来作图。

```{.python .input  n=3}
def optimize(batch_size, lr, num_epochs, log_interval, decay_epoch):
    w, b = init_params()
    ls = [loss(net(features, w, b), labels).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if decay_epoch and epoch > decay_epoch:
            lr *= 0.1
        for batch_i, (X, y) in enumerate(
            gb.data_iter(batch_size, features, labels)):
            with autograd.record():
                l = loss(net(X, w, b), y)
            # 先对 l 中元素求和，得到小批量损失之和，然后求参数的梯度。
            l.backward()
            sgd([w, b], lr, batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features, w, b), labels).mean().asnumpy())
    print('w[0]=%f, w[1]=%f, b=%f'%(w[0].asscalar(), 
                                    w[1].asscalar(), b.asscalar()))
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    gb.semilogy(es, ls, 'epoch', 'loss')
```

当批量大小为1时，优化使用的是随机梯度下降。在当前学习率下，损失函数值在早期快速下降后略有波动。这是由于随机梯度的方差在迭代过程中无法减小。当迭代周期大于2，学习率自我衰减后，损失函数值下降后较平稳。最终，优化所得的模型参数值`w`和`b`与它们的真实值[2, -3.4]和4.2较接近。

```{.python .input  n=4}
optimize(batch_size=1, lr=0.2, num_epochs=3, decay_epoch=2, log_interval=10)
```

当批量大小为1000时，由于数据样本总数也是1000，优化使用的是梯度下降。梯度下降无需自我衰减学习率（`decay_epoch=None`）。最终，优化所得的模型参数值与它们的真实值较接近。需要注意的是，梯度下降的1个迭代周期对模型参数只迭代1次。而随机梯度下降的批量大小为1，它在1个迭代周期对模型参数迭代了1000次。我们观察到，1个迭代周期后，梯度下降所得的损失函数值比随机梯度下降所得的损失函数值略大。而在3个迭代周期后，这两个算法所得的损失函数值很接近。

```{.python .input  n=5}
optimize(batch_size=1000, lr=0.999, num_epochs=3, decay_epoch=None, 
         log_interval=1000)
```

当批量大小为10时，由于数据样本总数也是1000，优化使用的是小批量随机梯度下降。最终，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=6}
optimize(batch_size=10, lr=0.2, num_epochs=3, decay_epoch=2, log_interval=10)
```

同样是批量大小为10，我们把学习率改大。这时损失函数值不断增大，直到出现“nan”（not a number，非数）。
这是因为，过大的学习率造成了模型参数越过最优解并发散。最终学到的模型参数也是“nan”。

```{.python .input  n=7}
optimize(batch_size=10, lr=5, num_epochs=3, decay_epoch=2, log_interval=10)
```

同样是批量大小为10，我们把学习率改小。这时我们观察到损失函数值下降较慢，直到3个迭代周期模型参数也没能接近它们的真实值。

```{.python .input  n=8}
optimize(batch_size=10, lr=0.002, num_epochs=3, decay_epoch=2,
         log_interval=10)
```

## 小结

* 当训练数据较大，梯度下降每次迭代计算开销较大，因而（小批量）随机梯度下降更受青睐。
* 学习率过大过小都有问题。一个合适的学习率通常是需要通过多次实验找到的。


## 练习

* 运行本节中实验代码。比较一下随机梯度下降和梯度下降的运行时间。
* 梯度下降和随机梯度下降虽然看上去有效，但可能会有哪些问题？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1877)


![](../img/qr_gd-sgd.svg)


## 参考文献

[1] Stewart, J. (2010). Calculus: Early Transcendentals (7th Edition). Brooks Cole.
