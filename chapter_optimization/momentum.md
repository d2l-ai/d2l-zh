# 动量法


我们已经介绍了梯度下降。每次迭代时，该算法根据自变量当前所在位置，沿着目标函数下降最快的方向更新自变量。因此，梯度下降有时也叫做最陡下降（steepest descent）。目标函数有关自变量的梯度代表了目标函数下降最快的方向。


## 梯度下降的问题

给定目标函数，在梯度下降中，自变量的迭代方向仅仅取决于自变量当前位置。这可能会带来一些问题。

考虑一个输入和输出分别为二维向量$\boldsymbol{x} = [x_1, x_2]^\top$和标量的目标函数$f: \mathbb{R}^2 \rightarrow \mathbb{R}$。图7.2展示了该函数的等高线示意图。

![目标函数$f$的等高线图和自变量$[x_1, x_2]$在梯度下降中的迭代。每条等高线（椭圆实线）代表所有函数值相同的自变量的坐标。实心圆代表自变量初始坐标。每个箭头头部代表自变量在每次迭代后的坐标。](../img/gd-move.svg)

由于目标函数在竖直方向（$x_2$轴方向）比在水平方向（$x_1$轴方向）更弯曲，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。因此，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。然而，这造成了图7.2中自变量向最优解移动较慢。


## 动量法

动量法的提出是为了应对梯度下降的上述问题。广义上，以小批量随机梯度下降为例（当批量大小等于训练集样本数时，该算法即为梯度下降；批量大小为1时即为随机梯度下降），我们对小批量随机梯度算法在每次迭代的步骤做如下修改：


$$
\begin{aligned}
\boldsymbol{v} &\leftarrow \gamma \boldsymbol{v} + \eta \nabla f_\mathcal{B}(\boldsymbol{x}),\\
\boldsymbol{x} &\leftarrow \boldsymbol{x} - \boldsymbol{v}.
\end{aligned}
$$


其中$\boldsymbol{v}$是速度变量，动量超参数$\gamma$满足$0 \leq \gamma \leq 1$。动量法中的学习率$\eta$和有关小批量$\mathcal{B}$的随机梯度$\nabla f_\mathcal{B}(\boldsymbol{x})$已在[“梯度下降和随机梯度下降”](gd-sgd.md)一节中描述。


### 指数加权移动平均

为了更清晰地理解动量法，让我们先解释指数加权移动平均（exponentially weighted moving average）。给定超参数$\gamma$且$0 \leq \gamma < 1$，当前时刻$t$的变量$y^{(t)}$是上一时刻$t-1$的变量$y^{(t-1)}$和当前时刻另一变量$x^{(t)}$的线性组合：

$$y^{(t)} = \gamma y^{(t-1)} + (1-\gamma) x^{(t)}.$$

我们可以对$y^{(t)}$展开：

$$
\begin{aligned}
y^{(t)}  &= (1-\gamma) x^{(t)} + \gamma y^{(t-1)}\\
         &= (1-\gamma)x^{(t)} + (1-\gamma) \cdot \gamma x^{(t-1)} + \gamma^2y^{(t-2)}\\
         &= (1-\gamma)x^{(t)} + (1-\gamma) \cdot \gamma x^{(t-1)} + (1-\gamma) \cdot \gamma^2x^{(t-2)} + \gamma^3y^{(t-3)}\\
         &\ldots
\end{aligned}
$$

由于

$$ \lim_{n \rightarrow \infty}  (1-\frac{1}{n})^n = \exp(-1) \approx 0.3679,$$

我们可以将$\gamma^{1/(1-\gamma)}$近似为$\exp(-1)$。例如$0.95^{20} \approx \exp(-1)$。如果把$\exp(-1)$当做一个比较小的数，我们可以在近似中忽略所有含$\gamma^{1/(1-\gamma)}$和比$\gamma^{1/(1-\gamma)}$更高阶的系数的项。例如，当$\gamma=0.95$时，

$$y^{(t)} \approx 0.05 \sum_{i=0}^{19} 0.95^i x^{(t-i)}.$$

因此，在实际中，我们常常将$y$看作是对最近$1/(1-\gamma)$个时刻的$x$值的加权平均。例如，当$\gamma = 0.95$时，$y$可以被看作是对最近20个时刻的$x$值的加权平均；当$\gamma = 0.9$时，$y$可以看作是对最近10个时刻的$x$值的加权平均：离当前时刻越近的$x$值获得的权重越大。


### 由指数加权移动平均理解动量法

现在，我们对动量法的速度变量做变形：

$$\boldsymbol{v} \leftarrow \gamma \boldsymbol{v} + (1 - \gamma) \frac{\eta \nabla f_\mathcal{B}(\boldsymbol{x})}{1 - \gamma}. $$

由指数加权移动平均的形式可得，速度变量$\boldsymbol{v}$实际上对$(\eta\nabla f_\mathcal{B}(\boldsymbol{x})) /(1-\gamma)$做了指数加权移动平均。给定动量超参数$\gamma$和学习率$\eta$，含动量法的小批量随机梯度下降可被看作使用了特殊梯度来迭代目标函数的自变量。这个特殊梯度是最近$1/(1-\gamma)$个时刻的$\nabla f_\mathcal{B}(\boldsymbol{x})/(1-\gamma)$的加权平均。


给定目标函数，在动量法的每次迭代中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决过去各个梯度在各个方向上是否一致。图7.3展示了使用动量法的梯度下降迭代图7.2中目标函数自变量的情景。我们将每个梯度代表的箭头方向在水平方向和竖直方向做分解。由于所有梯度的水平方向为正（向右）、在竖直上时正（向上）时负（向下），自变量在水平方向移动幅度逐渐增大，而在竖直方向移动幅度逐渐减小。这样，我们就可以使用较大的学习率，从而使自变量向最优解更快移动。

![目标函数$f$的等高线图和自变量$[x_1, x_2]$在使用动量法的梯度下降中的迭代。每条等高线（椭圆实线）代表所有函数值相同的自变量的坐标。实心圆代表自变量初始坐标。每个箭头头部代表自变量在每次迭代后的坐标。](../img/momentum-move.svg)


## 动量法的实现

动量法的实现也很简单。我们在小批量随机梯度下降的基础上添加速度变量。

```{.python .input  n=1}
def sgd_momentum(params, vs, lr, mom, batch_size):
    for param, v in zip(params, vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] -= v
```

## 实验

首先，导入本节中实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, nd
import numpy as np
```

实验中，我们以之前介绍过的线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。

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
    vs = []
    for param in params:
        param.attach_grad()
        # 把速度项初始化为和参数形状相同的零张量。
        vs.append(param.zeros_like())
    return params, vs
```

优化函数`optimize`与[“梯度下降和随机梯度下降”](gd-sgd.md)一节中的类似。

```{.python .input  n=3}
net = gb.linreg
loss = gb.squared_loss

def optimize(batch_size, lr, mom, num_epochs, log_interval):
    [w, b], vs = init_params()
    ls = [loss(net(features, w, b), labels).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        # 学习率自我衰减。
        if epoch > 2:
            lr *= 0.1
        for batch_i, (X, y) in enumerate(
            gb.data_iter(batch_size, features, labels)):
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()
            sgd_momentum([w, b], vs, lr, mom, batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features, w, b), labels).mean().asnumpy())
    print('w:', w, '\nb:', b, '\n')
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    gb.semilogy(es, ls, 'epoch', 'loss')
```

我们先将动量超参数$\gamma$（`mom`）设0.99。此时，小梯度随机梯度下降可被看作使用了特殊梯度：这个特殊梯度是最近100个时刻的$100\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后上升。这很可能是由于特殊梯度中较大的系数100造成的。

```{.python .input  n=4}
optimize(batch_size=10, lr=0.2, mom=0.99, num_epochs=3, log_interval=10)
```

假设学习率不变，为了降低上述特殊梯度中的系数，我们将动量超参数$\gamma$（`mom`）设0.9。此时，上述特殊梯度变成最近10个时刻的$10\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后下降。

```{.python .input}
optimize(batch_size=10, lr=0.2, mom=0.9, num_epochs=3, log_interval=10)
```

继续保持学习率不变，我们将动量超参数$\gamma$（`mom`）设0.5。此时，小梯度随机梯度下降可被看作使用了新的特殊梯度：这个特殊梯度是最近2个时刻的$2\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后下降，且下降曲线较平滑。最终，优化所得的模型参数值与它们的真实值较接近。

```{.python .input}
optimize(batch_size=10, lr=0.2, mom=0.5, num_epochs=3, log_interval=10)
```

## 小结

* 动量法使用了指数加权移动平均的思想。


## 练习

* 使用其他动量超参数和学习率的组合，观察实验结果。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1879)


![](../img/qr_momentum.svg)
