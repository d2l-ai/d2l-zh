# 动量法

（小批量）随机梯度修改了梯度下降中计算梯度的方式，它使用更少的样本来计算梯度。这一节我们将介绍动量法，它改变了如何使用梯度来更新自变量。在[“梯度下降和随机梯度下降”](./gd-sgd.md)一节中我们介绍了目标函数有关自变量的梯度代表了目标函数在当前点下降最快的方向。在每次迭代中，梯度下降算法根据自变量当前所在位置，沿着当前位置的梯度更新自变量。因此梯度下降也叫做最陡下降（steepest descent）。但自变量的迭代方向仅仅取决于自变量当前位置，这可能会带来一些问题。

## 梯度下降的问题

让我们考虑一个输入和输出分别为二维向量$\boldsymbol{x} = [x_1, x_2]^\top$和标量的目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2$。跟[“梯度下降和随机梯度下降”](./gd-sgd.md)一节中不同的在于我们将$x_1^2$系数从$1$减小到了$0.1$。然后观察梯度下降优化该目标函数的迭代过程。先导入实验所需的包或模块。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import nd
```

下面实现基于这个目标函数的梯度下降，并演示使用学习率为$0.4$时自变量的迭代轨迹。

```{.python .input  n=3}
eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

gb.show_trace_2d(f_2d, gb.train_2d(gd_2d))
```

可以看到，同一位置上，目标函数在竖直方向（$x_2$轴方向）比在水平方向（$x_1$轴方向）的斜率的绝对值更大。因此，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。因此，我们需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。然而，这造成了图中自变量在水平方向上朝最优解移动较慢。

下面我们试着将学习率调的稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散。

```{.python .input  n=4}
eta = 0.6
gb.show_trace_2d(f_2d, gb.train_2d(gd_2d))
```

## 动量法

动量法的提出是为了应对梯度下降的上述问题。在时间步$0$，动量法创建速度变量$\boldsymbol{v}_0\in\mathbb{R}^d$，并将其元素初始化成0。在时间步$t>0$，动量法对每次迭代的步骤做如下修改：

$$
\begin{aligned}
\boldsymbol{v}_t &\leftarrow \gamma \boldsymbol{v}_{t-1} + \eta_t \boldsymbol{g}_t \\
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{v}_t,
\end{aligned}
$$

其中，动量超参数$\gamma$满足$0 \leq \gamma < 1$。当$\gamma=0$时，动量法等价于小批量随机梯度下降。动量法中的学习率$\eta_t$和梯度$\boldsymbol{g_t}$已在[“小批量随机梯度下降”](minibatch-sgd.md)一节中描述。

在解释动量法的数学原理前，让我们先从实验中观察梯度下降在使用动量法后的迭代过程。

```{.python .input  n=5}
def momentum_2d(x1, x2, v1, v2):
    v1 = mom * v1 + eta * 0.2 * x1
    v2 = mom * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, mom = 0.4, 0.5
gb.show_trace_2d(f_2d, gb.train_2d(momentum_2d))
```

可以看到使用较小的学习率$\eta=0.4$和动量超参数$\gamma=0.5$时，动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解。我们还发现，使用较大的学习率$\eta=0.6$时，自变量也不再发散。

```{.python .input  n=11}
eta = 0.6
gb.show_trace_2d(f_2d, gb.train_2d(momentum_2d))
```

### 指数加权移动平均

为了从数学上理解动量法，让我们先解释指数加权移动平均（exponentially weighted moving average）。给定超参数$\gamma$且$0 \leq \gamma < 1$，当前时刻$t$的变量$y_t$是上一时刻$t-1$的变量$y_{t-1}$和当前时刻另一变量$x_t$的线性组合：

$$y_t = \gamma y_{t-1} + (1-\gamma) x_t.$$

我们可以对$y_t$展开：

$$
\begin{aligned}
y_t  &= (1-\gamma) x_t + \gamma y_{t-1}\\
         &= (1-\gamma)x_t + (1-\gamma) \cdot \gamma x_{t-1} + \gamma^2y_{t-2}\\
         &= (1-\gamma)x_t + (1-\gamma) \cdot \gamma x_{t-1} + (1-\gamma) \cdot \gamma^2x_{t-2} + \gamma^3y_{t-3}\\
         &\ldots
\end{aligned}
$$

由于

$$ \lim_{n \rightarrow \infty}  \left(1-\frac{1}{n}\right)^n = \exp(-1) \approx 0.3679,$$

令$n = 1/(1-\gamma)$，那么有 $\left(1-1/n\right)^n = \gamma^{1/(1-\gamma)}$。所以当$\gamma \rightarrow 1$时，$\gamma^{1/(1-\gamma)}=\exp(-1)$。例如$0.95^{20} = 0.358 \approx \exp(-1)$。如果把$\exp(-1)$当做一个比较小的数，我们可以在近似中忽略所有含$\gamma^{1/(1-\gamma)}$和比$\gamma^{1/(1-\gamma)}$更高阶的系数的项。例如，当$\gamma=0.95$时，

$$y_t \approx 0.05 \sum_{i=0}^{19} 0.95^i x_{t-i}.$$

因此，在实际中，我们常常将$y$看作是对最近$1/(1-\gamma)$个时刻的$x$值的加权平均。例如，当$\gamma = 0.95$时，$y$可以被看作是对最近20个时刻的$x$值的加权平均；当$\gamma = 0.9$时，$y$可以看作是对最近10个时刻的$x$值的加权平均。且离当前时刻越近的$x$值获得的权重越大（越接近1）。


### 由指数加权移动平均理解动量法

现在，我们对动量法的速度变量做变形：

$$\boldsymbol{v}_t \leftarrow \gamma \boldsymbol{v}_{t-1} + (1 - \gamma) \left(\frac{\eta_t}{1 - \gamma} \boldsymbol{g}_t\right). $$

由指数加权移动平均的形式可得，速度变量$\boldsymbol{v}_t$实际上对序列$\{\eta_{t-i}\boldsymbol{g}_{t-i} /(1-\gamma):i=0,\ldots,1/(1-\gamma)-1\}$做了指数加权移动平均。换句话说，相比于小批量随机梯度下降，动量法在每个时间步的自变量更新量近似于将前者对应的最近$1/(1-\gamma)$个时间步的更新量做了指数加权移动平均后再除以$1-\gamma$。所以动量法中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决过去各个梯度在各个方向上是否一致。在本节之前示例的优化问题中，由于所有梯度在水平方向上为正（向右）、而在竖直方向上时正（向上）时负（向下），自变量在水平方向的移动幅度逐渐增大，而在竖直方向的移动幅度逐渐减小。这样，我们就可以使用较大的学习率，从而使自变量向最优解更快移动。


## 从零开始实现

相对于随机梯度下降，动量法需要对每个自变量维护同它一样形状的状态变量$\boldsymbol{v}$，且超参数里多了动量超参数。

```{.python .input  n=13}
features, labels = gb.get_data_ch7()

def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    hp = hyperparams 
    for p, v in zip(params, states):
        v[:] = hp['mom'] * v + hp['lr'] * p.grad
        p[:] -= v
```

我们先将动量超参数`mom`设0.5，这时可以看成是使用最近2个时刻的$2\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均作为梯度的随机梯度下降，因此我们需要对应调下学习率（从上节的0.5减小到了0.02）。

```{.python .input  n=15}
gb.train_ch7(sgd_momentum, init_momentum_states(), 
             {'lr': 0.02, 'mom': 0.5}, features, labels)
```

将动量超参数`mom`增大到了0.9时，这个特殊梯度是最近10个时刻的$10\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。因此我们需要进一步调低学习率。

```{.python .input  n=8}
gb.train_ch7(sgd_momentum, init_momentum_states(), 
             {'lr': 0.004, 'mom': 0.9}, features, labels)
```

## 使用Gluon的实现

在Gluon中，只需要在随机梯度下降的训练器中通过`momentum`来指定动量超参数即可得到动量法。

```{.python .input  n=9}
gb.train_gluon_ch7('sgd', {'learning_rate': 0.02, 'momentum': 0.5},
                   features, labels)
```

## 小结

* 动量法使用了指数加权移动平均的思想，其将过去时刻的梯度做了加权平均，且权重按时间指数衰减。
* 动量法使得相邻时间步之间的自变量更新在方向更加一致。

## 练习

* 使用其他动量超参数和学习率的组合，观察实验结果。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1879)


![](../img/qr_momentum.svg)
