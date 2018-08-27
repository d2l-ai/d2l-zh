# 动量法


我们已经介绍了梯度下降。每次迭代时，该算法根据自变量当前所在位置，沿着目标函数下降最快的方向更新自变量。因此，梯度下降有时也叫做最陡下降（steepest descent）。目标函数有关自变量的梯度代表了目标函数下降最快的方向。


## 梯度下降的问题

给定目标函数，在梯度下降中，自变量的迭代方向仅仅取决于自变量当前位置。这可能会带来一些问题。考虑一个输入和输出分别为二维向量$\boldsymbol{x} = [x_1, x_2]^\top$和标量的目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2$。为了观察梯度下降优化该目标函数的迭代过程，下面导入实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import numpy as np
```

接下来实现梯度下降和作图函数。和上一节中的不同，这里的目标函数的输入是二维的。因此，在作图中我们使用等高线示意二维输入下的目标函数值。

```{.python .input  n=2}
f = lambda x1, x2: 0.1 * x1 ** 2 + 2 * x2 ** 2
f_grad = lambda x1, x2: (0.2 * x1, 2 * x2)

def gd(eta):    
    x1, x2 = -5, -2
    res = [(x1, x2)]
    for i in range(15):
        gx1, gx2 = f_grad(x1, x2)
        x1 = x1 - eta * gx1
        x2 = x2 - eta * gx2
        res.append((x1, x2))
    return res

def plot_iterate(res):
    x1, x2 = zip(*res)
    gb.set_figsize((3.5, 2.5))
    gb.plt.plot(x1, x2, '-o')
    x1 = np.arange(-5.0, 1.0, 0.1)
    x2 = np.arange(min(-3.0, min(x2) - 1), max(3.0, max(x2) + 1), 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    gb.plt.contour(x1, x2, f(x1, x2), colors='g')
    gb.plt.xlabel('x1')
    gb.plt.ylabel('x2')
```

下面演示使用学习率$\eta=0.9$时目标函数自变量的迭代轨迹。

```{.python .input  n=3}
plot_iterate(gd(0.9))
```

同一位置上，目标函数在竖直方向（$x_2$轴方向）比在水平方向（$x_1$轴方向）的斜率的绝对值更大。因此，给定学习率，梯度下降迭代自变量时会使自变量在竖直方向比在水平方向移动幅度更大。因此，我们需要一个较小的学习率（这里使用了0.9）从而避免自变量在竖直方向上越过目标函数最优解。然而，这造成了图中自变量向最优解移动较慢。

我们试着将学习率调的稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散。

```{.python .input  n=4}
plot_iterate(gd(1.1))
```

## 动量法

动量法的提出是为了应对梯度下降的上述问题。以小批量随机梯度下降为例，动量法对每次迭代的步骤做如下修改：

$$
\begin{aligned}
\boldsymbol{v} &\leftarrow \gamma \boldsymbol{v} + \eta \nabla f_\mathcal{B}(\boldsymbol{x}),\\
\boldsymbol{x} &\leftarrow \boldsymbol{x} - \boldsymbol{v}.
\end{aligned}
$$

其中$\boldsymbol{v}$是速度变量，动量超参数$\gamma$满足$0 \leq \gamma \leq 1$。动量法中的学习率$\eta$和有关小批量$\mathcal{B}$的随机梯度$\nabla f_\mathcal{B}(\boldsymbol{x})$已在[“梯度下降和随机梯度下降”](gd-sgd.md)一节中描述。

在解释动量法的原理前，让我们先从实验中观察梯度下降在使用动量法后的迭代过程。与本节上一个实验相比，这里目标函数和自变量的初始位置均保持不变。

```{.python .input  n=5}
def momentum(eta, mom):
    x1, x2 = -5, -2
    res = [(x1, x2)]
    v_x1, v_x2 = 0, 0
    for i in range(15):
        gx1, gx2 = f_grad(x1, x2)
        v_x1 = mom * v_x1 + eta * gx1
        v_x2 = mom * v_x2 + eta * gx2        
        x1 = x1 - v_x1
        x2 = x2 - v_x2
        res.append((x1, x2))
    return res

plot_iterate(momentum(0.9, 0.2))
```

可以看到使用学习率$\eta=0.9$和动量超参数$\gamma=0.2$时，动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解。

我们还发现，使用更大的学习率$\eta=1.1$时，自变量也不再发散。由于能够使用更大的学习率，自变量可以在水平方向上以更快的速度逼近最优解。

```{.python .input  n=6}
plot_iterate(momentum(1.1, 0.2))
```

### 指数加权移动平均

为了从数学上理解动量法，让我们先解释指数加权移动平均（exponentially weighted moving average）。给定超参数$\gamma$且$0 \leq \gamma < 1$，当前时刻$t$的变量$y^{(t)}$是上一时刻$t-1$的变量$y^{(t-1)}$和当前时刻另一变量$x^{(t)}$的线性组合：

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


给定目标函数，在动量法的每次迭代中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决过去各个梯度在各个方向上是否一致。在本节之前示例的优化问题中，由于所有梯度在水平方向上为正（向右）、而在竖直方向上时正（向上）时负（向下），自变量在水平方向的移动幅度逐渐增大，而在竖直方向的移动幅度逐渐减小。这样，我们就可以使用较大的学习率，从而使自变量向最优解更快移动。


## 实验

实验中，我们以之前介绍过的线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。我们把速度项$\boldsymbol{v}$初始化为和参数形状相同的零张量。

```{.python .input  n=7}
# 生成数据集。
num_inputs, num_examples, true_w, true_b, features, labels = gb.get_data_ch7()

# 初始化模型参数和中间变量。
def init_params_vars():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    vs = []
    for param in params:
        param.attach_grad()
        # 把速度项初始化为和参数形状相同的零张量。
        vs.append(param.zeros_like())
    return [params, vs]
```

动量法的实现也很简单。我们在小批量随机梯度下降的基础上添加速度变量。

```{.python .input  n=8}
def sgd_momentum(params_vars, hyperparams, batch_size):
    lr = hyperparams['lr']
    mom = hyperparams['mom']
    [w, b], vs = params_vars
    for param, v in zip([w, b], vs):
        v[:] = mom * v + lr * param.grad / batch_size
        param[:] -= v
```

我们先将动量超参数$\gamma$（`mom`）设0.99。此时，小梯度随机梯度下降可被看作使用了特殊梯度：这个特殊梯度是最近100个时刻的$100\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后上升。这很可能是由于特殊梯度中较大的系数100造成的。

```{.python .input  n=10}
gb.optimize(optimizer_fn=sgd_momentum, params_vars=init_params_vars(),
            hyperparams={'lr': 0.2, 'mom': 0.99}, features=features,
            labels=labels, decay_epoch=2)
```

假设学习率不变，为了降低上述特殊梯度中的系数，我们将动量超参数$\gamma$（`mom`）设0.9。此时，上述特殊梯度变成最近10个时刻的$10\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后下降。

```{.python .input  n=11}
gb.optimize(optimizer_fn=sgd_momentum, params_vars=init_params_vars(),
            hyperparams={'lr': 0.2, 'mom': 0.9}, features=features,
            labels=labels, decay_epoch=2)
```

继续保持学习率不变，我们将动量超参数$\gamma$（`mom`）设0.5。此时，小梯度随机梯度下降可被看作使用了新的特殊梯度：这个特殊梯度是最近2个时刻的$2\nabla f_\mathcal{B}(\boldsymbol{x})$的加权平均。我们观察到，损失函数值在3个迭代周期后下降，且下降曲线较平滑。最终，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=12}
gb.optimize(optimizer_fn=sgd_momentum, params_vars=init_params_vars(),
            hyperparams={'lr': 0.2, 'mom': 0.5}, features=features,
            labels=labels, decay_epoch=2)
```

## 使用Gluon的实现

下面我们展示如何使用Gluon实验动量法。我们可以在`Trainer`中定义动量超参数`momentum`来使用动量法。以下几组实验分别重现了本节中使用NDArray实现动量法的实验结果。这些结果有一定的随机性。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.99})
gb.optimize_gluon(trainer=trainer, features=features, labels=labels, net=net,
                  decay_epoch=2)
```

```{.python .input}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.9})
gb.optimize_gluon(trainer=trainer, features=features, labels=labels, net=net,
                  decay_epoch=2)
```

```{.python .input}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.5})
gb.optimize_gluon(trainer=trainer, features=features, labels=labels, net=net,
                  decay_epoch=2)
```

## 小结

* 动量法使用了指数加权移动平均的思想。
* 使用Gluon的`Trainer`可以方便地使用动量法。


## 练习

* 使用其他动量超参数和学习率的组合，观察实验结果。
* 如果想用本节中Gluon实现动量法的代码重现小批量随机梯度下降，应该把动量参数改为多少？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1879)


![](../img/qr_momentum.svg)
