# Adam

Adam是一个组合了动量法和RMSProp的优化算法 [1]。下面我们来介绍Adam算法。


## Adam算法

Adam算法使用了动量变量$\boldsymbol{v}$和RMSProp中小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，并将它们中每个元素初始化为0。在每次迭代中，时刻$t$的小批量随机梯度记作$\boldsymbol{g}_t$。


和动量法类似，给定超参数$\beta_1$且满足$0 \leq \beta_1 < 1$（算法作者建议设为0.9），将小批量随机梯度的指数加权移动平均记作动量变量$\boldsymbol{v}$，并将它在时刻$t$的值记作$\boldsymbol{v}_t$：

$$\boldsymbol{v}_t \leftarrow \beta_1 \boldsymbol{v}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t. $$

和RMSProp中一样，给定超参数$\beta_2$且满足$0 \leq \beta_2 < 1$（算法作者建议设为0.999），
将小批量随机梯度按元素平方后做指数加权移动平均得到$\boldsymbol{s}$，并将它在时刻$t$的值记作$\boldsymbol{s}_t$：

$$\boldsymbol{s}_t \leftarrow \beta_2 \boldsymbol{s}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t. $$

由于我们将$\boldsymbol{v}$和$\boldsymbol{s}$中的元素都初始化为0，
在时刻$t$我们得到$\boldsymbol{v}_t =  (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} \boldsymbol{g}_i$。将过去各时刻小批量随机梯度的权值相加，得到 $(1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} = 1 - \beta_1^t$。需要注意的是，当$t$较小时，过去各时刻小批量随机梯度权值之和会较小。例如当$\beta_1 = 0.9$时，$\boldsymbol{v}_1 = 0.1\boldsymbol{g}_1$。为了消除这样的影响，对于任意时刻$t$，我们可以将$\boldsymbol{v}_t$再除以$1 - \beta_1^t$，从而使得过去各时刻小批量随机梯度权值之和为1。这也叫做偏差修正。在Adam算法中，我们对变量$\boldsymbol{v}$和$\boldsymbol{s}$均作偏差修正：

$$\hat{\boldsymbol{v}}_t \leftarrow \frac{\boldsymbol{v}_t}{1 - \beta_1^t}, $$

$$\hat{\boldsymbol{s}}_t \leftarrow \frac{\boldsymbol{s}_t}{1 - \beta_2^t}. $$


接下来，Adam算法使用以上偏差修正后的变量$\hat{\boldsymbol{v}}_t$和$\hat{\boldsymbol{s}}_t$，将模型参数中每个元素的学习率通过按元素运算重新调整：

$$\boldsymbol{g}_t' \leftarrow \frac{\eta \hat{\boldsymbol{v}}_t}{\sqrt{\hat{\boldsymbol{s}}_t + \epsilon}},$$

其中$\eta$是初始学习率且$\eta > 0$，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-8}$。和Adagrad、RMSProp以及Adadelta一样，目标函数自变量中每个元素都分别拥有自己的学习率。

最后，时刻$t$的自变量$\boldsymbol{x}_t$的迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}_t'. $$

## 实验

首先，导入实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import numpy as np
```

实验中，我们依然以线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。我们把算法中变量$\boldsymbol{v}$和$\boldsymbol{s}$初始化为和模型参数形状相同的零张量。

```{.python .input  n=1}
# 生成数据集。
num_inputs, num_examples, true_w, true_b, features, labels = gb.get_data_ch7()

# 初始化模型参数和中间变量。
def init_params_vars():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    vs = []
    sqrs = []
    for param in params:
        param.attach_grad()
        # 把算法中基于指数加权移动平均的变量初始化为和参数形状相同的零张量。
        vs.append(param.zeros_like())
        sqrs.append(param.zeros_like())
    return [params, vs, sqrs]
```

接下来基于NDArray实现Adam算法。

```{.python .input}
def adam(params_vars, hyperparams, batch_size, t):
    lr = hyperparams['lr']
    [w, b], vs, sqrs = params_vars
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8
    for param, v, sqr in zip([w, b], vs, sqrs):      
        g = param.grad / batch_size
        v[:] = beta1 * v + (1 - beta1) * g
        sqr[:] = beta2 * sqr + (1 - beta2) * g.square()
        v_bias_corr = v / (1 - beta1 ** t)
        sqr_bias_corr = sqr / (1 - beta2 ** t)    
        param[:] = param - lr * v_bias_corr / (
            sqr_bias_corr.sqrt() + eps_stable)  
```

可以看出，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=3}
gb.optimize(optimizer_fn=adam, params_vars=init_params_vars(),
            hyperparams={'lr': 0.1}, features=features, labels=labels,
            is_adam=True)
```

## 使用Gluon的实现

下面我们展示如何使用Gluon实验Adam算法。我们可以在Trainer中定义优化算法名称`adam`并定义初始学习率。以下实验重现了本节中使用NDArray实现Adam的实验结果。该结果有一定的随机性。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.1})
gb.optimize_gluon(trainer=trainer, features=features, labels=labels, net=net)
```

## 小结

* Adam组合了动量法和RMSProp。
* Adam使用了偏差修正。
* 使用Gluon的`Trainer`可以方便地使用Adam。


## 练习

* 使用其他初始学习率，观察并分析实验结果。
* 总结本章各个优化算法的异同。
* 回顾前面几章中你感兴趣的模型，将训练部分的优化算法替换成其他算法，观察并分析实验现象。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2279)


![](../img/qr_adam.svg)

## 参考文献

[1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
