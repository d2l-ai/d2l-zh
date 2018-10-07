# RMSProp


我们在[“Adagrad”](adagrad.md)一节里提到，由于调整学习率时分母上的变量$\boldsymbol{s}$一直在累加按元素平方的小批量随机梯度，目标函数自变量每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad在迭代后期由于学习率过小，可能较难找到一个有用的解。为了应对这一问题，RMSProp算法对Adagrad做了一点小小的修改 [1]。

## RMSProp算法

我们在[“动量法”](momentum.md)一节里介绍过指数加权移动平均。不同于Adagrad里状态变量$\boldsymbol{s}$是到目前时间步里所有梯度按元素平方和，RMSProp将过去时间步里梯度按元素平方做指数加权移动平均。具体来说，给定超参数$\gamma$且$0 \leq \gamma < 1$，RMSProp在时间步$t>0$里计算

$$\boldsymbol{s}_t \leftarrow \gamma \boldsymbol{s}_{t-1} + (1 - \gamma) \boldsymbol{g}_t \odot \boldsymbol{g}_t. $$

和Adagrad一样，RMSProp将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下，然后更新自变量。

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta_t}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t, $$

其中$\eta_t$是学习率，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-6}$。

因为RMSProp的状态变量是对平方项$\boldsymbol{g}_t \odot \boldsymbol{g}_t$的指数加权移动平均，因此可以看作是最近$1/(1-\gamma)$个时间步的梯度平方项的加权平均，这样自变量每个元素的学习率在迭代过程中避免了“直降不升”的问题。

照例，让我们先观察RMSProp对目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2^2$中自变量的更新轨迹。首先，导入本节中实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
import math
from mxnet import nd
```

回忆在[“Adagrad”](adagrad.md)一节使用$0.4$的学习率，Adagrad对自变量的更新很缓慢。但在同样的学习率下，RMSProp可以快速的接近最优解。

```{.python .input  n=3}
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
gb.show_trace_2d(f_2d, gb.train_2d(rmsprop_2d))
```

## 从零开始实现

接下来按照公式实现RMSProp。

```{.python .input  n=22}
features, labels = gb.get_data_ch7()

def init_rmsprop_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
```

我们将初始学习率设为0.01，并将$\gamma$（`gamma`）设为0.9。此时，变量$\boldsymbol{s}$可看作是最近$1/(1-0.9) = 10$个时间步的平方项$\boldsymbol{g} \odot \boldsymbol{g}$的加权平均。我们观察到，损失函数在迭代后期较震荡。

```{.python .input  n=24}
features, labels = gb.get_data_ch7()
gb.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9},
             features, labels)
```

## Gluon实现

使用名称`rmsprop`可以获取Gluon中预实现的RMSProp算法。注意超参数$\gamma$此时通过`gamma1`指定。

```{.python .input  n=29}
gb.train_gluon_ch7('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                   features, labels)
```

## 小结

* RMSProp和Adagrad的不同在于，RMSProp使用了小批量随机梯度按元素平方的指数加权移动平均变量来调整学习率。
* 理解指数加权移动平均有助于我们调节RMSProp算法中的超参数，例如$\gamma$。

## 练习

* 把$\gamma$的值设为0或1，观察并分析实验结果。
* 试着使用其他的初始学习率和$\gamma$超参数的组合，观察并分析实验结果。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2275)


![](../img/qr_rmsprop.svg)

## 参考文献

[1] Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 26-31.
