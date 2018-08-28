# RMSProp


我们在[“Adagrad”](adagrad.md)一节里提到，由于调整学习率时分母上的变量$\boldsymbol{s}$一直在累加按元素平方的小批量随机梯度，目标函数自变量每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad在迭代后期由于学习率过小，可能较难找到一个有用的解。为了应对这一问题，RMSProp算法对Adagrad做了一点小小的修改 [1]。

## RMSProp算法

我们在[“动量法”](momentum.md)一节里介绍过指数加权移动平均。事实上，RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，并将其中每个元素初始化为0。
给定超参数$\gamma$且$0 \leq \gamma < 1$，
在每次迭代中，RMSProp首先计算小批量随机梯度$\boldsymbol{g}$，然后对该梯度按元素平方项$\boldsymbol{g} \odot \boldsymbol{g}$做指数加权移动平均，记为$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \gamma \boldsymbol{s} + (1 - \gamma) \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，和Adagrad一样，将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：

$$\boldsymbol{g}' \leftarrow \frac{\eta}{\sqrt{\boldsymbol{s} + \epsilon}} \odot \boldsymbol{g}, $$

其中$\eta$是初始学习率且$\eta > 0$，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-6}$。和Adagrad一样，模型参数中每个元素都分别拥有自己的学习率。同样地，最后的自变量迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}'. $$

需要强调的是，RMSProp只在Adagrad的基础上修改了变量$\boldsymbol{s}$的更新方法：对平方项$\boldsymbol{g} \odot \boldsymbol{g}$从累加变成了指数加权移动平均。由于变量$\boldsymbol{s}$可看作是最近$1/(1-\gamma)$个时刻的平方项$\boldsymbol{g} \odot \boldsymbol{g}$的加权平均，自变量每个元素的学习率在迭代过程中避免了“直降不升”的问题。

## 实验

首先，导入本节中实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import math
import gluonbook as gb
from mxnet import nd
```

```{.python .input  n=3}
def rmsprop_2d(x1, x2, s_x1, s_x2):    
    eps = 1e-6
    g_x1, g_x2 = 0.2 * x1, 4 * x2
    s_x1 = gamma * s_x1 + (1 - gamma) * g_x1 ** 2
    s_x2 = gamma * s_x2 + (1 - gamma) * g_x2 ** 2        
    x1 -= eta / math.sqrt(s_x1 + eps) * g_x1
    x2 -= eta / math.sqrt(s_x2 + eps) * g_x2
    return x1, x2, s_x1, s_x2

eta, gamma = 0.4, 0.9
f_2d = lambda x1, x2: 0.1 * x1 ** 2 + 2 * x2 ** 2
gb.show_trace_2d(f_2d, gb.train_2d(rmsprop_2d))
```

## 从零开始实现

```{.python .input  n=22}
# 生成数据集。
features, labels = gb.get_data_ch7()

# 初始化模型参数和中间变量。
def init_rmsprop_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    hp = hyperparams
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] = hp['gamma'] * s + (1 - hp['gamma']) * p.grad.square()
        p[:] -= hp['lr'] * p.grad / (s + eps).sqrt()
```

我们将初始学习率设为0.01，并将$\gamma$（`gamma`）设为0.9。此时，变量$\boldsymbol{s}$可看作是最近$1/(1-0.9) = 10$个时刻的平方项$\boldsymbol{g} \odot \boldsymbol{g}$的加权平均。我们观察到，损失函数在迭代后期较震荡。

```{.python .input  n=24}
features, labels = gb.get_data_ch7()
gb.train_ch7(rmsprop, init_rmsprop_states(), {'lr': .01, 'gamma':0.9}, features, labels)
```

## 使用Gluon的实现

使用名称`rmsprop`可以获取Gluon中预实现的RMSProp算法。注意超参数$\gamma$此时通过`gamma1`指定。

```{.python .input  n=29}
gb.train_gluon_ch7('rmsprop', {'learning_rate': 0.01,'gamma1': 0.9}, features, labels)
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
