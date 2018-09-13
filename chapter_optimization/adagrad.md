# Adagrad


在我们之前介绍过的优化算法中，无论是梯度下降、（小批量）随机梯度下降还是使用动量法，目标函数自变量的每一个元素在相同时刻都使用同一个学习率来自我迭代。举个例子，假设目标函数为$f$，自变量为一个多维向量$[x_1, x_2]^\top$，该向量中每一个元素在更新时都使用相同的学习率。例如在学习率为$\eta$的梯度下降中，元素$x_1$和$x_2$都使用相同的学习率$\eta$来自我迭代：

$$
x_1 \leftarrow x_1 - \eta \frac{\partial{f}}{\partial{x_1}}, \quad
x_2 \leftarrow x_2 - \eta \frac{\partial{f}}{\partial{x_2}}.
$$

在[“动量法”](./momentum.md)一节里我们看到当$x_1$和$x_2$的梯度值有较大差别时，我们需要选择足够小的学习率使得自变量在梯度值较大的维度上不发散。但这样会导致自变量在梯度值较小的维度上迭代过慢。动量法依赖指数加权移动平均使得自变量的更新方向更加一致，从而降低发散的可能。这一节我们介绍Adagrad算法，它根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。


## Adagrad算法

Adagrad的算法会使用一个小批量随机梯度按元素平方的累加变量$\boldsymbol{s}\in\mathbb{R}^d$。在时间步0，adagrad将$\boldsymbol{s}_0$中每个元素初始化为0。在每次迭代中，首先将梯度$\boldsymbol{g}_t$按元素平方后累加到变量$\boldsymbol{s}_t$：

$$\boldsymbol{s}_t \leftarrow \boldsymbol{s}_t + \boldsymbol{g}_t \odot \boldsymbol{g}_t,$$

其中$\odot$是按元素相乘（请参见[“数学基础”](../chapter_appendix/math.md)一节）。接着，我们将目标函数自变量中每个元素的学习率通过按元素运算重新调整一下：

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta_t}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t,$$

其中$\eta_t$是学习率且一般为常数，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-6}$。这里开方、除法和乘法的运算都是按元素进行的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

## Adagrad的特点

需要强调的是，小批量随机梯度按元素平方的累加变量$\boldsymbol{s}$出现在学习率的分母项中。因此，如果目标函数有关自变量中某个元素的偏导数一直都较大，那么就让该元素的学习率下降快一点；反之，如果目标函数有关自变量中某个元素的偏导数一直都较小，那么就让该元素的学习率下降慢一点。然而，由于$\boldsymbol{s}$一直在累加按元素平方的梯度，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad在迭代后期由于学习率过小，可能较难找到一个有用的解。

下面我们仍然以目标函数$f(\boldsymbol{x})=0.1x_1^2+2x_2$为例观察Adagrad对自变量的迭代过程。先导入实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
import math
from mxnet import nd
```

下面实现Adagrad并用同前一样的学习率$0.4$进行训练。可以看到自变量的更新轨迹更加平滑，且在$x_2$轴上没有抖动。但由于$\boldsymbol{s}$的累加效果使得学习率快速衰减，自变量在后期前进不够迅速。

```{.python .input  n=2}
def adagrad_2d(x1, x2, s1, s2):    
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度。
    s1 += g1 ** 2
    s2 += g2 ** 2        
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
gb.show_trace_2d(f_2d, gb.train_2d(adagrad_2d))
```

下面增大学习率到$2$，可以看到自变量非常迅速的接近了最优解。

```{.python .input  n=3}
eta = 2
gb.show_trace_2d(f_2d, gb.train_2d(adagrad_2d))
```

## 从零开始实现

同动量法一样，Adagrad需要对每个自变量维护同它一样形状的状态变量$\boldsymbol{s}$。接下来根据公式实现Adagrad。

```{.python .input  n=4}
features, labels = gb.get_data_ch7()

def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    hp, eps = hyperparams, 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hp['lr'] * p.grad / (s + eps).sqrt()
```

接下来使用$0.1$的学习率来训练模型，它是[“梯度下降和随机梯度下降”](./gd-sgd.md)一节中使用的2倍。

```{.python .input  n=5}
gb.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
```

## 使用Gluon的实现

使用名称`adagrad`可以获得Gluon对AdaGrad的实现。

```{.python .input  n=6}
gb.train_gluon_ch7('adagrad', {'learning_rate': 0.1}, features, labels)
```

## 小结

* Adagrad在迭代过程中不断调整学习率，并让目标函数自变量中每个元素都分别拥有自己的学习率。
* 使用Adagrad时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。

## 练习

* 在介绍Adagrad的特点时，我们提到了它可能存在的问题。你能想到什么办法来应对这个问题？
* 尝试使用其他的初始学习率，结果有什么变化？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2273)

![](../img/qr_adagrad.svg)


## 参考文献

[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.
