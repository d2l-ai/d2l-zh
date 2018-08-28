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

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import nd
```

```{.python .input  n=2}
features, labels = gb.get_data_ch7()

def init_adam_states():
    v_w, s_w = nd.zeros((features.shape[1], 1)), nd.zeros((features.shape[1], 1)), 
    v_b, s_b = nd.zeros(1), nd.zeros(1), 
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    hp = hyperparams
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hp['t'])
        s_bias_corr = s / (1 - beta2 ** hp['t'])    
        p[:] -= hp['lr'] * v_bias_corr / (s_bias_corr.sqrt() + eps)
    hp['t'] += 1
```

```{.python .input  n=5}
gb.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
```

## 使用Gluon的实现

```{.python .input  n=11}
gb.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)

```

## 小结

* Adam组合了动量法和RMSProp。
* Adam使用了偏差修正。

## 练习

* 使用其他初始学习率，观察并分析实验结果。
* 总结本章各个优化算法的异同。
* 回顾前面几章中你感兴趣的模型，将训练部分的优化算法替换成其他算法，观察并分析实验现象。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2279)


![](../img/qr_adam.svg)

## 参考文献

[1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
