# Adam

Adam是另一个对RMSProp的改进算法 [1]。但一个不同点在于Adam对梯度做了指数加权移动平均。

## 算法

首先将动量变量$\boldsymbol{v}\in\mathbb{R}^d$的元素在时间步0时初始化成0。
给定超参数$\beta_1$且满足$0 \leq \beta_1 < 1$（算法作者建议设为0.9），在时间步$t$计算

$$\boldsymbol{v}_t \leftarrow \beta_1 \boldsymbol{v}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t. $$

将其展开我们得到$\boldsymbol{v}_t =  (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} \boldsymbol{g}_i$。考虑一个简单情况：假设$\boldsymbol{g}_i=\boldsymbol{g}$对所有$i$成立，那么$\boldsymbol{v}_t = \left((1-\beta_1)\sum_{i=1}^t \beta_1^{t-i}\right)\boldsymbol{g} = \left(1 - \beta_1^t\right)\boldsymbol{g}$。

当$t$较小时，$1 - \beta_1^t$会较小。例如当$\beta_1 = 0.9$时且$t=1$时，$\boldsymbol{v}_1 = 0.1\boldsymbol{g}$，当$t=10$时，$\boldsymbol{v}_{10} = 0.65\boldsymbol{g}$。为了消除这样的影响，我们可以将$\boldsymbol{v}_t$再除以$1 - \beta_1^t$，从而使得过去各时间步小批量随机梯度权值之和为1。这也叫做偏差修正。也就是：

$$\boldsymbol{v}'_t \leftarrow \frac{\boldsymbol{v}_t}{1 - \beta_1^t}.$$

接下来和RMSProp中一样，给定超参数$\beta_2$且满足$0 \leq \beta_2 < 1$（算法作者建议设为0.999），更新状态变量$\boldsymbol{s}$：

$$\boldsymbol{s}_t \leftarrow \beta_2 \boldsymbol{s}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t \odot \boldsymbol{g}_t. $$

且同样做偏差修正：

$$\boldsymbol{s}'_t \leftarrow \frac{\boldsymbol{s}_t}{1 - \beta_2^t}. $$

最后使用修正后的变量$\boldsymbol{v}'_t$和$\boldsymbol{s}'_t$来更新自变量：

$$\boldsymbol{x}_{t} \leftarrow \boldsymbol{x}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{s}_{t}'+\epsilon}}\odot\boldsymbol{v}_{t}'$$

其中$\eta>0$是学习率且$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-6}$。

## 从零开始实现

首先，导入实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import nd
```

按照公式实现Adam，时间步$t$通过`hyperparams`传入：

```{.python .input  n=2}
features, labels = gb.get_data_ch7()

def init_adam_states():
    v_w, v_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])    
        p[:] -= hyperparams['lr'] * v_bias_corr / (s_bias_corr.sqrt() + eps)
    hyperparams['t'] += 1
```

使用学习率$0.01$来训练

```{.python .input  n=5}
gb.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
```

## Gluon实现

通过名称`adam`可以获取Gluon中的实现：

```{.python .input  n=11}
gb.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)
```

## 小结

* Adam在RMSProp基础上对梯度也做了指数加权移动平均。
* Adam使用了偏差修正。

## 练习

* 使用其他初始学习率，观察并分析实验结果。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2279)


![](../img/qr_adam.svg)

## 参考文献

[1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
