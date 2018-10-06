# Adadelta

我们在[“RMSProp”](rmsprop.md)一节中描述了RMSProp针对Adagrad在迭代后期可能较难找到有用解的问题。RMSProp对小批量随机梯度按元素平方项做指数加权移动平均而不是累加。另一种应对该问题的优化算法叫做Adadelta [1]。有意思的是，它没有学习率这一超参数。

Adadelta算法也像RMSProp一样，使用了小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，它的时间步0时被初始化为0。
给定超参数$\rho$且$0 \leq \rho < 1$（对应RMSProp中的$\gamma$），在时间步$t>0$，同RMSProp一样计算

$$\boldsymbol{s}_t \leftarrow \rho \boldsymbol{s}_{t-1} + (1 - \rho) \boldsymbol{g}_t \odot \boldsymbol{g}_t. $$

不同的在于Adadelta算法还维护一个额外的状态变量$\Delta\boldsymbol{x}\in\mathbb{R}^d$，其元素同样在时间步0时被初始化为0。然后使用它来计算自变量的变化量：

$$ \boldsymbol{g}_t' \leftarrow \sqrt{\frac{\Delta\boldsymbol{x}_{t-1} + \epsilon}{\boldsymbol{s}_t + \epsilon}}   \odot \boldsymbol{g}_t, $$

这里$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-5}$（注意我们不是使用前面常用的$10^{-6}$）。接着更新自变量：

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t. $$

最后，我们使用$\Delta\boldsymbol{x}$来记录$\boldsymbol{g}'$按元素平方的指数加权移动平均：

$$\Delta\boldsymbol{x}_t \leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1 - \rho) \boldsymbol{g}'_t \odot \boldsymbol{g}'_t. $$

可以看到，Adadelta跟RMSProp不同的地方在于使用$\Delta\boldsymbol{x}_t$来替代了超参数$\eta_t$，因此它的主要优势在于不需要手动选取学习率。


## 从零开始实现

首先，导入本节中实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import nd
```

Adadelta需要对每个自变量维护两个状态变量，$\boldsymbol{s}$和$\Delta\boldsymbol{x}$。按上面公式显示Adadelta：

```{.python .input  n=11}
features, labels = gb.get_data_ch7()

def init_adadelta_states():
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g        
```

使用$\rho=0.9$来训练模型。

```{.python .input  n=12}
gb.train_ch7(adadelta, init_adadelta_states(), {'rho': .9}, features, labels)
```

## Gluon实现

Adadelta在Gluon中名称为`adadelta`，其超参数可以通过`rho`来指定。

```{.python .input  n=9}
gb.train_gluon_ch7('adadelta', {'rho': .9}, features, labels)
```

## 小结

* Adadelta没有学习率参数，它通过使用自变量更新量平方的指数加权移动平均来替代学习率。

## 练习

* 如果把试验中的参数$\rho$改小会怎样？观察并分析实验结果。
* 如果将`eps`改成前面使用的`1e-6`会怎么样？分析观察的结果。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2277)

![](../img/qr_adadelta.svg)

## 参考文献

[1] Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
