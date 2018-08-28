# Adadelta

我们在[“RMSProp”](rmsprop.md)一节中描述了，RMSProp针对Adagrad在迭代后期可能较难找到有用解的问题，对小批量随机梯度按元素平方项做指数加权移动平均而不是累加。另一种应对该问题的优化算法叫做Adadelta [1]。有意思的是，它没有学习率超参数。


## Adadelta算法

Adadelta算法也像RMSProp一样，使用了小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，并将其中每个元素初始化为0。
给定超参数$\rho$且$0 \leq \rho < 1$，
在每次迭代中，RMSProp首先计算小批量随机梯度$\boldsymbol{g}$，然后对该梯度按元素平方项$\boldsymbol{g} \odot \boldsymbol{g}$做指数加权移动平均，记为$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \rho \boldsymbol{s} + (1 - \rho) \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，计算当前需要迭代的目标函数自变量的变化量$\boldsymbol{g}'$：

$$ \boldsymbol{g}' \leftarrow \frac{\sqrt{\Delta\boldsymbol{x} + \epsilon}}{\sqrt{\boldsymbol{s} + \epsilon}}   \odot \boldsymbol{g}, $$


其中$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-6}$。和Adagrad与RMSProp一样，目标函数自变量中每个元素都分别拥有自己的学习率。上式中$\Delta\boldsymbol{x}$初始化为零张量，并记录$\boldsymbol{g}'$按元素平方的指数加权移动平均：

$$\Delta\boldsymbol{x} \leftarrow \rho \Delta\boldsymbol{x} + (1 - \rho) \boldsymbol{g}' \odot \boldsymbol{g}'. $$

同样地，最后的自变量迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}'. $$

## 从零开始的实现

首先，导入本节中实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import nd
```

AdeDelta需要对每个自变量维护两个状态变量，$\boldsymbol{s}$和$\Delta\boldsymbol{x}$。

```{.python .input  n=11}
# 生成数据集。
features, labels = gb.get_data_ch7()

def init_adadelta_states():
    s_w, delta_w = nd.zeros((features.shape[1], 1)), nd.zeros((features.shape[1], 1)), 
    s_b, delta_b = nd.zeros(1), nd.zeros(1), 
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho = hyperparams['rho']
    eps = 1e-6
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps) / (s + eps)).sqrt() * p.grad
        delta[:] = rho * delta + (1 - rho) * g * g
        p[:] -= g
```

```{.python .input  n=12}
gb.train_ch7(adadelta, init_adadelta_states(), {'rho': .1}, features, labels)
```

## 使用Gluon的实现

下面我们展示如何使用Gluon实验Adadelta算法。我们可以在Trainer中定义优化算法名称`adadelta`并定义$\rho$超参数`rho`。以下实验重现了本节中使用NDArray实现Adadelta的实验结果。该结果有一定的随机性。

```{.python .input  n=9}
gb.train_gluon_ch7('adadelta', {'rho': .9}, features, labels)

```

## 小结

* Adadelta没有学习率参数。
* 使用Gluon的`Trainer`可以方便地使用Adadelta。


## 练习

* Adadelta为什么不需要设置学习率超参数？它被什么代替了？
* 如果把试验中的参数$\rho$改小会怎样？观察并分析实验结果。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2277)

![](../img/qr_adadelta.svg)

## 参考文献

[1] Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
