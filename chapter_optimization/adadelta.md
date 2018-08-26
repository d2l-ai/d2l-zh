# Adadelta

我们在[“RMSProp”](rmsprop.md)一节中描述了，RMSProp针对Adagrad在迭代后期可能较难找到有用解的问题，对小批量随机梯度按元素平方项做指数加权移动平均而不是累加。另一种应对该问题的优化算法叫做Adadelta [1]。有意思的是，它没有学习率超参数。


## Adadelta算法

Adadelta算法也像RMSProp一样，使用了小批量随机梯度按元素平方的指数加权移动平均变量$\boldsymbol{s}$，并将其中每个元素初始化为0。
给定超参数$\rho$且$0 \leq \rho < 1$，
在每次迭代中，RMSProp首先计算小批量随机梯度$\boldsymbol{g}$，然后对该梯度按元素平方项$\boldsymbol{g} \odot \boldsymbol{g}$做指数加权移动平均，记为$\boldsymbol{s}$：

$$\boldsymbol{s} \leftarrow \rho \boldsymbol{s} + (1 - \rho) \boldsymbol{g} \odot \boldsymbol{g}. $$

然后，计算当前需要迭代的目标函数自变量的变化量$\boldsymbol{g}'$：

$$ \boldsymbol{g}' \leftarrow \frac{\sqrt{\Delta\boldsymbol{x} + \epsilon}}{\sqrt{\boldsymbol{s} + \epsilon}}   \odot \boldsymbol{g}, $$


其中$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-5}$。和Adagrad与RMSProp一样，目标函数自变量中每个元素都分别拥有自己的学习率。上式中$\Delta\boldsymbol{x}$初始化为零张量，并记录$\boldsymbol{g}'$按元素平方的指数加权移动平均：

$$\Delta\boldsymbol{x} \leftarrow \rho \Delta\boldsymbol{x} + (1 - \rho) \boldsymbol{g}' \odot \boldsymbol{g}'. $$

同样地，最后的自变量迭代步骤与小批量随机梯度下降类似：

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \boldsymbol{g}'. $$

## 实验

首先，导入本节中实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import numpy as np
```

实验中，我们依然以线性回归为例。设数据集的样本数为1000，我们使用权重`w`为[2, -3.4]，偏差`b`为4.2的线性回归模型来生成数据集。该模型的平方损失函数即所需优化的目标函数，模型参数即目标函数自变量。我们把算法中变量$\boldsymbol{s}$和$\Delta\boldsymbol{x}$初始化为和模型参数形状相同的零张量。

```{.python .input  n=1}
# 生成数据集。
num_inputs, num_examples, true_w, true_b, features, labels = gb.get_data_ch7()

# 初始化模型参数和中间变量。
def init_params_vars():
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    sqrs = []
    deltas = []
    for param in params:
        param.attach_grad()
        # 把算法中基于指数加权移动平均的变量初始化为和参数形状相同的零张量。
        sqrs.append(param.zeros_like())
        deltas.append(param.zeros_like())
    return [params, sqrs, deltas]
```

接下来基于NDArray实现Adadelta算法。

```{.python .input}
def adadelta(params_vars, hyperparams, batch_size):
    rho = hyperparams['rho']
    [w, b], sqrs, deltas = params_vars
    eps_stable = 1e-5
    for param, sqr, delta in zip([w, b], sqrs, deltas):
        g = param.grad / batch_size
        sqr[:] = rho * sqr + (1 - rho) * g.square()
        cur_delta = ((delta + eps_stable).sqrt()
                     / (sqr + eps_stable).sqrt() * g)
        delta[:] = rho * delta + (1 - rho) * cur_delta * cur_delta
        param[:] -= cur_delta
```

可以看出，优化所得的模型参数值与它们的真实值较接近。

```{.python .input  n=3}
gb.optimize(optimizer_fn=adadelta, params_vars=init_params_vars(),
            hyperparams={'rho': 0.9999}, features=features, labels=labels)
```

## 使用Gluon的实现

下面我们展示如何使用Gluon实验Adadelta算法。我们可以在Trainer中定义优化算法名称`adadelta`并定义$\rho$超参数`rho`。以下实验重现了本节中使用NDArray实现Adadelta的实验结果。该结果有一定的随机性。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'rho': 0.9999})
gb.optimize_gluon(trainer=trainer, features=features, labels=labels, net=net)
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
