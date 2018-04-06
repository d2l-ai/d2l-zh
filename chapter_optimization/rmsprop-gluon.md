# RMSProp——使用`Gluon`


在`Gluon`里，使用RMSProp很方便，我们无需重新实现该算法。

首先，导入实验所需的包。

```{.python .input}
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import numpy as np
import sys
sys.path.append('..')
import utils
```

下面生成实验数据集并定义线性回归模型。

```{.python .input  n=1}
# 生成数据集。
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)

# 线性回归模型。
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
```

例如，以使用动量法的小批量随机梯度下降为例，我们可以在`Trainer`中定义动量超参数`momentum`。以下几组实验分别重现了[“RMSProp——从零开始”](rmsprop-scratch.md)一节中实验结果。

```{.python .input  n=3}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'rmsprop',
                        {'learning_rate': 0.03, 'gamma1': 0.9})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
               log_interval=10, X=X, y=y, net=net)
```

```{.python .input}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'rmsprop',
                        {'learning_rate': 0.03, 'gamma1': 0.999})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
               log_interval=10, X=X, y=y, net=net)
```

## 小结

* 使使用`Gluon`的`Trainer`可以方便地使用RMSProp。

## 练习

* 试着使用其他的初始学习率和$\gamma$超参数的组合，观察并分析实验现象。

## 讨论

欢迎扫码直达[本节内容讨论区](https://discuss.gluon.ai/t/topic/2276)：

![](../img/qr_rmsprop-gluon.svg)
