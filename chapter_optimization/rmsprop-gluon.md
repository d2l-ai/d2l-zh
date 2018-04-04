# RMSProp --- 使用Gluon


在`Gluon`里，使用RMSProp很容易。我们无需重新实现它。

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

```{.python .input  n=1}
# 生成数据集。
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)

# 创建模型和定义损失函数。
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
```

我们需要在`gluon.Trainer`中指定优化算法名称`rmsprop`并设置参数。例如设置初始学习率`learning_rate`和指数加权移动平均中gamma1参数。

我们将初始学习率设为0.03，并将gamma设为0.9。损失函数在迭代后期较震荡。

```{.python .input  n=3}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'rmsprop',
                        {'learning_rate': 0.03, 'gamma1': 0.9})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
               log_interval=10, X=X, y=y, net=net)
```

我们将gamma调大一点，例如0.999。这时损失函数在迭代后期较平滑。

```{.python .input}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'rmsprop',
                        {'learning_rate': 0.03, 'gamma1': 0.999})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
               log_interval=10, X=X, y=y, net=net)
```

## 小结

* 使用`Gluon`的`Trainer`可以轻松使用RMSProp。

## 练习

* 试着使用其他的初始学习率和gamma参数的组合，观察实验结果。

## 讨论

欢迎扫码直达[本节内容讨论区](https://discuss.gluon.ai/t/topic/2276)：

![](../img/qr_rmsprop-gluon.svg)
