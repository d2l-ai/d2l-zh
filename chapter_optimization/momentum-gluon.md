# 动量法 --- 使用Gluon


在`Gluon`里，使用动量法很容易。我们无需重新实现它。例如，在随机梯度下降中，我们可以定义`momentum`参数。

```{.python .input  n=1}
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import random

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

为了使学习率在两个epoch后自我衰减，我们需要访问`gluon.Trainer`的`learning_rate`属性和`set_learning_rate`函数。

```{.python .input}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
import sys
sys.path.append('..')
import utils
```

使用动量法，最终学到的参数值与真实值较接近。

```{.python .input  n=3}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'momentum': 0.9})
utils.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
               log_interval=10, X=X, y=y, net=net)
```

## 结论

* 使用`Gluon`的`Trainer`可以轻松使用动量法。

## 练习

* 如果想用以上代码重现随机梯度下降，应该把动量参数改为多少？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1880)
