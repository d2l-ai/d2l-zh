# Adadelta的Gluon实现

在Gluon里，使用Adadelta很容易，我们无需重新实现该算法。

首先，导入本节中实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import nn
```

下面生成实验数据集并定义线性回归模型。

```{.python .input  n=1}
# 生成数据集。
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 线性回归模型。
net = nn.Sequential()
net.add(nn.Dense(1))
```

我们可以在Trainer中定义优化算法名称`adadelta`并定义$\rho$超参数`rho`。以下实验重现了[“Adadelta”](adadelta.md)一节中实验结果。

```{.python .input  n=3}
net.initialize(init.Normal(sigma=0.01), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'rho': 0.9999})
gb.optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=None,
            log_interval=10, features=features, labels=labels, net=net)
```

## 小结

* 使用Gluon的`Trainer`可以方便地使用Adadelta。

## 练习

* 如果把试验中的参数$\rho$改小会怎样？观察并分析实验结果。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2278)


![](../img/qr_adadelta-gluon.svg)
