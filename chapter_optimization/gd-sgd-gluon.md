# 梯度下降和随机梯度下降——使用`Gluon`

在`Gluon`里，使用小批量随机梯度下降很容易。我们无需重新实现该算法。特别地，当批量大小等于训练集大小时，该算法即为梯度下降；批量大小为1即为随机梯度下降。

```{.python .input  n=1}
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import random

# 为方便比较同一优化算法的从零开始实现和Gluon实现，将输出保持确定。
random.seed(1)
mx.random.seed(1)

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

```{.python .input  n=2}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import utils

# 优化目标函数。
def optimize(batch_size, trainer, num_epochs, decay_epoch, log_interval, X, y,
             net, print_lr=True):
    dataset = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    square_loss = gluon.loss.L2Loss()
    y_vals = [nd.mean(square_loss(net(X), y)).asnumpy()]
    print('batch size', batch_size)
    for epoch in range(1, num_epochs + 1): 
        # 学习率自我衰减。
        if decay_epoch is not None and epoch > decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (features, label) in enumerate(data_iter):
            with autograd.record():
                output = net(features)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(nd.mean(square_loss(net(X), y)).asnumpy())
        if print_lr:
            print("epoch %d, learning rate %f, loss %.4e" %
                  (epoch, trainer.learning_rate, y_vals[-1]))
        else:
            print("epoch %d, loss %.4e" % (epoch, y_vals[-1]))
    print('w:', np.reshape(net[0].weight.data().asnumpy(), (1, -1)),
          'b:', net[0].bias.data().asnumpy()[0], '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    utils.set_fig_size(mpl)
    plt.semilogy(x_vals, y_vals)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show() 
```

当批量大小为1时，训练使用的是随机梯度下降。在当前学习率下，目标函数值在早期快速下降后略有波动。当epoch大于2，学习率自我衰减后，目标函数值下降后较平稳。最终学到的参数值与真实值较接近。

```{.python .input  n=3}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize(batch_size=1, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)
```

当批量大小为1000时，由于训练数据集含1000个样本，此时训练使用的是梯度下降。在当前学习率下，目标函数值在前两个epoch下降较快。当epoch大于2，学习率自我衰减后，目标函数值下降较慢。最终学到的参数值与真实值较接近。

```{.python .input  n=4}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.999})
optimize(batch_size=1000, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=1000, X=X, y=y, net=net)
```

当批量大小为10时，由于训练数据集含1000个样本，此时训练使用的是（小批量）随机梯度下降。最终学到的参数值与真实值较接近。

```{.python .input  n=5}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)
```

同样是批量大小为10，我们把学习率改大。这时我们观察到目标函数值不断增大。这时典型的overshooting问题。

```{.python .input  n=6}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 5})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)
```

同样是批量大小为10，我们把学习率改小。这时我们观察到目标函数值下降较慢，直到3个epoch也没能得到接近真实值的解。

```{.python .input  n=7}
net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.002})
optimize(batch_size=10, trainer=trainer, num_epochs=3, decay_epoch=2,
         log_interval=10, X=X, y=y, net=net)
```

## 结论

* 使用`Gluon`的`Trainer`可以使模型训练变得更容易。
* 使用`gluon.Trainer`的`learning_rate`属性和`set_learning_rate`函数可以随意调整学习率。


## 练习

* 你还能想到哪些学习率自我衰减的方法？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1878)
