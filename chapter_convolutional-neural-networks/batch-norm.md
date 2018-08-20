# 批量归一化

这一节我们介绍批量归一化（batch normalization）层，它能让深层卷积网络的训练变得更加容易 [1]。在 [“实战Kaggle比赛：预测房价和K折交叉验证”](../chapter_supervised-learning/kaggle-gluon-kfold.md) 小节里，我们对输入数据做了归一化处理，即将每个特征在所有样本上的值转归一化成均值0方差1。这个处理可以保证训练数据的值都在同一量级上，从而使得训练时模型参数更加稳定。

通常来说，数据归一化预处理对于浅层模型就足够有效了。输出数值在只经过几个神经层后一般不会出现剧烈变化。但对于深层神经网络来说，情况会变得比较复杂：每一层里都对输入乘以权重后得到输出。当很多层这样的相乘累计在一起时，一个输入数据较小的改变都可能导致输出产生巨大变化，从而带来不稳定性。

批量归一化层就是针对这个情况提出的。它将一个批量里的输入数据进行归一化后再输出。如果我们将批量归一化层放置在网络的各个层之间，那么就可以不断的对中间输出进行调整，从而保证整个网络的中间输出在数值上都是稳定的。

## 批量归一化层

我们首先看将批量归一化层放置在全连接层后时的情况，它的机制类似于数据归一处理。输入一个批量数据时，假设这个全连接层输出$n$个向量数据点 $X = \{x_1,\ldots,x_n\}$，其中$x_i\in\mathbb{R}^p$。我们可以计算数据点在这个批量里面的均值和方差，其均为长度$p$的向量：

$$\mu \leftarrow \frac{1}{n}\sum_{i = 1}^{n}x_i,$$
$$\sigma^2 \leftarrow \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2.$$

对于数据点 $x_i$，我们可以对它的每一个特征维进行归一化：

$$\hat{x}_i \leftarrow \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},$$

这里$\epsilon$是一个很小的常数，保证分母大于0。在上面归一化的基础上，批量归一化层引入了两个可以学习的模型参数，拉升参数 $\gamma$ 和偏移参数 $\beta$。它们是长为$p$的向量，作用在$\hat{x}_i$上：

$$y_i \leftarrow \gamma \hat{x}_i + \beta.$$

这里$Y = \{y_1, \ldots, y_n\}$是批量归一化层的输出。

如果批量归一化层是放置在卷积层后面，那么我们将通道维当做是特征维，空间维（高和宽）里的元素则当成是样本（参考[“多输入和输出通道”](channels.md)一节里我们对$1\times 1$卷积层的讨论）。

通常训练的时候我们使用较大的批量大小来获取更好的计算性能，这时批量内样本均值和方差的计算都较为准确。但在预测的时候，我们可能使用很小的批量大小，甚至每次我们只对一个样本做预测，这时我们无法得到较为准确的均值和方差。对此，一般的解决方法是维护一个移动平滑的样本均值和方差，从而在预测时使用。和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果是不一样的。

下面我们通过NDArray来实现这个计算。

```{.python .input  n=72}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import loss as gloss, nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过 autograd 来获取是不是在训练环境下。
    if not autograd.is_training():
        # 如果是在预测模式下，直接使用传入的移动平滑均值和方差。
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        # 接在全连接层后情况，计算特征维上的均值和方差。
        if len(X.shape) == 2:
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        # 接在二维卷积层后的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持 X
        # 的形状以便后面可以正常的做广播运算。
        else:
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下用当前的均值和方差做归一化。
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # 更新移动平滑均值和方差。
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 拉升和偏移。
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var
```

接下来我们自定义一个BatchNorm层。它保存参与求导和更新的模型参数`beta`和`gamma`，同时也维护移动平滑的均值和方差使得在预测时可以使用。

```{.python .input  n=73}
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1,
                                                         1)
        # 参与求导和更新的模型参数，分别初始化成 0 和 1。
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        # 不参与求导的模型参数。全在 CPU 上初始化成 0。
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        # 如果 X 不在 CPU 上，将 moving_mean 和 moving_varience 复制到对应设备上。
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # 保存更新过的 moving_mean 和 moving_var。
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

## 使用批量归一化层的LeNet

下面我们修改[“卷积神经网络”](lenet.md)这一节介绍的LeNet来使用批量归一化层。我们在所有的卷积层和全连接层与激活层之间加入批量归一化层，来使得每层的输出都被归一化。

```{.python .input  n=74}
net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    BatchNorm(120, num_dims=2),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    BatchNorm(84, num_dims=2),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)
```

使用同之前一样的超参数，可以发现前面五个迭代周期的收敛有明显加速。

```{.python .input  n=77}
lr = 1.0
num_epochs = 5
batch_size = 256
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx,
             num_epochs)
```

最后我们查看下第一个批量归一化层学习到的`beta`和`gamma`。

```{.python .input  n=60}
net[1].beta.data().reshape((-1,)), net[1].gamma.data().reshape((-1,))
```

## 小结

* 批量归一化层对网络中间层的输出做归一化，使得深层网络学习时数值更加稳定。
* 批量归一化层和丢弃层一样，在训练模式和预测模式的计算结果是不一样的。

## 练习

* 尝试调大学习率，看看跟前面的LeNet比，是不是可以使用更大的学习率。
* 尝试将批量归一化层插入到LeNet的其他地方，看看效果如何，想一想为什么。
* 尝试下不学习`beta`和`gamma`（构造的时候加入这个参数`grad_req='null'`来避免计算梯度），看看效果会怎么样。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1253)

![](../img/qr_batch-norm.svg)

## 参考文献

[1] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.
