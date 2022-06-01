# 批量规范化
:label:`sec_batch_norm`

训练深层神经网络是十分困难的，特别是在较短的时间内使他们收敛更加棘手。
在本节中，我们将介绍*批量规范化*（batch normalization） :cite:`Ioffe.Szegedy.2015`，这是一种流行且有效的技术，可持续加速深层网络的收敛速度。
再结合在 :numref:`sec_resnet`中将介绍的残差块，批量规范化使得研究人员能够训练100层以上的网络。

## 训练深层网络

为什么需要批量规范化层呢？让我们来回顾一下训练神经网络时出现的一些实际挑战。

首先，数据预处理的方式通常会对最终结果产生巨大影响。
回想一下我们应用多层感知机来预测房价的例子（ :numref:`sec_kaggle_house`）。
使用真实数据时，我们的第一步是标准化输入特征，使其平均值为0，方差为1。
直观地说，这种标准化可以很好地与我们的优化器配合使用，因为它可以将参数的量级进行统一。

第二，对于典型的多层感知机或卷积神经网络。当我们训练时，中间层中的变量（例如，多层感知机中的仿射变换输出）可能具有更广的变化范围：不论是沿着从输入到输出的层，跨同一层中的单元，或是随着时间的推移，模型参数的随着训练更新变幻莫测。
批量规范化的发明者非正式地假设，这些变量分布中的这种偏移可能会阻碍网络的收敛。
直观地说，我们可能会猜想，如果一个层的可变值是另一层的100倍，这可能需要对学习率进行补偿调整。

第三，更深层的网络很复杂，容易过拟合。
这意味着正则化变得更加重要。

批量规范化应用于单个可选层（也可以应用到所有层），其原理如下：在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。
接下来，我们应用比例系数和比例偏移。
正是由于这个基于*批量*统计的*标准化*，才有了*批量规范化*的名称。

请注意，如果我们尝试使用大小为1的小批量应用批量规范化，我们将无法学到任何东西。
这是因为在减去均值之后，每个隐藏单元将为0。
所以，只有使用足够大的小批量，批量规范化这种方法才是有效且稳定的。
请注意，在应用批量规范化时，批量大小的选择可能比没有批量规范化时更重要。

从形式上来说，用$\mathbf{x} \in \mathcal{B}$表示一个来自小批量$\mathcal{B}$的输入，批量规范化$\mathrm{BN}$根据以下表达式转换$\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在 :eqref:`eq_batchnorm`中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$是小批量$\mathcal{B}$的样本均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$是小批量$\mathcal{B}$的样本标准差。
应用标准化后，生成的小批量的平均值为0和单位方差为1。
由于单位方差（与其他一些魔法数）是一个主观的选择，因此我们通常包含
*拉伸参数*（scale）$\boldsymbol{\gamma}$和*偏移参数*（shift）$\boldsymbol{\beta}$，它们的形状与$\mathbf{x}$相同。
请注意，$\boldsymbol{\gamma}$和$\boldsymbol{\beta}$是需要与其他模型参数一起学习的参数。

由于在训练过程中，中间层的变化幅度不能过于剧烈，而批量规范化将每一层主动居中，并将它们重新调整为给定的平均值和大小（通过$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$）。

从形式上来看，我们计算出 :eqref:`eq_batchnorm`中的$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$，如下所示：

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

请注意，我们在方差估计值中添加一个小的常量$\epsilon > 0$，以确保我们永远不会尝试除以零，即使在经验方差估计值可能消失的情况下也是如此。估计值$\hat{\boldsymbol{\mu}}_\mathcal{B}$和${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$通过使用平均值和方差的噪声（noise）估计来抵消缩放问题。
你可能会认为这种噪声是一个问题，而事实上它是有益的。

事实证明，这是深度学习中一个反复出现的主题。
由于尚未在理论上明确的原因，优化中的各种噪声源通常会导致更快的训练和较少的过拟合：这种变化似乎是正则化的一种形式。
在一些初步研究中， :cite:`Teye.Azizpour.Smith.2018`和 :cite:`Luo.Wang.Shao.ea.2018`分别将批量规范化的性质与贝叶斯先验相关联。
这些理论揭示了为什么批量规范化最适应$50 \sim 100$范围中的中等批量大小的难题。

另外，批量规范化层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。
在训练过程中，我们无法得知使用整个数据集来估计平均值和方差，所以只能根据每个小批次的平均值和方差不断训练模型。
而在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和方差。

现在，我们了解一下批量规范化在实践中是如何工作的。

## 批量规范化层

回想一下，批量规范化和其他层之间的一个关键区别是，由于批量规范化在完整的小批量上运行，因此我们不能像以前在引入其他层时那样忽略批量大小。
我们在下面讨论这两种情况：全连接层和卷积层，他们的批量规范化实现略有不同。

### 全连接层

通常，我们将批量规范化层置于全连接层中的仿射变换和激活函数之间。
设全连接层的输入为u，权重参数和偏置参数分别为$\mathbf{W}$和$\mathbf{b}$，激活函数为$\phi$，批量规范化的运算符为$\mathrm{BN}$。
那么，使用批量规范化的全连接层的输出的计算详情如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在应用变换的"相同"小批量上计算的。

### 卷积层

同样，对于卷积层，我们可以在卷积层之后和非线性激活函数之前应用批量规范化。
当卷积有多个输出通道时，我们需要对这些通道的“每个”输出执行批量规范化，每个通道都有自己的拉伸（scale）和偏移（shift）参数，这两个参数都是标量。
假设我们的小批量包含$m$个样本，并且对于每个通道，卷积的输出具有高度$p$和宽度$q$。
那么对于卷积层，我们在每个输出通道的$m \cdot p \cdot q$个元素上同时执行每个批量规范化。
因此，在计算平均值和方差时，我们会收集所有空间位置的值，然后在给定通道内应用相同的均值和方差，以便在每个空间位置对值进行规范化。

### 预测过程中的批量规范化

正如我们前面提到的，批量规范化在训练模式和预测模式下的行为通常不同。
首先，将训练好的模型用于预测时，我们不再需要样本均值中的噪声以及在微批次上估计每个小批次产生的样本方差了。
其次，例如，我们可能需要使用我们的模型对逐个样本进行预测。
一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。
可见，和暂退法一样，批量规范化层在训练模式和预测模式下的计算结果也是不一样的。

## (**从零实现**)

下面，我们从头开始实现一个具有张量的批量规范化层。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过autograd来判断当前模式是训练模式还是预测模式
    if not autograd.is_training():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / np.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # 计算移动方差元平方根的倒数
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 缩放和移位
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input  n=2}
#@tab paddle
from d2l import paddle as d2l
import paddle
import paddle.nn as nn

import warnings
warnings.filterwarnings("ignore", category=Warning) # 过滤报警信息

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum, is_training=True):
    # 训练模式还与预测模式的BN处理不同
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / (moving_var + eps) ** 0.5
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = paddle.mean(X)
            var = paddle.mean(((X - mean) ** 2))
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = paddle.mean(X, axis=(0, 2, 3), keepdim=True)
            var = paddle.mean(((X - mean) ** 2), axis=(0, 2, 3), keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / (var + eps) ** 0.5
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean, moving_var
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "grep: warning: GREP_OPTIONS is deprecated; please use an alias or script\n"
 }
]
```

我们现在可以[**创建一个正确的`BatchNorm`层**]。
这个层将保持适当的参数：拉伸`gamma`和偏移`beta`,这两个参数将在训练过程中更新。
此外，我们的层将保存均值和方差的移动平均值，以便在模型预测期间随后使用。

撇开算法细节，注意我们实现层的基础设计模式。
通常情况下，我们用一个单独的函数定义其数学原理，比如说`batch_norm`。
然后，我们将此功能集成到一个自定义层中，其代码主要处理数据移动到训练设备（如GPU）、分配和初始化任何必需的变量、跟踪移动平均线（此处为均值和方差）等问题。
为了方便起见，我们并不担心在这里自动推断输入形状，因此我们需要指定整个特征的数量。
不用担心，深度学习框架中的批量规范化API将为我们解决上述问题，我们稍后将展示这一点。

```{.python .input}
class BatchNorm(nn.Block):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 非模型参数的变量初始化为0和1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # 非模型参数的变量初始化为0和1
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input  n=3}
#@tab paddle
class BatchNorm(nn.Layer):
    def __init__(self, num_features, num_dims=4):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=shape, dtype='float32')))
        self.beta = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=shape, dtype='float32')))
        self.moving_mean = paddle.zeros(shape=shape, dtype='float32')
        self.moving_var = paddle.zeros(shape=shape, dtype='float32')

    def forward(self, X):
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9, is_training=self.training)
        return Y
```

##  使用批量规范化层的 LeNet

为了更好理解如何[**应用`BatchNorm`**]，下面我们将其应用(**于LeNet模型**)（ :numref:`sec_lenet`）。
回想一下，批量规范化是在卷积层或全连接层之后、相应的激活函数之前应用的。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# 回想一下，这个函数必须传递给d2l.train_ch6。
# 或者说为了利用我们现有的CPU/GPU设备，需要在strategy.scope()建立模型
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

```{.python .input  n=4}
#@tab paddle
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "W0601 03:17:10.207521 23215 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 10.2\nW0601 03:17:10.212774 23215 device_context.cc:465] device: 0, cuDNN Version: 7.6.\n"
 }
]
```

和以前一样，我们将[**在Fashion-MNIST数据集上训练网络**]。
这个代码与我们第一次训练LeNet（ :numref:`sec_lenet`）时几乎完全相同，主要区别在于学习率大得多。

```{.python .input  n=8}
#@tab mxnet, pytorch, paddle
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.249, train acc 0.907, test acc 0.843\n37677.6 examples/sec on CUDAPlace(0)\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"238.965625pt\" height=\"180.65625pt\" viewBox=\"0 0 238.965625 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-06-01T03:18:57.005589</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.1, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nL 0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path id=\"ma8a22b9a5b\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#ma8a22b9a5b\" x=\"51.803125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#ma8a22b9a5b\" x=\"95.203125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#ma8a22b9a5b\" x=\"138.603125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#ma8a22b9a5b\" x=\"182.003125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#ma8a22b9a5b\" x=\"225.403125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-70\" d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-70\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"186.181641\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"241.162109\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path d=\"M 30.103125 114.691522 \nL 225.403125 114.691522 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path id=\"m77f0a55e88\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m77f0a55e88\" x=\"30.103125\" y=\"114.691522\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 118.490741)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path d=\"M 30.103125 85.516384 \nL 225.403125 85.516384 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use xlink:href=\"#m77f0a55e88\" x=\"30.103125\" y=\"85.516384\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 89.315603)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path d=\"M 30.103125 56.341246 \nL 225.403125 56.341246 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use xlink:href=\"#m77f0a55e88\" x=\"30.103125\" y=\"56.341246\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 60.140465)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path d=\"M 30.103125 27.166108 \nL 225.403125 27.166108 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use xlink:href=\"#m77f0a55e88\" x=\"30.103125\" y=\"27.166108\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 30.965327)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path d=\"M 12.743125 13.377273 \nL 17.083125 48.745108 \nL 21.423125 65.082604 \nL 25.763125 75.279374 \nL 30.103125 81.920819 \nL 34.443125 114.089439 \nL 38.783125 115.176195 \nL 43.123125 115.876951 \nL 47.463125 116.604259 \nL 51.803125 117.160276 \nL 56.143125 119.628381 \nL 60.483125 120.898558 \nL 64.823125 121.558025 \nL 69.163125 122.485658 \nL 73.503125 122.705885 \nL 77.843125 127.462425 \nL 82.183125 127.452882 \nL 86.523125 126.911008 \nL 90.863125 126.767947 \nL 95.203125 127.03558 \nL 99.543125 130.310991 \nL 103.883125 129.853562 \nL 108.223125 129.354773 \nL 112.563125 129.513323 \nL 116.903125 129.177919 \nL 121.243125 131.90711 \nL 125.583125 131.241219 \nL 129.923125 131.347319 \nL 134.263125 131.540028 \nL 138.603125 131.349154 \nL 142.943125 133.546695 \nL 147.283125 133.888935 \nL 151.623125 132.762831 \nL 155.963125 132.934482 \nL 160.303125 132.77012 \nL 164.643125 134.975835 \nL 168.983125 135.378605 \nL 173.323125 135.025454 \nL 177.663125 134.41258 \nL 182.003125 134.25072 \nL 186.343125 134.380341 \nL 190.683125 135.62173 \nL 195.023125 136.378404 \nL 199.363125 135.846019 \nL 203.703125 135.690989 \nL 208.043125 135.477542 \nL 212.383125 136.040822 \nL 216.723125 136.825471 \nL 221.063125 136.922727 \nL 225.403125 136.718317 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path d=\"M 12.743125 82.439319 \nL 17.083125 70.630565 \nL 21.423125 65.34047 \nL 25.763125 61.801279 \nL 30.103125 59.416792 \nL 34.443125 48.552803 \nL 38.783125 48.079968 \nL 43.123125 47.768786 \nL 47.463125 47.491955 \nL 51.803125 47.330991 \nL 56.143125 46.515975 \nL 60.483125 46.049202 \nL 64.823125 46.031016 \nL 69.163125 45.612739 \nL 73.503125 45.51727 \nL 77.843125 43.242501 \nL 82.183125 43.551663 \nL 86.523125 43.81637 \nL 90.863125 43.906289 \nL 95.203125 43.842131 \nL 99.543125 42.84241 \nL 103.883125 42.93334 \nL 108.223125 43.028311 \nL 112.563125 43.124293 \nL 116.903125 43.251334 \nL 121.243125 41.799748 \nL 125.583125 42.181653 \nL 129.923125 42.369575 \nL 134.263125 42.41504 \nL 138.603125 42.597325 \nL 142.943125 41.908864 \nL 147.283125 41.62395 \nL 151.623125 41.993732 \nL 155.963125 41.936143 \nL 160.303125 42.079466 \nL 164.643125 41.157177 \nL 168.983125 41.181425 \nL 173.323125 41.242045 \nL 177.663125 41.475432 \nL 182.003125 41.515413 \nL 186.343125 40.987442 \nL 190.683125 40.884388 \nL 195.023125 40.781334 \nL 199.363125 40.938946 \nL 203.703125 40.951361 \nL 208.043125 41.254169 \nL 212.383125 41.023814 \nL 216.723125 40.757086 \nL 221.063125 40.699497 \nL 225.403125 40.790897 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path d=\"M 30.103125 61.344782 \nL 51.803125 54.167698 \nL 73.503125 51.673224 \nL 95.203125 48.172207 \nL 116.903125 49.922716 \nL 138.603125 49.77684 \nL 160.303125 47.67623 \nL 182.003125 48.245145 \nL 203.703125 48.040919 \nL 225.403125 50.039416 \n\" clip-path=\"url(#p1dc8ce060c)\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #008000; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 140.634375 98.667187 \nL 218.403125 98.667187 \nQ 220.403125 98.667187 220.403125 96.667187 \nL 220.403125 53.632812 \nQ 220.403125 51.632812 218.403125 51.632812 \nL 140.634375 51.632812 \nQ 138.634375 51.632812 138.634375 53.632812 \nL 138.634375 96.667187 \nQ 138.634375 98.667187 140.634375 98.667187 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 142.634375 59.73125 \nL 152.634375 59.73125 \nL 162.634375 59.73125 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_11\">\n     <!-- train loss -->\n     <g transform=\"translate(170.634375 63.23125)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6c\" d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"232.763672\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"264.550781\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"292.333984\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"353.515625\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"405.615234\"/>\n     </g>\n    </g>\n    <g id=\"line2d_23\">\n     <path d=\"M 142.634375 74.409375 \nL 152.634375 74.409375 \nL 162.634375 74.409375 \n\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_12\">\n     <!-- train acc -->\n     <g transform=\"translate(170.634375 77.909375)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"232.763672\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"264.550781\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"325.830078\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"380.810547\"/>\n     </g>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 142.634375 89.0875 \nL 152.634375 89.0875 \nL 162.634375 89.0875 \n\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #008000; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_13\">\n     <!-- test acc -->\n     <g transform=\"translate(170.634375 92.5875)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"100.732422\"/>\n      <use xlink:href=\"#DejaVuSans-74\" x=\"152.832031\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"192.041016\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"223.828125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"285.107422\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"340.087891\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p1dc8ce060c\">\n   <rect x=\"30.103125\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

让我们来看看从第一个批量规范化层中学到的[**拉伸参数`gamma`和偏移参数`beta`**]。

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

```{.python .input  n=9}
#@tab paddle
param = net.parameters()
print('gamma:', param[2].numpy().reshape(-1))
print('beta:', param[3].numpy().reshape(-1))
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "gamma: [1.3471212  2.10055    2.0498872  1.0266933  1.6769049  0.96736234]\nbeta: [-1.6160808  -0.05820943  1.1569028  -0.8798132  -1.6196461  -1.1543956 ]\n"
 }
]
```

## [**简明实现**]

除了使用我们刚刚定义的`BatchNorm`，我们也可以直接使用深度学习框架中定义的`BatchNorm`。
该代码看起来几乎与我们上面的代码相同。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

```{.python .input  n=10}
#@tab paddle
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), nn.BatchNorm2D(6, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), nn.BatchNorm2D(16, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), 
    nn.Linear(256, 120), nn.BatchNorm1D(120, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(120, 84), nn.BatchNorm1D(84, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

下面，我们[**使用相同超参数来训练模型**]。
请注意，通常高级API变体运行速度快得多，因为它的代码已编译为C++或CUDA，而我们的自定义代码由Python实现。

```{.python .input  n=11}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.245, train acc 0.912, test acc 0.891\n38306.5 examples/sec on CUDAPlace(0)\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"238.965625pt\" height=\"180.65625pt\" viewBox=\"0 0 238.965625 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-06-01T03:20:26.673172</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.1, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nL 0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path id=\"m5e49cce4a7\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m5e49cce4a7\" x=\"51.803125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m5e49cce4a7\" x=\"95.203125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#m5e49cce4a7\" x=\"138.603125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#m5e49cce4a7\" x=\"182.003125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#m5e49cce4a7\" x=\"225.403125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-70\" d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-70\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"186.181641\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"241.162109\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path d=\"M 30.103125 112.088563 \nL 225.403125 112.088563 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path id=\"mab6e8e3a0a\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mab6e8e3a0a\" x=\"30.103125\" y=\"112.088563\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 115.887782)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-2e\" d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-34\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path d=\"M 30.103125 80.931208 \nL 225.403125 80.931208 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use xlink:href=\"#mab6e8e3a0a\" x=\"30.103125\" y=\"80.931208\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 84.730427)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-36\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path d=\"M 30.103125 49.773853 \nL 225.403125 49.773853 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use xlink:href=\"#mab6e8e3a0a\" x=\"30.103125\" y=\"49.773853\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 53.573072)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-38\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path d=\"M 30.103125 18.616498 \nL 225.403125 18.616498 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use xlink:href=\"#mab6e8e3a0a\" x=\"30.103125\" y=\"18.616498\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 22.415717)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-2e\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"95.410156\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path d=\"M 12.743125 13.377273 \nL 17.083125 45.464514 \nL 21.423125 62.221476 \nL 25.763125 71.680584 \nL 30.103125 78.889233 \nL 34.443125 113.773868 \nL 38.783125 113.45321 \nL 43.123125 113.959056 \nL 47.463125 114.043368 \nL 51.803125 114.688512 \nL 56.143125 118.448341 \nL 60.483125 120.902969 \nL 64.823125 121.246899 \nL 69.163125 121.893933 \nL 73.503125 121.966277 \nL 77.843125 125.190725 \nL 82.183125 124.914274 \nL 86.523125 125.830741 \nL 90.863125 125.638355 \nL 95.203125 125.591893 \nL 99.543125 129.275015 \nL 103.883125 128.235807 \nL 108.223125 127.573904 \nL 112.563125 128.355219 \nL 116.903125 128.262213 \nL 121.243125 131.914699 \nL 125.583125 131.106997 \nL 129.923125 130.456302 \nL 134.263125 130.315699 \nL 138.603125 130.377212 \nL 142.943125 132.509731 \nL 147.283125 132.424435 \nL 151.623125 132.455633 \nL 155.963125 131.962194 \nL 160.303125 131.889131 \nL 164.643125 134.161827 \nL 168.983125 134.47822 \nL 173.323125 134.078634 \nL 177.663125 133.863903 \nL 182.003125 133.402772 \nL 186.343125 135.382815 \nL 190.683125 134.883477 \nL 195.023125 134.456815 \nL 199.363125 134.479196 \nL 203.703125 134.611545 \nL 208.043125 136.761152 \nL 212.383125 136.747069 \nL 216.723125 136.922727 \nL 221.063125 136.395264 \nL 225.403125 136.302997 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path d=\"M 12.743125 71.44313 \nL 17.083125 62.386211 \nL 21.423125 56.976229 \nL 25.763125 54.080259 \nL 30.103125 51.757538 \nL 34.443125 40.057896 \nL 38.783125 40.070844 \nL 43.123125 39.988841 \nL 47.463125 40.06437 \nL 51.803125 39.946304 \nL 56.143125 38.801969 \nL 60.483125 38.012159 \nL 64.823125 37.986263 \nL 69.163125 37.691703 \nL 73.503125 37.661431 \nL 77.843125 35.901683 \nL 82.183125 36.296588 \nL 86.523125 35.940526 \nL 90.863125 36.070003 \nL 95.203125 36.075003 \nL 99.543125 35.098925 \nL 103.883125 35.480883 \nL 108.223125 35.586622 \nL 112.563125 35.302852 \nL 116.903125 35.363576 \nL 121.243125 34.542174 \nL 125.583125 34.593965 \nL 129.923125 34.593965 \nL 134.263125 34.584254 \nL 138.603125 34.540503 \nL 142.943125 34.050161 \nL 147.283125 33.907737 \nL 151.623125 33.834366 \nL 155.963125 34.03074 \nL 160.303125 34.052371 \nL 164.643125 33.506358 \nL 168.983125 33.338038 \nL 173.323125 33.312142 \nL 177.663125 33.386591 \nL 182.003125 33.564239 \nL 186.343125 32.923711 \nL 190.683125 33.04024 \nL 195.023125 33.161086 \nL 199.363125 33.202087 \nL 203.703125 33.195544 \nL 208.043125 32.548228 \nL 212.383125 32.289273 \nL 216.723125 32.125269 \nL 221.063125 32.363723 \nL 225.403125 32.330927 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path d=\"M 30.103125 56.488263 \nL 51.803125 40.675905 \nL 73.503125 54.416299 \nL 95.203125 88.050664 \nL 116.903125 52.359913 \nL 138.603125 55.553542 \nL 160.303125 40.115073 \nL 182.003125 49.415543 \nL 203.703125 42.015672 \nL 225.403125 35.612835 \n\" clip-path=\"url(#pf9845c43dd)\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #008000; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 140.634375 98.667187 \nL 218.403125 98.667187 \nQ 220.403125 98.667187 220.403125 96.667187 \nL 220.403125 53.632812 \nQ 220.403125 51.632812 218.403125 51.632812 \nL 140.634375 51.632812 \nQ 138.634375 51.632812 138.634375 53.632812 \nL 138.634375 96.667187 \nQ 138.634375 98.667187 140.634375 98.667187 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 142.634375 59.73125 \nL 152.634375 59.73125 \nL 162.634375 59.73125 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_11\">\n     <!-- train loss -->\n     <g transform=\"translate(170.634375 63.23125)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6c\" d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"232.763672\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"264.550781\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"292.333984\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"353.515625\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"405.615234\"/>\n     </g>\n    </g>\n    <g id=\"line2d_23\">\n     <path d=\"M 142.634375 74.409375 \nL 152.634375 74.409375 \nL 162.634375 74.409375 \n\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_12\">\n     <!-- train acc -->\n     <g transform=\"translate(170.634375 77.909375)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"232.763672\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"264.550781\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"325.830078\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"380.810547\"/>\n     </g>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 142.634375 89.0875 \nL 152.634375 89.0875 \nL 162.634375 89.0875 \n\" style=\"fill: none; stroke-dasharray: 9.6,2.4,1.5,2.4; stroke-dashoffset: 0; stroke: #008000; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_13\">\n     <!-- test acc -->\n     <g transform=\"translate(170.634375 92.5875)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"100.732422\"/>\n      <use xlink:href=\"#DejaVuSans-74\" x=\"152.832031\"/>\n      <use xlink:href=\"#DejaVuSans-20\" x=\"192.041016\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"223.828125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"285.107422\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"340.087891\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pf9845c43dd\">\n   <rect x=\"30.103125\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## 争议

直观地说，批量规范化被认为可以使优化更加平滑。
然而，我们必须小心区分直觉和对我们观察到的现象的真实解释。
回想一下，我们甚至不知道简单的神经网络（多层感知机和传统的卷积神经网络）为什么如此有效。
即使在暂退法和权重衰减的情况下，它们仍然非常灵活，因此无法通过常规的学习理论泛化保证来解释它们是否能够泛化到看不见的数据。

在提出批量规范化的论文中，作者除了介绍了其应用，还解释了其原理：通过减少*内部协变量偏移*（internal covariate shift）。
据推测，作者所说的“内部协变量转移”类似于上述的投机直觉，即变量值的分布在训练过程中会发生变化。
然而，这种解释有两个问题：
1、这种偏移与严格定义的*协变量偏移*（covariate shift）非常不同，所以这个名字用词不当。
2、这种解释只提供了一种不明确的直觉，但留下了一个有待后续挖掘的问题：为什么这项技术如此有效？
本书旨在传达实践者用来发展深层神经网络的直觉。
然而，重要的是将这些指导性直觉与既定的科学事实区分开来。
最终，当你掌握了这些方法，并开始撰写自己的研究论文时，你会希望清楚地区分技术和直觉。

随着批量规范化的普及，“内部协变量偏移”的解释反复出现在技术文献的辩论，特别是关于“如何展示机器学习研究”的更广泛的讨论中。
Ali Rahimi在接受2017年NeurIPS大会的“接受时间考验奖”（Test of Time Award）时发表了一篇令人难忘的演讲。他将“内部协变量转移”作为焦点，将现代深度学习的实践比作炼金术。
他对该示例进行了详细回顾 :cite:`Lipton.Steinhardt.2018`，概述了机器学习中令人不安的趋势。
此外，一些作者对批量规范化的成功提出了另一种解释：在某些方面，批量规范化的表现出与原始论文 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`中声称的行为是相反的。

然而，与机器学习文献中成千上万类似模糊的说法相比，内部协变量偏移没有更值得批评。
很可能，它作为这些辩论的焦点而产生共鸣，要归功于目标受众对它的广泛认可。
批量规范化已经被证明是一种不可或缺的方法。它适用于几乎所有图像分类器，并在学术界获得了数万引用。

## 小结

* 在模型训练过程中，批量规范化利用小批量的均值和标准差，不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定。
* 批量规范化在全连接层和卷积层的使用略有不同。
* 批量规范化层和暂退层一样，在训练模式和预测模式下计算不同。
* 批量规范化有许多有益的副作用，主要是正则化。另一方面，”减少内部协变量偏移“的原始动机似乎不是一个有效的解释。

## 练习

1. 在使用批量规范化之前，我们是否可以从全连接层或卷积层中删除偏置参数？为什么？
1. 比较LeNet在使用和不使用批量规范化情况下的学习率。
    1. 绘制训练和测试准确度的提高。
    1. 你的学习率有多高？
1. 我们是否需要在每个层中进行批量规范化？尝试一下？
1. 你可以通过批量规范化来替换暂退法吗？行为会如何改变？
1. 确定参数`beta`和`gamma`，并观察和分析结果。
1. 查看高级API中有关`BatchNorm`的在线文档，以查看其他批量规范化的应用。
1. 研究思路：想想你可以应用的其他“规范化”转换？你可以应用概率积分变换吗？全秩协方差估计可以么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1876)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1874)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1875)
:end_tab:
