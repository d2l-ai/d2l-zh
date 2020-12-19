# 批量归一化
:label:`sec_batch_norm`

训练深层神经网络是十分困难的，特别是在较短的时间内使他们收敛更加棘手。
在本节中，我们将介绍 *批量归一化*（batch normalization）层 :cite:`Ioffe.Szegedy.2015` ，这是一种流行且有效的技术，可持续加速深层网络的收敛速度。
再结合后一节在 :numref:`sec_resnet` 中介绍的残差块，批量归一化使得研究人员能够训练拥有 100 多层的网络。

## 训练深层网络

为什么需要批量归一化层呢？让我们来回顾一下训练神经网络时出现的一些实际挑战。

首先，数据预处理的方式通常会对最终结果产生巨大影响。
回想一下我们应用 MLP 来预测房价（:numref:`sec_kaggle_house`）。
使用真实数据时，我们的第一步是标准化输入特征，使其平均值为0，方差为1。
直观地说，这种标准化可以很好地与我们的优化器配合使用，因为它将参数统一规模。

第二，对于典型的 MLP 或 CNN，当我们训练时，中间层中的变量（例如，MLP 中的仿射变换输出）可能具有广泛变化的大小：不论是沿着从输入到输出的层，跨同一层中的单元，或是随着时间的推移，模型参数的随着训练更新变幻莫测。
批量归一化的发明者非正式地假定，这些变量分布中的这种偏移可能会阻碍网络的收敛。
直观地说，我们可能会猜想，如果一个层的可变值是另一层的100倍，这可能需要对学习率进行补偿调整。






第二，对于典型的 MLP 或 CNN，当我们训练时，中间层中的变量（例如，MLP 中的仿射变换输出）可能具有广泛变化的大小：沿着从输入到输出的图层、同一层中的单位以及随着时间的推移参数。批量归一化的发明者非正式地假定，这些变量分布中的这种漂移可能会阻碍网络的收敛。直观而言，我们可能会猜测，如果一个图层的可变值是另一个图层的 100 倍，这可能需要对学习率进行补偿性调整。

第三，更深层的网络很复杂，容易过度拟合。这意味着正规化变得更加重要。

批处理规范化应用于单个图层（可选），其工作方式如下：在每次训练迭代中，我们首先通过减去其均值并除以其标准差来规范输入（批量归一化），其中两者均基于当前微型批处理。接下来，我们应用比例系数和比例偏移。正是由于这个基于 * 批次 * 统计数据的 * 规范化 *，* 批量归一化 * 派生它的名称。

请注意，如果我们尝试使用大小为 1 的微批次应用批量归一化，我们将无法学到任何东西。这是因为在减去均值之后，每个隐藏的单位将取值 0！正如你可能猜到的那样，由于我们用一整部分来进行批量归一化，使用足够大的微批次，所以这种方法证明是有效和稳定的。这里的一个回报是，在应用批量归一化时，批量大小的选择可能比没有批量归一化更重要。

从形式上来说，用 $\mathbf{x} \in \mathcal{B}$ 表示一个来自小批次 $\mathcal{B}$ 的批量归一化输入（$\mathrm{BN}$），批量归一化根据以下表达式转换 $\mathbf{x}$：

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

在 :eqref:`eq_batchnorm` 中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$ 是样本均值，$\hat{\boldsymbol{\mu}}_\mathcal{B}$ 是微型批次 $\mathcal{B}$ 的样本标准差。应用标准化后，生成的微型批次的平均值和单位方差为零。由于单位方差（与其他一些幻数）的选择是一个任意的选择，因此我们通常包含
*尺度参数 * 和 * 移位参数 *
它们的形状与 $\mathbf{x}$ 相同。请注意，$\boldsymbol{\gamma}$ 和 $\boldsymbol{\beta}$ 是需要与其他模型参数一起学习的参数。

因此，在训练过程中，中间层的可变幅度不能发生分歧，因为批量归一化主动居中并将它们重新调整为给定的平均值和大小（通过 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$）。从业人员的直觉或智慧之一是批量正常化似乎允许更积极的学习率。

从形式上来看，我们计算出 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$，如下所示：

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

请注意，我们在方差估计值中添加一个小常量 $\epsilon > 0$，以确保我们永远不会尝试除以零，即使在经验方差估计值可能消失的情况下也是如此。估计值 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ 通过使用平均值和方差的噪声估计来抵消缩放问题。你可能会认为这种噪音应该是一个问题。事实证明，这实际上是有益的。

事实证明，这是深度学习中一个反复出现的主题。由于理论上尚未明确表述的原因，优化中的各种噪声源通常会导致更快的训练和较少的过度拟合：这种变化似乎是正则化的一种形式。在一些初步研究中，:cite:`Teye.Azizpour.Smith.2018` 和 :cite:`Luo.Wang.Shao.ea.2018` 分别将批量归一化的性质与贝叶斯先验和处罚相关。特别是，这揭示了为什么批量归一化最适合 $50 \sim 100$ 范围中的中等小批量尺寸的难题。

修复经过训练的模型时，您可能会认为我们更愿意使用整个数据集来估计平均值和方差。训练完成后，为什么我们希望同一影像根据其恰好驻留的批次对其进行不同的分类？在训练过程中，这种精确计算是不可行的，因为每次我们更新模型时，所有数据示例的中间变量都会发生变化。但是，训练模型后，我们可以根据整个数据集计算每个图层变量的均值和方差。事实上，这是使用批量归一化的模型的标准做法，因此批量归一化图层在 * 训练模式 *（通过小批量统计数据归一化）和 * 预测模式 *（通过数据集统计归一化）中的功能不同。

我们现在已经准备好了解批量归一化在实践中是如何工作的。

## 批量归一化图层

完全连接层和卷积层的批量归一化实现略有不同。我们在下面讨论这两种情况。回想一下，批量归一化和其他图层之间的一个关键区别是，由于批量归一化一次在完整的小批次上运行，因此我们不能像以前在引入其他图层时那样忽略批处理尺寸。

### 完全连接的层

当将批量归一化应用于完全连接的层时，原纸在仿射变换后和非线性激活函数之前插入批量归一化（以后的应用程序可以在激活函数后立即插入批量归一化）:cite:`Ioffe.Szegedy.2015`。通过 $\mathbf{x}$ 表示完全连接层的输入，仿射变换 $\mathbf{W}\mathbf{x} + \mathbf{b}$（权重参数 $\mathbf{W}$ 和偏置参数 $\mathbf{b}$），激活函数为 $\phi$，我们可以表达一个批量归一化的完全连接层输出的计算详情如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在应用变换的 * 相同 * 微型批次上计算的。

### 卷积层

同样，对于卷积层，我们可以在卷积之后和非线性激活函数之前应用批量归一化。当卷积有多个输出通道时，我们需要对这些通道的输出的 * 每个 * 执行批量归一化，每个通道都有自己的比例和移位参数，这两个参数都是标量。假设我们的微型批次包含 $m$ 示例，并且对于每个通道，卷积的输出具有高度 $p$ 和宽度 $q$。对于卷积层，我们在每个输出通道的 $m \cdot p \cdot q$ 个元素上同时执行每个批量归一化。因此，在计算平均值和方差时，我们会收集所有空间位置的值，然后在给定通道内应用相同的均值和方差，以便在每个空间位置对值进行归一化。

### 预测过程中的批量归一化

正如我们前面提到的，批量归一化在训练模式和预测模式下的行为通常不同。首先，一旦我们对模型进行了训练，样本均值中的噪声以及在微型批次上估计每个小批次产生的样本方差就不再需要了。其次，我们可能没有计算每批规范化统计数据的奢侈品。例如，我们可能需要应用我们的模型来一次进行一个预测。

通常，在训练后，我们使用整个数据集来计算变量统计数据的稳定估计值，然后在预测时修复它们。因此，批量归一化在训练期间和测试时的行为不同。回想一下，辍学也表现出这一特征。

## 从头开始实施

下面，我们从头开始实现一个具有张量的批量归一化层。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过 `autograd` 来判断当前模式是训练模式还是预测模式
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
    # 通过 `is_grad_enabled` 来判断当前模式是训练模式还是预测模式
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

我们现在可以创建一个正确的 `BatchNorm` 图层。我们的层将保持适当的参数规模 `gamma` 和移位 `beta`, 这两个将在训练过程中更新.此外，我们的图层将保持均值和方差的移动平均值，以便在模型预测期间随后使用。

撇开算法细节，注意我们实现图层的基础设计模式。通常情况下，我们在一个单独的函数中定义数学，比如说 `batch_norm`。然后，我们将此功能集成到一个自定义层中，其代码主要处理簿记问题，例如将数据移动到正确的设备上下文、分配和初始化任何必需的变量、跟踪移动平均线（此处为均值和方差）等。这种模式使数学与样板代码完全分离。另请注意，为了方便起见，我们并不担心在这里自动推断输入形状，因此我们需要指定整个要素的数量。不用担心，深度学习框架中的高级批量规范化 API 将为我们解决这个问题，我们稍后将展示这一点。

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`：完全连接层的输出数量或卷积层的输出通道数。
    # `num_dims`：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.zeros(shape)

    def forward(self, X):
        # 如果 `X` 不在内存上，将 `moving_mean` 和 `moving_var` 
        # 复制到 `X` 所在显存上
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # 保存更新过的 `moving_mean` 和 `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`：完全连接层的输出数量或卷积层的输出通道数。
    # `num_dims`：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果 `X` 不在内存上，将 `moving_mean` 和 `moving_var` 
        # 复制到 `X` 所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的 `moving_mean` 和 `moving_var`
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
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.zeros,
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

## 在 Lenet 中应用批量归一化

为了了解如何在上下文中应用 `BatchNorm`，下面我们将其应用于传统的 Lenet 模型 (:numref:`sec_lenet`)。回想一下，批量归一化是在卷积层或完全连接的层之后，但在相应的激活函数之前应用的。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
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
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# 回想一下，这个函数必须传递给 `d2l.train_ch6`。
# 或者说为了利用我们现有的CPU/GPU设备，需要在`战略范围（）`建立模型
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
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

和以前一样，我们将在时尚 MNist 数据集训练我们的网络。这个代码与我们第一次训练 Lenet（:numref:`sec_lenet`）时几乎完全相同。主要区别在于学习率大得多。

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

让我们来看看尺度参数 `gamma` 和从第一个批量归一化层中学到的移位参数 `beta`。

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

## 简明实施

与我们刚刚定义的 `BatchNorm` 类相比，我们可以直接使用深度学习框架中的高级 API 中定义的 `BatchNorm` 类。该代码看起来几乎与我们上面的应用程序相同。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
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
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
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
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
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

下面，我们使用相同的超参数来训练我们的模型。请注意，像往常一样，高级 API 变体运行速度快得多，因为它的代码已编译为 C ++ 或 CUDA，而我们的自定义实现必须由 Python 解释。

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 争议

直观地说，批量规范化被认为可以使优化环境更加平滑。然而，我们必须小心区分投机直觉和真实的解释，我们观察到的现象，当训练深层模型。回想一下，我们甚至不知道为什么简单的深度神经网络（MLP 和传统的有线电视网络）概括得很好。即使在衰减和权重衰减的情况下，它们仍然非常灵活，因此无法通过传统的学习理论泛化保证来解释它们是否能够概括到看不见的数据。

在提出批量规范化的原始文件中，作者除了引入一个强大而有用的工具之外，还解释了它的工作原因：通过减少 * 内部协变量偏移 *。据推测，通过 * 内部协变量转移 *，作者意味着类似于上面表达的直觉-变量值的分布在训练过程中发生变化的概念。然而，这个解释有两个问题：i) 这个漂移与 * 协变量移 * 非常不同，使名称成为一个错误的名称。ii) 解释提供了一个未指定的直觉，但留下了 * 为什么这种技术正确工作 * 一个未决问题，希望得到严格解释。在本书中，我们旨在传达从业者用来指导他们深度神经网络发展的直觉。然而，我们认为，重要的是将这些指导性直觉与既定的科学事实分开。最终，当你掌握这些材料并开始撰写自己的研究论文时，你会希望清楚地区分技术声明和预感。

随着批量规范化的成功，它在 * 内部协变量移 * 方面的解释反复出现在技术文献中的辩论和关于如何展示机器学习研究的更广泛的讨论中。在 2017 年 Neurips 会议上接受时间考验奖时，Ali Rahimi 在一次令人难忘的演讲中，将深度学习的现代实践比作炼金术的论点中使用了 * 内部协变量转移 * 作为焦点。随后，在一份概述机器学习 :cite:`Lipton.Steinhardt.2018` 令人不安的趋势的立场文件中对该示例进行了详细讨论。其他作者提出了批量规范化成功的替代解释，有些人声称批量规范化的成功仍然取得成功，尽管表现出的行为在某些方面与原文 :cite:`Santurkar.Tsipras.Ilyas.ea.2018` 中声称的行为相反。

我们注意到，* 内部协变量偏移 * 与技术机器学习文献中每年提出的成千上万类似模糊的声明相比，没有更值得批评。很可能，它作为这些辩论的焦点而产生共鸣，因为它对目标受众的广泛认可。批量标准化已经证明是一种不可缺少的方法，适用于几乎所有部署的图像分类器，赢得了介绍该技术数万引用的论文。

## 小结

* 在模型训练过程中，批量归一化通过利用微粒的均值和标准差，连续调整神经网络的中间输出，从而使神经网络中每个层的中间输出值更加稳定。
* 完全连接图层和卷积图层的批量归一化方法略有不同。
* 与压差图层一样，批量归一化图层在训练模式和预测模式下具有不同的计算结果。
* 批量归一化有许多有益的副作用，主要是正则化。另一方面，减少内部协变量偏移的原始动机似乎不是一个有效的解释。

## 练习

1. 我们是否可以在批量归一化之前从完全连接的层或卷积层中删除偏置参数？为什么？
1. 比较 LenNet 的学习速率，以及没有批量规范化的情况。
    1. 绘制训练和测试准确度的提高。
    1. 你可以做多大的学习率？
1. 我们是否需要在每个层中进行批量归一化？试验它？
1. 你可以通过批量规范化来替换辍学吗？行为如何改变？
1. 修复参数 `beta` 和 `gamma`，并观察和分析结果。
1. 查看高级 API 中有关 `BatchNorm` 的在线文档，以查看其他批量规范化应用程序。
1. 研究想法：想想你可以应用的其他规范化转换？你可以应用概率积分变换吗？完整排名协方差估计如何？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
