# 生成对抗网络
:label:`sec_basic_gan`

在本书的大部分时间里，都在讨论如何进行预测。以某种形式，使用深度神经网络学习从数据样本到标签的映射。这种学习被称为判别学习，比如，期望能够区分猫和狗的照片。分类器和回归器都是判别学习的例子。通过反向传播训练的神经网络颠覆了我们对大型复杂数据集的判别学习的所有认知。在短短 5-6 年的时间里，高分辨率图像的分类精度已经从无用到人类水平（有一些警示）。这里就不赘述深层神经网络做得非常出色的其他所有判别性任务了。

但机器学习不仅仅局限于解决判别任务。例如，给定没有任何标签的大型数据集，可能希望学习能够精确捕获该数据特征的模型。有了这样的模型，就可以对类似于训练数据分布的合成数据样本进行采样。例如，给定大的人脸照片库，可能希望能够生成新的逼真的图像，看起来似乎来自相同的数据集。这种学习被称为生成式建模。

直到最近，还没有办法合成出逼真的新图像。但是，深度神经网络在判别学习方面的成功带来了新的可能性。在过去的三年里，有一个很大的趋势是应用判别式网络来克服通常不认为是监督学习问题的挑战。循环神经网络语言模型是使用判别网络（经过训练来预测下一个字符）的一个例子，一旦经过训练，它就可以作为生成模型。

2014 年，一篇突破性的论文引入了生成式对抗网络（Generative adversarial network，简称 GANs） :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`，这是一种巧妙的新方法，可以利用判别模型的力量来获得好的生成模型。GANs 的核心思想是，如果我们不能区分假数据和真实数据，那么数据生成器就是好的。在统计学中，这被称为双样本检验——回答数据集  $X=\{x_1,\ldots, x_n\}$ 和 $X'=\{x'_1,\ldots, x'_n\}$ 是否来自同一个分布的测试。大多数统计论文和 GANs 之间的主要区别是后者以一种建设性的方式使用了这个想法。换句话说，他们不是仅仅训练模型说“嘿，这两个数据集看起来不像来自同一个分布”，而是使用 [两个样本测试](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) 为生成模型提供训练信号。这允许我们改进数据生成器，直到它生成一些类似于真实数据的东西。至少，它需要愚弄分类器。即使我们的分类器是最先进的深度神经网络。

![Generative Adversarial Networks](../img/gan.svg)
:label:`fig_gan`

GAN 架构的描述如下 :numref:`fig_gan`。
正如你所看到的，在 GAN 架构中有两个部分——首先，我们需要一个设备（比如，深层网络，但实际上它可以是任何东西，如游戏渲染引擎），它可能能够生成看起来像真实的东西一样的数据。如果我们处理的是图像，这就需要生成图像。如果我们处理的是语音，它需要生成音频序列，等等。我们称之为生成网络。第二部分是判别网络。它试图区分真实数据和虚假数据。这两个网络都在互相竞争。生成网络试图欺骗判别网络。这时，判别网络就会适应新的假数据。这些信息，反过来又被用来改进生成网络，等等。

判别器是二进制分类器，用于区分输入 $x$ 是真实的（来自真实数据）还是假的（来自生成器）。通常，对于输入 $\mathbf x$，判别器输出标量预测 $o\in\mathbb R$，例如使用隐藏层大小为 1 的 dense 层，然后应用 sigmoid 函数获得预测概率 $D(\mathbf x) = 1/(1+e^{-o})$。假设真实数据的标签 $y$ 是 $1$，假数据的标签 $0$。我们训练判别器使交叉熵损失最小，*即*，

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

对于生成器，它首先从随机性的来源中提取一些参数 $\mathbf z\in\mathbb R^d$，*例如*，正态分布 $\mathbf z \sim \mathcal{N} (0, 1)$。我们通常称 $\mathbf z$ 为隐变量。
然后应用函数来生成 $\mathbf x'=G(\mathbf z)$。生成器的目标是欺骗判别器，将 $\mathbf x'=G(\mathbf z)$ 分类为真实数据，*即*，我们想要 $D( G(\mathbf z)) \approx 1$。
换句话说，对于给定的判别器 $D$，我们更新生成器 $G$ 的参数，使 $y=0$ 时的交叉熵损失最大化，*即*，

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

如果生成器完成了完美的工作，那么 $D(\mathbf x')\approx 1$，所以上面的损失接近于 0，这导致梯度太小，不能使判别器取得良好的进展。所以我们通常会尽量最小化以下损失：

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

它只是将 $\mathbf x'=G(\mathbf z)$ 输入到判别器，但给出标签 $y=1$。

综上所述，$D$ 和 $G$ 正在进行具有综合目标函数的“极大极小”博弈：

$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$

许多GANs应用程序都位于图像环境中。作为演示目的，我们将首先满足于拟合简单得多的分布。我们将说明如果我们使用 GANs 来为高斯分布建立世界上最低效的参数估计会发生什么。让我们开始吧。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 生成一些“真实”数据

因为这将是世界上最蹩脚的例子，我们只是简单地从高斯函数中生成数据。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

看看我们得到了什么。这应该是高斯函数，以一种相当随意的方式移位均值 $b$ 和协方差矩阵 $A^TA$。

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## 生成器

我们的生成网络将是最简单的网络——单层线性模型。这是因为我们将使用高斯数据发生器来驱动线性网络。因此，它只需要学习参数就可以完美地伪造东西。

```{.python .input}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## 判别器

对于判别器，我们将更具有鉴别性：我们将使用一个带有 3 层的 MLP，使事情变得更有趣一些。

```{.python .input}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## 训练

首先，我们定义函数来更新判别器。

```{.python .input}
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) + 
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # Labels corresponding to real data
    zeros = tf.zeros((batch_size,)) # Labels corresponding to fake data
    # Do not need to compute gradient for `net_G`, so it's outside GradientTape
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogitsLoss
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

The generator is updated similarly. Here we reuse the cross-entropy loss but change the label of the fake data from $0$ to $1$.

```{.python .input}
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogits loss
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

判别器和生成器都进行了具有交叉熵损失的二元逻辑回归。我们用 Adam  来平滑训练过程。在每次迭代中，我们首先更新判别器，然后更新生成器。我们将损失和生成的样本可视化。

```{.python .input}
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # Visualize generated examples
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])
        
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
        
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

现在我们指定超参数来拟合高斯分布。

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## 小结

* 生成对抗网络由生成器和判别器两个深度网络组成。
* 生成器通过最大化交叉熵损失，生成尽可能接近真实图像的图像来欺骗判别器，即 $\max \log(D(\mathbf{x'}))$。
* 判别器通过最小化交叉熵损失，将生成的图像与真实的图像区分开来，即 $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$。

## 练习

* 是否存在生成获胜的均衡，即鉴别器最终无法在有限样本上区分这两个分布？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/408)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1082)
:end_tab:
