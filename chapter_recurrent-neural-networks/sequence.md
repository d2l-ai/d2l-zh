# 序列模型
:label:`sec_sequence`

想象一下，您正在观看 Netflix 电影。作为一个好的 Netflix 用户，您决定对每部电影进行宗教评价。毕竟，一部好电影是一部好电影，你想看更多的电影，对不对？事实证明，事情并不那么简单。随着时间的推移，人们对电影的看法可能会发生相当明显的事实上，心理学家甚至有一些效果的名字：

* 根据别人的意见，有 * 锚定 *。例如，奥斯卡奖后，相应电影的收视率上升，尽管它仍然是同一部电影。这种效果持续了几个月，直到该奖项被遗忘。它已经表明，效果提升了超过半点的评级
:cite:`Wu.Ahmed.Beutel.ea.2017`.
* 有 * 享乐适应 *，人类迅速适应，接受改善或恶化的情况作为新常态。例如，看了很多好电影之后，对下一部电影同样好或更好的期望也很高。因此，在许多伟大的电影被观看之后，即使是一个普通的电影也可能被认为是坏的。
* 有 * 季节性 *。很少有观众喜欢看圣诞老人电影在八月。
* 在某些情况下，由于导演或演员在制作中的不当行为，电影变得不受欢迎。
* 一些电影成为邪教电影，因为他们几乎是滑稽的坏。* 计划 9 从外太空 * 和 * 巨魔 2* 实现了很高程度的臭名昭著。

简而言之，电影评级只是静止的。因此，使用时间动力学导致更准确的电影建议 :cite:`Koren.2009`。当然，序列数据不仅仅是电影评级。下面给出了更多插图。

* 许多用户在打开应用时具有非常特殊的行为。例如，社交媒体应用程序在放学后更受学生欢迎。当市场开放时，股票市场交易应用程序更常用。
* 要预测明天的股票价格比填写我们昨天错过的股票价格的空白要困难得多，尽管两者都只是估计一个数字的问题。毕竟，远见是这么多比事后看来更难。在统计数据中，前者（超出已知观测值的预测值）称为 * 外推 *，而后者（在现有观测值之间进行估计）称为 * 插值 *。
* 音乐、语音、文字和视频本质上都是顺序的。如果我们把它们放在一起，他们就没有什么意义。标题 * 狗咬人 * 比 * 人咬狗少得多令人惊讶，即使这些词是相同的。
* 地震是密切相关的，也就是说，在大地震之后，很可能会出现几个较小的余震，而不是没有强地震的情况。事实上，地震在时空上是相关的，也就是说，余震通常在短时间内发生，并且在接近地区。
* 在 Twitter 的战斗、舞蹈模式和辩论中，人类以顺序的方式相互交流。

## 统计工具

我们需要统计工具和新的深度神经网络架构来处理序列数据。为了保持简单，我们以 :numref:`fig_ftse100` 中所示的股票价格（富时 100 指数）为例。

![FTSE 100 index over about 30 years.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

让我们用 $x_t$ 表示价格，也就是说，在 * 时间步骤 * $t \in \mathbb{Z}^+$ 时，我们观察价格 $x_t$。请注意，对于此文本中的序列，$t$ 通常是离散的，并且因整数或其子集而异。假设一个想要在股票市场上做得很好的交易者在 $t$ 日预测 $x_t$ 通过

$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1).$$

### 自回归模型

为了实现这一点，我们的交易者可以使用一个回归模型，如我们在 :numref:`sec_linear_concise` 中训练的模型。只有一个主要问题：输入数量，$x_{t-1}, \ldots, x_1$ 不同，具体取决于 $t$。也就是说，数量随着我们遇到的数据量而增加，我们需要一个近似值来使这个计算可追踪。本章接下来的许多内容将围绕如何有效地估计 $P(x_t \mid x_{t-1}, \ldots, x_1)$。简而言之，它归结为两个策略，如下所示。

首先，假设可能相当长的序列 $x_{t-1}, \ldots, x_1$ 并不是真正必要的。在这种情况下，我们可能会满足于一些长度为 $\tau$ 的时间跨度，并且只使用 $x_{t-1}, \ldots, x_{t-\tau}$ 观测值。直接的好处是，现在的参数数量始终是相同的，至少对于 $t > \tau$。这使我们能够训练一个深层网络，如上所示。这样的模型将被称为 * 自回归模型 *，因为它们完全从字面上执行回归。

第二个策略，在 :numref:`fig_sequence-model` 中显示，是保持过去的观测值的一些总结 $h_t$，并在同一时间更新 $h_t$，除了预测 $\hat{x}_t$。这导致模型的估计值为 $x_t$，并进一步更新了表格 $h_t = g(h_{t-1}, x_{t-1})$。由于从未观察到 $h_t$，因此这些模型也称为 * 潜在自回归模型 *。

![A latent autoregressive model.](../img/sequence-model.svg)
:label:`fig_sequence-model`

这两种情况都提出了如何生成训练数据的明显问题。通常使用历史观测值来预测到目前为止的下一个观测值。显然，我们不指望时间停滞不前。但是，一个常见的假设是，虽然 $x_t$ 的特定值可能会改变，但至少序列本身的动态不会发生变化。这是合理的，因为新的动态就是这样，新颖的，因此使用我们迄今为止所拥有的数据是不可预测的。统计学家调用不改变 * 固定 * 的动态。无论我们做什么，我们都会通过

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

请注意，如果我们处理离散对象（如单词）而不是连续数字，则上述注意事项仍然适用。唯一的区别是，在这种情况下，我们需要使用分类器而不是回归模型来估计 $P(x_t \mid  x_{t-1}, \ldots, x_1)$。

### 马尔科夫模型

回想一下，在自回归模型中，我们只使用 $x_{t-1}, \ldots, x_{t-\tau}$ 而不是 $x_{t-1}, \ldots, x_1$ 来估计 $x_t$ 的近似值。只要这个近似值是准确的，我们就会说序列满足 *Markov 条件 *。特别是，如果 $\tau = 1$，我们有一个 * 一阶马尔科夫模型 *，$P(x)$ 由

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ where } P(x_1 \mid x_0) = P(x_1).$$

每当 $x_t$ 只假设离散值时，这样的模型就特别好，因为在这种情况下，动态编程可以用于精确计算链上的值。例如，我们可以有效地计算 $P(x_{t+1} \mid x_{t-1})$：

$$\begin{aligned}
P(x_{t+1} \mid x_{t-1}) 
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}
$$

通过使用这样一个事实，即我们只需要考虑到过去的观察历史很短：$P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$。详细介绍动态编程超出了本节的范围。控制和强化学习算法广泛使用这些工具。

### 因果关系

原则上，以相反的顺序展开 $P(x_1, \ldots, x_T)$ 没有什么问题。毕竟，通过调理，我们总是可以通过

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

事实上，如果我们有一个 Markov 模型，我们也可以得到反向条件概率分布。然而，在许多情况下，数据有一个自然的方向，即时间前进。显然，未来的事件不能影响过去。因此，如果我们更改 $x_t$，我们或许能够影响 $x_{t+1}$ 的发展，但不能影响相反。也就是说，如果我们改变 $x_t$，过去事件的分布将不会改变。因此，应该更容易解释 $P(x_{t+1} \mid x_t)$，而不是解释 $P(x_t \mid x_{t+1})$。例如，已经表明，在某些情况下，我们可以找到 $x_{t+1} = f(x_t) + \epsilon$，用于某些添加剂噪声 $\epsilon$，而反之则不是真实的 :cite:`Hoyer.Janzing.Mooij.ea.2009`。这是一个好消息，因为它通常是我们有兴趣估计的前进方向。这本书由彼得斯等人. 已经解释了更多关于这个主题 :cite:`Peters.Janzing.Scholkopf.2017`.我们几乎没有划伤它的表面。

## 培训

在回顾了这么多的统计工具之后，让我们在实践中尝试一下。我们从生成一些数据开始。为了保持简单，我们通过使用带有一些加法噪声的正弦函数来生成序列数据，用于时间步长 $1, 2, \ldots, 1000$。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

```{.python .input}
#@tab tensorflow
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal([T], 0, 0.2)
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

接下来，我们需要将这样的序列转换为模型可以训练的要素和标注。基于嵌入维度 $\tau$，我们将数据映射成对。精明的读者可能已经注意到，这给了我们 $\tau$ 更少的数据示例，因为我们没有足够的历史记录来处理其中的第一个 $\tau$。一个简单的解决方法，特别是如果序列很长，就是丢弃这些几个术语。或者，我们可以用零填充序列。在这里，我们仅使用前 600 个要素标签对进行训练。

```{.python .input}
#@tab mxnet, pytorch
tau = 4
features = d2l.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab tensorflow
tau = 4
features = tf.Variable(d2l.zeros((T - tau, tau)))
for i in range(tau):
    features[:, i].assign(x[i: T - tau + i])
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab all
batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

在这里，我们保持架构相当简单：只需一个带有两个完全连接层的 MLP，即 RELU 激活和方损。

```{.python .input}
# A simple MLP
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Square loss
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Square loss
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
# Vanilla MLP architecture
def get_net():
    net = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                              tf.keras.layers.Dense(1)])
    return net

# Least mean squares loss
# Note: L2 Loss = 1/2 * MSE Loss. TensorFlow has MSE Loss that is slightly
# different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
# value to get L2Loss in TF
loss = tf.keras.losses.MeanSquaredError()
```

现在，我们已经准备好训练模型了。下面的代码与前面几节中的训练循环基本相同，例如 :numref:`sec_linear_concise`。因此，我们不会深入探讨太多细节。

```{.python .input}
def train(net, train_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab tensorflow
def train(net, train_iter, loss, epochs, lr):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out) / 2
                params = net.trainable_variables
                grads = g.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

## 预测

由于训练损失很小，我们希望我们的模型能够很好地运行。让我们看看这在实践中意味着什么。首先要检查的是模型能够预测下一个时间步骤中发生的情况，即 * 提前一步预测 *。

```{.python .input}
#@tab all
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [d2l.numpy(x), d2l.numpy(onestep_preds)], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
```

一步前进的预测看起来不错，正如我们预期的那样。即使超过 604 (`n_train + tau`) 的观察结果，预测仍然值得信赖。但是，这只有一个小问题：如果我们只观察到时间步长 604 之前的序列数据，我们不能希望收到所有未来提前一步预测的输入。相反，我们需要一次努力向前迈进一步：

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

通常，对于一个最高达 $x_t$ 的观测序列，其在时间步长 $t+k$ 时的预测输出值称为 *$k$ 提前步进预测 *。由于我们已经观察到了高达 $x_{604}$，因此其提前 $k$ 个步进预测是 $\hat{x}_{604+k}$。换句话说，我们将不得不使用我们自己的预测来做出多步预测。让我们看看这种情况有多好。

```{.python .input}
#@tab mxnet, pytorch
multistep_preds = d2l.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = d2l.reshape(net(
        multistep_preds[i - tau: i].reshape(1, -1)), 1)
```

```{.python .input}
#@tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(T))
multistep_preds[:n_train + tau].assign(x[:n_train + tau])
for i in range(n_train + tau, T):
    multistep_preds[i].assign(d2l.reshape(net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1))), ()))
```

```{.python .input}
#@tab all
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [d2l.numpy(x), d2l.numpy(onestep_preds),
          d2l.numpy(multistep_preds[n_train + tau:])], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

如上面的例子所示，这是一个惊人的失败。在几个预测步骤之后，预测衰减非常快。为什么算法工作如此糟糕？这最终是由于错误积累的事实。让我们说，在步骤 1 之后，我们有一些错误 $\epsilon_1 = \bar\epsilon$。现在，步骤 2 的 * 输入 * 被 $\epsilon_1$ 扰动，因此我们在某些常数 $c$ 的顺序上遇到了一些错误，依此类推。误差可以相当迅速地偏离真实的观察结果。这是一个常见的现象。例如，未来 24 小时的天气预报往往非常准确，但除此之外，准确性会迅速下降。我们将在本章及其后讨论改进这一点的方法。

让我们通过计算 $k = 1, 4, 16, 64$ 整个序列的预测数据，仔细看看 $k$ 提前预测中的困难。

```{.python .input}
#@tab all
max_steps = 64
```

```{.python .input}
#@tab mxnet, pytorch
features = d2l.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1].T

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = d2l.reshape(net(features[:, i - tau: i]), -1)
```

```{.python .input}
#@tab tensorflow
features = tf.Variable(d2l.zeros((T - tau - max_steps + 1, tau + max_steps)))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i].assign(x[i: i + T - tau - max_steps + 1].numpy().T)

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i].assign(d2l.reshape(net((features[:, i - tau: i])), -1))
```

```{.python .input}
#@tab all
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [d2l.numpy(features[:, tau + i - 1]) for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

这清楚地说明了预测质量如何在我们尝试进一步预测未来时发生变化。虽然提前 4 步的预测仍然看起来不错，但除此之外的任何东西几乎没有用处。

## 摘要

* 插值法和外推法之间的难度差别很大。因此，如果你有一个序列，在训练时始终尊重数据的时间顺序，即永远不要在未来的数据上进行训练。
* 序列模型需要专门的统计工具进行估计。两种常见的选择是自回归模型和延迟变量自回归模型。
* 对于因果模型（例如，前进时间），估计前进方向通常比反向方向容易得多。
* 对于直至时间步长 $t$ 的观测序列，其在时间步长 $t+k$ 的预测输出为 *$k$ 提前步进预测 *。当我们通过增加 $k$ 在时间上进一步预测时，误差累积，预测质量通常会显著降低。

## 练习

1. 在本节的实验中改进模型。
    1. 纳入比过去 4 个观察结果更多？你真的需要多少？
    1. 如果没有噪音，您需要多少过去的观察结果？提示：你可以写入 $\sin$ 和 $\cos$ 作为一个微分方程。
    1. 是否可以在保持要素总数不变的同时合并较旧的观测值？这是否提高了准确性？为什么？
    1. 更改神经网络架构并评估性能。
1. 投资者想要找到一个良好的安全性购买。他审视过去的回报，以决定哪一个可能做得很好。这个策略有什么可能出错？
1. 因果关系是否也适用于文本？在何种程度上？
1. 举一个示例，说明何时可能需要潜在的自回归模型来捕获数据的动态。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:
