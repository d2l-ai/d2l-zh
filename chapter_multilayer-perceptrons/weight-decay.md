# 权重衰减
:label:`sec_weight_decay`

现在我们已经对过拟合问题进行了特征，我们可以引入一些标准的模型规则化技术。回想一下，我们总是可以通过外出并收集更多的训练数据来减轻过拟合。这可能是昂贵的，耗时的，或者完全是我们无法控制的，因此在短期内不可能做到这一点。目前，我们可以假设我们已经拥有尽可能多的高质量数据，而且专注于正则化技术。

回想一下，在我们的多项式回归样本 (:numref:`sec_model_selection`) 中，我们可以通过调整拟合多项式的程度来限制模型的容量。实际上，限制要素数量是缓解过拟合的一种常用技术。但是，简单地抛出功能可能对工作来说太钝了。坚持多项式回归样本，考虑高维输入可能会发生什么情况。多项式对多变量数据的自然扩展称为 * 单词 *，它们仅仅是变量权力的产物。单项的程度是权力的总和。样本，$x_1^2 x_2$ 和 $x_3 x_5^2$ 都是三度的单一体式。

请注意，随着 $d$ 度的术语数量增加，$d$ 度的术语数量会迅速爆炸。鉴于 $k$ 个变量，单一体化的数量为 $d$ 度（即多选 $k$）。从 $2$ 到 $3$，即使程度上的微小变化也大大增加了我们模型的复杂性。因此，我们通常需要一个更精细的工具来调整函数复杂性。

## 规范和体重衰减

我们已经描述了 $L_2$ 规范和 $L_1$ 规范，这些规范是 :numref:`subsec_lin-algebra-norms` 中较为普遍的 $L_p$ 规范的特殊情况。
*重量减少 *（通常称为 $L_2$ 正则化），
可能是用于规则化参数机器学习模型的最广泛使用的技术。该技术的动机是基本的直觉，即在所有函数 $f$ 中，函数 $f = 0$（将值 $0$ 分配给所有输入）在某种意义上是 * 最简单 *，我们可以通过从零的距离来测量函数的复杂性。但是，我们应该如何精确地测量函数和零之间的距离？没有一个正确的答案。事实上，整个数学分支，包括部分功能分析和 Banach 空间理论，都致力于回答这个问题。

一个简单的解释可能是通过其权重向量的某个范数（例如 $\| \mathbf{w} \|^2$）来测量线性函数 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 的复杂性。确保小权重矢量的最常见方法是将其标准作为处罚术语加入最小化损失的问题。因此, 我们取代了我们最初的目标,
*最大限度地减少训练标签上的预测损失 *，
有新的目标,
*最大限度地减少预测损失和处罚条款的总和 *。
现在，如果我们的权重向量变得太大，我们的学习算法可能会侧重于最小化权重范数 $\| \mathbf{w} \|^2$，而最大限度地减少训练误差。这正是我们想要的。为了在代码中说明事情，让我们恢复前面的样本，从 :numref:`sec_linear_regression` 进行线性回归。在那里，我们的损失是由

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

回想一下，$\mathbf{x}^{(i)}$ 是要素，$y^{(i)}$ 是所有数据点的标签 $i$，$(\mathbf{w}, b)$ 分别是权重和偏差参数。为了惩罚权重向量的大小，我们必须以某种方式将 $\| \mathbf{w} \|^2$ 添加到损失函数中，但模型应该如何权衡这个新的加法惩罚的标准损耗？在实践中，我们通过 * 正则化常量 * $\lambda$ 来表征这种权衡，这是我们使用验证数据拟合的非负超参数：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

对于 $\lambda = 0$，我们恢复原来的损失函数。对于 $\lambda > 0$，我们限制了 $\| \mathbf{w} \|$ 的大小。我们按照惯例除以 $2$：当我们采取二次函数的导数时，$2$ 和 $1/2$ 取消，确保更新的表达式看起来很好和简单。精明的读者可能会想知道为什么我们使用平方范数而不是标准范数（即欧几里得距离）。我们这样做是为了方便计算。通过平方 $L_2$ 范数，我们删除平方根，留下权重向量每个分量的平方和。这使得罚款的衍生物易于计算：衍生物的总和等于总和的衍生物。

此外，你可能会问我们为什么首先使用 $L_2$ 规范，而不是说 $L_1$ 规范。事实上，其他选择在整个统计数据中是有效的，而且很受欢迎。$L_2$ 正则化线性模型构成经典的 * 脊回归 * 算法，$L_1$ 正则化线性回归是统计学中类似的基本模型，通常称为 * 套索回归 *。

使用 $L_2$ 规范的一个原因是，它会对权重向量的大分量施加大小惩罚。这使我们的学习算法偏向于在更多要素中均匀分配权重的模型。在实践中，这可能会使它们在单个变量中对测量误差更加稳定。相比之下，$L_1$ 处罚会导致模型通过将其他权重清除为零，将权重集中在一小组要素上。这称为 * 功能选择 *，由于其他原因，这可能是可取的。

使用 :eqref:`eq_linreg_batch_update` 中相同的符号，$L_2$ 正则化回归的小批次随机梯度下降更新如下：

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

和以前一样，我们根据我们的估计值与观测值不同的数量更新 $\mathbf{w}$。然而，我们也将 $\mathbf{w}$ 的大小缩小到零。这就是为什么该方法有时被称为 “体重衰减”：只考虑到惩罚术语，我们的优化算法 * 衰减 * 体重在训练的每个步骤。与特征选择不同，权重衰减为我们提供了一种连续的机制来调整函数的复杂性。$\lambda$ 的较小值对应于约束较小的 $\mathbf{w}$，而 $\lambda$ 的较大值则对 $\mathbf{w}$ 的约束越大。

我们是否包含相应的偏差罚款 $b^2$ 可能因实现而异，并且可能因神经网络的层而异。通常，我们不会规范网络输出图层的偏差项。

## 高维线性回归

我们可以通过一个简单的合成样本来说明体重衰减的好处。

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
import torch.nn as nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

首先，我们像以前一样生成一些数据

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$

我们选择我们的标签作为我们输入的线性函数，由零均值和标准差 0.01 的高斯噪声损坏。为了使过拟合的效果明显，我们可以将问题的维度提高到 $d = 200$，并使用一个仅包含 20 个例子的小型训练集。

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## 从头开始实施

在下面，我们将从头开始实现权重衰减，只需将 $L_2$ 的平方罚款添加到原始目标函数中即可。

### 初始化模型参数

首先，我们将定义一个函数来随机初始化我们的模型参数。

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### 规范惩罚的定义

也许最方便的方法来实现这一点罚是平方所有条款并总结它们。

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### 定义训练循环

以下代码将模型拟合到训练集中，并在测试集中对其进行评估。自 :numref:`chap_linear` 以来，线性网络和平方损失没有改变，所以我们将通过 `d2l.linreg` 和 `d2l.squared_loss` 导入它们。这里唯一的变化是，我们的损失现在包括罚款期限。

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### 没有正规化的训练

我们现在使用 `lambd = 0` 运行此代码，禁用体重衰减。请注意，我们过拟合，减少了训练误差，但不减少了测试误差，这是一种过度拟合的案例。

```{.python .input}
#@tab all
train(lambd=0)
```

### 使用权重衰减

下面，我们运行大量的权重衰减。请注意，训练误差增加，但测试误差减少。这正是我们期望正规化产生的效果。

```{.python .input}
#@tab all
train(lambd=3)
```

## 简明实施

由于权重衰减在神经网络优化中无处不在，深度学习框架使其特别方便，将权重衰减集成到优化算法本身中，以便与任何损失函数结合使用。此外，这种集成还具有计算优势，允许实现技巧为算法增加权重衰减，而无需任何额外的计算开销。由于更新的权重衰减部分仅取决于每个参数的当前值，因此优化器无论如何都必须触摸每个参数一次。

:begin_tab:`mxnet`
在下面的代码中，我们在实例化我们的 `Trainer` 时直接通过 `wd` 指定权重衰减超参数。默认情况下，Gluon 会同时衰减权重和偏差。请注意，在更新模型参数时，超参数 `wd` 将乘以 `wd_mult`。因此，如果我们将 `wd_mult` 设置为零，则偏差置参数 $b$ 不会衰减。
:end_tab:

:begin_tab:`pytorch`
在下面的代码中，我们在实例化我们的优化器时直接通过 `weight_decay` 指定权重衰减超参数。默认情况下，PyTorch 会同时衰减权重和偏差。在这里，我们只为权重设置 `weight_decay`，因此偏差置参数 $b$ 不会衰减。
:end_tab:

:begin_tab:`tensorflow`
在下面的代码中，我们创建了一个具有权衰减超参数 `wd` 的 $L_2$ 正则化程序，并通过 `kernel_regularizer` 参数将其应用于图层。
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

地块看起来相同的那些当我们从头开始实施权重衰减。然而，它们的运行速度明显快，而且更容易实施，对于更大的问题，这种好处将变得更加明显。

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

到目前为止，我们只触及一个构成简单线性函数的概念。此外，什么构成简单的非线性函数可能是一个更复杂的问题。实例，[重现卷积核 Hilbert 空间 (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 允许在非线性上下文中应用为线性函数引入的工具。遗憾的是，基于 RKH 的算法倾向于纯粹扩展到大型、高维数据。在本书中，我们将默认使用在深度网络的所有层上应用权重衰减的简单启发式方法。

## 摘要

* 正则化是处理过拟合的常见方法。它在训练集中的损失函数中添加了一个惩罚项，以降低学习模型的复杂性。
* 保持模型简单的一个特殊选择是使用 $L_2$ 处罚进行权重衰减。这会导致学习算法的更新步骤中的权重衰减。
* 权重衰减功能在深度学习框架的优化器中提供。
* 不同的参数集可以在同一个训练循环中具有不同的更新行为。

## 练习

1. 试验本节中估计问题中 $\lambda$ 的值。绘制训练和测试准确率为 $\lambda$ 的函数。你观察到什么？
1. 使用验证集查找 $\lambda$ 的最佳值。它真的是最佳价值吗？这是否重要？
1. 如果我们使用 $\sum_i |w_i|$ 作为我们的选择惩罚（$L_1$ 正则化），更新方程是什么样的？
1. 我们知道这一点。你能找到一个类似的矩阵方程（见 :numref:`subsec_lin-algebra-norms` 中的弗罗本纽斯规范）吗？
1. 查看训练误差和泛化误差之间的关系。除了体重衰减、训练增加以及使用适当复杂的模型之外，您还能想到什么其他方法来处理过拟合？
1. 在贝叶斯统计中，我们使用的是事先和可能性通过 $P(w \mid x) \propto P(x \mid w) P(w)$ 到达后。你如何识别正则化的 $P(w)$？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
