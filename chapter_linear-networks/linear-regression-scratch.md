# 线性回归的从零开始实现
:label:`sec_linear_scratch`

现在您已经了解了线性回归背后的关键想法，我们可以开始在代码中实现实践。在本节中，我们将从头开始实现整个方法，包括数据流水线、模型、损失函数和微型批次随机梯度下降优化器。虽然现代深度学习框架几乎可以自动执行所有这些工作，但从头开始实施操作是确保您真正了解自己在做什么的唯一方法。此外，当涉及到自定义模型时，定义我们自己的层或损失功能时，了解事情如何在引擎盖下工作将被证明是方便的。在本节中，我们将仅依靠张量和自动分化。之后，我们将利用深度学习框架的钟声和口哨，介绍一个更简洁的实现。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## 生成数据集

为了保持简单，我们将根据带有加法噪声的线性模型构建一个人工数据集。我们的任务是使用我们的数据集中包含的有限示例来恢复此模型的参数。我们将保持低维数据，以便我们可以轻松地显示数据。在以下代码片段中，我们生成一个包含 1000 个示例的数据集，每个示例由从标准正态分布采样的 2 个要素组成。因此，我们的综合数据集将是一个矩阵 $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$。

生成我们的数据集的真实参数将是 $\mathbf{w} = [2, -3.4]^\top$ 和 $b = 4.2$，我们的合成标注将根据以下线性模型进行分配，其噪声项为 $\epsilon$：

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

您可以将 $\epsilon$ 视为捕获特征和标签上的潜在测量误差。我们将假设标准假设维持，因此 $\epsilon$ 服从平均值为 0 的正态分布。为了让我们的问题变得容易，我们将其标准差设置为 0.01。以下代码生成我们的合成数据集。

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

请注意，`features` 中的每一行都包含一个二维数据点，`labels` 中的每一行都包含一维标签值（一个标量）。

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

通过使用第二个特征 `features[:, 1]` 和 `labels` 生成散点图，我们可以清楚地观察到两者之间的线性相关性。

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## 读取数据集

回想一下，训练模型包括对数据集进行多次传递，一次抓取一个小批示例，然后使用它们更新我们的模型。由于此过程对于训练机器学习算法非常重要，因此值得定义一个实用程序函数来对数据集进行洗牌并以小批量访问。

在下面的代码中，我们定义了 `data_iter` 函数来演示此功能的一个可能实现。该函数采用批量大小、要素矩阵和标注矢量，从而产生大小为 `batch_size` 的小批次。每个小批次由一个要素和标注的元组组成。

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

一般而言，请注意，我们希望使用大小合理的小批量来利用 GPU 硬件，这在并行化操作方面非常出色。由于每个样本都可以并行通过我们的模型提供，并且每个样本的损失函数的梯度也可以并行进行，因此 GPU 允许我们处理数百个样本的时间比处理单个示例所花费的时间少得多。

为了建立一些直觉，让我们阅读并打印第一批量数据示例。每个微型批次中要素的形状告诉我们微型批次大小和输入要素的数量。同样，我们的小批标签将采用 `batch_size` 给出的形状。

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

在我们运行迭代时，我们会先后获得不同的小批，直到整个数据集用尽（试试这个）。虽然上面实现的迭代对于教学目的很好，但效率低下，可能会使我们在实际问题上遇到麻烦。样本，它要求我们加载内存中的所有数据，并且我们执行大量的随机内存访问。在深度学习框架中实现的内置迭代器效率要高得多，它们可以处理存储在文件中的数据和通过数据流传输的数据。

## 初始化模型参数

在我们开始通过微型随机梯度下降优化模型的参数之前，我们首先需要一些参数。在下面的代码中，我们通过从平均值为 0，标准差为 0.01 的正态分布中采样随机数，并将偏差设置为 0 来初始化权重。

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

初始化我们的参数后，我们的下一个任务是更新它们，直到它们充分适合我们的数据。每次更新都需要采用我们的损失函数相对于参数的梯度。鉴于这个梯度，我们可以更新每个参数，以减少损失的方向。

由于没有人愿意明确计算梯度（这很乏味，容易出误差），我们使用 :numref:`sec_autograd` 中引入的自动差异来计算梯度。

## 定义模型

接下来，我们必须定义我们的模型，将其输入和参数与其输出联系起来。回想一下，要计算线性模型的输出，我们只需采用输入要素 $\mathbf{X}$ 的矩阵矢量点积和模型权重 $\mathbf{w}$，然后将偏移量 $b$ 添加到每个样本中。请注意，低于 $\mathbf{Xw}$ 是一个向量，$b$ 是一个标量。回想一下广播机制，如 :numref:`subsec_broadcasting` 所述。当我们添加矢量和标量时，标量被添加到矢量的每个组件中。

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## 定义损耗函数

由于更新我们的模型需要采用损失函数的梯度，我们应该先定义损失函数。这里我们将使用 :numref:`sec_linear_regression` 中描述的平方损失函数。在实现中，我们需要将真实值 `y` 变换为预测值的形状 `y_hat`。以下函数返回的结果也将具有与 `y_hat` 相同的形状。

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## 定义优化算法

正如我们在 :numref:`sec_linear_regression` 中讨论的那样，线性回归有一个封闭式解。然而，这不是一本关于线性回归的书：它是一本关于深度学习的书。由于本书介绍的其他模型都无法通过分析解决，因此我们将借此机会介绍您第一个微型批次随机梯度下降的工作样本。

在每一步，使用从我们的数据集中随机绘制的一个微型批次，我们将估计相对于我们的参数的损失梯度。接下来，我们将更新我们的参数，以减少损失的方向。以下代码应用微批次随机梯度下降更新，给定一组参数、学习率和批量大小。更新步骤的大小由学习率 `lr` 决定。因为我们的损失是以小批次示例的总和计算的，所以我们按批批量大小 (`batch_size`) 标准化我们的步长大小，这样一个典型步长大小的幅度不会在很大程度上取决于我们对批次大小的选择。

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## 培训

现在我们已经完成了所有部件，我们已经准备好实施主要的训练循环。理解此代码至关重要，因为在深度学习的整个职业生涯中，您将一遍又一遍地看到几乎相同的训练循环。

在每次迭代中，我们将获取一小批训练示例，并通过我们的模型传递它们以获得一组预测。计算损失后，我们启动向后通过网络，存储相对于每个参数的渐变。最后，我们将调用优化算法 `sgd` 来更新模型参数。

总之，我们将执行以下循环：

* 初始化参数
* 重复直到完成
    * 计算梯度
    * 更新参数

在每个 *epoch* 中，我们将遍历整个数据集（使用 `data_iter` 函数），一旦遍历训练数据集中的每个样本（假设示例数量可以被批量次大小除去）。时代的数量和学习率 `lr` 都是超参数，我们在这里分别设置为 3 和 0.03。不幸的是，设置超参数很棘手，需要通过试验和误差进行一些调整。我们现在保留这些细节，但稍后在 :numref:`chap_optimization` 中进行修改。

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

在这种情况下，因为我们自己合成了数据集，所以我们准确地知道真正的参数是什么。因此，我们可以通过比较真实参数和我们通过训练循环学习的参数来评估我们在训练中的成功。他们确实是相互亲近的。

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

请注意，我们不应该认为我们能够完美地恢复参数是理所当然的。然而，在机器学习中，我们通常不太关心恢复真实的基础参数，而更关心的是导致高度准确的预测参数。幸运的是，即使在困难的优化问题上，随机梯度下降往往可以找到非常好的解决方案，部分原因是，对于深度网络，存在许多参数配置，可以实现高度准确的预测。

## 摘要

* 我们看到了如何从头开始实施和优化深度网络，只需使用张量和自动分化，而无需定义层或花哨的优化器。
* 本部分只划伤可能的表面。在以下几节中，我们将根据我们刚才介绍的概念来描述其他模型，并学习如何更简洁地实现这些模型。

## 练习

1. 如果我们将权重初始化为零，会发生什么。算法仍然有效吗？
1. 假设你是 [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) 试图拿出电压和电流之间的模型。您可以使用自动分化来学习模型的参数吗？
1. 您可以使用 [普朗克定律](https://en.wikipedia.org/wiki/Planck%27s_law) 使用光谱能量密度来确定物体的温度吗？
1. 如果你想计算二阶导数，你可能会遇到什么问题？你会如何解决这些问题？
1.  为什么在 `squared_loss` 函数中需要使用 `reshape` 函数？
1. 使用不同的学习速率进行实验，以了解损失函数值下降的速度。
1. 如果示例数量不能除以批量大小，则 `data_iter` 函数的行为会发生什么情况？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
