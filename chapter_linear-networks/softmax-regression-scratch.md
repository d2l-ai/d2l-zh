# Softmax 回归的从零开始实现
:label:`sec_softmax_scratch`

就像我们从头开始实现线性回归一样，我们认为 softmax 回归也是类似的基础，你应该知道如何自己实现它的血腥细节。我们将使用刚刚在 :numref:`sec_fashion_mnist` 中引入的时尚 MNist 数据集，设置批量大小为 256 的数据迭代器。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

与我们的线性回归样本一样，这里的每个样本都将用固定长度向量表示。原始数据集中的每个样本都是 $28 \times 28$ 图像。在本节中，我们将拼合每个图像，将它们视为长度为 784 的矢量。未来，我们将讨论利用图像空间结构的更复杂的策略，但现在我们将每个像素位置视为另一个特征。

回想一下，在 softmax 回归中，我们的输出数量与类一样多。因为我们的数据集有 10 个类，所以我们的网络输出维度为 10。因此，我们的权重将构成一个 $784 \times 10$ 矩阵，偏差将构成一个 $1 \times 10$ 行向量。与线性回归一样，我们将使用高斯噪声初始化我们的权重 `W`，我们的偏差将初始值 0。

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## 定义软最大操作

在实现 softmax 回归模型之前，让我们简要地回顾一下总和运算符如何沿着张量中的特定维度工作，如 :numref:`subseq_lin-alg-reduction` 和 :numref:`subseq_lin-alg-non-reduction` 所述。给定矩阵 `X`，我们可以对所有元素进行总和（默认情况下）或仅对同一轴中的元素进行总和，即同一列（轴 0）或同一行（轴 1）。请注意，如果 `X` 是一个具有形状（2,3）的张量，并且我们对列进行总和，则结果将是一个具有形状（3，）的向量。在调用和运算符时，我们可以指定保留原始张量中的轴数，而不是折叠我们所汇总的尺寸。这将导致形状（1，3）的二维张量。

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

我们现在已经准备好实施 softmax 操作了。回想一下，softmax 由三个步骤组成：i) 我们对每个项进行指数（使用 `exp`）; ii) 我们对每行进行总和（我们在批量中每个样本中有一行）以获得每个样本的归一化量; iii) 我们将每行除以其标准化常量，确保结果总和为 1。在查看代码之前，让我们回顾一下这是如何表达的方程式：

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

分母或标归一化量，有时也称为 * 分区函数 *（其对数称为日志分区函数）。该名称的起源来自 [统计物理学](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)，其中一个相关的方程对粒子集合上的分布进行建模。

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

正如你所看到的，对于任何随机输入，我们将每个元素变成一个非负数。此外，每行总和最多 1，因为概率所需。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

请注意，虽然这看起来在数学上是正确的，但我们在实现中有点草率，因为我们没有采取预防措施来防止由于矩阵的大或非常小的元素导致的数字溢出或下溢。

## 定义模型

现在我们已经定义了 softmax 运算，我们可以实现 softmax 回归模型。下面的代码定义了输入如何通过网络映射到输出。请注意，在将数据传递到我们的模型之前，我们使用 `reshape` 函数将批量中的每个原始图像展平为矢量。

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 定义损耗函数

接下来，我们需要实现 :numref:`sec_softmax` 中引入的交叉熵损失函数。这可能是所有深度学习中最常见的损失函数，因为目前分类问题远远超过回归问题。

回想一下，交叉熵采用分配给真实标签的预测概率的负对数似然。我们可以通过单个运算符选择所有元素，而不是使用 Python for 循环迭代预测（这往往效率低下）。下面，我们创建了一个玩具数据 `y_hat`，其中包含 3 个类的预测概率的 2 个示例。然后我们选择第一个例子中第一个类的概率和第二个样本中第三个类的概率。

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

现在我们只需一行代码就可以有效地实现交叉熵损失函数。

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## 分类准确性

给定预测概率分布 `y_hat`，只要我们必须输出硬预测，我们通常选择具有最高预测概率的类。事实上，许多应用需要我们做出选择。Gmail 必须将电子邮件分类为 “主”、“社交”、“更新” 或 “论坛”。它可能会在内部估计概率，但在一天结束时，它必须在类中选择一个。

当预测与标签分类 `y` 一致时，它们是正确的。分类准确率是所有正确预测的一部分。虽然直接优化准确率性可能很困难（这是不可区分的），但它通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总是会报告它。

为了计算准确率，我们执行以下操作。首先，如果 `y_hat` 是矩阵，我们假设第二个维度存储每个类的预测分数。我们使用 `argmax` 通过每行中最大条目的索引获取预测类。然后我们将预测类与地面真实 `y` 元素进行比较。由于等式运算符 `==` 对数据类型很敏感，因此我们将 `y_hat` 的数据类型转换为与 `y` 的数据类型相匹配。结果是一个包含 0（false）和 1（true）条目的张量。获取总和会产生正确预测的数量。

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

我们将继续使用之前定义的变量 `y_hat` 和 `y` 分别作为预测的概率分布和标签。我们可以看到，第一个示例的预测类是 2（该行的最大元素为 0.6，索引 2），这与实际标签 0 不一致。第二个示例的预测类是 2（行的最大元素为 0.5，索引为 2），这与实际标签 2 一致。因此，这两个示例的分类准确率率为 0.5。

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

同样，我们可以评估通过数据迭代器 `data_iter` 访问的数据集上任何模型 `net` 的准确率。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

这里 `Accumulator` 是一个实用程序类，用于累积多个变量的总和。在上面的 `evaluate_accuracy` 函数中，我们在 `Accumulator` 实例中创建了 2 个变量，用于分别存储正确预测的数量和预测的数量。当我们遍历数据集时，两者都将随着时间的推移而累积。

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

由于我们使用随机权重初始化 `net` 模型，因此该模型的准确率应接近于随机猜测，即 10 个类的 0.1。

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## 培训

如果您阅读我们在 :numref:`sec_linear_scratch` 中的线性回归的实现，softmax 回归的训练循环应该看起来非常熟悉。在这里，我们重构实现以使其可重复使用。首先，我们定义一个函数来训练一个迭代周期（周期)。请注意，`updater` 是更新模型参数的常规函数，它接受批量次大小作为参数。它可以是 `d2l.sgd` 函数的包装器，也可以是框架的内置优化函数。

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

在显示训练函数的实现之前，我们定义了一个在动画中绘制数据的实用程序类。同样，它旨在简化本书其余部分的代码。

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

然后，以下训练函数在通过 `train_iter` 访问的训练数据集上训练一个模型 `net`，该数据集由 `num_epochs` 指定。在每个迭代周期（周期) 结束时，通过 `test_iter` 访问的测试数据集对模型进行评估。我们将利用 `Animator` 类来可视化培训进度。

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

作为一个从头开始的实现，我们使用 :numref:`sec_linear_scratch` 中定义的微型随机梯度下降来优化模型的损失函数，学习率为 0.1。

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

现在，我们训练模型与 10 个时代。请注意，周期数（`num_epochs`）和学习率（`lr`）都是可调节的超参数。通过更改它们的值，我们可以提高模型的分类准确率。

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 预测

现在培训已经完成，我们的模型已经准备好对某些图像进行分类。给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）。

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## 摘要

* 借助 softmax 回归，我们可以训练多类分类的模型。
* softmax 回归的训练循环与线性回归中的训练循环非常相似：检索和读取数据，定义模型和损失函数，然后使用优化算法训练模型。正如你很快就会发现的，大多数常见的深度学习模型都有类似的培训程序。

## 练习

1. 在本节中，我们直接实现了基于 softmax 运算的数学定义的 softmax 函数。这可能会导致什么问题？提示：尝试计算 $\exp(50)$ 的大小。
1. 本节中的函数 `cross_entropy` 是根据交叉熵损失函数的定义实现的。这个实现可能有什么问题？提示：考虑对数的域。
1. 你可以想到什么解决方案来解决上述两个问题？
1. 返回最有可能的标签总是一个好主意吗？样本，你会这样做医疗诊断吗？
1. 假设我们希望使用 softmax 回归来基于某些功能预测下一个单词。大量词汇可能会出现什么问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
