# Dropout
:label:`sec_dropout`

在 :numref:`sec_weight_decay` 中，我们引入了通过惩罚权重 $L_2$ 范数来规范统计模型的经典方法。从概率的角度来说，我们可以通过争辩说，我们已经假定权重从均值均为零的高斯分布中获取值来证明这种技术。更直观地说，我们可能会争辩说，我们鼓励模型在许多特征中分散其权重，而不是过分依赖少数潜在的虚假关联。

## 重新审视过度拟合

面对比示例更多的特征，线性模型往往过于拟合。但是，给出比特征更多的例子，我们通常可以指望线性模型不会过度拟合。不幸的是，线性模型概括的可靠性需要付出代成本。天真应用的线性模型不考虑要素之间的交互作用。对于每个特征，线性模型必须指定正权重或负权重，忽略上下文。

在传统文本中，广义性和灵活性之间的这种根本性张力被描述为 * 偏差-方差权衡 *。线性模型具有高偏差：它们只能表示一小类函数。但是，这些模型具有较低的方差：它们在数据的不同随机样本中给出类似的结果。

深度神经网络位于偏方方差谱的另一端。与线性模型不同，神经网络不局限于单独查看每个特征。他们可以学习要素组之间的交互作用。样本，他们可能推断电子邮件中出现的 “尼日利亚” 和 “Western Union” 表示垃圾邮件，但它们并不表示垃圾邮件。

即使我们有比特征多得多的例子，深度神经网络也能够过拟合。2017 年，一批研究人员通过在随机标记图像上训练深网，展示了神经网络的极端灵活性。尽管没有任何将输入连接到输出的真实模式，但他们发现，通过随机梯度下降优化的神经网络可以完美地标签训练集中的每个图像。考虑这意味着什么。如果标签随机均匀分配，并且有 10 个类，那么没有任何分类器在保持数据上的准确率超过 10%。这里的泛化差距高达 90%。如果我们的模型非常富有表现力，他们可以超过这种情况，那么我们什么时候才能期望它们不会过于适应？

深度网络令人费解的泛化特性的数学基础仍然是开放的研究问题，我们鼓励以理论为导向的读者更深入地研究这个主题。现在，我们转而研究实际工具，这些工具往往在经验上改进深网的泛化。

## 通过扰动实现的坚固性

让我们简要地思考一下我们对一个良好的预测模型的期望。我们希望它能够很好地利用看不见的数据。经典的泛化理论表明，为了缩小训练和测试性能之间的差距，我们应该瞄准一个简单的模型。简单可以在少数尺寸的形式。我们在 :numref:`sec_model_selection` 中讨论线性模型的单项基函数时，对此进行了探讨。此外，正如我们在 :numref:`sec_weight_decay` 中讨论权重衰减（$L_2$ 正则化）时所看到的，参数的（逆）范数也代表了简单性的有用措施。简单的另一个有用的概念是平滑性，即函数不应该对其输入的小变化敏感。实例，当我们对图像进行分类时，我们希望向像素添加一些随机杂色应该大多是无害的。

1995 年，克里斯托弗·毕晓普正规化这一想法时，他证明了与输入噪声的训练相当于季洪诺夫正则化 :cite:`Bishop.1995`。这项工作在函数平滑（因此简单）的要求与其对输入中的扰动具有弹性的要求之间建立了明确的数学联系。

然后，在 2014 年，斯里瓦斯塔娃等人 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` 开发了一个聪明的想法，如何将主教的想法应用到网络的内部层，太。也就是说，他们建议在训练期间计算后续层之前向网络的每一层注入噪音。他们意识到，在训练具有多层的深度网络时，注入噪声只会在输入-输出映射上强制平滑。

它们的想法被称为 *drop*，涉及在正向传播期间计算每个内部层时注入噪声，并且它已成为训练神经网络的标准技术。该方法被称为 * 退出 *，因为我们从字面上
*在训练过程中丢弃 * 一些神经元。
在整个训练过程中，在每次迭代中，标准压差包括在计算后续层之前将每个层中的一部分节点清零。

显而易见的是，我们将自己的叙述与主教的联系强加在一起。关于辍学的原始论文通过一个令人惊讶的类比性生殖提供了直觉。作者认为神经网络过拟合的特点是一种状态，即每个层依赖于前一层中的特定激活模式，称为这个条件 * 协同适应 *。他们声称，辍学打破了共同适应，就像性生殖被认为打破共同适应的基因一样。

因此，关键的挑战是如何注入这种噪音。一个想法是以 * 无偏 * 的方式注入噪声，以便每个图层的预期值（在固定其他图层的同时）等于它在没有噪声的情况下采取的值。

在 Bishop 的作品中，他将高斯噪声添加到线性模型的输入中。在每次训练迭代中，他将从 $\epsilon \sim \mathcal{N}(0,\sigma^2)$ 平均值为零的分布中采样的噪声添加到输入 $\mathbf{x}$，从而产生一个扰动点 $\mathbf{x}' = \mathbf{x} + \epsilon$。在预期的情况下，我们将会发出这样的信息。

在标准压差正则化中，一个通过按保留（未退出）的节点分数进行标准化来消除每个层的偏差。换句话说，对于 * 辍学概率 * $p$，每个中间激活 $h$ 都被随机变量 $h'$ 取代，如下所示：

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

根据设计，预期保持不变，即 $E[h'] = h$。

## 实践中的辍学

召回一个隐藏层和 5 个隐藏单位的 MLP :numref:`fig_mlp`.当我们将丢弃法应用于隐藏层，将每个隐藏单元归零概率为 $p$ 时，结果可以被视为只包含原始神经元子集的网络。在 :numref:`fig_dropout2` 和 $h_2$ 被删除。因此，输出的计算不再依赖于 $h_2$ 或 $h_5$，在执行反向传播时，它们各自的梯度也会消失。这样，输出图层的计算不能过分依赖于 $h_1, \ldots, h_5$ 的任何一个元素。

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

通常情况下，我们在测试时禁用丢弃法。给出一个训练有素的模型和一个新的样本，我们不会丢弃任何节点，因此不需要标准化。但是，也有一些例外：一些研究人员使用测试时的丢弃法作为启发式估计神经网络预测的 * 不确定性 *：如果预测在许多不同的压差掩码之间达成一致，那么我们可能会说网络更有信心。

## 从头开始实施

为了实现单一图层的丢弃法函数，我们必须从伯努利（二进制）随机变量中绘制尽可能多的样本，其中随机变量取值 $1$（保持），概率为 $1-p$ 和 $0$（降落），概率为 $p$。实现这一点的一个简单方法是首先从均匀分布 $U[0, 1]$ 中抽取样本。然后我们可以保留相应样本大于 $p$ 的节点，其余的节点。

在下面的代码中，我们实现了一个 `dropout_layer` 函数，它删除了张量输入 `X` 中的元素，概率为 `dropout`，如上所述重新缩放剩余部分：将幸存者除以 `1.0-dropout`。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

我们可以通过几个例子来测试 `dropout_layer` 函数。在以下代码行中，我们通过丢弃法操作传递输入 `X`，概率分别为 0，0.5 和 1。

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### 定义模型参数

再次，我们与 :numref:`sec_fashion_mnist` 引入的时尚多国主义数据集合作。我们定义一个 MLP，其中包含两个隐藏图层，每个图层包含 256 个单位。

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### 定义模型

下面的模型将丢弃法应用于每个隐藏层的输出（遵循激活函数）。我们可以分别为每个图层设置丢弃法概率。一个常见的趋势是将较低的压差概率设置在更靠近输入图层的位置。下面我们将第一个和第二个隐藏图层分别设置为 0.2 和 0.5。我们确保仅在训练期间丢弃法。

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()

        self.num_inputs = num_inputs
        self.training = is_training

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### 培训和测试

这类似于前面描述的 MLP 的培训和测试。

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 简明实施

对于高级 API，我们所需要做的就是在每个完全连接的层之后添加一个 `Dropout` 层，将丢弃法概率作为其构造函数的唯一参数传递。在训练过程丢弃法，`Dropout` 图层将根据指定的压差概率随机丢弃前一图层的输出（或相当于后续图层的输入）。如果不在训练模式下，`Dropout` 层只是在测试过程中传递数据。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

接下来，我们训练和测试模型。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 摘要

* 除了控制尺寸数量和权重向量的大小之外，压差也是避免过拟合的另一个工具。通常它们被共同使用。
* 压差将激活 $h$ 替换为具有预期值 $h$ 的随机变量。
* 辍学仅在训练期间使用。

## 练习

1. 如果您更改第一层和第二层的丢弃法概率，会发生什么情况？特别是，如果您切换两个层的图层，会发生什么情况？设计一个实验来回答这些问题，定量描述您的结果，并总结定性。
1. 增加周期的数量，并将使用丢弃法时获得的结果与不使用时的结果进行比较。
1. 当丢弃法差和未应用时，每个隐藏层的激活方差是什么？绘制一个图，以显示这两个模型的数量随着时间的推移而变化。
1. 为什么测试时不通常使用丢弃法？
1. 以本节中的模型为样本，比较使用丢弃法和体重衰减的影响。当同时使用丢弃法和体重衰减时会发生什么情况？结果是否添加剂？是否有收益减少（或更糟）？他们互相取消吗？
1. 如果我们将丢弃法应用于体重矩阵的单个权重而不是激活，会发生什么情况？
1. 发明另一种在每个层注入随机噪声的技术，这种技术与标准丢弃法技术不同。你可以开发一种方法，优于时尚 MNist 数据集（对于固定的架构）？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
