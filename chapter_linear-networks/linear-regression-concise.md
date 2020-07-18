# 线性回归的简明实现
:label:`sec_linear_concise`

过去几年对深度学习的广泛兴趣激发了公司、学术界和业余爱好者开发了各种成熟的开源框架，以实现基于梯度的学习算法的重复工作的自动化。在 :numref:`sec_linear_scratch` 中，我们仅依赖 (i) 数据存储和线性代数的张量；(ii) 计算梯度的自动分化。在实践中，由于数据迭代器、损失函数、优化器和神经网络层非常常见，现代库也为我们实现了这些组件。

在本节中，我们将向您介绍如何通过使用深度学习框架的高级 API 来简洁地实现 :numref:`sec_linear_scratch` 的线性回归模型。

## 生成数据集

首先，我们将生成与 :numref:`sec_linear_scratch` 中相同的数据集。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 读取数据集

我们可以在框架中调用现有 API 来读取数据，而不是滚动我们自己的迭代器。我们将 `features` 和 `labels` 作为参数传递，并在实例化数据迭代器对象时指定 `batch_size`。此外，布尔值 `is_train` 表示我们是否希望数据迭代器对象洗牌每个迭代周期（周期) 上的数据（通过数据集）。

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

现在我们可以使用 `data_iter` 的方式与我们在 :numref:`sec_linear_scratch` 中调用 `data_iter` 函数的方式相同。为了验证它是否正常工作，我们可以读取和打印第一个小批示例。与 :numref:`sec_linear_scratch` 相比，这里我们使用 `iter` 来构建一个 Python 迭代器，并使用 `next` 从迭代器中获取第一个项目。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 定义模型

当我们在 :numref:`sec_linear_scratch` 中实现从头开始的线性回归时，我们明确地定义了模型参数，并对计算进行编码，以使用基本的线性代数运算生成输出。你应该知道如何做到这一点。但是，一旦你的模型变得更加复杂，一旦你几乎每天都需要这样做，你会很高兴得到帮助。这种情况类似于从头开始编写自己的博客。这样做一次或两次是有益的和有启发性的，但如果每次你需要一个博客，你花了一个月的时间来重新发明轮子，你将是一个糟糕的 Web 开发人员。

对于标准操作，我们可以使用框架的预定义图层，这使我们能够特别关注用于构建模型的图层，而不必专注于实现。我们将首先定义一个模型变量 `net`，它将引用 `Sequential` 类的一个实例。`Sequential` 类为将链接在一起的多个图层定义了一个容器。给定输入数据时，`Sequential` 实例将其传递到第一层，然后将输出作为第二层的输入传递等。在下面的样本中，我们的模型只包含一个层，因此我们不需要 `Sequential`。但是，由于我们几乎所有未来的模型都会涉及多个层次，我们将使用它只是为了让您熟悉最标准的工作流程。

回想一下单层网络的体系结构，如 :numref:`fig_single_neuron` 所示。该图层被认为是 * 完全连接 *，因为它的每个输入通过矩阵矢量乘法连接到每个输出。

:begin_tab:`mxnet`
在 Gluon 中，完全连接的层在 `Dense` 类中定义。由于我们只想生成一个标量输出，所以我们将该数字设置为 1。

值得注意的是，为了方便起见，Gluon 并不要求我们为每个层指定输入形状。所以在这里，我们不需要告诉 Gluon 有多少输入进入这个线性层。当我们第一次尝试通过我们的模型传递数据时，例如，当我们稍后执行 `net(X)` 时，Gluon 会自动推断每个层的输入数量。我们稍后将详细介绍这是如何工作的。
:end_tab:

:begin_tab:`pytorch`
在 PyTorch 中，完全连接的层是在 `Linear` 类中定义的。请注意，我们将两个参数传递到 `nn.Linear` 中。第一个指定输入特征尺寸，即 2，第二个指定输出要素尺寸，输出要素尺寸为单个标量，因此为 1。
:end_tab:

:begin_tab:`tensorflow`
在 Keras 中，完全连接的层是在 `Dense` 类中定义的。由于我们只想生成一个标量输出，所以我们将该数字设置为 1。

值得注意的是，为了方便起见，Keras 不要求我们为每个图层指定输入形状。所以在这里，我们不需要告诉 Keras 有多少输入进入这个线性层。当我们第一次尝试通过我们的模型传递数据时，例如，当我们稍后执行 `net(X)` 时，Keras 会自动推断每个层的输入数量。我们稍后将详细介绍这是如何工作的。
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## 初始化模型参数

在使用 `net` 之前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。深度学习框架通常有一种预定义的方式来初始化参数。在这里，我们指定每个权重参数应从平均值 0 和标准差 0.01 的正态分布随机采样。偏差参数将初始化为零。

:begin_tab:`mxnet`
我们将从 MxNet 导入 `initializer` 模块。本模块为模型参数初始化提供了各种方法。粘合剂使 `init` 可作为访问 `initializer` 软件包的快捷方式（缩写）。我们只指定如何通过调用 `init.Normal(sigma=0.01)` 来初始化权重。默认情况下，偏置参数初始化为零。
:end_tab:

:begin_tab:`pytorch`
正如我们在构造 `nn.Linear` 时指定的输入和输出尺寸。现在我们直接访问参数以指定初始值。我们首先通过 `net[0]` 定位图层，这是网络中的第一个图层，然后使用 `weight.data` 和 `bias.data` 方法访问参数。接下来我们使用替换方法 `normal_` 和 `fill_` 来覆盖参数值。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow 中的 `initializers` 模块为模型参数初始化提供了多种方法。在 Keras 中指定初始化方法的最简单方法是通过指定 `kernel_initializer` 创建图层时。在这里，我们再次重新创建了 `net`。
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
上面的代码可能看起来简单，但你应该注意到这里发生了一些奇怪的事情。我们正在初始化网络的参数，即使 Gluon 还不知道输入将具有多少维度！它可能是 2，如我们的例子，也可能是 2000 年。Gluon 让我们摆脱这一点，因为在幕后，初始化实际上是 * 推迟 *。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只要小心记住，由于参数尚未被初始化，我们无法访问或操作它们。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上面的代码可能看起来简单，但你应该注意到这里发生了一些奇怪的事情。我们正在初始化网络的参数，即使 Keras 还不知道输入将具有多少维度！它可能是 2，如我们的例子，也可能是 2000 年。Keras 让我们摆脱这一点，因为幕后，初始化实际上是 * 延迟 *。只有当我们第一次尝试通过网络传递数据时，才会进行真正的初始化。只要小心记住，由于参数尚未被初始化，我们无法访问或操作它们。
:end_tab:

## 定义损耗函数

:begin_tab:`mxnet`
在 Gluon 中，`loss` 模块定义了各种损耗函数。在这个样本中，我们将使用平方损耗的 Gluon 实现 (`L2Loss`)。
:end_tab:

:begin_tab:`pytorch`
`MSELoss` 类计算均方误差，也称为平方 $L_2$ 范数。默认情况下，它返回示例的平均损失。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` 类计算均方误差，也称为平方 $L_2$ 范数。默认情况下，它返回示例的平均损失。
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## 定义优化算法

:begin_tab:`mxnet`
Minibatch 随机梯度下降是一种用于优化神经网络的标准工具，因此 Gluon 通过其 `Trainer` 类支持该算法的许多变化。当我们实例化 `Trainer` 时，我们将指定要优化的参数（可从我们的模型 `net` 通过 `net.collect_params()` 获得），我们希望使用的优化算法（`sgd`），以及我们的优化算法所需的超参数字典。小批次随机梯度下降只需要我们设置值 `learning_rate`，这里设置为 0.03。
:end_tab:

:begin_tab:`pytorch`
微型随机梯度下降是优化神经网络的标准工具，因此 PyTorch 在 `optim` 模块中支持该算法的许多变化。当我们实例化一个 `SGD` 实例时，我们将指定要优化的参数（可通过 `net.parameters()` 从我们的网络获得），并使用我们的优化算法所需的超参数字典。小批次随机梯度下降只需要我们设置值 `lr`，这里设置为 0.03。
:end_tab:

:begin_tab:`tensorflow`
Minibatch 随机梯度下降是一种用于优化神经网络的标准工具，因此 Keras 在 `optimizers` 模块中支持该算法的许多变化。小批次随机梯度下降只需要我们设置值 `learning_rate`，这里设置为 0.03。
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## 培训

您可能已经注意到，通过深度学习框架的高级 API 表达我们的模型需要相对较少的代码行。我们不必单独分配参数，定义我们的损失函数，或实现小批次随机梯度下降。一旦我们开始使用更复杂的模型，高级 API 的优势将大幅增长。但是，一旦我们有所有的基本部分，训练循环本身就与我们从头开始实施所有内容时所做的非常相似。

刷新内存：对于一些时代，我们将完全传递数据集（`train_data`），迭代地抓取一个小批输入和相应的地面真实标签。对于每个小批次，我们将通过以下仪式：

* 通过调用 `net(X)` 生成预测并计算损失 `l`（正向传播）。
* 通过运行反向传播来计算渐变。
* 通过调用我们的优化器来更新模型参数。

为了良好的衡量，我们计算每个迭代周期（周期) 后的损失，并打印它来监控进度。

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

下面，我们比较了通过有限数据培训学习的模型参数和生成数据集的实际参数。要访问参数，我们首先从 `net` 访问所需的图层，然后访问该图层的权重和偏差。正如我们从头开始实施一样，请注意，我们的估计参数接近于实际真相对应的参数。

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## 摘要

:begin_tab:`mxnet`
* 使用 Gluon，我们可以更简洁地实现模型。
* 在 Gluon 中，`data` 模块提供了数据处理工具，`nn` 模块定义了大量的神经网络层，`loss` 模块定义了许多常见的损耗函数。
* MxNet 的模块 `initializer` 为模型参数初始化提供了各种方法。
* 维度和存储是自动推断的，但请注意不要在初始化参数之前尝试访问参数。
:end_tab:

:begin_tab:`pytorch`
* 使用 PyTorch 的高级 API，我们可以更简洁地实现模型。
* 在 PyTorch 中，`data` 模块提供了数据处理工具，`nn` 模块定义了大量的神经网络层和常见损耗函数。
* 我们可以通过将参数替换为 `_` 结尾的方法来初始化参数。
:end_tab:

:begin_tab:`tensorflow`
* 使用 TensorFlow 的高级 API，我们可以更简洁地实现模型。
* 在 TensorFlow 中，`data` 模块提供了数据处理工具，`keras` 模块定义了大量神经网络层和常见损耗函数。
* TensorFlow 模块 `initializers` 为模型参数初始化提供了多种方法。
* 自动推断维度和存储（但请注意，不要在初始化参数之前尝试访问参数）。
:end_tab:

## 练习

:begin_tab:`mxnet`
1. 如果我们用 `l = loss(output, y).mean()` 替换 `l = loss(output, y)`，我们需要将 `trainer.step(batch_size)` 更改为 `trainer.step(1)`，以使代码的行为相同。为什么？
1. 查看 MxNet 文档，了解模块 `gluon.loss` 和 `init` 中提供了哪些丢失函数和初始化方法。以 Huber 的损失补偿损失。
1. 你如何访问 `dense.weight` 的梯度？

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. 如果我们用 `nn.MSELoss()` 替换 `nn.MSELoss(reduction='sum')`，我们怎样才能改变代码行为相同的学习率。为什么？
1. 查看 PyTorch 文档，了解提供了哪些丢失函数和初始化方法。以 Huber 的损失补偿损失。
1. 你如何访问 `net[0].weight` 的梯度？

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. 查看 TensorFlow 文档，了解提供了哪些损失函数和初始化方法。以 Huber 的损失补偿损失。

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
