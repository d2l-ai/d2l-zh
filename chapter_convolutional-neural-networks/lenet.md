# 卷积神经网络 (Lenet)
:label:`sec_lenet`

我们现在拥有组装一个功能齐全的 CNN 所需的所有成分。在我们之前遇到的图像数据中，我们将软最大回归模型 (:numref:`sec_softmax_scratch`) 和 MLP 模型 (:numref:`sec_mlp_scratch`) 应用于时尚多国主义数据集中的服装图片。为了使这些数据适用于 softmax 回归和 MLP，我们首先将 $28\times28$ 矩阵中的每个图像拼合成一个固定长度的 $784$ 维矢量，然后使用完全连接的图层对它们进行处理。现在我们有一个卷积图层的句柄，我们可以保留图像中的空间结构。作为用卷积层替换完全连接的图层的另一个好处，我们将享受更多需要更少参数的偏差模型。

在本节中，我们将介绍 *Lenet*，它是首次发布的 CNN 之一，以吸引人们对其在计算机视觉任务上的性能的广泛关注。该模型由当时的 AT&T 贝尔实验室研究员 Yann Lecun 介绍（并命名），目的是识别图像 :cite:`LeCun.Bottou.Bengio.ea.1998` 中的手写数字。这项工作标志着十年研究开发技术的结果。1989 年，Lecun 公布了第一项通过反向传播成功训练有线电视网络的研究。

当时 LenNet 取得了出色的结果，与支持向量机的性能相匹配，然后是监督学习的主导方法。Lenet 最终被调整为识别数字，用于处理 ATM 机的存款。到目前为止，一些自动取款机仍然运行 Yann 和他的同事莱昂·博图在 20 世纪 90 年代写的代码！

## Lenet

在高层次上，Lenet (Lenet-5) 由两个部分组成：(i) 卷积编码器由两个卷积层组成; 和 (ii) 由三个完全连接层组成的密集块; 该架构在 :numref:`img_lenet` 中总结。

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

每个卷积块中的基本单位为卷积层、sigmoid 激活函数和后续的平均池操作。请注意，虽然 RelU 和最大汇集工作效果更好，但这些发现在 1990 年代还没有取得。每个卷积层都使用一个 $5\times 5$ 核和一个符号激活函数。这些图层将空间排列的输入映射到许多二维要素地图，通常会增加通道数。第一个卷积层有 6 个输出通道，而第二个有 16 个输出通道。通过空间缩减采样，每次 $2\times2$ 池化操作（步进 2）将维度降低系数 $4$。卷积块发出形状由（批量大小、通道数、高度、宽度）给定的输出。

为了将卷积块的输出传递给密集块，我们必须将微型批处理中的每个示例平整。换句话说，我们将这个四维输入转换为完全连接图层所期望的二维输入：作为一个提醒，我们希望的二维表示使用第一个维度来索引微批次中的示例，第二个维度给出平坦向量表示每个示例。Lenet 的密集模块有三个完全连接的层，分别有 120、84 和 10 个输出。由于我们仍在执行分类，因此 10 维输出图层对应于可能的输出类的数量。

虽然你真正理解 LenNet 内部发生的事情可能需要一些工作，但希望下面的代码片段能够说服你，使用现代深度学习框架实现这些模型非常简单。我们只需要实例化一个 `Sequential` 块，并将适当的层链接在一起。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,1,28,28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy, OneDeviceStrategy

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

我们采取了一个小的自由与原来的模型, 去除高斯激活在最后一层.除此之外，此网络与原来的 Lenet-5 体系结构相匹配。

通过通过网络传递单通道（黑白）$28 \times 28$ 图像并在每一层打印输出形状，我们可以检查模型，以确保其操作符合我们期望的 :numref:`img_lenet_vert`。

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

请注意，整个卷积块中每个图层的制图表达的高度和宽度都会减少（与上一层相比）。第一个卷积图层使用 2 个像素的填充来补偿因使用 $5 \times 5$ 内核而导致的高度和宽度减少。相比之下，第二个卷积图层放弃填充，因此高度和宽度都减少了 4 个像素。随着层叠的上升，通道的数量将从输入中的 1 个增加到第一个卷积层后的 6 个，第二个卷积层后的 16 个。但是，每个池化图层将高度和宽度降低一半。最后，每个完全连接的图层都会降低维度，最后发出尺寸与类数相匹配的输出。

## 培训

现在，我们已经实施了模型, 让我们运行一个实验，看看 Lenet 如何在时尚 MNist 票价.

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

虽然 CNN 的参数较少，但与类似深度的 MLP 相比，它们的计算成本仍然更高，因为每个参数都参与更多的乘法。如果您有权访问 GPU，现在可能是将其付诸实施以加快培训的好时机。

:begin_tab:`mxnet, pytorch`
为了进行评估，我们需要对我们在 :numref:`sec_softmax_scratch` 中描述的 `evaluate_accuracy` 函数进行轻微的修改。由于完整的数据集位于主内存中，因此在模型使用 GPU 计算数据集之前，我们需要将其复制到 GPU 内存中。
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

我们还需要更新我们的培训功能来处理 GPU。与 :numref:`sec_softmax_scratch` 中定义的 `train_epoch_ch3` 不同，我们现在需要将每个微型数据移动到我们指定的设备（希望是 GPU），然后才能进行前向和向后传播。

训练功能也与第 :numref:`sec_softmax_scratch` 号文件中所定义的训练功能相似。由于我们将实现具有多层次的网络，我们将主要依赖于高级 API。以下训练函数假定从高级 API 创建的模型作为输入，并进行相应优化。我们使用 :numref:`subsec_xavier` 中引入的 Xavier 初始化，在 `device` 参数指示的设备上初始化模型参数。就像 MLP 一样，我们的损失函数是交叉熵，并且我们通过小批量随机梯度下降最小化它。由于每个时代的运行需要几十秒钟，所以我们更频繁地可视化训练损失。

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference compared with `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[0, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch+1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

现在让我们训练和评估 Lenet-5 模型。

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## 摘要

* CNN 是一种使用卷积层的网络。
* 在 CNN 中，我们交错卷积、非线性和（通常）合并操作。
* 在 CNN 中，通常对卷积图层进行排列，以便它们逐渐降低制图表达的空间分辨率，同时增加通道数。
* 在传统的 CNN 中，由卷积块编码的表示在发射输出之前由一个或多个完全连接的层处理。
* Lenet 可以说是这样一个网络的第一个成功部署。

## 练习

1. 将平均池替换为最大池。会发生什么？
1. 尝试构建一个更复杂的基于 Lenet 的网络，以提高其准确性。
    1. 调整卷积窗口大小。
    1. 调整输出通道的数量。
    1. 调整激活功能（如 RELU）。
    1. 调整卷积层的数量。
    1. 调整完全连接的图层的数量。
    1. 调整学习率和其他培训详细信息（例如，初始化和周期数）。
1. 在原始 MNIST 数据集上尝试改进的网络。
1. 显示 LenNet 的第一层和第二层激活不同输入（例如毛衣和大衣）。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
