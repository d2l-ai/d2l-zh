# 卷积神经网络 (LenNet)
:label:`sec_lenet`

我们现在拥有组装全功能 CNN 所需的所有成分。在我们早些时候遇到的图像数据中，我们将软最大回归模型 (:numref:`sec_softmax_scratch`) 和 MLP 模型 (:numref:`sec_mlp_scratch`) 应用于时尚多国主义数据集中的服装图片。为了使这些数据适用于软最大回归和 MLP，我们首先将 $28\times28$ 矩阵中的每个图像拼合成一个固定长度的 $784$ 维矢量，然后用完全连接的图层处理它们。现在我们有了卷积图层的句柄，我们可以在图像中保留空间结构。作为用卷积层替换完全连接的图层的另一个好处，我们将享受更多需要更少参数的模型。

在本节中，我们将介绍 [LeNet](http://yann.lecun.com/exdb/lenet/)，这是首批出版的 CNN，以吸引对计算机视觉任务表现的广泛关注。该模型由（并命名为）Yann Lecun，当时在 AT&T 贝尔实验室的研究员，用于识别图像 :cite:`LeCun.Bottou.Bengio.ea.1998` 中的手写数字的目的。这项工作代表了十年来开发该技术的研究的结果。1989 年，LecuN 公布了第一份通过反向传播成功培训 CNN 的研究报告。

当时，LenNet 取得了与支持向量机的性能相匹配的优异成果，然后是监督学习的主导方法。LenNet 最终被调整为识别数字，用于处理 ATM 机中的存款。到目前为止，一些自动取款机仍然运行 Yann 和他的同事莱昂·博托在 20 世纪 90 年代写的代码！

## 列内

在较高层次上，Lenet (Lenet-5) 由两部分组成：(i) 卷积编码器由两个卷积层组成；(ii) 由三个完全连接层组成的密集块；该架构概述于 :numref:`img_lenet`。

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

每个卷积块中的基本单位是卷积层、西格莫图激活函数和随后的平均池化操作。请注意，虽然 Relus 和最大池工作得更好，但这些发现在 1990 年代还没有发现。每个卷积层都使用 $5\times 5$ 内核和一个符号激活函数。这些图层将空间排列的输入映射到多个二维要素地图，通常会增加通道数量。第一个卷积层有 6 个输出通道，第二个有 16 个。每个 $2\times2$ 池操作（步长 2）通过空间缩减采样将维度降低 $4$ 系数。卷积块发出的输出形状为（批量大小，通道数量，高度，宽度）。

为了将输出从卷积块传递到密集块，我们必须在小批中压平每个示例。换句话说，我们采用这个四维输入并将其转换为完全连接图层所期望的二维输入：作为一个提醒，我们希望的二维表示使用第一维索引小批中的示例，第二维给出平坦矢量每个示例的表示形式。Lenet 的密集模块有三个完全连接的层，分别提供 120、84 和 10 个输出。由于我们仍在执行分类，因此 10 维输出图层对应于可能的输出类的数量。

虽然你真正了解 LenNet 内部发生的事情可能需要一些工作，但希望下面的代码片段能够说服你使用现代深度学习框架实现这些模型非常简单。我们只需要实例化 `Sequential` 块并将适当的层链接在一起。

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
from dataclasses import dataclass

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

我们采取了一个小的自由与原始模型, 删除了高斯激活在最后一层.除此之外，这个网络与原来的 Lenet-5 体系结构相匹配。

通过网络传递单通道（黑白）$28 \times 28$ 图像并在每个图层打印输出形状，我们可以检查模型，以确保其操作符合我们对 :numref:`img_lenet_vert` 的期望。

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

请注意，卷积块中每个图层的制图表达的高度和宽度都会减小（与前一图层相比）。第一个卷积图层使用 2 个像素的填充来补偿因使用 $5 \times 5$ 内核而导致的高度和宽度的减少。相比之下，第二个卷积图层放弃填充，因此高度和宽度都减少了 4 个像素。当我们上升图层堆栈时，通道的数量将图层从输入中的 1 个增加到第一个卷积层后的 6 个，第二个卷积层后的 16 个。但是，每个池图层将高度和宽度减半。最后，每个完全连接的图层都会降低维度，最终会发出其尺寸与类数匹配的输出。

## 培训

现在，我们已经实现了该模型，让我们运行一个实验来看看 Lenet 如何在时尚 MNist 上的票价。

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

虽然 CNN 的参数较少，但与类似深度 MLP 相比，它们的计算成本仍然更高，因为每个参数参与更多的乘法。如果您有权访问 GPU，这可能是将其付诸实施以加快培训的好时机。

:begin_tab:`mxnet, pytorch`
为了评估，我们需要对 :numref:`sec_softmax_scratch` 中描述的 `evaluate_accuracy` 函数进行稍微修改。由于完整数据集位于主内存中，因此在模型使用 GPU 计算数据集之前，我们需要将其复制到 GPU 内存中。
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
    return metric[0]/metric[1]
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

我们还需要更新我们的培训功能来处理 GPU。与 :numref:`sec_softmax_scratch` 中定义的 `train_epoch_ch3` 不同，我们现在需要在进行正向和后向传播之前将每个小批数据移动到我们的指定设备（希望是 GPU）。

培训职能也类似于 :numref:`sec_softmax_scratch` 号文件中定义的 `train_ch3` 号文件。由于我们将实施具有许多层次的网络，我们将主要依靠高级 API。以下训练函数假定一个由高级 API 创建的模型作为输入，并相应地进行了优化。我们使用 :numref:`subsec_xavier` 中引入的 Xavier 初始化，初始化 `device` 参数所指示的设备上的模型参数。就像 MLP 一样，我们的损耗函数是交叉熵，我们通过微型随机梯度下降最小化。由于每个时代运行需要几十秒钟，因此我们更频繁地显示训练损失。

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

* CNN 是采用卷积层的网络。
* 在 CNN 中，我们交叉卷积、非线性和（通常）池操作。
* 在 CNN 中，卷积图层的排列通常使其逐渐降低制图表达的空间分辨率，同时增加通道数量。
* 在传统的 CNN 中，卷积块编码的表示在发射输出之前由一个或多个完全连接的图层处理。
* LenNet 可以说是这样一个网络的第一个成功部署。

## 练习

1. 将平均池替换为最大池。会发生什么？
1. 尝试构建一个基于 LenNet 的更复杂的网络，以提高其准确性。
    1. 调整卷积窗口大小。
    1. 调整输出通道的数量。
    1. 调整激活功能（例如 RelU）。
    1. 调整卷积层的数量。
    1. 调整完全连接的层数。
    1. 调整学习率和其他培训细节（例如，初始化和时代数量）。
1. 在原始 MNIST 数据集上尝试改进的网络。
1. 显示不同输入（例如毛衣和外套）的 LenNet 第一层和第二层的激活情况。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
