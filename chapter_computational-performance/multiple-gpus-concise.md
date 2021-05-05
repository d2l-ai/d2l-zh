# 多个 GPU 的简明实施
:label:`sec_multi_gpu_concise`

为每个新模型从头开始实施并行性并不乐趣。此外，优化同步工具以实现高性能也有很大的好处。在下面我们将展示如何使用深度学习框架的高级 API 来完成此操作。数学和算法与 :numref:`sec_multi_gpu` 中的算法相同。毫不奇怪，你需要至少两个 GPU 来运行本节的代码。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 玩具网

让我们使用一个比 :numref:`sec_multi_gpu` 的 Lenet 稍有意义的网络，该网络仍然足够简单快捷地训练。我们选择了 Resnet-18 变体 :cite:`He.Zhang.Ren.ea.2016`。由于输入图像很小，我们对其进行稍微修改。特别是，与 :numref:`sec_resnet` 的区别在于，我们在开始时使用较小的卷积内核、步幅和填充。此外，我们删除了最大池层。

```{.python .input}
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## 网络初始化

:begin_tab:`mxnet`
`initialize` 函数允许我们在我们选择的设备上初始化参数。有关初始化方法的复习，请参阅 :numref:`sec_numerical_stability`。特别方便的是，它还允许我们同时在 * 多个 * 设备上初始化网络。让我们试试这在实践中是如何运作的。
:end_tab:

:begin_tab:`pytorch`
我们将在训练循环中初始化网络。有关初始化方法的复习，请参阅 :numref:`sec_numerical_stability`。
:end_tab:

```{.python .input}
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
使用 :numref:`sec_multi_gpu` 中引入的 `split_and_load` 函数，我们可以划分一小批数据并将部分内容复制到 `devices` 变量提供的设备列表中。网络实例 * 自动 * 使用适当的 GPU 来计算正向传播的值。在这里，我们生成 4 个观测结果并通过 GPU 将它们拆分。
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
数据通过网络后，相应的参数将在数据通过的设备上初始化 *。这意味着初始化是基于每台设备进行的。由于我们选择了 GPU 0 和 GPU 1 进行初始化，因此网络仅在那里初始化，而不是在 CPU 上初始化。事实上，CPU 上甚至不存在这些参数。我们可以通过打印参数并观察可能出现的任何错误来验证这一点。
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
接下来，让我们将代码替换为在多个设备上并行工作的代码来评估准确性。这可以替代 :numref:`sec_lenet` 的 `evaluate_accuracy_gpu` 功能。主要区别在于我们在调用网络之前拆分了一个小批次。其他一切基本上是相同的。
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## 培训

与之前一样，训练代码需要执行几个基本功能以实现高效的并行性： 

* 需要在所有设备上初始化网络参数。
* 迭代数据集时，小批次将在所有设备之间划分。
* 我们在不同设备之间并行计算损失及其梯度。
* 渐变将进行聚合，并相应地更新参数。

最后，我们计算报告网络最终性能的准确性（同样并行）。训练例程与前几章中的实施非常相似，只是我们需要拆分和聚合数据。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

让我们看看这在实践中是如何运作的。作为热身活动，我们在单个 GPU 上训练网络。

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

接下来我们使用 2 个 GPU 进行培训。与 :numref:`sec_multi_gpu` 中评估的 Lenet 相比，Resnet-18 的模型要复杂得多。这就是并行化显示其优势的地方。计算时间明显大于同步参数的时间。这提高了可扩展性，因为并行化的开销没有那么重要。

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 小结

:begin_tab:`mxnet`
* Gluon 通过提供上下文列表为跨多个设备的模型初始化提供了基元。
:end_tab:

* 数据将在可以找到数据的设备上自动评估。
* 在尝试访问每台设备上的参数之前，请注意初始化每台设备上的网络。否则你会遇到错误。
* 优化算法会自动聚合多个 GPU。

## 练习

:begin_tab:`mxnet`
1. 本节使用 Resnet-18。尝试不同的时代、批量大小和学习率。使用更多 GPU 进行计算。如果您使用 16 个 GPU（例如，在 AWS p2.16xlarge 实例上）尝试此操作会怎样？
1. 有时，不同的设备提供不同的计算能力。我们可以同时使用 GPU 和 CPU。我们应该如何划分工作？值得努力吗？为什么？为什么不？
1. 如果我们丢弃 `npx.waitall()` 会怎么样？你将如何修改训练，使你最多可以重叠两个步骤来实现并行性？
:end_tab:

:begin_tab:`pytorch`
1. 本节使用 Resnet-18。尝试不同的时代、批量大小和学习率。使用更多 GPU 进行计算。如果您使用 16 个 GPU（例如，在 AWS p2.16xlarge 实例上）尝试此操作会怎样？
1. 有时，不同的设备提供不同的计算能力。我们可以同时使用 GPU 和 CPU。我们应该如何划分工作？值得努力吗？为什么？为什么不？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
