# 多GPU的简洁实现
:label:`sec_multi_gpu_concise`

为每一个新模型从零开始实现并行性并不有趣。此外，优化同步工具以获得高性能也有很大的好处。下面我们将展示如何使用深度学习框架的高级API来实现这一点。数学和算法与 :numref:`sec_multi_gpu` 中的相同。毫不奇怪，你至少需要两个GPU来运行本节的代码。

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

## 简单网络

让我们使用一个比 :numref:`sec_multi_gpu` 的LeNet稍微有意义的网络，它仍然足够容易和快速地训练。我们选择了ResNet-18 :cite:`He.Zhang.Ren.ea.2016`。因为输入的图像很小，所以我们稍微修改一下。与 :numref:`sec_resnet` 的区别在于，我们在开始时使用了更小的卷积核、步长和填充。此外，我们删除了最大池化层。

```{.python .input}
#@save
def resnet18(num_classes):
    """稍加修改的ResNet-18模型。"""
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
    # 该模型使用了更小的卷积核、步长和填充，且删除了最大池化层。
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
    """稍加修改的ResNet-18模型。"""
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

    # 该模型使用了更小的卷积核、步长和填充，且删除了最大池化层。
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
`initialize`函数允许我们在所选设备上初始化参数。有关初始化方法的复习内容，请参阅 :numref:`sec_numerical_stability` 。特别方便的是，它还允许我们同时在多个设备上初始化网络。让我们在实践中尝试一下这是如何运作的。
:end_tab:

:begin_tab:`pytorch`
我们将初始化训练部分代码内的网络。有关初始化方法的复习内容，请参见 :numref:`sec_numerical_stability`。
:end_tab:

```{.python .input}
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 初始化网络的所有参数
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 我们将在训练代码实现中初始化网络
```

:begin_tab:`mxnet`
使用 :numref:`sec_multi_gpu` 中引入的 `split_and_load` 函数，我们可以切分一小批数据，并将部分数据复制到`devices`变量提供的设备列表中。网络实例自动使用适当的GPU来计算前向传播的值。在这里，我们生成4个观测值，并通过GPU将它们拆分。
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
一旦数据通过网络，相应的参数就会在数据通过的设备上初始化。这意味着初始化是在每个设备的基础上进行的。因为我们选择GPU 0和GPU 1进行初始化，所以网络只在那里初始化，而不是在CPU上初始化。事实上，这些参数甚至不存在于CPU上。我们可以通过打印出参数并观察可能出现的任何错误来验证这一点。
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
接下来，让我们用一个在多个设备上并行工作的代码来替换评估准确性的代码。这是 :numref:`sec_lenet` 的`evaluate_accuracy_gpu`函数的替代。主要区别在于，我们在调用网络之前拆分了一个小批量。其他的基本上都是一样的。
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """使用多个GPU计算数据集上模型的精度。"""
    # 查询设备列表
    devices = list(net.collect_params().values())[0].list_ctx()
    # 正确预测的数量，预测的总数量
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # 并行运行
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## 训练

如前所述，训练代码需要执行几个基本功能才能实现高效并行：

* 需要在所有设备上初始化网络参数。
* 在数据集上迭代时，要将小批量划分到所有设备上。
* 我们跨设备并行计算损失及其梯度。
* 聚合梯度，并相应地更新参数。

最后我们计算精度（同样是并行地）来报告网络的最终性能。训练代码与前几章中的实现非常相似，只是我们需要拆分和聚合数据。

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
    # 在多个GPU上设置模型
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

让我们看看这在实践中是如何运作的。作为热身，我们在单个GPU上训练网络。

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

接下来我们使用2个GPU进行训练。与 :numref:`sec_multi_gpu` 中评估的LeNet相比，ResNet-18的模型要复杂得多。这就是并行化显示其优势的地方。计算时间明显大于同步参数的时间。这提高了可伸缩性，因为并行化的开销不太相关。

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 小结

:begin_tab:`mxnet`
* Gluon通过提供上下文列表，为跨多个设备的模型初始化提供原语。
:end_tab:

* 在可以找到数据的设备上自动评估数据。
* 在尝试访问每台设备上的参数之前，请注意初始化该设备上的网络。否则，你将遇到错误。
* 优化算法在多个GPU上自动聚合。

## 练习

:begin_tab:`mxnet`
1. 本节使用ResNet-18。尝试不同的迭代周期数、批量大小和学习率。使用更多GPU进行计算。如果使用16个GPU（例如，在AWS p2.16xlarge实例上）尝试此操作，会发生什么情况？
1. 有时，不同的设备提供不同的计算能力。我们可以同时使用GPU和CPU。我们应该如何分工？值得付出努力吗？为什么呢？
1. 如果我们丢掉`npx.waitall()`会发生什么？你将如何修改训练，以使并行操作最多有两个步骤重叠？
:end_tab:

:begin_tab:`pytorch`
1. 本节使用ResNet-18。尝试不同的迭代周期数、批量大小和学习率。使用更多GPU进行计算。如果使用16个GPU（例如，在AWS p2.16xlarge实例上）尝试此操作，会发生什么情况？
1. 有时，不同的设备提供不同的计算能力。我们可以同时使用GPU和CPU。我们应该如何分工？值得付出努力吗？为什么呢？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
