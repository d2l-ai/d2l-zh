# 在多个 GPU 上进行培训
:label:`sec_multi_gpu`

到目前为止，我们讨论了如何在 CPU 和 GPU 上高效训练模型。我们甚至展示了深度学习框架如何允许在 :numref:`sec_auto_para` 中自动并行化它们之间的计算和通信。我们还在 :numref:`sec_use_gpu` 中展示了如何使用 `nvidia-smi` 命令列出计算机上的所有可用 GPU。我们没有 * 讨论的是如何真正并行化深度学习训练。相反，我们顺便暗示，人们会以某种方式将数据拆分到多个设备之间并使其工作。本节将填充详细信息，并展示如何从头开始并行训练网络。有关如何利用高级 API 中功能的详细信息降级为 :numref:`sec_multi_gpu_concise`。我们假设您熟悉微型批次随机梯度下降算法，例如 :numref:`sec_minibatch_sgd` 中描述的算法。 

## 分解问题

让我们从一个简单的计算机视觉问题和稍微陈旧的网络开始，例如，最后有多层复杂、集中，还有几个完全连接的层。也就是说，让我们从一个看起来与 Lenet :cite:`LeCun.Bottou.Bengio.ea.1998` 或 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` 非常相似的网络开始。假定多个 GPU（如果是台式机服务器，则为 2 个 GPU，AWS g4dn.12xlarge 实例上 4 个，p3.16xlarge 上的 8 个，p2.16xlarge 上 16 个），我们希望以实现良好的加速的方式对训练进行分区，同时从简单且可重复的设计选择中受益。毕竟，多个 GPU 可以同时增加 * 内存 * 和 * 计算 * 能力。简而言之，鉴于我们想要分类的一小批训练数据，我们有以下选择。 

首先，我们可以在多个 GPU 之间对网络进行分区。也就是说，每个 GPU 都会将流入特定层的数据作为输入，跨多个后续图层处理数据，然后将数据发送到下一个 GPU。与单个 GPU 可以处理的数据相比，这使我们能够使用更大的网络处理数据。此外，每个 GPU 的内存占用可以很好地控制（占总网络占用空间的一小部分）。 

但是，层之间的接口（以及 GPU）需要严格的同步。这可能很棘手，特别是如果计算工作负载在图层之间没有正确匹配的情况下。对于大量 GPU 来说，这个问题更加严重。图层之间的接口还需要大量的数据传输，例如激活和梯度。这可能会超过 GPU 总线的带宽。此外，计算密集型但连续的操作对于分区来说不是微不足道的。例如，请参阅 :cite:`Mirhoseini.Pham.Le.ea.2017` 以了解这方面的最佳努力。这仍然是一个困难的问题，目前尚不清楚是否有可能在非平凡的问题上实现良好的（线性）扩展。除非有出色的框架或操作系统支持将多个 GPU 链接在一起，否则我们不推荐使用。 

其次，我们可以逐层分割工作。例如，我们可以将问题分成 4 个 GPU，而不是在单个 GPU 上计算 64 个通道，每个 GPU 都会生成 16 个通道的数据。同样，对于完全连接的层，我们可以拆分输出单元的数量。:numref:`fig_alexnet_original`（取自 :cite:`Krizhevsky.Sutskever.Hinton.2012`）说明了这种设计，该策略用于处理内存占用非常小的 GPU（当时为 2 GB）。如果通道（或单位）数量不太少，这样就可以在计算方面进行良好的缩放。此外，由于可用内存可以线性扩展，多个 GPU 可以处理越来越大的网络。 

![Model parallelism in the original AlexNet design due to limited GPU memory.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

但是，我们需要 * 非常大 * 数的同步或障碍操作，因为每个层都取决于所有其他层的结果。此外，需要传输的数据量可能甚至超过在 GPU 之间分布层时的数据量。因此，由于带宽成本和复杂性，我们不推荐使用此方法。 

最后，我们可以在多个 GPU 之间对数据进行分区。这样，所有 GPU 都执行相同类型的工作，尽管观察结果不同。在每个小批量训练数据之后，梯度会在 GPU 中进行汇总。这是最简单的方法，它可以在任何情况下应用。我们只需要在每个小批次之后进行同步。也就是说，在计算其他梯度参数的同时，开始交换梯度参数是非常可取的。此外，更多的 GPU 会导致更大的小批量尺寸，从而提高训练效率。但是，添加更多 GPU 并不允许我们训练更大的模型。 

![Parallelization on multiple GPUs. From left to right: original problem, network partitioning, layer-wise partitioning, data parallelism.](../img/splitting.svg)
:label:`fig_splitting`

:numref:`fig_splitting` 中描述了多个 GPU 上的不同并行化方式的比较。总的来说，如果我们能够访问具有足够大内存的 GPU，数据并行性是最方便的继续方式。另请参阅 :cite:`Li.Andersen.Park.ea.2014` 以了解分布式培训的分区的详细说明。在深度学习的早期，GPU 内存曾经是一个问题。到目前为止，除了最不寻常的情况外，所有这个问题都已解决。我们将重点放在以下内容中的数据并行性。 

## 数据并行

假设计算机上有 $k$ GPU。鉴于要训练的模型，每个 GPU 将独立维护一套完整的模型参数，尽管整个 GPU 的参数值是相同且同步的。例如，:numref:`fig_data_parallel` 说明了 $k=2$ 时使用数据并行性进行培训。 

![Calculation of minibatch stochastic gradient descent using data parallelism on two GPUs.](../img/data-parallel.svg)
:label:`fig_data_parallel`

一般来说，培训的进展情况如下： 

* 在训练的任何迭代中，只要有一个随机的微型批量，我们将批次中的示例拆分为 $k$ 部分，然后在 GPU 中均匀分布。
* 每个 GPU 都根据分配给模型的小批次子集来计算模型参数的损耗和梯度。
* 汇总 $k$ GPU 中每个 GPU 的局部梯度，以获得当前的微型批次随机梯度。
* 聚合渐变将重新分配到每个 GPU。
* 每个 GPU 都使用这个微型批次随机梯度来更新它维护的完整模型参数集。

请注意，在实践中，我们在 $k$ GPU 上进行培训时，我们将小批量 $k$ 倍增加 *，这样每个 GPU 的工作量就像我们只在单个 GPU 上进行培训一样。在 16-GPU 服务器上，这可能会大大增加小批量的大小，我们可能必须相应地提高学习率。另请注意，:numref:`sec_batch_norm` 中的批量标准化需要进行调整，例如，通过每个 GPU 保持单独的批量标准化系数。在下面的内容中，我们将使用玩具网络来说明多 GPU 训练。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## 玩具网

我们使用 :numref:`sec_lenet` 中引入的 Lenet（稍作修改）。我们从头开始定义它，以详细说明参数交换和同步。

```{.python .input}
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## 数据同步

为了高效的多 GPU 培训，我们需要两种基本操作。首先，我们需要有能力将参数列表分发到多个设备并附加渐变（`get_params`）。如果没有参数，就不可能在 GPU 上评估网络。其次，我们需要跨多个设备对参数进行求和的能力，即我们需要 `allreduce` 函数。

```{.python .input}
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

让我们通过将模型参数复制到一个 GPU 来尝试一下。

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

由于我们还没有执行任何计算，所以有关偏差参数的梯度仍然为零。现在让我们假设我们有一个向量分布在多个 GPU 上。以下 `allreduce` 函数将所有向量加起来，并将结果广播回所有 GPU。请注意，为了实现这一点，我们需要将数据复制到累计结果的设备。

```{.python .input}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)
```

让我们通过在不同设备上创建具有不同值的矢量来测试这一点，然后聚合它们

```{.python .input}
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## 分发数据

我们需要一个简单的实用程序函数才能在多个 GPU 之间均匀分配微型批次。例如，在两个 GPU 上，我们希望将一半的数据复制到任何一个 GPU 中。由于它更方便、更简洁，我们使用深度学习框架中的内置函数在 $4 \times 5$ 矩阵上进行试用。

```{.python .input}
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

为了以后重复使用，我们定义了一个分割数据和标签的 `split_batch` 函数。

```{.python .input}
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 训练

现在我们可以在单个小批量上实施多 GPU 训练。其实施主要基于本节中描述的数据并行方法。我们将使用刚才讨论的辅助函数 `allreduce` 和 `split_and_load`，在多个 GPU 之间同步数据。请注意，我们不需要编写任何特定的代码即可实现并行性。由于计算图在微型批次内的设备之间没有任何依赖关系，因此它是并行 * 自动执行的。

```{.python .input}
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

现在，我们可以定义训练功能。它与前几章中使用的略有不同：我们需要分配 GPU 并将所有模型参数复制到所有设备。显然，每个批次都使用 `train_batch` 函数来处理多个 GPU。为方便起见（以及代码的简洁性），我们在单个 GPU 上计算准确性，尽管这是 * 效率低的 *，因为其他 GPU 处于空闲状态。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

让我们看看这在单个 GPU 上的效果。我们首先使用 256 个批量大小，学习率为 0.2。

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

通过保持批量大小和学习速率不变并将 GPU 的数量增加到 2，我们可以看到，与之前的实验相比，测试准确度大致保持不变。就优化算法而言，它们是相同的。不幸的是，这里没有任何有意义的加速：模型太小了；此外，我们只有一个小数据集，在这里我们略微不完善的多 GPU 训练方法受到了巨大的 Python 开销。今后，我们将遇到更复杂的模型和更复杂的并行化方式。尽管如此，让我们看看时尚 Mnist 会发生什么。

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## 小结

* 有多种方法可以将深度网络训练分成多个 GPU。我们可以在图层之间、跨图层或跨数据拆分它们。前两者需要严格编排的数据传输。数据并行性是最简单的策略。
* 数据并行培训非常简单。但是，它增加了有效的微型批量以提高效率。
* 在数据并行度中，数据被拆分到多个 GPU 中，其中每个 GPU 执行自己的向前和向后操作，然后聚合梯度，结果将广播回 GPU。
* 我们可能会对较大的小批量使用略微提高的学习率。

## 练习

1. 在 $k$ GPU 上进行培训时，将小批量大小从 $b$ 更改为 $k \cdot b$，即按 GPU 的数量向上扩展。
1. 比较不同学习率的准确性。它如何随着 GPU 的数量进行扩展？
1. 实施一个更高效的 `allreduce` 函数来聚合不同的 GPU 上的不同参数？为什么效率更高？
1. 实施多 GPU 测试精度计算。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab:
