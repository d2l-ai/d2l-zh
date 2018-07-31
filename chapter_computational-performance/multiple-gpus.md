# 多GPU计算

本教程我们将展示如何使用多个GPU计算，例如使用多个GPU训练模型。正如你期望的那样，运行本节中的程序需要至少两块GPU。事实上，一台机器上安装多块GPU非常常见。这是因为主板上通常会有多个PCIe插槽。如果正确安装了NVIDIA驱动，我们可以通过`nvidia-smi`命令来查看当前机器上的全部GPU。

```{.python .input  n=1}
!nvidia-smi
```

在[“自动并行计算”](auto-parallelism.md)一节里，我们介绍过，大部分的运算可以使用所有的CPU的全部计算资源，或者单个GPU的全部计算资源。但如果使用多个GPU训练模型，我们仍然需要实现相应的算法。这些算法中最常用的叫做数据并行。


## 数据并行

数据并行目前是深度学习里使用最广泛的将模型训练任务划分到多个GPU的办法。回忆一下我们在[“梯度下降和随机梯度下降”](../chapter_optimization/gd-sgd.md)一节中介绍的使用优化算法训练模型的过程。下面我们就以小批量随机梯度下降为例来介绍数据并行是如何工作的。

假设一台机器上有$k$个GPU。给定需要训练的模型，每个GPU将分别独立维护一份完整的模型参数。在模型训练的任意一次迭代中，给定一个小批量，我们将该批量中的样本划分成$k$份并分给每个GPU一份。然后，每个GPU将分别根据自己分到的训练数据样本和自己维护的模型参数计算模型参数的梯度。
接下来，我们把$k$个GPU上分别计算得到的梯度相加，从而得到当前的小批量梯度。
之后，每个GPU都使用这个小批量梯度分别更新自己维护的那一份完整的模型参数。

为了从零开始实现多GPU训练中的数据并行，让我们先导入需要的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
from time import time
```

## 定义模型

我们使用[“卷积神经网络：LeNet”](../chapter_convolutional-neural-networks/lenet.md)一节里介绍的LeNet来作为本节的样例模型。

```{.python .input  n=2}
# 初始化模型参数。
scale = 0.01
W1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(shape=20)
W2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(shape=50)
W3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(shape=128)
W4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(shape=10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型。
def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1],
                             kernel=(3, 3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                    stride=(2, 2))
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3],
                             kernel=(5, 5), num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                    stride=(2, 2))
    h2 = nd.flatten(h2)
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数。
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 多GPU之间同步数据

我们需要实现一些多GPU之间同步数据的辅助函数。下面函数将模型参数复制到某个特定GPU并初始化梯度。

```{.python .input  n=3}
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

试一试把`params`复制到`mx.gpu(0)`上。

```{.python .input}
new_params = get_params(params, mx.gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

给定分布在多个GPU之间的数据。以下函数可以把各个GPU上的数据加起来，然后再广播到所有GPU上。

```{.python .input  n=4}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

简单测试一下`allreduce`函数。

```{.python .input}
data = [nd.ones((1,2), ctx=mx.gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:', data)
allreduce(data)
print('after allreduce:', data)
```

给定一个批量的数据样本，以下函数可以划分它们并复制到各个GPU上。

```{.python .input  n=5}
def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k
    assert m * k == n, '# examples is not divided by # devices.'
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]
```

让我们试着用`split_and_load`函数将6个数据样本平均分给2个GPU。

```{.python .input}
batch = nd.arange(24).reshape((6, 4))
ctx = [mx.gpu(0), mx.gpu(1)]
splitted = split_and_load(batch, ctx)
print('input: ', batch)
print('load into', ctx)
print('output:', splitted)
```

## 单个小批量上的多GPU训练

现在我们可以实现单个小批量上的多GPU训练了。它的实现主要依据本节介绍的数据并行方法。我们将使用刚刚定义的多GPU之间同步数据的辅助函数，例如`split_and_load`和`allreduce`。

```{.python .input  n=6}
def train_batch(X, y, gpu_params, ctx, lr):
    # 当 ctx 包含多个GPU时，划分小批量数据样本并复制到各个 GPU 上。
    gpu_Xs = split_and_load(X, ctx)
    gpu_ys = split_and_load(y, ctx)
    # 在各个 GPU 上计算损失。
    with autograd.record():
        ls = [loss(lenet(gpu_X, gpu_W), gpu_y) 
              for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
    # 在各个 GPU 上反向传播。
    for l in ls:
        l.backward()
    # 把各个 GPU 上的梯度加起来，然后再广播到所有 GPU 上。
    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx))])
    for param in gpu_params:
        gb.sgd(param, lr, X.shape[0])
```

## 训练函数

现在我们可以定义训练函数。这里的训练函数和之前章节里的训练函数稍有不同。例如，在这里我们需要依据本节介绍的数据并行，将完整的模型参数复制到多个GPU上，并在每次迭代时对单个小批量上进行多GPU训练。

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    # 将模型参数复制到 num_gpus 个 GPU 上。
    gpu_params = [get_params(params, c) for c in ctx]
    for epoch in range(1, 6):
        start = time()
        for X, y in train_iter:
            # 对单个小批量上进行多 GPU 训练。
            train_batch(X, y, gpu_params, ctx, lr)
        nd.waitall()
        print('epoch %d, time: %.1f sec' % (epoch, time() - start))
        # 在 GPU 0 上验证模型。
        net = lambda x: lenet(x, gpu_params[0])
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('validation accuracy: %.4f' % test_acc)
```

我们使用2个GPU和较大的批量大小来训练，以使得GPU的计算资源能够得到较充分利用。

```{.python .input  n=10}
train(num_gpus=2, batch_size=512, lr=0.3)
```

## 小结

* 我们可以使用数据并行更充分地利用多个GPU的计算资源，实现多GPU训练模型。

## 练习

* 在本节实验中，试一试不同的迭代周期、批量大小和学习率。
* 将本节实验的模型预测部分改为用多GPU预测。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1884)

![](../img/qr_multiple-gpus.svg)
