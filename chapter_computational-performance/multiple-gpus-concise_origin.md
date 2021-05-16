# Concise Implementation for Multiple GPUs
:label:`sec_multi_gpu_concise`

Implementing parallelism from scratch for every new model is no fun. Moreover, there is significant benefit in optimizing synchronization tools for high performance. In the following we will show how to do this using high-level APIs of deep learning frameworks.
The math and the algorithms are the same as in :numref:`sec_multi_gpu`.
Quite unsurprisingly you will need at least two GPUs to run code of this section.

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

## A Toy Network

Let us use a slightly more meaningful network than LeNet from :numref:`sec_multi_gpu` that is still sufficiently easy and quick to train. 
We pick a ResNet-18 variant :cite:`He.Zhang.Ren.ea.2016`. Since the input images are tiny we modify it slightly. In particular, the difference from :numref:`sec_resnet` is that we use a smaller convolution kernel, stride, and padding at the beginning.
Moreover, we remove the maximum pooling layer.

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

## Network Initialization

:begin_tab:`mxnet`
The `initialize` function allows us to initialize parameters on a device of our choice.
For a refresher on initialization methods see :numref:`sec_numerical_stability`. What is particularly convenient is that it also allows us to initialize the network on *multiple* devices simultaneously. Let us try how this works in practice.
:end_tab:

:begin_tab:`pytorch`
We will initialize the network inside the training loop.
For a refresher on initialization methods see :numref:`sec_numerical_stability`.
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
Using the `split_and_load` function introduced in :numref:`sec_multi_gpu` we can divide a minibatch of data and copy portions to the list of devices provided by the `devices` variable. The network instance *automatically* uses the appropriate GPU to compute the value of the forward propagation. Here we generate 4 observations and split them over the GPUs.
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Once data pass through the network, the corresponding parameters are initialized *on the device the data passed through*.
This means that initialization happens on a per-device basis. Since we picked GPU 0 and GPU 1 for initialization, the network is initialized only there, and not on the CPU. In fact, the parameters do not even exist on the CPU. We can verify this by printing out the parameters and observing any errors that might arise.
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
Next, let us replace the code to evaluate the accuracy by one that works in parallel across multiple devices. This serves as a replacement of the `evaluate_accuracy_gpu` function from :numref:`sec_lenet`. The main difference is that we split a minibatch before invoking the network. All else is essentially identical.
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

## Training

As before, the training code needs to perform several basic functions for efficient parallelism:

* Network parameters need to be initialized across all devices.
* While iterating over the dataset minibatches are to be divided across all devices.
* We compute the loss and its gradient in parallel across devices.
* Gradients are aggregated and parameters are updated accordingly.

In the end we compute the accuracy (again in parallel) to report the final performance of the network. The training routine is quite similar to implementations in previous chapters, except that we need to split and aggregate data.

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

Let us see how this works in practice. As a warm-up we train the network on a single GPU.

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Next we use 2 GPUs for training. Compared with LeNet 
evaluated in :numref:`sec_multi_gpu`,
the model for ResNet-18 is considerably more complex. This is where parallelization shows its advantage. The time for computation is meaningfully larger than the time for synchronizing parameters. This improves scalability since the overhead for parallelization is less relevant.

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Summary

:begin_tab:`mxnet`
* Gluon provides primitives for model initialization across multiple devices by providing a context list.
:end_tab:
* Data are automatically evaluated on the devices where the data can be found.
* Take care to initialize the networks on each device before trying to access the parameters on that device. Otherwise you will encounter an error.
* The optimization algorithms automatically aggregate over multiple GPUs.



## Exercises

:begin_tab:`mxnet`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 16 GPUs (e.g., on an AWS p2.16xlarge instance)?
1. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
1. What happens if we drop `npx.waitall()`? How would you modify training such that you have an overlap of up to two steps for parallelism?
:end_tab:

:begin_tab:`pytorch`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 16 GPUs (e.g., on an AWS p2.16xlarge instance)?
1. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
:end_tab:



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
