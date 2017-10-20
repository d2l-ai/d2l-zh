#  Training on multiple GPUs with `gluon`

Gluon makes it easy to implement data parallel training.
In this notebook, we'll implement data parallel training for a convolutional neural network.
If you'd like a finer grained view of the concepts,
you might want to first read the previous notebook,
[multi gpu from scratch](./multiple-gpus-scratch.ipynb) with `gluon`.

To get started, let's first define a simple convolutional neural network and loss function.

## Initialize on multiple devices

Gluon supports initialization of network parameters over multiple devices. We accomplish this by passing in an array of device contexts, instead of the single contexts we've used in earlier notebooks.
When we pass in an array of contexts, the parameters are initialized
to be identical across all of our devices.

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
from mxnet import gpu

net = utils.resnet18_28(10)

ctx = [gpu(0), gpu(1)]
net.collect_params().initialize(ctx=ctx)
```

Given a batch of input data,
we can split it into parts (equal to the number of contexts)
by calling `gluon.utils.split_and_load(batch, ctx)`.
The `split_and_load` function doesn't just split the data,
it also loads each part onto the appropriate device context.

So now when we call the forward pass on two separate parts,
each one is computed on the appropriate corresponding device and using the version of the parameters stored there.

```{.python .input  n=3}
#batch = mnist['train_data'][0:GPU_COUNT*2, :]
#data = gluon.utils.split_and_load(batch, ctx)
#print(net(data[0]))
#print(net(data[1]))
```

At any time, we can access the version of the parameters stored on each device.
Recall from the first Chapter that our weights may not actually be initialized
when we call `initialize` because the parameter shapes may not yet be known.
In these cases, initialization is deferred pending shape inference.

```{.python .input  n=6}
#weight = net.collect_params()['cnn_conv0_weight']

#for c in ctx:
#    print('=== channel 0 of the first conv on {} ==={}'.format(
#        c, weight.data(ctx=c)[0]))

```

Similarly, we can access the gradients on each of the GPUs. Because each GPU gets a different part of the batch (a different subset of examples), the gradients on each GPU vary.

```{.python .input  n=2}
from mxnet import autograd
from mxnet import gluon

loss = gluon.loss.SoftmaxCrossEntropyLoss()

def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()

#label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
#forward_backward(net, data, label)
#for c in ctx:
#    print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
#        c, weight.grad(ctx=c)[0]))
```

## Put all things together

Now we can implement the remaining functions. Most of them are the same as [when we did everything by hand](./chapter07_distributed-learning/multiple-gpus-scratch.ipynb); one notable difference is that if a `gluon` trainer recognizes multi-devices, it will automatically aggregate the gradients and synchronize the parameters.

```{.python .input  n=6}
from time import time
from mxnet import init
from mxnet import nd

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.05})

def train(num_gpus, batch_size, lr):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on', ctx)

    net = utils.resnet18_28(10)
    net.initialize(init=init.Xavier(), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': lr})

    for epoch in range(5):
        # train
        start = time()
        for data, label in train_data:
            data = gluon.utils.split_and_load(data, ctx)
            label = gluon.utils.split_and_load(label, ctx)

            forward_backward(net, data, label)
            trainer.step(batch_size)
        nd.waitall()
        print('Epoch %d, training time = %.1f sec'%(
            epoch, time()-start))

        # validating on GPU 0
        test_acc = utils.evaluate_accuracy(test_data, net, ctx[0])
        print('         validation accuracy = %.4f'%(test_acc))

train(1, 128, .1)
train(2, 256, .1)
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Running on [gpu(0)]\nEpoch 0, training time = 15.5 sec\n         validation accuracy = 0.8770\nEpoch 1, training time = 15.1 sec\n         validation accuracy = 0.8888\n"
 }
]
```

## Conclusion

Both parameters and trainers in `gluon` support multi-devices. Moving from one device to multi-devices is straightforward.

## Next
[Distributed training with multiple machines](../chapter07_distributed-learning/training-with-multiple-machines.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
