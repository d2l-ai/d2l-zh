# Designing a custom layer with ``gluon``

Now that we've peeled back some of the syntactic sugar conferred by ``nn.Sequential()`` and given you a feeling for how ``gluon`` works under the hood, you might feel more comfortable when writing your high-level code. But the real reason to get to know ``gluon`` more intimately is so that we can mess around with it and write our own Blocks.

Up until now, we've presented two versions of each tutorial. One from scratch and one in ``gluon``. Empowered with such independence, you might be wondering, "if I wanted to write my own layer, why wouldn't I just do it from scratch?"

In reality, writing every model completely from scratch can be cumbersome.  Just like there's only so many times a developer can code up a blog from scratch without hating life, there's only so many times that you'll want to write out a convolutional layer, or define the stochastic gradient descent updates. Even in pure research environment, we usually want to customize one part of the model. For example, we might want to implement a new layer, but still rely on other common layers, loss functions, optimizers, etc. In some cases it might be nontrivial to compute the gradient efficiently and the automatic differentiation subsystem might need some help: When was the last time you performed backprop through a log-determinant, a Cholesky factorization, or a matrix exponential? In other cases things might not be numerically very stable when calculated straightforwardly (e.g. taking logs of exponentials of some arguments).

By hacking ``gluon``, we can get the desired flexibility in one part of our model, without screwing up everything else that makes our life easy.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
mx.random.seed(1)

###########################
#  Speficy the context we'll be using
###########################
ctx = mx.cpu()

###########################
#  Load up our dataset
###########################
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Defining a (toy) custom layer

To start, let's pretend that we want to use ``gluon`` for its optimizer, serialization, etc, but that we need a new layer. Specifically, we want a layer that centers its input about 0 by subtracting its mean. We'll go ahead and define the simplest possible ``Block``. Remember from the last tutorial that in ``gluon`` a layer is called a ``Block`` (after all, we might compose multiple blocks into a larger block, etc.).

```{.python .input  n=2}
class CenteredLayer(Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - nd.mean(x)
```

That's it. We can just call instantiate this block and make a forward pass.
Note that this layer doesn't actually care
what it's input dimension or output dimensions are.
So we can just feed in an arbitrary array
and should expect appropriately transformed output. Whenever we are happy with whatever the automatic differentiation generates, this is all we need.

```{.python .input  n=3}
net = CenteredLayer()
net(nd.array([1,2,3,4,5]))
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "\n[-2. -1.  0.  1.  2.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can also incorporate this layer into a more complicated network, as by using ``nn.Sequential()``.

```{.python .input  n=4}
net2 = nn.Sequential()
net2.add(nn.Dense(128))
net2.add(nn.Dense(10))
net2.add(CenteredLayer())
```

This network contains Blocks (Dense) that contain parameters and thus require initialization

```{.python .input  n=5}
net2.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

Now we can pass some data through it, say the first image from our MNIST dataset.

```{.python .input  n=6}
for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
output = net2(data[0:1])
print(output)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.47172713 -0.27308482 -0.95690644  0.17887072  0.03658688  0.30082721\n   0.15491416 -0.09305321  0.02917496  0.15094344]]\n<NDArray 1x10 @cpu(0)>\n"
 }
]
```

And we can verify that as expected, the resulting vector has mean 0.

```{.python .input  n=7}
nd.mean(output)
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "\n[  4.47034854e-09]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

There's a good chance you'll see something other than 0. When I ran this code, I got ``2.68220894e-08``.
That's roughly ``.000000027``. This is due to the fact that MXNet often uses low precision arithmetics.
For deep learning research, this is often a compromise that we make.
In exchange for giving up a few significant digits, we get tremendous speedups on modern hardware.
And it turns out that most deep learning algorihtms don't suffer too much from the loss of precision. This is probably due to the fact that work and worse on account of the loss of precision.

## Custom layers with parameters

While ``CenteredLayer`` should give you some sense of how to implement a custom layer, it's missing a few important pieces. Most importantly, ``CenteredLayer`` doesn't care about the dimensions of its input or output, and it doesn't contain any trainable parameters. Since you already know how to implement a fully-connected layer from scratch, let's learn how to make parameteric ``Block`` by implementing MyDense, our own version of a fully-connected (Dense) layer.

## Parameters

Before we can add parameters to our custom ``Block``, we should get to know how ``gluon`` deals with parmaeters generally. Instead of working with NDArrays directly, each ``Block`` is associated with some number (as few as zero) of ``Parameter`` (groups).

At a high level, you can think of a ``Parameter`` as a wrapper on an ``NDArray``. However, the ``Parameter`` can be instantiated before the corresponding NDArray is. For example, when we instantiate a ``Block`` but the shapes of each parameter still need to be inferred, the Parameter will wait for the shape to be inferred before allocating memory.

To get a hands-on feel for mxnet.Parameter, let's just instantiate one outside of a ``Block``:

```{.python .input  n=8}
my_param = gluon.Parameter("exciting_parameter_yay", grad_req='write', shape=(5,5))
print(my_param)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Parameter exciting_parameter_yay (shape=(5, 5), dtype=<type 'numpy.float32'>)\n"
 }
]
```

Here we've instantiated a parameter, giving it the name "exciting_parameter_yay". We've also specified that we'll want to capture gradients for this Parameter. Under the hood, that lets ``gluon`` know that it has to call ``.attach_grad()`` on the underlying NDArray. We also specified the shape. Now that we have a Parameter, we can initialize its values via ``.initialize()`` and print extract its data by calling ``.data()``.

```{.python .input  n=9}
my_param.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print(my_param.data())
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.50735062 -0.65750605 -0.56013602  0.46934015  0.1596154 ]\n [-0.65080845 -0.11559016  0.31085443 -0.49285054  0.57047993]\n [ 0.35613006  0.29938424  0.61431509  0.13020623  0.21408975]\n [-0.38888294  0.65209502 -0.08793807 -0.03835624  0.63372332]\n [-0.42945772 -0.36274379 -0.06317961 -0.58671117  0.2023437 ]]\n<NDArray 5x5 @cpu(0)>\n"
 }
]
```

For data parallelism, a Parameter can also be initialzied on multiple contexts. The Parameter will then keep a copy of its value on each context. Keep in mind that you need to maintain consistency amount the copies when updating the Parameter (usually `gluon.Trainer` does this for you).

Note that you need at least two GPUs to run this section.

```{.python .input  n=10}
# my_param = gluon.Parameter("exciting_parameter_yay", grad_req='write', shape=(5,5))
# my_param.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.gpu(0), mx.gpu(1)])
# print(my_param.data(mx.gpu(0)), my_param.data(mx.gpu(1)))
```

## Parameter dictionaries (introducing ``ParameterDict``)

Rather than directly store references to each of its ``Parameters``, ``Block``s typicaly contain a parameter dictionary (``ParameterDict``). In practice, we'll rarely instantiate our own ``ParamterDict``. That's because whenever we call the ``Block`` constructor it's generated automatically. For pedagogical purposes, we'll do it from scratch this one time.

```{.python .input  n=11}
pd = gluon.ParameterDict(prefix="block1_")
```

MXNet's ``ParameterDict`` does a few cool things for us. First, we can instantiate a new Parameter by calling ``pd.get()``

```{.python .input  n=12}
pd.get("exciting_parameter_yay", grad_req='write', shape=(5,5))
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "Parameter block1_exciting_parameter_yay (shape=(5, 5), dtype=<type 'numpy.float32'>)"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Note that the new parameter is (i) contained in the ParameterDict and (ii) appends the prefix to it's name. This naming convention helps us to know which parameters belong to which ``Block`` or sub-``Block``. It's especially useful when we want to write parameters to disc (i.e. serialize), or read them from disc (i.e. deserialize).

Like a regular Python dictionary, we can get the names of all parameters with ``.keys()`` and can

```{.python .input  n=13}
pd["block1_exciting_parameter_yay"]
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "Parameter block1_exciting_parameter_yay (shape=(5, 5), dtype=<type 'numpy.float32'>)"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Craft a bespoke fully-connected ``gluon`` layer

Now that we know how parameters work, we're ready to create our very own fully-connected layer. We'll use the familiar relu activation from previous tutorials.

```{.python .input  n=14}
def relu(X):
    return nd.maximum(X, 0)
```

Now we can define our ``Block``.

```{.python .input  n=15}
class MyDense(Block):
    ####################
    # We add arguments to our constructor (__init__)
    # to indicate the number of input units (``in_units``)
    # and output units (``units``)
    ####################
    def __init__(self, units, in_units=0, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.units = units
            self._in_units = in_units
            #################
            # We add the required parameters to the ``Block``'s ParameterDict ,
            # indicating the desired shape
            #################
            self.weight = self.params.get(
                'weight', init=mx.init.Xavier(magnitude=2.24),
                shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    #################
    #  Now we just have to write the forward pass.
    #  We could rely upong the FullyConnected primitative in NDArray,
    #  but it's better to get our hands dirty and write it out
    #  so you'll know how to compose arbitrary functions
    #################
    def forward(self, x):
        with x.context:
            linear = nd.dot(x, self.weight.data()) + self.bias.data()
            activation = relu(linear)
            return activation
```

Recall that every Block can be run just as if it were an entire network.
In fact, linear models are nothing more than neural networks
consisting of a single layer as a network.

So let's go ahead and run some data though our bespoke layer.
We'll want to first instantiate the layer and initialize its parameters.

```{.python .input  n=16}
dense = MyDense(20, in_units=10)
dense.collect_params().initialize(ctx=ctx)
```

```{.python .input  n=17}
dense.params
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "mydense0_ (\n  Parameter mydense0_weight (shape=(10, 20), dtype=<type 'numpy.float32'>)\n  Parameter mydense0_bias (shape=(20,), dtype=<type 'numpy.float32'>)\n)"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now we can run through some dummy data.

```{.python .input  n=18}
dense(nd.ones(shape=(2,10)))
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[[ 0.          0.59868848  0.          1.08994353  0.          0.\n   0.02280135  0.26122352  0.15244918  0.          0.          1.23705149\n   0.53500706  0.          0.          0.61897928  0.09488952  0.          0.\n   0.46094608]\n [ 0.          0.59868848  0.          1.08994353  0.          0.\n   0.02280135  0.26122352  0.15244918  0.          0.          1.23705149\n   0.53500706  0.          0.          0.61897928  0.09488952  0.          0.\n   0.46094608]]\n<NDArray 2x20 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Using our layer to build an MLP

While it's a good sanity check to run some data though the layer, the real proof that it works will be if we can compose a network entirely out of ``MyDense`` layers and achieve respectable accuracy on a real task. So we'll revist the MNIST digit classification task, and use the familiar ``nn.Sequential()`` syntax to build our net.

```{.python .input  n=19}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(MyDense(128, in_units=784))
    net.add(MyDense(64, in_units=128))
    net.add(MyDense(10, in_units=64))
```

## Initialize Parameters

```{.python .input  n=20}
net.collect_params().initialize(ctx=ctx)
```

## Instantiate a loss

```{.python .input  n=21}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=22}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Evaluation Metric

```{.python .input  n=23}
metric = mx.metric.Accuracy()

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.

    for i, (data, label) in enumerate(data_iterator):
        with autograd.record():
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            output = net(data)

        metric.update([label], [output])
    return metric.get()[1]
```

## Training loop

```{.python .input  n=24}
epochs = 10
moving_loss = 0.

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
            cross_entropy.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(cross_entropy).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

```

```{.json .output n=24}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.662074091477, Train_acc 0.756814285714, Test_acc 0.7559\nEpoch 1. Loss: 0.588702041132, Train_acc 0.763771428571, Test_acc 0.758025\nEpoch 2. Loss: 0.572036757495, Train_acc 0.769761904762, Test_acc 0.764713333333\nEpoch 3. Loss: 0.538418785097, Train_acc 0.773721428571, Test_acc 0.770240909091\nEpoch 4. Loss: 0.538585911311, Train_acc 0.776274285714, Test_acc 0.773951724138\nEpoch 5. Loss: 0.309081547325, Train_acc 0.793864285714, Test_acc 0.778911111111\nEpoch 6. Loss: 0.291082330554, Train_acc 0.806795918367, Test_acc 0.795658139535\nEpoch 7. Loss: 0.291905886004, Train_acc 0.816110714286, Test_acc 0.808068\nEpoch 8. Loss: 0.261846415965, Train_acc 0.824328571429, Test_acc 0.817166666667\nEpoch 9. Loss: 0.273562002079, Train_acc 0.830831428571, Test_acc 0.8251140625\n"
 }
]
```

## Conclusion

It works! There's a lot of other cool things you can do. In more advanced chapters, we'll show how you can make a layer that takes in multiple inputs, or one that cleverly calls down to MXNet's symbolic API to squeeze out extra performance without screwing up your convenient imperative workflow.

## Next
[Serialization: saving your models and parameters for later re-use](../chapter03_deep-neural-networks/serialization.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
