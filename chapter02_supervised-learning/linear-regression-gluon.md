# 线性回归 --- 使用Gluon

[前一章](linear-regression-scratch.md)我们仅仅使用了ndarray和autograd来实现线性回归，这一章我们仍然实现同样的模型，但是使用高层抽象包`gluon`。

## 创建数据集

我们生成同样的数据集

```{.python .input  n=1}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
```

## 数据读取

但这里使用`data`模块来读取数据。

```{.python .input  n=2}
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
```

读取跟前面一致：

```{.python .input  n=3}
for data, label in data_iter:
    print(data, label)
    break
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.38450566  1.0117768 ]\n [-1.8815192  -0.17634277]\n [-1.01968884 -0.26588425]\n [-0.95514405 -0.3151958 ]\n [ 1.07606435 -0.44802737]\n [ 0.35113546 -0.67558455]\n [-0.14714409  0.35308638]\n [ 0.14286116  1.63560939]\n [-0.2893931   0.08320017]\n [-0.69495088 -0.00414529]]\n<NDArray 10x2 @cpu(0)> \n[ -3.17799393e-03   1.03284538e+00   3.06117892e+00   3.36742139e+00\n   7.86236525e+00   7.20398426e+00   2.69482207e+00  -1.08427429e+00\n   3.35248947e+00   2.82675838e+00]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## 定义模型

当我们手写模型的时候，我们需要先声明模型参数，然后再使用它们来构建模型。但`gluon`提供大量提前定制好的层，使得我们只需要主要关注使用哪些层来构建模型。例如线性模型就是使用对应的`Dense`层。

虽然我们之后会介绍如何构造任意结构的神经网络，构建模型最简单的办法是利用`Sequential`来把所有层串起来。首先我们定义一个空的模型：

```{.python .input  n=44}
net = gluon.nn.Sequential()
```

然后我们加入一个`Dense`层，它唯一必须要定义的参数就是输出节点的个数，在线性模型里面是1.

```{.python .input  n=45}
net.add(gluon.nn.Dense(1))
```

（注意这里我们并没有定义说这个层的输入节点是多少，这个在之后真正给数据的时候系统会自动赋值。我们之后会详细介绍这个特性是如何工作的。）

## 初始化模型参数

在使用前`net`我们必须要初始化模型权重，这里我们使用默认随机初始化方法（之后我们会介绍更多的初始化方法）。

```{.python .input  n=46}
net.initialize()
```

## 损失函数

`gluon`提供了平方误差函数：

```{.python .input  n=47}
square_loss = gluon.loss.L2Loss()
```

## 优化

同样我们无需手动实现随机梯度下降，我们可以用创建一个`Trainer`的实例，并且将模型参数传递给它就行。

```{.python .input  n=48}
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.05})
```

## 训练

这里的训练跟前面没有太多区别，唯一的就是我们不再是调用`SGD`，而是`trainer.step`来更新模型。

```{.python .input  n=50}
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
```

```{.json .output n=50}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, average loss: 0.000051\nEpoch 1, average loss: 0.000051\nEpoch 2, average loss: 0.000051\nEpoch 3, average loss: 0.000052\nEpoch 4, average loss: 0.000051\n"
 }
]
```

比较学到的和真实模型。我们先从`net`拿到需要的层，然后访问其权重和位移。

```{.python .input  n=52}
dense = net[0]
true_w, dense.weight.data()
```

```{.json .output n=52}
[
 {
  "data": {
   "text/plain": "([2, -3.4], \n [[ 1.99987423 -3.40039515]]\n <NDArray 1x2 @cpu(0)>)"
  },
  "execution_count": 52,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=51}
true_b, dense.bias.data()
```

```{.json .output n=51}
[
 {
  "data": {
   "text/plain": "(4.2, \n [ 4.19955826]\n <NDArray 1 @cpu(0)>)"
  },
  "execution_count": 51,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=53}
help(dense.weight)
```

```{.json .output n=53}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on Parameter in module mxnet.gluon.parameter object:\n\nclass Parameter(builtins.object)\n |  A Container holding parameters (weights) of `Block`s.\n |  \n |  `Parameter` holds a copy of the parameter on each `Context` after\n |  it is initialized with `Parameter.initialize(...)`. If `grad_req` is\n |  not `null`, it will also hold a gradient array on each `Context`::\n |  \n |      ctx = mx.gpu(0)\n |      x = mx.nd.zeros((16, 100), ctx=ctx)\n |      w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())\n |      b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Zero())\n |      w.initialize(ctx=ctx)\n |      b.initialize(ctx=ctx)\n |      out = mx.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)\n |  \n |  Parameters\n |  ----------\n |  name : str\n |      Name of this parameter.\n |  grad_req : {'write', 'add', 'null'}, default 'write'\n |      Specifies how to update gradient to grad arrays.\n |  \n |      - 'write' means everytime gradient is written to grad `NDArray`.\n |      - 'add' means everytime gradient is added to the grad `NDArray`. You need\n |        to manually call `zero_grad()` to clear the gradient buffer before each\n |        iteration when using this option.\n |      - 'null' means gradient is not requested for this parameter. gradient arrays\n |        will not be allocated.\n |  shape : tuple of int, default None\n |      Shape of this parameter. By default shape is not specified. Parameter with\n |      unknown shape can be used for `Symbol` API, but `init` will throw an error\n |      when using `NDArray` API.\n |  dtype : numpy.dtype or str, default 'float32'\n |      Data type of this parameter. For example, numpy.float32 or 'float32'.\n |  lr_mult : float, default 1.0\n |      Learning rate multiplier. Learning rate will be multiplied by lr_mult\n |      when updating this parameter with optimizer.\n |  wd_mult : float, default 1.0\n |      Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.\n |  init : Initializer, default None\n |      Initializer of this parameter. Will use the global initializer by default.\n |  \n |  Attributes\n |  ----------\n |  grad_req : {'write', 'add', 'null'}\n |      This can be set before or after initialization. Setting grad_req to null\n |      with `x.grad_req = 'null'` saves memory and computation when you don't\n |      need gradient w.r.t x.\n |  \n |  Methods defined here:\n |  \n |  __init__(self, name, grad_req='write', shape=None, dtype=<class 'numpy.float32'>, lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False, differentiable=True)\n |      Initialize self.  See help(type(self)) for accurate signature.\n |  \n |  __repr__(self)\n |      Return repr(self).\n |  \n |  data(self, ctx=None)\n |      Returns a copy of this parameter on one context. Must have been\n |      initialized on this context before.\n |      \n |      Parameters\n |      ----------\n |      ctx : Context\n |          Desired context.\n |      \n |      Returns\n |      -------\n |      NDArray on ctx\n |  \n |  grad(self, ctx=None)\n |      Returns a gradient buffer for this parameter on one context.\n |      \n |      Parameters\n |      ----------\n |      ctx : Context\n |          Desired context.\n |  \n |  initialize(self, init=None, ctx=None, default_init=<mxnet.initializer.Uniform object at 0x110394080>, force_reinit=False)\n |      Initializes parameter and gradient arrays. Only used for `NDArray` API.\n |      \n |      Parameters\n |      ----------\n |      init : Initializer\n |          The initializer to use. Overrides `Parameter.init` and default_init.\n |      ctx : Context or list of Context, defaults to `context.current_context()`.\n |          Initialize Parameter on given context. If ctx is a list of Context, a\n |          copy will be made for each context.\n |      \n |          .. note:: Copies are independent arrays. User is responsible for keeping\n |          their values consistent when updating. Normally `gluon.Trainer` does this for you.\n |      default_init : Initializer\n |          Default initializer is used when both `init` and `Parameter.init` are `None`.\n |      force_reinit : bool, default False\n |          Whether to force re-initialization if parameter is already initialized.\n |      \n |      Examples\n |      --------\n |      >>> weight = mx.gluon.Parameter('weight', shape=(2, 2))\n |      >>> weight.initialize(ctx=mx.cpu(0))\n |      >>> weight.data()\n |      [[-0.01068833  0.01729892]\n |       [ 0.02042518 -0.01618656]]\n |      <NDArray 2x2 @cpu(0)>\n |      >>> weight.grad()\n |      [[ 0.  0.]\n |       [ 0.  0.]]\n |      <NDArray 2x2 @cpu(0)>\n |      >>> weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])\n |      >>> weight.data(mx.gpu(0))\n |      [[-0.00873779 -0.02834515]\n |       [ 0.05484822 -0.06206018]]\n |      <NDArray 2x2 @gpu(0)>\n |      >>> weight.data(mx.gpu(1))\n |      [[-0.00873779 -0.02834515]\n |       [ 0.05484822 -0.06206018]]\n |      <NDArray 2x2 @gpu(1)>\n |  \n |  list_ctx(self)\n |      Returns a list of contexts this parameter is initialized on.\n |  \n |  list_data(self)\n |      Returns copies of this parameter on all contexts, in the same order\n |      as creation.\n |  \n |  list_grad(self)\n |      Returns gradient buffers on all contexts, in the same order\n |      as `values`.\n |  \n |  reset_ctx(self, ctx)\n |      Re-assign Parameter to other contexts.\n |      \n |      ctx : Context or list of Context, default `context.current_context()`.\n |          Assign Parameter to given context. If ctx is a list of Context, a\n |          copy will be made for each context.\n |  \n |  set_data(self, data)\n |      Sets this parameter's value on all contexts to data.\n |  \n |  var(self)\n |      Returns a symbol representing this parameter.\n |  \n |  zero_grad(self)\n |      Sets gradient buffer on all contexts to 0. No action is taken if\n |      parameter is uninitialized or doesn't require gradient.\n |  \n |  ----------------------------------------------------------------------\n |  Data descriptors defined here:\n |  \n |  __dict__\n |      dictionary for instance variables (if defined)\n |  \n |  __weakref__\n |      list of weak references to the object (if defined)\n |  \n |  grad_req\n\n"
 }
]
```

## 结论

可以看到`gluon`可以帮助我们更快更干净地实现模型。


## 练习

- 在训练的时候，为什么我们用了比前面要大10倍的学习率呢？（提示：可以尝试运行 `help(trainer.step)`来寻找答案。）
- 如何拿到`weight`的梯度呢？（提示：尝试 `help(dense.weight), dense.weight?, dense.weight.??`）

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/742)

```{.python .input}

```
