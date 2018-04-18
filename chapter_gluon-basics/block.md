上一章介绍了简单的深度学习模型，例如多层感知机。为了引入深度学习计算的问题，我们以该模型为例，对输入数据做计算。

在[“多层感知机——使用Gluon”](../chapter_supervised-learning/mlp-gluon.md)一节中，
我们通过在`nn.Sequential`里依次添加两个全连接层构造出多层感知机。其中第一层的输出大小为256，即隐藏层单元个数；第二层的输出大小为10，即输出层单元个数。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
```

接下来，让模型根据输入数据做一次计算。

```{.python .input  n=2}
net.initialize()
x = nd.random.uniform(shape=(2, 20))
print(net(x))
print('hidden layer: ', net[0])
print('output layer: ', net[1])
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.09543004  0.04614332 -0.00286654 -0.07790349 -0.05130243  0.02942037\n   0.08696642 -0.0190793  -0.04122177  0.05088576]\n [ 0.0769287   0.03099705  0.00856576 -0.04467199 -0.06926839  0.09132434\n   0.06786595 -0.06187842 -0.03436673  0.04234694]]\n<NDArray 2x10 @cpu(0)>\nhidden layer:  Dense(20 -> 256, Activation(relu))\noutput layer:  Dense(256 -> 10, linear)\n"
 }
]
```

在上面的例子中，`net`的输入数据`x`包含2个样本，每个样本的特征向量长度为20（`shape=(2, 20)`）。在按照默认方式初始化好模型参数后，`net`计算得到一个$2 \times 10$的矩阵作为模型的输出。其中4是数据样本个数，10是输出层单元个数。

实际上，这个多层感知机计算的例子涉及到了深度学习计算的方方面面，例如模型的构造、模型参数的初始化、模型的层等。在本章中，我们将主要使用Gluon来介绍深度学习计算中的重要组成部分：模型构造、模型参数、自定义层、读写和GPU计算。通过本章的学习，读者将能够动手实现和训练更复杂的深度学习模型，例如之后章节里的一些模型。

# 模型构造

本节中，我们将通过Gluon里的`nn.Block`来介绍如何构造深度学习模型。相信读者在学习完本节后，也会对上一章中使用的`nn.Sequential`有更深刻的认识。

## 使用 `nn.Block` 构造模型

使用Gluon提供的`nn.Block`，我们可以很方便地构造各种模型。例如，我们可以通过`nn.Block`来构造与本章开头例子中相同的多层感知机。

```{.python .input  n=3}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(256, activation='relu')
            self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))
```

这里，我们通过创建`nn.Block`的子类构造模型。任意一个`nn.Block`的子类至少实现以下两个函数：

* `__init__`：创建模型的参数。在上面的例子里，模型的参数被包含在了两个`nn.Dense`层里。
* `forward`：定义模型的计算。

接下来我们解释一下`MLP`里面用的其他命令：

* `super(MLP, self).__init__(**kwargs)`：这句话调用`MLP`父类`nn.Block`的构造函数`__init__`。这样，我们在调用`MLP`的构造函数时还可以指定函数参数`prefix`（名字前缀）或`params`（模型参数，下一节会介绍）。这两个函数参数将通过`**kwargs`传递给`nn.Block`的构造函数。

* `with self.name_scope()`：本例中的两个`nn.Dense`层和其中模型参数的名字前面都将带有模型名前缀。该前缀可以通过构造函数参数`prefix`指定。若未指定，该前缀将自动生成。我们建议，在构造模型时将每个层至少放在一个`name_scope()`里。

我们可以实例化`MLP`类得到`net2`，并让`net2`根据输入数据`x`做一次计算。其中，`y = net2(x)`明确调用了`MLP`中的`__call__`函数（从`nn.Block`继承得到）。在Gluon中，这将进一步调用`MLP`中的`forward`函数从而完成一次模型计算。

```{.python .input  n=4}
net = MLP()
net.initialize()
print(net(x))
print('hidden layer name with default prefix:', net.hidden.name)
print('output layer name with default prefix:', net.output.name)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.00362228  0.00633332  0.03201144 -0.01369375  0.10336449 -0.03508018\n  -0.00032164 -0.01676023  0.06978628  0.01303309]\n [ 0.03871715  0.02608213  0.03544959 -0.02521311  0.11005433 -0.0143066\n  -0.03052466 -0.03852827  0.06321152  0.0038594 ]]\n<NDArray 2x10 @cpu(0)>\nhidden layer name with default prefix: mlp0_dense0\noutput layer name with default prefix: mlp0_dense1\n"
 }
]
```

在上面的例子中，隐藏层和输出层的名字前都加了默认前缀。接下来我们通过`prefix`指定它们的名字前缀。

```{.python .input  n=5}
net = MLP(prefix='my_mlp_')
print('hidden layer name with "my_mlp_" prefix:', net.hidden.name)
print('output layer name with "my_mlp_" prefix:', net.output.name)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "hidden layer name with \"my_mlp_\" prefix: my_mlp_dense0\noutput layer name with \"my_mlp_\" prefix: my_mlp_dense1\n"
 }
]
```

接下来，我们重新定义`MLP_NO_NAMESCOPE`类。它和`MLP`的区别就是不含`with self.name_scope():`。这是，隐藏层和输出层的名字前都不再含指定的前缀`prefix`。

```{.python .input  n=6}
class MLP_NO_NAMESCOPE(nn.Block):
    def __init__(self, **kwargs):
        super(MLP_NO_NAMESCOPE, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP_NO_NAMESCOPE(prefix='my_mlp_')
print('hidden layer name without prefix:', net.hidden.name)
print('output layer name without prefix:', net.output.name)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "hidden layer name without prefix: dense0\noutput layer name without prefix: dense1\n"
 }
]
```

需要指出的是，在Gluon里，`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们还可以反复嵌套`nn.Block`来构建新的`nn.Block`。

`nn.Block`类主要提供模型参数的存储、模型计算的定义和自动求导。读者也许已经发现了，以上`nn.Block`的子类中并没有定义如何求导，或者是`backward`函数。事实上，`MXNet`会使用`autograd`对`forward`自动生成相应的`backward`函数。


### `nn.Sequential`是特殊的`nn.Block`

在Gluon里，`nn.Sequential`是`nn.Block`的子类。它也可以被看作是一个`nn.Block`的容器：通过`add`函数来添加`nn.Block`。在`forward`函数里，`nn.Sequential`把添加进来的`nn.Block`逐一运行。

一个简单的实现是这样的：

```{.python .input  n=7}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

它的使用和`nn.Sequential`类似：

```{.python .input  n=8}
net = MySequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[[ 0.07787765  0.00216401  0.01682201  0.03059879 -0.00702019  0.01668714\n   0.04822845  0.00394321 -0.09300036 -0.044943  ]\n [ 0.08891079 -0.00625484 -0.01619132  0.03807178 -0.01451489  0.02006172\n   0.0303478   0.02463485 -0.07605445 -0.04389167]]\n<NDArray 2x10 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 构造更复杂的模型

与`nn.Sequential`相比，使用`nn.Block`可以构造更复杂的模型。下面是一个例子。

```{.python .input  n=9}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = nd.random_uniform(shape=(10, 20))
        with self.name_scope():
            self.dense = nn.Dense(10, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight) + 1)
        x = self.dense(x)
        return x
```

在这个`FancyMLP`模型中，我们使用了常数权重`rand_weight`（注意它不是模型参数）、做了矩阵乘法操作（`nd.dot`）并重复使用了相同的`nn.Dense`层。测试一下：

```{.python .input  n=10}
net = FancyMLP()
net.initialize()
net(x)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[[0.         0.         0.         0.2370123  0.07323837 0.\n  0.06294025 0.         0.09714817 0.        ]\n [0.         0.         0.         0.2564658  0.07756092 0.\n  0.07705911 0.         0.1034155  0.        ]]\n<NDArray 2x10 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

由于`nn.Sequential`是`nn.Block`的子类，它们还可以嵌套使用。下面是一个例子。

```{.python .input  n=12}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(64, activation='relu'))
            self.net.add(nn.Dense(32, activation='relu'))
            self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP())
net.add(nn.Dense(10))
net.initialize()
print(net(x))
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "\n[[-3.26217065e-04 -1.74427973e-04 -1.15589692e-05  4.81838943e-05\n   1.92852458e-04 -1.91162762e-04 -3.87010841e-05 -8.06279277e-05\n  -1.04320447e-04  1.19986660e-04]\n [-3.05865280e-04 -9.83882128e-05 -8.97376740e-05  3.69372603e-04\n   1.09407074e-05 -6.82246638e-04  1.09486784e-04 -2.26356962e-04\n  -4.30053333e-04  2.89441086e-04]]\n<NDArray 2x10 @cpu(0)>"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* 我们可以通过`nn.Block`来构造复杂的模型。
* `nn.Sequential`是特殊的`nn.Block`。


## 练习

* 比较使用`nn.Sequential`和使用`nn.Block`构造模型的方式。如果希望访问模型中某一层（例如隐藏层）的某个属性（例如名字），这两种方式有什么不同？
* 如果把`NestMLP`中的`self.net`和`self.dense`改成`self.denses = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu'), nn.Dense(16)]`，并在`forward`中用for循环实现相同计算，会有什么问题吗？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_block.svg)
