# 自定义层

深度学习的一个魅力之处在于神经网络中各式各样的层，例如全连接层和后面章节中将要介绍的卷积层、池化层与循环层。虽然`Gluon`提供了大量常用的层，但有时候我们依然希望自定义层。本节将介绍如何使用`NDArray`来自定义一个`Gluon`的层，从而以后可以被重复调用。


## 不含模型参数的自定义层

我们先介绍如何定义一个不含模型参数的自定义层。事实上，这和[“模型构造”](block.md)一节中介绍的使用`nn.Block`构造模型类似。下面通过继承`nn.Block`自定义了一个将输入减掉均值的层`CenteredLayer`，并将层的计算放在`forward`函数里。这个层里不含模型参数。

```{.python .input  n=1}
from mxnet import nd, gluon
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

我们可以实例化这个层用起来。

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[-2. -1.  0.  1.  2.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以用它来构造更复杂的模型。

```{.python .input  n=3}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

打印自定义层输出的均值。由于均值是浮点数，它的值是个很接近0的数。

```{.python .input  n=4}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[-6.635673e-10]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 含模型参数的自定义层

我们还可以自定义含模型参数的自定义层。这样，自定义层里的模型参数就可以通过训练学出来了。我们在[“模型参数”](parameters.md)一节里介绍了`Parameter`类。其实，在自定义层的时候我们还可以使用`nn.Block`自带的`ParameterDict`类型的成员变量`params`。顾名思义，这是一个由字符串类型的参数名字映射到`Parameter`类型的模型参数的字典。我们可以通过`get`从`ParameterDict`创建`Parameter`。

```{.python .input  n=15}
params = gluon.ParameterDict(prefix="block1_")
params.get("param2", shape=(2, 3))
params
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "block1_ (\n  Parameter block1_param2 (shape=(2, 3), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

现在我们看下如何实现一个含权重参数和偏差参数的全连接层。它使用ReLU作为激活函数。其中`in_units`和`units`分别是输入单元个数和输出单元个数。

```{.python .input  n=6}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))        

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

下面，我们实例化`MyDense`来看下它的模型参数。这里我们特意加了名字前缀`prefix`。在[“模型构造”](block.md)一节中介绍过，这是`nn.Block`的构造函数自带的参数。

```{.python .input  n=7}
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "o_my_dense_ (\n  Parameter o_my_dense_weight (shape=(10, 5), dtype=<class 'numpy.float32'>)\n  Parameter o_my_dense_bias (shape=(5,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以直接使用自定义层做计算。

```{.python .input  n=8}
dense.initialize()
dense(nd.random.uniform(shape=(2, 10)))
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[[0.         0.09092736 0.         0.17156085 0.        ]\n [0.         0.06395531 0.         0.09730551 0.        ]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

自定义层跟`Gluon`提供的层用起来很类似。

```{.python .input  n=9}
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[[0. 0.]\n [0. 0.]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

现在我们知道了如何把前面手写过的层全部包装了Gluon能用的Block，之后再用到的时候就可以飞起来了！

## 练习

* 怎么修改自定义层里参数的默认初始化函数。
* (这个比较难），在一个代码Cell里面输入`nn.Dense??`，看看它是怎么实现的。为什么它就可以支持延迟初始化了。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1256)

![](../img/qr_custom-layer.svg)
