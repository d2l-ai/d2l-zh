# 自定义层

深度学习成功的一个因素是可用的各种层，这些层可以以创造性的方式组成，从而设计适合各种任务的体系结构。例如，研究人员已经发明了专门用于处理图像、文本、循环连续数据和执行动态编程的图层。迟早，您会遇到或发明深度学习框架中尚不存在的图层。在这些情况下，您必须构建自定义图层。在本节中，我们将向您展示如何使用。

## 无参数的图层

首先，我们构建一个没有自己参数的自定义图层。如果您还记得我们在 :numref:`sec_model_construction` 中阻止的介绍，这应该看起来很熟悉。以下 `CenteredLayer` 类简单地从其输入中减去均值。要构建它，我们只需要从基层类继承并实现正向传播函数。

```{.python .input}
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

让我们通过输入一些数据来验证我们的图层是否按预期工作。

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

现在，我们可以将我们的图层作为一个组件来构建更复杂的模型。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

作为额外的完整性检查，我们可以通过网络发送随机数据，并检查均值是否实际上是 0。由于我们正在处理浮点数，因此由于量化，我们可能仍然看到一个非常小的非零数。

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## 带有参数的图层

现在我们已经知道了如何定义简单的图层，让我们继续定义具有可通过训练进行调整的参数的图层。我们可以使用内置功能来创建参数，这些参数提供了一些基本的管理功能。特别是，它们管理访问、初始化、共享、保存和加载模型参数。这样，除其他好处外，我们不需要为每个自定义层编写自定义序列化例程。

现在让我们实现自己版本的完全连接层。回想一下，这个层需要两个参数，一个表示权重，另一个表示偏差。在此实现中，我们将 RELU 激活中烘焙为默认值。此层需要输入参数：`in_units` 和 `units`，它们分别表示输入和输出的数量。

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        return torch.matmul(X, self.weight.data) + self.bias.data
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        return tf.matmul(X, self.weight) + self.bias
```

接下来，我们实例化 `MyDense` 类并访问其模型参数。

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
dense = MyLinear(5, 3)
dense.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

我们可以使用自定义图层直接执行前向传播计算。

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
dense(torch.randn(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

我们还可以使用自定义图层构建模型。一旦我们得到了，我们就可以像内置的完全连接层一样使用它。

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), nn.ReLU(), MyLinear(8, 1))
net(torch.randn(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## 摘要

* 我们可以通过基本图层类设计自定义图层。这使我们能够定义灵活的新图层，这些图层的行为与库中任何现有图层不同。
* 定义后，可在任意上下文和架构中调用自定义图层。
* 图层可以具有局部参数，可以通过内置函数创建。

## 练习

1. 设计一个接受输入并计算张量减少的图层，即它返回 $y_k = \sum_{i, j} W_{ijk} x_i x_j$。
1. 设计一个返回数据傅里叶系数前半的图层。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
