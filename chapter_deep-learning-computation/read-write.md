# 读写文件

到目前为止，我们讨论了如何处理数据以及如何构建、训练和测试深度学习模型。但是，在某些时候，我们希望能够对学习的模型感到满意，我们希望将结果保存在各种情况下以后使用（甚至可以在部署中做出预测）。此外，在运行长时间的培训过程时，最佳做法是定期保存中间结果（检查点），以确保如果我们穿过服务器的电源线，我们不会失去几天的计算。因此，现在是学习如何加载和存储单个权重向量和整个模型的时候了。本节讨论了这两个问题。

## 加载和保存张量

对于单个张量，我们可以直接调用 `load` 和 `save` 函数来读取和写入它们。这两个函数都要求我们提供一个名称，`save` 要求将要保存的变量作为输入。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save("x-file.npy", x)
```

我们现在可以将存储的文件中的数据读回内存。

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load("x-file")
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

我们可以存储张量列表并将它们读回内存。

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

我们甚至可以编写和读取从字符串映射到张量的字典。当我们想要读取或写入模型中的所有权重时，这很方便。

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## 加载和保存模型参数

保存单个权重向量（或其他张量）是有用的，但如果我们想保存（以后加载）整个模型，它会变得非常乏味。毕竟，我们可能会有数百个参数组洒在整个过程中。因此，深度学习框架提供了内置功能来加载和保存整个网络。需要注意的一个重要细节是，这将保存模型 * 参数 *，而不是保存整个模型。例如，如果我们有一个 3 层 MLP，我们需要单独指定架构。这样做的原因是模型本身可以包含任意代码，因此它们不能像自然那样被序列化。因此，为了恢复模型，我们需要在代码中生成体系结构，然后从磁盘加载参数。让我们从熟悉的 MLP 开始吧。

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

接下来，我们将模型的参数存储为名为 “mlp.params” 的文件。

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

为了恢复模型，我们实例化原始 MLP 模型的克隆。我们直接读取存储在文件中的参数，而不是随机初始化模型参数。

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights("mlp.params")
```

由于两个实例具有相同的模型参数，因此相同输入 `X` 的计算结果应该是相同的。让我们验证这一点。

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## 摘要

* `save` 和 `load` 函数可用于为张量对象执行文件 I/O 操作。
* 我们可以通过参数字典保存和加载网络的整组参数。
* 保存体系结构必须在代码中而不是在参数中完成。

## 练习

1. 即使不需要将训练模型部署到不同的设备，存储模型参数的实际好处是什么？
1. 假设我们只想重复使用网络中要合并到不同体系结构的网络中的部分。如何使用新网络中先前网络的前两层？
1. 您将如何保存网络架构和参数？您会对架构施加什么限制？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
