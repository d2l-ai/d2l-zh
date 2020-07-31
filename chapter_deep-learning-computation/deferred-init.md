# 延迟初始化
:label:`sec_deferred_init`

到目前为止，似乎我们在建立我们的网络方面被草率地逃脱了。具体来说，我们做了以下不直观的事情，这可能看起来不应该工作：

* 我们在没有指定输入维度的情况下定义了网络架构。
* 我们添加了图层，而不指定前一图层的输出维度。
* 我们甚至 “初始化” 这些参数，然后提供足够的信息来确定我们的模型应该包含多少个参数。

您可能会惊讶我们的代码运行。毕竟，深度学习框架无法判断网络的输入维度。这里的诀窍是框架 * 推迟初始化 *，等到我们第一次通过模型传递数据，以便动态推断每个层的大小。

后来，当使用卷积神经网络时，这种技术将变得更加方便，因为输入维度（即图像的分辨率）将影响每个后续图层的维度。因此，无需在编写代码时知道维度是什么的情况下设置参数的能力可以极大地简化指定和随后修改我们的模型的任务。接下来，我们深入了解初始化的机制。

## 实例化网络

首先，让我们实例化 MLP。

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

此时，网络不可能知道输入图层权重的尺寸，因为输入维度仍未知。因此，框架尚未初始化任何参数。我们通过尝试访问以下参数来确认。

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
请注意，当参数对象存在时，每个图层的输入维度将列为-1。MxNet 使用特殊值-1 表示参数维度仍未知。此时，尝试访问 `net[0].weight.data()` 将触发运行时错误，指出必须先初始化网络才能访问参数。现在让我们看看当我们尝试通过 `initialize` 函数初始化参数时会发生什么。
:end_tab:

:begin_tab:`tensorflow`
请注意，每个图层对象都存在，但权重为空。使用 `net.get_weights()` 会抛出错误，因为权重尚未初始化。
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
正如我们所看到的，什么都没有改变。当输入维度未知时，初始化调用不会真正初始化参数。相反，这个调用注册到 MxNet，我们希望（也可以根据哪个分布）初始化参数。
:end_tab:

接下来让我们通过网络传递数据，使框架最终初始化参数。

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

只要我们知道输入维度 20，框架就可以通过插入值 20 来识别第一层权重矩阵的形状。在识别第一层的形状之后，框架将继续进入第二层，依此类推通过计算图形，直到所有形状都已知。请注意，在这种情况下，只有第一层需要延迟初始化，但框架按顺序初始化。一旦所有参数形状都知道，框架最终可以初始化参数。

## 摘要

* 延迟初始化可以很方便，允许框架自动推断参数形状，使得修改体系结构并消除一个常见的错误来源。
* 我们可以通过模型传递数据，使框架最终初始化参数。

## 练习

1. 如果您将输入维度指定给第一个图层而不是后续图层，会发生什么情况？您是否立即进行初始化？
1. 如果指定不匹配的维度，会发生什么情况？
1. 如果你有不同维度的输入，你需要做什么？提示：看看参数绑定。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
