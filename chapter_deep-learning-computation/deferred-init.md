# 延后初始化
:label:`sec_deferred_init`

到目前为止，我们忽略了建立网络时需要做的以下这些事情：

* 我们定义了网络架构，但没有指定输入维度。
* 我们添加层时没有指定前一层的输出维度。
* 我们在初始化参数时，甚至没有足够的信息来确定模型应该包含多少参数。

有些读者可能会对我们的代码能运行感到惊讶。
毕竟，深度学习框架无法判断网络的输入维度是什么。
这里的诀窍是框架的*延后初始化*（defers initialization），
即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

在以后，当使用卷积神经网络时，
由于输入维度（即图像的分辨率）将影响每个后续层的维数，
有了该技术将更加方便。
现在我们在编写代码时无须知道维度是什么就可以设置参数，
这种能力可以大大简化定义和修改模型的任务。
接下来，我们将更深入地研究初始化机制。

## 实例化网络

首先，让我们实例化一个多层感知机。

```{.python .input}
from mxnet import np, npx
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

此时，因为输入维数是未知的，所以网络不可能知道输入层权重的维数。
因此，框架尚未初始化任何参数，我们通过尝试访问以下参数进行确认。

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
注意，当参数对象存在时，每个层的输入维度为-1。
MXNet使用特殊值-1表示参数维度仍然未知。
此时，尝试访问`net[0].weight.data()`将触发运行时错误，
提示必须先初始化网络，然后才能访问参数。
现在让我们看看当我们试图通过`initialize`函数初始化参数时会发生什么。
:end_tab:

:begin_tab:`tensorflow`
请注意，每个层对象都存在，但权重为空。
使用`net.get_weights()`将抛出一个错误，因为权重尚未初始化。
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
如我们所见，一切都没有改变。
当输入维度未知时，调用`initialize`不会真正初始化参数。
而是会在MXNet内部声明希望初始化参数，并且可以选择初始化分布。
:end_tab:

接下来让我们将数据通过网络，最终使框架初始化参数。

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

一旦我们知道输入维数是20，框架可以通过代入值20来识别第一层权重矩阵的形状。
识别出第一层的形状后，框架处理第二层，依此类推，直到所有形状都已知为止。
注意，在这种情况下，只有第一层需要延迟初始化，但是框架仍是按顺序初始化的。
等到知道了所有的参数形状，框架就可以初始化参数。

## 小结

* 延后初始化使框架能够自动推断参数形状，使修改模型架构变得容易，避免了一些常见的错误。
* 我们可以通过模型传递数据，使框架最终初始化参数。

## 练习

1. 如果指定了第一层的输入尺寸，但没有指定后续层的尺寸，会发生什么？是否立即进行初始化？
1. 如果指定了不匹配的维度会发生什么？
1. 如果输入具有不同的维度，需要做什么？提示：查看参数绑定的相关内容。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5770)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5770)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1833)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11779)
:end_tab:
