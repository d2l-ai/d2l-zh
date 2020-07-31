# Softmax 回归的简洁实现
:label:`sec_softmax_concise`

正如深度学习框架的高级 API 使得在 :numref:`sec_linear_concise` 中实现线性回归变得更加容易一样，我们会发现它类似（或可能更方便）实现分类模型。让我们坚持使用时尚多国主义数据集，并将批量大小保持在 256，如 :numref:`sec_softmax_scratch` 中。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

如 :numref:`sec_softmax` 所述，softmax 回归的输出图层是一个完全连接的图层。因此，为了实现我们的模型，我们只需在 `Sequential` 中添加一个带有 10 个输出的完全连接层。同样，在这里，`Sequential` 并不是真正必要的，但我们可能会形成这种习惯，因为在实现深度模型时，它将无处不在。我们再次以零平均值和标准差 0.01 随机初始化权重。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## 重新审视软件最大实施
:label:`subsec_softmax-implementation-revisited`

在前面的 :numref:`sec_softmax_scratch` 样本中，我们计算了模型的输出，然后通过交叉熵损耗运行此输出。从数学上讲，这是一件完全合理的事情。然而，从计算角度来看，指数可能是数值稳定性问题的一个来源。

回想一下，软最大函数计算 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中 $\hat y_j$ 是预测概率分布的 $j^\mathrm{th}$ 元素和 $o_j$ 是对数的 $j^\mathrm{th}$ 元素。如果 $o_k$ 中的一些非常大（即非常积极），那么 $\exp(o_k)$ 可能大于我们可以拥有的特定数据类型（即 * 溢出 *）的最大数字。这将使分母（和/或分子）`inf`（无穷大），我们最后遇到的是 0，`inf` 或 `nan`（不是数字）的 $\hat y_j$。在这些情况下，我们没有得到一个明确定义的交叉熵返回值。

解决这个问题的一个诀窍是先从所有 $o_k$ 中减去 $\max(o_k)$，然后再继续进行软最大计算。您可以验证每个 $o_k$ 按常数因子进行的移动是否不会更改 softmax 的返回值。在减法和归一化步骤之后，可能有些 $o_j$ 具有较大的负值，因此相应的 $\exp(o_j)$ 将接近零的值。由于有限的精度（即 * 下水 *），这些值可能会四舍五入为零，使 $\hat y_j$ 为零，并且给我们 $\log(\hat y_j)$ 的值 `-inf`。在反向传播的道路上，我们可能会发现自己面临着一个可怕的 `nan` 结果的屏幕。

幸运的是，尽管我们正在计算指数函数，但我们最终打算采用它们的日志（在计算交叉熵损失时）。通过将这两个运算符 softmax 和交叉熵结合在一起，我们可以避免反向传播过程中可能会困扰我们的数值稳定性问题。如下面的公式所示，我们避免计算 $\exp(o_j)$，并且可以直接使用 $o_j$，直接由于在 $\log(\exp(\cdot))$ 中取消。

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\log{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

如果我们想要通过模型评估输出概率，我们希望保持传统的 softmax 函数方便。但不是将 softmax 概率传递到我们的新损失函数中，我们只是传递日志并计算 softmax 及其日志在交叉熵损失函数中一次性，这做了像 [“LogSumExp 技巧”]（https://en.wikipedia.org/wiki/LogSumExp）这样的聪明事情。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 优化算法

在这里，我们使用学习率为 0.1 的小批次随机梯度下降作为优化算法。请注意，这与我们在线性回归样本中应用的相同，它说明了优化器的一般适用性。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## 培训

接下来我们调用 :numref:`sec_softmax_scratch` 中定义的训练函数来训练模型。

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

和以前一样，这个算法收敛到一个实现了体面准确率的解决方案，尽管这次代码行比以前少。

## 摘要

* 使用高级 API，我们可以更简洁地实现 softmax 回归。
* 从计算的角度来看，实现 softmax 回归具有复杂性。请注意，在许多情况下，深度学习框架除了这些最知名的技巧之外，还需要额外的预防措施来确保数值稳定性，从而使我们免受更多的陷阱，如果我们试图在实践中从头开始编写我们的所有模型，我们会遇到这些陷阱。

## 练习

1. 尝试调整超参数，例如批量大小、周期数和学习率，以查看结果。
1. 增加训练时代的数量。为什么测试准确率会在一段时间后降低？我们怎么能解决这个问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
