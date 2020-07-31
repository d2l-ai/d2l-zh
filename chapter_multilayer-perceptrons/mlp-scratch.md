# 多层感知器的从零开始实现
:label:`sec_mlp_scratch`

现在我们已经从数学上对多层感知进行了描述，让我们自己尝试实现一个。为了与我们之前使用 softmax 回归 (:numref:`sec_softmax_scratch`) 取得的结果进行比较，我们将继续使用时尚多国主义图像分类数据集 (:numref:`sec_fashion_mnist`)。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
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

回想一下，时尚 MNist 包含 10 类，并且每个图像由灰度像素值的 $28 \times 28 = 784$ 网格组成。同样，我们现在将忽视像素之间的空间结构，因此我们可以将其视为仅仅是一个包含 784 个输入要素和 10 个类的分类数据集。首先，我们将实现一个具有一个隐藏层和 256 个隐藏单位的 MLP。请注意，我们可以将这两个数量视为超参数。通常，我们选择以 2 幂为单位的图层宽度，由于内存在硬件中的分配和寻址方式，这些图层宽度往往具有计算效率。

再次，我们将用几个张量来表示我们的参数。请注意，* 对于每个层 *，我们必须跟踪一个权重矩阵和一个偏差向量。与往常一样，我们为损失的梯度分配内存相对于这些参数。

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## 激活函数

为了确保我们知道一切是如何工作的，我们将使用最大函数自己实现 RelU 激活，而不是直接调用内置的 `relu` 函数。

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## 模型

因为我们忽视空间结构，我们 `reshape` 每个二维图像成长度为 `num_inputs` 的平坦矢量。最后，我们只用几行代码来实现我们的模型。

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## 损失函数

为了确保数值稳定性，并且由于我们已经从头开始实现了 softmax 函数 (:numref:`sec_softmax_scratch`)，我们利用高级 API 的集成函数来计算软最大和交叉熵损耗。回想一下我们早些时候在 :numref:`subsec_softmax-implementation-revisited` 中对这些错综复杂的讨论。我们鼓励有兴趣的读者检查损失函数的源代码，以深化他们对实施细节的了解。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 培训

幸运的是，MLP 的训练循环与 softmax 回归完全相同。再次利用 `d2l` 软件包，我们调用 `train_ch3` 函数（见 :numref:`sec_softmax_scratch`），将周期数设置为 10，学习率设置为 0.5。

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

为了评估学习的模型，我们将其应用于一些测试数据。

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## 摘要

* 我们看到，即使手动完成，实施简单的 MLP 也很容易。
* 但是，对于大量层，从头开始实现 MLP 仍然会变得混乱（例如，命名和跟踪模型的参数）。

## 练习

1. 更改超参数 `num_hiddens` 的值，并查看此超参数如何影响结果。确定此超参数的最佳值，保持所有其他参数不变。
1. 尝试添加额外的隐藏层以查看其对结果的影响。
1. 学习率的变化如何改变您的成绩？修复模型体系结构和其他超参数（包括周期数），什么学习率可为您提供最佳结果？
1. 通过优化所有超参数（学习率，周期数，隐藏层数，每层隐藏单位数），您可以获得什么最佳结果？
1. 描述为什么处理多个超参数更具挑战性。
1. 您可以想到什么最聪明的策略来构建多个超参数的搜索？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
