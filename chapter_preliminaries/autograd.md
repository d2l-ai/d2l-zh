# 自动求导
:label:`sec_autograd`

正如我们在 :numref:`sec_calculus` 中所解释的那样，差异是几乎所有深度学习优化算法的关键步骤。虽然采用这些衍生物的计算非常简单，只需要一些基本的微积分，但对于复杂的模型，手动完成更新可能是一个痛苦（而且往往容易出错）。

深度学习框架通过自动计算衍生品（即 * 自动差异 *）来加快这项工作。实际上，根据我们设计的模型，系统构建一个 * 计算图 *，跟踪哪些数据组合哪些操作来生成输出。自动分化使系统能够随后反向传播渐变。在这里，*backpropagate* 只是意味着跟踪计算图，填充相对于每个参数的部分导数。

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

## 一个简单的例子

作为一个玩具样本，假设我们有兴趣区分函数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ 相对于柱向量 $\mathbf{x}$。首先，让我们创建变量 `x` 并为其分配一个初始值。

```{.python .input}
x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

在我们甚至计算 $y$ 相对于 $\mathbf{x}$ 的梯度之前，我们需要一个地方来存储它。重要的是，我们不要每次对参数采取衍生物时分配新的内存，因为我们经常会更新相同的参数数千次或数百万次，并且可能会很快耗尽内存。请注意，标量值函数相对于矢量 $\mathbf{x}$ 的梯度本身就是矢量值，并且具有与 $\mathbf{x}$ 相同的形状。

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

现在让我们计算出 $y$。

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

由于 `x` 是一个长度为 4 的向量，因此执行了 `x` 和 `x` 的内部积，产生了我们分配给 `y` 的标量输出。接下来，我们可以通过调用反向传播函数并打印梯度来自动计算 `x` 的每个分量 `y` 的梯度。

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

函数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ 的梯度应为 $\mathbf{x}$ 的梯度。让我们快速验证我们想要的梯度是否正确计算。

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

现在让我们计算 `x` 的另一个函数。

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## 非标量变量向后

从技术上讲，当 `y` 不是标量时，对矢量 `y` 分化的最自然的解释就是一个矩阵。对于高阶和高维 `y` 和 `x`，分析结果可能是高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中），但当我们向后调用矢量时，我们正在尝试为 * 批次 * 训练示例的每个组成部分计算损失函数的衍生物。在这里，我们的目的不是计算分化矩阵，而是计算批量中每个样本单独计算的部分衍生物的总和。

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## 分离计算

有时，我们希望将某些计算移动到记录的计算图之外。样本，假设 `y` 被计算为 `x` 的函数，并且随后的 `z` 被计算为 `y` 和 `x` 的函数。现在，想象一下，我们想计算 `z` 相对于 `x` 的梯度，但由于某种原因，希望将 `y` 视为一个常数，并且只考虑到 `x` 在计算后发挥的作用。

在这里，我们可以分离 `y` 来返回一个新变量 `u`，该变量与 `y` 具有相同的值，但丢弃有关计算图中如何计算 `y` 的任何信息。换句话说，梯度不会向后流经 `u` 到 `x`。因此，下面的反向传播函数计算 `z = u * x` 相对于 `x` 的偏导数分导数，同时将 `u` 作为常数处理，而不是相对于 `x` 的部分导数。

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

由于记录了 `y` 的计算结果，我们可以随后在 `y` 上调用反向传播，得到 `y = x * x` 的导数，这是 `2 * x`。

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Python 控制流梯度的计算

使用自动区分的一个好处是，即使构建函数的计算图需要通过 Python 控制流的迷宫（例如，条件，循环和任意函数调用），我们仍然可以计算结果变量的梯度。在下面的代码段中，请注意 `while` 循环的迭代次数和 `if` 语句的评估都取决于输入 `a` 的值。

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

让我们计算梯度。

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

我们现在可以分析上面定义的 `f` 函数。请注意，它在其输入 `a` 中是分段线性的。换句话说，对于任何 `a`，存在一些常量标量 `k`，这样 `f(a) = k * a`，其中 `k` 的值取决于输入 `a`。因此，`d / a` 允许我们验证梯度是否正确。

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## 摘要

* 深度学习框架可以自动计算衍生品。为了使用它，我们首先将渐变附加到我们希望部分导数的变量。然后，我们记录目标值的计算，执行其反向传播函数，并访问生成的梯度。

## 练习

1. 为什么第二衍生物比第一衍生物更昂贵？
1. 运行反向传播函数后，立即再次运行它，看看会发生什么。
1. 在控制流的样本中，我们计算 `d` 相对于 `a` 的导数，如果我们将变量 `a` 更改为随机向量或矩阵，会发生什么。此时，计算结果 `f(a)` 不再是标量。结果会发生什么？我们如何分析这个？
1. 重新设计查找控制流梯度的样本。运行并分析结果。
1. 让我们来吧地块 $f(x)$ 和 $\frac{df(x)}{dx}$，其中后者是在没有利用该地块的情况下计算的。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
