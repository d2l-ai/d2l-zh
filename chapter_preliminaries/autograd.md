# 自动微分
:label:`sec_autograd`

正如 :numref:`sec_calculus`中所说，求导是几乎所有深度学习优化算法的关键步骤。
虽然求导的计算很简单，只需要一些基本的微积分。
但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。
实际中，根据设计好的模型，系统会构建一个*计算图*（computational graph），
来跟踪计算是哪些数据通过哪些操作组合起来产生输出。
自动微分使系统能够随后反向传播梯度。
这里，*反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

## 一个简单的例子

作为一个演示例子，(**假设我们想对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导**)。
首先，我们创建变量`x`并为其分配一个初始值。

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

x = paddle.arange(4, dtype='float32')
x
```

```{.python .input}
#@tab mindspore
import mindspore as ms
import mindspore.ops as ops

x = ops.arange(4.0)
x
```


[**在我们计算$y$关于$\mathbf{x}$的梯度之前，需要一个地方来存储梯度。**]
重要的是，我们不会在每次对一个参数求导时都分配新的内存。
因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。
注意，一个标量函数关于向量$\mathbf{x}$的梯度是向量，并且与$\mathbf{x}$具有相同的形状。

:begin_tab:`mindspore`
在我们$y$计算关于$\\{x}$的梯度之前，我们需要介绍一下MindSpore的自动微分实现方式

MindSpore现有版本不像Pytorch一样将梯度grad直接绑定在Tensor上，而是整体运算后，再通过获取梯度的算子进行梯度的提取。因此，和Pytorch有如下差异：

   1.想要自动微分的函数需要显式注册为function

   2.需要通过自动微分接口`mindspore.grad`或者`mindspore.value_and_grad`来获取梯度
:end_tab:


```{.python .input}
# 通过调用attach_grad来为一个张量的梯度分配内存
x.attach_grad()
# 在计算关于x的梯度后，将能够通过'grad'属性访问它，它的值被初始化为0
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

```{.python .input}
#@tab paddle
x = paddle.to_tensor(x, stop_gradient=False)
x.grad  # 默认值是None
```

```{.python .input}
#@tab mindspore
# MindSpore现有版本不像Pytorch一样将梯度grad直接绑定在Tensor上，而是整体运算后，再通过获取梯度的算子进行梯度的提取。所以x默认是需要梯度的，不需要再次定义，也无法直接通过x获得它的梯度
```

(**现在计算$y$。**)

```{.python .input}
# 把代码放到autograd.record内，以建立计算图
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
# 把所有计算记录在磁带上
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
#@tab paddle
y = 2 * paddle.dot(x, x)
y
```

```{.python .input}
#@tab mindspore
import mindspore.numpy as mnp
from mindspore import grad

def forward(x):
    return 2 * mnp.dot(x, x)

y = forward(x)
y
```

`x`是一个长度为4的向量，计算`x`和`x`的点积，得到了我们赋值给`y`的标量输出。
接下来，[**通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度**]，并打印这些梯度。

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

```{.python .input}
#@tab paddle
y.backward()
x.grad
```

```{.python .input}
#@tab mindspore
x_grad = grad(forward)(x)
x_grad
```

函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于$\mathbf{x}$的梯度应为$4\mathbf{x}$。
让我们快速验证这个梯度是否计算正确。

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

```{.python .input}
#@tab paddle
x.grad == 4 * x
```

```{.python .input}
#@tab mindspore
x.grad == 4 * x
```
[**现在计算`x`的另一个函数。**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # 被新计算的梯度覆盖
```

```{.python .input}
#@tab pytorch
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # 被新计算的梯度覆盖
```

```{.python .input}
#@tab paddle
# 在默认情况下，PaddlePaddle会累积梯度，我们需要清除之前的值
x.clear_gradient()
y = paddle.sum(x)
y.backward()
x.grad
```

```{.python .input}
#@tab mindspore
def forward(x):
    return x.sum()

x_grad = grad(forward)(x)
x_grad
```
## 非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。
对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括[**深度学习中**]），
但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。
这里(**，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。**)

```{.python .input}
# 当对向量值变量y（关于x的函数）调用backward时，将通过对y中的元素求和来创建
# 一个新的标量变量。然后计算这个标量变量相对于x的梯度
with autograd.record():
    y = x * x  # y是一个向量
y.backward()
x.grad  # 等价于y=sum(x*x)
```

```{.python .input}
#@tab pytorch
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # 等价于y=tf.reduce_sum(x*x)
```

```{.python .input}
#@tab paddle
x.clear_gradient()
y = x * x
paddle.sum(y).backward() 
x.grad
```

```{.python .input}
#@tab mindspore
def forward(x):
    y = x * x
    return y.sum()

x_grad = grad(forward)(x)
x_grad
```


## 分离计算

有时，我们希望[**将某些计算移动到记录的计算图之外**]。
例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。
想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数，
并且只考虑到`x`在`y`被计算后发挥的作用。

这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值，
但丢弃计算图中如何计算`y`的任何信息。
换句话说，梯度不会向后流经`u`到`x`。
因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为常数处理，
而不是`z=x*x*x`关于`x`的偏导数。

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
# 设置persistent=True来运行t.gradient多次
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
#@tab paddle
x.clear_gradient()
y = x * x
u = y.detach()
z = u * x

paddle.sum(z).backward()
x.grad == u
```

```{.python .input}
#@tab mindspore
def forward(x):
    y = x * x
    u = ops.stop_gradient(y)
    z = u * x
    return z, u

z, u = forward(x)
x_grad = grad(forward)(x)
x_grad == u
```

由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播，
得到`y=x*x`关于的`x`的导数，即`2*x`。

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

```{.python .input}
#@tab paddle
x.clear_gradient()
paddle.sum(y).backward()
x.grad == 2 * x
```

```{.python .input}
#@tab mindspore
def forward(x):
    y = x * x
    return y.sum()
x_grad = grad(forward)(x)
x_grad == 2 * x
```

## Python控制流的梯度计算

使用自动微分的一个好处是：
[**即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度**]。
在下面的代码中，`while`循环的迭代次数和`if`语句的结果都取决于输入`a`的值。

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

```{.python .input}
#@tab paddle
def f(a):
    b = a * 2
    while paddle.norm(b) < 1000:
        b = b * 2
    if paddle.sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab mindspore
def f(a):
    b = a * 2
    while ops.norm(b, axis=0) < 1000:
        b = b * 2
    if b.sum() > 0:
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

```{.python .input}
#@tab paddle
a = paddle.to_tensor(paddle.randn(shape=[1]), stop_gradient=False)
d = f(a)
d.backward()
```

```{.python .input}
#@tab mindspore
a = ops.randn(())
d = f(a)
a_grad = grad(f)(a)
```

我们现在可以分析上面定义的`f`函数。
请注意，它在其输入`a`中是分段线性的。
换言之，对于任何`a`，存在某个常量标量`k`，使得`f(a)=k*a`，其中`k`的值取决于输入`a`，因此可以用`d/a`验证梯度是否正确。

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

```{.python .input}
#@tab paddle
a.grad == d / a
```

```{.python .input}
#@tab mindspore
a.grad == d / a
```
## 小结

* 深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。

## 练习

1. 为什么计算二阶导数比一阶导数的开销要更大？
1. 在运行反向传播函数之后，立即再次运行它，看看会发生什么。
1. 在控制流的例子中，我们计算`d`关于`a`的导数，如果将变量`a`更改为随机向量或矩阵，会发生什么？
1. 重新设计一个求控制流梯度的例子，运行并分析结果。
1. 使$f(x)=\sin(x)$，绘制$f(x)$和$\frac{df(x)}{dx}$的图像，其中后者不使用$f'(x)=\cos(x)$。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1758)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1759)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1757)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11684)
:end_tab:
