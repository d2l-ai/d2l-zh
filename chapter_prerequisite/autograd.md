# 自动求梯度

在深度学习中，我们经常需要对函数求梯度（gradient）。如果你对本节中的数学概念（例如梯度）不是很熟悉，可以参阅附录中[“数学基础”](../chapter_appendix/math.md)一节。本节将介绍如何使用MXNet提供的`autograd`包来自动求梯度。

```{.python .input  n=2}
from mxnet import autograd, nd
```

## 简单例子

我们先看一个简单例子：对函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 求关于列向量 $\boldsymbol{x}$ 的梯度。我们先创建变量`x`，并赋初值。

```{.python .input}
x = nd.arange(4).reshape((4, 1))
x
```

为了求有关变量`x`的梯度，我们需要先调用`attach_grad`函数来申请存储梯度所需要的内存。

```{.python .input}
x.attach_grad()
```

下面定义有关变量`x`的函数。为了减少计算和内存开销，默认条件下MXNet不会记录用于求梯度的计算。我们需要调用`record`函数来要求MXNet记录与求梯度有关的计算。

```{.python .input}
with autograd.record():
    y = 2 * nd.dot(x.T, x)
```

由于`x`的形状为(4, 1)，`y`是一个标量。接下来我们可以通过调用`backward`函数自动求梯度。需要注意的是，如果`y`不是一个标量，MXNet将默认先对`y`中元素求和得到新的变量，再求该变量有关`x`的梯度。

```{.python .input}
y.backward()
```

函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 关于$\boldsymbol{x}$ 的梯度应为$4\boldsymbol{x}$。现在我们来验证一下求出来的梯度是正确的。

```{.python .input}
assert (x.grad - 4 * x).norm().asscalar() == 0
x.grad
```

## 训练模式和预测模式

从上面可以看出，在调用`record`函数后，MXNet会记录并计算梯度。此外，默认下`autograd`还会将运行模式从预测模式转为训练模式。这可以通过调用`is_training`函数来查看。

```{.python .input}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

除了调用`record`函数来切换运行模式以外，我们还可以使用`set_training`函数来设定运行模式。

```{.python .input}
# 设置为训练模式。
autograd.set_training(train_mode=True)
print(autograd.is_training())

# 设置为预测模式。
autograd.set_training(train_mode=False)
print(autograd.is_training())
```

在有些情况下，同一个模型在训练模式和预测模式下的行为并不相同。我们会在后面的章节详细介绍这些区别。


## 对Python控制流求梯度

使用MXNet的一个便利之处是，即使函数的计算图包含了Python的控制流（例如条件和循环控制），我们也有可能对变量求梯度。

考虑下面程序，其中包含Python的条件和循环控制。需要强调的是，这里循环（while循环）迭代的次数和条件判断（if语句）的执行都取决于输入`b`的值。

```{.python .input  n=3}
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们依然跟之前一样使用`record`函数记录计算，并调用`backward`函数求梯度。

```{.python .input  n=5}
a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

让我们仔细观察上面定义的$f$函数。事实上，给定任意输入`a`，其输出必然是 $f(a)= xa$的形式，且标量系数$x$的值取决于输入`a`。由于`c`有关`a`的梯度为$x =  c / a$，我们可以像下面这样验证对本例中控制流求梯度的结果是正确的。

```{.python .input  n=8}
a.grad == c / a
```

## 小结

* MXNet提供`autograd`包来自动化求导过程。
* MXNet的`autograd`包可以对正常的命令式程序进行求导。
* 我们可以通过`autograd.is_training()`来判断运行模式，并使用`autograd.set_training()`来设定运行模式。

## 练习

* 在本节对控制流求梯度的例子中，把变量`a`改成一个随机向量或矩阵。此时计算结果`c`不再是标量，运行结果将有何变化？该如何分析此结果？
* 自己重新设计一个对控制流求梯度的例子。运行并分析结果。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/744)

![](../img/qr_autograd.svg)
