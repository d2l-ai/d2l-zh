# 使用autograd来自动求导

在机器学习中，我们通常使用**梯度下降（gradient descent）**来更新模型参数从而求解。损失函数关于模型参数的梯度指向一个可以降低损失函数值的方向，我们不断地沿着梯度的方向更新模型从而最小化损失函数。虽然梯度计算比较直观，但对于复杂的模型，例如多达数十层的神经网络，手动计算梯度非常困难。

为此MXNet提供autograd包来自动化求导过程。虽然大部分的深度学习框架要求编译计算图来自动求导，`mxnet.autograd`可以对正常的命令式程序进行求导，它每次在后端实时创建计算图，从而可以立即得到梯度的计算方法。

下面让我们一步步介绍这个包。我们先导入`autograd`。

```{.python .input  n=2}
import mxnet.ndarray as nd
import mxnet.autograd as ag
```

## 为变量附上梯度

假设我们想对函数 $f=2\times x^2$ 求关于 $x$ 的导数。我们先创建变量`x`，并赋初值。

```{.python .input}
x = nd.array([[1, 2], [3, 4]])
```

当进行求导的时候，我们需要一个地方来存`x`的导数，这个可以通过NDArray的方法`attach_grad()`来要求系统申请对应的空间。

```{.python .input}
x.attach_grad()
```

下面定义`f`。默认条件下，MXNet不会自动记录和构建用于求导的计算图，我们需要使用autograd里的`record()`函数来显式的要求MXNet记录我们需要求导的程序。

```{.python .input}
with ag.record():
    y = x * 2
    z = y * x
```

接下来我们可以通过`z.backward()`来进行求导。如果`z`不是一个标量，那么`z.backward()`等价于`nd.sum(z).backward()`.

```{.python .input}
z.backward()
```

现在我们来看求出来的导数是不是正确的。注意到`y = x * 2`和`z = x * y`，所以`z`等价于`2 * x * x`。它的导数那么就是 $\frac{dz}{dx} = 4 \times {x}$ 。

```{.python .input}
print('x.grad: ', x.grad)
x.grad == 4*x
```

## 对控制流求导

命令式的编程的一个便利之处是几乎可以对任意的可导程序进行求导，即使里面包含了Python的控制流。考虑下面程序，里面包含控制流`for`和`if`，但循环迭代的次数和判断语句的执行都是取决于输入的值。不同的输入会导致这个程序的执行不一样。（对于计算图框架来说，这个对应于动态图，就是图的结构会根据输入数据不同而改变）。

```{.python .input  n=3}
def f(a):
    b = a * 2
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们可以跟之前一样使用`record`记录和`backward`求导。

```{.python .input  n=5}
a = nd.random_normal(shape=3)
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()
```

注意到给定输入`a`，其输出 $\\f(a)= {xa}$，$x$ 的值取决于输入`a`。所以有 $\frac{df}{da} = {x}$，我们可以很简单地评估自动求导的导数：

```{.python .input  n=8}
a.grad == c/a
```

## 头梯度和链式法则

**注意：读者可以跳过这一小节，不会影响阅读之后的章节**

当我们在一个`NDArray`上调用`backward`方法时，例如`y.backward()`，此处`y`是一个关于`x`的函数，我们将求得`y`关于`x`的导数。数学家们会把这个求导写成 $\frac{dy(x)}{dx}$ 。还有些更复杂的情况，比如`z`是关于`y`的函数，且`y`是关于`x`的函数，我们想对`z`关于`x`求导，也就是求 $\frac{d}{dx} z(y(x))$ 的结果。回想一下链式法则，我们可以得到$\frac{d}{dx} z(y(x)) = \frac{dz(y)}{dy} \frac{dy(x)}{dx}$。当`y`是一个更大的`z`函数的一部分，并且我们希望求得 $\frac{dz}{dx}$ 保存在`x.grad`中时，我们可以传入**头梯度（head gradient）** $\frac{dz}{dy}$ 的值作为`backward()`方法的输入参数，系统会自动应用链式法则进行计算。这个参数的默认值是`nd.ones_like(y)`。关于链式法则的详细解释，请参阅[Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)。

```{.python .input}
with ag.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)
```

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/744)
