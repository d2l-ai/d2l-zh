无论是当数据集很大还是计算资源或应用有约束条件时，深度学习十分关注计算性能。本章将重点介绍影响计算性能的重要因子：命令式编程、符号式编程、惰性计算、自动并行计算和多GPU计算。通过本章的学习，读者很可能进一步提升已有模型的计算性能，例如在不影响模型精度的前提下减少模型的训练时间。


# 命令式和符号式混合编程

其实，到目前为止我们一直都在使用命令式编程：使用编程语句改变程序状态。考虑下面这段简单的命令式编程代码。

```{.python .input}
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

fancy_func(1, 2, 3, 4)
```

和我们预期的一样，在运行`e = add(a, b)`时，Python会做加法运算并将结果存储在变量`e`，从而令程序的状态发生了改变。类似地，后面的两个语句`f = add(c, d)`和`g = add(e, f)`会依次做加法运算并存储变量。

虽然使用命令式编程很方便，但它的运行可能会慢。一方面，即使`fancy_func`函数中的`add`是被重复调用的函数，Python也会逐一执行这三个函数调用语句。另一方面，我们需要保存变量`e`和`f`的值直到`fancy_func`中所有语句执行结束。这是因为在执行`e = add(a, b)`和`f = add(c, d)`之前我们并不知道变量`e`和`f`是否会被程序的其他部分使用。

与命令式编程不同，符号式编程通常在计算流程完全定义好后才被执行。大部分的深度学习框架，例如Theano和TensorFlow，都使用了符号式编程。通常，符号式编程的程序需要下面三个步骤：

1. 定义计算流程；
2. 把计算流程编译成可执行的程序；
3. 给定输入，调用编译好的程序执行。

下面我们用符号式编程重新实现本节开头给出的命令式编程代码。

```{.python .input}
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

以上定义的三个函数都只是返回计算流程。最后，我们编译完整的计算流程并运行。由于在编译时系统能够完整地看到整个程序，因此有更多空间优化计算。例如，编译的时候可以将程序改写成`print((1 + 2) + (3 + 4))`，甚至直接改写成`print(10)`。这样不仅减少了函数调用，还节省了内存。

总结一下，

- 命令式编程更方便。当我们在Python里使用命令式编程时，大部分代码编写起来都符合直觉。同时，命令式编程更容易除错。这是因为我们可以很方便地拿到所有的中间变量值并打印，或者使用Python的除错工具。

- 符号式编程更高效并更容易移植。一方面，在编译的时候系统可以容易地做更多优化；另一方面，符号式编程可以将程序变成一个与Python无关的格式，从而可以使程序在非Python环境下运行。


## 混合式编程取二者之长

大部分的深度学习框架在命令式编程和符号式编程之间二选一。例如Theano和受其启发的后来者TensorFlow使用了符号式编程；Chainer和它的追随者PyTorch使用了命令式编程。开发人员在设计Gluon时思考了这个问题：有没有可能既拿到命令式编程的好处，又享受符号式编程的优势？开发者们认为，用户应该用纯命令式编程进行开发和调试；当需要产品级别的性能和部署时，用户可以将至少大部分程序转换成符号式来运行。

值得强调的是，Gluon可以通过混合式编程做到这一点。在混合式编程中，我们可以通过使用HybridBlock或者`HybridSequential`类构建模型。默认情况下，它们和Block或者`Sequential`类一样依据命令式编程的方式执行。当我们调用`hybridize`函数后，Gluon会转换成依据符号式编程的方式执行。事实上，绝大多数模型都可以享受符号式编程的优势。

本节将通过实验展示混合式编程的魅力。首先，导入实验所需的包。

```{.python .input}
from mxnet.gluon import nn
from mxnet import nd, sym
from time import time
```

## 使用`HybridSequential`类构造模型

我们之前学习了如何使用`Sequential`类来串联多个层。为了使用混合式编程，下面我们将`Sequential`类替换成`HybridSequential`类。

```{.python .input}
def get_net():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Dense(256, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(2)
        )
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

我们可以通过调用`hybridize`函数来编译和优化`HybridSequential`实例中串联的层的计算。模型的计算结果不变。

```{.python .input}
net.hybridize()
net(x)
```

需要注意的是，只有继承HybridBlock的层才会被优化。例如，`HybridSequential`类和Gluon提供的`Dense`层都是HybridBlock的子类，它们都会被优化计算。如果一个层只是继承自Block而不是HybridBlock，那么它将不会被优化。我们接下会讨论如何使用HybridBlock。


### 性能

我们比较`hybridize`前和后的计算时间来展示符号式执行的性能提升。这里我们计时1000次forward：

```{.python .input}
def bench(net, x):
    start = time()
    for i in range(1000):
        y = net(x)
    # 等待所有计算完成
    nd.waitall()
    return time() - start

net = get_net()
print('Before hybridizing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec'%(bench(net, x)))
```

可以看到`hybridize`提供近似两倍的加速。


### 获取符号式的程序

之前我们给`net`输入NDArray类型的`x`，然后`net(x)`会直接返回结果。对于调用过`hybridize()`后的网络，我们可以给它输入一个`Symbol`类型的变量，其会返回同样是`Symbol`类型的程序。

```{.python .input}
x = sym.var('data')
y = net(x)
y
```

我们可以通过`export()`来保存这个程序到硬盘。它可以之后不仅被Python，同时也可以其他支持的前端语言，例如C++, Scala, R...，读取。

TODO(mli) `export`需要`mxnet>=0.11.1b20171015`，样例之后放进来。

## 通过HybridBlock深入理解`hybridize`工作机制

前面我们展示了通过`hybridize`我们可以获得更好的性能和更高的移植性。现在我们来解释这个是如何影响灵活性的。记得我们提过Gluon里面的`Sequential`是`Block`的一个便利形式，同理，可以`HybridSequential`是`HybridBlock`的子类。跟`Block`需要实现`forward`方法不一样，对于`HybridBlock`我们需要实现`hybrid_forward`方法。

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(10)
            self.fc2 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print(F)
        print(x)
        x = F.relu(self.fc1(x))
        print(x)
        return self.fc2(x)
```

`hybrid_forward`方法加入了额外的输入`F`，它使用了MXNet的一个独特的特征。MXNet有一个符号式的API (`symbol`) 和命令式的API (`ndarray`)。这两个接口里面的函数基本是一致的。系统会根据输入来决定`F`是使用`symbol`还是`ndarray`。

我们实例化一个样例，然后可以看到默认`F`是使用`ndarray`。而且我们打印出了输入和第一层relu的输出。

```{.python .input}
net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
y = net(x)
```

再运行一次会得到同样的结果。

```{.python .input}
y = net(x)
```

接下来看看`hybridze`后会发生什么。

```{.python .input}
net.hybridize()
y = net(x)
```

可以看到：

1. `F`变成了`symbol`.
2. 即使输入数据还是`NDArray`的类型，但`hybrid_forward`里不论是输入还是中间输出，全部变成了`Symbol`

再运行一次看看

```{.python .input}
y = net(x)
```

可以看到什么都没有输出。这是因为第一次`net(x)`的时候，会先将输入替换成`Symbol`来构建符号式的程序，之后运行的时候系统将不再访问Python的代码，而是直接在C++后端执行这个符号式程序。这是为什么`hybridze`后会变快的一个原因。

但它可能的问题是我们损失写程序的灵活性。因为Python的代码只执行一次，而且是符号式的执行，那么使用`print`来调试，或者使用`if`和`for`来做复杂的控制都不可能了。

## 小结

* 通过`HybridSequential`和`HybridBlock`，我们可以简单的用`hybridize`来将将命令式的程序转成符号式程序。我们推荐大家尽可能的使用这个来获得最好的性能加速。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1665)

![](../img/qr_hybridize.svg)
