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


## 混合式编程取两者之长

大部分的深度学习框架在命令式编程和符号式编程之间二选一。例如Theano和受其启发的后来者TensorFlow使用了符号式编程；Chainer和它的追随者PyTorch使用了命令式编程。开发人员在设计Gluon时思考了这个问题：有没有可能既拿到命令式编程的好处，又享受符号式编程的优势？开发者们认为，用户应该用纯命令式编程进行开发和调试；当需要产品级别的性能和部署时，用户可以将至少大部分程序转换成符号式来运行。

值得强调的是，Gluon可以通过混合式编程做到这一点。在混合式编程中，我们可以通过使用HybridBlock或者HybridSequential类构建模型。默认情况下，它们和Block或者Sequential类一样依据命令式编程的方式执行。当我们调用`hybridize`函数后，Gluon会转换成依据符号式编程的方式执行。事实上，绝大多数模型都可以享受符号式编程的优势。

本节将通过实验展示混合式编程的魅力。首先，导入本节中实验所需的包或模块。

```{.python .input}
from mxnet import nd, sym
from mxnet.gluon import nn
from time import time
```

## 使用HybridSequential类构造模型

我们之前学习了如何使用Sequential类来串联多个层。为了使用混合式编程，下面我们将Sequential类替换成HybridSequential类。

```{.python .input}
def get_net():
    net = nn.HybridSequential()
    net.add(
        nn.Dense(256, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(2)
    )
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

我们可以通过调用`hybridize`函数来编译和优化HybridSequential实例中串联的层的计算。模型的计算结果不变。

```{.python .input}
net.hybridize()
net(x)
```

需要注意的是，只有继承HybridBlock的层才会被优化。例如，HybridSequential类和Gluon提供的Dense类都是HybridBlock的子类，它们都会被优化计算。如果一个层只是继承自Block而不是HybridBlock类，那么它将不会被优化。我们接下会讨论如何使用HybridBlock类。


### 性能

我们比较调用`hybridize`函数前后的计算时间来展示符号式编程的性能提升。这里我们计时1000次`net`模型计算。在`net`调用`hybridize`函数前后，它分别依据命令式编程和符号式编程做模型计算。

```{.python .input}
def benchmark(net, x):
    start = time()
    for i in range(1000):
        _ = net(x)
    # 等待所有计算完成。
    nd.waitall()
    return time() - start

net = get_net()
print('before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridizing: %.4f sec' % (benchmark(net, x)))
```

由上面结果可见，在一个HybridSequential实例调用`hybridize`函数后，它可以通过符号式编程提升计算性能。


### 获取符号式程序

在模型`net`根据输入计算模型输出后，例如`benchmark`函数中的`net(x)`，我们就可以通过`export`函数来保存符号式程序和模型参数到硬盘。

```{.python .input}
net.export('my_mlp')
```

此时生成的.json和.params文件分别为符号式程序和模型参数。它们可以被Python或MXNet支持的其他前端语言读取，例如C++。这样，我们就可以很方便地使用其他前端语言或在其他设备上部署训练好的模型。同时，由于部署时使用的是基于符号式编程的程序，计算性能往往比基于命令式编程更好。

在MXNet中，符号式程序指的是Symbol类型的程序。我们知道，当给`net`提供NDArray类型的输入`x`后，`net(x)`会根据`x`直接计算模型输出并返回结果。对于调用过`hybridize`函数后的模型，我们还可以给它输入一个Symbol类型的变量，`net(x)`会返回同样是Symbol类型的程序。

```{.python .input}
x = sym.var('data')
net(x)
```

## 使用HybridBlock类构造模型

和Sequential类与Block之间的关系一样，HybridSequential类是HybridBlock的子类。跟Block实例需要实现`forward`函数不太一样的是，对于HybridBlock实例我们需要实现`hybrid_forward`函数。

前面我们展示了调用`hybridize`函数后的模型可以获得更好的计算性能和移植性。另一方面，调用`hybridize`后的模型会影响灵活性。为了解释这一点，我们先使用HybridBlock构造模型。

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)
```

在继承HybridBlock类时，我们需要在`hybrid_forward`函数中添加额外的输入`F`。我们知道，MXNet既有基于命令式编程的NDArray类，又有基于符号式编程的Symbol类。由于这两个类的函数基本一致，MXNet会根据输入来决定`F`使用NDArray或Symbol。

下面创建了一个HybridBlock实例。可以看到默认下`F`使用NDArray。而且，我们打印出了输入`x`和使用ReLU激活函数的隐藏层的输出。

```{.python .input}
net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
net(x)
```

再运行一次会得到同样的结果。

```{.python .input}
net(x)
```

接下来看看调用`hybridize`函数后会发生什么。

```{.python .input}
net.hybridize()
net(x)
```

可以看到，`F`变成了Symbol。而且，虽然输入数据还是NDArray，但`hybrid_forward`函数里，相同输入和中间输出全部变成了Symbol。

再运行一次看看。

```{.python .input}
net(x)
```

可以看到`hybrid_forward`函数里定义的三行打印语句都没有打印任何东西。这是因为上一次在调用`hybridize`函数后运行`net(x)`的时候，符号式程序已经得到。之后再运行`net(x)`的时候MXNet将不再访问Python代码，而是直接在C++后端执行符号式程序。这也是调用`hybridize`后模型计算性能会提升的一个原因。但它可能的问题是我们损失了写程序的灵活性。在上面这个例子中，如果我们希望使用那三行打印语句调试代码，执行符号式程序时会跳过它们无法打印。此外，对于少数像`asnumpy`这样的Symbol不支持的函数，以及像`a += b`和`a[:] = a + b`（需改写为`a = a + b`）这样的in-place操作，我们无法在`hybrid_forward`函数中使用并在调用`hybridize`函数后进行模型计算。


## 小结

* 命令式编程和符号式编程各有优劣。MXNet通过混合式编程取两者之长。
* 通过HybridSequential类和HybridBlock构建的模型可以调用`hybridize`来将将命令式程序转成符号式程序。我们建议大家使用这种方法获得计算性能的提升。


## 练习

* 在本节HybridNet类`hybrid_forward`函数中第一行添加`x.asnumpy()`，运行本节全部代码，观察报错的位置和错误类型。
* 回顾前面几章中你感兴趣的模型，改用HybridBlock或HybridSequential类实现。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1665)

![](../img/qr_hybridize.svg)
