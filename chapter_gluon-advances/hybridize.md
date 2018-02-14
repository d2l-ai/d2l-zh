# Hybridize：更快和更好移植

到目前为止我们看到的教程都使用了**命令式**的编程。你可能之前都从来没有听说这个词。不过，一直以来我们都是用这个方式写Python代码。

考虑下面这段代码：

```{.python .input}
def add(A, B):
    return A + B

def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G

fancy_func(1,2,3,4)
```

正如大家希望的那样，在运行`E = add(A, B)`的时候，我们实际上会做加法运算并返回结果。之后的指令`F = `和`G = `会跟在后面顺序执行。

这个编程方式的主要优点是很自然，大部分用户可能都不会意识还有别的其他的编程方式。但它的不足是可能会慢。这是因为我们不断跟（可能很慢的）Python的运行环境打交道。即使我们重复调用了`add`三次，我们还是会跟Python打三次交道。另外一点是，我们需要保存`E`和`F`的结果直到`fancy_func`结束。因为之前我们不知道是不是还会有谁用这些结果。

事实上这里不同的打开方式。其中一个叫**符号式**编程，大部分的深度学习框架包括Theano和TensorFlow用了这个方式。通常这个方式的程序需要下面三个步骤：

1. 定义计算流程
2. 编译成可执行的程序
3. 给定输入调用编译好的程序

我们重新实现上面的程序：

```{.python .input}
def add_str():
    return '''
def add(A, B):
    return A + B
'''

def fancy_func_str():
    return '''
def fancy_func(A, B, C, D):
    E = add(A, B)
    F = add(C, D)
    G = add(E, F)
    return G
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1,2,3,4))
'''

prog = evoke_str()
y = compile(prog, '', 'exec')
exec(y)
```

可以看到我们定义的三个函数都只是返回计算流程。之后我们编译再执行。在编译的时候系统能够看到整个程序，因此有更多的优化空间。例如编译的时候可以将程序改写成`print((1+2)+(3+4))`，甚至直接`print(10)`。这里我们不仅减少了函数调用，同时节省了内存。

总结一下

- **命令式编程更方便。** 当我们在Python里用一个命令式编程库时，我们在写Python代码，绝大部分代码很符合直觉。同样很容易逮BUG，因为我们可以拿到所有中间变量值，我们可以简单打印它们，或者使用Python的debug工具。

- **符号式编程更加高效而且更容易移植。** 之前我们提到在编译的时候系统可以容易的做更多的优化。另外一个好处是可以将程序变成一个与Python无关的格式，从而我们可以在非Python环境下运行。

## 使用`hybridize`来拿到两者的好处

大部分的深度学习框架通常在命令式和符号式之间二选一。例如Theano和它启发的后来者，例如TensorFlow，使用了符号式。Chainer和它的追随者PyTorch使用了命令式。在设计Gluon的时候我们问了这个问题：可能拿到命令式的*全部*好处，但仍然享受符号式的优势吗？从另一方面来说，用户应该用纯命令式的方法来使用Gluon进行开发和调试。但当需要产品级别的性能和部署的时候，我们可以将代码，至少大部分，转换成符号式来运行。

事实这一点可以做到。我们可以通过使用`HybridBlock`或者`HybridSequential`来构建神经网络。默认他们跟`Block`和`Sequential`一样使用命令式执行。当我们调用`.hybridize()`后，系统会转换成符号式来执行。事实上，所有Gluon里定义的层全是`HybridBlock`，这个意味着大部分的神经网络都可以享受符号式执行的优势。


## HybridSequential

我们之前学习了如何使用`Sequential`来串联多个层。如果你想要它跑得飞快，那你应该考虑替换成`HybridSequential`。

```{.python .input}
from mxnet.gluon import nn
from mxnet import nd

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

我们可以通过`hybridize`来编译和优化`HybridSequential`。

```{.python .input}
net.hybridize()
net(x)
```

注意到只有继承自`HybridBlock`的层才会被优化。`HybridSequential`和Gluon提供的层都是它的子类。如果一个层只是继承自`Block`，那么我们将跳过优化。我们会接下会讨论如何使用`HybridBlock`。


## 性能

我们比较`hybridize`前和后的计算时间来展示符号式执行的性能提升。这里我们计时1000次forward：

```{.python .input}
from time import time

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


## 获取符号式的程序

之前我们给`net`输入`NDArray`类型的`x`，然后`net(x)`会直接返回结果。对于调用过`hybridize()`后的网络，我们可以给它输入一个`Symbol`类型的变量，其会返回同样是`Symbol`类型的程序。

```{.python .input}
from mxnet import sym

x = sym.var('data')
y = net(x)
y
```

我们可以通过`export()`来保存这个程序到硬盘。它可以之后不仅被Python，同时也可以其他支持的前端语言，例如C++, Scala, R...，读取。

TODO(mli) `export`需要`mxnet>=0.11.1b20171015`，样例之后放进来。

## 通过HybridBlock深入理解`hybridize`工作机制

前面我们展示了通过`hybridize`我们可以获得更好的性能和更高的移植性。现在我们来解释这个是如何影响灵活性的。记得我们提过gluon里面的`Sequential`是`Block`的一个便利形式，同理，可以`HybridSequential`是`HybridBlock`的子类。跟`Block`需要实现`forward`方法不一样，对于`HybridBlock`我们需要实现`hybrid_forward`方法。

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

## 结论

通过`HybridSequential`和`HybridBlock`，我们可以简单的用`hybridize`来将将命令式的程序转成符号式程序。我们推荐大家尽可能的使用这个来获得最好的性能加速。


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1665)
