# 惰性计算

MXNet使用惰性计算（lazy evaluation）来提升计算性能。理解它的工作原理既有助于开发更高效的程序，又有助于在内存资源有限的情况下主动降低计算性能从而减小内存开销。

我们先导入本节中实验需要的包。

```{.python .input  n=1}
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
import os
import subprocess
from time import time
```

惰性计算的含义是，程序中定义的计算仅在结果真正被取用的时候才执行。我们先看下面这个例子。

```{.python .input  n=2}
a = 1 + 1
a = 2 + 2
a = 3 + 3
print(a)
```

在这个例子中，前三句都在对变量`a`赋值，最后一句打印变量`a`的计算结果。事实上，我们可以把三条赋值语句的计算延迟到即将执行打印语句之前。这样的主要好处是系统在即将计算变量`a`时已经看到了全部有关计算`a`的语句，从而有更多空间优化计算。例如，这里我们并不需要对前两条赋值语句做计算。


## MXNet中的惰性计算

广义上，MXNet包括用户直接用来交互的前端和系统用来执行计算的后端。例如，用户可以使用不同的前端语言编写MXNet程序，像Python、R、Scala和C++。无论使用何种前端编程语言，MXNet程序的执行主要都发生在C++实现的后端。换句话说，用户写好的前端MXNet程序会传给后端执行计算。后端有自己的线程来不断收集任务，构造、优化并执行计算图。后端优化的方式有很多种，其中包括本章将介绍的惰性计算。

假设我们在前端调用以下四条语句。MXNet后端的线程会分析它们的依赖关系并构建出如图8.1所示的计算图。

```{.python .input  n=3}
a = nd.ones((1, 2))
b = nd.ones((1, 2))
c = a * b + 2
print(c)
```

![MXNet后端的计算图](../img/frontend-backend.svg)

在惰性计算中，前端执行前三条语句的时候，仅仅是把任务放进后端的队列里就返回了。当最后一条语句需要打印计算结果时，前端会等待后端线程把`c`的结果计算完。此设计的一个好处是，这里的Python前端线程不需要做实际计算。因此，无论Python的性能如何，它对整个程序性能的影响会很小。只要C++后端足够高效，那么不管前端语言性能如何，MXNet都可以提供一致的高性能。

下面的例子通过计时来展示惰性计算的效果。可以看到，当`y = nd.dot(x, x)`返回的时候并没有等待它真正被计算完。

```{.python .input  n=4}
start = time()
x = nd.random.uniform(shape=(2000, 2000))
y = nd.dot(x, x)
print('workloads are queued: %f sec' % (time() - start))
print(y)
print('workloads are completed: %f sec' % (time() - start))
```

的确，除非我们需要打印或者保存计算结果，我们基本无需关心目前结果在内存中是否已经计算好了。只要数据是保存在NDArray里并使用MXNet提供的运算符，MXNet后端将默认使用惰性计算来获取最高的计算性能。


## 用同步函数让前端等待计算结果

除了前面介绍的`print`外，我们还有其他方法让前端线程等待后端的计算结果完成。我们可以使用`wait_to_read`函数让前端等待某个的NDArray的计算结果完成，再执行前端中后面的语句。或者，我们可以用`waitall`函数令前端等待前面所有计算结果完成。后者是性能测试中常用的方法。

下面是使用`wait_to_read`的例子。输出用时包含了`y`的计算时间。

```{.python .input  n=5}
start = time()
y = nd.dot(x, x)
y.wait_to_read()
time() - start
```

下面是使用`waitall`的例子。输出用时包含了`y`和`z`的计算时间。

```{.python .input  n=6}
start = time()
y = nd.dot(x, x)
z = nd.dot(x, x)
nd.waitall()
time() - start
```

此外，任何将NDArray转换成其他不支持惰性计算的数据结构的操作都会让前端等待计算结果。例如当我们调用`asnumpy`和 `asscalar`函数时。

```{.python .input  n=7}
start = time()
y = nd.dot(x, x)
y.asnumpy()
time() - start
```

```{.python .input  n=8}
start = time()
y = nd.dot(x, x)
y.norm().asscalar()
time() - start
```

由于`asnumpy`、`asscalar`和`print`函数会触发让前端等待后端计算结果的行为，我们通常把这类函数称作同步函数。

## 使用惰性计算提升计算性能

在下面例子中，我们不断对`y`进行赋值。如果不使用惰性计算，我们可以在for循环内使用`wait_to_read`做1000次赋值计算。在惰性计算中，MXNet会省略掉一些不必要执行。

```{.python .input  n=9}
start = time()
for i in range(1000):
    y = x + 1
    y.wait_to_read()
print('no lazy evaluation: %f sec' % (time() - start))

start = time()
for i in range(1000):
    y = x + 1
nd.waitall()
print('with lazy evaluation: %f sec' % (time() - start))
```

## 惰性计算对内存使用的影响

在惰性计算中，只要不影响最终计算结果，MXNet后端不一定会按前端代码中定义的执行顺序来执行。

考虑下面的例子。

```{.python .input  n=10}
a = 1
b = 2
a + b
```

上例中，第一句和第二句之间没有依赖。所以，把`b = 2`提前到`a = 1`前执行也是可以的。但这样可能会导致内存使用的变化。

为了解释惰性计算对内存使用的影响，让我们先回忆一下前面章节的内容。在前面章节中实现的模型训练过程中，我们通常会在每个小批量上评测一下模型，例如模型的损失或者精度。细心的你也许发现了，这类评测常用到同步函数，例如`asscalar`或者`asnumpy`。如果去掉这些同步函数，前端会将大量的小批量计算任务同时放进后端，从而可能导致较大的内存开销。当我们在每个小批量上都使用同步函数时，前端在每次迭代时仅会将一个小批量的任务放进后端执行计算。换言之，我们通过适当减少惰性计算，从而减小内存开销。这也是一种“时间换空间”的策略。

由于深度学习模型通常比较大，而内存资源通常有限，我们建议大家在训练模型时对每个小批量都使用同步函数。类似地，在使用模型预测时，为了减小内存开销，我们也建议大家对每个小批量预测时都使用同步函数，例如直接打印出当前批量的预测结果。

下面我们来演示惰性计算对内存使用的影响。我们先定义一个数据获取函数，它会从被调用时开始计时，并定期打印到目前为止获取数据批量总共耗时。

```{.python .input  n=11}
num_batches = 41
def get_data():
    start = time()
    batch_size = 1024
    for i in range(num_batches):
        if i % 10 == 0:
            print('batch %d, time %f sec' % (i, time() - start))
        x = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield x, y
```

以下定义多层感知机、优化器和损失函数。

```{.python .input  n=12}
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(1),
    )
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate':0.005})
square_loss = gluon.loss.L2Loss()
```

这里定义辅助函数来监测内存的使用。需要注意的是，这个函数只能在Linux运行。

```{.python .input  n=13}
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
```

现在我们可以做测试了。我们先试运行一次让系统把`net`的参数初始化。相关内容请参见[“模型参数”](../chapter_gluon-basics/parameters.md)一节。

```{.python .input  n=14}
for x, y in get_data():
    break
square_loss(y, net(x)).wait_to_read()
```

如果我们用`net`来做预测，通常情况下我们可以把每个小批量的结果通过同步函数从NDArray格式中取出，例如打印或者保存在磁盘上。这里我们使用`wait_to_read`来模拟。此时，每个小批量的生成间隔较长，不过内存开销较小。

```{.python .input  n=15}
mem = get_mem()
for x, y in get_data():
    square_loss(y, net(x)).wait_to_read()
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

假设我们不使用`wait_to_read()`，默认惰性计算下，前端会将所有小批量计算一次性添加进后端。可以看到，每个小批量的生成间隔较短。然而，此时内存开销较大：它包括了在内存中保存的所有`x`和`y`。

```{.python .input  n=16}
mem = get_mem()
for x, y in get_data():
    square_loss(y, net(x))
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

对于训练`net`来说，假设我们希望打印每个迭代周期后的模型损失，我们可以自然地使用同步函数`asscalar`和`print`来避免内存开销过大。

```{.python .input  n=17}
mem = get_mem()
for epoch in range(1, 3):
    total_loss = 0
    for x, y in get_data():
        with autograd.record():
            loss = square_loss(y, net(x))
        total_loss += loss.mean().asscalar()
        loss.backward()
        trainer.step(x.shape[0])
    print('epoch', epoch, ' loss: ', total_loss / num_batches)
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

但如果去掉同步函数，训练过程中可能会导致内存开销过大。

```{.python .input  n=18}
mem = get_mem()
for epoch in range(1, 3):
    for x, y in get_data():
        with autograd.record():
            loss = square_loss(y, net(x))
        loss.backward()
        trainer.step(x.shape[0])
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

## 小结

* MXNet包括用户直接用来交互的前端和系统用来执行计算的后端。

* MXNet能够通过惰性计算提升计算性能。

* 我们建议使用每个小批量训练或预测时至少使用一个同步函数，从而避免将过多计算任务同时添加进后端。


## 练习

* 本节中提到了“时间换空间”的策略。本节中哪些部分与“空间换时间”的策略有关？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1881)

![](../img/qr_lazy-evaluation.svg)
