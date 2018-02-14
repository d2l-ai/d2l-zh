# 延迟执行

MXNet使用**延迟执行**来提升系统性能。绝大情况下我们不用知道它的存在，因为它不会对正常使用带来影响。但理解它的工作原理有助于开发更高效的程序。

延迟执行是指命令可以等到之后它的结果真正的需要的时候再执行。我们先来看一个例子：

```{.python .input  n=1}
a = 1 + 1
# some other things
print(a)
```

第一句对`a`赋值，再执行一些其指令后打印`a`的结果。因为这里我们可能很久以后才用`a`的值，所以我们可以把它的执行延迟到后面。这样的主要好处是在执行之前系统可以看到后面指令，从而有更多机会来对程序进行优化。例如如果`a`在被使用前被重新赋值了，那么我们可以不需要真正执行第一条语句。

在MXNet里，我们把用户打交道的部分叫做前端。例如这个教程里我们一直在使用Python前端写代码。除了Python外，MXNet还支持其他例如Scala，R，C++的前端。不管使用什么前端，MXNet的程序执行主要都在C++后端。前端只是把程序传给后端。后端有自己的线程来不断的收集任务，构造计算图，优化，并执行。本章我们介绍后端优化之一：延迟执行。

考虑下图的样例，我们在前端调用四条语句，它们被后端的线程分析依赖并构建成计算图。

![](../img/frontend-backend.svg)

在延迟执行中，前端执行前三个语句的时候，它仅仅是把任务放进后端的队列里就返回了。当在需要打印结果时，前端会等待后端线程把`c`的结果计算完。

这个设计的一个好处是前端，就是Python线程，不需要做实际计算工作，从而不管Python的性能如何，它对整个程序的影响会很小。只需要C++后端足够高效，那么不管前端语言性能如何，都可以提供一致的性能。

下面的例子通过计时来展示了延后执行的效果。可以看到，当`y=...`返回的时候并没有等待它真的被计算完。

```{.python .input  n=2}
from mxnet import nd
from time import time

start = time()
x = nd.random_uniform(shape=(2000,2000))
y = nd.dot(x, x)
print('workloads are queued:\t%f sec' % (time() - start))
print(y)
print('workloads are finished:\t%f sec' % (time() - start))
```

延迟执行大部分情况是对用户透明的。因为除非我们需要打印或者保存结果外，我们基本不需要关心目前是不是结果在内存里面已经计算好了。

事实上，只要数据是保存在NDArray里，而且使用MXNet提供的运算子，后端将默认使用延迟执行来获取最大的性能。

## 立即获取结果

除了前面介绍的`print`外，我们还有别的方法可以让前端线程等待直到结果完成。我们可以使用`nd.NDArray.wait_to_read()`等待直到特定结果完成，或者`nd.waitall()`等待所有前面结果完成。后者是测试性能常用方法。

```{.python .input  n=3}
start = time()
y = nd.dot(x, x)
y.wait_to_read()
time() - start
```

```{.python .input  n=4}
start = time()
y = nd.dot(x, x)
z = nd.dot(x, x)
nd.waitall()
time() - start
```

任何方法将内容从NDArray搬运到其他不支持延迟执行的数据结构里都会触发等待，例如`asnumpy()`, `asscalar()`

```{.python .input  n=5}
start = time()
y = nd.dot(x, x)
y.asnumpy()
time() - start
```

```{.python .input  n=6}
start = time()
y = nd.dot(x, x)
y.norm().asscalar()
time() - start
```

## 延迟执行带来的便利

下面例子中，我们不断的对`y`进行赋值。如果每次我们需要等到`y`的值，那么我们必须要要计算它。而在延迟执行里，系统有可能省略掉一些执行。

```{.python .input}
start = time()

for i in range(1000):
    y = x + 1
    y.wait_to_read()

print('No lazy evaluation: %f sec' % (time()-start))

start = time()
for i in range(1000):
    y = x + 1
nd.waitall()
print('With evaluation: %f sec' % (time()-start))
```

## 延迟执行带来的影响

在延迟执行里，只要最终结果是一致的，系统可能使用跟代码不一样的顺序来执行，例如假设我们写

```{.python .input  n=7}
a = 1
b = 2
a + b
```

第一句和第二句之间没有依赖，所以把`b=2`提前到`a=1`前执行也是可以的。但这样可能会导致内存使用的变化。

下面我们列举几个在训练和预测中常见的现象。一般每个批量我们都会评测一下，例如计算损失或者精度，其中会用到`asscalar`或者`asnumpy`。这样我们会每次仅仅将一个批量的任务放进后端系统执行。但如果我们去掉这些同步函数，会导致我们将大量的批量任务同时放进系统，从而可能导致系统占用过多资源。

为了演示这种情况，我们定义一个数据获取函数，它会打印什么数据是什么时候被请求的。

```{.python .input  n=8}
def get_data():
    start = time()
    batch_size = 1024
    for i in range(60):
        if i % 10 == 0:
            print('batch %d, time %f sec' %(i, time()-start))
        x = nd.ones((batch_size, 1024))
        y = nd.ones((batch_size,))
        yield x, y
```

使用两层网络和和L2损失函数作为样例

```{.python .input  n=9}
from mxnet import gluon
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Dense(1024, activation='relu'),
        nn.Dense(1024, activation='relu'),
        nn.Dense(1),
    )
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {})
loss = gluon.loss.L2Loss()
```

我们定义辅助函数来监测内存的使用（只能在Linux运行）

```{.python .input  n=10}
import os
import subprocess

def get_mem():
    """get memory usage in MB"""
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15])/1e3
```

现在我们可以做测试了。我们先试运行一次让系统把`net`的参数初始化（回忆[延后初始化](../chapter_gluon-basics/parameters.md)）。

```{.python .input  n=11}
for x, y in get_data():
    break
loss(y, net(x)).wait_to_read()
```

如果我们用`net`来做预测，正常情况下对每个批量的结果我们把它复制出NDArray，例如打印或者保存在磁盘上。这里我们简单使用`wait_to_read`来模拟。

```{.python .input  n=12}
mem = get_mem()

for x, y in get_data():
    loss(y, net(x)).wait_to_read()
nd.waitall()

print('Increased memory %f MB' % (get_mem() - mem))
```

假设我们不使用`wait_to_read()`， 那么前端会将所有批量的计算一次性的添加进后端。可以看到每个批量的数据都会在很短的时间内生成，同时在接下来的数秒钟内，我们看到了内存的增长（包括了在内存中保存所有`x`和`y`）。

```{.python .input  n=13}
mem = get_mem()

for x, y in get_data():
    loss(y, net(x))

nd.waitall()
print('Increased memory %f MB' % (get_mem() - mem))
```

同样对于训练，如果我们每次计算损失，那么就加入了同步

```{.python .input  n=14}
from mxnet import autograd

mem = get_mem()

total_loss = 0
for x, y in get_data():
    with autograd.record():
        L = loss(y, net(x))
    total_loss += L.sum().asscalar()
    L.backward()
    trainer.step(x.shape[0])

nd.waitall()
print('Increased memory %f MB' % (get_mem() - mem))
```

但如果不去掉同步，同样会首先把数据全部生成好，导致占用大量内存。

```{.python .input  n=15}
from mxnet import autograd

mem = get_mem()

total_loss = 0
for x, y in get_data():
    with autograd.record():
        L = loss(y, net(x))
    L.backward()
    trainer.step(x.shape[0])

nd.waitall()
print('Increased memory %f MB' % (get_mem() - mem))
```

## 总结

延后执行使得系统有更多空间来做性能优化。但我们推荐每个批量里至少有一个同步函数，例如对损失函数进行评估，来避免将过多任务同时丢进后端系统。

## 练习

为什么同步版本的训练中，我们看到了内存使用的大量下降？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1881)
