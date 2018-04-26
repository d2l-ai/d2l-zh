# 惰性计算

MXNet使用惰性计算（lazy evaluation）来提升计算性能。理解它的工作原理既有助于开发更高效的程序，又有助于在内存资源有限的情况下主动降低计算性能从而减小内存消耗。

我们先导入本节中实验需要的包。

```{.python .input  n=1}
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
import os
import subprocess
from time import time
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/ubuntu/miniconda3/lib/python3.6/site-packages/urllib3/contrib/pyopenssl.py:46: DeprecationWarning: OpenSSL.rand is deprecated - you should use os.urandom instead\n  import OpenSSL.SSL\n"
 }
]
```

惰性计算的含义是，程序中定义的计算仅在结果真正被取用的时候才执行。我们先看下面这个例子。

```{.python .input  n=2}
a = 1 + 1
a = 2 + 2
a = 3 + 3
print(a)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "6\n"
 }
]
```

在这个例子中，前三句都在对变量`a`赋值，最后一句打印变量`a`的计算结果。事实上，我们可以把三个赋值语句的计算延迟到即将执行打印语句之前。这样的主要好处是系统在即将计算变量`a`时已经看到了全部有关计算`a`的指令，从而有更多空间优化计算。例如，这里我们并不需要对前两个赋值语句做计算。


## MXNet中的惰性计算

广义上，MXNet包括用户直接用来交互的前端和系统用来执行计算的后端。例如，用户可以使用不同的前端语言编写MXNet程序，例如Python、R、Scala和C++。无论使用何种前端编程语言，MXNet程序的执行主要都发生在C++后端。换句话说，用户写好的前端MXNet程序会传给后端执行计算。后端有自己的线程来不断收集任务，构造、优化并执行计算图。

本章我们介绍后端优化之一：延迟执行。

考虑下图的样例，我们在前端调用四条语句，它们被后端的线程分析依赖并构建成计算图。

```{.python .input  n=3}
a = nd.ones((1, 2))
b = nd.ones((1, 2))
c = a * b + 2
print(c)
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[3. 3.]]\n<NDArray 1x2 @cpu(0)>\n"
 }
]
```

![](../img/frontend-backend.svg)

在延迟执行中，前端执行前三个语句的时候，它仅仅是把任务放进后端的队列里就返回了。当在需要打印结果时，前端会等待后端线程把`c`的结果计算完。

这个设计的一个好处是前端，就是Python线程，不需要做实际计算工作，从而不管Python的性能如何，它对整个程序的影响会很小。只需要C++后端足够高效，那么不管前端语言性能如何，都可以提供一致的性能。

下面的例子通过计时来展示了延后执行的效果。可以看到，当`y=...`返回的时候并没有等待它真的被计算完。

```{.python .input  n=4}
start = time()
x = nd.random.uniform(shape=(2000, 2000))
y = nd.dot(x, x)
print('workloads are queued:\t%f sec' % (time() - start))
print(y)
print('workloads are finished:\t%f sec' % (time() - start))
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "workloads are queued:\t0.001552 sec\n\n[[501.1584  508.29724 495.65237 ... 492.84705 492.69092 490.0481 ]\n [508.81058 507.1822  495.1743  ... 503.10526 497.29315 493.67917]\n [489.56598 499.47015 490.17722 ... 490.99945 488.05008 483.28836]\n ...\n [484.0019  495.7179  479.92142 ... 493.69952 478.89194 487.2074 ]\n [499.64932 507.65094 497.5938  ... 493.0474  500.74512 495.82712]\n [516.0143  519.1715  506.354   ... 510.08878 496.35608 495.42523]]\n<NDArray 2000x2000 @cpu(0)>\nworkloads are finished:\t0.197700 sec\n"
 }
]
```

延迟执行大部分情况是对用户透明的。因为除非我们需要打印或者保存结果外，我们基本不需要关心目前是不是结果在内存里面已经计算好了。

事实上，只要数据是保存在NDArray里，而且使用MXNet提供的运算子，后端将默认使用延迟执行来获取最大的性能。

## 立即获取结果

除了前面介绍的`print`外，我们还有别的方法可以让前端线程等待直到结果完成。我们可以使用`nd.NDArray.wait_to_read()`等待直到特定结果完成，或者`nd.waitall()`等待所有前面结果完成。后者是测试性能常用方法。

```{.python .input  n=5}
start = time()
y = nd.dot(x, x)
y.wait_to_read()
time() - start
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "0.13035035133361816"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=6}
start = time()
y = nd.dot(x, x)
z = nd.dot(x, x)
nd.waitall()
time() - start
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "0.2879159450531006"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

任何方法将内容从NDArray搬运到其他不支持延迟执行的数据结构里都会触发等待，例如`asnumpy()`, `asscalar()`

```{.python .input  n=7}
start = time()
y = nd.dot(x, x)
y.asnumpy()
time() - start
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "0.13819432258605957"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
start = time()
y = nd.dot(x, x)
y.norm().asscalar()
time() - start
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "0.16870975494384766"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 延迟执行带来的便利

下面例子中，我们不断的对`y`进行赋值。如果每次我们需要等到`y`的值，那么我们必须要要计算它。而在延迟执行里，系统有可能省略掉一些执行。

```{.python .input  n=9}
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

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "No lazy evaluation: 0.905849 sec\nWith evaluation: 0.700691 sec\n"
 }
]
```

## 延迟执行带来的影响

在延迟执行里，只要最终结果是一致的，系统可能使用跟代码不一样的顺序来执行，例如假设我们写

```{.python .input  n=10}
a = 1
b = 2
a + b
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "3"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

第一句和第二句之间没有依赖，所以把`b=2`提前到`a=1`前执行也是可以的。但这样可能会导致内存使用的变化。

下面我们列举几个在训练和预测中常见的现象。一般每个批量我们都会评测一下，例如计算损失或者精度，其中会用到`asscalar`或者`asnumpy`函数。这样我们会每次仅仅将一个批量的任务放进后端系统执行。但如果我们去掉这些同步函数，会导致我们将大量的批量任务同时放进系统，从而可能导致系统占用过多资源。

为了演示这种情况，我们定义一个数据获取函数，它会打印什么数据是什么时候被请求的。

```{.python .input  n=11}
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

```{.python .input  n=12}
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

```{.python .input  n=13}
def get_mem():
    """get memory usage in MB"""
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15])/1e3
```

现在我们可以做测试了。我们先试运行一次让系统把`net`的参数初始化（回忆[延后初始化](../chapter_gluon-basics/parameters.md)）。

```{.python .input  n=14}
for x, y in get_data():
    break
loss(y, net(x)).wait_to_read()
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "batch 0, time 0.000005 sec\n"
 }
]
```

如果我们用`net`来做预测，正常情况下对每个批量的结果我们把它复制出NDArray，例如打印或者保存在磁盘上。这里我们简单使用`wait_to_read`来模拟。

```{.python .input  n=15}
mem = get_mem()

for x, y in get_data():
    loss(y, net(x)).wait_to_read()
nd.waitall()

print('Increased memory %f MB' % (get_mem() - mem))
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "batch 0, time 0.000005 sec\nbatch 10, time 0.504875 sec\nbatch 20, time 0.934479 sec\nbatch 30, time 1.328565 sec\nbatch 40, time 1.740662 sec\nbatch 50, time 2.340153 sec\nIncreased memory 0.032000 MB\n"
 }
]
```

假设我们不使用`wait_to_read()`， 那么前端会将所有批量的计算一次性的添加进后端。可以看到每个批量的数据都会在很短的时间内生成，同时在接下来的数秒钟内，我们看到了内存的增长（包括了在内存中保存所有`x`和`y`）。

```{.python .input  n=16}
mem = get_mem()

for x, y in get_data():
    loss(y, net(x))

nd.waitall()
print('Increased memory %f MB' % (get_mem() - mem))
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "batch 0, time 0.000005 sec\nbatch 10, time 0.024832 sec\nbatch 20, time 0.038855 sec\nbatch 30, time 0.052305 sec\nbatch 40, time 0.062469 sec\nbatch 50, time 0.068811 sec\nIncreased memory 241.716000 MB\n"
 }
]
```

同样对于训练，如果我们每次计算损失，那么就加入了同步

```{.python .input  n=17}
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

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "batch 0, time 0.000013 sec\nbatch 10, time 1.250264 sec\nbatch 20, time 2.410555 sec\nbatch 30, time 3.649215 sec\nbatch 40, time 4.798485 sec\nbatch 50, time 5.957984 sec\nIncreased memory -224.596000 MB\n"
 }
]
```

但如果不去掉同步，同样会首先把数据全部生成好，导致占用大量内存。

```{.python .input  n=18}
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

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "batch 0, time 0.000005 sec\nbatch 10, time 0.032923 sec\nbatch 20, time 0.057777 sec\nbatch 30, time 0.081526 sec\nbatch 40, time 0.104298 sec\nbatch 50, time 0.125155 sec\nIncreased memory 245.776000 MB\n"
 }
]
```

## 小结

* 延后执行使得系统有更多空间来做性能优化。但我们推荐每个批量里至少有一个同步函数，例如对损失函数进行评估，来避免将过多任务同时丢进后端系统。

## 练习

* 为什么同步版本的训练中，我们看到了内存使用的大量下降？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1881)

![](../img/qr_lazy-evaluation.svg)
