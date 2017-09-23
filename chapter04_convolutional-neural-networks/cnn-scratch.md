# 卷积神经网络 --- 从0开始

之前的教程里，在输入神经网络前我们将输入图片直接转成了向量。这样做有两个不好的地方：

- 在图片里相近的像素在向量表示里可能很远，从而模型很难捕获他们的空间关系。
- 对于大图片输入，模型可能会很大。例如输入是$256\times 256\times3$的照片（仍然远比手机拍的小），输入层是1000，那么这一层的模型大小是将近1GB.

这一节我们介绍卷积神经网络，其有效了解决了上述两个问题。

## 卷积神经网络

卷积神经网络是指主要由卷积层构成的神经网络。

### 卷积层

卷积层跟前面的全连接层类似，但输入和权重不是做简单的矩阵乘法，而是使用每次作用在一个窗口上的卷积。下图演示了输入是一个$4\times 4$矩阵，使用一个$3\times 3$的权重，计算得到$2\times 2$结果的过程。每次我们采样一个跟权重一样大小的窗口，让它跟权重做按元素的乘法然后相加。通常我们也是用卷积的术语把这个权重叫kernel或者filter。

![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif)

（图片版权属于vdumoulin@github）

我们使用`nd.Convlution`来演示这个。

```{.python .input  n=47}
from mxnet import nd

# 输入输出数据格式是 batch x channel x height x width，这里batch和channel都是1
# 权重格式是 input_filter x output_filter x height x width，这里input_filter和output_filter都是1。
w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1])

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

我们可以控制如何移动窗口，和在边缘的时候如何填充窗口。下图演示了`stride=1`和`pad=1`。

![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides.gif)

```{.python .input  n=48}
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1],
                     stride=(2,2), pad=(1,1))

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

当输入数据有多个通道的时候，每个通道会有对应的权重，然后会对每个通道做卷积之后在通道之间求和

$$conv(data, w, b) = \sum_i conv(data[:,i,:,:], w[0,i,:,:], b)$$

```{.python .input  n=49}
w = nd.arange(8).reshape((1,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

当输入需要多通道时，每个输出通道有对应权重，然后每个通道上做卷积。

$$conv(data, w, b)[:,i,:,:] = conv(data, w[i,:,:,:], b[i])$$

```{.python .input  n=50}
w = nd.arange(16).reshape((2,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))
b = nd.array([1,2])

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

### 池化层（pooling）

因为卷积层每次作用在一个窗口，它对位置很敏感。池化层能够很好的缓解这个问题。它跟卷积类似每次看一个小窗口，然后选出窗口里面最大的元素，或者平均元素作为输出。

```{.python .input  n=53}
data = nd.arange(18).reshape((1,2,3,3))

max_pool = nd.Pooling(data=data, pool_type="max", kernel=(2,2))
avg_pool = nd.Pooling(data=data, pool_type="avg", kernel=(2,2))

print('data:', data, '\n\nmax pooling:', max_pool, '\n\navg pooling:', avg_pool)
```

下面我们可以开始使用这些层构建模型了。


## 获取数据

我们继续使用FashionMNIST（希望你还没有彻底厌烦这个数据）

```{.python .input  n=22}
import sys
sys.path.append('..')
from utils import load_data_fashion_mnist

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)
```

## 定义模型

因为卷积网络计算比全连接要复杂，这里我们默认使用GPU来计算。如果GPU不能用，默认使用CPU。

```{.python .input  n=65}
import mxnet as mx

try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx
```

我们使用MNIST常用的LeNet，它有两个卷积层，之后是两个全连接层。注意到我们将权重全部创建在`ctx`上：

```{.python .input  n=66}
weight_scale = .01

# output channels = 20, kernel = (5,5)
W1 = nd.random_normal(shape=(20,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

# output channels = 50, kernel = (3,3)
W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(W2.shape[0], ctx=ctx)

# output dim = 128
W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

# output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
    param.attach_grad()
```

卷积模块通常是“卷积层-激活层-池化层”。然后转成2D矩阵输出给后面的全连接层。

```{.python .input  n=74}
def net(X, verbose=False):
    X = X.as_in_context(W1.context)
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear
```

测试一下，输出中间结果形状（当然可以直接打印结果)和最终结果。

```{.python .input  n=76}
for data, _ in train_data:
    net(data, verbose=True)
    break
```

## 训练

跟前面没有什么不同的

```{.python .input  n=60}
from mxnet import autograd as autograd
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 结论

可以看到卷积神经网络比前面的多层感知的分类精度更好。事实上，如果你看懂了这一章，那你基本知道了计算视觉里最重要的几个想法。LeNet早在90年代就提出来了。不管你相信不详细，如果你5年前懂了这个而且开了家公司，那么你很可能现在已经把公司作价几千万卖个某大公司了。幸运的是，或者不幸的是，现在的算法已经更加高级些了，接下来我们会看到一些更加新的想法。

## 练习

- 试试改改卷积层设定，例如filter数量，kernel大小
- 试试把池化层从`max`改到`avg`
- 如果你有GPU，那么尝试用CPU来跑一下看看
- 你可能注意到比前面的多层感知机慢了很多，那么尝试计算下这两个模型分别需要多少浮点计算。例如$n\times m$和$m \times k$的矩阵乘法需要浮点运算 $2nmk$。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/736)
