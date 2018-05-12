# 卷积层

卷积层（Convolutional layer）是卷积神经网络里的基石。本节我们将介绍最简单形式的二维卷积层的是怎么工作的。虽然最早卷积层是使用卷积（Convolution）运算符，但目前的主流实现使用的是更加直观的相关（Correlation）运算符。

## 二维相关运算符

一个二维相关运算符将一个二维矩阵核（kernel）作用在一个二维输入数据上计算出一个二维矩阵输出。

下面例子里我们构造了一个`(3, 3)`形状的输入`X`和`(2, 2)`形状的核`K`来计算输出`Y`。

![二维相关运算符。](../img/correlation.svg)

可以看到`Y`的形状是`(2, 2)`，而且第一个元素是由`X`的左上的`(2, 2)`子矩阵与核做按元素乘法然后相加得来，即`Y[0, 0] = (X[0:2, 0:2] * K).sum()`，这里我们使用假设数据类型是NDArray。然后我们将`X`上子矩阵向左滑动一个元素来计算`Y`的第一行第二个元素。以此类推计算下面所有结果。

我们将这一过程实现在下面的`corr2d`函数里。

```{.python .input  n=21}
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

from mxnet import nd, autograd
from mxnet.gluon import nn

def corr2d(X, K):
    n, m = K.shape
    Y = nd.zeros((X.shape[0]-n+1, X.shape[1]-m+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+n, j:j+m]*K).sum()
    return Y

X = nd.array([[0,1,2], [3,4,5], [6,7,8]])
K = nd.array([[0,1], [2,3]])
corr2d(X, K)
```

## 图片物体边缘检测

在这里让我们看一个如何构造一个核来进行图片里的物体边缘检测。假如我们定义一个$6\times 8$的图片，它中间4列为黑，其余为白。

```{.python .input  n=66}
X = nd.ones((6, 8))
X[:, 2:6] = 0
plt.imshow(X.asnumpy(), cmap='gray')
plt.show()
```

然后我们构造一个形状为`(1, 2)`的核，使得如果是作用在相同色块上时输出为0，有颜色变化时为1。因为这里颜色只在水平方向上变化，所以我们设计是第一个元素为1，第二个元素为-1。

```{.python .input  n=67}
K = nd.array([[1, -1]])
K
```

对输入图片作用我们设计的核后可以发现，从白到黑的边缘我们检测成了1，从黑到白则是-1，其余全是0。

```{.python .input  n=69}
Y = corr2d(X, K)
Y
```

## 二维卷积层

二维卷积层则将核作为权重，外加一个标量偏差。在前向计算时，它的输出就是输入数据和核的二维相关运算加上偏差。在训练的时候，我们同全连接一样将权重进行随机初始化，然后不断迭代优化权重和偏差来拟合数据。

下面的我们基于`corr2d`函数来实现一个自定义的二维卷基层。在初始化函数里我们声明`weight`和`bias`这两个模型参数，前向计算函数则是直接调用`corr2d`再加上偏差。

```{.python .input  n=70}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

你也许会好奇为什么不使用二维卷积运算符呢？其实它的计算于与二维相关运算符类似，唯一的区别是我们将反向访问`K`，即`Y[0, 0] = (X[0:2, 0:2] * K[::-1, ::-1]).sum()`。因为在卷基层里我们会通过数据来学习`K`，所以不管是正向还是反向访问，我们最后得到的结果是一样的。

## 学习核参数

最后我们来看一个例子，我们使用图片边缘检测例子里的`X`和`Y`来学习`K`. 虽然我们之前定义了Conv2D，但由于`corr2d`使用了对单个元素赋值（`[i, j]=`）的操作会导致系统无法对其自动求导，所以我们使用`nn`模块里的Conv2D层来实现这个例子。它的使用跟我们定义的非常相似。

每一个迭代里，我们使用平方误差来比较`Y`和卷积层的输入，然后计算梯度来更新权重（为了简单起见这里忽略了偏差）。

```{.python .input  n=83}
# 构造一个输出通道是 1 的二维卷基层，我们会在后面小节里解释什么是通道。
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 二维卷基层使用 4 维输入输出，格式为（批量大小，通道数，高，宽），这里批量和通道均为 1.
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        pY = conv2d(X)
        loss = (pY - Y) ** 2
        print('batch %d, loss %.3f' % (i, loss.sum().asscalar()))
    loss.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
```

可以看到10次迭代后误差已经降到了一个比较小的值，现在来看一下学习到的权重。

```{.python .input}
conv2d.weight.data()
```

如果忽略掉这里权重是一个4维数组（会在之后小节解释），我们学到的权重与我们之前定义的`K`已经非常接近。

## 小结

- 二维卷基层的核心计算是二维相关运算。在最简单的形式下，它将一个二维核矩阵作用在二维输入上。
- 我们可以设计核矩阵来检测图片中的边缘，同时也可以通过数据来学习这个核矩阵。

## 练习

- 构造一个`X`它有水平方向的边缘，如何设计`K`来检测它？如果是对角方向的边缘呢？
- 试着对我们构造的`Conv2D`进行自动求导，会有什么样的错误信息？
- 在Conv2D的`forward`函数里，将`corr2d`替换成`nd.Convolution`使得其可以求导。
- 试着将conv2d的核构造成`(2, 2)`，会学出什么样的结果？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6314)

![](../img/qr_conv-layer.svg)
