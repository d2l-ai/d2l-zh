# 多层感知机——从零开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input}
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss
```

```{.python .input  n=1}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一个到多个隐含层。


这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=2}
num_inputs = 784
num_outputs = 10

num_hiddens = 256

W1 = nd.random_normal(shape=(num_inputs, num_hiddens), scale=0.01)
b1 = nd.zeros(num_hiddens)
W2 = nd.random_normal(shape=(num_hiddens, num_outputs), scale=0.01)
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 激活函数

如果我们就用线性操作符来构造多层神经网络，那么整个模型仍然只是一个线性函数。这是因为

$$\hat{y} = X \cdot W_1 \cdot W_2 = X \cdot W_3 $$

这里$W_3 = W_1 \cdot W_2$。为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。这里我们使用ReLU

$$\textrm{rel}u(x)=\max(x, 0)$$

```{.python .input  n=3}
def relu(X):
    return nd.maximum(X, 0)
```

## 定义模型

我们的模型就是将层（全连接）和激活函数（Relu）串起来：

```{.python .input  n=4}
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=8}
num_epochs = 5
lr = 0.5

gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params, lr)
```

## 小结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/739)

![](../img/qr_mlp-scratch.svg)
