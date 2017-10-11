# 多层感知机 --- 从0开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter01_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一到多个隐含层。

![](../img/multilayer-perceptron.png)

这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=2}
from mxnet import ndarray as nd

num_inputs = 28*28
num_outputs = 10

num_hidden = 256
weight_scale = .01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
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
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=6}
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=8}
from mxnet import autograd as autograd

learning_rate = .5

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), 
        train_acc/len(train_data), test_acc))
```

## 总结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 我们使用了 `weight_scale` 来控制权重的初始化值大小，增大或者变小这个值会怎么样？
- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/739)
