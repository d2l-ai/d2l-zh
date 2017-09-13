# 正则化 --- 从0开始

本章从0开始介绍如何的正则化来应对[过拟合](underfit-overfit.md)问题。

## 高维线性回归

我们使用高维线性回归为例来引入一个过拟合问题。


具体来说我们使用如下的线性函数来生成每一个数据样本

$$y = 0.05 + \sum_{i = 1}^p 0.01x_i +  \text{noise}$$

这里噪音服从均值0和标准差为0.01的正态分布。

需要注意的是，我们用以上相同的数据生成函数来生成训练数据集和测试数据集。为了观察过拟合，我们特意把训练数据样本数设低，例如$n=20$，同时把维度升高，例如$p=200$。这个高维线性回归也属于$p << n$问题。


```python
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 20
num_test = 1000
num_inputs = 200
```

## 生成数据集


这里定义模型真实参数。


```python
true_w = nd.ones(shape=(num_inputs, 1)) * 0.01
true_b = 0.05
```

我们接着生成训练和测试数据集。


```python
X = nd.random_normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w)
y += .01 * nd.random_normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]
```

当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回`batch_size`个随机的样本和对应的目标。我们通过python的`yield`来构造一个迭代器。


```python
import random
batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X, j), nd.take(y, j)
```

## 初始化模型参数

下面我们随机初始化模型参数。之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。


```python
def getParams():
    w = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params
```

## $L_2$范数正则化


线性模型就是将输入和模型做乘法再加上偏移。

这里我们引入$L_2$范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，我们在训练时其实在最小化

$$\text{原损失函数} + \lambda \times \text{模型所有参数的平方和}。$$

直观上，$L_2$范数正则化试图惩罚较大绝对值的参数值。在训练模型时，如果$\lambda = 0$，则未使用正则化。需要注意的是，在测试模型时，$\lambda$必须为0。


```python
def net(X, lambd, params):
    w = params[0]
    b = params[1]
    return nd.dot(X, w) + b + lambd * nd.dot(w.T, w)
```

## 损失函数和优化算法

我们使用平方误差和随机梯度下降。


```python
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## 定义训练和测试

我们定义一个训练和测试的函数，这样在跑不同的实验时不需要重复实现相同的步骤。


```python
def train(params, epochs, learning_rate, lambd):
    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, lambd, params)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)

        train_loss = nd.sum(square_loss(net(X_train, 0, params), y_train)).asscalar() / num_train
        print("Epoch %d, train loss: %f" % (e, train_loss))
    return params
```

以下是测试步骤。


```python
def test(params):
    loss_test = nd.sum(square_loss(net(X_test, 0, params), y_test)).asscalar() / num_test
    print("Test loss: %f" % loss_test)
    print('Learned weights (first 10): ', params[0][:10], 'Learned bias: ', params[1])
```

以下函数将调用模型的定义、训练和测试。


```python
def learn(learning_rate, lambd):
    epochs = 10
    params = getParams()
    params_trained = train(params, epochs, learning_rate, lambd)
    test(params_trained)
```

## 观察过拟合

接下来我们训练并测试我们的高维线性回归模型。注意这时我们并未使用正则化。


```python
learning_rate = 0.0025
lambd = 0
learn(learning_rate, lambd)
```

即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。


## 使用正则化

下面我们重新初始化模型参数并设置一个正则化参数。


```python
lambd = 0.8
learn(learning_rate, lambd)
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 结论

* 我们可以使用正则化来应对过拟合问题。

## 练习

* 除了正则化、增大训练量、以及使用合适的模型，你觉得还有哪些办法可以应对过拟合现象？
* 如果你了解贝叶斯统计，你觉得$L_2$范数正则化对应贝叶斯统计里的哪个重要概念？


**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
