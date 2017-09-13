# 正则化 --- 使用Gluon

本章介绍如果使用``Gluon``的正则化来应对[过拟合](/Users/astonz/WorkDocs/Programs/git_repo/gluon-tutorials-zh/chapter01_crashcourse/underfit-overfit.md)问题。

## 过拟合

我们使用五阶多项式拟合为例来引入一个过拟合问题。

### 生成数据集

具体来说我们使用如下的五阶多项式来生成每一个数据样本

$$y = 0.001x - 0.002x^2 + 0.003x^3 -0.004x^4 + 0.005x^5 + 0.005 +  \text{noise}$$

这里噪音服从均值0和标准差为0.01的正态分布。

需要注意的是，我们用以上相同的数据生成函数来生成训练数据集和测试数据集。为了观察过拟合，我们特意把训练数据样本数设低，例如6。


```python
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 6
num_test = 1000
```

这里定义模型真实参数。


```python
true_order = 5
true_w = [0.001, -0.002, 0.003, -0.004, 0.005]
true_b = 0.005
```

我们需要用五阶多项式生成数据样本。


```python
x = nd.random_normal(shape=(num_train + num_test, 1))
X = x
y = true_w[0] * X[:, 0]
for i in range(1, true_order):
    X = nd.concat(X, nd.power(x, i + 1))
    y += true_w[i] * X[:, i]
y += true_b + .01 * nd.random_normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]
```

这里使用`data`模块来读取数据。


```python
batch_size = 10
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
```

### 定义模型

我们照旧定义模型并初始化模型参数。损失函数依然是平方误差。


```python
net = gluon.nn.Sequential()
dense = gluon.nn.Dense(1)
net.add(dense)
square_loss = gluon.loss.L2Loss()
net.initialize()
```

### 训练模型并观察过拟合

接下来我们训练并测试我们的五阶多项式模型。


```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.004})
epochs = 10
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter_train:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, train loss: %f" % (e, total_loss / num_train))
```

即便训练误差较低，但是测试数据集上的误差很高。这是典型的过拟合现象。


```python
loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
print("Test loss: %f" % loss_test)
```

观察学习的参数。事实上，学到的参数的绝对值比真实参数的绝对值要大一些。


```python
print("True params: ", true_w, true_b)
print("Learned params: ", dense.weight.data()[0], dense.bias.data()[0])
```

## 使用``Gluon``的正则化

我们通过优化算法的``wd``参数 (weight decay)实现对模型的正则化。这相当于$L_2$范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，如果weight decay设为$\lambda$, 我们在训练时其实在最小化

$$\text{原损失函数} + \lambda \times \text{模型所有参数的平方和}。$$

直观上，$L_2$范数正则化试图惩罚较大绝对值的参数值。

下面我们重新初始化模型参数并在`Trainer`里设置一个较大的`wd`参数。


```python
net.collect_params().initialize(force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.004,
                                                     'wd': 150.0})
epochs = 10
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter_train:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, train loss: %f" % (e, total_loss / num_train))
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。


```python
loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
print("Test loss: %f" % loss_test)
```

但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本太少。


```python
print("True params: ", true_w, true_b)
print("Learned params: ", dense.weight.data()[0], dense.bias.data()[0])
```

## 结论

* 使用``Gluon``的`weight decay`参数可以实现正则化而应对过拟合。

## 练习

* 除了正则化、增大训练量、以及使用合适的模型，你觉得还有哪些办法可以应对过拟合现象？
* 如果你了解贝叶斯统计，你觉得$L_2$范数正则化对应贝叶斯统计里的哪个重要概念？


**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
