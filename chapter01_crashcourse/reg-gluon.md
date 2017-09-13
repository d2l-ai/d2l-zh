# 正则化 --- 使用Gluon

本章介绍如何使用``Gluon``的正则化来应对[过拟合](/Users/astonz/WorkDocs/Programs/git_repo/gluon-tutorials-zh/chapter01_crashcourse/underfit-overfit.md)问题。

## 过拟合

我们使用高维线性回归为例来引入一个过拟合问题。

### 生成数据集

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

这里使用`data`模块来读取数据。


```python
batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
```

我们把损失函数定义为平方误差。


```python
square_loss = gluon.loss.L2Loss()
```

### 定义模型

我们将模型的定义放在一个函数里供多次调用。


```python
def getNet():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

我们定义一个训练和测试的函数，这样在跑不同的实验时不需要重复实现相同的步骤。

你也许发现了，`Trainer`有一个新参数`wd`。我们将在本节的正则化部分详细描述它。


```python
def train(net, data_iter_train, epochs, square_loss, lr, wd):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        total_loss = 0
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d, train loss: %f" % (epoch, total_loss / num_train))
    return net
```

以下是测试步骤。


```python
def test(net):
    loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
    print("Test loss: %f" % loss_test)
    dense = net[0]
    print('Learned weights (first 10): ', dense.weight.data()[0][:10], 
          'Learned bias: ', dense.bias.data())
```

以下函数将调用模型的定义、训练和测试。


```python
def learn(data_iter_train, square_loss, learning_rate, weight_decay):
    epochs = 10
    net = getNet()
    net_trained = train(net, data_iter_train, epochs, square_loss,
                        learning_rate, weight_decay)
    test(net_trained)
```

### 训练模型并观察过拟合

接下来我们训练并测试我们的高维线性回归模型。


```python
learning_rate = 0.005
weight_decay = 0
learn(data_iter_train, square_loss, learning_rate, weight_decay)
```

即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。

## 使用``Gluon``的正则化

我们通过优化算法的``wd``参数 (weight decay)实现对模型的正则化。这相当于$L_2$范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，如果把weight decay设为$\lambda$, 我们在训练时其实在最小化

$$\text{原损失函数} + \lambda \times \text{模型所有参数的平方和}。$$

直观上，$L_2$范数正则化试图惩罚较大绝对值的参数值。

下面我们重新初始化模型参数并在`Trainer`里设置一个`wd`参数。


```python
weight_decay = 1.0
learn(data_iter_train, square_loss, learning_rate, weight_decay)
```

我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。
但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 结论

* 使用``Gluon``的`weight decay`参数可以很容易地使用正则化来应对过拟合问题。

## 练习

* 如果把`weight decay`调高或调低，对模型的训练误差有何影响？
* 除了正则化、增大训练量、以及使用合适的模型，你觉得还有哪些办法可以应对过拟合现象？
* 如果你了解贝叶斯统计，你觉得$L_2$范数正则化对应贝叶斯统计里的哪个重要概念？


**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
