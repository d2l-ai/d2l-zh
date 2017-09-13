# 正则化 --- 使用Gluon

本章介绍如何使用``Gluon``的正则化来应对[过拟合](underfit-overfit.md)问题。

## 高维线性回归

我们使用与[上一节](reg-scratch.md)相同的高维线性回归为例来引入一个过拟合问题。

## 生成数据集

我们定义训练和测试数据集的样本数量以及维度。


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

## 定义模型

我们将模型的定义放在一个函数里供多次调用。


```python
def getNet():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

我们定义一个训练和测试的函数，这样在跑不同的实验时不需要重复实现相同的步骤。

你也许发现了，`Trainer`有一个新参数`wd`。我们通过优化算法的``wd``参数 (weight decay)实现对模型的正则化。这相当于$L_2$范数正则化。


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

    Epoch 0, train loss: 0.111031
    Epoch 1, train loss: 0.019290
    Epoch 2, train loss: 0.002763
    Epoch 3, train loss: 0.000244
    Epoch 4, train loss: 0.000014
    Epoch 5, train loss: 0.000002
    Epoch 6, train loss: 0.000000
    Epoch 7, train loss: 0.000000
    Epoch 8, train loss: 0.000000
    Epoch 9, train loss: 0.000000
    Test loss: 0.169621
    Learned weights (first 10):  
    [ 0.02233147  0.02927095  0.03952093 -0.04752691 -0.00439923 -0.04209704
      0.06403525 -0.04425619 -0.04155682 -0.02823593]
    <NDArray 10 @cpu(0)> Learned bias:  
    [-0.01081783]
    <NDArray 1 @cpu(0)>


即便训练误差可以达到0.000000，但是测试数据集上的误差很高。这是典型的过拟合现象。

观察学习的参数。事实上，大部分学到的参数的绝对值比真实参数的绝对值要大一些。

## 使用``Gluon``的正则化

下面我们重新初始化模型参数并在`Trainer`里设置一个`wd`参数。


```python
weight_decay = 1.0
learn(data_iter_train, square_loss, learning_rate, weight_decay)
```

    Epoch 0, train loss: 0.225623
    Epoch 1, train loss: 0.006438
    Epoch 2, train loss: 0.000377
    Epoch 3, train loss: 0.000113
    Epoch 4, train loss: 0.000048
    Epoch 5, train loss: 0.000052
    Epoch 6, train loss: 0.000036
    Epoch 7, train loss: 0.000072
    Epoch 8, train loss: 0.000053
    Epoch 9, train loss: 0.000037
    Test loss: 0.027216
    Learned weights (first 10):  
    [-0.0279177   0.00097269 -0.00401345  0.00359961 -0.01495902 -0.02302548
      0.00151448  0.00749522 -0.00127769  0.02289863]
    <NDArray 10 @cpu(0)> Learned bias:  
    [-0.0043129]
    <NDArray 1 @cpu(0)>


我们发现训练误差虽然有所提高，但测试数据集上的误差有所下降。过拟合现象得到缓解。
但打印出的学到的参数依然不是很理想，这主要是因为我们训练数据的样本相对维度来说太少。

## 结论

* 使用``Gluon``的`weight decay`参数可以很容易地使用正则化来应对过拟合问题。

## 练习

* 如何从字面正确理解`weight decay`的含义？它为何相当于$L_2$范式正则化？


**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
