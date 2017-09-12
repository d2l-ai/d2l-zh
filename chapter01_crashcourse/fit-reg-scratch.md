# 拟合问题和从0开始的正则化

你有没有类似这样的体验？考试前突击背了模拟题的答案，模拟题随意秒杀。但考试时出的题即便和模拟题相关，只要不是原题依然容易考挂。换种情况来说，如果考试前通过自己的学习能力从模拟题的答案里总结出一个比较通用的解题套路，考试时碰到这些模拟题的变种更容易答对。

有人曾依据这种现象对学生群体简单粗暴地做了如下划分：

![](../img/student_categories.png)

这里简要总结上图中四类学生的特点：

* 学渣：训练量较小，学习能力较差，容易考挂
* 学痞：训练量较小，学习能力较强，通常考不过学霸但比学渣好
* 学痴：训练量较大，学习能力较差，通常考不过学霸但比学渣好
* 学霸：训练量较大，学习能力较强，容易考好


学生的考试成绩和看起来与自身的训练量以及自身的学习能力有关。但即使是在科技进步的今天，我们依然没有完全知悉人类大脑学习的所有奥秘。的确，依赖数据训练的机器学习和人脑学习不一定完全相同。但有趣的是，机器学习模型也可能由于自身不同的训练量和不同的学习能力而产生不同的测试效果。为了科学地阐明这个现象，我们需要从若干机器学习的重要概念开始讲解。

## 训练误差和泛化误差

在实践中，机器学习模型通常在训练数据集上训练并不断调整模型里的参数。之后，我们通常把训练得到的模型在一个区别于训练数据集的测试数据集上测试，并根据测试结果评价模型的好坏。机器学习模型在训练数据集上表现出的误差叫做**训练误差**，在任意一个测试数据样本上表现出的误差的期望值叫做**泛化误差**。（如果对严谨的数学定义感兴趣，可参考Mohri的[Foundations of Machine Learning](http://www.cs.nyu.edu/~mohri/mlbook/)）

训练误差和泛化误差的计算可以利用我们之前提到的损失函数，例如[从0开始的线性回归](linear-regression-scratch.md)里用到的平方误差和[从0开始的多类逻辑回归](softmax-regression-scratch.md)里用到的交叉熵损失函数。

之所以要了解训练误差和泛化误差，是因为统计学习理论基于这两个概念可以科学解释本节教程一开始提到的模型不同的测试效果。我们知道，理论的研究往往需要基于一些假设。而统计学习理论的一个假设是：

> 训练数据集和测试数据集里的每一个数据样本都是独立同分布。

基于以上独立同分布假设，给定任意一个机器学习模型及其参数，它的训练误差的期望值和泛化误差都是一样的。然而从之前的章节中我们了解到，在机器学习的过程中，模型的参数并不是事先给定的，而是通过训练数据学习得出的：模型的参数在训练中使训练误差不断降低。所以，如果模型参数是通过训练数据学习得出的，那么泛化误差必然无法低于训练误差的期望值。换句话说，由训练数据学到的模型参数通常使模型在训练数据上的表现不差于在测试数据上的表现。

因此，一个重要结论是：

> 训练误差的降低不一定意味着泛化误差的降低。机器学习既需要降低训练误差，又需要降低泛化误差。

## 欠拟合和过拟合

实践中，如果测试数据集是给定的，我们通常用机器学习模型在该测试数据集的误差来反映泛化误差。基于上述重要结论，以下两种拟合问题值得注意：

* **欠拟合**：机器学习模型无法得到较低训练误差。
* **过拟合**：机器学习模型的训练误差远小于其在测试数据集上的误差。

我们要尽可能同时避免欠拟合和过拟合的出现。虽然有很多因素可能导致这两种拟合问题，在这里我们重点讨论两个因素：模型的选择和训练数据集的大小。


### 模型的选择

在本节的开头，我们提到一个学生可以有特定的学习能力。类似地，一个机器学习模型也有特定的拟合能力。拿多项式函数举例来说，高阶多项式函数比低阶多项式函数更容易在相同的训练数据集上得到较低的训练误差。


### 训练数据集的大小

在本节的开头，我们同样提到一个学生可以有特定的训练量。类似地，一个机器学习模型的训练数据集的样本数也可大可小。统计学习理论中有个结论是：泛化误差不会随训练数据集里样本数量增加而增大。

为了理解这两个因素对拟合和过拟合的影响，下面让我们来动手学习。

## 案例分析——多项式拟合

【以下代码需改进且模块化，文字解释待加】


```python
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 1000
num_test = 1000

max_order = 5

true_w = [1.2, -3.4]
true_b = 3

x = nd.random_normal(shape=(num_train + num_test, 1))
X = nd.concat(x, nd.power(x, 2))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
y_train, y_test = y[:num_train], y[num_train:]

X_train_true_order, X_test_true_order = X[:num_train, :], X[num_train:, :]
x_train_order1, x_test_order1 = x[:num_train, :], x[num_train:, :]

X_max_order = x
for i in range(1, max_order):
    X_max_order = nd.concat(X_max_order, nd.power(x, i + 1))
X_train_max_order, X_test_max_order = X_max_order[:num_train, :], \
                                      X_max_order[num_train:, :]

for X_train, X_test, lr in zip([x_train_order1, X_train_true_order, X_train_max_order],
                           [x_test_order1, X_test_true_order, X_test_max_order],
                               [0.01, 0.05, 0.0001]):
    batch_size = 10
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

    net = gluon.nn.Sequential()
    dense = gluon.nn.Dense(1)
    net.add(dense)
    square_loss = gluon.loss.L2Loss()
    net.collect_params().initialize(force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

    epochs = 5
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

    loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
    print("Test loss: %f" % loss_test)

    print(true_w, true_b)
    print(dense.weight.data(), dense.bias.data())
```

    Epoch 0, train loss: 12.915853
    Epoch 1, train loss: 12.558763
    Epoch 2, train loss: 12.513711
    Epoch 3, train loss: 12.526407
    Epoch 4, train loss: 12.499932
    Test loss: 10.925433
    ([1.2, -3.4], 3)
    (
    [[ 1.25542045]]
    <NDArray 1x1 @cpu(0)>, 
    [-0.45684317]
    <NDArray 1 @cpu(0)>)
    Epoch 0, train loss: 1.193887
    Epoch 1, train loss: 0.001781
    Epoch 2, train loss: 0.000056
    Epoch 3, train loss: 0.000052
    Epoch 4, train loss: 0.000051
    Test loss: 0.000053
    ([1.2, -3.4], 3)
    (
    [[ 1.19974732 -3.40114522]]
    <NDArray 1x2 @cpu(0)>, 
    [ 2.99917698]
    <NDArray 1 @cpu(0)>)
    Epoch 0, train loss: 9.331360
    Epoch 1, train loss: 3.749909
    Epoch 2, train loss: 3.621049
    Epoch 3, train loss: 3.702738
    Epoch 4, train loss: 4.023941
    Test loss: 3.323413
    ([1.2, -3.4], 3)
    (
    [[ 0.00946653 -0.04145175  0.07508904 -0.38557202  0.01632602]]
    <NDArray 1x5 @cpu(0)>, 
    [ 0.03163745]
    <NDArray 1 @cpu(0)>)


## 结论



## 练习



**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
