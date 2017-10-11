# 多层感知机 --- 使用Gluon

我们只需要稍微改动[多类Logistic回归](../chapter01_crashcourse/softmax-regression-gluon.md)来实现多层感知机。

## 定义模型

唯一的区别在这里，我们加了一行进来。

```{.python .input  n=5}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## 读取数据并训练

```{.python .input  n=6}
import sys
sys.path.append('..')
from mxnet import ndarray as nd
from mxnet import autograd
import utils


batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 结论

通过Gluon我们可以更方便地构造多层神经网络。

## 练习

- 尝试多加入几个隐含层，对比从0开始的实现。
- 尝试使用一个另外的激活函数，可以使用`help(nd.Activation)`或者[线上文档](https://mxnet.apache.org/api/python/ndarray.html#mxnet.ndarray.Activation)查看提供的选项。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/738)
