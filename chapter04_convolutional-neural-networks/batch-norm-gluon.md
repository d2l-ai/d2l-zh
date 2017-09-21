# 批量归一化 --- 使用Gluon

本章介绍如何使用``Gluon``在训练和测试深度学习模型中使用批量归一化。


## 定义模型并添加批量归一化层

有了`Gluon`，我们模型的定义工作变得简单了许多。我们只需要添加`gluon.nn.BatchNorm`层并指定对二维卷积的通道(`axis=1`)进行批量
归一化。

```{.python .input  n=1}
import mxnet as mx
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    # 第一层卷积
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5))
    net.add(gluon.nn.BatchNorm(axis=1))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # 第二层卷积
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=3))
    net.add(gluon.nn.BatchNorm(axis=1))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    # 第一层全连接
    net.add(gluon.nn.Dense(128, activation="relu"))
    # 第二层全连接
    net.add(gluon.nn.Dense(10))
```

我们推荐使用GPU运行并教程代码。

```{.python .input  n=2}
import sys
sys.path.append('..')
import utils

ctx = utils.try_gpu()
net.initialize(mx.init.Normal(sigma=0.01), ctx=ctx)

print('initialize weight on', ctx)
```

这里训练并测试模型。

```{.python .input  n=3}
from mxnet import autograd 
from mxnet import gluon
from mxnet import nd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 总结

使用``Gluon``我们可以很轻松地添加批量归一化层。

## 练习

如果在全连接层添加批量归一化结果会怎么样？
