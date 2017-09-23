# 卷积神经网络 --- 使用Gluon

现在我们使用Gluon来实现[上一章的卷积神经网络](cnn-scratch.md)。

## 定义模型

下面是LeNet在Gluon里的实现，注意到我们不再需要实现去计算每层的输入大小，尤其是接在卷积后面的那个全连接层。

```{.python .input}
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Flatten())
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(10))
```

然后我们尝试将模型权重初始化在GPU上

```{.python .input}
import sys
sys.path.append('..')
import utils

ctx = utils.try_gpu()
net.initialize(ctx=ctx)

print('initialize weight on', ctx)
```

## 获取数据然后训练

跟之前没什么两样。

```{.python .input}
from mxnet import autograd 
from mxnet import gluon
from mxnet import nd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

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
        epoch, train_loss/len(train_data), 
        train_acc/len(train_data), test_acc))
```

## 结论

使用Gluon来实现卷积网络轻松加随意。

## 练习

再试试改改卷积层设定，是不是会比上一章容易很多？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/737)
