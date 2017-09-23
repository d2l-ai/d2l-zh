# 丢弃法 --- 使用Gluon

本章介绍如何使用``Gluon``在训练和测试深度学习模型中使用丢弃法。


## 定义模型并添加丢弃层

有了`Gluon`，我们模型的定义工作变得简单了许多。我们只需要在全连接层后添加`gluon.nn.Dropout`层并指定元素丢弃概率。一般情况下，我们推荐把
更靠近输入层的元素丢弃概率设的更小一点。这个试验中，我们把第一层全连接后的元素丢弃概率设为0.2，把第二层全连接后的元素丢弃概率设为0.5。

```{.python .input  n=5}
from mxnet.gluon import nn

net = nn.Sequential()
drop_prob1 = 0.2
drop_prob2 = 0.5

with net.name_scope():
    net.add(nn.Flatten())
    # 第一层全连接。
    net.add(nn.Dense(256, activation="relu"))
    # 在第一层全连接后添加丢弃层。
    net.add(nn.Dropout(drop_prob1))
    # 第二层全连接。
    net.add(nn.Dense(256, activation="relu"))
    # 在第二层全连接后添加丢弃层。
    net.add(nn.Dropout(drop_prob2))
    net.add(nn.Dense(10))
net.initialize()
```

## 读取数据并训练

这跟之前没什么不同。

```{.python .input  n=6}
import sys
sys.path.append('..')
import utils
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 
                        'sgd', {'learning_rate': 0.5})

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
        epoch, train_loss/len(train_data), 
        train_acc/len(train_data), test_acc))
```

## 结论

通过`Gluon`我们可以更方便地构造多层神经网络并使用丢弃法。

## 练习

- 尝试不同元素丢弃概率参数组合，看看结果有什么不同。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1279)
