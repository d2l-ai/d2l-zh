# 使用Gluon的多类Logistic回归

现在让我们使用gluon来更快速的实现一个多类Logistic回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../mnist.py](../mnist.py)这样这里不用复制一遍。

```{.python .input  n=1}
import sys
sys.path.append('..')
from mnist import load_data

batch_size = 256
train_data, test_data = load_data(batch_size)
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size x ?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input  n=2}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=3}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=4}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
```

## 训练

```{.python .input  n=5}
from mxnet import ndarray as nd
from mxnet import autograd
from utils import accuracy, evaluate_accuracy

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
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

再尝试调大下学习率看看？
