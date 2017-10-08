# 多类逻辑回归 --- 使用Gluon

现在让我们使用gluon来更快速地实现一个多类逻辑回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../utils.py](../utils.py)这样这里不用复制一遍。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size` x `?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

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
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
```

## 训练

```{.python .input  n=10}
from mxnet import ndarray as nd
from mxnet import autograd
import time

for epoch in range(5):
    start_time = time.time()
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
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f, Cost time %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc, time.time() - start_time))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.725811, Train acc 0.828291, Test acc 0.843359, Cost time 10.299233\nEpoch 1. Loss: 0.748478, Train acc 0.825792, Test acc 0.821094, Cost time 10.215502\nEpoch 2. Loss: 0.712756, Train acc 0.828441, Test acc 0.840430, Cost time 10.271214\nEpoch 3. Loss: 0.672995, Train acc 0.832945, Test acc 0.849414, Cost time 10.657582\nEpoch 4. Loss: 0.733748, Train acc 0.826867, Test acc 0.849219, Cost time 10.455018\n"
 }
]
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

- 再尝试调大下学习率看看？
- 为什么参数都差不多，但gluon版本比从0开始的版本精度更高？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/740)
