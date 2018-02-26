# 多类逻辑回归 --- 使用Gluon

现在让我们使用gluon来更快速地实现一个多类逻辑回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../utils.py](../utils.py)这样这里不用复制一遍。

```{.python .input  n=39}
import sys
sys.path.append('..')
import utils
# utils??
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

```{.python .input  n=40}
train_data
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "<mxnet.gluon.data.dataloader.DataLoader at 0x7ff2e0800518>"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=41}
test_data
```

```{.json .output n=41}
[
 {
  "data": {
   "text/plain": "<mxnet.gluon.data.dataloader.DataLoader at 0x7ff2e0800128>"
  },
  "execution_count": 41,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size` x `?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input  n=42}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=43}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=44}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1})
```

## 训练

```{.python .input  n=45}
from mxnet import ndarray as nd
from mxnet import autograd

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
#         print(data.shape)
#         print(label.shape)
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

```{.json .output n=45}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 3.423383, Train acc 0.695157, Test acc 0.814258\nEpoch 1. Loss: 2.017116, Train acc 0.769570, Test acc 0.846387\nEpoch 2. Loss: 1.754002, Train acc 0.784918, Test acc 0.830957\nEpoch 3. Loss: 1.703902, Train acc 0.791029, Test acc 0.835059\nEpoch 4. Loss: 1.618749, Train acc 0.794986, Test acc 0.840137\n"
 }
]
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

- 再尝试调大下学习率看看？
- 为什么参数都差不多，但gluon版本比从0开始的版本精度更高？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/740)
