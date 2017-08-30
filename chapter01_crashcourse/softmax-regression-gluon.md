# 使用Gluon的多类Logistic回归

现在让我们使用gluon来更快速的实现一个多类Logistic回归。

## 获取和读取数据

```{.python .input}
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size x ?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=7}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=8}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
```

## 计算精度

手写一下也挺方便

```{.python .input}
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net):
    acc = 0.    
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
```

## 训练

```{.python .input  n=18}
epochs = 5

for e in range(epochs):
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
    print("Epoch %d. Loss: %f, Train_acc %f, Test_acc %f" % (
            e, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

再尝试调大下学习率看看？
