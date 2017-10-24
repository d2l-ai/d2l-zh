# 卷积神经网络 --- 使用Gluon

现在我们使用Gluon来实现[上一章的卷积神经网络](cnn-scratch.md)。

## 定义模型

下面是LeNet在Gluon里的实现，注意到我们不再需要实现去计算每层的输入大小，尤其是接在卷积后面的那个全连接层。

```{.python .input}
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
```

## 获取数据和训练

剩下的跟上一章没什么不同，我们重用`utils.py`里定义的函数。

```{.python .input}
from mxnet import gluon
import sys
sys.path.append('..')
import utils

# 初始化
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)

# 获取数据
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 训练
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=5)
```

## 结论

使用Gluon来实现卷积网络轻松加随意。

## 练习

再试试改改卷积层设定，是不是会比上一章容易很多？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/737)
