# VGG：使用重复元素的非常深的网络

我们从Alexnet看到网络的层数的激增。这个意味着即使是用Gluon手动写代码一层一层的堆每一层也很麻烦，更不用说从0开始了。幸运的是编程语言提供了很好的方法来解决这个问题：函数和循环。如果网络结构里面有大量重复结构，那么我们可以很紧凑来构造这些网络。第一个使用这种结构的深度网络是VGG。

## VGG架构

VGG的一个关键是使用很多有着相对小的kernel（$3\times 3$）的卷积层然后接上一个池化层，之后再将这个模块重复多次。下面我们先定义一个这样的块：

```{.python .input}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                          padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
```

我们实例化一个这样的块，里面有两个卷积层，每个卷积层输出通道是128：

```{.python .input}
from mxnet import nd

blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2,3,16,16))
y = blk(x)
y.shape
```

可以看到经过一个这样的块后，长宽会减半，通道也会改变。

然后我们定义如何将这些块堆起来：

```{.python .input}
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
```

这里我们定义一个最简单的一个VGG结构，它有8个卷积层，和跟Alexnet一样的3个全连接层。这个网络又称VGG 11.

```{.python .input}
num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))
```

## 模型训练

这里跟Alexnet的训练代码一样：

```{.python .input}
import sys
sys.path.append('..')
import utils
from mxnet import image

def transform(data, label):
    # resize from 28 x 28 to 96 x 96
    data = image.imresize(data, 96, 96)
    return utils.transform_mnist(data, label)

batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size, transform)

from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import init
ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.05})

for epoch in range(1):
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

## 总结

通过使用重复的元素，我们可以通过循环和函数来定义模型。使用不同的配置(`architecture`)可以得到一系列不同的模型。


## 练习

- 尝试多跑几轮，看看跟LeNet/Alexnet比怎么样？
- 尝试下构造VGG其他常用模型，例如VGG16， VGG19. （提示：可以参考[VGG论文](https://arxiv.org/abs/1409.1556)里的表1。）
- 把图片从默认的$224\times 224$降到$96\times 96$有什么影响？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1277)
