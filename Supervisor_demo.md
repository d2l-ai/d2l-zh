# 微信监管MXNet训练
>本系列受szha回复影响，花了半天时间将论坛帖子改为callback来实现这一功能。测试结果还行，但毕竟新手，可能bug不少。contribute希望MXNe
t团队能够给予支持！：）

## 基本改动

在工具类utils.py中做了如下改动：

1.导入itchat和threading

```{.python .input  n=1}
#pip install itchat
import itchat
import threading
```

2.添加基础变量

```{.python .input  n=2}
lock = threading.Lock()
running = False

batch_size = 256
learning_rate = 0.5
training_iters = 2
```

3.定义`lock_start()`、`chat_inf()`、`lock_end()`和`chat_supervisor()`

## 具体实现

以[卷积神经网络 — 使用Gluon](https://zh.gluon.ai/cnn-gluon.html)一章为例，我们将实现cnn与itchat结合。

首先，将原本网络和训练部分封装到`nn_train`内,其中加入`utils.lock_start()`、`utils.chat_inf()`和`utils.l
ock_end()`作为控制和输出：

```{.python .input  n=3}
def nn_train(wechat_name,params):
    learning_rate, training_iters, batch_size = params
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Flatten())
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(10))
    ctx = utils.try_gpu()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':learning_rate})
    epoch = 1
    while utils.lock_start() and epoch < training_iters:
        train_loss = 0
        train_acc = 0
        for data,label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output,label)
            loss.backward()
            trainer.step(batch_size)
            train_acc += utils.accuracy(output,label)
            train_loss += nd.mean(loss).asscalar()
        test_acc = utils.evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f\n" % (epoch, train_loss/len(train_data),train_acc/len(train_data), test_acc))
        utils.chat_inf(wechat_name,epoch,train_loss,train_acc,train_data,test_acc)
        epoch += 1
    utils.lock_end(wechat_name)
```

2.然后在main函数里调用 `utils.chat_supervisor()` 来实现交互：

```{.python .input  n=4}
if __name__ == '__main__':
    utils.chat_supervisor(nn_train)
    itchat.auto_login(hotReload=True)
    itchat.run()
```

3.完整代码如下：

```{.python .input  n=5}
from mxnet.gluon import nn
import utils
import itchat
from mxnet import autograd
from mxnet import gluon
from mxnet import nd

def nn_train(wechat_name,params):
    learning_rate, training_iters, batch_size = params
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Flatten())
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(10))
    ctx = utils.try_gpu()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':learning_rate})
    epoch = 1
    while utils.lock_start() and epoch < training_iters:
        train_loss = 0
        train_acc = 0
        for data,label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output,label)
            loss.backward()
            trainer.step(batch_size)
            train_acc += utils.accuracy(output,label)
            train_loss += nd.mean(loss).asscalar()
        test_acc = utils.evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f\n" % (epoch, train_loss/len(train_data),train_acc/len(train_data), test_acc))
        utils.chat_inf(wechat_name,epoch,train_loss,train_acc,train_data,test_acc)
        epoch += 1
    utils.lock_end(wechat_name)
            
if __name__ == '__main__':
    utils.chat_supervisor(nn_train)
    itchat.auto_login(hotReload=True)
    itchat.run()
```

## 实现效果
扫码登录微信A，通过微信B向微信A实现交互：

1.通过输入**param\_name value**的方式（learning_rate, training_iters,
batch_size参数名分别简写为lr，ti和bs）进行参数修改；

2.输入**开始**，开始进入训练；

3.输入**停止**，暂定训练。具体如下所示：

<center>

![](https://i.imgur.com/y5ZukZb.png)

</center>
