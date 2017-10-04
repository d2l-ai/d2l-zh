# 实战Kaggle比赛——使用Gluon对原始图像文件分类（CIFAR-10）


我们在[监督学习中的一章](../chapter02_supervised-learning/kaggle-gluon-
kfold.md)里，以[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-t
echniques)为例，介绍了如何使用``Gluon``来实战[Kaggle比赛](https://www.kaggle.com)。社区中的很多小伙伴实践了房
价预测问题并[分享了自己的成绩和方法](https://discuss.gluon.ai/t/topic/1039)。有小伙伴[反馈希望增加Kaggle比赛章节
](https://discuss.gluon.ai/t/topic/1341)。还有小伙伴[反馈希望提供对原始图像分类的例子](https://discuss
.gluon.ai/t/topic/499)，例如输入格式是jpg或png而非教程中封装好的图像数据，这样甚至可能[更好地管理手机图片](https://dis
cuss.gluon.ai/t/topic/1372)。


有鉴于大家的反馈，我们在本章中选择了Kaggle中著名的[CIFAR-10原始图像分类问题](https://www.kaggle.com/c/cifar-10
)。我们以该问题为例，为大家提供使用`Gluon`对原始图像文件进行分类的示例代码。

本章的示例代码主要分五部分：

* 整理原始数据集；
* 使用`Gluon`读取整理后的数据集；
* 设计模型；
* 使用部分训练数据训练模型并以其它训练数据为验证集调参；
* 使用全部训练数据训练模型并对测试数据分类。


需要注意的是，本章仅提供一些基本实战流程供大家参考。对于数据的预处理、模型的设计和参数的选择等，我们特意提供一个仅供示例的版本。希望大家通过动手实战、仔细观察实
验现象、认真分析实验结果并不断调整方法，从而不断提升成绩。

计算机视觉一直是深度学习的主战场，请

> Get your hands dirty。




## Kaggle中的CIFAR-10原始图像分类问题

[Kaggle](https://www.kaggle.com)是一个著名的供机器学习爱好者交流的平台。为了便于提交结果，请大家注册[Kaggle](https
://www.kaggle.com)账号。然后请大家先点击[CIFAR-10原始图像分类问题](https://www.kaggle.com/c/cifar-1
0)了解有关本次比赛的信息。

![](../img/kaggle_cifar10.png)



## 整理原始数据集

比赛数据分为训练数据集和测试数据集。训练集包含5万张图片。测试集包含30万张图片：其中有1万张图片用来计分，但为了防止人工标注测试集，里面另加了29万张不计分的
图片。

两个数据集都是png彩色图片，大小为$32\times 32 \times 3$。训练集一共有10类图片，分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

（那么问题来了，你觉得你用肉眼能把下面100个图片正确分类吗？）

![](../img/cifar10.png)


### 下载数据集


登录Kaggle后，数据可以从[CIFAR-10原始图像分类问题](https://www.kaggle.com/c/cifar-10)中下载。

* [训练数据集train.7z下载地址](https://www.kaggle.com/c/cifar-10/download/train.7z)
* [测试数据集test.7z下载地址](https://www.kaggle.com/c/cifar-10/download/test.7z)
* [训练数据标签trainLabels.csv下载地址](https://www.kaggle.com/c/cifar-10/download/trainLa
bels.csv)

如果不登录Kaggle，比赛数据也可通过以下地址下载：

* [训练数据集train.7z下载地址](https://apache-
mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kaggle-
cifar10/train.7z)
* [测试数据集test.7z下载地址](https://apache-
mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kaggle-
cifar10/test.7z)
* [训练数据标签trainLabels.csv下载地址](https://apache-
mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kaggle-
cifar10/trainLabels.csv)


### 解压数据集

训练数据集train.7z和测试数据集test.7z都是压缩格式，下载后请解压缩。假设解压缩后原始数据集的路径如下（data_dir是存放数据的路径，例如'..
/data/kaggle_cifar10'，'train_dir'和'test_dir'分别是对train.7z和test.7z解压后的文件夹名称，例如'tra
in'和'test'）：

* data_dir/train_dir/1.png
* data_dir/train_dir/...
* data_dir/train_dir/50000.png
* data_dir/test_dir/1.png
* data_dir/test_dir/...
* data_dir/test_dir/300000.png
* data_dir/trainLabels.csv


### 整理数据集

我们定义下面的reorg_cifar10_data函数来整理数据集。整理后，同一类图片将出现在在同一个文件夹下，便于`Gluon`稍后读取。

函数中的参数如data_dir、train_dir和test_dir对应上述数据存放路径及训练和测试的图片集文件夹名称。参数label_file为训练数据标签的
文件名称。参数input_dir时整理后数据集文件夹名称。参数valid_ratio是验证集占原始训练集的比重。以valid_ratio=0.1为例，由于原始训
练数据有5万张图片，调参时将有4万5千张图片用于训练（整理后存放在input_dir/train）而另外5千张图片为验证集（整理后存放在input_dir/va
lid）。

```{.python .input  n=1}
import os
import shutil

def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    idx_label = dict()
    labels = set()
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as label_f:
        # 跳过文件头行（栏名称）。
        next(label_f)
        for line in label_f:
            idx, label = line.rstrip().split(',')
            idx_label[int(idx)] = label
            labels.add(label)

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int(num_train * (1 - valid_ratio))
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels)
    label_count = dict()
    
    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file), 
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file), 
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file), 
                        os.path.join(data_dir, input_dir, 'valid', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file), 
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
```

为了使网页编译快一点，我们在这里仅仅使用1000个训练样本和100个测试样本。训练和测试数据的文件夹名称分别为'train_1000samples'和'test
_100samples'。我们将10%的训练样本作为调参时的验证集。

```{.python .input  n=2}
# 注意：Kaggle的完整数据集应包括5万训练样本，此处为便于网页编译。
train_dir = 'train_1000samples'
# 注意：Kaggle的完整数据集应包括30万测试样本，此处为便于网页编译。
test_dir = 'test_100samples'

data_dir = '../data/kaggle_cifar10'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1
reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
```

## 使用Gluon读取整理后的数据集

我们可以使用`Gluon`中的`ImageFolderDataset`类来读取整理后的数据集。

```{.python .input  n=3}
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision

def transform(data, label):
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    return (nd.transpose(data.astype('float32'), (2,0,1)) / 255, 
            nd.array([label]).asscalar().astype('float32'))

batch_size = 100
input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, 
                                     transform=transform)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, 
                                     transform=transform)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', 
                                           flag=1, transform=transform)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                     transform=transform)

train_data = gluon.data.DataLoader(train_ds, batch_size, shuffle=True)
valid_data = gluon.data.DataLoader(valid_ds, batch_size, shuffle=True)
train_valid_data = gluon.data.DataLoader(train_valid_ds, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_ds, batch_size, shuffle=False)

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 设计模型

我们这里使用了[VGG-16模型](vgg-gluon.md)并有意做了一点改动。

请注意：模型可以重新设计，参数也可以重新调整。

```{.python .input  n=4}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for i in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
        out.add(nn.BatchNorm(axis=1))
        out.add(nn.Activation(activation='relu'))
        if i < num_convs - 1:
            out.add(nn.Dropout(0.4))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

def get_net(ctx):
    num_outputs = 10
    architecture = ((2,64), (2,128), (3,256), (3,512), (3,512))
    net = nn.Sequential()
    with net.name_scope():
        net.add(vgg_stack(architecture))
        net.add(nn.Flatten())
        net.add(nn.Dense(512, activation="relu"))
        net.add(nn.Dropout(.5))
        net.add(nn.Dense(512, activation="relu"))
        net.add(nn.Dropout(.5))
        net.add(nn.Dense(num_outputs))
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
```

## 训练模型并调参

在[过拟合](underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。由于图像分类训练时间可能较长，为
了方便，我们这里不再使用K折交叉验证，而是依赖验证集的结果来调参。

我们定义模型训练函数。这里我们记录每个epoch的训练时间。这有助于我们比较不同模型设计的时间成本。

```{.python .input  n=5}
import datetime

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, " 
                         % (epoch, train_loss / len(train_data), 
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, " 
                         % (epoch, train_loss / len(train_data), 
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
```

以下定义训练参数并训练模型。这些参数均可调。为了使网页编译快一点，我们这里将epoch数量有意设为1。事实上，epoch一般可以调大些，例如100。

我们将依据验证集的结果不断优化模型设计和调整参数。

```{.python .input  n=3}
num_epochs = 1
learning_rate = 0.1
weight_decay = 5e-4

import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()

net = get_net(ctx)
train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx)
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 2.800513, Train acc 0.107778, Valid acc 0.100000, Time 00:01:13\n"
 }
]
```

## 对测试集分类

当得到一组满意的模型设计和参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。

```{.python .input  n=6}
import numpy as np
import pandas as pd

net = get_net(ctx)
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())
    
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

上述代码执行完会生成一个`submission.csv`的文件用于在Kaggle上提交。这是Kaggle要求的提交格式。这时我们可以在Kaggle上把对测试集分
类的结果提交并查看分类准确率。你需要登录Kaggle网站，打开[CIFAR-10原始图像分类问题](https://www.kaggle.com/c/cifar
-10)，并点击下方右侧`Late Submission`按钮。

![](../img/kaggle_submit3.png)



请点击下方`Upload Submission File`选择需要提交的预测结果。然后点击下方的`Make Submission`按钮就可以查看结果啦！

![](../img/kaggle_submit4.png)



## 作业（[汇报作业和查看其他小伙伴作业]()）：

* 运行本教程，把epoch次数改为100，可以拿到什么样的准确率？
* 你还有什么其他办法可以继续改进模型和参数？小伙伴们都期待你的分享。

**吐槽和讨论欢迎点**[这里]()
