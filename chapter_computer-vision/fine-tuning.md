# 迁移学习


在前面的章节里我们展示了如何训练神经网络来识别小图片里的问题。我们也介绍了ImageNet这个学术界默认的数据集，它有超过一百万的图片和一千类的物体。这个数据集很大的改变计算机视觉这个领域，展示了很多事情虽然在小的数据集上做不到，但在数GB的大数据上是可能的。事实上，我们目前还不知道有什么技术可以在类似的但小图片数据集上，例如一万张图片，训练出一个同样强大的模型。

所以这是一个问题。尽管深度卷积神经网络在ImageNet上有了很惊讶的结果，但大部分人不关心Imagenet这个数据集本身。他们关心他们自己的问题。例如通过图片里面的人脸识别身份，或者识别图片里面的10种不同的珊瑚。通常大部分在非BAT类似大机构里的人在解决计算机视觉问题的时候，能获得的只是相对来说中等规模的数据。几百张图片很正常，找到几千张图片也有可能，但很难同Imagenet一样获得上百万张图片。

于是我们会有一个很自然的问题，如何使用在百万张图片上训练出来的强大的模型来帮助提升在小数据集上的精度呢？这种在源数据上训练，然后将学到的知识应用到目标数据集上的技术通常被叫做**迁移学习**。幸运的是，我们有一些有效的技术来解决这个问题。

对于深度神经网络来首，最为流行的一个方法叫做微调（fine-tuning）。它的想法很简单但有效：


* 在源数据 $S$ 上训练一个神经网络。
* 砍掉它的头，将它的输出层改成适合目标数据 $S$ 的大小
* 将输出层的权重初始化成随机值，但其它层保持跟原先训练好的权重一致
* 然后开始在目标数据集开始训练

下图图示了这个算法：

![](../img/fine-tuning.svg)

## 热狗识别

这一章我们将通过[ResNet](../chapter_convolutional-neural-networks/resnet-gluon.md)来演示如何进行微调。因为通常不会每次从0开始在ImageNet上训练模型，我们直接从Gluon的模型园下载已经训练好的。然后将其迁移到一个我们感兴趣的问题上：识别热狗。

![hot dog](../img/comic-hot-dog.png)


热狗识别是一个二分类问题。我们这里使用的热狗数据集是从网上抓取的，它有$1400$张正类和同样多的负类，负类主要是食品相关图片。我们将各类的$1000$张作为训练集合，其余的作为测试集合。

### 获取数据

我们首先从网上下载数据并解压到`../data/hotdog`。每个文件夹下会有对应的`png`文件。

```{.python .input  n=17}
from mxnet import gluon
import zipfile

data_dir = '../data'
fname = gluon.utils.download(
    'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')

with zipfile.ZipFile(fname, 'r') as f:
    f.extractall(data_dir)
```

我们使用[图片增强](../image-augmentation.md)里类似的方法来处理图片。

```{.python .input  n=18}
from mxnet import nd
from mxnet import image
from mxnet import gluon

train_augs = [
    image.HorizontalFlipAug(.5),
    image.RandomCropAug((224,224))
]

test_augs = [
    image.CenterCropAug((224,224))
]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')
```

读取文件夹下的图片，并且画出一些图片

```{.python .input  n=20}
%matplotlib inline
import sys
sys.path.append('..')
import utils

train_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/hotdog/train',
    transform=lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/hotdog/test',
    transform=lambda X, y: transform(X, y, test_augs))

data = gluon.data.DataLoader(train_imgs, 32, shuffle=True)
for X, _ in data:
    X = X.transpose((0,2,3,1)).clip(0,255)/255
    utils.show_images(X, 4, 8)
    break
```

### 模型和训练

这里我们将使用Gluon提供的ResNet18来训练。我们先从模型园里获取改良过ResNet。使用`pretrained=True`将会自动下载并加载从ImageNet数据集上训练而来的权重。

```{.python .input  n=21}
from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.resnet18_v2(pretrained=True)
```

通常预训练好的模型由两块构成，一是`features`，二是`output`。后者主要包括最后一层全连接层，前者包含从输入开始的大部分层。这样的划分的一个主要目的是为了更方便做微调。我们先看下`output`的内容：

```{.python .input  n=22}
pretrained_net.output
```

我们可以看一下第一个卷积层的部分权重。

```{.python .input  n=23}
pretrained_net.features[1].weight.data()[0][0]
```

在微调里，我们一般新建一个网络，它的定义跟之前训练好的网络一样，除了最后的输出数等于当前数据的类别数。新网络的`features`被初始化前面训练好网络的权重，而`output`则是从头开始训练。

```{.python .input  n=24}
from mxnet import init

finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
```

我们先定义一个可以重复使用的训练函数。

```{.python .input  n=25}

def train(net, ctx, batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)

    # 确保net的初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # 训练
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)
```

现在我们可以训练了。

```{.python .input  n=10}
ctx = utils.try_all_gpus()
train(finetune_net, ctx)
```

对比起见我们尝试从随机初始值开始训练一个网络。

```{.python .input  n=11}
scratch_net = models.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train(scratch_net, ctx)
```

可以看到，微调版本收敛比从随机值开始的要快很多。

### 图片预测

```{.python .input  n=12}
import matplotlib.pyplot as plt

def classify_hotdog(net, fname):
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())
    data, _ = transform(img, -1, test_augs)
    plt.imshow(data.transpose((1,2,0)).asnumpy()/255)
    data = data.expand_dims(axis=0)
    out = net(data.as_in_context(ctx[0]))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    label = train_imgs.synsets
    return 'With prob=%f, %s'%(prob, label[pred])
```

接下来我们用训练好的模型来预测几张图片：

```{.python .input  n=13}
classify_hotdog(finetune_net, '../img/real_hotdog.jpg')
```

```{.python .input  n=14}
classify_hotdog(finetune_net, '../img/leg_hotdog.jpg')
```

```{.python .input  n=15}
classify_hotdog(finetune_net, '../img/dog_hotdog.jpg')
```

## 小结

* 我们看到，通过一个预先训练好的模型，我们可以在即使较小的数据集上训练得到很好的分类器。这是因为这两个任务里面的数据表示有很多共通性，例如都需要如何识别纹理、形状、边等等。而这些通常被在靠近数据的层有效的处理。因此，如果你有一个相对较小的数据在手，而且担心它可能不够训练出很好的模型，你可以寻找跟你数据类似的大数据集来先训练你的模型，然后再在你手上的数据集上微调。

## 练习

- 多跑几个`epochs`直到收敛（你可以也需要调调参数），看看`scratch_net`和`finetune_net`最后的精度是不是有区别
- 这里`finetune_net`重用了`pretrained_net`除最后全连接外的所有权重，试试少重用些权重，有会有什么区别
- 事实上`ImageNet`里也有`hotdog`这个类，它的index是713。例如它对应的weight可以这样拿到。试试如何重用这个权重

```{.python .input  n=16}
weight = pretrained_net.output.weight
hotdog_w = nd.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

- 试试不让`finetune_net`里重用的权重参与训练，就是不更新权重
- 如果图片预测这一章里我们训练的模型没有分对所有的图片，如何改进？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2272)

![](../img/qr_fine-tuning.svg)
