# Fine-tuning: 通过微调来迁移学习


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

这一章我们将通过[ResNet](../chapter_convolutional-neural-networks/resnet-gluon.md)来演示如何进行微调。因为通常不会每次从0开始在ImageNet上训练模型，我们直接从Gluon的模型园下载已经训练好的。然后将其迁移到一个我们感兴趣的问题上：识别**热狗**。

![hot dog](../img/comic-hot-dog.png)



## 数据集

热狗识别是一个二分类问题。我们用$1$来表示图片里面的是热狗，用$0$来表示不是热狗。我们的热狗数据集合是从网上抓取的，它有18,141张图片。可以想象说大部分正常图片里都不会含有热狗，实际上这个数据集里只有2091张图片含有热狗。所以这个是不均衡的问题。

### 获取数据

我们将数据
We prepare the dataset in the format of MXRecord using
[im2rec](http://mxnet.io/how_to/recordio.html?highlight=im2rec) tool. As of the
current draft, rec files are not yet explained in the book, but if you're
reading after November or December 2017 and you still see this note, [open an
issue on GitHub](https://github.com/zackchase/mxnet-the-straight-dope) and let
us know to stop slacking off.

- not_hotdog_train.rec 641M (1882 positive, 10000 interesting negative, and 5000
random negative)
- not_hotdog_validation.rec 49M (209 positive, 700 interesting negative, and 350
random negative)

```{.python .input  n=1}

dataset = {'train': ('not_hotdog_train-e6ef27b4.rec', '0aad7e1f16f5fb109b719a414a867bbee6ef27b4'),
                 'validation': ('not_hotdog_validation-c0201740.rec', '723ae5f8a433ed2e2bf729baec6b878ac0201740')}
```

To demo the model here, we're justgoing to use the smaller validation set.
But if you're interested in training on the full set,
set 'demo' to False in the settings at the beginning.
Now we're ready to download and verify the dataset.

```{.python .input  n=2}
from mxnet import gluon
base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/'
download_dir = '../data/'

def get_file(name, sha1):
    gluon.utils.download(base_url+name, 
                         path=download_dir+name, sha1_hash=sha1)
get_file(*dataset['train'])
get_file(*dataset['validation'])
```

### 读取

The record files can be read using [mx.io.ImageRecordIter](http://mxnet.io/api/p
ython/io.html#mxnet.io.ImageRecordIter)

```{.python .input  n=3}
from mxnet import io

train_iter = io.ImageRecordIter(
    path_imgrec=download_dir+dataset['train'][0],
    batch_size=128,
    data_shape=(3, 224, 224),
    min_img_size=256,
    rand_crop=True, 
    rand_mirror=True, 
    shuffle=True,
    max_random_scale=1.5, 
    min_random_scale=0.75
)
val_iter = io.ImageRecordIter(
    path_imgrec=download_dir+dataset['validation'][0],
    batch_size=128,
    data_shape=(3, 224, 224),
    min_img_size=256
)
```

## 模型和训练

这里我们将使用Gluon提供的ResNet18来训练。我们先从模型园里获取改良过ResNet。使用`pretrained=True`将会自动下载并加载从ImageNet数据集上训练而来的权重。

```{.python .input  n=6}
from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.resnet18_v2(pretrained=True)
```

通常预训练好的模型由两块构成，一是`features`，二是`classifier`。后者主要包括最后一层全连接层，前者包含从输入开始的大部分层。这样的划分的一个主要目的是为了更方便做微调。我们先看下`classifer`的内容：

```{.python .input  n=7}
pretrained_net.classifier
```

【注意】未来几天我们可能会将`classifier`重命名成`output`，并在里面只保留最后的Dense层。

我们可以看一下第一个卷积层的部分权重。

```{.python .input  n=5}
pretrained_net.features[1].params.get('weight').data()[0][0]
```

在微调里，我们一般新建一个网络，它的定义跟之前训练好的网络一样，除了最后的输出数等于当前数据的类别数。新网络的`features`被初始化前面训练好网络的权重，而`classfier`则是从头开始训练。

```{.python .input  n=6}
from mxnet import init

finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.classifier.initialize(init.Xavier())
```

我们先定义一个可以重复使用的训练函数。跟之前教程不一样的地方在于这里我们对正例预测错误做了更大惩罚。

```{.python .input  n=7}
import sys
sys.path.append('..')
import utils

def train(net, ctx, epochs=10, learning_rate=0.1, wd=0.001):
    # 确保net的初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # 自定义损失函数，加大了正例错误的惩罚
    sce = gluon.loss.SoftmaxCrossEntropyLoss()
    def loss(yhat, y):
        return sce(yhat, y) * (1+y*5)/6
    # 训练
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_iter, val_iter, net, loss, trainer, ctx, epochs)
```

现在我们可以训练了。

```{.python .input  n=8}
ctx = utils.try_all_gpus()
train(finetune_net, ctx)
```

对比起见我们尝试从随机初始值开始训练一个网络

```{.python .input  n=9}
scratch_net = models.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train(scratch_net, ctx)
```

可以看到，微调版本收敛比从随机值开始的要快很多。

## 预测

我们定义一个预测函数，它可以读取一张图片并返回预测概率：

```{.python .input  n=10}
%matplotlib inline
import matplotlib.pyplot as plt
from mxnet import image
from mxnet import nd

def classify_hotdog(net, fname):
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())
    # reisze and crop
    img = image.resize_short(img, 256)
    img, _ = image.center_crop(img, (224,224))
    plt.imshow(img.asnumpy())
    # from h x w x c -> b x c x h x w
    data = img.transpose((2,0,1)).expand_dims(axis=0)
    # predict
    out = net(data.astype('float32').as_in_context(ctx[0]))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    label = ['Not hotdog!', 'Hotdog!']
    return 'With prob=%f, %s'%(prob, label[pred])
```

接下来我们用训练好的图片来预测几张图片：

```{.python .input  n=11}
classify_hotdog(finetune_net, '../img/real_hotdog.jpg')
```

```{.python .input  n=12}
classify_hotdog(finetune_net, '../img/leg_hotdog.jpg')
```

```{.python .input  n=13}
classify_hotdog(finetune_net, '../img/dog_hotdog.jpg')
```

## 结论

我们看到通过一个预先训练好的模型，我们可以在即使小的数据集上训练得到很好的分类器。这是因为这两个任务里面的数据表示有很多共通性。例如他们都需要如何识别纹理，形状，边等等。而这些通常被在靠近数据的层有效的处理。因此，如果你有一个相对较小的数据在手，而且当心它可能不够训练出很好的模型，你可以寻找跟你数据类似的大数据集来先训练你的模型，然后再在你手上的数据集上微调。

## 练习

- 多跑几个`epochs`直到收敛（你可以也需要调调参数），看看`scratch_net`和`finetune_net`最后的精度是不是有区别
- 这里`finetune_net`重用了`pretrained_net`除最后全连接外的所有权重，试试少重用些权重，有会有什么区别
- 事实上`ImageNet`里也有`hotdog`这个类，它的index是713。例如它对应的weight可以这样拿到
  ```
  weight = pretrained_net.classifier[4].params.get('weight')
  nd.split(weight,1000, axis=0)[713]
  ```
  试试重用`classifer`里重用hotdog的权重
- 试试不让`finetune_net`里重用的权重参与训练，就是不更新权重
