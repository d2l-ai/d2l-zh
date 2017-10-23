# 【草稿】Fine-tuning: 通过微调来迁移学习

【代码完成，但文字还没翻译完】

In previous chapters,
we demonstrated how to train a neural network
to recognize the categories corresponding to objects in images.
We looked at toy datasets like hand-written digits,
and thumbnail-sized pictures of animals.
And we talked about the ImageNet dataset,
the default academic benchmark,
which contains 1M million images,
1000 each from 1000 separate classes.

The ImageNet dataset categorically changed what was possible in computer vision.
It turns out some things are possible (these days, even easy)
on gigantic datasets, that simply aren't with smaller datasets.
In fact, we don't know of any technique that can comparably powerful model
on a similar photograph dataset but containing only, say, 10k images.

And that's a problem.
Because however impressive the results of CNNs on ImageNet may be,
most people aren't interested in ImageNet itself.
They're interested in their own problems.
Recognize people based on pictures of their faces.
Distinguish between photographs of $10$ different types of corral on the ocean
floor.
Usually when individuals (and not Amazon, Google, or inter-institutional *big
science* initiatives)
are interested in solving a computer vision problem,
they come to the table with modestly sized datasets.
A few hundred examples may be common and a few thousand examples may be as much
as you can reasonably ask for.

So one natural question emerges.
Can we somehow use the powerful models trained on millions of examples for one
dataset,
and apply them to improve performance on a new problem
with a much smaller dataset?
This kind of problem (learning on source dataset, bringing knowledge to target
dataset),
is appropriately called *transfer learning*.
Fortunately, we have some effective tools for solving this problem.

For deep neural networks, the most popular approach is called finetuning
and the idea is both simple and effective:

* Train a neural network on the source task $S$.
* Decapitate it, replacing it's output layer appropriate to target task $T$.
* Initialize the weights on the new output layer randomly, keeping all other
(pretrained) weights the same.
* Begin training on the new dataset.

This might be clearer if we visualize the algorithm:

![](../img/fine-tune.png)


In this section, we'll demonstrate fine-tuning,
using the popular and compact SqueezeNet architecture.
Since we don't want to saddle you with the burden of downloading ImageNet,
or of training on ImageNet from scratch,
we'll pull the weights of the pretrained Squeeze net from the internet.
Specifically, we'll be fine-tuning a squeezenet-1.1
that was pre-trained on imagenet-12.
Finally, we'll fine-tune it to recognize **hotdogs**.

![hot dog](../img/comic-hot-dog.png)



## 数据集

Formally, hot dog recognition is a binary classification problem.
We'll use $1$ to represent the hotdog class,
and $0$ for the *not hotdog* class.
Our hot dog dataset (the target dataset which we'll fine-tune the model to)
contains 18,141 sample images, 2091 of which are hotdogs.
Because the dataset is imbalanced (e.g. hotdog class is only 1% in mscoco
dataset),
sampling interesting negative samples can help to improve the performance of our
algorithm.
Thus, in the negative class in the our dataset,
two thirds are images from food categories (e.g. pizza) other than hotdogs,
and 30% are images from all other categories.

### 获取数据
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

```{.python .input  n=4}
from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.resnet18_v2(pretrained=True)
pretrained_net
```

可以看到这个模型有两部分组成，一是`features`，二是`classifier`。后者主要包括最后一层全连接层。我们可以看一下第一个卷积层的部分权重。

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

```{.python .input}
import sys
sys.path.append('..')
import utils

def train(net, ctx, epochs=2, learning_rate=0.1, wd=0.001):
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

```{.python .input}
ctx = utils.try_gpu()
train(finetune_net, ctx, epochs=5)
```

对比起见我们尝试从随机初始值开始训练一个网络

```{.python .input}
scratch_net = models.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train(scratch_net, ctx)
```

可以看到，微调版本收敛比从随机值开始的要快很多。

## 预测

我们定义一个预测函数，它可以读取一张图片并返回预测概率：

```{.python .input}
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
    data = img.astype('float32')
    data = nd.transpose(data, (2,1,0))
    data = nd.expand_dims(data, axis=0)
    # predict
    out = net(data.as_in_context(ctx))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    label = ['Not hotdog!', 'Hotdog!']
    return 'With prob=%f, %s'%(prob, label[pred])
```

接下来我们用训练好的图片来预测几张图片：

```{.python .input  n=17}
classify_hotdog(finetune_net, '../img/real_hotdog.jpg')
```

```{.python .input  n=18}
classify_hotdog(finetune_net, '../img/leg_hotdog.jpg')
```

```{.python .input  n=19}
classify_hotdog(finetune_net, '../img/dog_hotdog.jpg')
```

## Conclusions
As you can see, given a pretrained model, we can get a great classifier,
even for tasks where we simply don't have enough data to train from scratch.
That's because the representations necessary to perform both tasks have a lot in
common.
Since they both address natural images, they both require recognizing textures,
shapes, edges, etc.
Whenever you have a small enough dataset that you fear impoverishing your model,
try thinking about what larger datasets you might be able to pre-train your
model on,
so that you can just perform fine-tuning on the task at hand.

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
