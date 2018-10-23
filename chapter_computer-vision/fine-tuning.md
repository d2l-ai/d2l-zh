# 微调

在早先的一些章节中，我们介绍了如何在只有6万张图像的Fashion-MNIST训练数据集上训练模型。我们还描述了学术界当下使用最广泛的大规模图像数据集ImageNet，它有超过一千万的图像和一千类的物体。然而，我们平常接触到数据集的规模通常在这两者之间。

假设我们想从图像中识别出不同种类的椅子，然后将购买链接推荐给用户。一个可能的方法是先找出一百种常见的椅子，为每种椅子拍摄一千张不同角度的图像；然后在收集到的图像数据集上训练一个分类模型。这个数据集虽然可能比Fashion-MNIST要庞大，但样本数仍然不及ImageNet样本数的十分之一。这可能会导致适用于ImageNet的复杂模型在这个数据集上过拟合。同时，因为数据量有限，最终训练得到的模型的精度也可能达不到实用的要求。

为了应对上述问题，一个显而易见的解决办法是收集更多的数据。然而，收集和标注数据会花费大量的时间和资金。例如，为了收集ImageNet数据集，研究人员花费了数百万美元的研究经费。虽然目前的数据采集成本已降低了不少，但其成本仍然不可忽略。

另外一种解决办法是应用迁移学习（transfer learning），将从源数据集学到的知识迁移到目标数据集上。例如，虽然ImageNet的图像大多跟椅子无关，但在该数据集上训练的模型可以抽取较通用图像特征，从而能够帮助识别边缘、纹理、形状和物体组成等。这些类似的特征对于识别椅子也可能同样有效。

本小节我们介绍迁移学习中的一个常用技术：微调。如图9.1所示，微调由以下四步构成：

1. 在源数据集（例如ImageNet数据集）上训练一个神经网络模型$A$。
2. 创建一个新的神经网络模型$B$，它复制了模型$A$上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设模型$A$的输出层跟源数据集的标签紧密相关，因此不予采用。
3. 为模型$B$添加一个输出大小为目标数据集类别个数的输出层，并对该层的权重参数做随机初始化。
4. 在目标数据集（例如椅子数据集）上训练模型$B$。我们将从头开始学习输出层，而其余层的参数都是基于模型$A$的参数进行微调得到的。

![微调。](../img/finetune.svg)


## 热狗识别

接下来我们来实践一个具体的例子：热狗识别。我们将基于一个小数据集对在ImageNet数据集上训练好的ResNet模型进行微调。该小数据集含有数千张包含热狗和不包含热狗的图像。我们将使用微调得到的模型来识别一张图像中是否包含热狗。

首先，导入实验所需的包或模块。Gluon的`model_zoo`包提供了常用的预训练模型。如果你希望获取更多的计算机视觉的预训练模型，可以使用GluonCV工具包 [1]。

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils
import os
import zipfile
```

### 获取数据集

我们使用的热狗数据集是从网上抓取的，它含有1400张包含热狗的正例图像，和同样多包含其他食品的负例图像。各类的1000张图像被用作训练，其余则用于测试。

我们首先将数据下载到`../data`。在下载目录将下载好的数据集进行解压后得到`hotdog/train`和`hotdog/test`这两个文件夹。在这两个文件夹下面均有`hotdog`和`not-hotdog`两个类别文件夹，每个类别文件夹里面是对应的图像文件。

```{.python .input  n=4}
data_dir = '../data'
base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
fname = gutils.download(
    base_url + 'gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')

with zipfile.ZipFile(fname, 'r') as z:
    z.extractall(data_dir)
```

我们使用`ImageFolderDataset`类来读取数据。它将每个文件夹当做一个类，并读取下面所有的图像。

```{.python .input  n=6}
train_imgs = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'hotdog/train'))
test_imgs = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'hotdog/test'))
```

下面画出前8张正例图像和最后的8张负例图像，可以看到它们的大小和长宽比各不相同。

```{.python .input}
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
gb.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

在训练时，我们先从图像中剪裁出随机大小和随机长宽比的一块，然后将它们统一缩放为长宽都是224的输入。测试时，则使用简单的中心剪裁。此外，我们对输入的RGB通道数值进行了归一化。

```{.python .input  n=3}
# 指定 RGB 三个通道的均值和方差来将图像通道归一化。
normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize
])

test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize
])
```

### 微调模型

我们用在ImageNet上预先训练的ResNet-18作为基础模型。这里指定`pretrained=True`来自动下载并加载预先训练的权重。

```{.python .input  n=6}
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
```

预训练好的模型由两部分构成：`features`和`output`。前者包含从输入开始的所有卷积和全连接层，后者主要包括最后一层全连接层。这样划分的主要目的是为了更方便做微调。我们来看一下`output`的内容：

```{.python .input  n=7}
pretrained_net.output
```

它将ResNet最后的全局平均池化层输出转化成1000类的输出。

在微调中，我们新建一个网络，它的定义跟之前训练好的网络一样，但是最后的输出数等于当前数据的类别数。也就是说新网络的`features`被初始化成前面训练好网络的权重，而`output`则是从头开始训练的。

```{.python .input  n=9}
finetune_net = model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
```

### 训练模型

我们先定义一个可以重复使用的训练函数。

```{.python .input  n=12}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gdata.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)

    ctx = gb.try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

因为微调的网络中的主要层已经训练的足够好，所以一般采用比较小的学习率，以防止过大的步长对其产生过多影响。

```{.python .input  n=13}
train_fine_tuning(finetune_net, 0.01)
```

作为对比，我们训练一个同样的模型，但将所有参数都初始化为随机值。我们使用较大的学习率来加速收敛。

```{.python .input  n=14}
scratch_net = model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

可以看到，微调的模型因为初始值更好，在相同迭代周期下能够取得更好的结果。在很多情况下，微调的模型最终都会比非微调的模型取得更好的结果。


## 小结

* 微调通过将模型部分权重初始化成在源数据集上预训练的模型权重，从而将模型在源数据集上学到的知识迁移到目标数据上。

## 练习

- 试着增大`finetune_net`的学习率看看收敛变化。
- 多跑几个`num_epochs`直到收敛（其他参数可能也需要微调），看看`scratch_net`和`finetune_net`最后的精度是不是有区别。
- 这里`finetune_net`重用了`pretrained_net`除最后全连接外的所有权重，试试少重用些权重，又会有什么区别？
- 事实上`ImageNet`里也有`hotdog`这个类，它对应的输出层参数可以用如下代码拿到。试试如何使用它。

```{.python .input  n=16}
weight = pretrained_net.output.weight
hotdog_w = nd.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

- 试试不让`finetune_net`里重用的权重参与训练，也就是不更新他们的权重。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2272)

![](../img/qr_fine-tuning.svg)

## 参考文献

[1] GluonCV工具包。https://gluon-cv.mxnet.io/
