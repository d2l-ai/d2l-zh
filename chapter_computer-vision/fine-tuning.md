# 微调
:label:`sec_fine_tuning`

在前面的章节中，我们讨论了如何在 Fashion-Mnist 训练数据集上训练模型，只有 60000 张图像。我们还描述了 iMagenNet，这是学术界中使用最广泛的大型图像数据集，它拥有 1000 多万张图像和 1000 个对象。但是，我们通常遇到的数据集的大小介于两个数据集中的大小之间。 

假设我们想识别图片中不同类型的椅子，然后向用户推荐购买链接。一种可能的方法是首先识别 100 把普通椅子，为每把椅子拍摄 1000 张不同角度的图像，然后在收集的影像数据集上训练一个分类模型。尽管这个椅子数据集可能大于 Fashion-Mnist 数据集，但实例数量仍然不到 iMagenet 中的十分之一。这可能会导致这个椅子数据集上适合 iMagenNet 的复杂模型过度拟合。此外，由于训练示例数量有限，训练模型的准确性可能无法满足实际要求。 

为了解决上述问题，一个显而易见的解决方案是收集更多的数据。但是，收集和标记数据可能需要大量的时间和金钱。例如，为了收集 iMagenet 数据集，研究人员从研究资金中花费了数百万美元。尽管目前的数据收集成本已大幅降低，但这一成本仍不能忽视。 

另一种解决方案是应用 * 传输学习 * 将从 * 源数据集 * 学到的知识传输到 * 目标数据集 *。例如，尽管 iMagenet 数据集中的大多数图像与椅子无关，但在此数据集上训练的模型可能会提取更常规的图像特征，这有助于识别边缘、纹理、形状和对象合成。这些类似的功能也可能有效地识别椅子。 

## 步骤

在本节中，我们将介绍转移学习中的常见技巧 : *fine-tuning*. As shown in :numref:`fig_finetune`，微调包括以下四个步骤： 

1. 在源数据集（例如 iMagenet 数据集）上预训练神经网络模型，即 * 源模型 *。
1. 创建一个新的神经网络模型，即 * 目标模型 *。这将复制源模型上的所有模型设计及其参数，但输出层除外。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出图层与源数据集的标签密切相关；因此不在目标模型中使用该图层。
1. 向目标模型添加输出图层，其输出数量是目标数据集中的类别数。然后随机初始化该层的模型参数。
1. 在目标数据集（如椅子数据集）上训练目标模型。输出图层将从头开始进行训练，而所有其他图层的参数将根据源模型的参数进行微调。

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

当目标数据集比源数据集小得多时，微调有助于提高模型的泛化能力。 

## 热狗识别

让我们通过具体案例演示微调：热狗识别。我们将在一个小型数据集上微调 ReSnet 模型，该数据集已在 iMagenet 数据集上进行了预训练。这个小型数据集包含数千张带热狗和不带热狗的图像。我们将使用微调模型来识别图像中的热狗。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### 阅读数据集

我们使用的热狗数据集取自在线图片。该数据集包含 1400 张包含热狗的正面类图像以及包含其他食物的尽可能多的负面级图像。两个课程的 1000 张图片用于训练，其余的则用于测试。 

解压下载的数据集后，我们获得了两个文件夹 `hotdog/train` 和 `hotdog/test`。这两个文件夹都有 `hotdog` 和 `not-hotdog` 个子文件夹，其中任何一个文件夹都包含相应类的图像。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

我们创建两个实例来分别读取训练和测试数据集中的所有图像文件。

```{.python .input}
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

下面显示了前 8 个正面示例和最后 8 张负面图片。正如你所看到的，图像的大小和纵横比有所不同。

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

在训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为 $224 \times 224$ 输入图像。在测试过程中，我们将图像的高度和宽度都缩放到 256 像素，然后裁剪中央 $224 \times 224$ 区域作为输入。此外，对于三个 RGB（红、绿和蓝）颜色通道，我们按频道 * 标准化 * 它们的价值通道。具体而言，通道的平均值将从该通道的每个值中减去，然后将结果除以该通道的标准差。

```{.python .input}
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### 定义和初始化模型

我们使用在 iMagenet 数据集上预训练的 Resnet-18 作为源模型。在这里，我们指定 `pretrained=True` 以自动下载预训练的模型参数。如果首次使用此模型，则需要互联网连接才能下载。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
预训练的源模型实例包含两个成员变量：`features` 和 `output`。前者包含除输出层以外的模型的所有层，后者是模型的输出层。此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调。源模型的成员变量 `output` 如下所示。
:end_tab:

:begin_tab:`pytorch`
预训练的源模型实例包含许多要素图层和一个输出图层 `fc`。此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调。下面给出了源模型的成员变量 `fc`。
:end_tab:

```{.python .input}
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

作为一个完全连接的层，它将 RESNet 的最终全球平均池输出转换为 iMagenet 数据集的 1000 个类输出。然后，我们构建一个新的神经网络作为目标模型。它的定义方式与预训练源模型的定义方式相同，只是最终图层中的输出数量被设置为目标数据集中的类数（而不是 1000 个）。 

在下面的代码中，目标模型实例 finetune_net 的成员变量特征中的模型参数被初始化为源模型相应层的模型参数。由于功能中的模型参数是在 iMagenNet 数据集上预训练的，并且足够好，因此通常只需要较小的学习速率即可微调这些参数。  

成员变量输出中的模型参数是随机初始化的，通常需要更高的学习速率才能从头开始训练。假设 Trainer 实例中的学习速率为，我们将迭代中成员变量输出中模型参数的学习速率设置为 10。 

在下面的代码中，初始化目标模型实例 `finetune_net` 输出层之前的模型参数，以对源模型中相应层的参数进行建模。由于这些模型参数是通过 iMagenet 上的预训练获得的，因此它们很有效。因此，我们只能使用较小的学习速率进行 * 微调 * 这样的预训练参数。相比之下，输出层中的模型参数是随机初始化的，通常需要从头开始学习更高的学习速率。让基本学习速率为 $\eta$，学习速率 $10\eta$ 将用于迭代输出层中的模型参数。

```{.python .input}
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### 微调模型

首先，我们定义了一个训练函数 `train_fine_tuning`，该函数使用微调，因此可以多次调用。

```{.python .input}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

我们将基本学习速率设置为小值，以便 * 微调 * 通过预训练获得的模型参数。根据之前的设置，我们将使用高十倍的学习率从头开始训练目标模型的输出层参数。

```{.python .input}
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

为了进行比较，我们定义了一个相同的模型，但是将其所有模型参数初始化为随机值。由于整个模型需要从头开始训练，因此我们可以使用更大的学习率。

```{.python .input}
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

正如我们所看到的，微调模型在同一纪元中往往表现更好，因为它的初始参数值更有效。 

## 摘要

* 转移学习将从源数据集中学到的知识传输到目标数据集。微调是转移学习的常见技巧。
* 目标模型将从源模型中复制所有模型设计及其参数，但输出层除外，并根据目标数据集对这些参数进行微调。相比之下，需要从头开始训练目标模型的输出层。
* 通常，微调参数使用较小的学习速率，而从头开始训练输出层可以使用更大的学习速率。

## 练习

1. 继续提高 `finetune_net` 的学习率。模型的准确性如何变化？
2. 在比较实验中进一步调整 `finetune_net` 和 `scratch_net` 的超参数。它们的准确性还有不同吗？
3. 将输出层 `finetune_net` 之前的参数设置为源模型的参数，在训练期间不要 * 更新它们。模型的准确性如何变化？你可以使用以下代码。

```{.python .input}
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. 事实上，`ImageNet` 数据集中有一个 “热狗” 类。可以通过以下代码获取输出层中的相应权重参数。我们怎样才能利用这个权重参数？

```{.python .input}
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[713]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
