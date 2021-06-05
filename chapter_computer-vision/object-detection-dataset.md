# 对象检测数据集
:label:`sec_object-detection-dataset`

目标检测领域没有像 MNIST 和 Fashion-Mnist 这样的小数据集。为了快速演示对象检测模型，我们收集并标记了一个小型数据集。首先，我们从办公室拍摄了免费香蕉的照片，并生成了 1000 张不同旋转和大小的香蕉图像。然后我们将每张香蕉图像放在一些背景图片上随机的位置。最后，我们在图片上为这些香蕉标记了边界框。 

## 下载数据集

包含所有图像和 csv 标签文件的香蕉检测数据集可以直接从互联网下载。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 阅读数据集

我们将阅读下面 `read_data_bananas` 函数中的香蕉检测数据集。该数据集包括一个用于对象类标签的 csv 文件以及位于左上角和右下角的地面真实边界框坐标。

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

通过使用 `read_data_bananas` 函数读取图像和标签，以下 `BananasDataset` 类将允许我们创建一个自定义的 `Dataset` 实例来加载香蕉检测数据集。

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最后，我们将 `load_data_bananas` 函数定义为训练集和测试集返回两个数据加载器实例。对于测试数据集，无需按随机顺序读取它。

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

让我们阅读一个小批次，然后在这个小批中打印图像和标签的形状。图像微型批次的形状（批量大小、通道数、高度、宽度）看起来很熟悉：它与我们之前的图像分类任务相同。标签微型批次的形状是（批量大小，$m$，5），其中 $m$ 是数据集中任何图像中尽可能多的边界框。 

尽管微型批量计算效率更高，但它要求所有图像示例包含相同数量的边界框才能通过串联形成微型批次。通常，图像可能具有不同数量的边界框；因此，在达到 $m$ 之前，边界框少于 $m$ 的图像将被非法边界框填充。然后，每个边界框的标签由长度为 5 的数组表示。数组中的第一个元素是边界框中对象的类，其中-1 表示填充的非法边界框。数组的其余四个元素是左上角和边界框右下角的 ($x$, $y$) 坐标值（范围介于 0 和 1 之间）。对于 Banana 数据集，由于每张图像上只有一个边界框，我们有 $m=1$。

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## 示范

让我们用标记的地面真实边界框来演示十幅图像。我们可以看到，所有这些图像中香蕉的旋转、大小和位置都有所不同。当然，这只是一个简单的人工数据集。实际上，真实世界的数据集通常要复杂得多。

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 摘要

* 我们收集的香蕉检测数据集可用于演示物体检测模型。
* 用于对象检测的数据加载与图像分类的数据加载类似。但是，在物体检测中，标签还包含地面真实边界框的信息，这在图像分类中缺失。

## 练习

1. 在香蕉检测数据集中使用地面真实边界框演示其他图像。它们在边界框和物体方面有什么不同？
1. 假设我们想要将数据增强（例如随机裁剪）应用于对象检测。它与图像分类中的不同有什么不同？提示：如果裁剪的图像只包含物体的一小部分会怎么办？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
