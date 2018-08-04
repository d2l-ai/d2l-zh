# 物体检测数据集

在物体检测领域并没有类似MNIST那样的小数据集方便我们快速测试模型。为此我们合成了一个小的人工数据集。我们首先使用一个开源的皮卡丘3D模型生成1000张不同角度和大小的图片。然后我们收集了一系列背景图片，并在每张图的随机位置放置一张皮卡丘图片。我们使用MXNet提供的[tools/im2rec.py](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py)来将图片打包成二进制rec文件。（这是MXNet在Gluon开发出来之前常用的数据格式。注意GluonCV这个包里已经提供了更简单的类似之前我们读取图片的函数，从而可以省略打包图片的步骤。但由于这个工具包目前仍在快速开发迭代中，这里我们仍使用rec格式。）

## 下载数据集

打包好的数据集可以直接在网上下载。下载数据集的操作定义在`_download_pikachu`函数中。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import gluon, image
from mxnet.gluon import utils as gutils
import os

def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)
```

## 读取数据集

我们使用`image.ImageDetIter`来读取数据。这是针对物体检测的迭代器，(Det表示Detection)。在读取训练图片时我们做了随机剪裁。我们将读取数据集的`load_data_pikachu`函数定义在`gluonbook`包中供后面章节调用。

```{.python .input  n=2}
# edge_size：输出图片的宽和高。
def load_data_pikachu(batch_size, edge_size=256): 
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)                                                                                                                 
    train_iter = image.ImageDetIter(
        path_imgrec =os.path.join(data_dir, 'train.rec'),
        # 每张图片在rec中的位置，使用随机顺序时需要。
        path_imgidx=os.path.join(data_dir, 'train.idx'), 
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图片形状。
        shuffle=True,  # 用随机顺序访问。
        rand_crop=1,  # 一定使用随机剪裁。
        min_object_covered=0.95,  # 剪裁出的图片至少覆盖每个物体95%的区域。
        max_attempts=200)  # 最多尝试 200 次随机剪裁。如果失败则不进行剪裁。
    val_iter = image.ImageDetIter(  # 测试图片则去除了随机访问和随机剪裁。
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False)
    return train_iter, val_iter

batch_size = 32
edge_size = 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
```

下面我们读取一个批量。

```{.python .input  n=3}
batch = train_iter.next()
batch.data[0].shape, batch.label[0].shape
```

可以看到图片的形状跟之前图片分类时一样，但标签的形状是（批量大小，每张图片中最大边界框数，5）。每个边界框的由长为5的数组表示，第一个元素是其对用物体的标号，其中`-1`表示非法，仅做填充使用。后面4个元素表示边界框位置。这里使用的数据相对简单，每张图片只有一个边界框。实际使用的物体检测数据集中每张图片可能会有多个边界框，但我们要求每张图片有相同数量的边界框使得多张图可以放在一个批量里高效处理。所以我们会使用一个最大边界框数，对于物体数量偏少的图片填充非法边界框直到最大边界框数。

## 图示数据

我们画出几张图片和其对应的标号。可以看到比卡丘的角度大小位置在每张图图片都不一样。当然，这是一个简单的人工数据集，物体和背景的区别较大。实际中遇到的数据集通常会复杂很多。

```{.python .input  n=4}
imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))).clip(0, 254) / 254
axes = gb.show_images(imgs, 2, 5).flatten()
for ax, label in zip(axes, batch.label[0][0:10]):
    gb.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 小结

* 物体识别的数据读取跟图片分类类似，但引入了边界框后导致标注形状和图片增强均有所不同。

## 练习

* 了解下`image.ImageDetIter`和`image.CreateDetAugmenter`这两个类的创建参数。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7022)

![](../img/qr_object-detection-dataset.svg)
