# 物体检测数据集

在物体检测领域并没有类似MNIST那样的小数据集方便我们快速测试模型。为此我们合成了一个小的人工数据集。我们首先使用一个开源的皮卡丘3D模型生成1000张不同角度和大小的图片。然后我们收集了一系列背景图片，并在每张图的随机位置放置一张皮卡丘图片。我们使用MXNet提供的[tools/im2rec.py](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py)来将图片打包成二进制rec文件。（这是MXNet在Gluon开发出来之前常用的数据格式。注意到GluonCV这个包里已经提供了更简单的类似之前我们读取图片的方式，从而无需打包图片。但由于这个包目前仍在快速迭代中，所以这里我们使用rec格式。）

## 下载数据

打包好的数据可以直接在网上下载：

```{.python .input}
%matplotlib inline
import sys
sys.path.insert(0, '..')
import gluonbook as gb
from mxnet import image, gluon
```

```{.python .input  n=81}
root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
            'gluon/dataset/pikachu/')
data_dir = '../data/pikachu/'
dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in dataset.items():
    gluon.utils.download(root_url+k, data_dir+k, sha1_hash=v)
```

## 读取数据集

我们使用`image.ImageDetIter`来读取数据。这是针对物体检测的迭代器，(Det表示Detection)。在读取训练图片时我们做了随机剪裁。读取数据集的方法我们保存在GluonBook中的`load_data_pikachu`函数里。

```{.python .input  n=85}
edge_size = 256 # 输出图片的宽和高。
batch_size = 32

train_data = image.ImageDetIter(
    path_imgrec=data_dir+'train.rec',
    path_imgidx=data_dir+'train.idx',  # 每张图片在rec中的位置，使用随机顺序时需要。
    batch_size=batch_size,
    data_shape=(3, edge_size, edge_size), # 输出图片形状。
    shuffle=True,  # 用随机顺序访问。
    rand_crop=1,  # 一定使用随机剪裁。
    min_object_covered=0.95,  # 剪裁出的图片至少覆盖每个物体95%的区域。
    max_attempts=200) # 最多尝试 200 次随机剪裁。如果失败则不进行剪裁。

val_data = image.ImageDetIter(  # 测试图片则去除了随机访问和随机剪裁。
    path_imgrec=data_dir+'val.rec',
    batch_size=batch_size,
    data_shape=(3, edge_size, edge_size),
    shuffle=False)
```

```{.python .input}
image.ImageDetIter?
```

下面我们读取一个批量。

```{.python .input  n=86}
batch = train_data.next()
(batch.data[0].shape, batch.label[0].shape)
```

可以看到图片的形状跟之前图片分类时一样，但标签的形状是（批量大小，每张图片中最大边界框数，5）。每个边界框的由长为5的数组表示，第一个元素是其对用物体的标号，其中`-1`表示非法，仅做填充使用。后面4个元素表示边界框位置。这里使用的数据相对简单，每张图片只有一个边界框。一般使用的物体检测数据中每张图片可能会有多个边界框，但我们要求每张图片有相同数量的边界框使得可以放在一个批量里。所以我们会使用一个最大边界框数，对于不够的图片使用填充边界框。

## 图示数据

我们先定义一个函数可以画出多个边界框（以及他们的标注），它将被保存在GluonBook里以便后面使用。

```{.python .input}
def show_bboxes(axes, bboxs, labels=None, colors=['b','g','r','m','k']):
    for i, bbox in enumerate(bboxs):
        color = colors[i%len(colors)]
        rect = gb.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va="center", ha="center", fontsize=9, color='white',
                      bbox=dict(facecolor=color, lw=0))
```

我们画出几张图片和其对应的标号。可以看到比卡丘的角度大小位置在每张图图片都不一样。当然，这是一个简单的人工数据集，物体和背景的区别较大。实际中遇到的数据集通常会复杂很多。

```{.python .input  n=19}
imgs = (batch.data[0][0:10].transpose((0,2,3,1)) ).clip(0, 254)/254
axes = gb.show_images(imgs, 2, 5).flatten()
for ax, label in zip(axes, batch.label[0][0:10]):
    show_bboxes(ax, [label[0][1:5]*edge_size], colors=['w'])
```

## 小结

* 物体识别的数据读取跟图片分类类似，但引入了边界框后导致标注形状和图片增强均有所不同。

## 练习

* 了解下`image.ImageDetIter`和`image.CreateDetAugmenter`这两个类的创建参数。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7022)

![](../img/qr_pikachu.svg)
