# 语义分割和数据集

图片分类关心识别图片里面的主要物体，物体识别则进一步找出图片的多个物体以及它们的方形边界框。本小节我们将介绍语义分割（semantic segmentation），它在物体识别上更进一步的找出物体的精确边界框。换句话说，它识别图片中的每个像素属于哪类我们感兴趣的物体还是只是背景。图9.9演示猫和狗图片在语义分割中的标注。可以看到，跟物体识别相比，语义分割预测的边框更加精细。

![语义分割的训练数据和标注。](../img/segmentation.svg)

在计算机视觉里，还有两个跟语义分割相似的任务。一个是图片分割（image segmentation），它也是将像素划分到不同的类。不同的是，语义分割里我们赋予像素语义信息，例如属于猫、狗或者背景。而图片分割则通常根据像素本身之间的相似性，它训练时不需要像素标注信息，其预测结果也不能保证有语义性。例如图片分割可能将图9.9中的狗划分成两个区域，其中一个嘴巴和眼睛，其颜色以黑色为主，另一个是身体其余部分，其主色调是黄色。

另一个应用是实例分割（instance segementation），它不仅需要知道每个像素的语义，即属于那一类物体，还需要进一步区分物体实例。例如如果图片中有两只狗，那么对于预测为对应狗的像素是属于第一只狗还是第二只。

## Pascal VOC语义分割数据集

下面我们使用一个常用的语义分割数据集
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)来介绍这个应用。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import gluon, image, nd
from mxnet.gluon import data as gdata, utils as gutils
import os
import sys
import tarfile
```

我们首先下载这个数据集到`../data`下。压缩包大小是2GB，下载需要一定时间。解压之后这个数据集将会放置在`../data/VOCdevkit/VOC2012`下。

```{.python .input  n=2}
def download_voc_pascal(data_dir='../data'):
    """Download the Pascal VOC2012 Dataset."""
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
           '/VOCtrainval_11-May-2012.tar')
    sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
    return voc_dir

voc_dir = download_voc_pascal()
```

在`ImageSets/Segmentation`下有文本文件指定哪些样本用来训练，哪些用来测试。样本图片放置在`JPEGImages`下，标注则放在`SegmentationClass`下。这里标注也是图片格式，图片大小与对应的样本图片一致，其中颜色相同的像素属于同一个类。

下面定义函数将图片和标注全部读进内存。

```{.python .input  n=3}
def read_voc_images(root=voc_dir, train=True):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data, label = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        label[i] = image.imread('%s/SegmentationClass/%s.png' % (root, fname))
    return data, label

train_images, train_labels = read_voc_images()
```

我们画出前面五张图片和它们对应的标注。在标注，白色代表边框黑色代表背景，其他不同的颜色对应不同物体类别。

```{.python .input  n=4}
n = 5
imgs = train_images[0:n] + train_labels[0:n]
gb.show_images(imgs, 2, n);
```

接下来我们列出标注中每个 RGB 颜色值对应的类别。

```{.python .input  n=5}
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

这样给定一个标号图片，我们就可以将每个像素对应的物体标号找出来。

```{.python .input  n=6}
colormap2label = nd.zeros(256**3)
for i, cm in enumerate(voc_colormap):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def voc_label_indices(img, colormap2label):
    data = img.astype('int32')
    idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
    return colormap2label[idx]
```

可以看到第一张样本中飞机头部对应的标注里属于飞机的像素被标记成了1。

```{.python .input  n=7}
y = voc_label_indices(train_labels[0], colormap2label)
y[105:115, 130:140], voc_classes[1]
```

### 数据预处理

我们知道小批量训练需要输入图片的形状一致。之前我们通过图片缩放来得到同样形状的输入。但语义分割里，如果对样本图片进行缩放，那么重新映射每个像素对应的类别将变得困难，特别是对应物体边缘的像素。

为了避免这个困难，这里我们将图片剪裁成固定大小而不是缩放。特别的，我们使用随机剪裁来附加图片增广。下面定义随机剪裁函数，其对样本图片和标注使用同样的剪裁。

```{.python .input  n=8}
def voc_rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_images[0], train_labels[0], 200, 300)
gb.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### 数据读取

下面我们定义Gluon可以使用的数据集类，它可以返回任意的第$i$个样本图片和标号。除了随机剪裁外，这里我们将样本图片进行了归一化，同时过滤了小于剪裁尺寸的图片。

```{.python .input  n=9}
class VOCSegDataset(gdata.Dataset):
    def __init__(self, train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size        
        data, label = read_voc_images(root=voc_dir, train=train)
        self.data = [self.normalize_image(im) for im in self.filter(data)]
        self.label = self.filter(label)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.data)) + ' examples')
        
    def normalize_image(self, data):
        return (data.astype('float32') / 255 - self.rgb_mean) / self.rgb_std
    
    def filter(self, images):
        return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        data, label = voc_rand_crop(self.data[idx], self.label[idx],
                                    *self.crop_size)
        return (data.transpose((2, 0, 1)),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.data)
```

假设我们剪裁$320\times 480$图片来进行训练，我们可以查看训练和测试各保留了多少图片。

```{.python .input  n=10}
output_shape = (320, 480)  # 高和宽。
voc_train = VOCSegDataset(True, output_shape, voc_dir, colormap2label)
voc_test = VOCSegDataset(False, output_shape, voc_dir, colormap2label)
```

最后定义批量读取，这里使用4个进程来加速读取（暂不支持Windows操作系统）。

```{.python .input  n=11}
batch_size = 64
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = gdata.DataLoader(voc_train, batch_size, shuffle=True,
                              last_batch='discard', num_workers=num_workers)
test_iter = gdata.DataLoader(voc_test, batch_size, last_batch='discard',
                             num_workers=num_workers)
```

打印第一个批量可以看到，不同于图片分类和物体识别，这里的标签是一个三维的数组。

```{.python .input  n=12}
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

我们将本节中描述的`voc_classes`和`voc_colormap`变量、`download_voc_pascal`、`read_voc_images`、`voc_label_indices`和`voc_rand_crop`函数，以及`VOCSegDataset`类定义在`gluonbook`包中供后面章节使用。

## 小结

* 语义分割识别图片中每个像素的类别。
* 语义分割数据集中的标注由于跟跟样本图片有像素级的对应，使得其对应用图片变形类增广时有一定的限制。

## 练习

* 对比下有哪些图片分类上使用的图片增广比较难在语义分割上使用。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7218)

![](../img/qr_semantic-segmentation-and-dataset.svg)
