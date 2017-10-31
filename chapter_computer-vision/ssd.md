# 物体检测

Object Detection Using Convolutional Neural Networks

当我们讨论对图片进行预测时，到目前为止我们都是谈论分类。我们问过这个数字是0到9之间的哪一个，这个图片是鞋子还是衬衫。相对于图片分类，**物体检测**更有挑战性。我们不仅是要分析图片里有什么，而且需要识别它在什么位置。我们使用在[机器学习简介](../chapter_crashcourse/introduction.md)那章讨论过的图片作为样例，并对它标上主要物体和位置。

![](../img/catdog_label.svg)

可以看出物体检测跟图片分类有几个不同点：

1. 图片分类器通常只需要输出对图片中的主物体的分类。但物体检测必须能够识别多个物体，即使有些物体可能在图片中不是占主要版面。严格上来说，这个任务一般叫**多类物体检测**，但绝大部分研究都是针对多类的设置，所以我们这里为了简单去掉了”多类“
1. 图片分类器只需要输出将图片物体识别成某类的概率，但物体检测不仅需要输出识别概率，还需要识别物体在图片中的位置。这个通常是一个括住这个物体的方框,通常也被称之为**边界框**（bounding box）。

同时也看到物体检测跟图片分类有诸多类似之处。这类算法通常的一个做法是先预先提出一些边界框，然后用图片分类器（通常是卷积神经网络）来判断这里面是不是包含了感兴趣的物体，还是就只是背景。

这一章我们介绍物体检测问题中的一个流行模型：单发多框检测器(single shot multiple box object detector)，它通常被简称为SSD。这个方法最早在[这篇论文](https://arxiv.org/abs/1512.02325)提出。它之所以被称之为**单发**是因为它不同于之前提出是R-CNN系列算法其将提出边界框和预测做成两部，SSD一次将两件事情做完，从而获得性能上的提升。


## SSD:  单发多框检测器

SSD overview TODO

The SSD model predicts anchor boxes at multiple scales. The model architecture is illustrated in the following figure.

[image]

We first use a `body` network to extract the image features,
which are used as the input to the first scale (scale 0). The class labels and the corresponding anchor boxes
are predicted by `class_predictor` and `box_predictor`, respectively.
We then downsample the representations to the next scale (scale 1).
Again, at this new resolution, we predict both classes and anchor boxes.
This downsampling and predicting routine
can be repeated in multiple times to obtain results on multiple resolution scales.
Let's walk through the components one by one in a bit more detail.

### 锚框：默认的边界框

边界框可以出现在图片中的任何位置，并且可以有任何大小。为了简化计算，SSD跟Faster R-CNN一样使用一些默认的边界框，被称之为锚框（anchor box），做为搜索起点。具体来说，对输入的每个像素，以其为中心采样数个有不同形状和不同比例的边界框。假设输入大小是 $w \times h$，

- 给定大小 $s\in (0,1]$，那么生成的边界框形状是 $ws \times hs$
- 给定比例 $r > 0$，那么生成的边界框形状是 $w\sqrt{r} \times \frac{h}{\sqrt{r}}$

在采样的时候我们提供 $n$ 个大小（`sizes`）和 $m$ 个比例（`ratios`）。为了计算简单这里不生成$nm$个锚框，而是$n+m-1$个。其中第 $i$ 个锚框使用

- `sizes[i]`和`ratios[0]` 如果 $i\le n$
- `sizes[0]`和`ratios[i-n]` 如果 $i>n$

我们可以使用`contribe.ndarray`里的`MultiBoxPrior`来采样锚框。这里锚框通过左下角和右上角两个点来确定，而且被标准化成了区间$[0,1]$的实数。

```{.python .input  n=1}
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior

# shape: batch x channel x height x weight
n = 40
x = nd.random.uniform(shape=(1, 3, n, n))

y = MultiBoxPrior(x, sizes=[.5,.25,.1], ratios=[1,2,.5])

boxes = y.reshape((n, n, -1, 4))
print(boxes.shape)
# The first anchor box centered on (20, 20)
# its format is (x_min, y_min, x_max, y_max)
boxes[20, 20, 0, :]
```

我们可以画出以`(20,20)`为中心的所有锚框：

```{.python .input  n=2}
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)
colors = ['blue', 'green', 'red', 'black', 'magenta']
plt.imshow(nd.ones((n, n, 3)).asnumpy())
anchors = boxes[20, 20, :, :]
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
plt.show()
```

### 预测物体类别

对每一个锚框我们需要预测它是不是包含了我们感兴趣的物体，还是只是背景。这里我们使用一个$3\times 3$的卷积层来做预测，加上`pad=1`使用它的输出和输入一样。同时输出的通道数是`num_anchors*(num_classes+1)`，每个通道对应一个锚框对某个类的置信度。假设输出是`Y`，那么对应输入中第$n$个样本的第$(i,j)$像素的置信值是在`Y[n,:,i,j]`里。具体来说，对于以`(i,j)`为中心的第`a`个锚框，

- 通道 `a*(num_class+1)` 是其只包含背景的分数
- 通道 `a*(num_class+1)+1+b` 是其包含第`b`个物体的分数

我们定义个一个这样的类别分类器函数：

```{.python .input  n=3}
from mxnet.gluon import nn
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

cls_pred = class_predictor(5, 10)
cls_pred.initialize()
X = nd.zeros((2, 3, 20, 20))
Y = cls_pred(X)
Y.shape
```

### 预测边界框

因为真实的边界框可以是任意形状，我们需要预测如何从一个锚框变换成真正的边界框。

这里的目标是
The goal is predict how to transfer the current anchor box to the correct box. That is, assume $b$ is one of the sampled default box, while $Y$ is the ground truth, then we want to predict the delta positions $\Delta(Y, b)$, which is a 4-length vector.

More specifically, the we define the delta vector as:
[$t_x$, $t_y$, $t_{width}$, $t_{height}$], where

- $t_x = (Y_x - b_x) / b_{width}$
- $t_y = (Y_y - b_y) / b_{height}$
- $t_{width} = (Y_{width} - b_{width}) / b_{width}$
- $t_{height} = (Y_{height} - b_{height}) / b_{height}$

Normalizing the deltas with box width/height tends to result in better convergence behavior.

Similar to classes, we use a convolution layer here. The only difference is that the output channel size is now `num_anchors * 4`, with the predicted delta positions for the *i*-th box stored from channel `i*4` to `i*4+3`.

```{.python .input  n=4}
def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

box_pred = box_predictor(10)
box_pred.initialize()
X = nd.zeros((2, 3, 20, 20))
Y = box_pred(X)
Y.shape
```

### 减半模块

我们定义一个卷积块，它由两个`Conv-BatchNorm-Relu`组成，我们使用`padding=1`的$3\times 3$卷积使得输入和输入有同样的长宽，然后再通过`strides=2`的最大池化层将长宽减半。

```{.python .input  n=5}
def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

blk = down_sample(10)
blk.initialize()
X = nd.zeros((2, 3, 20, 20))
Y = blk(X)
Y.shape
```

### 合并来自不同层的预测输出

前面我们提到过SSD的一个重要性质是它会在多个层同时做预测。每个层由于长宽和锚框选择不一样，导致输出的数据形状会不一样。这里我们用物体类别预测作为样例，边框预测是类似的。

我们首先创建一个特定大小的输入，然后对它输出类别预测。然后对输入减半，再输出类别预测。

```{.python .input  n=6}
X = nd.zeros((2, 8, 20, 20))
print('X:', X.shape)

cls_pred1 = class_predictor(5, 10)
cls_pred1.initialize()
Y1 = cls_pred1(X)
print('Class prediction 1:', Y1.shape)

ds = down_sample(16)
ds.initialize()
X = ds(X)
print('X:', X.shape)

cls_pred2 = class_predictor(3, 10)
cls_pred2.initialize()
Y2 = cls_pred2(X)
print('Class prediction 1:', Y2.shape)
```

可以看到`Y1`和`Y2`形状完全不同。为了之后处理简单，我们将不同层的输入合并成一个输出。首先我们将通道移到最后的维度，然后将其展成2D数组。因为第一个维度是样本个数，所以不同输出之间是不变，我们可以将所有输出在第二个维度上拼接起来。

```{.python .input  n=7}
def flatten_prediction(pred):
    return pred.transpose(axes=(0,2,3,1)).flatten()

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)

flat_y1 = flatten_prediction(Y1)
print('Flatten class prediction 1', flat_y1.shape)
flat_y2 = flatten_prediction(Y2)
print('Flatten class prediction 2', flat_y2.shape)
y = concat_predictions([flat_y1, flat_y2])
print('Concat class predictions', y.shape)
```

### 主体网络

主体网络用来从原始像素抽取特征。通常前面介绍的用来图片分类的卷积神经网络，例如ResNet，都可以用来作为主体网络。这里为了示范，我们简单叠加几个减半模块作为主体网络。


```{.python .input  n=8}
def body():
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out

bnet = body()
bnet.initialize()
X = nd.random.uniform(shape=(2,3,256,256))
Y = bnet(X)
Y.shape
```

### 创建一个玩具SSD模型

现在我们可以创建一个玩具SSD模型了。我们称之为玩具是因为这个网络不管是层数还是锚框个数都比较小，仅仅适合之后我们之后使用的一个小数据集。但这个模型不会影响我们介绍SSD。

这个网络包含四块。主体网络，三个减半模块，以及五个物体类别和边框预测模块。其中预测分别应用在在主体网络输出，减半模块输入，和最后的全局池化层上。

```{.python .input  n=9}
def toy_ssd_model(num_anchors, num_classes):
    downsamplers = nn.Sequential()
    for _ in range(3):
        downsamplers.add(down_sample(128))
        
    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()    
    for _ in range(5):
        class_predictors.add(class_predictor(num_anchors, num_classes))
        box_predictors.add(box_predictor(num_anchors))

    model = nn.Sequential()
    model.add(body(), downsamplers, class_predictors, box_predictors)
    return model
```

### 计算预测

给定模型和每层预测输出使用的锚框大小和形状，我们可以定义前向函数。

```{.python .input  n=10}
def toy_ssd_forward(X, model, sizes, ratios, verbose=False):    
    body, downsamplers, class_predictors, box_predictors = model
    anchors, class_preds, box_preds = [], [], []
    # feature extraction    
    X = body(X)
    for i in range(5):
        # predict
        anchors.append(MultiBoxPrior(
            X, sizes=sizes[i], ratios=ratios[i]))
        class_preds.append(
            flatten_prediction(class_predictors[i](X)))
        box_preds.append(
            flatten_prediction(box_predictors[i](X)))
        if verbose:
            print('Predict scale', i, X.shape, 'with', 
                  anchors[-1].shape[1], 'anchors')
        # down sample
        if i < 3:
            X = downsamplers[i](X)
        elif i == 3:
            X = nd.Pooling(
                X, global_pool=True, pool_type='max', 
                kernel=(X.shape[2], X.shape[3]))
    return (concat_predictions(anchors),
            concat_predictions(class_preds),
            concat_predictions(box_preds))
#    return anchors, class_preds, box_preds
```

### Put all things together

```{.python .input  n=11}
from mxnet import gluon
class ToySSD(gluon.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes and ratios for 5 feature scales
        self.sizes = [[.2,.272], [.37,.447], [.54,.619], 
                      [.71,.79], [.88,.961]]
        self.ratios = [[1,2,.5]]*5
        self.num_classes = num_classes
        self.verbose = verbose
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        # use name_scope to guard the names
        with self.name_scope():
            self.model = toy_ssd_model(num_anchors, num_classes)

    def forward(self, X):
        anchors, class_preds, box_preds = toy_ssd_forward(
            X, self.model, self.sizes, self.ratios, 
            verbose=self.verbose)
        # concat results
        # it is better to have class predictions reshaped for softmax computation       
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))
        return anchors, class_preds, box_preds
```

### Outputs of ToySSD

```{.python .input  n=12}
net = ToySSD(num_classes=2, verbose=True)
net.initialize()
X = nd.zeros((1, 3, 256, 256))
anchors, class_preds, box_preds = net(X)
print('Output achors:', anchors.shape)
print('Output class predictions:', class_preds.shape)
print('Output box predictions:', box_preds.shape)
```

## Dataset

For demonstration purposes, we'll build a train our model to detect Pikachu in the wild.
We generated a a synthetic toy dataset by rendering images from open-sourced 3D Pikachu models.
The dataset consists of 1000 pikachus with random pose/scale/position in random background images.
The exact locations are recorded as ground-truth for training and validation.

![](https://user-images.githubusercontent.com/3307514/29479494-5dc28a02-8427-11e7-91d0-2849b88c17cd.png)


### Download dataset

```{.python .input  n=13}
from mxnet.test_utils import download
import os.path as osp
def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        print('Found hash mismatch in file {}, possibly due to incomplete download.'.format(file_path))
    return matched

url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/{}'
hashes = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in hashes.items():
    fname = 'pikachu_' + k
    target = osp.join('data', fname)
    url = url_format.format(k)
    if not osp.exists(target) or not verified(target, v):
        print('Downloading', target, url)
        download(url, fname=fname, dirname='data', overwrite=True)
```

### Load dataset

```{.python .input  n=14}
import mxnet.image as image
data_shape = 256
batch_size = 32
def get_iterators(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_train.rec',
        path_imgidx='./data/pikachu_train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class

train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
batch = train_data.next()
print(batch)
```

### Illustration

Let's display one image loaded by ImageDetIter.

```{.python .input  n=15}
import numpy as np

img = batch.data[0][0].asnumpy()  # grab the first image, convert to numpy array
img = img.transpose((1, 2, 0))  # we want channel to be the last dimension
img += np.array([123, 117, 104])
img = img.astype(np.uint8)  # use uint8 (0-255)
# draw bounding boxes on image
for label in batch.label[0][0].asnumpy():
    if label[0] < 0:
        break
    print(label)
    xmin, ymin, xmax, ymax = [int(x * data_shape) for x in label[1:5]]
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=(1, 0, 0), linewidth=3)
    plt.gca().add_patch(rect)
plt.imshow(img)
plt.show()
```

## Train

### Losses

Network predictions will be penalized for incorrect class predictions and wrong box deltas.

```{.python .input  n=16}
from mxnet.contrib.ndarray import MultiBoxTarget
def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(*[default_anchors, labels, class_predicts])
    box_target = z[0]  # box offset target for (x, y, width, height)
    box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
    cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
    return box_target, box_mask, cls_target
```

Pre-defined losses are provided in `gluon.loss` package, however, we can define losses manually.

First, we need a Focal Loss for class predictions.

```{.python .input  n=17}
class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pt = F.pick(output, label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

# cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
cls_loss = FocalLoss()
print(cls_loss)
```

Next, we need a SmoothL1Loss for box predictions.

```{.python .input  n=18}
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)

box_loss = SmoothL1Loss()
print(box_loss)
```

### Evaluation metrics

Here, we define two metrics that we'll use to evaluate our performance whien training.
You're already familiar with accuracy unless you've been naughty and skipped straight to object detection.
We use the accuracy metric to assess the quality of the class predictions.
Mean absolute error (MAE) is just the L1 distance, introduced in our [linear algebra chapter](../chapter01_crashcourse/linear-algebra.ipynb).
We use this to determine how close the coordinates of the predicted bounding boxes are to the ground-truth coordinates.
Because we are jointly solving both a classification problem and a regression problem, we need an appropriate metric for each task.

```{.python .input  n=19}
import mxnet as mx
cls_metric = mx.metric.Accuracy()
box_metric = mx.metric.MAE()  # measure absolute difference between prediction and target
```

```{.python .input  n=20}
### Set context for training
ctx = mx.gpu()  # it may takes too long to train using CPU
try:
    _ = nd.zeros(1, ctx=ctx)
    # pad label for cuda implementation
    train_data.reshape(label_shape=(3, 5))
    train_data = test_data.sync_label_shape(train_data)
except mx.base.MXNetError as err:
    print('No GPU enabled, fall back to CPU, sit back and be patient...')
    ctx = mx.cpu()
```

### Initialize parameters

```{.python .input  n=21}
net = ToySSD(num_class)
net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
```

### Set up trainer

```{.python .input  n=22}
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})
```

### Start training

Optionally we load pretrained model for demonstration purpose. One can set `from_scratch = True` to training from scratch, which may take more than 30 mins to finish using a single capable GPU.

```{.python .input  n=23}
epochs = 150  # set larger to get better performance
log_interval = 20
from_scratch = False  # set to True to train from scratch
if from_scratch:
    start_epoch = 0
else:
    start_epoch = 148
    pretrained = 'ssd_pretrained.params'
    sha1 = 'fbb7d872d76355fff1790d864c2238decdb452bc'
    url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/ssd_pikachu-fbb7d872.params'
    if not osp.exists(pretrained) or not verified(pretrained, sha1):
        print('Downloading', pretrained, url)
        download(url, fname=pretrained, overwrite=True)
    net.load_params(pretrained, ctx)
```

```{.python .input  n=24}
import time
from mxnet import autograd as ag
for epoch in range(start_epoch, epochs):
    # reset iterator and tick
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(train_data):
        btic = time.time()
        # record gradients
        with ag.record():
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            default_anchors, class_predictions, box_predictions = net(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
            # losses
            loss1 = cls_loss(class_predictions, cls_target)
            loss2 = box_loss(box_predictions, box_target, box_mask)
            # sum all losses
            loss = loss1 + loss2
            # backpropagate
            loss.backward()
        # apply
        trainer.step(batch_size)
        # update metrics
        cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
        box_metric.update([box_target], [box_predictions * box_mask])
        if (i + 1) % log_interval == 0:
            name1, val1 = cls_metric.get()
            name2, val2 = box_metric.get()
            print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                  %(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))

    # end of epoch logging
    name1, val1 = cls_metric.get()
    name2, val2 = box_metric.get()
    print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

# we can save the trained parameters to disk
net.save_params('ssd_%d.params' % epochs)
```

## Test

Testing is similar to training, except that we don't need to compute gradients and training targets. Instead, we take the predictions from network output, and combine them to get the real detection output.

### Prepare the test data

```{.python .input  n=25}
import numpy as np
def preprocess(im):
    """Takes an image and apply preprocess"""
    # resize to data_shape
    im = image.imresize(im, data_shape, data_shape)
    # swap BGR to RGB
    # im = im[:, :, (2, 1, 0)]
    # convert to float before subtracting mean
    im = im.astype('float32')
    # subtract mean
    im -= nd.array([123, 117, 104])
    # organize as [batch-channel-height-width]
    im = im.transpose((2, 0, 1))
    im = im.expand_dims(axis=0)
    return im

with open('../img/pikachu.jpg', 'rb') as f:
    im = image.imdecode(f.read())
x = preprocess(im)
print('x', x.shape)
```

### Network inference

In a single line of code!

```{.python .input  n=26}
# if pre-trained model is provided, we can load it
# net.load_params('ssd_%d.params' % epochs, ctx)
anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
print('anchors', anchors)
print('class predictions', cls_preds)
print('box delta predictions', box_preds)
```

### Convert predictions to real object detection results

```{.python .input  n=27}
from mxnet.contrib.ndarray import MultiBoxDetection
# convert predictions to probabilities using softmax
cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
# apply shifts to anchors boxes, non-maximum-suppression, etc...
output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
print(output)
```

Each row in the output corresponds to a detection box, as in format [class_id, confidence, xmin, ymin, xmax, ymax].

Most of the detection results are -1, indicating that they either have very small confidence scores, or been suppressed through non-maximum-suppression.

### Display results

```{.python .input  n=28}
def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img.asnumpy())
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()

display(im, output[0].asnumpy(), thresh=0.45)
```

## Conclusion

Detection is harder than classification, since we want not only class probabilities, but also localizations of different objects including potential small objects. Using sliding window together with a good classifier might be an option, however, we have shown that with a properly designed convolutional neural network, we can do single shot detection which is blazing fast and accurate!

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
