# 单次多盒检测
:label:`sec_ssd`

在 :numref:`sec_bbox`—:numref:`sec_object-detection-dataset` 中，我们引入了边界框、锚框、多尺度物体检测和用于对象检测的数据集。现在我们已经准备好使用这样的背景知识来设计物体检测模型：单次多盒检测 (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`。该模型简单、快速且广泛使用。尽管这只是大量目标检测模型中的一个，但本节中的一些设计原则和实现细节也适用于其他模型。 

## 模型

:numref:`fig_ssd` 概述了单次多盒检测的设计。此模型主要由基础网络组成，然后是几个多尺度要素地图块。基本网络用于从输入图像中提取要素，因此它可以使用深度 CNN。例如，原始的单次多盒检测论文采用了在分类层 :cite:`Liu.Anguelov.Erhan.ea.2016` 之前被截断的 VGG 网络，而 ResNet 也被普遍使用。通过我们的设计，我们可以使基本网络输出更大的要素地图，从而生成更多的锚框来检测较小的物体。随后，每个多比例要素地图块都会减少（例如，减少一半）要素地图的高度和宽度，并使要素地图的每个单元能够增加其在输入图像上的接受范围。 

回想一下 :numref:`sec_multiscale-object-detection` 通过深度神经网络对图像进行分层表示的多尺度物体检测的设计。由于接近 :numref:`fig_ssd` 顶部的多比例要素地图较小但具有较大的接收场，因此它们适合检测较少但较大的物体。 

简而言之，通过其基本网络和几个多尺度要素地图块，单击多盒检测生成不同大小的锚框，并通过预测这些锚框的类和偏移量（因此边界框）来检测变化大小的对象；因此，这是一个多尺度对象检测模型。 

![As a multiscale object detection model, single-shot multibox detection mainly consists of a base network followed by several multiscale feature map blocks.](../img/ssd.svg)
:label:`fig_ssd`

在下面，我们将介绍 :numref:`fig_ssd` 中不同模块的实施细节。首先，我们将讨论如何实施班级和边界框预测。 

### 班级预测层

让对象类的数量为 $q$。然后锚盒有 $q+1$ 个类，其中 0 类是背景。在某种规模上，假设要素地图的高度和宽度分别为 $h$ 和 $w$。当生成 $a$ 个锚点框以这些要素地图的每个空间位置为中心时，总共需要对 $hwa$ 个锚框进行分类。由于参数化成本可能很高，这通常会导致使用完全连接的图层进行分类不可行。回想一下我们是如何使用卷积层通道来预测 :numref:`sec_nin` 中的类别。单次多盒检测使用相同的技术来降低模型复杂性。 

具体来说，类预测图层使用卷积图层而不改变要素地图的宽度或高度。这样，输出和输入之间可以在要素地图的相同空间维度（宽度和高度）下进行一对一的对应关系。更具体地说，任何空间位置（$x$、$y$）的输出要素地图的通道表示以输入要素地图（$x$、$y$）为中心的所有锚点框的类预测。要生成有效的预测，必须有 $a(q+1)$ 个输出通道，其中对于相同的空间位置，索引 $i(q+1) + j$ 的输出通道表示 $j$ ($0 \leq j \leq q$) 对定位盒 $i$ ($0 \leq j \leq q$) 的预测。 

下面我们定义了这样一个类预测层，分别通过参数 `num_anchors` 和 `num_classes` 指定 $q$ 和 `num_classes`。该图层使用填充为 1 的 $3\times3$ 卷积图层。此卷积图层的输入和输出的宽度和高度保持不变。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### 边界框预测层

边界框预测图层的设计类似于类预测图层的设计。唯一的区别在于每个锚盒的输出数量：在这里我们需要预测四个偏移，而不是 $q+1$ 类。

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### 连接多个尺度的预测

正如我们所提到的，单次多盒检测使用多尺度要素地图来生成锚框并预测其类别和偏移量。在不同的比例下，要素地图的形状或以同一单元为中心的锚框的数量可能会有所不同。因此，不同尺度的预测输出的形状可能会有所不同。 

在以下示例中，我们为同一个微型批次的两个不同比例（`Y1` 和 `Y2`）构建要素地图，其中 `Y2` 的高度和宽度是 `Y1` 的一半。让我们以课堂预测为例。假设分别在 `Y1` 和 `Y2` 中为每个单元生成 5 个和 3 个锚箱。进一步假设对象类的数量为 10。对于功能地图 `Y1` 和 `Y2`，类预测输出中的通道数分别为 $5\times(10+1)=55$ 和 $3\times(10+1)=33$，其中任一输出形状是（批量大小、通道数、高度、宽度）。

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

正如我们所看到的，除了批量尺寸维度外，其他三个维度都有不同的尺寸。为了将这两个预测输出连接起来以提高计算效率，我们将将这些张量转换为更一致的格式。 

请注意，通道维度保存中心相同的锚点框的预测。我们首先将这个维度移到最内层。由于不同尺度的批量大小保持不变，因此我们可以将预测输出转换为具有形状的二维张量（批量大小，高度 $\times$ 宽度 $\times$ 通道数）。然后我们可以沿维度 1 以不同的比例连接此类输出。

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

这样，尽管 `Y1` 和 `Y2` 在通道、高度和宽度方面具有不同的大小，但我们仍然可以在同一个微型批次的两个不同尺度上连接这两个预测输出。

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### 降采样模块

为了在多个比例下检测物体，我们定义了以下降采样模块 `down_sample_blk`，该模块将输入要素贴图的高度和宽度减半。事实上，该模块在 :numref:`subsec_vgg-blocks` 中应用了 VGG 模块的设计。更具体地说，每个缩减采样模块由两个 $3\times3$ 卷积层组成，填充为 1，其次是步幅为 2 的 $2\times2$ 最大池层。正如我们所知，填充为 1 的 $3\times3$ 卷积图层不会改变要素地图的形状。但是，随后的 $2\times2$ 最大池将输入要素地图的高度和宽度减少了一半。对于此降采样模块的输入和输出特征映射，因为 $1\times 2+(3-1)+(3-1)=6$，输出中的每个单元在输入上都有一个 $6\times6$ 接收场。因此，降采样模块会扩大每个单元在其输出特征图中的接受场。

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

在以下示例中，我们构建的降采样模块会更改输入通道的数量，并将输入要素映射的高度和宽度减半。

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### 基本网络块

基本网络块用于从输入图像中提取要素。为简单起见，我们构建了一个由三个降采样模块组成的小型基础网络，使每个模块的通道数量增加一倍。给定 $256\times256$ 输入图像，此基本网络模块输出 $32 \times 32$ 要素地图 ($256/2^3=32$)。

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### 完整模型

完整的单击多盒检测模型由五个模块组成。每个区块生成的特征图既用于 (i) 生成锚框，也可用于 (ii) 预测这些锚框的类和偏移量。在这五个区块中，第一个是基本网块，第二个到第四个是降采样块，最后一个区块使用全局最大池将高度和宽度都减少到 1。从技术上讲，第二到第五个区块都是 :numref:`fig_ssd` 中的多比例要素地图块。

```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

现在我们为每个模块定义向前传播。与图像分类任务不同，此处的输出包括：(i) CNN 功能地图 `Y`，(ii) 在当前比例下使用 `Y` 生成的锚盒，以及 (iii) 这些锚盒预测的类和偏移量（基于 `Y`）。

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

回想一下，在 :numref:`fig_ssd` 中，一个更接近顶部的多比例要素地图块是用于检测较大的物体；因此，它需要生成更大的锚框。在上面的正向传播中，在每个多尺度要素映射块中，我们通过调用的 `multibox_prior` 函数的 `sizes` 参数传递两个比例值的列表（见 :numref:`sec_anchor`）。在以下内容中，0.2 和 1.05 之间的间隔被均匀分成五个部分，以确定五个区块的较小比例值：0.2、0.37、0.54、0.71 和 0.88。然后，他们的较大尺度值由 $\sqrt{0.2 \times 0.37} = 0.272$、$\sqrt{0.37 \times 0.54} = 0.447$ 等给出。

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

现在我们可以按如下方式定义完整的模型 `TinySSD`。

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

我们创建了一个模型实例，然后使用它在 $256 \times 256$ 图像 `X` 的小批量上执行前向传播。 

如本节前面所示，第一个模块输出 $32 \times 32$ 功能地图。回想一下，第二到第四个缩减采样模块的高度和宽度减半，第五个模块使用全局池。由于沿要素地图的空间维度为每个单位生成 4 个锚点框，因此在所有五个比例下，每个图像总共生成 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ 个锚点框。

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 培训

现在我们将解释如何训练用于目标检测的单次多盒检测模型。 

### 阅读数据集并初始化模型

首先，让我们阅读 :numref:`sec_object-detection-dataset` 中描述的香蕉检测数据集。

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

香蕉检测数据集中只有一个类。定义模型后，我们需要初始化其参数并定义优化算法。

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### 定义损失和评估函数

对象检测有两种类型的损失。第一个损失涉及锚盒的类别：它的计算可以简单地重复使用我们用于图像分类的交叉熵损失函数。第二个损失涉及正面（非背景）锚框的抵消：这是一个回归问题。但是，对于这个回归问题，我们在这里不使用 :numref:`subsec_normal_distribution_and_squared_loss` 中描述的平方损失。相反，我们使用 $L_1$ 标准损失，即预测和基本真相差值的绝对值。掩码变量 `bbox_masks` 过滤掉损失计算中的负锚框和非法（填充的）锚框。最后，我们总结了锚箱类损失和锚箱偏移损耗，以获得模型的损失函数。

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

我们可以使用准确性来评估分类结果。由于偏移使用了 $L_1$ 标准损失，我们使用 * 平均绝对误差 * 来评估预测的边界框。这些预测结果是从生成的锚框和它们的预测偏移量中获得的。

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### 训练模型

在训练模型时，我们需要生成多尺度锚盒 (`anchors`)，并预测前传播中的类别 (`cls_preds`) 和偏移 (`bbox_preds`)。然后，我们根据标签信息 `Y` 标记此类生成的锚盒的类别（`cls_labels`）和偏移量（`bbox_labels`）。最后，我们使用类和偏移量的预测值和标记值来计算损失函数。对于简洁的实现，此处省略了测试数据集的评估。

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## 预测

在预测期间，目标是检测图像上的所有感兴趣对象。下面我们阅读并调整测试图像的大小，将其转换为卷积层所需的四维张量。

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

使用下面的 `multibox_detection` 函数，可从锚点框及其预测的偏移量中获得预测的边界框。然后使用非最大抑制来删除类似的预测边界框。

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

最后，我们以 0.9 或更高的信心显示所有预测的边界框作为输出。

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## 摘要

* 单镜多盒检测是一种多尺度目标检测模型。通过其基本网络和多个多尺度要素地图块，单击多盒检测可生成不同大小的锚框，并通过预测这些锚框的类和偏移量（即边界框）来检测变化大小的对象。
* 训练单次多盒检测模型时，损失函数是根据锚箱类和偏移量的预测值和标记值计算的。

## 练习

1. 你能通过改进损失功能来改善单次多盒检测吗？例如，用平滑 $L_1$ 标准损失替换 $L_1$ 标准损失，以预测的抵消量。此损耗函数使用大约零的方形函数来获得平滑度，该函数由超参数 $\sigma$ 控制：

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

当 $\sigma$ 非常大时，这种损失类似于 $L_1$ 标准损失。当它的值较小时，损失函数会更平滑。

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

此外，在实验中，我们使用交叉熵损失进行类预测：用 $p_j$ 表示地面真相类 $j$ 的预测概率，交叉熵损失为 $-\log p_j$。我们还可以使用焦点损失 :cite:`Lin.Goyal.Girshick.ea.2017`：给定超参数 $\gamma > 0$ 和 $\alpha > 0$，此损失被定义为： 

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

正如我们所看到的，增加 $\gamma$ 可以有效地减少分类良好的例子（例如 $p_j > 0.5$）的相对损失，因此培训可以更多地关注那些错误分类的困难示例。

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 由于空间限制，我们在本节中省略了单一拍摄多盒检测模型的一些实现细节。你能否在以下几个方面进一步改进模型：
    1. 当对象比图像小得多时，模型可以调整输入图像的大小。
    1. 通常有大量的负面锚箱。为了使课堂分布更加平衡，我们可以减少负锚框的样本。
    1. 在损失函数中，将不同的重量超参数分配给类损失和偏移损失。
    1. 使用其他方法评估物体检测模型，例如单次拍摄多盒检测论文 :cite:`Liu.Anguelov.Erhan.ea.2016` 中的方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
