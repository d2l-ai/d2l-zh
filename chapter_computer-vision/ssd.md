# 单发多框检测（SSD）

我们在前几节分别介绍了边界框、锚框、多尺度目标检测和数据集，下面我们基于这些背景知识来构造一个目标检测模型：单发多框检测（single shot multibox detection，简称SSD）[1]。它简单、快速，并得到了广泛使用。该模型的一些设计思想和实现细节常适用于其他目标检测模型。


## 模型

图9.4描述了SSD模型的设计。它主要由一个基础网络块和若干个多尺度特征块串联而成。其中基础网络块可以是一个卷积神经网络，例如在分类层之前截断的VGG。我们可以设计基础网络，使它输出的高和宽较大。这样一来，基于该特征图生成的锚框数量较多，可以用来检测尺寸较小的目标。接下来的每个多尺度特征块将上一层提供的特征图的高和宽缩小（例如减半），并使特征图中每个单元在输入图像上的感受野变得更广阔。如此一来，图9.4中越靠近顶部的多尺度特征块输出的特征图越小，基于特征图生成的锚框故而也越少，加之特征图中每个单元感受野越大，因此更适合检测尺寸较大的目标。由于SSD基于基础网络块和各个多尺度特征块生成不同数量和不同大小的锚框，并通过预测锚框的类别和偏移量（即预测边界框）检测不同大小的目标，因此SSD是一个多尺度的目标检测模型。

![SSD模型主要由一个基础网络块和若干多尺度特征块串联而成。](../img/ssd.svg)


接下来我们介绍如何实现图中的各个模块。我们先介绍如何实现类别预测和边界框预测。

### 类别预测层

设目标的类别个数为$q$。每个锚框的类别个数将是$q+1$，其中类别0表示锚框只包含背景。在某个尺度下，设特征图的高和宽分别为$h$和$w$，如果以其中每个单元为中心生成$a$个锚框，那么我们需要对$hwa$个锚框进行分类。如果使用全连接层作为输出，很容易导致模型参数过多。回忆[“网络中的网络（NiN）”](../chapter_convolutional-neural-networks/nin.md)一节介绍的使用卷积层的通道来输出类别预测的方法。SSD采用同样的方法来降低模型复杂度。

具体来说，类别预测层使用一个保持输入高和宽的卷积层。这样一来，输出和输入在特征图宽和高上的空间坐标一一对应。考虑输出和输入同一空间坐标$(x,y)$：输出特征图上$(x,y)$坐标的通道里包含了以输入特征图$(x,y)$坐标为中心生成的所有锚框的类别预测。因此输出通道数为$a(q+1)$，其中索引为$i(q+1) + j$（$0 \leq j \leq q$）的通道代表了索引为$i$的锚框的类别索引为$j$的预测。

下面我们定义一个这样的类别预测层。指定参数$a$和$q$后，它使用一个填充为1的$3\times3$卷积层。该卷积层的输入和输出的高和宽保持不变。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

### 边界框预测层

边界框预测层的设计与类别预测层的设计类似。唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是$q+1$个类别。

```{.python .input  n=2}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

### 连结多尺度的预测

我们提到，SSD根据多个尺度下的特征图生成锚框并预测类别和偏移量。由于每个尺度上特征图的形状或以同一单元为中心生成的锚框个数可能不同，不同尺度的预测输出可能形状不同。

在下面的例子中，我们对同一批量数据构造两个不同尺度下的特征图`Y1`和`Y2`，其中`Y2`相对于`Y1`来说高和宽分别减半。以类别预测为例，假设以`Y1`和`Y2`特征图中每个单元生成的锚框个数分别是5和3，当目标类别个数为10时，类别预测输出的通道数分别为$5\times(10+1)=55$和$3\times(10+1)=33$。预测输出的格式为（批量大小，通道数，高，宽）。可以看到，除了批量大小外，其他维度大小均不一样。我们需要将它们变形成统一的格式并将多尺度的预测连结，从而让后续计算更简单。

```{.python .input  n=3}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
(Y1.shape, Y2.shape)
```

通道维包含中心相同的锚框的预测结果。我们首先将通道维移到最后一维。因为不同尺度下批量大小仍保持不变，我们可以将预测结果转成二维的（批量大小，高$\times$宽$\times$通道数）的格式，以方便之后在维度1上的连结。

```{.python .input  n=4}
def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)
```

这样一来，尽管`Y1`和`Y2`形状不同，我们仍然可以将这两个同一批量不同尺度的预测结果连结在一起。

```{.python .input  n=6}
concat_preds([Y1, Y2]).shape
```

### 高和宽减半块

为了在多尺度检测目标，下面定义高和宽减半块`down_sample_blk`。它串联了两个填充为1的$3\times3$卷积层和步幅为2的$2\times2$最大池化层。我们知道，填充为1的$3\times3$卷积层不改变特征图的形状，而后面的池化层直接将特征图的高和宽减半。由于$1\times 2+(3-1)+(3-1)=6$，输出特征图中每个单元在输入特征图上的感受野形状为$6\times6$。可以看出，高和宽减半块使得输出特征图中每个单元的感受野变得更广阔。

```{.python .input  n=7}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

测试高和宽减半块的前向计算。可以看到，它改变了输入的通道数，并将高和宽减半。

```{.python .input  n=8}
forward(nd.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

### 基础网络块

基础网络块用来从原始图像抽取特征，一般会选择常用的深度卷积神经网络。SSD论文中选用了在分类层之前截断的VGG [1]，现在也常用ResNet替代。为了计算简洁，我们在这里构造一个小的基础网络。该网络串联三个高和宽减半块，并逐步将通道数翻倍。当输入的原始图像的形状为$256\times256$时，基础网络块输出的特征图的形状为$32 \times 32$。

```{.python .input  n=9}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(nd.zeros((2, 3, 256, 256)), base_net()).shape
```

### 完整的模型

我们已经介绍了SSD模型中的各个功能模块，现在我们将构建整个模型。这个模型有五个模块，每个模块对输入进行特征抽取，并且预测锚框的类和偏移。第一个模块使用主体网络，第二到四模块使用减半模块，最后一个模块则使用全局的最大池化层来将高宽降到1。下面函数定义如何构建这些模块。

```{.python .input  n=10}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

接下来我们定义每个模块如何进行前向计算。它跟之前的卷积神经网络不同在于，我们不仅输出卷积块的输出，而且还返回在输出上生成的锚框，以及每个锚框的类别预测和偏移预测。

```{.python .input  n=11}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchor = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_pred = cls_predictor(Y)
    bbox_pred = bbox_predictor(Y)
    return (Y, anchor, cls_pred, bbox_pred)
```

对每个模块我们要定义其输出上的锚框如何生成。比例固定成1、2和0.5，但大小上则不同，用于覆盖不同的尺度。

```{.python .input  n=12}
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

完整的模型定义如下。

```{.python .input  n=13}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors,
                                                      num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # 每个模块的锚框需要连结。
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape(
                    (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))


net = TinySSD(num_classes=1)
net.initialize()
X = nd.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class predictions:', cls_preds.shape)
print('output box predictions:', bbox_preds.shape)
```

## 训练

下面我们描述如何一步步训练SSD模型来进行目标检测。

### 读取数据和初始化训练

我们使用之前构造的皮卡丘数据集。

```{.python .input  n=14}
batch_size = 32
train_data, test_data = gb.load_data_pikachu(batch_size)
# GPU 实现里要求每张图像至少有三个边界框，我们加上两个标号为 -1 的边界框。
train_data.reshape(label_shape=(3, 5))
```

模型和训练器的初始化跟之前类似。

```{.python .input  n=15}
ctx, net = gb.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

### 损失和评估函数

目标检测有两个损失函数，一是对每个锚框的类别预测，我们可以重用之前图像分类问题里一直使用的Softmax和交叉熵损失。二是正类锚框的偏移预测。它是一个回归问题，但我们这里不使用前面介绍过的L2损失函数，而是使用惩罚相对更小的线性L1损失函数，即$l_1(\hat y, y) = |\hat y - y|$。

```{.python .input  n=16}
cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()
```

```{.python .input  n=17}
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

对于分类好坏我们可以沿用之前的分类精度。因为使用了L1损失，我们用平均绝对误差评估边框预测的性能。

```{.python .input  n=18}
def cls_metric(cls_preds, cls_labels):
    # 注意这里类别预测结果放在最后一维，argmax 的时候指定使用最后一维。
    return (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()

def bbox_metric(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().mean().asscalar()
```

### 训练模型

训练函数跟前面的不一样在于网络会有多个输出，而且有两个损失函数。为了代码简单起见我们没有评估测试数据集。

```{.python .input  n=19}
for epoch in range(20):
    acc, mae = 0, 0
    train_data.reset()  # 从头读取数据。
    start = time.time()
    for i, batch in enumerate(train_data):
        # 复制数据到 GPU。
        X = batch.data[0].as_in_context(ctx)
        Y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            # 对每个锚框预测输出。
            anchors, cls_preds, bbox_preds = net(X)
            # 对每个锚框生成标号。
            bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                anchors, Y, cls_preds.transpose((0, 2, 1)))
            # 计算类别预测和边界框预测损失。
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        # 计算梯度和更新模型。
        l.backward()
        trainer.step(batch_size)
        # 更新类别预测和边界框预测评估。
        acc += cls_metric(cls_preds, cls_labels)
        mae += bbox_metric(bbox_preds, bbox_labels, bbox_masks)
    if (epoch + 1) % 5 == 0:
        print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
            epoch + 1, 1 - acc / (i + 1), mae / (i + 1), time.time() - start))
```

## 预测

在预测阶段，我们希望能把图像里面所有感兴趣的目标找出来。我们首先定义一个图像预处理函数，它对图像进行变换然后转成卷积层需要的四维格式。

```{.python .input  n=20}
img = image.imread('../img/pikachu.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
```

在预测的时候，我们通过`MultiBoxDetection`函数来合并预测偏移和锚框得到预测边界框，并使用NMS去除重复的预测边界框。

```{.python .input  n=21}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]

output = predict(X)
```

最后我们将预测出置信度超过某个阈值的边框画出来：

```{.python .input  n=22}
gb.set_figsize((5, 5))

def display(img, output, threshold):
    fig = gb.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        gb.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.3)
```

## 小结

* SSD在多尺度上对每个锚框同时预测类别以及与真实边界框的位移来进行目标检测。

## 练习

* 限于篇幅原因我们忽略了SSD实现的许多细节。我们将选取其中数个作为练习。

### 损失函数

边界框预测时使用了$L_1$损失，但这个函数在0点处导数不唯一，因此可能会影响收敛。一个常用改进是在0点附近使用平方函数使得它更加平滑。它被称之为平滑$L_1$损失函数。它通过一个参数$\sigma$来控制平滑的区域：

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

当$\sigma$很大时它类似于$L_1$损失，变小时函数更加平滑。

```{.python .input  n=23}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = nd.arange(-2, 2, 0.1)
gb.set_figsize()

for l, s in zip(lines, sigmas):
    y = nd.smooth_l1(x, scalar=s)
    gb.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
gb.plt.legend();
```

对于类别预测我们使用了交叉熵损失。假设对真实类别$j$的概率预测是$p_j$，交叉熵损失为$\log(p_j)$。我们可以使用一个被称为关注损失（focal loss）的函数来对之稍微变形。给定正的$\gamma$和$\alpha$，它的定义是

$$ - \alpha (1-p_j)^{\gamma} \log(p_j) $$

可以看到，增加$\gamma$可以减小正类预测值比较大时的损失。

```{.python .input  n=24}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * x.log()

x = nd.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = gb.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                    label='gamma=%.1f' % gamma)
gb.plt.legend();
```

### 训练和预测

* 当目标在图像中占比很小时，我们通常会使用比较大的输入图像尺寸。
* 尝试分析不同尺寸上锚框的大小和比例是如何选取的。
* 对锚框赋予标号时，通常会有大量的负类锚框。我们可以对负例采样来使得分类时数据更加平衡。这个可以通过设置`MultiBoxTarget`的参数来完成。
* 分类和回归损失我们直接加起来了，并没有给予各自权重。
* 训练中我们没有实现验证数据集的评估。
* 目标检测算法好坏通常用用mAP（mean Average Precision）来评估，查找它的定义。
* 在展示的时候如何选取阈值，特别是在修改训练算法时（例如增加迭代周期）。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2511)

![](../img/qr_ssd.svg)

## 参考文献

[1] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016, October). Ssd: Single shot multibox detector. In European conference on computer vision (pp. 21-37). Springer, Cham.
