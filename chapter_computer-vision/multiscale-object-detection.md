# 多尺度对象检测

在 :numref:`sec_anchor` 中，我们生成了以输入图像的每个像素为中心的多个锚框。基本上，这些锚框代表图像不同区域的样本。但是，如果是为 * 每个 * 像素生成的锚框，我们最终可能会得到太多的锚框来计算。想想一下 $561 \times 728$ 输入图像。如果为每个像素生成五个形状不同的锚框作为中心，则需要在图像上标记和预测超过 200 万个锚框 ($561 \times 728 \times 5$)。 

## 多尺度锚框
:label:`subsec_multiscale-anchor-boxes`

你可能会意识到减少图像上的锚框并不困难。例如，我们只能从输入图像中均匀地采样一小部分像素来生成以它们为中心的锚框。此外，在不同的比例下，我们可以生成不同大小的不同数量的锚框。直观地说，较小的物体比较大的物体更有可能出现在图像上。例如，$1 \times 1$、$1 \times 2$ 和 $2 \times 2$ 对象可以分别以 4、2 和 1 种可能的方式出现在 $2 \times 2$ 图像上。因此，当使用较小的锚框检测较小的物体时，我们可以采样更多的区域，而对于较大的物体，我们可以采样较少的区域。 

为了演示如何在多个比例下生成锚框，让我们阅读图片。它的高度和宽度分别为 561 和 728 像素。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

回想一下，在 :numref:`sec_conv_layer` 中，我们将卷积图层的二维数组输出称为特征映射。通过定义特征映射的形状，我们可以确定任何图像上均匀采样锚框的中心。 

`display_anchors` 函数定义如下。我们在功能地图 (`fmap`) 上生成锚框 (`anchors`)，每个单位（像素）作为锚框的中心。由于锚点框中的 $(x, y)$ 轴坐标值 (`anchors`) 被除以要素地图的宽度和高度 (`fmap`)，因此这些值介于 0 和 1 之间，表示要素地图中锚点框的相对位置。 

由于锚框 (`anchors`) 的中心分布在特征映射 (`fmap`) 上的所有单位上，因此这些中心必须根据其相对空间位置在任何输入图像上 * 统一 * 分布。更具体地说，鉴于功能地图 `fmap_w` 和 `fmap_h` 的宽度和高度，以下函数将 *统一* 对任何输入图像中的 `fmap_h` 行和 `fmap_w` 列中的像素进行采样。以这些均匀采样像素为中心，将生成大小 `s` 的锚框（假设列表 `s` 的长度为 1）和不同的纵横比（`ratios`）。

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

首先，让我们考虑探测小物体。为了更容易区分显示时间，此处具有不同中心的锚框不会重叠：锚框比例设置为 0.15，要素地图的高度和宽度设置为 4。我们可以看到，图像上 4 行和 4 列的锚框的中心分布均匀。

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

我们继续将要素地图的高度和宽度减少一半，然后使用较大的锚框来检测较大的物体。当比例设置为 0.4 时，一些锚框将彼此重叠。

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

最后，我们进一步将要素地图的高度和宽度减少一半，然后将锚框的比例增加到 0.8。现在锚框的中心是图像的中心。

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 多尺度检测

由于我们已经生成了多尺度的锚框，我们将使用它们来检测不同尺寸下各种尺寸的物体。在下面我们将介绍一种基于 CNN 的多尺度目标检测方法，我们将在 :numref:`sec_ssd` 中实现。 

在某种规模上，假设我们有 $c$ 形状为 $h \times w$ 的特征地图。使用 :numref:`subsec_multiscale-anchor-boxes` 中的方法，我们生成了 $hw$ 套锚框，其中每组都有 $a$ 个中心相同的锚框。例如，在 :numref:`subsec_multiscale-anchor-boxes` 实验的第一个尺度上，给出 10 个（通道数量）$4 \times 4$ 特征地图，我们生成了 16 套锚盒，每组包含 3 个中心相同的锚框。接下来，每个锚框都用类标记，并根据地面真实边界框进行偏移。在当前规模下，物体检测模型需要预测输入图像上 $hw$ 组锚框的类和偏移量，其中不同集合具有不同的中心。 

假设此处的 $c$ 特征图是 CNN 基于输入图像的正向传播获得的中间输出。由于每个要素地图上有 $hw$ 个不同的空间位置，因此可以认为相同的空间位置具有 $c$ 个单位。根据 :numref:`sec_conv_layer` 中对接受场的定义，这些 $c$ 单位位于要素地图同一空间位置的单位在输入图像上具有相同的接受场：它们代表同一接收场中的输入图像信息。因此，我们可以将同一空间位置的 $c$ 个要素地图单位转换为使用此空间位置生成的 $a$ 锚框的类和偏移量。实质上，我们使用某个接收字段中输入图像的信息来预测接近输入图像上该接收场的锚点框的类和偏移量。 

当不同图层的要素地图在输入图像上具有变化大小的接受字段时，它们可用于检测不同大小的对象。例如，我们可以设计一个神经网络，其中靠近输出图层的要素地图单位具有更宽的接受场，这样它们就可以从输入图像中检测到较大的对象。 

## 小结

* 在多个比例下，我们可以生成不同尺寸的锚框来检测不同尺寸的物体。
* 通过定义特征地图的形状，我们可以确定任何图像上均匀采样的锚框的中心。
* 我们使用特定接受字段中输入图像的信息来预测接近输入图像上该接收场的锚点框的类和偏移量。

## 练习

1. 在 :numref:`subsec_multiscale-anchor-boxes` 年实验中的第一个尺度（`fmap_w=4, fmap_h=4`），生成可能重叠的均匀分布的锚盒。
1. 给定形状为 $1 \times c \times h \times w$ 的特征图变量，其中 $h$ 和 $w$ 分别是功能地图的通道数、高度和宽度。你怎样才能将这个变量转换为锚框的类和偏移量？输出的形状是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
