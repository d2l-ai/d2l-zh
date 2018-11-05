# 多尺度检测目标

在[“锚框”](anchor.md)一节中，我们在实验中以输入图像的每个像素为中心生成多个锚框。这些锚框是对输入图像不同区域的采样。然而，如果以图像每个像素为中心都生成锚框，很容易生成过多锚框而造成计算量过大。举个例子，假设输入图像的高和宽分别为561和728像素，如果以每个像素为中心生成5个不同形状的锚框，那么一张图像上则需要标注并预测两百多万个锚框（$561 \times 728 \times 5$）。

减少锚框个数并不难。一个简单的方法是在输入图像中均匀采样一小部分像素，并以采样的像素为中心生成锚框。此外，在不同尺度下，我们可以生成不同数量和不同大小的锚框。实际上，较小目标比较大目标在图像上出现位置的可能性更多。举个简单的例子：形状为$1 \times 1$、$1 \times 2$和$2 \times 2$的目标在形状为$2 \times 2$的图像上可能出现的位置分别有4、2和1种。因此，当使用较小锚框来检测较小目标时，我们可以采样较多的区域；而当使用较大锚框来检测较大目标时，我们可以采样较少的区域。

为了演示如何多尺度生成锚框，我们先读取一张图像。它的高和宽分别为561和728像素。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import contrib, image, nd

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```

我们在[“二维卷积层”](../chapter_convolutional-neural-networks/conv-layer.md)一节中将卷积神经网络的二维数组输出称为特征图。
我们可以通过定义特征图的形状来确定任一图像上均匀采样的锚框中心。

下面定义`display_anchors`函数。我们在特征图`fmap`上以每个像素（单元）为中心生成锚框`anchors`。由于锚框`anchors`中$x$和$y$轴的坐标值分别已除以特征图`fmap`的宽和高，这些值域在0和1之间的值表达了锚框在特征图中的相对位置。由于锚框`anchors`的中心遍布特征图`fmap`上的所有像素，`anchors`的中心在任一图像宽和高的相对位置一定是均匀分布的。具体来说，当特征图的宽和高分别设为`fmap_w`和`fmap_h`时，该函数将在任一图像上均匀采样`fmap_h`行`fmap_w`列个像素，并分别以它们为中心生成大小为`s`（假设列表`s`长度为1）的三个不同宽高比（`ratios`）的锚框。

```{.python .input  n=2}
def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 3, fmap_w, fmap_h))
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    gb.show_bboxes(gb.plt.imshow(img.asnumpy()).axes, anchors[0] * bbox_scale)
```

我们先关注小目标的检测。为了在显示时更容易分辨，这里令不同中心的锚框不重合：设锚框大小为0.15，特征图的高和宽分别为4。可以看出，图像上4行4列的锚框中心分布均匀。

```{.python .input  n=3}
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

我们将特征图的高和宽分别减半，并用更大的锚框检测更大的目标。当锚框大小设0.4时，有些锚框的区域有重合。

```{.python .input  n=4}
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

最后，我们将特征图的高和宽进一步减半至1，并将锚框大小增至0.8。此时锚框中心即图像中心。

```{.python .input  n=5}
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 小结

* SSD在多尺度上对每个锚框同时预测类别以及与真实边界框的位移来进行目标检测。

## 练习

* 限于篇幅原因我们忽略了SSD实现的许多细节。我们将选取其中数个作为练习。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/8859)

![](../img/qr_multiscale-object-detection.svg)
