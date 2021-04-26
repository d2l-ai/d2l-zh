# 目标检测和边界框
:label:`sec_bbox`

在上一节中，我们介绍了许多用于图像分类的模型。在图像分类任务中，我们假设图像中只有一个主要目标，我们只关注如何识别目标的类别。
然而，在许多情况下，图像中有多个我们感兴趣的目标。我们不仅要对它们进行分类，还要获取它们在图像中的具体位置。在计算机视觉中，我们把这些任务称为目标检测（或目标识别）。

目标检测在许多领域有着广泛的应用。例如，在自动驾驶技术中，我们需要通过在捕获的视频图像中识别车辆、行人、道路和障碍物的位置来规划路线。机器人通常执行这类任务来检测感兴趣的目标。安全领域的系统则需要检测异常目标，如入侵者或炸弹。

在接下来的几节中，我们将介绍用于目标检测的多种深度学习模型。在此之前，我们应该先讨论一下目标定位的概念。首先，导入实验所需的包和模块。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

接下来，我们要加载即将在本节中使用的示例图像。我们可以看到在图片的左边有一只狗，右边有一只猫。它们是这张照片中的两个主要目标。

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 边界框

在目标检测中，我们通常使用边界框来描述目标的位置。边界框是一个矩形框，可由矩形左上角的$x$和$y$轴坐标以及右下角的$x$和$y$轴坐标确定。另一种常用的边界框表示法是边界框中心的$x$和$y$轴坐标、及其宽度和高度。这里我们定义了在这两种表示法之间转换的函数， `box_corner_to_center` 将两个角表示法转换为中心宽度-高度表示法， `box_center_to_corner` 反之。输入参数 `boxes` 可以是长度为$4$的一维张量，也可以是$(N, 4)$二维张量。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper_left, bottom_right)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

我们将根据坐标信息定义图像中狗和猫的边界框。图像中坐标的原点是图像的左上角，向右和向下分别是$x$轴和$y$轴的正方向。

```{.python .input}
#@tab all
# bbox is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

我们可以通过转换两次来验证box转换函数的正确性。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) - boxes
```

我们可以在图像中画出边界框来检查它是否准确。在绘制框之前，我们将定义一个助手函数 `bbox_to_rect` 。它以 `matplotlib` 的边界框格式表示边界框。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

在图像上加载边界框后，我们可以看到目标的主要轮廓大体上都在框内。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 小结

* 在目标检测中，我们不仅要识别图像中所有感兴趣的目标，还要识别它们的位置。位置通常由矩形的边界框表示。

## 练习

1. 找一些图像并尝试标记出包含目标的边界框。比较一下标记边界框和标记类别所需的时间。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
