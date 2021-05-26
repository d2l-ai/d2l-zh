# 目标检测和边界框
:label:`sec_bbox`

在前面的章节（例如 :numref:`sec_alexnet`—:numref:`sec_googlenet`）中，我们介绍了各种图像分类模型。在图像分类任务中，我们假设图像中只有 *一个* 主要对象，我们只关注如何识别其类别。但是，感兴趣的图像中通常会有 *多个* 对象。我们不仅想知道他们的类别，还想知道他们在图片中的具体位置。在计算机视觉中，我们指的是 *目标检测*（object detection）或 *物体检测*。 

目标检测已广泛应用于许多领域。例如，自动驾驶需要通过检测拍摄的视频图像中的车辆、行人、道路和障碍物的位置来规划行驶路线。此外，机器人可能会在整个环境导航过程中使用这种技术来检测和本地化感兴趣的对象。此外，安全系统可能需要检测异常物体，例如入侵者或炸弹。 

在接下来的几节中，我们将介绍几种用于目标检测的深度学习方法。我们将首先介绍对象的 *位置*。

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

我们将加载在本节中使用的示例图像。我们可以看到图片的左侧有一只狗，右边有一只猫。它们是这张图片中的两个主要对象。

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

在目标检测中，我们通常使用*边界框*（bounding box）来描述对象的空间位置。边界框是矩形的，由矩形左上角的 $x$ 和 $y$ 坐标以及右下角的坐标决定。另一种常用的边界框表示方法是边界框中心的 $(x, y)$ 轴坐标以及框的宽度和高度。 

在这里，我们定义了在这两种表示之间进行转换的函数：`box_corner_to_center` 从两角表示转换为中心宽度表示，而 `box_center_to_corner` 反之亦然。输入参数 `boxes` 可以是长度为 4 的张量，也可以是形状的二维张量（$n$，4），其中 $n$ 是边界框的数量。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """改变边界框表示形式从(左上角坐标, 右下角坐标) 变为 (中心点坐标, 宽度, 高度)。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """改变边界框表示形式从(左上角坐标, 右下角坐标) 变为 (中心点坐标, 宽度, 高度)。"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

我们将根据坐标信息定义图像中狗和猫的边界框。图像中坐标的原点是图像的左上角，右侧和向下分别是 $x$ 和 $y$ 轴的正方向。

```{.python .input}
#@tab all
# 这里的 `bbox` 是边界框（bounding box）的缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

我们可以通过转换两次来验证两个边界框转换函数的正确性。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

让我们在图像中绘制边界框来检查它们是否准确。在绘图之前，我们将定义一个辅助函数 `bbox_to_rect`。它代表 `matplotlib` 包装的边界框格式的边界框。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """转换边界框到matplotlib格式。"""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

在图像上添加边界框之后，我们可以看到两个物体的主要轮廓基本上在两个框内。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 小结

* 目标检测不仅可以识别图像中所有感兴趣的物体，还能识别它们的位置。该位置通常由矩形边界框表示。
* 我们可以在两种常用的边界框表示之间进行转换。

## 练习

1. 找到另一张图像，然后尝试标记包含该对象的边界框。比较标签边界框和类别：哪些通常需要更长时间？
1. 为什么 `box_corner_to_center` 和 `box_center_to_corner` 的输入参数的最内层维度总是 4？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
