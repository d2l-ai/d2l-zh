# 物体识别：边界框和预测

前面小节里我们介绍了诸多用于图片分类的模型。在这个任务里，我们假设图片里只有一个主体物体，然后目标是识别这个物体的类别。但很多时候图片里有多个感兴趣的物体，我们不仅仅想知道它们是什么，而且想得到它们在图片中的具体位置。例如在无人驾驶任务里，我们需要识别拍摄到的图片里的车辆、行人、道路和障碍的位置来规划行进线路。在计算机视觉里，我们将这类任务称之为物体识别。

在接下来的数小节里我们将介绍物体识别里的多个深度学习模型。在此之前，让我们先讨论物体位置这个概念。首先我们加载本小节将使用的示例图片。

```{.python .input  n=1}
%matplotlib inline
import sys
sys.path.insert(0, '..')
import gluonbook as gb
import numpy as np
from mxnet import image, nd, contrib

img = image.imread('../img/catdog.jpg').asnumpy()
gb.plt.imshow(img);  # 用 ; 使得不要显示它的输出。
```

可以看到图片左边是一只小狗，右边是一只小猫。跟前面使用的图片的主要不同点在于这里有两个主体物体。

## 边界框

在物体识别里，我们通常使用边界框（bounding box）来确定物体位置。它一个矩形框，可以由左上角的x、y轴位置与右下角x、y轴位置确定。我们根据上图坐标信息来定义图中小狗和小猫的边界框。

```{.python .input  n=2}
# 注意坐标轴原点是图片的左上角。bbox 是 bounding box 的缩写。
dog_bbox = [60, 0, 340, 365]
cat_bbox = [360, 80, 580, 365]
```

我们可以在图中将边框画出来检查其准确性。画之前我们定义一个辅助函数，它将边界框表示成matplotlib的边框格式，这个函数将保存在GluonBook里方便之后使用。

```{.python .input  n=3}
# 将边界框（左上 x、左上 y，右下 x，右下 y）格式转换成 matplotlib 格式：
# （（左上 x，左上 y），宽，高）。
def bbox_to_rect(bbox, color):
    return gb.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

我们将边界框加载在图上，可以看到物体的主体基本在框内。

```{.python .input}
fig = gb.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 预测边界框

相比于图片分类，物体识别的一个主要复杂点在于需要预测物体的边界框。一般这是通过下面两个步骤来完成：首先针对输入图片提出数个区域，然后对每个区域判断其是否包含感兴趣的物体，如果是则进一步预测其边界框。

## 锚框

不同的模型使用不同的区域生成方法，这里我们介绍其中常用的一种：它以每个输入像素为中心生成数个大小和比例不同的默认边界框，或者称之为锚框（anchor box）。假设输入高为$h$，宽为$w$，那么大小为$s\in (0,1]$和比例为$r > 0$的锚框形状是

$$\left( ws\sqrt{r}, \  \frac{hs}{\sqrt{r}}\right).$$

注意到便利不同的$s$的$r$会生成大量的锚框，这样将使得计算很复杂。一般我们需要对其进行采样。一个例子是首先固定一个比例$r_1$，然后采样$n$个不同的大小$s_1,\ldots,s_n$。然后固定一个大小$s_1$，采样$m$个不同的比例$r_1,\ldots,r_m$。这样对每个像素我们一共生成$n+m-1$个锚框。对于整个输入图片，我们将一共生成$wh(n+m-1)$个锚框。

上面描述的采样方法实现在`contribe.ndarray`中的`MultiBoxPrior`函数。通过指定输入数据（我们只需要访问其形状），锚框的采样大小和比例，这个函数将返回所有采样到的锚框。

```{.python .input  n=4}
h, w = img.shape[0:2]
x = nd.random.uniform(shape=(1, 3, h, w))
y = contrib.ndarray.MultiBoxPrior(x, sizes=[.75, .5, .25], ratios=[1, 2, .5])
('total #anchor boxes', y.shape[1])
```

将返回结果变形成（高，宽，$n+m-1$，4）后，我们可以方便的访问以任何一个像素为中心的所有锚框。下面例子里我们访问以（200，200）为中心的第一个锚框。它有四个元素，同前一样是左上和右下的x、y轴坐标，但被分别除以了高和宽使得数值在0和1之间。

```{.python .input}
boxes = y.reshape((h, w, 5, 4))
boxes[200, 200, 0, :]
```

在画出这些锚框的具体样子前我们先定义一个函数来图上画出多个边界框，它将被保存在GluonBook里以便后面使用。

```{.python .input  n=5}
def show_bboxs(axes, bboxs, labels=None):
    colors = ['b', 'g', 'r', 'k', 'm']
    for i, bbox in enumerate(bboxs):
        color = colors[i%len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)        
        if labels and len(labels) > i:
            axes.text(rect.xy[0], rect.xy[1], labels[i], 
                      va="center", ha="center", fontsize=9, color='white',
                      bbox=dict(facecolor=color, lw=0))
```

然后我们画出以（200，200）为中心的所有锚框。

```{.python .input  n=6}
bbox_scale = nd.array((w, h, w, h))  # 需要乘以高和宽使得符合我们的画图格式。
fig = gb.plt.imshow(img)
show_bboxs(fig.axes, boxes[200, 200, :, :] * bbox_scale, [
    's=.75, r=1', 's=.5, r=1', 's=.25, r=1', 's=.75, r=2', 's=.75, r=.5'])
```

可以看到大小为0.75比例为0.5的洋红色锚框比较好的覆盖了图片中的小狗。

在预测的时候，我们对每个锚框预测一个到真实边界框的偏移。例如对于洋红锚框来说，我们希望其预测值是：

```{.python .input}
boxes[200, 200, 3, :] - nd.array(dog_bbox) / bbox_scale
```

## IoU：交集除并集

上面例子里我们生成了一百万以上的边界框，从而会有同样的预测边界框。而图片中只有两个物体，这导致大量的边界框都非常相似。我们需要去除冗余，使得预测结果更有可读性。

为了去除冗余，我们需要先定义如何判断两个边界框的相似性。我们知道集合相似度的最常用衡量标准叫做Jaccard距离。给定集合$A$和$B$，它的定义是集合的交集除以集合的并集：

$$J(A,B) = \frac{|A\cap B|}{| A \cup B|}$$

边界框指定了一块像素区域，其也可以看成是像素点的集合。因此我们可以定类似的距离，即将两个边界框的相交面积除以相并面积。

![](../img/iou.svg)

我们将这个测量方法称之为交集除并集（Intersection over Union，IoU）。它的取值范围在0和1之间。0表示边界框不相关，1则表示完全一样。

## NMS：非最大抑制

对于相似的预测边界框，非最大抑制（Non-Maximum Suppression，NMS）只保留置信度最高的那个来去除冗余。它的工作机制如下：对于每个类别，我们首先拿到每个预测边界框被判断包含这个类别物体的概率。然后我们找到概率最大的那个边界框并保留它到输出，接下来移除掉（抑制）其它所有的跟这个边界框的IoU大于某个阈值的边界框。在剩下的边界框里我们再找出预测概率最大的边界框，重复前面的移除过程。直到我们要么保留或者移除了每个边界框。

非最大抑制的实现包含在`contrib.ndarray`的`MultiBoxDetection`里。这里锚框的格式是（批量大小，锚框个数，4），预测偏移则是（批量大小，锚框个数$\times 4$），这里偏移使用了更符合网络输出层的格式。假设锚框是`A`，预测偏移是`P`，那么预测边界框就是`A+P.reshape(A.shape)`。类别预测的格式是（批量大小，类别数$+1$，锚框个数），这里第0类预留给了背景，即不含有需要被识别的物体。

下面我们构造四个预测框。为了简单起见我们在锚框直接放置预测边界框内容，预测偏移则设成0。

```{.python .input  n=7}
anchors = nd.array([[[.09, .00, .53, 1],  # 每一行是一个预测框。
                     [.08, .09, .56, .95], 
                     [.15, .16, .62, .91],
                     [.55, .22, .89, 1]]])
loc_preds = nd.array([[.0]*anchors.size])
cls_probs = nd.array([[[0]*4,  # 是背景的概率。
                       [.9, .8, .7, .1],  # 是狗的概率 。
                       [.1, .2, .3, .9]]])  # 是猫的概率。
```

在实际图片上看一下他们的位置和预测概率：

```{.python .input  n=8}
fig = gb.plt.imshow(img)
show_bboxs(fig.axes, anchors.reshape(-1, 4) * bbox_scale,
           ['dog=.9', 'dog=.8', 'dog=.7', 'cat=.9'])
```

给定锚框、预测偏移、类别预测概率、和IoU阈值，`MultiBoxDetection`返回格式为（批量大小，锚框个数，6）的结果。每一行对应一个预测边界框，其有六个元素，依次为预测类别（-1表示被该边界框抑制了，其余物体类别从0开始，且移除了背景类）、预测物体属于此类的概率和预测边界框的位置。

```{.python .input  n=9}
np.set_printoptions(2)  # 使得打印更加简洁。
ret = contrib.ndarray.MultiBoxDetection(
    cls_probs, loc_preds, anchors, nms_threshold=.5).asnumpy()
ret
```

我们移除掉-1类的结果来可视化NMS保留的结果。

```{.python .input  n=11}
bboxs = [nd.array(i[2:])*bbox_scale for i in ret[0] if i[0] != -1]
labels = [('dog=','cat=')[int(i[0])]+str(i[1]) for i in ret[0] if i[0] != -1]
fig = gb.plt.imshow(img)        
show_bboxs(fig.axes, bboxs, labels) 
```

NMS在70年代提出后有数个变种。例如假设一张图片里出现的物体数远小于类别数，所以对每个类做抑制可能意义不大，因为绝大部分类别物体不会出现。因此我们可以忽略掉类别来做来全局抑制，这样避免某些类即使最大的预测概率很低但仍然被输出。此外，我们也可以指定只返回预测值最高的固定数目的结果。

## 小节

在物体识别里我们不仅需要找出图片里面所有感兴趣的物体，而且要知道它们的位置。位置一般由方形边界框来表示。这一小节我们讨论了如何生产一系列的锚框来预测边界框，和在预测过程中如何用IoU来判断边界框的相似度并使用NMS来消除相似的输出使得预测边界框更加简洁。

## 练习

- 找一些图片，尝试标注下其中物体的边界框。比较下同图片分类标注所花时间的区别。
- 改变锚框生成里面的大小和比例采样来看看可视化时的区别。
- 修改NMS的阈值来看其对结果的影响。
- 试试`MultiBoxDetection`里的其他选项，例如`force_suppress`和`nms_topk`。
