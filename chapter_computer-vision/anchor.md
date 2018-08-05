# 锚框

物体识别算法通常会在输入图片中采样大量的区域，然后判断这些区域是否有我们感兴趣的物体，以及进一步调整区域边缘来更准确预测物体的真实边界框。不同的模型使用不同的区域采样方法，这里我们介绍其中的一种：它以每个像素为中心生成数个大小和比例不同的被称之为锚框（anchor box）的边界框。

导入本小节需要的包。注意我们新引入了`contrib`这个模块，以及使用NumPy修改了打印精度，这是因为NDArray的打印实际上调用了NumPy的打印函数。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import contrib, gluon, image, nd
import numpy as np
np.set_printoptions(2)
```

## 锚框的生成

假设输入图片高为$h$，宽为$w$，那么大小为$s\in (0,1]$和比例为$r > 0$的锚框形状是

$$\left( ws\sqrt{r}, \  \frac{hs}{\sqrt{r}}\right),$$

确定其中心点位置便可以固定一个锚框。

当然我们可以通过使用不同的$s$和$r$，以及改变中心位置，来遍历所有可能的区域。虽然这样可以覆盖真实边界框，但会使得计算很复杂。通常我们进行采样，使得锚框尽量贴近真实边界框。 例如我们可以首先固定一个比例$r_1$，然后采样$n$个不同的大小$s_1,\ldots,s_n$。然后固定一个大小$s_1$，采样$m$个不同的比例$r_1,\ldots,r_m$。这样对每个像素我们一共生成$n+m-1$个锚框。对于整个输入图片，我们将一共生成$wh(n+m-1)$个锚框。

上述的采样方法实现在`contrib.ndarray`中的`MultiBoxPrior`函数。通过指定输入数据（我们只需要访问其形状），锚框的采样大小和比例，这个函数将返回所有采样到的锚框。

```{.python .input  n=52}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]
x = nd.random.uniform(shape=(1, 3, h, w))  # 构造一个输入数据。
y = contrib.nd.MultiBoxPrior(x, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
y.shape
```

其返回结果格式为（批量大小，锚框个数，4）。可以看到我们生成了2百万以上个锚框。将其变形成（高，宽，$n+m-1$，4）后，我们可以方便的访问以任何一个像素为中心的所有锚框。下面例子里我们访问以（250，250）为中心的第一个锚框。它有四个元素，同之前一样是左上和右下的x、y轴坐标，但被分别除以了高和宽使得数值在0和1之间。

```{.python .input  n=10}
boxes = y.reshape((h, w, 5, 4))
boxes[250, 250, 0, :]
```

在画出这些锚框的具体样子前，我们需要定义`show_bboxes`函数在图上画出多个边界框。我们将该函数定义在`gluonbook`包中供后面章节调用。

```{.python .input  n=11}
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = gb.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

然后我们画出以（200，200）为中心的所有锚框。

```{.python .input  n=12}
gb.set_figsize()
bbox_scale = nd.array((w, h, w, h))  # 需要乘以高和宽使得符合我们的画图格式。
fig = gb.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

可以看到大小为0.75比例为1的蓝色锚框比较好的覆盖了图片中的小狗。

## IoU：交集除并集

在介绍如何使用锚框参与训练和预测前，我们先介绍如何判断两个边界框的距离。我们知道集合相似度的最常用衡量标准叫做Jaccard距离。给定集合$A$和$B$，它的定义是集合的交集除以集合的并集：

$$J(A,B) = \frac{|A\cap B|}{| A \cup B|}.$$

边界框指定了一块像素区域，其可以看成是像素点的集合。因此我们可以定义类似的距离，即我们使用两个边界框的相交面积除以相并面积来衡量它们的相似度。这被称之为交集除并集（Intersection over Union，简称IoU），如图9.2所示。它的取值范围在0和1之间。0表示边界框不相关，1则表示完全重合。

![交集除并集。](../img/iou.svg)


## 训练

在训练时，每个锚框都表示成一个样本。对每个样本我们需要预测它是否含有我们感兴趣的物体，以及如果是那么预测它的真实边界框。在训练前我们首先需要为每个锚框生成标签。这里标签有两类，第一类是对应的真实物体的标号。一个常用的构造办法是对每个真实的边界框，我们选取一个或多个与其相似的锚框赋予它们这个真实边界框里的物体标号。具体来说，对一个训练数据中提供的真实边界框，假设其对应物体标号$i$，我们选取所有与其IoU大于某个阈值（例如0.5）的锚框。如果没有这样的锚框，我们就选取IoU值最大的那个（例如0.4）。然后将选中的锚框的物体标号设成$i+1$。如果一个锚框没有被任何真实边界框选中，即不与任何训练数据中的物体足够重合，那么将赋予标号0，代表只含有背景。我们经常将这类锚框叫做负类锚框，其余的则称之为正类。

对于正类锚框，我们还需要构造第二类标号，即它们与真实边界框的距离。一个简单的方法是它与真实边界框的坐标差。但因为有图片边框的限制，这些差值都在-1与1之间， 而且分布差异很大，这使得模型预测变得复杂。所以通常我们会将其进行非线性变化来使得其数值上更加均匀来方便模型预测。

下面来看一个具体的例子。我们将读取的图片中的猫和狗边界框定义成真实边界框，其中第一个元素为类别号（从0开始）。然后我们构造四个锚框，其与真实边界框的位置如下图示。

```{.python .input  n=25}
ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.5, 0.6],
                    [0.5, 0.25, 0.85, 0.85], [0.57, 0.45, 0.85, 0.85]])

fig = gb.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3'] );
```

我们可以通过`contrib.nd`模块中的MultiBoxTarget函数来对锚框生成标号。我们把锚框和真实边界框加上批量维（实际中我们会批量处理数据），然后构造一个任意的锚框预测结果，其形状为（批量大小，类别数+1，锚框数），其中第0类为背景。

```{.python .input  n=26}
out = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
                                ground_truth.expand_dims(axis=0),
                                nd.zeros((1, 3, 4)))
```

返回的结果里有三个`NDArray`。首先看第三个值，其表示赋予锚框的标号。

```{.python .input  n=27}
out[2]
```

我们可以逐一分析每个锚框被赋予给定标号的理由：

1. 锚框0里被认定只有背景，因为它与所有真实边界框的IoU都小于0.5。
2. 锚框1里被认定为有狗，虽然它与狗的边界框的IoU小于0.5，但是它是这四个锚框里离狗的边界框最近的那个。
3. 锚框2里被认定为有猫，因为它与猫的边界框的IoU大于0.5。
4. 锚框3里被认定只有背景，虽然它与猫的边界框的IoU类似于锚框1与狗的边界框的IoU，但由于其小于0.5，且锚框2已经获得了猫的标号，所以不予理会。

返回值的第二项用来遮掩不需要的负类锚框，其形状为（批量大小，锚框数$\times 4$）。其中正类锚框对应的元素为1，负类为0。

```{.python .input  n=29}
out[1]
```

返回的第一项是锚框与真实边界框的偏移，只有正类锚框有非0值。

```{.python .input  n=36}
out[0]
```

## 预测

预测同训练类似，是对每个锚框预测其包含的物体类别和与真实边界框的位移。因为我们会生成大量的锚框，所以可能导致对同一个物体产生大量相似的预测边界框。为了使得结果更加简洁，我们需要消除相似的冗余预测框。这里常用的方法是非最大抑制（Non-Maximum Suppression，简称NMS）。对于相近的预测边界框，NMS只保留物体标号预测置信度最高的那个。

具体来说，对于每个物体类别（非背景），我们先获取每个预测边界框里被判断包含这个类别物体的概率。然后我们找到概率最大的那个边界框，如果其置信度大于某个阈值，那么保留它到输出。接下来移除掉其它所有的跟这个边界框的IoU大于某个阈值的边界框。在剩下的边界框里我们再找出预测概率最大的边界框，一直重复前面的移除过程，直到我们遍历保留或者移除了每个边界框。

下面来看一个具体的例子。我们先构造四个锚框，为了简单起见我们假设预测偏移全是0，然后构造了类别预测。

```{.python .input  n=48}
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
loc_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,  # 是背景的概率。
                      [0.9, 0.8, 0.7, 0.1],  # 是狗的概率 。
                      [0.1, 0.2, 0.3, 0.9]])  # 是猫的概率。
```

在实际图片上查看预测边界框的位置和预测置信度：

```{.python .input  n=49}
fig = gb.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7',' cat=0.9'])
```

我们使用`contrib.nd`模块的`MultiBoxDetection`函数来执行NMS，这里为NDArray输入都增加了批量维。

```{.python .input  n=50}
ret = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), loc_preds.expand_dims(axis=0),
    anchors.expand_dims(axis=0), nms_threshold=.5)
ret
```

其返回格式为（批量大小，锚框个数，6）。每一行对应一个预测边界框，包含六个元素，依次为预测类别（移除了背景类，因为不需要标注背景，其中-1表示被该边界框为背景或由于NMS被移除了）、预测物体属于此类的概率和预测边界框。我们移除掉-1类的结果来可视化NMS保留的结果。

```{.python .input  n=51}
fig = gb.plt.imshow(img)
for i in ret[0].asnumpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)
```

## 小结

- 以每个像素为中心我们生成多个大小比例不同的锚框来预测真实边界框。
- 训练时我们根据真实边界框来为每个锚框赋予类别标号和偏移这两类标签。
- 预测时我们移除重合度很高的预测值来保持结果简洁。

## 练习

- 改变锚框生成里面的大小和比例采样来看看可视化时的区别。
- 构造IoU是0.5的两个边界框，看看视觉上他们的重合度。
- 修改训练和预测里的`anchors`来看他们对结果的影响。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7024)

![](../img/qr_anchor.svg)
