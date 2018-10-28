# 锚框

目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边缘从而更准确预测目标的真实边界框（ground-truth bounding box）。不同的模型所使用的区域采样方法可能不同。这里我们介绍其中的一种方法：它以每个像素为中心生成数个大小和宽高比（aspect ratio）不同的边界框。这些边界框被称为锚框（anchor box）。我们将在后面的小节中基于锚框实践目标检测。

首先，导入本小节需要的包或模块。这里我们新引入了`contrib`包，并修改了NumPy的打印精度。由于NDArray的打印实际调用NumPy的打印函数，本节打印出的NDArray中的浮点数更简洁一些。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import contrib, gluon, image, nd
import numpy as np
np.set_printoptions(2)
```

## 生成锚框

假设输入图像高为$h$，宽为$w$。我们分别以图像的每个像素为中心生成不同形状的锚框。设大小为$s\in (0,1]$且宽高比为$r > 0$，锚框的宽和高分别为$ws\sqrt{r}$和$hs/\sqrt{r}$。当中心位置给定时，已知宽和高的锚框是确定的。

下面我们分别设定好一组大小$s_1,\ldots,s_n$和一组宽高比$r_1,\ldots,r_m$。如果以每个像素为中心时使用所有的大小与宽高比的组合，输入图像将一共得到$whnm$个锚框。虽然这些锚框可能覆盖了所有的真实边界框，但计算复杂度容易过高。因此，我们通常只对包含$s_1$或$r_1$的大小与宽高比的组合感兴趣，即

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

也就是说，以相同像素为中心的锚框数量为$n+m-1$。对于整个输入图像，我们将一共生成$wh(n+m-1)$个锚框。

以上生成锚框的方法已实现在`MultiBoxPrior`函数中。指定输入、一组大小和一组宽高比，该函数将返回输入的所有锚框。

```{.python .input  n=52}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]
x = nd.random.uniform(shape=(1, 3, h, w))  # 构造输入数据。
y = contrib.nd.MultiBoxPrior(x, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
y.shape
```

我们看到，返回锚框变量`y`的形状为（批量大小，锚框个数，4）。将锚框变量`y`的形状变为（图像高，图像宽，以相同像素为中心的锚框个数，4）后，我们就可以通过指定像素位置来获取所有以该像素为中心的锚框了。下面例子里我们访问以（250，250）为中心的第一个锚框。它有四个元素，分别是锚框左上角和右下角的$x$和$y$轴坐标。其中$x$和$y$轴的坐标值分别已除以图像的宽和高，因此值域均为0和1之间。

```{.python .input  n=10}
boxes = y.reshape((h, w, 5, 4))
boxes[250, 250, 0, :]
```

为了描绘图像中以某个像素为中心的所有锚框，我们先定义`show_bboxes`函数以便在图像上画出多个边界框。

```{.python .input  n=11}
# 本函数已保存在 gluonbook 包中方便以后使用。
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

刚刚我们看到，变量`boxes`中$x$和$y$轴的坐标值分别已除以图像的宽和高。在绘图时，我们需要恢复锚框的原始坐标值，并因此定义了变量`bbox_scale`。现在，我们可以画出图像中以（250，250）为中心的所有锚框了。可以看到，大小为0.75且宽高比为1的蓝色锚框较好地覆盖了图像中的狗。

```{.python .input  n=12}
gb.set_figsize()
bbox_scale = nd.array((w, h, w, h))
fig = gb.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## 交并比

我们刚刚提到某个锚框较好地覆盖了图像中的狗。如果该目标的真实边界框已知，这里的“较好”该如何量化呢？一个直观的方法是衡量锚框和真实边界框之间的相似度。我们知道，Jaccard系数（Jaccard index）可以衡量两个集合相似度。给定集合$\mathcal{A}$和$\mathcal{B}$，它们的Jaccard系数即二者交集大小除以二者并集大小：

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$


实际上，我们可以把边界框内的像素区域看成是像素的集合。如此一来，我们可以用两个边界框的像素集合的Jaccard系数衡量这两个边界框的相似度。当衡量两个边界框的相似度时，我们通常将Jaccard系数称为交并比（Intersection over Union，简称IoU），即两个边界框相交面积与相并面积之比，如图9.2所示。交并比的取值范围在0和1之间：0表示两个边界框无重合像素，1表示两个边界框相等。

![交并比。](../img/iou.svg)


## 训练

在训练时，每个锚框都表示成一个样本。对每个样本我们需要预测它是否含有我们感兴趣的目标，以及如果是那么预测它的真实边界框。在训练前我们首先需要为每个锚框生成标签。这里标签有两类，第一类是对应的真实目标的标号。一个常用的构造办法是对每个真实的边界框，我们选取一个或多个与其相似的锚框赋予它们这个真实边界框里的目标标号。具体来说，对一个训练数据中提供的真实边界框，假设其对应目标标号$i$，我们选取所有与其IoU大于某个阈值（例如0.5）的锚框。如果没有这样的锚框，我们就选取IoU值最大的那个（例如0.4）。然后将选中的锚框的目标标号设成$i+1$。如果一个锚框没有被任何真实边界框选中，即不与任何训练数据中的目标足够重合，那么将赋予标号0，代表只含有背景。我们经常将这类锚框叫做负类锚框，其余的则称之为正类。

对于正类锚框，我们还需要构造第二类标号，即它们与真实边界框的距离。一个简单的方法是它与真实边界框的坐标差。但因为有图像边框的限制，这些差值都在-1与1之间， 而且分布差异很大，这使得模型预测变得复杂。所以通常我们会将其进行非线性变化来使得其数值上更加均匀来方便模型预测。

下面来看一个具体的例子。我们将读取的图像中的猫和狗边界框定义成真实边界框，其中第一个元素为类别号（从0开始）。然后我们构造四个锚框，其与真实边界框的位置如下图示。

```{.python .input  n=25}
ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.5, 0.6],
                    [0.5, 0.25, 0.85, 0.85], [0.57, 0.45, 0.85, 0.85]])

fig = gb.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3']);
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

预测同训练类似，是对每个锚框预测其包含的目标类别和与真实边界框的位移。因为我们会生成大量的锚框，所以可能导致对同一个目标产生大量相似的预测边界框。为了使得结果更加简洁，我们需要消除相似的冗余预测框。这里常用的方法是非最大抑制（Non-Maximum Suppression，简称NMS）。对于相近的预测边界框，NMS只保留目标标号预测置信度最高的那个。

具体来说，对于每个目标类别（非背景），我们先获取每个预测边界框里被判断包含这个类别目标的概率。然后我们找到概率最大的那个边界框，如果其置信度大于某个阈值，那么保留它到输出。接下来移除掉其它所有的跟这个边界框的IoU大于某个阈值的边界框。在剩下的边界框里我们再找出预测概率最大的边界框，一直重复前面的移除过程，直到我们遍历保留或者移除了每个边界框。

下面来看一个具体的例子。我们先构造四个锚框，为了简单起见我们假设预测偏移全是0，然后构造了类别预测。

```{.python .input  n=48}
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
loc_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,  # 是背景的概率。
                      [0.9, 0.8, 0.7, 0.1],  # 是狗的概率 。
                      [0.1, 0.2, 0.3, 0.9]])  # 是猫的概率。
```

在实际图像上查看预测边界框的位置和预测置信度：

```{.python .input  n=49}
fig = gb.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

我们使用`contrib.nd`模块的`MultiBoxDetection`函数来执行NMS，这里为NDArray输入都增加了批量维。

```{.python .input  n=50}
ret = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), loc_preds.expand_dims(axis=0),
    anchors.expand_dims(axis=0), nms_threshold=0.5)
ret
```

其返回格式为（批量大小，锚框个数，6）。每一行对应一个预测边界框，包含六个元素，依次为预测类别（移除了背景类，因为不需要标注背景，其中-1表示被该边界框为背景或由于NMS被移除了）、预测目标属于此类的概率和预测边界框。我们移除掉-1类的结果来可视化NMS保留的结果。

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
