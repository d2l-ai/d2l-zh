# 锚盒
:label:`sec_anchor`

对象检测算法通常会对输入图像中的大量区域进行采样，确定这些区域是否包含感兴趣的对象，并调整区域的边界以预测
*地面真实边界框 *
更准确的对象。不同的模型可能采用不同的区域抽样方案在这里我们介绍一下这样的方法：它会生成多个边界框，每个像素都以不同的比例和长宽比为中心。这些边界框被称为 * 锚框 *。我们将在 :numref:`sec_ssd` 中基于锚框设计一个物体检测模型。 

首先，让我们修改打印精度，以获得更简洁的输出。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## 生成多个锚框

假设输入图像的高度为 $h$，宽度为 $w$。我们以图像的每个像素为中心生成不同形状的锚框。让 * 比例 * 为 $s\in (0, 1]$，* 宽高比 *（宽高比）为 $r > 0$。那么锚箱的宽度和高度分别是 $ws\sqrt{r}$ 和 $hs/\sqrt{r}$。请注意，当给出中心位置时，将确定一个已知宽度和高度的锚框。 

要生成多个不同形状的锚框，让我们设置一系列刻度 $s_1,\ldots, s_n$ 和一系列宽高比 $r_1,\ldots, r_m$。当使用这些比例和长宽比的所有组合以每个像素为中心时，输入图像将总共有 $whnm$ 个锚框。尽管这些锚框可能会覆盖所有地面真实边界框，但计算复杂性很容易太高。在实践中，我们只能考虑包含 $s_1$ 或 $r_1$ 的组合： 

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

也就是说，以同一像素为中心的锚框的数量是 $n+m-1$。对于整个输入图像，我们将共生成 $wh(n+m-1)$ 个锚框。 

上述生成锚框的方法在以下 `multibox_prior` 函数中实现。我们指定输入图像、比例列表和纵横比列表，然后此函数将返回所有的锚框。

```{.python .input}
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

我们可以看到返回的锚框变量 `Y` 的形状是（批量大小，锚框的数量，4）。

```{.python .input}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

将锚框的形状变量 `Y` 更改为（图像高度、图像宽度、以同一像素为中心的锚框的数量，4）后，我们可以获得以指定像素位置为中心的所有锚框。在以下内容中，我们访问以 (250, 250) 为中心的第一个锚框。它有四个元素：左上角的 $(x, y)$ 轴坐标，锚点框右下角的 $(x, y)$ 轴坐标。两个轴的坐标值分别除以图像的宽度和高度；因此，范围介于 0 和 1 之间。

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

为了显示以图像中一个像素为中心的所有锚框，我们定义了以下 `show_bboxes` 函数来在图像上绘制多个边界框。

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

正如我们刚才看到的，变量 `boxes` 中 $y$ 轴的坐标值已分别除以图像的宽度和高度。绘制锚点框时，我们需要恢复它们的原始坐标值；因此，我们在下面定义了变量 `bbox_scale`。现在，我们可以绘制图像中所有以（250、250）为中心的锚框。正如你所看到的，刻度为 0.75 且纵横比为 1 的蓝色锚框很好地围绕着图像中的狗。

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## 联盟上的交叉点 (IoU)

我们刚才提到图像中的狗周围有一个锚盒 “好”。如果已知物体的地面真相边界框，那么如何量化这里的 “好”？直观地说，我们可以测量锚框和地面真相边界框之间的相似性。我们知道 *Jaccard 索引 * 可以衡量两组之间的相似性。给定系列 $\mathcal{A}$ 和 $\mathcal{B}$，他们的 Jaccard 指数是他们的交叉点的大小除以工会的规模： 

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

事实上，我们可以将任何边界框的像素区域视为一组像素。通过这种方式，我们可以通过其像素集的 Jaccard 索引来测量两个边界框的相似性。对于两个边界框，我们通常将他们的 Jaccard 指数称为 * 跨工会 * (*iOU*)，这是他们的交叉区域与联盟区的比率，如 :numref:`fig_iou` 所示。iOU 的范围介于 0 到 1：0 之间，表示两个边界框根本不重叠，而 1 表示两个边界框相等。 

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

在本节的剩余部分中，我们将使用 iOU 来测量锚框和地面真实边界框之间以及不同锚框之间的相似性。给定两个锚点或边界框列表，以下 `box_iou` 将在这两个列表中计算它们的成对 iOU。

```{.python .input}
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## 在训练数据中标记锚箱
:label:`subsec_labeling-anchor-boxes`

在训练数据集中，我们将每个锚点框视为训练示例。为了训练对象检测模型，我们需要每个锚框的 *class * 和 *offset* 标签，其中前者是与锚框相关的对象的类，后者是地面真相边界框相对于锚框的偏移量。在预测期间，我们为每个图像生成多个锚框，预测所有锚点框的类和偏移量，根据预测的偏移调整它们的位置以获得预测的边界框，最后只输出符合特定条件的预测边界框。 

正如我们所知，物体检测训练套装附带了 * 地面真实边界框 * 及其包围物体类别的标签。要标记任何生成的 * 锚框 *，我们可以参考其 * 分配的 * 地面真实边界框的标记位置和类别，即锚框的壁橱。在下文中，我们介绍了将最接近的地面真实边界框分配给锚框的算法。  

### 将地面真实边界框分配给锚框

给定图像，假设锚盒是 $A_1, A_2, \ldots, A_{n_a}$，地面真实边界框是 $B_1, B_2, \ldots, B_{n_b}$，其中 $n_a \geq n_b$。让我们定义一个矩阵 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$，其中 $i^\mathrm{th}$ 行中的元素 $x_{ij}$ 和 $j^\mathrm{th}$ 列是锚盒 $A_i$ 的 iOU 和地面真实边界框 $B_j$。该算法包括以下步骤： 

1. 在矩阵 $\mathbf{X}$ 中找到最大的元素，并将其行索引和列索引分别表示为 $i_1$ 和 $j_1$。然后将地面真实边界框 $B_{j_1}$ 分配给锚框 $A_{i_1}$。这很直观，因为 $A_{i_1}$ 和 $B_{j_1}$ 是所有锚盒和地面真实边界盒中的衣柜。在第一个赋值之后，丢弃 ${i_1}^\mathrm{th}$ 行中的所有元素和矩阵 ${j_1}^\mathrm{th}$ 列中的所有元素。 
1. 在矩阵 $\mathbf{X}$ 中找到剩余元素中最大的元素，并将其行索引和列索引分别表示为 $i_2$ 和 $j_2$。我们将地面真实边界框 $B_{j_2}$ 分配给锚框 $A_{i_2}$ 并丢弃 ${i_2}^\mathrm{th}$ 行和矩阵 ${j_2}^\mathrm{th}$ 列中的所有元素。
1. 此时，矩阵 $\mathbf{X}$ 中两行和两列中的元素已被丢弃。我们继续直到丢弃矩阵 $\mathbf{X}$ 中 $n_b$ 列中的所有元素。目前，我们已经为 $n_b$ 个锚盒中的每个分配了一个地面真相边界框。
1. 只能穿过剩下的 $n_a - n_b$ 锚盒。例如，给定任何锚框 $A_i$，找到地面真实边界框 $A_i$，在 $i^\mathrm{th}$ 行矩阵 $\mathbf{X}$ 中找到最大的 IOU $A_i$ 的地面真实边界框 $A_i$，只有当此 IOU 大于预定义的阈值时，才将 $B_j$ 分配给 $A_i$。

让我们用一个具体的例子来说明上述算法。如 :numref:`fig_anchor_label`（左）所示，假设矩阵 $\mathbf{X}$ 中的最大值为 $x_{23}$，我们将地面真实边界框 $B_3$ 分配给锚框 $A_2$。然后，我们丢弃矩阵第 2 行和第 3 列中的所有元素，在剩余元素（阴影区域）中找到最大的 $x_{71}$，然后将地面真相边界框 $B_1$ 分配给锚框 $A_7$。接下来，如 :numref:`fig_anchor_label`（中）所示，丢弃矩阵第 7 行和第 1 列中的所有元素，在剩余元素（阴影区域）中找到最大的 $x_{54}$，然后将地面真值边界框 $B_4$ 分配给锚框 $A_5$。最后，如 :numref:`fig_anchor_label`（右）所示，丢弃矩阵第 5 行和第 4 列中的所有元素，在剩余元素（阴影区域）中找到最大的 $x_{92}$，然后将地面真值边界框 $B_2$ 分配给锚框 $A_9$。之后，我们只需要遍历剩余的锚盒 $A_1, A_3, A_4, A_6, A_8$，然后根据阈值确定是否为它们分配地面真实边界框。 

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

此算法在以下 `assign_anchor_to_bbox` 函数中实现。

```{.python .input}
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= 0.5)[0]
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### 标记类和偏移

现在我们可以为每个锚框标记分类和偏移量。假设一个锚框 $A$ 被分配了一个地面真相边界框 $B$。一方面，锚箱 $A$ 的类将被标记为 $B$。另一方面，锚框 $A$ 的偏移量将根据 $B$ 和 $A$ 中心坐标之间的相对位置以及这两个框之间的相对大小进行标记。鉴于数据集中不同框的位置和大小不同，我们可以对那些可能导致更均匀分布的偏移量的相对位置和大小应用变换，从而更容易适应。在这里我们描述一种常见的转变鉴于中心坐标为 $A$ 和 $B$，即 $(x_a, y_a)$ 和 $(x_b, y_b)$，其宽度分别为 $w_a$ 和 $w_b$，高度分别为 $h_a$ 和 $h_b$。我们可能会将 $A$ 的偏移标签为 

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$

其中常量的默认值是 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$ 和 $\sigma_w=\sigma_h=0.2$。这种转换在下面的 `offset_boxes` 函数中实现。

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

如果没有为锚框分配地面真实边界框，我们只需将锚点框的类标记为 “背景”。类为背景的锚框通常被称为 * 负面 * 锚框，其余的被称为 * 正面 * 锚框。我们使用地面真值边界框（`labels` 参数）实现以下 `multibox_target` 函数来标注锚盒的分类和偏移量（`anchors` 参数）。此函数将后台类设置为零，然后将新类的整数索引递增一。

```{.python .input}
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 一个例子

让我们通过一个具体的例子来说明锚箱标签。我们在加载的图像中为狗和猫定义了地面真实边界框，其中第一个元素是类（0 代表狗，1 代表猫），其余四个元素是左上角和右下角的 $(x, y)$ 轴坐标（范围介于 0 和 1 之间）。我们还构建了五个锚框，用左上角和右下角的坐标进行标记：$A_0, \ldots, A_4$（索引从 0 开始）。然后我们在图像中绘制这些地面真相边界框和锚框。

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

使用上面定义的 `multibox_target` 函数，我们可以根据狗和猫的地面真相边界框标注这些锚箱的分类和偏移量。在此示例中，背景、狗和猫类的索引分别为 0、1 和 2。下面我们添加了锚框和地面真实边界框示例的维度。

```{.python .input}
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

返回的结果中有三个项目，所有这些项目都是张量格式。第三个项目包含输入锚框的标记类。 

让我们根据图像中的锚框和地面真实边界框位置来分析下面返回的类标签。首先，在所有对锚盒和地面真实边界盒中，锚盒 $A_4$ 的 iOU 和猫的地面真实边界框是最大的。因此，$A_4$ 的类被标记为猫。取出包含 $A_4$ 或猫的地面真实边界盒的对，其余的是锚盒 $A_1$ 和狗的地面真实边界盒中最大的 iOU。因此，$A_1$ 的班级被标记为狗。接下来，我们需要遍历剩下的三个未标记的锚盒：$A_0$、$A_2$ 和 $A_3$。对于 $A_0$，拥有最大 iOU 的地面真实边界框的类是狗，但 iOU 低于预定义的阈值 (0.5)，因此该类被标记为背景；$A_2$，拥有最大 iOU 的地面真实边界框的类是猫，iOU 超过阈值，所以类被标记为猫；对于 $A_3$，具有最大 iOU 的地面真实边界框的类是猫，但值低于阈值，因此该类被标记为背景。

```{.python .input}
#@tab all
labels[2]
```

返回的第二个项目是形状的遮罩变量（批量大小，锚框数的四倍）。遮罩变量中的每四个元素对应于每个锚点框的四个偏移值。由于我们不关心后台检测，这个负类的偏移不应影响目标函数。通过元素乘法，掩码变量中的零将在计算目标函数之前过滤掉负类偏移量。

```{.python .input}
#@tab all
labels[1]
```

返回的第一个项目包含为每个锚点框标记的四个偏移值。请注意，负类锚框的偏移量被标记为零。

```{.python .input}
#@tab all
labels[0]
```

## 使用非最大抑制预测边界框
:label:`subsec_predicting-bounding-boxes-nms`

在预测期间，我们为图像生成多个锚框，并预测每个定位框的类和偏移量。因此，根据带有预测偏移量的锚框获得 * 预测的边界框 *。下面我们实现了 `offset_inverse` 函数，该函数将锚点和偏移预测作为输入，并应用逆偏移变换来返回预测的边界框坐标。

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

当有许多锚框时，可能会输出许多相似的（具有明显重叠）的预测边界框用于围绕同一对象。为了简化输出，我们可以使用 * 非最大抑制 * (NMS) 合并属于同一对象的类似预测边界框。 

以下是非最大抑制的工作方式。对于预测边界框 $B$，对象检测模型计算每个类的预测可能性。以 $p$ 为预测的最大可能性，与此概率对应的类别是 $B$ 的预测类别。具体来说，我们将 $p$ 称为预测边界框 $B$ 的 * 信心 *（分数）。在同一张图像中，所有预测的非背景边界框都按置信度降序排序，以生成列表 $L$。然后我们通过以下步骤操作排序列表 $L$： 

1. 选择 $L$ 以最高置信度为基准的预测边界框 $B_1$ 作为基准，然后删除所有具有 $B_1$ iOU 超过 $L$ 预定阈值 $\epsilon$ 的非基础预测边界框。此时，$L$ 将预测的边界框保持最高的可信度，但丢弃了与它太类似的其他边界框。简而言之，那些具有 * 非最大 * 置信度分数的人被 * 抑制 *。
1. 选择第二高置信度从 $L$ 的预测边界框 $B_2$ 作为另一个基准，然后移除所有非基础预测边界框，其 IOU 的 $B_2$ 超过 $\epsilon$ 的 $\epsilon$。
1. 重复上述过程，直到 $L$ 中的所有预测边界框都被用作基准。目前，$L$ 中任何一对预测边界盒的 IoU 都低于 $\epsilon$ 的门槛；因此，没有一对彼此太相似。 
1. 输出列表中的所有预测边界框 $L$。

以下 `nms` 函数按降序对置信度分数进行排序并返回其指数。

```{.python .input}
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

我们定义以下 `multibox_detection` 以将非最大抑制应用于预测边界框。如果你发现实现有点复杂，请不要担心：我们将在实施之后立即用一个具体的示例来展示它是如何工作的。

```{.python .input}
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

现在让我们将上述实现应用到带有四个锚框的具体示例中。为简单起见，我们假设预测的偏移都是零。这意味着预测的边界框是锚框。对于背景、狗和猫中的每个课程，我们还定义了它的预测可能性。

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

我们可以在图像上自信地绘制这些预测的边界框。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

现在我们可以调用 `multibox_detection` 函数来执行非最大抑制，其中阈值设置为 0.5。请注意，我们在张量输入中添加了示例的维度。 

我们可以看到返回结果的形状是（批量大小，锚框的数量，6）。最内层维度中的六个元素提供了同一预测边界框的输出信息。第一个元素是预测的类索引，从 0 开始（0 代表狗，1 是猫）。值-1 表示在非最大抑制情况下背景或移除。第二个要素是预测的边界框的信心。其余四个元素分别是预测边界框左上角和右下角的 $(x, y)$ 轴坐标（范围介于 0 和 1 之间）。

```{.python .input}
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

删除-1 类的预测边界框后，我们可以输出由非最大抑制保存的最终预测边界框。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

实际上，我们甚至可以在执行非最大抑制之前以较低的置信度删除预测的边界框，从而减少此算法中的计算。例如，我们也可以对非最大抑制的输出进行后处理，例如，只能通过保持结果对最终输出的信心更高。 

## 摘要

* 我们以图像的每个像素为中心生成不同形状的锚框。
* 联盟交叉点（IoU），也被称为 Jaccard 指数，衡量两个边界框的相似性。这是他们的交叉口区域与联盟区的比率。
* 在训练集中，我们需要为每个锚箱提供两种类型的标签。一个是与锚框相关的对象的类，另一个是地面真实边界框相对于锚框的偏移量。
* 在预测期间，我们可以使用非最大抑制（NMS）来移除类似的预测边界框，从而简化输出。

## 练习

1. 在 `multibox_prior` 函数中更改 `sizes` 和 `ratios` 的值。生成的锚框有什么变化？
1. 构建和可视化两个 iOU 为 0.5 的边界框。它们如何彼此重叠？
1. 在 :numref:`subsec_labeling-anchor-boxes` 和 :numref:`subsec_predicting-bounding-boxes-nms` 中修改变量 `anchors`。结果如何变化？
1. 非最大抑制是一种贪婪的算法，它通过 * 移除 * 来抑制预测的边界框。这些被删除的一些可能实际上是有用的吗？如何修改这个算法来抑制 * 软性 *？你可以参考 Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`。
1. 不是手工制作，可以学习非最大限度的抑制吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:
