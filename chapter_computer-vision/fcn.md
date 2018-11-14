# 全卷积网络（FCN）

我们在上节介绍了，语义分割对图像中的每个像素预测类别。全卷积网络（fully convolutional network，简称FCN）采用卷积神经网络实现了从图像像素到像素类别的变换。与之前介绍的卷积神经网络有所不同，全卷积网络通过转置卷积（transposed convolution）层将中间层特征图的高和宽变换回输入图像的尺寸，从而令预测结果与输入图像在空间维（高和宽）上一一对应：给定空间维上的位置，通道维的输出即该位置对应像素的类别预测。

我们先导入实验所需的包或模块，然后解释什么是转置卷积层。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import numpy as np
import sys
```

## 转置卷积层

顾名思义，转置卷积层得名于矩阵的转置操作。事实上，卷积运算还可以通过矩阵乘法来实现。在下面的例子中，我们定义高和宽分别为4的输入`X`，以及高和宽分别为3的卷积核`K`。打印二维卷积运算的输出以及卷积核。可以看到，输出的高和宽分别为2。

```{.python .input}
X = nd.arange(1, 17).reshape((1, 1, 4, 4))
K = nd.arange(1, 10).reshape((1, 1, 3, 3))
conv = nn.Conv2D(channels=1, kernel_size=3)
conv.initialize(init.Constant(K))
conv(X), K
```

下面我们将卷积核`K`改写成含有大量零元素的稀疏矩阵`W`，即权重矩阵。权重矩阵的形状为（4，16），其中的非零元素来自卷积核`K`中的元素。将输入`X`逐行连结，得到长度为16的向量。然后将`W`与向量化的`X`做矩阵乘法，得到长度为4的向量。对其变形后，我们可以得到和上面卷积运算相同的结果。可见，我们在这个例子中使用矩阵乘法实现了卷积运算。

```{.python .input}
W, k = nd.zeros((4, 16)), nd.zeros(11)
k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)), W
```

现在我们从矩阵乘法的角度来描述卷积运算。设输入向量为$\boldsymbol{x}$，权重矩阵为$\boldsymbol{W}$，卷积的前向计算函数的实现可以看作将函数输入乘以权重矩阵，并输出向量$\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x}$。我们知道，反向传播需要依据链式法则。由于$\nabla_\boldsymbol{x} \boldsymbol{y} = \boldsymbol{W}^\top$，卷积的反向传播函数的实现可以看作将函数输入乘以转置后的权重矩阵$\boldsymbol{W}^\top$。而转置卷积层正是交换了卷积层的前向计算函数与反向传播函数：这两个函数可以看作将函数输入向量分别乘以$\boldsymbol{W}^\top$和$\boldsymbol{W}$。

不难想象，转置卷积层可以用来交换卷积层输入和输出的形状。让我们继续用矩阵乘法描述卷积。设权重矩阵是形状为$4\times16$的矩阵，对于长度为16的输入向量，卷积前向计算输出长度为4的向量。假如输入向量的长度为4，转置权重矩阵的形状为$16\times4$，那么转置卷积层将输出长度为16的向量。在模型设计中，转置卷积层常用于将较小的特征图变换为更大的特征图。在全卷积网络中，当输入是高和宽较小的特征图时，转置卷积层可以用来将高和宽放大到输入图像的尺寸。

我们来看一个例子。构造一个卷积层`conv`，并设输入`X`的形状为（1，3，64，64）。卷积输出`Y`的通道数增加到10，但高和宽分别缩小了一半。

```{.python .input  n=3}
conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
conv.initialize()

X = nd.random.uniform(shape=(1, 3, 64, 64))
Y = conv(X)
Y.shape
```

下面我们通过创建`Conv2DTranspose`实例来构造转置卷积层`conv_trans`。这里我们设`conv_trans`的卷积核形状、填充以及步幅与`conv`中的相同，并设输出通道数为3。当输入为卷积层`conv`的输出`Y`时，转置卷积层输出与卷积层输入的高和宽相同：转置卷积层将特征图的高和宽分别放大了2倍。

```{.python .input  n=4}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize()
conv_trans(Y).shape
```

## FCN模型

FCN的核心思想是将一个卷积网络的最后全连接输出层替换成转置卷积层来获取对每个输入像素的预测。具体来说，它去掉了过于损失空间信息的全局池化层，并将最后的全连接层替换成输出通道是原全连接层输出大小的$1\times 1$卷积层，最后接上转置卷积层来得到需要形状的输出。图9.11描述了FCN模型。

![FCN模型。](../img/fcn.svg)

下面我们基于ResNet-18来创建FCN。首先我们下载一个预先训练好的模型，并打印其最后的数个神经层。

```{.python .input  n=5}
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-4:], pretrained_net.output
```

可以看到`feature`模块最后两层是`GlobalAvgPool2D`和`Flatten`，在FCN里均不需要，`output`模块里的全连接层也需要舍去。下面我们定义一个新的网络，它复制`feature`里除去最后两层的所有神经层以及权重。

```{.python .input  n=6}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

给定高宽为224的输入，`net`的输出将减少为输入高宽的$1/32$。

```{.python .input  n=7}
X = nd.random.uniform(shape=(1, 3, 224, 224))
net(X).shape
```

为了使得输出跟输入有同样的高宽，我们构建一个步幅为32的转置卷积层，卷积核的窗口高宽设置成步幅的2倍，并补充适当的填充。在转置卷积层之前，我们加上$1\times 1$卷积层来将通道数从512降到标注类别数，对Pascal VOC数据集来说是21。

```{.python .input  n=8}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,
                           strides=32))
```

## 模型初始化

模型`net`中的最后两层需要对权重进行初始化，通常我们会使用随机初始化。但新加入的转置卷积层的功能有些类似于将输入调整到更大的尺寸。在图像处理里面，我们可以通过有适当卷积核的卷积运算符来完成这个操作。常用的包括双线性插值核，以下函数构造核权重。

```{.python .input  n=9}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)
```

接下来我们构造一个步幅为2的转置卷积层，将其权重初始化为双线性插值核。

可以看到这个转置卷积层的前向函数的效果是将输入图像高宽扩大2倍。

```{.python .input  n=11}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))

img = image.imread('../img/catdog.jpg')
X = img.astype('float32').transpose((2, 0, 1)).expand_dims(axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose((1, 2, 0))
```

```{.python .input}
gb.set_figsize()
print('input image shape:', img.shape)
gb.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
gb.plt.imshow(out_img.asnumpy());
```

下面对`net`的最后两层进行初始化。其中$1\times 1$卷积层使用Xavier，转置卷积层则使用双线性插值核。

```{.python .input  n=12}
trans_conv_weights = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(trans_conv_weights))
net[-2].initialize(init=init.Xavier())
```

## 读取数据

我们使用较大的输入图像尺寸，其值选成了32的倍数。数据的读取方法已在上一节描述。

```{.python .input  n=13}
input_shape, batch_size, colormap2label = (320, 480), 32, nd.zeros(256**3)
for i, cm in enumerate(gb.VOC_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
voc_dir = gb.download_voc_pascal(data_dir='../data')

num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = gdata.DataLoader(
    gb.VOCSegDataset(True, input_shape, voc_dir, colormap2label), batch_size,
    shuffle=True, last_batch='discard', num_workers=num_workers)
test_iter = gdata.DataLoader(
    gb.VOCSegDataset(False, input_shape, voc_dir, colormap2label), batch_size,
    last_batch='discard', num_workers=num_workers)
```

## 训练

这时候我们可以开始训练了。因为我们使用转置卷积层的通道来预测像素的类别，所以softmax是作用在通道这个维度（维度1）上的。于是，我们在`SoftmaxCrossEntropyLoss`里加入了额外的`axis=1`选项。

```{.python .input  n=12}
ctx = gb.try_all_gpus()
loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1,
                                                      'wd': 1e-3})
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=5)
```

## 预测

预测一张新图像时，我们只需要将其归一化并转成卷积网络需要的4D格式。

```{.python .input  n=13}
def predict(img):
    x = test_iter._dataset.normalize_image(img)
    x = x.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmax(net(x.as_in_context(ctx[0])), axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))
```

同时我们根据每个像素预测的类别找出其RGB颜色以便画图。

```{.python .input  n=14}
def label2image(pred):
    colormap = nd.array(gb.VOC_COLORMAP, ctx=ctx[0], dtype='uint8')
    x = pred.astype('int32')
    return colormap[x, :]
```

现在我们读取前几张测试图像并对其进行预测。

```{.python .input  n=15}
test_images, test_labels = gb.read_voc_images(is_train=False)
n, imgs = 5, []
for i in range(n):
    X = test_images[i]
    pred = label2image(predict(X))
    imgs += [X, pred, test_labels[i]]
gb.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n);
```

## 小结

* FCN通过转置卷积层来为每个像素预测类别。

## 练习

* 用矩阵乘法来实现卷积运算是否高效？为什么？
* 试着改改最后的转置卷积层的参数设定。
* 看看双线性插值初始化是不是必要的。
* 试着改改训练参数来使得收敛更好些。
* FCN论文中提到了不只是使用主体卷积网络输出，还可以考虑其中间层的输出 [1]。试着实现这个想法。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/3041)

![](../img/qr_fcn.svg)


## 参考文献

[1] Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
