# 图片增广

在[“深度卷积神经网络：AlexNet”](../chapter_convolutional-neural-networks/alexnet.md)小节里我们提到过，大规模数据集是成功使用深度网络的前提。图片增广（image augmentation）技术通过对训练图片做一系列随机变化，来产生相似但又有不同的训练样本，从而扩大训练数据集规模。图片增广的另一种解释是，通过对训练样本做一些随机变形，可以降低模型对某些属性的依赖，从而提高泛化能力。例如我们可以对图片进行不同的裁剪，使得感兴趣的物体出现在不同的位置中，从而使得模型减小对物体出现位置的依赖性。也可以调整亮度色彩等因素来降低模型对色彩的敏感度。在AlexNet的成功中，图片增广技术功不可没。本小节我们将讨论这个在计算机视觉里被广泛使用的技术。

首先，导入本节实验所需的包或模块。

```{.python .input  n=21}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd 
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
from time import time
```

## 常用增广方法

我们先读取一张$400\times 500$的图片作为样例。

```{.python .input  n=22}
gb.set_figsize()
img = image.imread('../img/cat1.jpg')
gb.plt.imshow(img.asnumpy())
```

下面定义绘图函数`show_images`。该函数也被定义在`gluonbook`包中供后面章节调用。

```{.python .input  n=23}
def show_images(imgs, num_rows, num_cols, scale=2):                                                                              
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = gb.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes
```

因为大部分的增广方法都有一定的随机性。接下来我们定义一个辅助函数，它对输入图片`img`运行多次增广方法`aug`并显示所有结果。

```{.python .input  n=24}
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
```

### 变形

左右翻转图片通常不改变物体的类别，它是最早也是最广泛使用的一种增广。下面我们使用transform模块里的`RandomFlipLeftRight`类来实现按0.5的概率左右翻转图片：

```{.python .input  n=25}
apply(img, gdata.vision.transforms.RandomFlipLeftRight())
```

上下翻转不如水平翻转通用，但是至少对于样例图片，上下翻转不会造成识别障碍。

```{.python .input  n=26}
apply(img, gdata.vision.transforms.RandomFlipTopBottom())
```

我们使用的样例图片里，猫在图片正中间，但一般情况下可能不是这样。[“池化层”](../chapter_convolutional-neural-networks/pooling.md)一节里我们解释了池化层能弱化卷积层对目标位置的敏感度，另一方面我们可以通过对图片随机剪裁来让物体以不同的比例出现在不同位置。

下面代码里我们每次随机裁剪一片面积为原面积10%到100%的区域，其宽和高的比例在0.5和2之间，然后再将高宽缩放到200像素大小。

```{.python .input  n=27}
shape_aug = gdata.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 颜色变化

另一类增广方法是变化颜色。我们可以从四个维度改变图片的颜色：亮度、对比、饱和度和色相。在下面的例子里，我们将随机亮度改为原图的50%到150%。

```{.python .input  n=28}
apply(img, gdata.vision.transforms.RandomBrightness(0.5))
```

类似的，我们可以修改色相。

```{.python .input  n=29}
apply(img, gdata.vision.transforms.RandomHue(0.5))
```

或者用使用`RandomColorJitter`来一起使用。

```{.python .input  n=30}
color_aug = gdata.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 使用多个增广

实际应用中我们会将多个增广叠加使用。`Compose`类可以将多个增广串联起来。

```{.python .input  n=31}
augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

## 使用图片增广来训练

接下来我们来看一个将图片增广应用在实际训练中的例子，并比较其与不使用时的区别。这里我们使用CIFAR-10数据集，而不是之前我们一直使用的Fashion-MNIST。原因在于Fashion-MNIST中物体位置和尺寸都已经归一化了，而CIFAR-10中物体颜色和大小区别更加显著。下面我们展示CIFAR-10中的前32张训练图片。

```{.python .input  n=32}
show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8);
```

我们通常将图片增广用在训练样本上，但是在预测的时候并不使用随机增广。这里我们仅仅使用最简单的随机水平翻转。此外，我们使用`ToTensor`变换来将图片转成MXNet需要的格式，即格式为（批量，通道，高，宽）以及类型为32位浮点数。

```{.python .input  n=33}
train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
])

test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
])
```

接下来我们定义一个辅助函数来方便读取图片并应用增广。Gluon的数据集提供`transform_first`函数来对数据里面的第一项（数据一般有图片和标签两项）来应用增广。另外图片增广将增加计算复杂度，这里使用4个进程来加速读取（暂不支持 Windows 操作系统）。

```{.python .input  n=34}
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size):
    return gdata.DataLoader(
        gdata.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
```

### 使用多GPU训练模型

我们在CIFAR-10数据集上训练[“残差网络：ResNet”](../chapter_convolutional-neural-networks/resnet.md)一节介绍的ResNet-18模型。我们将应用[“多GPU计算的Gluon实现”](../chapter_computational-performance/multiple-gpus-gluon.md)一节中介绍的方法，使用多GPU训练模型。

首先，我们定义`try_all_gpus`函数，从而能够使用所有可用的GPU。

```{.python .input  n=35}
def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes
```

然后，我们定义`evaluate_accuracy`函数评价模型的分类准确率。与[“Softmax回归的从零开始实现”](../chapter_deep-learning-basics/softmax-regression-scratch.md)和[“卷积神经网络（LeNet）”](../chapter_convolutional-neural-networks/lenet.md)两节中描述的`evaluate_accuracy`函数不同，当`ctx`包含多个GPU时，这里定义的函数通过辅助函数`_get_batch`将小批量数据样本划分并复制到各个GPU上。

```{.python .input  n=36}
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    # 当 ctx 包含多个GPU时，划分小批量数据样本并复制到各个 GPU 上。
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1)==y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n
```

接下来，我们定义`train`函数使用多GPU训练并评价模型。

```{.python .input  n=37}
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        start = time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch, train_l_sum / n, train_acc_sum / m, test_acc,
                 time() - start))
```

现在，我们可以定义函数使用图片增广来训练模型了。

```{.python .input  n=38}
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size = 256
    ctx = try_all_gpus()
    net = gb.resnet18(10)
    net.initialize(ctx=ctx, init=init.Xavier())
    # 这里使用了 Adam 优化算法。
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=15)
```

我们先观察使用了图片增广的结果。

```{.python .input  n=39}
train_with_data_aug(train_augs, test_augs)
```

作为对比，下面我们尝试不使用图片增广。

```{.python .input  n=40}
train_with_data_aug(test_augs, test_augs)
```

可以看到，即使添加了简单的随机翻转也会对训练产生一定的影响。图片增广通常会使训练准确率变低，但有可能提高测试准确率。

本节中描述的`try_all_gpus`、`evaluate_accuracy`和`train`函数被定义在`gluonbook`包中供后面章节调用。

## 小结

* 图片增广基于现有训练数据生成大量随机图片来有效避免过拟合。

## 练习

* 尝试在CIFAR-10训练中增加不同的增广方法。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1666)

![](../img/qr_image-augmentation.svg)
