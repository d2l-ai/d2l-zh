# 样式迁移

如果你是一位摄影爱好者，你也许接触过滤镜。它能改变照片的颜色样式，从而使风景照更加锐利或者令人像更加美白。但一个滤镜通常只能改变照片的某个方面。如果要照片达到理想中的样式，这经常需要尝试大量不同的组合：其复杂程度不亚于模型调参。

在本节中，我们将介绍如何使用卷积神经网络自动将某图像中的样式应用在另一图像之上，即样式迁移（style transfer）[1]。这里我们需要两张输入图像，一张是内容图像，另一张是样式图像：我们将使用神经网络修改内容图像使得其样式接近样式图像。图9.12中的内容图像为本书作者在西雅图郊区的雷尼尔山国家公园（Mount Rainier National Park）拍摄的风景照，而样式图像则是一副主题为秋天橡树的油画。最终输出的合成图像在保留了内容图像中物体主体形状的情况下应用了样式图像的油画笔触，同时也让整体颜色更加鲜艳。

![输入内容图像和样式图像，输出样式迁移后的合成图像。](../img/style-transfer.svg)

## 方法

图9.13用一个例子来阐述基于卷积神经网络的样式迁移方法。首先，我们初始化合成图像，例如将其初始化成内容图像。该合成图像是样式迁移过程中唯一需要更新的变量，即样式迁移所需迭代的模型参数。然后，我们选择一个预训练的卷积神经网络来抽取图像的特征，其中的模型参数在训练中无需更新。深度卷积神经网络凭借多个神经层逐级抽取图像的特征。我们可以选择其中某些层的输出作为内容特征或样式特征。以图9.13为例，这里选取的预训练的神经网络含有三个卷积层，其中第二层则输出图像的内容特征，而第一层和第三层的输出被作为图像的样式特征。接下来，我们通过正向传播（实线箭头方向）计算样式迁移的损失函数，并通过反向传播（虚线箭头方向）迭代模型参数，即不断更新合成图像。样式迁移常用的损失函数由三部分组成：内容损失（content loss）使合成图像与内容图像在内容特征上接近，样式损失（style loss）令合成图像与样式图像在样式特征上接近，而总变差损失（total variation loss）则有助于减少合成图像中的噪点。最后，当模型训练结束时，我们输出样式迁移的模型参数，即得到最终的合成图像。

![基于卷积神经网络的样式迁移。实线箭头和虚线箭头分别表示正向传播和反向传播。](../img/neural-style.svg)

下面，我们通过实验来进一步了解样式迁移的技术细节。

## 读取内容图像和样式图像

首先，我们分别读取内容图像和样式图像。从打印出的图像坐标轴可以看出，它们的尺寸并不一样。

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import model_zoo, nn
import time

gb.set_figsize()
content_img = image.imread('../img/rainier.jpg')
gb.plt.imshow(content_img.asnumpy());
```

```{.python .input  n=2}
style_img = image.imread('../img/autumn_oak.jpg')
gb.plt.imshow(style_img.asnumpy());
```

## 预处理和后处理图像

下面定义图像的预处理和后处理函数。预处理函数`preprocess`对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式。后处理函数`postprocess`则将输出图像中的像素值还原回标准化之前的值。由于图像打印函数要求每个像素的浮点数值在0到1之间，我们使用`clip`函数对小于0和大于1的值分别取0和1。

```{.python .input  n=3}
rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)
```

## 抽取特征

我们使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征 [1]。

```{.python .input  n=4}
pretrained_net = model_zoo.vision.vgg19(pretrained=True)
```

为了抽取图像的内容特征和样式特征，我们可以选择VGG网络中某些层的输出。一般来说，越靠近输入层的输出越容易抽取图像的细节信息，反之则越容易抽取图像的全局信息。为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，也称内容层，来输出图像的内容特征。我们还从VGG中选择不同层的输出来匹配局部和全局的样式：这些层也叫样式层。在[“使用重复元素的网络（VGG）”](../chapter_convolutional-neural-networks/vgg.md)一节中我们曾介绍过，VGG网络使用了5个卷积块。实验中，我们选择第四卷积块的最后一个卷积层作为内容层，以及每个卷积块的第一个卷积层作为样式层。这些层的索引可以通过打印`pretrained_net`实例来获取。

```{.python .input  n=5}
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

在抽取特征时，我们只需要用到VGG从输入层到最靠近输出层的内容层或样式层之间的所有层。下面构建一个新的网络`net`，它只保留需要用到的VGG的所有层。我们将使用`net`来抽取特征。

```{.python .input  n=6}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

给定输入`X`，如果简单调用前向计算`net(X)`只能获得最后一层的输出。由于我们还需要中间层的输出，因此这里我们逐层计算，并保留内容层和样式层的输出。

```{.python .input  n=7}
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

下面定义两个函数：`get_contents`函数对内容图像抽取内容特征，而`get_styles`函数则对样式图像抽取样式特征。因为在训练时无需改变预训练的VGG的模型参数，所以我们可以在训练开始之前就提取出内容图像的内容特征，以及样式图像的样式特征。由于合成图像是样式迁移所需迭代的模型参数，我们只能在训练过程中通过调用`extract_features`函数来抽取合成图像的内容和样式特征。

```{.python .input  n=8}
def get_contents(image_shape, ctx):
    content_X = preprocess(content_img, image_shape).copyto(ctx)
    content_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, content_Y

def get_styles(image_shape, ctx):
    style_X = preprocess(style_img, image_shape).copyto(ctx)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y
```

## 损失函数

在训练时，我们需要定义如何比较合成图像和内容图像的内容层输出（内容损失函数），以及比较和样式图像的样式层输出（样式损失函数）。内容损失函数可以使用回归用的均方误差。

```{.python .input  n=9}
def content_loss(Y_hat, Y):
    return (Y_hat - Y).square().mean()
```

对于样式，我们可以简单将它看成是像素点在每个通道的统计分布。例如要匹配两张图像的样式，我们可以匹配这两张图像在RGB这三个通道上的直方图。更一般的，假设卷积层的输出格式是$c \times h \times w$，既（通道，高，宽）。那么我们可以把它变形成 $c \times hw$ 的二维数组，并将它看成是一个维度为$c$ 的随机变量采样到的 $hw$ 个点。所谓的样式匹配就是使得两个 $c$ 维随机变量统计分布一致。

匹配统计分布常用的做法是冲量匹配，就是说使得他们有一样的均值，协方差，和其他高维的冲量。为了计算简单起见，我们只匹配二阶信息，即协方差。下面定义如何计算协方差矩阵，

```{.python .input  n=10}
def gram(X):
    c, n = X.shape[1], X.size // X.shape[1]
    Y = X.reshape((c, n))
    return nd.dot(Y, Y.T) / (c * n)
```

和对应的损失函数，这里假设样式图像的样式特征协方差已经预先计算好了。

```{.python .input  n=11}
def style_loss(Y_hat, gram_Y):
    return (gram(Y_hat) - gram_Y).square().mean()
```

当我们使用靠近输出层的神经层输出来匹配时，经常可以观察到学到的合成图像里面有大量高频噪音，即有特别亮或者暗的颗粒像素。一种常用的降噪方法是总变差降噪（total variation denoising）。假设 $x_{i,j}$ 表示像素 $(i,j)$的值，总变差损失使得邻近的像素值相似：

$$\sum_{i,j} \left|x_{i,j} - x_{i+1,j}\right| + \left|x_{i,j} - x_{i,j+1}\right|.$$

```{.python .input  n=12}
def tv_loss(Y_hat):
    return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                  (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())
```

训练中我们将上述三个损失函数加权求和。通过调整权重值我们可以控制学到的图像是否保留更多样式，更多内容，还是更加干净。

```{.python .input  n=13}
style_channels = [net[l].weight.shape[0] for l in style_layers]
content_weight, style_weight, tv_weight = 1, 1e3, 10
```

```{.python .input  n=14}
def compute_loss(X, content_Y_hat, style_Y_hat, content_Y, style_Y_gram):
    # 分别计算内容、样式和噪音损失。
    content_l = [content_loss(y_hat, y) * content_weight
                 for y_hat, y in zip(content_Y_hat, content_Y)]
    style_l = [style_loss(y_hat, y) * style_weight
               for y_hat, y in zip(style_Y_hat, style_Y_gram)] 
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和。
    l = nd.add_n(*style_l) + nd.add_n(*content_l) + tv_l
    return content_l, style_l, tv_l, l
```

## 定义输出图像

```{.python .input  n=15}
class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input  n=16}
def get_inits(X, ctx, lr, style_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    style_Y_gram = [gram(Y) for Y in style_Y]
    return gen_img(), style_Y_gram, trainer
```

## 训练

这里的训练跟前面章节的主要不同在于我们只对输入`x`进行更新。此外我们将`x`的梯度除以它的绝对平均值来降低对学习率的敏感度，而且每隔一定的批量我们减小一次学习率。

```{.python .input  n=17}
def train(X, content_Y, style_Y, ctx, lr, max_epochs, lr_decay_epoch):
    X, style_Y_gram, trainer = get_inits(X, ctx, lr, style_Y)
    for i in range(max_epochs):
        start = time.time()
        with autograd.record():
            # 对 X 抽取样式和内容特征。
            content_Y_hat, style_Y_hat = extract_features(
                X, content_layers, style_layers)
            content_l, style_l, tv_l, l = compute_loss(
                X, content_Y_hat, style_Y_hat, content_Y, style_Y_gram)
        l.backward()     
        trainer.step(1)
        # 如果不加的话会导致每 50 轮迭代才同步一次，可能导致过大内存使用。
        nd.waitall()
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, nd.add_n(*content_l).asscalar(),
                     nd.add_n(*style_l).asscalar(), tv_l.asscalar(),
                     time.time() - start))
        if i % lr_decay_epoch == 0 and i != 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            print('change lr to %.1e' % trainer.learning_rate)
    return X
```

现在我们可以真正开始训练了。首先将图像调整到高300宽200，这样能使训练更加快速。合成图像的初始值设成了内容图像，使得初始值能尽可能接近训练输出，从而加速收敛。

```{.python .input  n=18}
ctx, image_shape = gb.try_gpu(), (300, 200)
net.collect_params().reset_ctx(ctx)
content_X, content_Y = get_contents(image_shape, ctx)
style_X, style_Y = get_styles(image_shape, ctx)
output = train(content_X, content_Y, style_Y, ctx, 0.01, 500, 200)
```

因为使用了内容图像作为初始值，所以一开始内容误差远小于样式误差。随着迭代的进行样式误差迅速减少，最终它们值在相近的范围。下面我们将训练好的合成图像保存下来。

```{.python .input  n=19}
gb.plt.imsave('../img/neural-style-1.png', postprocess(output).asnumpy())
```

![$300 \times 200$ 尺寸的合成图像。](../img/neural-style-1.png)

可以看到图9.14中的合成图像保留了样式图像的风景物体，同时借鉴了样式图像的色彩。由于图像尺寸较小，所以细节上比较模糊。下面我们在更大的$1200 \times 800$的尺寸上训练，希望可以得到更加清晰的合成图像。为了加速收敛，我们将训练到的合成图像高宽放大4倍来作为初始值。

```{.python .input  n=20}
image_shape = (1200, 800)
content_X, content_Y = get_contents(image_shape, ctx)
style_X, style_Y = get_styles(image_shape, ctx)
X = preprocess(postprocess(output) * 255, image_shape)
output = train(X, content_Y, style_Y, ctx, 0.01, 300, 100)
gb.plt.imsave('../img/neural-style-2.png', postprocess(output).asnumpy())
```

可以看到这一次由于初始值离最终输出更近使得收敛更加迅速。但同时由于图像尺寸更大，每一次迭代需要花费更多的时间和内存。

![$1200 \times 800$ 尺寸的合成图像。](../img/neural-style-2.png)

从训练得到的图9.15中的可以看到它保留了更多的细节，里面不仅有大块的类似样式图像的油画色彩块，色彩块里面也有细微的纹理。

## 小结

* 通过匹配神经网络的中间层输出可以有效的融合不同图像的内容和样式。

## 练习

* 选择不同的内容和样式层。
* 使用不同的损失权重来得到更偏向内容或样式或平滑的输出。
* 一个得到更加干净的合成图的办法是使用更大的尺寸。
* 换别的样式和内容图像试试。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/3273)

![](../img/qr_neural-style.svg)

## 参考文献

[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2414-2423).
