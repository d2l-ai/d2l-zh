# 神经风格转移

如果你是摄影爱好者，你可能会熟悉滤镜。它可以改变照片的颜色样式，使风景照片变得更清晰，或者肖像照片使皮肤变白。但是，一个滤镜通常只改变照片的一个方面。要将理想的风格应用于照片，你可能需要尝试许多不同的滤镜组合。这个过程与调整模型的超参数一样复杂。 

在本节中，我们将利用 CNN 的分层表示方式自动将一张图片的样式应用到另一张图片，即 * 样式转移 * :cite:`Gatys.Ecker.Bethge.2016`。此任务需要两张输入图片：一张是 * 内容图片 *，另一张是 * 样式图片 *。我们将使用神经网络来修改内容图像，使其在风格上接近样式图像。例如，:numref:`fig_style_transfer` 中的内容图像是我们在西雅图郊区雷尼尔山国家公园拍摄的风景照片，而风格图像是以秋天橡树为主题的油画。在输出合成图像中，应用样式图像的油画笔笔触，从而产生更鲜艳的色彩，同时保留内容图像中对象的主要形状。 

![Given content and style images, style transfer outputs a synthesized image.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## 方法

:numref:`fig_style_transfer_model` 用简化的示例说明了基于 CNN 的样式传递方法。首先，我们将合成的图像初始化为内容图像。此合成图像是唯一需要在风格转移过程中更新的变量，即训练期间要更新的模型参数。然后，我们选择一个预训练的 CNN 来提取图像要素并在训练期间冻结其模型参数。这个深度 CNN 使用多个图层来提取图像的分层要素。我们可以选择其中一些图层的输出作为内容要素或样式要素。以 :numref:`fig_style_transfer_model` 为例。这里预训练的神经网络有 3 个卷积图层，其中第二层输出内容要素，第一层和第三层输出样式要素。  

![CNN-based style transfer process. Solid lines show the direction of forward propagation and dotted lines show backward propagation. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

接下来，我们计算通过正向传播（实心箭的方向）的样式传递的损失函数，并通过反向传播（虚线箭的方向）更新模型参数（输出的合成图像）。风格转移中常用的损失函数由三个部分组成：(i) * 内容损失 * 使合成图像和内容图像在内容要素中接近；(ii) * 样式损失 * 使合成图像和样式图像在样式特征中接近；(iii) * 全部变体损失 * 有助于减少合成图像中的噪音。最后，当模型训练结束时，我们输出样式转移的模型参数以生成最终的合成图像。 

在下面，我们将通过具体的实验解释风格转移的技术细节。 

## [** 阅读内容和风格图片 **]

首先，我们阅读内容和样式图片。从他们打印的坐标轴中，我们可以看出这些图像的尺寸不同。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [** 预处理和后处理 **]

下面，我们定义了预处理和后处理图像的两个函数。`preprocess` 功能对输入图像的三个 RGB 通道中的每个通道进行标准化，并将结果转换为 CNN 输入格式。`postprocess` 函数将输出图像中的像素值恢复为标准化之前的原始值。由于图像打印功能要求每个像素的浮点值介于 0 到 1 之间，因此我们分别用 0 或 1 替换任何小于 0 或大于 1 的值。

```{.python .input}
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [** 提取功能 **]

我们使用 ImageNet 数据集上预训练的 VGG-19 模型来提取影像要素 :cite:`Gatys.Ecker.Bethge.2016`。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

为了提取图像的内容要素和样式要素，我们可以选择 VGG 网络中某些图层的输出。一般来说，离输入图层越近，提取图像的细节越容易，反之亦然，提取图像的全局信息就越容易。为了避免在合成图像中过度保留内容图像的细节，我们选择了一个更接近输出的 VGG 图层作为 * 内容图层 * 来输出图像的内容要素。我们还选择不同 VGG 图层的输出来提取局部和全局样式要素。这些图层也称为 * 样式图层 *。正如 :numref:`sec_vgg` 中所述，VGG 网络使用 5 个卷积块。在实验中，我们选择第四个卷积块的最后一个卷积层作为内容图层，选择每个卷积块的第一个卷积层作为样式图层。这些图层的索引可以通过打印 `pretrained_net` 实例获得。

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

使用 VGG 图层提取要素时，我们只需要使用从输入图层到最接近输出图层的内容图层或样式图层的所有要素。让我们构建一个新的网络实例 `net`，它只保留用于要素提取的所有 VGG 图层。

```{.python .input}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

给定输入 `X`，如果我们简单地调用正向传播 `net(X)`，我们只能获得最后一层的输出。由于我们还需要中间图层的输出，因此我们需要执行逐层计算并保留内容和样式图层输出。

```{.python .input}
#@tab all
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

下面定义了两个函数：`get_contents` 函数从内容图像中提取内容要素，`get_styles` 函数从样式图像中提取样式特征。由于在训练期间无需更新预训练的 VGG 的模型参数，因此我们甚至可以在训练开始之前提取内容和样式特征。由于合成图像是一组要更新用于风格传输的模型参数，因此我们只能在训练期间调用 `extract_features` 函数来提取合成图像的内容和样式特征。

```{.python .input}
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [** 定义亏损函数 **]

现在我们将描述风格转移的损失函数。丢失函数包括内容丢失、样式损失和总变体损失。 

### 内容丢失

与线性回归中的损失函数类似，内容损失通过平方损失函数衡量合成图像和内容图像之间的内容特征差异。平方损失函数的两个输入都是 `extract_features` 函数计算的内容层的输出。

```{.python .input}
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### 风格损失

样式损失与内容丢失类似，还使用平方损失函数来衡量合成图像和样式图像之间的风格差异。要表达任何样式图层的样式输出，我们首先使用 `extract_features` 函数来计算样式图层输出。假设输出有 1 个示例，$c$ 个通道，高度为 $h$ 和宽度 $w$，我们可以将此输出转换为矩阵 $\mathbf{X}$，其中包含 $c$ 行和 $hw$ 列。这个矩阵可以被认为是 $c$ 矢量 $\mathbf{x}_1, \ldots, \mathbf{x}_c$ 的连接，每个矢量的长度为 $hw$。在这里，矢量 $\mathbf{x}_i$ 代表了 $i$ 频道的风格特征。  

在这些向量的 * 克矩阵 * $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$ 中，$i$ 行中的元素 $x_{ij}$ 和 $j$ 列是矢量 $\mathbf{x}_i$ 和 $\mathbf{x}_j$ 的内积。它代表了 $i$ 和 $j$ 频道风格特征的相关性。我们使用此 Gram 矩阵来表示任何样式图层的样式输出。请注意，当 $hw$ 的值较大时，它可能会导致革兰氏矩阵中的值更大。另请注意，格兰氏矩阵的高度和宽度都是通道数 $c$。为了允许风格损失不受这些值的影响，下面的 `gram` 函数将革兰氏矩阵除以其元素的数量，即 $chw$。

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

显然，用于样式损失的平方损失函数的两个革兰氏矩阵输入基于合成图像的样式图层输出和样式图像。这里假设基于样式图像的 Gram 矩阵 `gram_Y` 已经预先计算出来。

```{.python .input}
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### 总变体损失

有时，学习的合成图像会有很多高频噪点，即特别明亮或暗的像素。一种常见的降噪方法是 
*总变体去噪 *。 
用 $x_{i, j}$ 表示坐标 $(i, j)$ 处的像素值。减少总变体损失 

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

使合成图像上相邻像素的值更接近。

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 亏损函数

[** 风格转移的损失函数是内容损失、风格损失和总变体损失的加权总和 **]。通过调整这些权重超参数，我们可以在合成图像上的内容保留、样式转移和降噪之间取得平衡。

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [** 初始化合成的图像 **]

在风格转移中，合成的图像是训练期间唯一需要更新的变量。因此，我们可以定义一个简单的模型 `SynthesizedImage`，并将合成的图像视为模型参数。在此模型中，正向传播只是返回模型参数。

```{.python .input}
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

接下来，我们定义 `get_inits` 函数。此函数创建一个合成图像模型实例并将其初始化为 `X` 图像。不同样式图层中样式图像的革兰矩阵 `styles_Y_gram` 是在训练之前计算的。

```{.python .input}
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [** 培训 **]

在训练模型进行样式转移时，我们会不断提取合成图像的内容特征和样式特征，然后计算损失函数。下面定义了训练循环。

```{.python .input}
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

现在我们 [** 开始训练模型 **]。我们将内容和样式图像的高度和宽度重新缩放为 300 乘 450 像素。我们使用内容图像来初始化合成的图像。

```{.python .input}
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

我们可以看到，合成的图像保留了内容图像的景色和对象，并同时传输样式图像的颜色。例如，合成的图像具有与样式图像中的颜色块一样。其中一些块甚至具有画笔笔触的微妙纹理。 

## 摘要

* 风格转移中常用的损失函数由三部分组成：(i) 内容丢失使合成图像和内容图像在内容要素中接近；(ii) 样式损失使合成图像和样式图像与样式特征接近；(iii) 总变体损失有助于降低噪音合成的图像。
* 我们可以使用预训练的 CNN 来提取图像特征并最大限度地减少损失函数，以便在训练期间不断更新合成的图像作为模型参数。
* 我们使用 Gram 矩阵来表示样式图层的样式输出。

## 练习

1. 选择不同的内容和样式图层时，输出会如何变化？
1. 调整损失函数中的体重超参数。输出是否保留更多的内容还是噪音较少？
1. 使用不同的内容和样式图片。你能创建更有趣的合成图像吗？
1. 我们可以为文本申请样式转移吗？Hint: you may refer to the survey paper by Hu et al. :cite:`Hu.Lee.Aggarwal.ea.2020`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
