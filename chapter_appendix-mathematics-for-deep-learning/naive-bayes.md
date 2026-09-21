# 朴素贝叶斯
:label:`sec_naive_bayes`

在前面几节中，我们学习了概率与随机变量理论。为了应用这些理论，下面介绍*朴素贝叶斯*（naive Bayes）分类器。该分类器只使用概率论的基础知识，就可以完成数字分类。

学习离不开假设。如果要对从未见过的新数据样本进行分类，就必须对哪些数据样本彼此相似作出某些假设。朴素贝叶斯分类器是一种流行且非常清晰的算法，它假设所有特征彼此独立，从而简化计算。本节将应用该模型识别图像中的字符。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 光学字符识别

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` 是使用最广泛的数据集之一。它包含60000张训练图像和10000张验证图像，每张图像都包含一个$0$到$9$的手写数字。任务是将每张图像分类为相应的数字。

Gluon在`data.vision`模块中提供了`MNIST`类，可以自动从互联网获取该数据集。
此后，Gluon会使用已经下载到本地的副本。
将参数`train`分别设为`True`或`False`，即可指定获取训练集或测试集。
每张图像都是宽和高均为$28$的灰度图像，其形状为(28, 28, 1)。我们使用自定义变换移除最后的通道维。此外，该数据集使用无符号$8$位整数表示每个像素。为了简化问题，我们将其量化为二值特征。

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# MNIST的原始像素值范围为0--255（数字以uint8格式存储）
# 本节将原图中大于128的像素值转换为1，将小于128的值转换为0
# 原因参见第18.9.2节和第18.9.3节
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

我们可以访问某个特定样本，其中包含图像及其对应标签。

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

这里存储在变量`image`中的样本对应一张高和宽均为$28$像素的图像。

```{.python .input}
#@tab all
image.shape, image.dtype
```

代码将每张图像的标签存储为标量，其类型为$32$位整数。

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

我们也可以同时访问多个样本。

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

下面将这些样本可视化。

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 分类的概率模型

在分类任务中，我们把一个样本映射到某个类别。这里的样本是一张$28\times28$灰度图像，类别则是一个数字（更详细的解释参见 :numref:`sec_softmax`）。
一种自然的分类任务表述方式是提出概率问题：给定特征（即图像像素）时，最可能的标签是什么？用$\mathbf{x}\in\mathbb{R}^d$表示样本的特征，用$y\in\mathbb{R}$表示标签。这里特征就是图像像素，可以把二维图像重塑为向量，从而$d=28^2=784$；标签则是数字。
给定特征时标签的概率为$p(y\mid\mathbf{x})$。在本例中，如果能对$y=0,\ldots,9$分别计算$p(y\mid\mathbf{x})$，分类器就会输出由下式给出的预测$\hat{y}$：

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

遗憾的是，这要求我们对$\mathbf{x}=x_1,...,x_d$的每个可能取值估计$p(y\mid\mathbf{x})$。假设每个特征只能取$2$个值。例如，$x_1=1$可能表示给定文档中出现了单词apple，而$x_1=0$表示没有出现。如果有$30$个这样的二值特征，就必须准备对输入向量$\mathbf{x}$的$2^{30}$种可能取值（超过十亿种）进行分类。

更重要的是，这样做还算学习吗？如果必须见过每个可能的样本才能预测相应标签，那么模型并没有真正学习规律，只是在记忆数据集。

## 朴素贝叶斯分类器

幸运的是，通过对条件独立性作出一些假设，我们可以引入归纳偏置，并构建一个只需相对少量训练样本就能泛化的模型。首先，利用贝叶斯定理可以将分类器表示为

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

注意，分母是归一化项$p(\mathbf{x})$，它不依赖标签$y$的取值。因此，只需比较不同$y$取值所对应的分子。即使分母难以计算，只要能够计算分子，也可以将其忽略。事实上，即便需要归一化常数，也总能利用$\sum_y p(y\mid\mathbf{x})=1$将其恢复出来。

现在来看$p(\mathbf{x}\mid y)$。利用概率的链式法则，可以将$p(\mathbf{x}\mid y)$表示为

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

仅凭这个表达式还无法简化问题，我们仍需估计大约$2^d$个参数。然而，如果假设*给定标签时，各个特征彼此条件独立*，情况就会好得多：该项可简化为$\prod_i p(x_i\mid y)$，从而得到预测器

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

如果能够对每个$i$和$y$估计$p(x_i=1\mid y)$，并将其值保存在$P_{xy}[i,y]$中，其中$P_{xy}$是一个$d\times n$矩阵，$n$为类别数且$y\in\{1,\ldots,n\}$，那么也可以据此估计$p(x_i=0\mid y)$，即

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & t_i=1;\\
    1 - P_{xy}[i, y] & t_i=0.
\end{cases}
$$

此外，对每个$y$估计$p(y)$并将其保存在$P_y[y]$中，其中$P_y$是长度为$n$的向量。于是，对于任意新样本$\mathbf t=(t_1,t_2,\ldots,t_d)$，可以计算

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

其中$y$可取任意类别。因此，条件独立假设将模型复杂度从关于特征数的指数复杂度$\mathcal{O}(2^dn)$降至线性复杂度$\mathcal{O}(dn)$。


## 训练

现在的问题是，我们还不知道$P_{xy}$和$P_y$。因此，首先需要根据训练数据估计它们的值，这个过程就是模型的*训练*（training）。估计$P_y$并不困难。由于这里只有$10$个类别，可以统计每个数字出现的次数$n_y$，再除以数据总量$n$。例如，如果数字8出现了$n_8=5800$次，而图像总数为$n=60000$，那么概率估计为$p(y=8)=0.0967$。

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # 所有训练样本

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

接下来估计稍难一些的$P_{xy}$。由于使用的是黑白图像，$p(x_i\mid y)$表示类别为$y$时像素$i$被点亮的概率。与之前类似，可以统计该事件发生的次数$n_{iy}$，再除以$y$出现的总次数$n_y$。不过这里存在一个问题：某些像素可能从未变黑（例如，对裁剪良好的图像而言，角落像素可能始终为白色）。统计学中一种方便的处理方法是为所有计数加入伪计数。因此，用$n_{iy}+1$代替$n_{iy}$，用$n_y+2$代替$n_y$（因为像素$i$有黑、白两个可能取值）。这也称为*拉普拉斯平滑*（Laplace smoothing）。虽然这种做法看似临时拼凑，但从贝叶斯视角可以用Beta—二项模型解释。

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

将这$10\times28\times28$个概率（每个类别中每个像素的概率）可视化，可以得到一些具有代表性的数字图像。

现在可以使用 :eqref:`eq_naive_bayes_estimation`预测新图像。给定$\mathbf{x}$，下面的函数对每个$y$计算$p(\mathbf{x}\mid y)p(y)$。

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

结果完全不对！要找出原因，先看每个像素的概率。它们通常是$0.001$到$1$之间的数，而这里要将其中$784$个相乘。需要注意的是，这些数是在计算机上计算的，因此指数的表示范围有限。此时会发生*数值下溢*（numerical underflow）：许多小数相乘会得到越来越小的结果，直至被舍入为零。我们曾在 :numref:`sec_maximum_likelihood`中从理论上讨论过这个问题，现在则在实践中清楚地看到了这一现象。

如该节所述，可以利用$\log ab=\log a+\log b$解决这一问题，也就是改为对各项的对数求和。
即使$a$和$b$都是小数，其对数值通常仍处于可表示的范围内。

```{.python .input}
#@tab mxnet
a = 0.1
print('下溢：', a**784)
print('对数值正常：', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('下溢：', a**784)
print('对数值正常：', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('下溢：', a**784)
print('对数值正常：', 784*tf.math.log(a).numpy())
```

由于对数函数单调递增，可以将 :eqref:`eq_naive_bayes_estimation`改写为

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

由此可以实现下面这个数值稳定的版本：

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

现在可以检查预测是否正确。

```{.python .input}
#@tab mxnet
# 将int32类型的标量张量标签转换为Python标量整数
# 以便进行比较
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

现在对几个验证样本进行预测，可以看到贝叶斯分类器的表现相当不错。

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

最后，计算分类器的总体准确率。

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # 验证准确率
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # 验证准确率
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# 验证准确率
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

现代深度网络可以达到低于$0.01$的错误率。这个模型表现相对较差，是因为采用了不正确的统计假设：我们假设每个像素都只依赖于标签，并且彼此*独立*地生成。显然，人类并不是以这种方式书写数字的；正是这个错误假设导致了这个过于朴素的（贝叶斯）分类器表现不佳。

## 小结

* 利用贝叶斯规则，并假设所有观测特征彼此独立，可以构造一个分类器。
* 统计标签与像素值各种组合的出现次数，即可在数据集上训练该分类器。
* 数十年来，这种分类器一直是垃圾邮件检测等任务的黄金标准。

## 练习

1. 考虑数据集$[[0,0],[0,1],[1,0],[1,1]]$，其标签由两个元素的异或给出，即$[0,1,1,0]$。在该数据集上构建的朴素贝叶斯分类器会得到哪些概率？它能否正确地对这些点进行分类？如果不能，违反了哪些假设？
1. 假设估计概率时没有使用拉普拉斯平滑，并且测试时出现了一个包含训练中从未观察到的取值的数据样本。模型会输出什么？
1. 朴素贝叶斯分类器是贝叶斯网络的一个具体例子。在贝叶斯网络中，随机变量之间的依赖关系由图结构编码。完整理论超出了本节范围（详见 :citet:`Koller.Friedman.2009`），请解释为什么在异或模型中允许两个输入变量之间存在显式依赖，就能构造出成功的分类器。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1101)
:end_tab:
