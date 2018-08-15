# 多类图片分类数据集（Fashion-MNIST）

在介绍softmax回归的实现前我们先引入一个多类图片分类数据集。它将多次在后面的章节中使用，方便我们观察比较算法之间的模型精度、计算速度和收敛率之间的区别。多类图片分类数据集中最常用的是手写数字识别数据集MNIST [1]。但由于大部分模型在MNIST上的分类精度都超过了95%，这里我们将使用一个图片内容更加复杂的变种 Fashion-MNIST [2]，使得我们可以更加清楚的观察算法之间的差异。

在Fashion-MNIST中，图片的高和宽均为28像素，一共包括了10个类别，分别为：t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。训练数据集中和测试数据集中的每个类别的图片数分别为6,000和1,000。

## 获取数据集

首先导入本节需要的包。

```{.python .input}
%matplotlib inline
import sys
sys.path.insert(0, '..')

import sys
import gluonbook as gb
from mxnet.gluon import data as gdata
from IPython.display import set_matplotlib_formats 
```

下面，我们通过Gluon的`data`包来下载这个数据集，第一次调用时其自动从网上获取数据。我们通过`train`来指定获取训练集还是测试集。

```{.python .input  n=23}
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```

因为有10个类别，所以训练集和测试集样本数分别为60,000和10,000。

```{.python .input}
len(mnist_train), len(mnist_test)
```

我们可以通过`[]`来访问任意一个样本，下面获取第一个样本和图片和标签。

```{.python .input  n=24}
feature, label = mnist_train[0]
```

`feature`对应高和宽均为28像素的图片。每个像素的数值为0到255之间8位无符号整数（uint8）。它使用3维的NDArray储存。其中的最后一维是通道数。因为是灰度图片，所以通道数为1。

```{.python .input}
feature.shape, feature.dtype
```

下面定义一个可以在一行里画出多张图片的函数。

```{.python .input}
def show_fashion_imgs(images, labels):
    # 这里的 _ 表示我们忽略（不使用）的变量。
    set_matplotlib_formats('svg')
    _, figs = gb.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```

图片的标签使用NumPy的标量表示。它的类型为32位整数。

```{.python .input}
label, type(label), label.dtype
```

以下函数可以将数值标签转成相应的文本标签。

```{.python .input  n=25}
def get_text_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

现在，我们看一下训练数据集中前9个样本的图片内容和文本标签。

```{.python .input  n=27}
X, y = mnist_train[0:9]
show_fashion_imgs(X, get_text_labels(y))
```

## 读取数据

Fashion-MNIST包括训练数据集和测试数据集（testing data set）。我们将在训练数据集上训练模型，并将训练好的模型在测试数据集上评价模型的表现。对于训练数据集，我们需要使用随机顺序访问其样本。

我们可以像[“线性回归的从零开始实现”](linear-regression-scratch.md)一节中那样通过`yield`来定义读取小批量数据样本的函数。为了代码简洁，这里我们直接创建DataLoader实例，其每次读取一个样本数为`batch_size`的小批量。这里的批量大小`batch_size`是一个超参数。在实际中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。Gluon的`DataLoader`中一个很方便的功能是允许使用多进程来加速数据读取（暂不支持Windows操作系统）。这里我们通过参数`num_workers`来设置4个进程读取数据。

此外，我们通过`ToTensor`将图片数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间。`ToTensor`还将图片通道从最后一维调整到最前一维来方便之后介绍的卷积神经网络使用。通过数据集的`transform_first`函数，我们将`ToTensor`的变换应用在每个数据样本（图片和标签）的第一个元素，也即图片之上。

```{.python .input  n=28}
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```

我们将获取并读取Fashion-MNIST数据集的逻辑封装在`gluonbook.load_data_fashion_mnist`函数中供后面章节调用。随着本书内容的不断深入，我们会进一步改进该函数。它的完整实现将在[“深度卷积神经网络（AlexNet）”](../chapter_convolutional-neural-networks/alexnet.md)一节中描述。

## 小结

* 与训练线性回归相比，你会发现训练Softmax回归的步骤跟其非常相似：获取并读取数据、定义模型和损失函数并使用优化算法训练模型。事实上，绝大多数深度学习模型的训练都有着类似的步骤。

* 我们可以使用Softmax回归做多类别分类。

## 练习

* 本节中，我们直接按照Softmax运算的数学定义来实现`softmax`函数。这可能会造成什么问题？（试一试计算$e^{50}$的大小。）
* 本节中的`cross_entropy`函数同样是按照交叉熵损失函数的数学定义实现的。这样的实现方式可能有什么问题？（思考一下对数函数的定义域。）
* 你能想到哪些办法来解决上面这两个问题？
* 修改`DataLoader`里的参数`num_workers`，查看这个改动对计算性能的影响。

## 扫码直达讨论区

TODO


## 参考文献

[1] LeCun, Y., Cortes, C., & Burges, C. http://yann.lecun.com/exdb/mnist/

[2] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.
