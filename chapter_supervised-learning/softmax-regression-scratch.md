# Softmax回归——从零开始

前几节介绍的线性回归模型适用于输出为连续值的情景，例如输出为房价。在其他情景中，模型输出还可以是一个离散值，例如图片类别。对于这样的分类问题，我们可以使用Softmax回归模型。和线性回归不同，Softmax回归的输出单元从一个变成了多个。


## Softmax回归模型

让我们考虑一个简单的分类问题。为了便于讨论，让我们假设输入图片的尺寸为$2 \times 2$，并设图片的四个特征值，即像素值分别为$x_1, x_2, x_3, x_4$。假设训练数据集中图片的真实标签为狗、猫或鸡，这些标签分别对应离散值$y_1, y_2, y_3$。举个例子，如果$y_1=0, y_2=1, y_3=2$，任意一张狗图片的标签记作0。


### 单样本分类

下面我们一步步地描述Softmax回归是怎样对单个$2 \times 2$图片样本分类的。

设带下标的$w$和$b$分别为Softmax回归的权重和偏差参数。给定单个图片的输入特征$x_1, x_2, x_3, x_4$，我们有
$$
o_1 = x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1,\\
o_2 = x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2,\\
o_3 = x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3.
$$

图3.2用神经网络图描绘了上面的计算。
和线性回归一样，Softmax回归也是一个单层神经网络。和线性回归有所不同的是，Softmax回归输出层中的输出个数等于类别个数，因此从一个变成了多个。在Softmax回归中，$o_1, o_2, o_3$的计算都要依赖于$x_1, x_2, x_3, x_4$。所以，Softmax回归的输出层是一个全连接层。

![Softmax回归是一个单层神经网络](../img/softmaxreg.svg)

在得到输出层的三个输出后，我们需要预测输出分别为狗、猫或鸡的概率。不妨设它们分别为$\hat{y}_1, \hat{y}_2, \hat{y}_3$。下面，我们通过对$o_1, o_2, o_3$做Softmax运算，得到模型最终输出

$$
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\\
\hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\\
\hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$$

由于$\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$且$\hat{y}_1 \geq 0, \hat{y}_2 \geq 0, \hat{y}_3 \geq 0$，$\hat{y}_1, \hat{y}_2, \hat{y}_3$是一个合法的概率分布。我们可将上面三式记作

$$\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{Softmax}(o_1, o_2, o_3).$$


### 单样本分类的矢量计算表达式

为了提高计算效率，我们可以将单样本分类通过矢量计算来表达。在上面的图片分类问题中，假设Softmax回归的权重和偏差参数分别为

$$
\boldsymbol{W} = 
\begin{bmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43}
\end{bmatrix},\quad
\boldsymbol{b} = 
\begin{bmatrix}
    b_1 & b_2 & b_3
\end{bmatrix},
$$




设$2 \times 2$图片样本$i$的特征为

$$\boldsymbol{x}^{(i)} = \begin{bmatrix}x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix},$$

输出层输出为
$$\boldsymbol{o}^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix},$$

预测为狗、猫或鸡的概率分布为

$$\boldsymbol{\hat{y}}^{(i)} = \begin{bmatrix}\hat{y}_1^{(i)} & \hat{y}_2^{(i)} & \hat{y}_3^{(i)}\end{bmatrix}.$$


我们对样本$i$分类的矢量计算表达式为

$$
\boldsymbol{o}^{(i)} = \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{y}}^{(i)} = \text{Softmax}(\boldsymbol{o}^{(i)}).
$$


### 小批量样本分类的矢量计算表达式


为了进一步提升计算效率，我们通常对小批量数据做矢量计算。广义上，给定一个小批量样本，其批量大小为$n$，输入个数（特征数）为$x$，输出个数（类别数）为$y$。设批量特征为$\boldsymbol{X} \in \mathbb{R}^{n \times x}$，批量标签$\boldsymbol{y} \in \mathbb{R}^{n \times 1}$。
假设Softmax回归的权重和偏差参数分别为$\boldsymbol{W} \in \mathbb{R}^{x \times y}, \boldsymbol{b} \in \mathbb{R}^{1 \times y}$。Softmax回归的矢量计算表达式为

$$
\boldsymbol{O} = \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{Y}} = \text{Softmax}(\boldsymbol{O}),
$$

其中的加法运算使用了广播机制，$\boldsymbol{O}, \boldsymbol{\hat{Y}} \in \mathbb{R}^{n \times y}$且这两个矩阵的第$i$行分别为$\boldsymbol{o}^{(i)}$和$\boldsymbol{\hat{y}}^{(i)}$。


### 交叉熵损失函数

Softmax回归使用了交叉熵损失函数（cross-entropy loss）。以本节中的图片分类为例，真实标签狗、猫或鸡分别对应离散值$y_1, y_2, y_3$，它们的预测概率分别为$\hat{y}_1, \hat{y}_2, \hat{y}_3$。为了便于描述，设样本$i$的标签的被预测概率为$p_{\text{label}_i}$。例如，如果样本$i$的标签为$y_3$，那么$p_{\text{label}_i} = \hat{y}_3$。直观上，训练数据集上每个样本的真实标签的被预测概率越大（最大为1），分类越准确。假设训练数据集的样本数为$n$。由于对数函数是单调递增的，且最大化函数与最小化该函数的相反数等价，我们希望最小化

$$
\ell(\boldsymbol{\Theta}) = -\frac{1}{n} \sum_{i=1}^n \log p_{\text{label}_i},
$$
其中$\boldsymbol{\Theta}$为模型参数。该函数即交叉熵损失函数。在训练Softmax回归时，我们将使用优化算法来迭代模型参数并最小化该损失函数。


### 模型预测及评价

在训练好Softmax回归模型后，给定任一样本特征，我们可以预测每个输出类别的概率。通常，我们把预测概率最大的类别作为输出类别。如果它与真实类别（标签）一致，说明这次预测是正确的。在本节的实验中，我们将使用准确率（accuracy）来评价模型的表现。它等于正确预测数量与总预测数量的比。


## Softmax回归实现

下面我们来动手实现Softmax回归。首先，导入实验所需的包或模块。


### 获取数据

演示这个模型的常见数据集是手写数字识别MNIST。这里我们用了一个稍微复杂点的数据集，它跟MNIST非常像，但是内容不再是分类数字，而是服饰。我们通过gluon的data.vision模块自动下载这个数据。

```{.python .input  n=1}
import matplotlib.pyplot as plt
from mxnet import autograd, nd
from mxnet.gluon import data as gdata
import sys
sys.path.append('..')
import utils
```

```{.python .input  n=2}
def transform(feature, label):
    return feature.astype('float32') / 255, label.astype('float32')

mnist_train = gdata.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gdata.vision.FashionMNIST(train=False, transform=transform)
```

打印一个样本的形状和它的标号

```{.python .input  n=3}
feature, label = mnist_train[0]
'feature shape: ', feature.shape, 'label: ', label
```

我们画出前几个样本的内容，和对应的文本标号

```{.python .input  n=4}
def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
```

```{.python .input}
def get_text_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]
```

```{.python .input}
X, y = mnist_train[0:9]
show_images(X)
print(get_text_labels(y))
```

### 数据读取

虽然我们可以像前面那样通过`yield`来定义获取批量数据函数，这里我们直接使用gluon.data的DataLoader函数，它每次`yield`一个批量。

```{.python .input  n=5}
batch_size = 256
train_iter = gdata.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = gdata.DataLoader(mnist_test, batch_size, shuffle=False)
```

注意到这里我们要求每次从训练数据里读取一个由随机样本组成的批量，但测试数据则不需要这个要求。

### 初始化模型参数

跟线性模型一样，每个样本会表示成一个向量。我们这里数据是 $28 \times 28$ 大小的图片，所以输入向量的长度是 $28 \times 28 = 784$。因为我们要做多类分类，我们需要对每一个类预测这个样本属于此类的概率。因为这个数据集有10个类型，所以输出应该是长为10的向量。这样，我们需要的权重将是一个 $784 \times 10$ 的矩阵：

```{.python .input  n=6}
num_inputs = 784
num_outputs = 10

W = nd.random.normal(shape=(num_inputs, num_outputs))
b = nd.random.normal(shape=num_outputs)

params = [W, b]
```

同之前一样，我们要对模型参数附上梯度：

```{.python .input  n=7}
for param in params:
    param.attach_grad()
```

### 定义模型

在线性回归教程里，我们只需要输出一个标量`yhat`使得尽可能的靠近目标值。但在这里的分类里，我们需要属于每个类别的概率。这些概率需要值为正，而且加起来等于1. 而如果简单的使用 $\boldsymbol{\hat y} = \boldsymbol{W} \boldsymbol{x}$, 我们不能保证这一点。一个通常的做法是通过softmax函数来将任意的输入归一化成合法的概率值。

```{.python .input  n=8}
from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition
```

可以看到，对于随机输入，我们将每个元素变成了非负数，而且每一行加起来为1。

```{.python .input  n=9}
X = nd.random_normal(shape=(2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))
```

现在我们可以定义模型了：

```{.python .input  n=10}
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
```

### 交叉熵损失函数

我们需要定义一个针对预测为概率值的损失函数。其中最常见的是交叉熵损失函数，它将两个概率分布的负交叉熵作为目标值，最小化这个值等价于最大化这两个概率的相似度。

具体来说，我们先将真实标号表示成一个概率分布，例如如果`y=1`，那么其对应的分布就是一个除了第二个元素为1其他全为0的长为10的向量，也就是 `yvec=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`。那么交叉熵就是`yvec[0]*log(yhat[0])+...+yvec[n]*log(yhat[n])`。注意到`yvec`里面只有一个1，那么前面等价于`log(yhat[y])`。所以我们可以定义这个损失函数了

```{.python .input  n=11}
def cross_entropy(y_hat, y):
    return - nd.pick(nd.log(y_hat), y)
```

```{.python .input}
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2])
cross_entropy(y_hat, y)
```

### 计算精度

给定一个概率输出，我们将预测概率最高的那个类作为预测的类，然后通过比较真实标号我们可以计算精度：

```{.python .input  n=12}
def accuracy(output, label):
    return (output.argmax(axis=1) == label).mean().asscalar()
```

我们可以评估一个模型在这个数据上的精度。（这两个函数我们之后也会用到，所以也都保存在[../utils.py](../utils.py)。）

```{.python .input  n=13}
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)
```

因为我们随机初始化了模型，所以这个模型的精度应该大概是`1/num_outputs = 0.1`.

```{.python .input  n=14}
evaluate_accuracy(test_iter, net)
```

### 训练

训练代码跟前面的线性回归非常相似：

```{.python .input  n=15}
num_epochs = 5
lr = 0.1

for epoch in range(1, num_epochs + 1):
    train_l_sum = 0
    train_acc_sum = 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = cross_entropy(y_hat, y)
        l.backward()
        utils.sgd(params, lr, batch_size)
        train_l_sum += l.mean().asscalar()
        train_acc_sum += accuracy(y_hat, y)
    test_acc = evaluate_accuracy(test_iter, net)
    print("epoch %d, loss %f, train acc %f, test acc %f" 
          % (epoch, train_l_sum / len(train_iter),
             train_acc_sum / len(train_iter), test_acc))
```

### 预测

训练完成后，现在我们可以演示对输入图片的标号的预测

```{.python .input  n=16}
data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))
predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
```

## 小结

与前面的线性回归相比，你会发现多类逻辑回归教程的结构跟其非常相似：获取数据、定义模型及优化算法和求解。事实上，几乎所有的实际神经网络应用都有着同样结构。他们的主要区别在于模型的类型和数据的规模。每一两年会有一个新的优化算法出来，但它们基本都是随机梯度下降的变种。

## 练习

尝试增大学习率，你会马上发现结果变得很糟糕，精度基本徘徊在随机的0.1左右。这是为什么呢？提示：

- 打印下output看看是不是有什么异常
- 前面线性回归还好好的，这里我们在net()里加了什么呢？
- 如果给exp输入个很大的数会怎么样？
- 即使解决exp的问题，求出来的导数是不是还是不稳定？

请仔细想想再去对比下我们小伙伴之一@[pluskid](https://github.com/pluskid)早年写的一篇[blog解释这个问题](http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/)，看看你想的是不是不一样。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/741)

![](../img/qr_softmax-regression-scratch.svg)
