# 多类逻辑回归 --- 从0开始

如果你读过了[从0开始的线性回归](linear-regression-scratch.md)，那么最难的部分已经过去了。现在你知道如果读取和操作数据，如何构造目标函数和对它求导，如果定义损失函数，模型和求解。

下面我们来看一个稍微有意思一点的问题，如何使用多类逻辑回归进行多类分类。这个模型跟线性回归的主要区别在于输出节点从一个变成了多个。

![](../img/simple-softmax-net.png)


## 获取数据

演示这个模型的常见数据集是手写数字识别MNIST，它长这个样子。

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/mnist.png)

这里我们用了一个稍微复杂点的数据集，它跟MNIST非常像，但是内容不再是分类数字，而是服饰。我们通过gluon的data.vision模块自动下载这个数据。

```{.python .input  n=238}
from mxnet import gluon
from mxnet import ndarray as nd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
```

打印一个样本的形状和它的标号

```{.python .input  n=239}
data, label = mnist_train[0]
('example shape: ', data.shape, 'label:', label)
```

```{.json .output n=239}
[
 {
  "data": {
   "text/plain": "('example shape: ', (28, 28, 1), 'label:', 2.0)"
  },
  "execution_count": 239,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们画出前几个样本的内容，和对应的文本标号

```{.python .input  n=240}
import matplotlib.pyplot as plt

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))
```

```{.json .output n=240}
[
 {
  "data": {
   "image/png": "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\n0TmEW0wAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f76940607f0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['pullover', 'ankle boot', 'shirt', 't-shirt', 'dress,', 'coat', 'coat', 'sandal', 'coat']\n"
 }
]
```

## 数据读取

虽然我们可以像前面那样通过`yield`来定义获取批量数据函数，这里我们直接使用gluon.data的DataLoader函数，它每次`yield`一个批量。

```{.python .input  n=241}
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
```

注意到这里我们要求每次从训练数据里读取一个由随机样本组成的批量，但测试数据则不需要这个要求。

## 初始化模型参数

跟线性模型一样，每个样本会表示成一个向量。我们这里数据是 28 * 28 大小的图片，所以输入向量的长度是 28 * 28 = 784。因为我们要做多类分类，我们需要对每一个类预测这个样本属于此类的概率。因为这个数据集有10个类型，所以输出应该是长为10的向量。这样，我们需要的权重将是一个 784 * 10 的矩阵：

```{.python .input  n=242}
num_inputs = 784
num_outputs = 10

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)

params = [W, b]
print(params)
```

```{.json .output n=242}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[\n[[ 0.31628925 -0.51747596 -0.56972587 ..., -0.44373724 -1.15453124\n  -1.3114748 ]\n [ 0.36605787  0.24623856  0.45515403 ..., -1.01639283 -0.51937789\n  -0.42492595]\n [-1.7905978   0.47361377  0.05115852 ...,  0.87459302  0.58828896\n  -0.00791701]\n ..., \n [ 0.45501056  0.95532054  0.2882992  ...,  1.08507073  0.85066479\n  -0.82347429]\n [ 0.42549416  0.74736625  0.40008491 ...,  0.81855476 -0.55872959\n   2.37984657]\n [ 0.1386409  -0.22651403  0.62381667 ..., -0.08955938  1.01807368\n  -1.06890678]]\n<NDArray 784x10 @cpu(0)>, \n[-0.08249321  1.41823101 -0.50847739  0.24808507  1.02948284  1.4690367\n  1.85592437  0.60471541 -0.49085623 -0.50624841]\n<NDArray 10 @cpu(0)>]\n"
 }
]
```

同之前一样，我们要对模型参数附上梯度：

```{.python .input  n=243}
for param in params:
    param.attach_grad()
```

## 定义模型

在线性回归教程里，我们只需要输出一个标量`yhat`使得尽可能的靠近目标值。但在这里的分类里，我们需要属于每个类别的概率。这些概率需要值为正，而且加起来等于1. 而如果简单的使用 $\boldsymbol{\hat y} = \boldsymbol{W} \boldsymbol{x}$, 我们不能保证这一点。一个通常的做法是通过softmax函数来将任意的输入归一化成合法的概率值。

```{.python .input  n=244}
from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
#     print(partition)
#     print(exp.shape)
    return exp / partition
```

可以看到，对于随机输入，我们将每个元素变成了非负数，而且每一行加起来为1。

```{.python .input  n=245}
X = nd.random_normal(shape=(4,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))
```

```{.json .output n=245}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.17035699  0.48914027  0.14033821  0.07881127  0.12135328]\n [ 0.20942886  0.08640888  0.10505491  0.59362262  0.00548483]\n [ 0.4968085   0.32582662  0.11188766  0.04129994  0.02417732]\n [ 0.13839263  0.4463208   0.19035615  0.17861113  0.04631924]]\n<NDArray 4x5 @cpu(0)>\n\n[ 1.          1.00000012  1.          1.        ]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

现在我们可以定义模型了：

```{.python .input  n=246}
def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)), W) + b)
```

## 交叉熵损失函数

我们需要定义一个针对预测为概率值的损失函数。其中最常见的是交叉熵损失函数，它将两个概率分布的负交叉熵作为目标值，最小化这个值等价于最大化这两个概率的相似度。

具体来说，我们先将真实标号表示成一个概率分布，例如如果`y=1`，那么其对应的分布就是一个除了第二个元素为1其他全为0的长为10的向量，也就是 `yvec=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`。那么交叉熵就是`yvec[0]*log(yhat[0])+...+yvec[n]*log(yhat[n])`。注意到`yvec`里面只有一个1，那么前面等价于`log(yhat[y])`。所以我们可以定义这个损失函数了

```{.python .input  n=247}
def cross_entropy(yhat, y):
#     return - nd.pick(nd.log(yhat), y)
    return - nd.log(nd.pick(yhat, y))
```

```{.python .input  n=248}
import math
yhat = math.exp(1) ** (-1 * nd.arange(20).reshape((2,10)))
y = nd.array([6,2])
print(yhat)
print(y)
cross_entropy(yhat, y)
```

```{.json .output n=248}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[  1.00000000e+00   3.67879450e-01   1.35335296e-01   4.97870706e-02\n    1.83156412e-02   6.73794793e-03   2.47875252e-03   9.11882147e-04\n    3.35462682e-04   1.23409831e-04]\n [  4.53999419e-05   1.67017060e-05   6.14421424e-06   2.26033012e-06\n    8.31529007e-07   3.05902432e-07   1.12535218e-07   4.13993959e-08\n    1.52299862e-08   5.60279911e-09]]\n<NDArray 2x10 @cpu(0)>\n\n[ 6.  2.]\n<NDArray 2 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[  6.  12.]\n<NDArray 2 @cpu(0)>"
  },
  "execution_count": 248,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 计算精度

给定一个概率输出，我们将预测概率最高的那个类作为预测的类，然后通过比较真实标号我们可以计算精度：

```{.python .input  n=249}
def accuracy(output, label):
#     print((output.argmax(axis=1)==label).shape)
    return nd.mean(output.argmax(axis=1)==label).asscalar()
```

我们可以评估一个模型在这个数据上的精度。（这两个函数我们之后也会用到，所以也都保存在[../utils.py](../utils.py)。）

```{.python .input  n=250}
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
```

因为我们随机初始化了模型，所以这个模型的精度应该大概是`1/num_outputs = 0.1`.

```{.python .input  n=251}
evaluate_accuracy(test_data, net)
```

```{.json .output n=251}
[
 {
  "data": {
   "text/plain": "0.068750000000000006"
  },
  "execution_count": 251,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 训练

训练代码跟前面的线性回归非常相似：

```{.python .input  n=252}
import sys
sys.path.append('..')
from utils import SGD
from mxnet import autograd

learning_rate = .5

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=252}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 2.255614, Train acc 0.638436, Test acc 0.742676\nEpoch 1. Loss: 1.343939, Train acc 0.738115, Test acc 0.772656\nEpoch 2. Loss: 1.181268, Train acc 0.759236, Test acc 0.785059\nEpoch 3. Loss: nan, Train acc 0.525892, Test acc 0.103516\nEpoch 4. Loss: nan, Train acc 0.100094, Test acc 0.103516\n"
 }
]
```

## 预测

训练完成后，现在我们可以演示对输入图片的标号的预测

```{.python .input  n=185}
data, label = mnist_test[0:9]
print(data.shape)
print(label.shape)
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print(net(data).max(axis=1))
print(predicted_labels)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
```

```{.json .output n=185}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(9, 28, 28, 1)\n(9,)\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f7699e7f1d0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "true labels\n['t-shirt', 'trouser', 'pullover', 'pullover', 'dress,', 'pullover', 'bag', 'shirt', 'sandal']\n\n[ 0.97994012  0.99999994  0.75621921  0.76367581  0.87824315  0.89354783\n  0.99945712  0.79184598  0.99994409]\n<NDArray 9 @cpu(0)>\n\n[ 6.  1.  2.  2.  4.  6.  8.  6.  5.]\n<NDArray 9 @cpu(0)>\npredicted labels\n['shirt', 'trouser', 'pullover', 'pullover', 'coat', 'shirt', 'bag', 'shirt', 'sandal']\n"
 }
]
```

## 结论

与前面的线性回归相比，你会发现多类逻辑回归教程的结构跟其非常相似：获取数据、定义模型及优化算法和求解。事实上，几乎所有的实际神经网络应用都有着同样结构。他们的主要区别在于模型的类型和数据的规模。每一两年会有一个新的优化算法出来，但它们基本都是随机梯度下降的变种。

## 练习

尝试增大学习率，你会发现结果马上回变成很糟糕，精度基本徘徊在随机的0.1左右。这是为什么呢？提示：

- 打印下output看看是不是有有什么异常
- 前面线性回归还好好的，这里我们在net()里加了什么呢？
- 如果给exp输入个很大的数会怎么样？
- 即使解决exp的问题，求出来的导数是不是还是不稳定？

请仔细想想再去对比下我们小伙伴之一@[pluskid](https://github.com/pluskid)早年写的一篇[blog解释这个问题](http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/)，看看你想的是不是不一样。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/741)
