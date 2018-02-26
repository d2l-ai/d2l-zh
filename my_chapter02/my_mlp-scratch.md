# 多层感知机 --- 从0开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input  n=6}
import sys
sys.path.append('..')
import utils
batch_size = 256
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

```{.python .input  n=7}
print(train_data)
print(test_data)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<mxnet.gluon.data.dataloader.DataLoader object at 0x7f41485502e8>\n<mxnet.gluon.data.dataloader.DataLoader object at 0x7f4148550828>\n"
 }
]
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter01_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一到多个隐含层。

![](../img/multilayer-perceptron.png)

这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=8}
from mxnet import ndarray as nd

num_inputs = 28*28
num_outputs = 10

num_hidden = 256
weight_scale = .01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 激活函数

如果我们就用线性操作符来构造多层神经网络，那么整个模型仍然只是一个线性函数。这是因为

$$\hat{y} = X \cdot W_1 \cdot W_2 = X \cdot W_3 $$

这里$W_3 = W_1 \cdot W_2$。为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。这里我们使用ReLU

$$\textrm{rel}u(x)=\max(x, 0)$$

```{.python .input  n=9}
def relu(X):
    return nd.maximum(X, 0)  # 将0广播，然后返回X的同维矩阵
```

## 定义模型

我们的模型就是将层（全连接）和激活函数（Relu）串起来：

```{.python .input  n=10}
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=11}
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=12}
from mxnet import autograd as autograd

learning_rate = .5

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.785235, Train acc 0.707680, Test acc 0.820996\nEpoch 1. Loss: 0.494183, Train acc 0.818368, Test acc 0.852148\nEpoch 2. Loss: 0.427540, Train acc 0.843068, Test acc 0.860449\nEpoch 3. Loss: 0.396314, Train acc 0.853751, Test acc 0.866699\nEpoch 4. Loss: 0.374866, Train acc 0.861120, Test acc 0.857422\n"
 }
]
```

## 预测

```{.python .input  n=13}
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
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

```

```{.python .input  n=14}
from mxnet import gluon

data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))


```

```{.json .output n=14}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f41485c0ba8>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "true labels\n['t-shirt', 'trouser', 'pullover', 'pullover', 'dress,', 'pullover', 'bag', 'shirt', 'sandal']\n"
 }
]
```

```{.python .input  n=15}
data.shape
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "(9, 28, 28, 1)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=16}
net(data).shape
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "(9, 10)"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=17}
predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "predicted labels\n['t-shirt', 'trouser', 'pullover', 'pullover', 'coat', 'shirt', 'bag', 'coat', 'sandal']\n"
 }
]
```

## 总结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 我们使用了 `weight_scale` 来控制权重的初始化值大小，增大或者变小这个值会怎么样？
- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/739)
