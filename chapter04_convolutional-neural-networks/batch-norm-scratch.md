# 批量归一化 --- 从0开始

在[Kaggle实战](../chapter02_supervised-learning/kaggle-gluon-kfold.md#预处理数据)我们输入数据做了归一化。在实际应用中，我们通常将输入数据的每个样本或者每个特征进行归一化，就是将均值变为0方差变为1，来使得数值更稳定。

这个对
我们在之前的课程里学过了[线性回归](../chapter02_supervised-learning/linear-regression-
scratch.md)和[逻辑回归](../chapter02_supervised-learning/softmax-regression-
scratch.md)很有效。因为输入层的输入值的大小变化不剧烈，那么输入也不会。但是，对于一个可能有很多层的深度学习模型来说，情况可能会比较复杂。

举个例子，随着第一层和第二层的参数在训练时不断变化，第三层所使用的激活函数的输入值可能由于乘法效应而变得极大或极小，例如和第一层所使用的激活函数的输入值不在一个数量级上。这种在训练时可能出现的情况会造成模型训练的不稳定性。例如，给定一个学习率，某次参数迭代后，目标函数值会剧烈变化或甚至升高。数学的解释是，如果把目标函数
$f$ 根据参数 $\mathbf{w}$ 迭代（如 $f(\mathbf{w} - \eta \nabla f(\mathbf{w}))$
）进行泰勒展开，有关学习率 $\eta$ 的高阶项的系数可能由于数量级的原因（通常由于层数多）而不容忽略。然而常用的低阶优化算法（如梯度下降）对于不断降低目标函
数的有效性通常基于一个基本假设：在以上泰勒展开中把有关学习率的高阶项通通忽略不计。

为了应对上述这种情况，Sergey Ioffe和Christian Szegedy在2015年提出了批量归一化的方法。简而言之，在训练时给定一个批量输入，批量归一化试图对深度学习模型的某一层所使用的激活函数的输入进行归一化：使批量呈标准正态分布（均值为0，标准差为1）。

批量归一化通常应用于输入层或任意中间层。

## 简化的批量归一化层

给定一个批量 $B = \{x_{1, ..., m}\}$, 我们需要学习拉升参数 $\gamma$ 和偏移参数 $\beta$。

我们定义：

$$\mu_B \leftarrow \frac{1}{m}\sum_{i = 1}^{m}x_i$$
$$\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_B)^2$$
$$\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i \leftarrow \gamma \hat{x_i} + \beta \equiv \mbox{BN}_{\gamma,\beta}(x_i)$$

批量归一化层的输出是 $\{y_i = BN_{\gamma, \beta}(x_i)\}$。

我们现在来动手实现一个简化的批量归一化层。实现时对全连接层和二维卷积层两种情况做了区分。对于全连接层，很明显我们要对每个批量进行归一化。然而这里需要注意的是，对
于二维卷积，我们要对每个通道进行归一化，并需要保持四维形状使得可以正确地广播。

```{.python .input  n=1}
from mxnet import nd
def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确地广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        
    # 均一化
    X_hat = (X - mean) / nd.sqrt(variance + eps)
    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)
```

下面我们检查一下。我们先定义全连接层的输入是这样的。每一行是批量中的一个实例。

```{.python .input  n=2}
A = nd.arange(6).reshape((3,2))
A
```

我们希望批量中的每一列都被归一化。结果符合预期。

```{.python .input  n=3}
pure_batch_norm(A, gamma=nd.array([1,1]), beta=nd.array([0,0]))
```

下面我们定义二维卷积网络层的输入是这样的。

```{.python .input  n=4}
B = nd.arange(18).reshape((1,2,3,3))
B
```

结果也如预期那样，我们对每个通道做了归一化。

```{.python .input  n=5}
pure_batch_norm(B, gamma=nd.array([1,1]), beta=nd.array([0,0]))
```

## 批量归一化层

你可能会想，既然训练时用了批量归一化，那么测试时也该用批量归一化吗？其实这个问题乍一想不是很好回答，因为：

* 不用的话，训练出的模型参数很可能在测试时就不准确了；
* 用的话，万一测试的数据就只有一个数据实例就不好办了。

事实上，在测试时我们还是需要继续使用批量归一化的，只是需要做些改动。在测试时，我们需要把原先训练时用到的批量均值和方差替换成**整个**训练数据的均值和方差。但
是当训练数据极大时，这个计算开销很大。因此，我们用移动平均的方法来近似计算（参见实现中的`moving_mean`和`moving_variance`）。

为了方便讨论批量归一化层的实现，我们先看下面这段代码来理解``Python``变量可以如何修改。

```{.python .input  n=7}
def batch_norm(X, gamma, beta, is_training, moving_mean, moving_variance,
               eps = 1e-5, moving_momentum = 0.9):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确的广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        # 变形使得可以正确的广播
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)
        
    # 均一化
    if is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)
        #!!! 更新全局的均值和方差
        moving_mean[:] = moving_momentum * moving_mean + (
            1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (
            1.0 - moving_momentum) * variance
    else:
        #!!! 测试阶段使用全局的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)
    
    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)
```

## 定义模型

我们尝试使用GPU运行本教程代码。

```{.python .input  n=8}
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
ctx
```

先定义参数。

```{.python .input  n=9}
weight_scale = .01

# output channels = 20, kernel = (5,5)
c1 = 20
W1 = nd.random.normal(shape=(c1,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(c1, ctx=ctx)

# batch norm 1
gamma1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(c1, ctx=ctx)
moving_variance1 = nd.zeros(c1, ctx=ctx)

# output channels = 50, kernel = (3,3)
c2 = 50
W2 = nd.random_normal(shape=(c2,c1,3,3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(c2, ctx=ctx)

# batch norm 2
gamma2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(c2, ctx=ctx)
moving_variance2 = nd.zeros(c2, ctx=ctx)

# output dim = 128
o3 = 128
W3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(o3, ctx=ctx)

# output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

# 注意这里moving_*是不需要更新的
params = [W1, b1, gamma1, beta1, 
          W2, b2, gamma2, beta2, 
          W3, b3, W4, b4]

for param in params:
    param.attach_grad()
```

下面定义模型。我们添加了批量归一化层。特别要注意我们添加的位置：在卷积层后，在激活函数前。

```{.python .input  n=10}
def net(X, is_training=False, verbose=False):
    X = X.as_in_context(W1.context)
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=c1)
    ### 添加了批量归一化层 
    h1_bn = batch_norm(h1_conv, gamma1, beta1, is_training, 
                       moving_mean1, moving_variance1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=c2)
    ### 添加了批量归一化层 
    h2_bn = batch_norm(h2_conv, gamma2, beta2, is_training, 
                       moving_mean2, moving_variance2)        
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear
```

下面我们训练并测试模型。

```{.python .input  n=11}
from mxnet import autograd 
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = 0.2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 总结

相比[卷积神经网络 --- 从0开始](cnn-scratch.md)来说，通过加入批量归一化层，即使是同样的参数，测试精度也有明显提升，尤其是最开始几轮。

## 练习

尝试调大学习率，看看跟前面比，是不是可以使用更大的学习率。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1253)
