# 线性回归的简洁实现
:label:`sec_linear_concise`

在过去的几年里，出于对深度学习强烈的兴趣，
许多公司、学者和业余爱好者开发了各种成熟的开源框架。
这些框架可以自动化基于梯度的学习算法中重复性的工作。
在 :numref:`sec_linear_scratch`中，我们只运用了：
（1）通过张量来进行数据存储和线性代数；
（2）通过自动微分来计算梯度。
实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用，
现代深度学习库也为我们实现了这些组件。

本节将介绍如何(**通过使用深度学习框架来简洁地实现**)
 :numref:`sec_linear_scratch`中的(**线性回归模型**)。

## 生成数据集

与 :numref:`sec_linear_scratch`中类似，我们首先[**生成数据集**]。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import paddle
```

```{.python .input}
#@tab mindspore
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor, grad
from d2l import mindspore as d2l

```


```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 读取数据集

我们可以[**调用框架中现有的API来读取数据**]。
我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。
此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个Gluon数据迭代器"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个TensorFlow数据迭代器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab paddle
#@save
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Paddle数据迭代器"""
    dataset = paddle.io.TensorDataset(data_arrays)
    return paddle.io.DataLoader(dataset, batch_size=batch_size,
                                shuffle=is_train,
                                return_list=True)
```

```{.python .input}
#@tab mindspore
class SyntheticData():
    def __init__(self):
        self.features, self.labels = d2l.synthetic_data(true_w, true_b, 1000)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def load_array(data_arrays, column_names, batch_size, is_train=True):  
    """构造一个MindSpore数据迭代器。"""
    dataset = ds.GeneratorDataset(data_arrays, column_names, shuffle=is_train)
    dataset = dataset.batch(batch_size)
    return dataset

```


```{.python .input}
#@tab mxnet, pytorch, tensorflow, paddle
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab mindspore
batch_size = 10
dataset = SyntheticData()
data_iter = load_array(dataset, ['features', 'labels'], batch_size)
```

使用`data_iter`的方式与我们在 :numref:`sec_linear_scratch`中使用`data_iter`函数的方式相同。为了验证是否正常工作，让我们读取并打印第一个小批量样本。
与 :numref:`sec_linear_scratch`不同，这里我们使用`iter`构造Python迭代器，并使用`next`从迭代器中获取第一项。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 定义模型

当我们在 :numref:`sec_linear_scratch`中实现线性回归时，
我们明确定义了模型参数变量，并编写了计算的代码，这样通过基本的线性代数运算得到输出。
但是，如果模型变得更加复杂，且当我们几乎每天都需要实现模型时，自然会想简化这个过程。
这种情况类似于为自己的博客从零开始编写网页。
做一两次是有益的，但如果每个新博客就需要工程师花一个月的时间重新开始编写网页，那并不高效。

对于标准深度学习模型，我们可以[**使用框架的预定义好的层**]。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。
我们首先定义一个模型变量`net`，它是一个`Sequential`类的实例。
`Sequential`类将多个层串联在一起。
当给定输入数据时，`Sequential`实例将数据传入到第一层，
然后将第一层的输出作为第二层的输入，以此类推。
在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。
但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`会让你熟悉“标准的流水线”。

回顾 :numref:`fig_single_neuron`中的单层网络架构，
这一单层被称为*全连接层*（fully-connected layer），
因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

:begin_tab:`mxnet`
在Gluon中，全连接层在`Dense`类中定义。
由于我们只想得到一个标量输出，所以我们将该数字设置为1。

值得注意的是，为了方便使用，Gluon并不要求我们为每个层指定输入的形状。
所以在这里，我们不需要告诉Gluon有多少输入进入这一层。
当我们第一次尝试通过我们的模型传递数据时，例如，当后面执行`net(X)`时，
Gluon会自动推断每个层输入的形状。
本节稍后将详细介绍这种工作机制。
:end_tab:

:begin_tab:`pytorch`
在PyTorch中，全连接层在`Linear`类中定义。
值得注意的是，我们将两个参数传递到`nn.Linear`中。
第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
:end_tab:

:begin_tab:`tensorflow`
在Keras中，全连接层在`Dense`类中定义。
由于我们只想得到一个标量输出，所以我们将该数字设置为1。


值得注意的是，为了方便使用，Keras不要求我们为每个层指定输入形状。
所以在这里，我们不需要告诉Keras有多少输入进入这一层。
当我们第一次尝试通过我们的模型传递数据时，例如，当后面执行`net(X)`时，
Keras会自动推断每个层输入的形状。
本节稍后将详细介绍这种工作机制。
:end_tab:

:begin_tab:`paddle`
在PaddlePaddle中，全连接层在`Linear`类中定义。
值得注意的是，我们将两个参数传递到`nn.Linear`中。
第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
:end_tab:

:begin_tab:`mindspore`
在mindspore中，全连接层在`Dense`类中定义。
值得注意的是，我们将两个参数传递到`nn.Dense`中。
第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
:end_tab:

```{.python .input}
# nn是神经网络的缩写
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# keras是TensorFlow的高级API
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

```{.python .input}
#@tab paddle
# nn是神经网络的缩写
from paddle import nn
net = nn.Sequential(nn.Linear(2, 1))
```
```{.python .input}
#@tab mindspore
# nn是神经网络的缩写
from mindspore import nn
from mindspore.common.initializer import initializer, Normal

net = nn.SequentialCell([nn.Dense(2, 1)])
```

## (**初始化模型参数**)

在使用`net`之前，我们需要初始化模型参数。
如在线性回归模型中的权重和偏置。
深度学习框架通常有预定义的方法来初始化参数。
在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，
偏置参数将初始化为零。

:begin_tab:`mxnet`
我们从MXNet导入`initializer`模块，这个模块提供了各种模型参数初始化方法。
Gluon将`init`作为访问`initializer`包的快捷方式。
我们可以通过调用`init.Normal(sigma=0.01)`来指定初始化权重的方法。
默认情况下，偏置参数初始化为零。
:end_tab:

:begin_tab:`pytorch`
正如我们在构造`nn.Linear`时指定输入和输出尺寸一样，
现在我们能直接访问参数以设定它们的初始值。
我们通过`net[0]`选择网络中的第一个图层，
然后使用`weight.data`和`bias.data`方法访问参数。
我们还可以使用替换方法`normal_`和`fill_`来重写参数值。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的`initializers`模块提供了多种模型参数初始化方法。
在Keras中最简单的指定初始化方法是在创建层时指定`kernel_initializer`。
在这里，我们重新创建了`net`。
:end_tab:

:begin_tab:`paddle`
正如我们在构造`nn.Linear`时指定输入和输出尺寸一样，
现在我们能直接访问参数以设定它们的初始值。 
我们通过`net[0]`选择网络中的第一个图层，
然后使用`weight`和`bias`方法访问参数。
我们可以通过调用`nn.initializer.Normal(0, 0.01)`来指定初始化权重的方法。
默认情况下，偏置参数初始化为零。
:end_tab:

:begin_tab:`mindspore`
正如我们在构造`nn.Dense`时指定输入和输出尺寸一样，
现在我们能直接访问参数以设定它们的初始值。
我们通过`net[0]`选择网络中的第一个图层，
然后使用`weight`和`bias`方法访问参数。
我们可以通过调用`initializer(init, shape, dtype)`来指定初始化权重的方法。
其中，`init`可以是常数，也可以是某种分布，比如正态分布`Normal(sigma=0.01, mean=0.0)`
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

```{.python .input}
#@tab paddle
weight_attr = paddle.ParamAttr(initializer=
                               paddle.nn.initializer.Normal(0, 0.01))
bias_attr = paddle.ParamAttr(initializer=None)
net = nn.Sequential(nn.Linear(2, 1, weight_attr=weight_attr,
                              bias_attr=bias_attr))
```

```{.python .input}
#@tab mindspore
net[0].weight = initializer(Normal(), net[0].weight.shape, mindspore.float32)
net[0].bias = initializer('zero', net[0].bias.shape, mindspore.float32)
```


:begin_tab:`mxnet`
上面的代码可能看起来很简单，但是这里有一个应该注意到的细节：
我们正在为网络初始化参数，而Gluon还不知道输入将有多少维!
网络的输入可能有2维，也可能有2000维。
Gluon让我们避免了这个问题，在后端执行时，初始化实际上是*推迟*（deferred）执行的，
只有在我们第一次尝试通过网络传递数据时才会进行真正的初始化。
请注意，因为参数还没有初始化，所以我们不能访问或操作它们。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上面的代码可能看起来很简单，但是这里有一个应该注意到的细节：
我们正在为网络初始化参数，而Keras还不知道输入将有多少维!
网络的输入可能有2维，也可能有2000维。
Keras让我们避免了这个问题，在后端执行时，初始化实际上是*推迟*（deferred）执行的。
只有在我们第一次尝试通过网络传递数据时才会进行真正的初始化。
请注意，因为参数还没有初始化，所以我们不能访问或操作它们。
:end_tab:

## 定义损失函数

:begin_tab:`mxnet`
在Gluon中，`loss`模块定义了各种损失函数。
在这个例子中，我们将使用Gluon中的均方误差（`L2Loss`）。
:end_tab:

:begin_tab:`pytorch`
[**计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数**]。
默认情况下，它返回所有样本损失的平均值。
:end_tab:

:begin_tab:`tensorflow`
计算均方误差使用的是`MeanSquaredError`类，也称为平方$L_2$范数。
默认情况下，它返回所有样本损失的平均值。
:end_tab:

:begin_tab:`paddle`
[**计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数**]。
默认情况下，它返回所有样本损失的平均值。
:end_tab:

:begin_tab:`mindspore`
[**计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数**]。
默认情况下，它返回所有样本损失的平均值。
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

```{.python .input}
#@tab paddle
loss = nn.MSELoss()
```

```{.python .input}
#@tab mindspore
loss = nn.MSELoss()
```

## 定义优化算法

:begin_tab:`mxnet`
小批量随机梯度下降算法是一种优化神经网络的标准工具，
Gluon通过`Trainer`类支持该算法的许多变种。
当我们实例化`Trainer`时，我们要指定优化的参数
（可通过`net.collect_params()`从我们的模型`net`中获得）、
我们希望使用的优化算法（`sgd`）以及优化算法所需的超参数字典。
小批量随机梯度下降只需要设置`learning_rate`值，这里设置为0.03。
:end_tab:

:begin_tab:`pytorch`
小批量随机梯度下降算法是一种优化神经网络的标准工具，
PyTorch在`optim`模块中实现了该算法的许多变种。
当我们(**实例化一个`SGD`实例**)时，我们要指定优化的参数
（可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。
小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。
:end_tab:

:begin_tab:`tensorflow`
小批量随机梯度下降算法是一种优化神经网络的标准工具，
Keras在`optimizers`模块中实现了该算法的许多变种。
小批量随机梯度下降只需要设置`learning_rate`值，这里设置为0.03。
:end_tab:

:begin_tab:`paddle`
小批量随机梯度下降算法是一种优化神经网络的标准工具，
PaddlePaddle在`optimizer`模块中实现了该算法的许多变种。
小批量随机梯度下降只需要设置`learning_rate`值，这里设置为0.03。
:end_tab:

:begin_tab:`mindspore`
小批量随机梯度下降算法是一种优化神经网络的标准工具，
mindspore在`nn`模块中实现了该算法的许多变种。
当我们(**实例化一个`SGD`实例**)时，我们要指定优化的参数
（可通过`net.trainable_params()`从我们的模型中获得）以及优化算法所需的超参数字典。
小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

```{.python .input}
#@tab paddle
trainer =  paddle.optimizer.SGD(learning_rate=0.03,
                                parameters=net.parameters())
```

```{.python .input}
#@tab mindspore
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.03)
```

## 训练

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。
我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。
当我们需要更复杂的模型时，高级API的优势将大大增加。
当我们有了所有的基本组件，[**训练过程代码与我们从零开始实现时所做的非常相似**]。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（`train_data`），
不停地从中获取一个小批量的输入和相应的标签。
对于每一个小批量，我们会进行以下步骤:

* 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
* 通过进行反向传播来计算梯度。
* 通过调用优化器来更新模型参数。

为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab paddle
num_epochs = 3
for epoch in range(num_epochs):
    for i,(X, y) in enumerate (data_iter()):
        l = loss(net(X) ,y)
        trainer.clear_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1},'f'loss {l}')
```

```{.python .input}
#@tab mindspore
# 构造前向网络
def forward_fn(x, y):
    y_hat = net(x)
    l = loss(y_hat, y)
    return l

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        grad_fn = mindspore.value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters)
        l, grads = grad_fn(X, y)
        optimizer(grads)
    l = forward_fn(mindspore.Tensor(data_iter.features), mindspore.Tensor(data_iter.labels))
    print(f'epoch {epoch + 1}, loss {l.asnumpy():f}')
```

下面我们[**比较生成数据集的真实参数和通过有限数据训练获得的模型参数**]。
要访问参数，我们首先从`net`访问所需的层，然后读取该层的权重和偏置。
正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。

```{.python .input}
w = net[0].weight.data()
print(f'w的估计误差： {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'b的估计误差： {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('w的估计误差：', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('w的估计误差：', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b的估计误差：', true_b - b)
```

```{.python .input}
#@tab paddle
w = net[0].weight
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias
print('b的估计误差：', true_b - b)
```

```{.python .input}
#@tab mindspore
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

## 小结

:begin_tab:`mxnet`
* 我们可以使用Gluon更简洁地实现模型。
* 在Gluon中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层，`loss`模块定义了许多常见的损失函数。
* MXNet的`initializer`模块提供了各种模型参数初始化方法。
* 维度和存储可以自动推断，但注意不要在初始化参数之前尝试访问参数。
:end_tab:

:begin_tab:`pytorch`
* 我们可以使用PyTorch的高级API更简洁地实现模型。
* 在PyTorch中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层和常见损失函数。
* 我们可以通过`_`结尾的方法将参数替换，从而初始化参数。
:end_tab:

:begin_tab:`tensorflow`
* 我们可以使用TensorFlow的高级API更简洁地实现模型。
* 在TensorFlow中，`data`模块提供了数据处理工具，`keras`模块定义了大量神经网络层和常见损耗函数。
* TensorFlow的`initializers`模块提供了多种模型参数初始化方法。
* 维度和存储可以自动推断，但注意不要在初始化参数之前尝试访问参数。
:end_tab:

:begin_tab:`mindspore`
* 我们可以使用MindSpore的高级API更简洁地实现模型。
* 在MindSpore中，`data`模块提供了数据处理工具，`nn`模块定义了大量神经网络层和常见损耗函数。
* MindSpore的`initializers`模块提供了多种模型参数初始化方法。
:end_tab:

## 练习

1. 如果将小批量的总损失替换为小批量损失的平均值，需要如何更改学习率？
1. 查看深度学习框架文档，它们提供了哪些损失函数和初始化方法？用Huber损失代替原损失，即
    $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \text{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \text{ 其它情况}\end{cases}$$
1. 如何访问线性回归的梯度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1782)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1781)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1780)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11690)
:end_tab:
