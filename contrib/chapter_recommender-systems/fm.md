# 因子分解机

由Steffen Rendle发表于2010年的因子分解机（Factorization machines，FM）模型:cite:`Rendle.2010`是一种监督学习算法，它可以用于分类、回归和排序等任务。它很快就引起了人们的注意，而后成为一个流行的、有影响的，可以用于预测和推荐的方法。具体而言，它是线性回归模型和矩阵分解模型的推广。此外，它还让人想起具有多项式核函数的支持向量机。和线性回归以及矩阵分解相比，因子分解机的优势在于：一，它可以建模$\chi$路变量交互，其中$\chi$为多项式阶数，通常为二；二、与因子分解机相关联的快速优化算法可以将计算时间从多项式复杂度降低为线性复杂度，如此一来，对于高维度稀疏输入，它的计算效率会非常高。基于以上这些原因，因子分解机广泛应用于现代广告和产品推荐之中。技术细节和实现如下所示。

## 双路因子分解机

使用$x \in \mathbb{R}^d$表示样本的特征向量，使用$y$表示样本标签。这里的标签$y$可以是实数值，也可以是二分类任务的类别标签，例如点击/不点击。二阶的因子分解机模型可以定义为：

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

其中，$\mathbf{w}_0 \in \mathbb{R}$为全局偏置，$\mathbf{w} \in \mathbb{R}^d$为第i个变量的权重，$\mathbf{V} \in \mathbb{R}^{d\times k}$为特征嵌入，$\mathbf{v}_i$为$\mathbf{V}$的第i行，$k$为隐向量的维度，$\langle\cdot, \cdot \rangle$为两个向量的内积。$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$对第i个特征和第j个特征的交互进行建模。有一些特征交互很容易理解，因此它们可以由专家设计得到。但是，其他大多数特征交互都隐藏在了数据之中，很难被识别出来。因此，自动化地建模特征交互可以极大地减轻特征工程的工作量。显然，公式的前两项对应了线性回归，而最后一项则对应了因子分解机。如果特征$i$表示物品，而特征$j$代表用户，那么第三项则恰好是用户和物品嵌入向量的内积。需要注意的是，因子分解机可以推广到更高的阶数（阶数大于2）。不过，数值稳定性可能会削弱模型的泛化性能。

## 高效的优化标准

直接对因子分解机进行优化的复杂度为$\mathcal{O}(kd^2)$，因为每一对交互作用都需要计算。为了解决效率低下的问题，我们可以重组第三项。重组后计算时间复杂度为$\mathcal{O}(kd)$)，计算成本大大降低。逐对交互项重组后的公式如下所示：

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

重组后，模型的（优化）复杂度大大降低。此外，对于稀疏特征，只有非零元素才需要计算，因此整体复杂度与非零特征的数量呈线性关系。

为了学习因子分解机模型，我们在回归任务中使用MSE损失，在分类任务中使用交叉熵损失，在排序任务中使用贝叶斯个性化排序损失。标准优化器（如SGD和Adam等）均可用于参数优化过程。

```python
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os
import sys
npx.set_np()
```

## 模型实现

下面的代码实现了因子分解机。可以清楚地从中看到，因子分解机包含了一个线性回归模块和一个高效的特征交互模块。由于点击率预测是一个分类任务，所以我们在最后的得分上应用了sigmoid函数。

```python
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        
    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2 # self.embedding(x).shape == (b, num_inputs, num_factors)
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) # self.fc(x).shape == (b, num_inputs, 1)
        x = npx.sigmoid(x)
        return x
```

## 加载广告数据集

我们使用上一节定义的数据装饰器加载在线广告数据集。

```python
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 训练模型

之后，我们使用`Adam`优化器和`SigmoidBinaryCrossEntropyLoss`损失训练模型。默认的学习率为0.01，而嵌入尺寸则设置为20。

```python
ctx = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=ctx)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
```

## 小结

* 因子分解机是一种通用框架，它可以应用在回归、分类和排序等一系列不同的任务上。
* 对于预测任务来说，特征交互/交叉非常重要，而使用因子分解机可以高效地建模双路特征交互。

## 练习

* 你可以在Avazu、MovieLens和Criteo数据集上测试因子分级机吗？
* 改变嵌入尺寸，观察它对模型性能的影响。你能观察到和矩阵分解模型相似的模式吗？

[讨论](https://discuss.d2l.ai/t/406)
