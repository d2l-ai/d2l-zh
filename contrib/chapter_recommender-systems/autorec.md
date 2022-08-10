# AutoRec：使用自动编码器预测评分

尽管矩阵分解模型在评分预测任务上取得了不错的表现，但是它本质上仍然只是一个线性模型。因此，这类模型无法描述能够预测用户偏好的非线性复杂关系。在本节，我们介绍一个基于非线性神经网络的协同过滤模型AutoRec:cite:`Sedhain.Menon.Sanner.ea.2015`。AutoRec是一个基于显式评分和自动编码器架构，并将非线性变换集成到协同过滤（collaborative filtering，CF）中的模型。神经网络已经被证明能够逼近任意连续函数，因此它能够解决矩阵分解的不足，增强矩阵分解的表示能力。

一方面，AutoRec和自动编码器拥有一样的架构：输入层、隐含层、重构层（输出层）。自动编码器是一种可以将输入复制到输出的神经网络，它能够将输入编码成隐含层（通常维度更低）表示。AutoRec没有显式地将用户和物品嵌入到低维空间。它使用交互矩阵的行或着列作为输入，然后在输出层重构交互矩阵。

另一方面，AutoRec和常规的自动编码器也有所不同。AutoRec专注于学习重构层输出，而不是隐含层表示。它使用一个只有部分数据的交互矩阵作为输入，然后试图重构一个完整的评分矩阵。同时，出于推荐的目的，重构过程在输出层中将输入层中缺失的条目补齐。

AutoRec有基于用户的和基于物品的两种变体。们在这里只介绍基于物品的AutoRec，基于用户的AutoRec可以据此导出。

## 模型

$\mathbf{R}_{*i}$表示评分矩阵的第$i$列，其中未知评分在默认情况下设置为0。神经网络的定义如下所示：

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

其中，$f(\cdot)$和$g(\cdot)$表示激活函数，$\mathbf{W}$和$\mathbf{V}$表示权重矩阵，$\mu$和$b$表示偏置。使用$h( \cdot )$表示AutoRec的整个网络，因此$h(\mathbf{R}_{*i})$表示评分矩阵第$i$列的重构结果。

下面的目标函数旨在降低重构误差。

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

其中，$\| \cdot \|_{\mathcal{O}}$表示在训练过程中只考虑已知评分。这也就是说，只有和已知输入相关联的权重矩阵才会在反向传播的过程中得到更新。

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import sys
npx.set_np()
```

## 模型实现

一个典型的自动编码器由编码器和解码器两部分组成。编码器将输入映射为隐含层表示，解码器则将隐含层表示映射到重构层。按照这一做法，我们使用全连接层构建编码器和解码器。在默认情况下，编码器的激活函数为`sigmoid`，而解码器不使用激活函数。为了减轻过拟合，在编码器后添加了dropout层。通过掩码屏蔽未定输入值的梯度，如此一来，只有已确定的评分才能帮助到模型的学习。

```python
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 重新实现评估器

由于输入和输出均已改变，为了能继续使用RMSE作为评估指标，我们需要重新实现评估函数。

```python
def evaluator(network, inter_matrix, test_data, ctx):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, ctx, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 训练和评估模型

现在，让我们使用MovieLens数据集训练和评估一下AutoRec模型。我们可以清楚地看到，测试集的RMSE低于矩阵分解模型，这表明神经网络在评分预测任务上的有效性。

```python
ctx = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        ctx, evaluator, inter_mat=test_inter_mat)
```

## 小结

* 我们可以使用自动编码器构建矩阵分解算法，同时还可以在其中整合非线性层和dropout正则化层。
* MovieLens-100K数据集上的实验表明，自动编码器的性能优于矩阵分解模型。

## 练习

* 修改自动编码器的隐含层维度，观察模型性能的变化。
* 尝试添加更多的隐含层。这对提高模型的性能有帮助吗？
* 可以找到更好的编码器激活函数和解码器激活函数吗？

[讨论](https://discuss.d2l.ai/t/401)
