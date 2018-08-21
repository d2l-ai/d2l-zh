# 实战Kaggle比赛：房价预测

作为深度学习基础篇章的总结，我们将对本章内容学以致用。下面，让我们动手实战一个Kaggle比赛：房价预测。本节将提供简单的数据的预处理、模型的设计和超参数的选择。我们希望你通过动手实战、仔细观察实验现象、认真分析实验结果并不断调整方法，从而得到令自己满意的结果。

## Kaggle比赛

Kaggle（网站地址：https://www.kaggle.com ）是一个著名的供机器学习爱好者交流的平台。图3.8展示了Kaggle网站首页。为了便于提交结果，请大家注册Kaggle账号。

![Kaggle网站首页。](../img/kaggle.png)

我们可以在房价预测比赛的网页上了解比赛信息和参赛者成绩、下载数据集并提交自己的预测结果。该比赛的网页地址是

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques


图3.9展示了房价预测比赛的网页信息。

![房价预测比赛的网页信息。比赛数据集可通过点击“Data”标签获取。](../img/house_pricing.png)

## 获取和读取数据集

比赛数据分为训练数据集和测试数据集。两个数据集都包括每栋房子的特征，例如街道类型、建造年份、房顶类型、地下室状况等特征值。这些特征值有连续的数字、离散的标签甚至是缺失值“na”。只有训练数据集包括了每栋房子的价格。我们可以访问比赛网页，点击图3.9中的“Data”标签，并下载这些数据集。

下面，我们通过使用`pandas`读入数据，请简单介绍如何处理离散数据、处理丢失的数据特征和对数据进行标准化。在导入本节需要的包前请确保已安装`pandas`，否则请参考下面代码注释。

```{.python .input  n=3}
# 如果没有安装pandas，请反注释下面一行。
# !pip install pandas

import sys
sys.path.insert(0, '..')

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, init, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
```

数据解压放在`../data`目录里，它包括两个csv文件。下面使用pandas读取着这两个文件

```{.python .input  n=14}
train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')
```

训练数据集包括1460个样本、80个特征和1个标签。

```{.python .input  n=11}
train_data.shape
```

测试数据集包括1459个样本和80个特征。我们需要预测测试数据集上每个样本的标签。

```{.python .input  n=5}
test_data.shape
```

让我们来前4个样本的前4个特征、后2个特征和标签（SalePrice）：

```{.python .input  n=28}
train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]]
```

可以看到第一个特征是Id，它能帮助模型记住每个训练样本，但难以推广到测试样本，所以我们不使用它来训练。我们将训练数据剩下的79维特征和测试数据对应的特征放在一起，得到整个数据的特征。

```{.python .input  n=30}
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 预处理数据

我们对连续数值的特征做标准化处理：设该特征在训练数据集和测试数据集上的均值为$\mu$，标准差为$\sigma$。那么，我们可以将该特征的每个值先减去$\mu$再除以$\sigma$得到标准化后的每个特征值。对于值为NaN的特征，我们将其替换成特征均值，即为0。

```{.python .input  n=6}
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(all_features.mean())
```

接下来将离散数值转成指示特征。例如假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning\_RL和MSZoning\_RM，其值为0或1。如果一个样本在MSZoning里的值为RL，那么有MSZoning\_RL=0且MSZoning\_RM=1。

```{.python .input  n=7}
# dummy_na=True 将 NaN 也当做合法的特征值并为其创建指示特征。
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

可以看到这一步转换将特征数从79增加到了331。

最后，通过`values`属性得到Numpy格式的数据，并接下来转成NDArray方便后面的训练。

```{.python .input  n=9}
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
```

## 模型训练

我们使用一个基本的线性回归模型和平方损失函数来训练模型。

```{.python .input  n=13}
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

下面定义比赛用来评价模型的对数均方根误差。给定预测值$\hat y_1, \ldots, \hat y_n$和对应的真实标签$y_1,\ldots, y_n$，它的定义为

$$\left[\frac{1}{n}\sum_{i=1}^n\left(\log(y_i)-\log(\hat y_i)\right)^2\right]^{\frac{1}{2}}.$$

```{.python .input  n=11}
def log_rmse(net, train_features, train_labels):
    # 将小于 1 的预测值设成 1，使得取对数时数值更稳定。
    clipped_preds = nd.clip(net(train_features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), train_labels.log()).mean())
    return rmse.asscalar()
```

下面的训练函数跟本章中前几节的不同在于使用了Adam优化算法，相对之前使用的SGD，它对学习率相对不那么敏感。我们将在之后的“优化算法”一章里详细介绍它。

```{.python .input  n=14}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了 Adam 优化算法。
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## $K$折交叉验证

我们在[“欠拟合、过拟合和模型选择”](underfit-overfit.md)一节中介绍了$K$折交叉验证。我们将使用它来选择模型设计并调参。首先实现一个函数它能返回第$i$折交叉验证时需要的训练和验证数据。

```{.python .input}
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_test, y_test
```

在$K$折交叉验证中我们训练$K$次并返回平均训练和测试误差。

```{.python .input  n=15}
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, test_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, test_ls = train(net, *data, num_epochs, learning_rate,
                                  weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        test_l_sum += test_ls[-1]
        if i == 0:
            gb.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse',
                        range(1, num_epochs+1), test_ls, ['train', 'test'])
        print('fold %d, train rmse: %f, test rmse: %f' % (
            i, train_ls[-1], test_ls[-1]))
    return train_l_sum / k, test_l_sum / k
```

## 模型选择

我们使用一组简单的超参数并计算交叉验证误差。你可以改动这些超参数来尽可能减小平均测试误差。

```{.python .input  n=16}
k = 5
num_epochs = 100
verbose_epoch = num_epochs - 2
lr = 5
weight_decay = 0
batch_size = 64

train_l, test_l = k_fold(k, train_features, train_labels,
                         num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg test rmse: %f'
      % (k, train_l, test_l))
```

有时候你会发现一组参数的训练误差可以达到很低，但是在$K$折交叉验证上的误差可能反而较高。这种现象这很可能是由于过拟合造成的。因此，当训练误差特别低时，我们要观察$K$折交叉验证上的误差是否同时降低，以避免模型的过拟合。

## 预测并在Kaggle提交结果

我们首先定义预测函数。在预测之前，我们会使用完整的训练数据集来重新训练模型，并将预测结果存成提交需要的格式。

```{.python .input  n=18}
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    gb.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

设计好模型并调好超参数之后，下一步就是对测试数据集上的房屋样本做价格预测。如果我们得到跟交叉验证时差不多的训练误差，那么这个结果很可能是好的，可以在Kaggle上提交结果。

```{.python .input  n=19}
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

上述代码执行完之后会生成一个“submission.csv”文件。这个文件是符合Kaggle比赛要求的提交格式的。这时，我们可以在Kaggle上把我们预测得出的结果进行提交，并且查看与测试数据集上真实房价（标签）的误差。具体来说有以下几个步骤：你需要登录Kaggle网站，访问房价预测比赛网页，并点击右侧“Submit Predictions”或“Late Submission”按钮。然后，点击页面下方“Upload Submission File”选择需要提交的预测结果文件。最后，点击页面最下方的“Make Submission”按钮就可以查看结果了。如图3.11所示。

![Kaggle预测房价比赛的预测结果提交页面。](../img/kaggle_submit2.png)


## 小结

* 我们通常需要对真实数据做预处理。
* 我们可以使用$K$折交叉验证来选择模型并调参。


## 练习

* 在Kaggle提交本教程的预测结果。观察一下，这个结果能在Kaggle上拿到什么样的分数？
* 对照$K$折交叉验证结果，不断修改模型（例如添加隐藏层）和调参，你能提高Kaggle上的分数吗？
* 如果不使用本节中对连续数值特征的标准化处理，结果会有什么变化?
* 扫码直达讨论区，在社区交流方法和结果。你能发掘出其他更好的技巧吗？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1039)

![](../img/qr_kaggle-house-price.svg)
