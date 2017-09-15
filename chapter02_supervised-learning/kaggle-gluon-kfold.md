# 实战Kaggle比赛——使用Gluon预测房价和K折交叉验证

本章介绍如何使用``Gluon``来实战[Kaggle比赛](https://www.kaggle.com)。我们以[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)为例，为大家提供一整套实战中常常需要的工具，例如**K折交叉验证**。我们还以pandas为工具介绍如何对**真实世界**中的数据进行重要的预处理，例如：

* 处理离散数据
* 处理丢失的数据特征
* 对数据进行标准化

需要注意的是，本章仅提供一些基本实战流程供大家参考。对于数据的预处理、模型的设计和参数的选择等，我们特意只提供最基础的版本。希望大家一定要通过动手实战、仔细观察实验现象、认真分析实验结果、不断调整方法，从而得到令自己满意的结果。

这是一次宝贵的实战机会，我们相信你一定能从动手的过程中学到很多。

> 请务必 Get your hands dirty。

## Kaggle中的房价预测问题

[Kaggle](https://www.kaggle.com)是一个著名的供机器学习爱好者交流的平台。

为了便于提交结果，请大家注册[Kaggle](https://www.kaggle.com)账号。请注意，**目前Kaggle仅限每个账号一天以内10次提交结果的机会**。所以提交结果前务必三思。


我们以[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)为例教大家如何实战一次Kaggle比赛。请大家在动手开始之前点击[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)了解相关信息。

## 读入数据

比赛数据分为训练数据集和测试数据集。两个数据集都包括每个房子的特征，例如街道类型、建造年份、房顶类型、地下室状况等特征值。这些特征值有连续的数字、离散的标签甚至是缺失值'na'。只有训练数据集包括了我们需要在测试数据集中预测的每个房子的价格。数据可以从[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)中下载。

我们通过使用pandas读入数据。请确保安装了pandas (pip install pandas)。

```{.python .input  n=1}
import pandas as pd
import numpy as np

train = pd.read_csv("../data/kaggle_house_pred_train.csv")
test = pd.read_csv("../data/kaggle_house_pred_test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))
```

我们看看数据长什么样子。

```{.python .input  n=8}
train.head()
```

```{.json .output n=8}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Id</th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>...</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n      <th>SalePrice</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>8450</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>208500</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>20</td>\n      <td>RL</td>\n      <td>80.0</td>\n      <td>9600</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>5</td>\n      <td>2007</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>181500</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>68.0</td>\n      <td>11250</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>9</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>223500</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>70</td>\n      <td>RL</td>\n      <td>60.0</td>\n      <td>9550</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2006</td>\n      <td>WD</td>\n      <td>Abnorml</td>\n      <td>140000</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>84.0</td>\n      <td>14260</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>12</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>250000</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 81 columns</p>\n</div>",
   "text/plain": "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n0   1          60       RL         65.0     8450   Pave   NaN      Reg   \n1   2          20       RL         80.0     9600   Pave   NaN      Reg   \n2   3          60       RL         68.0    11250   Pave   NaN      IR1   \n3   4          70       RL         60.0     9550   Pave   NaN      IR1   \n4   5          60       RL         84.0    14260   Pave   NaN      IR1   \n\n  LandContour Utilities    ...     PoolArea PoolQC Fence MiscFeature MiscVal  \\\n0         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n1         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n2         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n3         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n4         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n\n  MoSold YrSold  SaleType  SaleCondition  SalePrice  \n0      2   2008        WD         Normal     208500  \n1      5   2007        WD         Normal     181500  \n2      9   2008        WD         Normal     223500  \n3      2   2006        WD        Abnorml     140000  \n4     12   2008        WD         Normal     250000  \n\n[5 rows x 81 columns]"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

数据大小如下。

```{.python .input  n=6}
train.shape
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(1460, 81)"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=7}
test.shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(1459, 80)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 预处理数据

我们使用pandas对数值特征做标准化处理：

$$x_i = \frac{x_i - \mathbb{E} x_i}{\text{std}(x_i)}。$$

```{.python .input  n=9}
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
```

现在把离散数据点转换成数值标签。

```{.python .input  n=10}
all_X = pd.get_dummies(all_X, dummy_na=True)
```

把缺失数据用本特征的平均值估计。

```{.python .input  n=11}
all_X = all_X.fillna(all_X.mean())
```

下面把数据转换一下格式。

```{.python .input  n=12}
num_train = train.shape[0]

X_train = all_X[:num_train]
X_test = all_X[num_train:]
y_train = train.SalePrice

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
```

## 导入NDArray格式数据

为了便于和``Gluon``交互，我们需要导入NDArray格式数据。

```{.python .input  n=13}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

X_test = nd.array(X_test)
```

我们把损失函数定义为平方误差。

```{.python .input  n=14}
square_loss = gluon.loss.L2Loss()
```

我们定义比赛中测量结果用的函数。

```{.python .input  n=15}
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)
```

## 定义模型

我们将模型的定义放在一个函数里供多次调用。这是一个基本的线性回归模型。

```{.python .input  n=16}
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

我们定义一个训练的函数，这样在跑不同的实验时不需要重复实现相同的步骤。

```{.python .input  n=17}
def train(net, X_train, y_train, epochs, verbose_epoch, learning_rate,
          weight_decay):
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size,
                                            shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            avg_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, avg_loss))
    return net, avg_loss
```

## K折交叉验证

在[过拟合](underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。事实上，当我们调参时，往往需要基于K折交叉验证。

> 在K折交叉验证中，我们把初始采样分割成$K$个子样本，一个单独的子样本被保留作为验证模型的数据，其他$K-1$个样本用来训练。

我们关心K次验证模型的测试结果的平均值和训练误差的平均值，因此我们定义K折交叉验证函数如下。

```{.python .input  n=18}
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_idx in range(k):
        X_val_test = X_train[test_idx * fold_size: (test_idx + 1) *
                                                   fold_size, :]
        y_val_test = y_train[test_idx * fold_size: (test_idx + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_idx:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        net_trained, train_loss = train(net, X_val_train, y_val_train, epochs,
                                        verbose_epoch, learning_rate,
                                        weight_decay)
        train_loss_sum += train_loss
        test_loss = get_rmse_log(net_trained, X_val_test, y_val_test)
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k
```

### 训练模型并交叉验证

以下的模型参数都是可以调的。

```{.python .input  n=23}
k = 5
epochs = 100
verbose_epoch = 95
learning_rate = 5
weight_decay = 0.0
```

给定以上调好的参数，接下来我们训练并交叉验证我们的模型。

```{.python .input  n=24}
train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                           y_train, learning_rate, weight_decay)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
      (k, train_loss, test_loss))
```

```{.json .output n=24}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 96, train loss: 0.201754\nEpoch 97, train loss: 0.199677\nEpoch 98, train loss: 0.197680\nEpoch 99, train loss: 0.195824\nTest loss: 0.188714\nEpoch 96, train loss: 0.197866\nEpoch 97, train loss: 0.195674\nEpoch 98, train loss: 0.193603\nEpoch 99, train loss: 0.191686\nTest loss: 0.211255\nEpoch 96, train loss: 0.199089\nEpoch 97, train loss: 0.196955\nEpoch 98, train loss: 0.194979\nEpoch 99, train loss: 0.193046\nTest loss: 0.201710\nEpoch 96, train loss: 0.201462\nEpoch 97, train loss: 0.199319\nEpoch 98, train loss: 0.197295\nEpoch 99, train loss: 0.195372\nTest loss: 0.178485\nEpoch 96, train loss: 0.196540\nEpoch 97, train loss: 0.194323\nEpoch 98, train loss: 0.192214\nEpoch 99, train loss: 0.190255\nTest loss: 0.206328\n5-fold validation: Avg train loss: 0.193237, Avg test loss: 0.197298\n"
 }
]
```

即便训练误差可以达到很低（调好参数之后），但是K折交叉验证上的误差可能更高。当训练误差特别低时，要观察K折交叉验证上的误差是否同时降低并小心过拟合。我们通常依赖K折交叉验证误差结果来调节参数。


## 预测并在Kaggle提交预测结果

我们定义预测函数。

```{.python .input  n=25}
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
    net = get_net()
    net_trained, _ = train(net, X_train, y_train, epochs, verbose_epoch,
                        learning_rate, weight_decay)
    preds = net_trained(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

调好参数以后，下面我们预测并在Kaggle提交预测结果。

```{.python .input  n=26}
learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
      weight_decay)
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 96, train loss: 0.171103\nEpoch 97, train loss: 0.170528\nEpoch 98, train loss: 0.169987\nEpoch 99, train loss: 0.169484\n"
 }
]
```

请注意，**目前Kaggle仅限每个账号一天以内10次提交结果的机会**。所以提交结果前务必三思。

## 作业

* 运行本教程，并提交本教程的预测结果，观察这个结果能在Kaggle上拿到什么样的loss
* 通过重新设计模型、调参并对照K折交叉验证结果，新模型是否比其他小伙伴的更好？
* 不使用对数值特征做标准化处理能拿到什么样的loss？
* 还有什么其他办法可以继续改进模型？
