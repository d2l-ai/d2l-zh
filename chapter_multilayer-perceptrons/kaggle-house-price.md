# 实战Kaggle比赛：预测房价
:label:`sec_kaggle_house`

之前几节我们学习了一些训练深度网络的基本工具和网络正则化的技术（如权重衰减、暂退法等）。
本节我们将通过Kaggle比赛，将所学知识付诸实践。
Kaggle的房价预测比赛是一个很好的起点。
此数据集由Bart de Cock于2011年收集 :cite:`De-Cock.2011`，
涵盖了2006-2010年期间亚利桑那州埃姆斯市的房价。
这个数据集是相当通用的，不会需要使用复杂模型架构。
它比哈里森和鲁宾菲尔德的[波士顿房价](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)
数据集要大得多，也有更多的特征。

本节我们将详细介绍数据预处理、模型设计和超参数选择。
通过亲身实践，你将获得一手经验，这些经验将指导你数据科学家职业生涯。

## 下载和缓存数据集

在整本书中，我们将下载不同的数据集，并训练和测试模型。
这里我们(**实现几个函数来方便下载数据**)。
首先，我们建立字典`DATA_HUB`，
它可以将数据集名称的字符串映射到数据集相关的二元组上，
这个二元组包含数据集的url和验证文件完整性的sha-1密钥。
所有类似的数据集都托管在地址为`DATA_URL`的站点上。

```{.python .input  n=1}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

下面的`download`函数用来下载数据集，
将数据集缓存在本地目录（默认情况下为`../data`）中，
并返回下载文件的名称。
如果缓存目录中已经存在此数据集文件，并且其sha-1与存储在`DATA_HUB`中的相匹配，
我们将使用缓存的文件，以避免重复的下载。

```{.python .input  n=2}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

我们还需实现两个实用函数：
一个将下载并解压缩一个zip或tar文件，
另一个是将本书中使用的所有数据集从`DATA_HUB`下载到缓存目录中。

```{.python .input  n=3}
#@tab all
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)
```

## Kaggle

[Kaggle](https://www.kaggle.com)是一个当今流行举办机器学习比赛的平台，
每场比赛都以至少一个数据集为中心。
许多比赛有赞助方，他们为获胜的解决方案提供奖金。
该平台帮助用户通过论坛和共享代码进行互动，促进协作和竞争。
虽然排行榜的追逐往往令人失去理智：
有些研究人员短视地专注于预处理步骤，而不是考虑基础性问题。
但一个客观的平台有巨大的价值：该平台促进了竞争方法之间的直接定量比较，以及代码共享。
这便于每个人都可以学习哪些方法起作用，哪些没有起作用。
如果你想参加Kaggle比赛，你首先需要注册一个账户（见 :numref:`fig_kaggle`）。

![Kaggle网站](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

在房价预测比赛页面（如 :numref:`fig_house_pricing` 所示），
你在"Data"选项卡下可以找到数据集。
你可以通过下面的网址提交预测，并查看排名：

>https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![房价预测比赛页面](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 访问和读取数据集

注意，竞赛数据分为训练集和测试集。
每条记录都包括房屋的属性值和属性，如街道类型、施工年份、屋顶类型、地下室状况等。
这些特征由各种数据类型组成。
例如，建筑年份由整数表示，屋顶类型由离散类别表示，其他特征由浮点数表示。
这就是现实让事情变得复杂的地方：例如，一些数据完全丢失了，缺失值被简单地标记为“NA”。
每套房子的价格只出现在训练集中（毕竟这是一场比赛）。
我们将希望划分训练集以创建验证集，但是在将预测结果上传到Kaggle之后，
我们只能在官方测试集中评估我们的模型。
在 :numref:`fig_house_pricing` 中，"Data"选项卡有下载数据的链接。

开始之前，我们将[**使用`pandas`读入并处理数据**]，
这是我们在 :numref:`sec_pandas`中引入的。
因此，在继续操作之前，你需要确保已安装`pandas`。
幸运的是，如果你正在用Jupyter阅读该书，你可以在不离开笔记本的情况下安装`pandas`。

```{.python .input}
# 如果你没有安装pandas，请取消下一行的注释
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# 如果你没有安装pandas，请取消下一行的注释
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# 如果你没有安装pandas，请取消下一行的注释
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

```{.python .input  n=4}
#@tab paddle
# 如果pandas没有被安装，请取消下一句的注释。
# !pip install pandas

%matplotlib inline
from d2l import paddle as d2l
import paddle
from paddle import nn
import pandas as pd
import numpy as np
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "grep: warning: GREP_OPTIONS is deprecated; please use an alias or script\n"
 }
]
```

为方便起见，我们可以使用上面定义的脚本下载并缓存Kaggle房屋数据集。

```{.python .input  n=5}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

我们使用`pandas`分别加载包含训练数据和测试数据的两个CSV文件。

```{.python .input  n=6}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\u6b63\u5728\u4ecehttp://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv\u4e0b\u8f7d../data/kaggle_house_pred_train.csv...\n\u6b63\u5728\u4ecehttp://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv\u4e0b\u8f7d../data/kaggle_house_pred_test.csv...\n"
 }
]
```

训练数据集包括1460个样本，每个样本80个特征和1个标签，
而测试数据集包含1459个样本，每个样本80个特征。

```{.python .input  n=7}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1460, 81)\n(1459, 80)\n"
 }
]
```

让我们看看[**前四个和最后两个特征，以及相应标签**]（房价）。

```{.python .input  n=8}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice\n0   1          60       RL         65.0       WD        Normal     208500\n1   2          20       RL         80.0       WD        Normal     181500\n2   3          60       RL         68.0       WD        Normal     223500\n3   4          70       RL         60.0       WD       Abnorml     140000\n"
 }
]
```

我们可以看到，(**在每个样本中，第一个特征是ID，**)
这有助于模型识别每个训练样本。
虽然这很方便，但它不携带任何用于预测的信息。
因此，在将数据提供给模型之前，(**我们将其从数据集中删除**)。

```{.python .input  n=9}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 数据预处理

如上所述，我们有各种各样的数据类型。
在开始建模之前，我们需要对数据进行预处理。
首先，我们[**将所有缺失的值替换为相应特征的平均值。**]然后，为了将所有特征放在一个共同的尺度上，
我们(**通过将特征重新缩放到零均值和单位方差来标准化数据**)：

$$x \leftarrow \frac{x - \mu}{\sigma},$$

其中$\mu$和$\sigma$分别表示均值和标准差。
现在，这些特征具有零均值和单位方差，即 $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$和$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$。
直观地说，我们标准化数据有两个原因：
首先，它方便优化。
其次，因为我们不知道哪些特征是相关的，
所以我们不想让惩罚分配给一个特征的系数比分配给其他任何特征的系数更大。

```{.python .input  n=10}
#@tab all
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

接下来，我们[**处理离散值。**]
这包括诸如“MSZoning”之类的特征。
(**我们用独热编码替换它们**)，
方法与前面将多类别标签转换为向量的方式相同
（请参见 :numref:`subsec_classification-problem`）。
例如，“MSZoning”包含值“RL”和“Rm”。
我们将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
根据独热编码，如果“MSZoning”的原始值为“RL”，
则：“MSZoning_RL”为1，“MSZoning_RM”为0。
`pandas`软件包会自动为我们实现这一点。

```{.python .input  n=11}
#@tab all
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "(2919, 331)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

你可以看到，此转换会将特征的总数量从79个增加到331个。
最后，通过`values`属性，我们可以
[**从`pandas`格式中提取NumPy格式，并将其转换为张量表示**]用于训练。

```{.python .input  n=12}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## [**训练**]

首先，我们训练一个带有损失平方的线性模型。
显然线性模型很难让我们在竞赛中获胜，但线性模型提供了一种健全性检查，
以查看数据中是否存在有意义的信息。
如果我们在这里不能做得比随机猜测更好，那么我们很可能存在数据处理错误。
如果一切顺利，线性模型将作为*基线*（baseline）模型，
让我们直观地知道最好的模型有超出简单的模型多少。

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input  n=13}
#@tab pytorch, paddle
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

房价就像股票价格一样，我们关心的是相对数量，而不是绝对数量。
因此，[**我们更关心相对误差$\frac{y - \hat{y}}{y}$，**]
而不是绝对误差$y - \hat{y}$。
例如，如果我们在俄亥俄州农村地区估计一栋房子的价格时，
假设我们的预测偏差了10万美元，
然而那里一栋典型的房子的价值是12.5万美元，
那么模型可能做得很糟糕。
另一方面，如果我们在加州豪宅区的预测出现同样的10万美元的偏差，
（在那里，房价中位数超过400万美元）
这可能是一个不错的预测。

(**解决这个问题的一种方法是用价格预测的对数来衡量差异**)。
事实上，这也是比赛中官方用来评价提交质量的误差指标。
即将$\delta$ for $|\log y - \log \hat{y}| \leq \delta$
转换为$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$。
这使得预测价格的对数与真实标签价格的对数之间出现以下均方根误差：

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

```{.python .input  n=14}
#@tab paddle
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = paddle.clip(net(features), 1, float('inf'))
    rmse = paddle.sqrt(loss(paddle.log(clipped_preds),
                            paddle.log(labels)))
    return rmse.item()
```

与前面的部分不同，[**我们的训练函数将借助Adam优化器**]
（我们将在后面章节更详细地描述它）。
Adam优化器的主要吸引力在于它对初始学习率不那么敏感。

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
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

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

```{.python .input  n=27}
#@tab paddle
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate*1.0, parameters=net.parameters(),weight_decay = weight_decay*1.0)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            optimizer.clear_grad()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## $K$折交叉验证

你可能还记得，我们在讨论模型选择的部分（ :numref:`sec_model_selection`）
中介绍了[**K折交叉验证**]，
它有助于模型选择和超参数调整。
我们首先需要定义一个函数，在$K$折交叉验证过程中返回第$i$折的数据。
具体地说，它选择第$i$个切片作为验证数据，其余部分作为训练数据。
注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。

```{.python .input  n=28}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

当我们在$K$折交叉验证中训练$K$次后，[**返回训练和验证误差的平均值**]。

```{.python .input  n=29}
#@tab all
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## [**模型选择**]

在本例中，我们选择了一组未调优的超参数，并将其留给读者来改进模型。
找到一组调优的超参数可能需要时间，这取决于一个人优化了多少变量。
有了足够大的数据集和合理设置的超参数，$K$折交叉验证往往对多次测试具有相当的稳定性。
然而，如果我们尝试了不合理的超参数，我们可能会发现验证效果不再代表真正的误差。

```{.python .input  n=30}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
```

```{.json .output n=30}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\u62981\uff0c\u8bad\u7ec3log rmse0.169814, \u9a8c\u8bc1log rmse0.156861\n\u62982\uff0c\u8bad\u7ec3log rmse0.162232, \u9a8c\u8bc1log rmse0.189275\n\u62983\uff0c\u8bad\u7ec3log rmse0.164553, \u9a8c\u8bc1log rmse0.168585\n\u62984\uff0c\u8bad\u7ec3log rmse0.167454, \u9a8c\u8bc1log rmse0.154226\n\u62985\uff0c\u8bad\u7ec3log rmse0.162457, \u9a8c\u8bc1log rmse0.182564\n5-\u6298\u9a8c\u8bc1: \u5e73\u5747\u8bad\u7ec3log rmse: 0.165302, \u5e73\u5747\u9a8c\u8bc1log rmse: 0.170302\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"257.521875pt\" height=\"180.65625pt\" viewBox=\"0 0 257.521875 180.65625\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n <metadata>\n  <rdf:RDF xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-05-08T04:30:56.179250</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.5.1, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 257.521875 180.65625 \nL 257.521875 0 \nL 0 0 \nL 0 180.65625 \nz\n\" style=\"fill: none\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \nL 240.778125 7.2 \nL 45.478125 7.2 \nz\n\" style=\"fill: #ffffff\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path d=\"M 82.959943 143.1 \nL 82.959943 7.2 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path id=\"m883a60846d\" d=\"M 0 0 \nL 0 3.5 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#m883a60846d\" x=\"82.959943\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 20 -->\n      <g transform=\"translate(76.597443 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-32\" d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" transform=\"scale(0.015625)\"/>\n        <path id=\"DejaVuSans-30\" d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path d=\"M 122.414489 143.1 \nL 122.414489 7.2 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use xlink:href=\"#m883a60846d\" x=\"122.414489\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 40 -->\n      <g transform=\"translate(116.051989 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-34\" d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path d=\"M 161.869034 143.1 \nL 161.869034 7.2 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use xlink:href=\"#m883a60846d\" x=\"161.869034\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 60 -->\n      <g transform=\"translate(155.506534 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-36\" d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path d=\"M 201.32358 143.1 \nL 201.32358 7.2 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use xlink:href=\"#m883a60846d\" x=\"201.32358\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 80 -->\n      <g transform=\"translate(194.96108 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-38\" d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path d=\"M 240.778125 143.1 \nL 240.778125 7.2 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use xlink:href=\"#m883a60846d\" x=\"240.778125\" y=\"143.1\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 100 -->\n      <g transform=\"translate(231.234375 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path id=\"DejaVuSans-31\" d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"63.623047\"/>\n       <use xlink:href=\"#DejaVuSans-30\" x=\"127.246094\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <g transform=\"translate(127.9 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-65\" d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-70\" d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6f\" d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-63\" d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-68\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use xlink:href=\"#DejaVuSans-70\" x=\"61.523438\"/>\n      <use xlink:href=\"#DejaVuSans-6f\" x=\"125\"/>\n      <use xlink:href=\"#DejaVuSans-63\" x=\"186.181641\"/>\n      <use xlink:href=\"#DejaVuSans-68\" x=\"241.162109\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path d=\"M 45.478125 64.855883 \nL 240.778125 64.855883 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #b0b0b0; stroke-width: 0.8; stroke-linecap: square\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path id=\"mb6d3b07708\" d=\"M 0 0 \nL -3.5 0 \n\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mb6d3b07708\" x=\"45.478125\" y=\"64.855883\" style=\"stroke: #000000; stroke-width: 0.8\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- $\\mathdefault{10^{0}}$ -->\n      <g transform=\"translate(20.878125 68.655102)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\" transform=\"translate(0 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(63.623047 0.765625)\"/>\n       <use xlink:href=\"#DejaVuSans-30\" transform=\"translate(128.203125 39.046875)scale(0.7)\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <defs>\n       <path id=\"mcd5e281b2e\" d=\"M 0 0 \nL -2 0 \n\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </defs>\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"127.469075\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_14\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"111.694957\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_15\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"100.503041\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_16\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"91.821917\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_17\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"84.728923\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_18\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"78.731881\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_8\">\n     <g id=\"line2d_19\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"73.537007\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_9\">\n     <g id=\"line2d_20\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"68.954804\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_10\">\n     <g id=\"line2d_21\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"37.889849\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_11\">\n     <g id=\"line2d_22\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"22.115731\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_12\">\n     <g id=\"line2d_23\">\n      <g>\n       <use xlink:href=\"#mcd5e281b2e\" x=\"45.478125\" y=\"10.923815\" style=\"stroke: #000000; stroke-width: 0.6\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_8\">\n     <!-- rmse -->\n     <g transform=\"translate(14.798437 87.669531)rotate(-90)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-72\" d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6d\" d=\"M 3328 2828 \nQ 3544 3216 3844 3400 \nQ 4144 3584 4550 3584 \nQ 5097 3584 5394 3201 \nQ 5691 2819 5691 2113 \nL 5691 0 \nL 5113 0 \nL 5113 2094 \nQ 5113 2597 4934 2840 \nQ 4756 3084 4391 3084 \nQ 3944 3084 3684 2787 \nQ 3425 2491 3425 1978 \nL 3425 0 \nL 2847 0 \nL 2847 2094 \nQ 2847 2600 2669 2842 \nQ 2491 3084 2119 3084 \nQ 1678 3084 1418 2786 \nQ 1159 2488 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1356 3278 1631 3431 \nQ 1906 3584 2284 3584 \nQ 2666 3584 2933 3390 \nQ 3200 3197 3328 2828 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-73\" d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use xlink:href=\"#DejaVuSans-6d\" x=\"39.363281\"/>\n      <use xlink:href=\"#DejaVuSans-73\" x=\"136.775391\"/>\n      <use xlink:href=\"#DejaVuSans-65\" x=\"188.875\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_24\">\n    <path d=\"M 45.478125 13.547232 \nL 47.450852 21.466068 \nL 49.42358 26.914542 \nL 51.396307 31.287999 \nL 53.369034 35.002557 \nL 55.341761 38.293894 \nL 57.314489 41.306225 \nL 59.287216 44.087305 \nL 61.259943 46.702334 \nL 63.23267 49.179164 \nL 65.205398 51.549203 \nL 67.178125 53.818004 \nL 69.150852 56.018587 \nL 71.12358 58.1197 \nL 73.096307 60.180584 \nL 75.069034 62.199107 \nL 77.041761 64.167588 \nL 79.014489 66.098891 \nL 80.987216 67.974921 \nL 82.959943 69.838162 \nL 84.93267 71.667254 \nL 86.905398 73.475542 \nL 88.878125 75.270762 \nL 90.850852 77.020849 \nL 92.82358 78.763653 \nL 94.796307 80.493572 \nL 96.769034 82.186478 \nL 98.741761 83.869527 \nL 100.714489 85.552947 \nL 102.687216 87.226706 \nL 104.659943 88.868314 \nL 106.63267 90.476248 \nL 108.605398 92.095172 \nL 110.578125 93.689873 \nL 112.550852 95.264206 \nL 114.52358 96.853422 \nL 116.496307 98.397868 \nL 118.469034 99.926867 \nL 120.441761 101.439499 \nL 122.414489 102.899253 \nL 124.387216 104.36949 \nL 126.359943 105.81328 \nL 128.33267 107.208145 \nL 130.305398 108.627261 \nL 132.278125 110.007441 \nL 134.250852 111.344587 \nL 136.22358 112.679563 \nL 138.196307 113.94645 \nL 140.169034 115.164486 \nL 142.141761 116.349883 \nL 144.114489 117.504923 \nL 146.087216 118.616046 \nL 148.059943 119.677748 \nL 150.03267 120.708871 \nL 152.005398 121.697649 \nL 153.978125 122.635095 \nL 155.950852 123.503356 \nL 157.92358 124.332256 \nL 159.896307 125.123108 \nL 161.869034 125.875082 \nL 163.841761 126.575204 \nL 165.814489 127.2474 \nL 167.787216 127.8448 \nL 169.759943 128.43017 \nL 171.73267 128.946751 \nL 173.705398 129.431003 \nL 175.678125 129.860093 \nL 177.650852 130.265537 \nL 179.62358 130.639267 \nL 181.596307 130.974451 \nL 183.569034 131.28687 \nL 185.541761 131.570384 \nL 187.514489 131.817826 \nL 189.487216 132.04983 \nL 191.459943 132.249141 \nL 193.43267 132.449703 \nL 195.405398 132.628484 \nL 197.378125 132.775245 \nL 199.350852 132.904537 \nL 201.32358 133.013682 \nL 203.296307 133.125973 \nL 205.269034 133.216423 \nL 207.241761 133.291592 \nL 209.214489 133.355659 \nL 211.187216 133.406148 \nL 213.159943 133.464121 \nL 215.13267 133.517643 \nL 217.105398 133.58246 \nL 219.078125 133.63427 \nL 221.050852 133.700346 \nL 223.02358 133.724754 \nL 224.996307 133.745698 \nL 226.969034 133.75661 \nL 228.941761 133.793897 \nL 230.914489 133.765963 \nL 232.887216 133.781942 \nL 234.859943 133.791792 \nL 236.83267 133.797512 \nL 238.805398 133.824205 \nL 240.778125 133.834302 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n   </g>\n   <g id=\"line2d_25\">\n    <path d=\"M 45.478125 13.377273 \nL 47.450852 21.271181 \nL 49.42358 26.701683 \nL 51.396307 31.054299 \nL 53.369034 34.757298 \nL 55.341761 38.027044 \nL 57.314489 41.024467 \nL 59.287216 43.787529 \nL 61.259943 46.387273 \nL 63.23267 48.847378 \nL 65.205398 51.202489 \nL 67.178125 53.457303 \nL 69.150852 55.642684 \nL 71.12358 57.732148 \nL 73.096307 59.779743 \nL 75.069034 61.785194 \nL 77.041761 63.741315 \nL 79.014489 65.656742 \nL 80.987216 67.519159 \nL 82.959943 69.369351 \nL 84.93267 71.189075 \nL 86.905398 72.986136 \nL 88.878125 74.777051 \nL 90.850852 76.517305 \nL 92.82358 78.252851 \nL 94.796307 79.975724 \nL 96.769034 81.663813 \nL 98.741761 83.341611 \nL 100.714489 85.023281 \nL 102.687216 86.697166 \nL 104.659943 88.341608 \nL 106.63267 89.954309 \nL 108.605398 91.579285 \nL 110.578125 93.182761 \nL 112.550852 94.772353 \nL 114.52358 96.372808 \nL 116.496307 97.937974 \nL 118.469034 99.48896 \nL 120.441761 101.029804 \nL 122.414489 102.519572 \nL 124.387216 104.023432 \nL 126.359943 105.505738 \nL 128.33267 106.944303 \nL 130.305398 108.412986 \nL 132.278125 109.852764 \nL 134.250852 111.248349 \nL 136.22358 112.651027 \nL 138.196307 113.990308 \nL 140.169034 115.284698 \nL 142.141761 116.555456 \nL 144.114489 117.800984 \nL 146.087216 119.009945 \nL 148.059943 120.168688 \nL 150.03267 121.315331 \nL 152.005398 122.419775 \nL 153.978125 123.470387 \nL 155.950852 124.451038 \nL 157.92358 125.399491 \nL 159.896307 126.310733 \nL 161.869034 127.175895 \nL 163.841761 127.997598 \nL 165.814489 128.789102 \nL 167.787216 129.499303 \nL 169.759943 130.200497 \nL 171.73267 130.828875 \nL 173.705398 131.416663 \nL 175.678125 131.946611 \nL 177.650852 132.456069 \nL 179.62358 132.930348 \nL 181.596307 133.348909 \nL 183.569034 133.746768 \nL 185.541761 134.108572 \nL 187.514489 134.422944 \nL 189.487216 134.723388 \nL 191.459943 134.968822 \nL 193.43267 135.235529 \nL 195.405398 135.468714 \nL 197.378125 135.659671 \nL 199.350852 135.824484 \nL 201.32358 135.969631 \nL 203.296307 136.12019 \nL 205.269034 136.240214 \nL 207.241761 136.33878 \nL 209.214489 136.426137 \nL 211.187216 136.470136 \nL 213.159943 136.544292 \nL 215.13267 136.623038 \nL 217.105398 136.69139 \nL 219.078125 136.747951 \nL 221.050852 136.816789 \nL 223.02358 136.841496 \nL 224.996307 136.854399 \nL 226.969034 136.869805 \nL 228.941761 136.912325 \nL 230.914489 136.896702 \nL 232.887216 136.909299 \nL 234.859943 136.919545 \nL 236.83267 136.914353 \nL 238.805398 136.922727 \nL 240.778125 136.920964 \n\" clip-path=\"url(#p49a4e7d3d1)\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 45.478125 143.1 \nL 45.478125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 240.778125 143.1 \nL 240.778125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 45.478125 7.2 \nL 240.778125 7.2 \n\" style=\"fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 177.826562 44.55625 \nL 233.778125 44.55625 \nQ 235.778125 44.55625 235.778125 42.55625 \nL 235.778125 14.2 \nQ 235.778125 12.2 233.778125 12.2 \nL 177.826562 12.2 \nQ 175.826562 12.2 175.826562 14.2 \nL 175.826562 42.55625 \nQ 175.826562 44.55625 177.826562 44.55625 \nz\n\" style=\"fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter\"/>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 179.826562 20.298437 \nL 189.826562 20.298437 \nL 199.826562 20.298437 \n\" style=\"fill: none; stroke: #1f77b4; stroke-width: 1.5; stroke-linecap: square\"/>\n    </g>\n    <g id=\"text_9\">\n     <!-- train -->\n     <g transform=\"translate(207.826562 23.798437)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-74\" d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-61\" d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-69\" d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6e\" d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use xlink:href=\"#DejaVuSans-72\" x=\"39.208984\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"80.322266\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"141.601562\"/>\n      <use xlink:href=\"#DejaVuSans-6e\" x=\"169.384766\"/>\n     </g>\n    </g>\n    <g id=\"line2d_27\">\n     <path d=\"M 179.826562 34.976562 \nL 189.826562 34.976562 \nL 199.826562 34.976562 \n\" style=\"fill: none; stroke-dasharray: 5.55,2.4; stroke-dashoffset: 0; stroke: #bf00bf; stroke-width: 1.5\"/>\n    </g>\n    <g id=\"text_10\">\n     <!-- valid -->\n     <g transform=\"translate(207.826562 38.476562)scale(0.1 -0.1)\">\n      <defs>\n       <path id=\"DejaVuSans-76\" d=\"M 191 3500 \nL 800 3500 \nL 1894 563 \nL 2988 3500 \nL 3597 3500 \nL 2284 0 \nL 1503 0 \nL 191 3500 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-6c\" d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-64\" d=\"M 2906 2969 \nL 2906 4863 \nL 3481 4863 \nL 3481 0 \nL 2906 0 \nL 2906 525 \nQ 2725 213 2448 61 \nQ 2172 -91 1784 -91 \nQ 1150 -91 751 415 \nQ 353 922 353 1747 \nQ 353 2572 751 3078 \nQ 1150 3584 1784 3584 \nQ 2172 3584 2448 3432 \nQ 2725 3281 2906 2969 \nz\nM 947 1747 \nQ 947 1113 1208 752 \nQ 1469 391 1925 391 \nQ 2381 391 2643 752 \nQ 2906 1113 2906 1747 \nQ 2906 2381 2643 2742 \nQ 2381 3103 1925 3103 \nQ 1469 3103 1208 2742 \nQ 947 2381 947 1747 \nz\n\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-76\"/>\n      <use xlink:href=\"#DejaVuSans-61\" x=\"59.179688\"/>\n      <use xlink:href=\"#DejaVuSans-6c\" x=\"120.458984\"/>\n      <use xlink:href=\"#DejaVuSans-69\" x=\"148.242188\"/>\n      <use xlink:href=\"#DejaVuSans-64\" x=\"176.025391\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p49a4e7d3d1\">\n   <rect x=\"45.478125\" y=\"7.2\" width=\"195.3\" height=\"135.9\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

请注意，有时一组超参数的训练误差可能非常低，但$K$折交叉验证的误差要高得多，
这表明模型过拟合了。
在整个训练过程中，你将希望监控训练误差和验证误差这两个数字。
较少的过拟合可能表明现有数据可以支撑一个更强大的模型，
较大的过拟合可能意味着我们可以通过正则化技术来获益。

##  [**提交你的Kaggle预测**]

既然我们知道应该选择什么样的超参数，
我们不妨使用所有数据对其进行训练
（而不是仅使用交叉验证中使用的$1-1/K$的数据）。
然后，我们通过这种方式获得的模型可以应用于测试集。
将预测保存在CSV文件中可以简化将结果上传到Kaggle的过程。

```{.python .input  n=31}
#@tab all
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = d2l.numpy(net(test_features))
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

如果测试集上的预测与$K$倍交叉验证过程中的预测相似，
那就是时候把它们上传到Kaggle了。
下面的代码将生成一个名为`submission.csv`的文件。

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

接下来，如 :numref:`fig_kaggle_submit2`中所示，
我们可以提交预测到Kaggle上，并查看在测试集上的预测与实际房价（标签）的比较情况。
步骤非常简单：

* 登录Kaggle网站，访问房价预测竞赛页面。
* 点击“Submit Predictions”或“Late Submission”按钮（在撰写本文时，该按钮位于右侧）。
* 点击页面底部虚线框中的“Upload Submission File”按钮，选择你要上传的预测文件。
* 点击页面底部的“Make Submission”按钮，即可查看你的结果。

![向Kaggle提交数据](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 小结

* 真实数据通常混合了不同的数据类型，需要进行预处理。
* 常用的预处理方法：将实值数据重新缩放为零均值和单位方法；用均值替换缺失值。
* 将类别特征转化为指标特征，可以使我们把这个特征当作一个独热向量来对待。
* 我们可以使用$K$折交叉验证来选择模型并调整超参数。
* 对数对于相对误差很有用。

## 练习

1. 把你的预测提交给Kaggle，它有多好？
1. 你能通过直接最小化价格的对数来改进你的模型吗？如果你试图预测价格的对数而不是价格，会发生什么？
1. 用平均值替换缺失值总是好主意吗？提示：你能构造一个不随机丢失值的情况吗？
1. 通过$K$折交叉验证调整超参数，从而提高Kaggle的得分。
1. 通过改进模型（例如，层、权重衰减和dropout）来提高分数。
1. 如果我们没有像本节所做的那样标准化连续的数值特征，会发生什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1823)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1824)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1825)
:end_tab:
