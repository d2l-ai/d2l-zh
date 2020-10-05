# 实战 Kaggle 比赛：预测房价
:label:`sec_kaggle_house`

现在，我们已经引入了一些基本工具来构建和培训深度网络，并通过包括体重衰减和丢弃法在内的技术正规化它们，我们已经准备好通过参加 Kaggle 竞赛将所有这些知识付诸实践。房价预测竞争是一个很好的开始。这些数据是相当通用的，不表现出可能需要专门模型的异国情调的结构（音频或视频可能）。这个数据集由巴特·德科克于 2011 年 :cite:`De-Cock.2011` 收集，涵盖了 2006-2010 年期间在爱姆斯的房价。它比著名的哈里森和鲁宾费尔德（1978 年）[Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) 大得多，拥有更多的例子和更多的功能。

在本节中，我们将介绍数据预处理、模型设计和超参数选择的详细信息。我们希望通过实践方法，您将获得一些直觉，指导您作为数据科学家的职业生涯。

## 下载和缓存数据集

在整本书中，我们将对各种已下载数据集进行培训和测试模型。在这里，我们实现了几个实用程序功能，以方便数据下载。首先，我们维护一个字典 `DATA_HUB`，该字典将字符串（数据集的 * 名称 *）映射到包含用于定位数据集的 URL 和验证文件完整性的 SHA-1 键的元组。所有这些数据集都托管在地址为 `DATA_URL` 的站点上。

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

DATA_HUB = dict()  #@save
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  #@save
```

以下 `download` 函数下载一个数据集，将其缓存在本地目录中（默认情况下为 `../data`）并返回已下载文件的名称。如果缓存目录中已存在与此数据集对应的文件，并且其 SHA-1 与存储在 `DATA_HUB` 中的文件相匹配，我们的代码将使用缓存的文件来避免冗余下载堵塞您的互联网。

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
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
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

我们还实现了两个额外的实用程序功能：一个是下载和解压缩 zip 或 tar 文件，另一个是将本书中使用的所有数据集从 `DATA_HUB` 下载到缓存目录中。

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## kaggle

[Kaggle](https://www.kaggle.com) 是一个受欢迎的平台，主办机器学习比赛。每个竞赛都以数据集为中心，许多竞赛由利益相关方赞助，他们为获胜的解决方案提供奖品。该平台帮助用户通过论坛和共享代码进行互动，促进协作和竞争。虽然排行榜的追逐往往不受控制，研究人员近视侧重于预处理步骤，而不是提出基本问题，但平台的客观性也具有巨大价值，该平台有助于在竞争方法和代码之间进行直接定量比较共享，以便每个人都可以了解什么做了和没有工作。如果您想参加 Kaggle 竞赛，您首先需要注册一个帐户（参见 :numref:`fig_kaggle`）。

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

在房价预测竞争页面上，如 :numref:`fig_house_pricing` 所示，您可以找到数据集（在 “数据” 选项卡下），提交预测，并查看您的排名，URL 就在这里：

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 访问和读取数据集

请注意，竞争数据分为培训和测试集。每个记录都包括房屋的属性值和属性，如街道类型、建筑年份、屋顶类型、地下室条件等。这些要素由各种数据类型组成。样本，施工年份由整数表示，屋顶类型表示离散类别分配，其他要素用浮点数表示。这里是现实使事情复杂化的地方：对于一些例子，一些数据完全缺失，缺失的值简单地标记为 “na”。每个房子的价格仅包括训练套装（毕竟这是一个竞争）。我们希望对训练集进行分区以创建验证集，但我们只能在将预测上传到 Kaggle 后在官方测试集中评估我们的模型。:numref:`fig_house_pricing` 竞赛选项卡上的 “数据” 选项卡包含下载数据的链接。

为了开始使用，我们将使用 `pandas` 读取和处理数据，我们在 :numref:`sec_pandas` 中介绍了这些数据。因此，您需要确保您已安装 `pandas`，然后再继续操作。幸运的是，如果你在 Jupyter 阅读，我们可以安装熊猫甚至不离开笔记本电脑。

```{.python .input}
# If pandas is not installed, please uncomment the following line:
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
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

为了方便起见，我们可以使用我们上面定义的脚本下载和缓存 Kaggle 外壳数据集。

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

我们使用 `pandas` 加载分别包含训练和测试数据的两个 csv 文件。

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

训练数据集包括 1460 个示例、80 个要素和 1 个标签，而测试数据包含 1459 个示例和 80 个要素。

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

让我们来看看前四个和后两个功能，以及前四个示例中的标签（SalePrice）。

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

我们可以看到，在每个样本中，第一个特征是 ID。这有助于模型识别每个训练样本。虽然这很方便，但它不会携带任何信息用于预测目的。因此，在将数据输入模型之前，我们将其从数据集中移除。

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 数据预处理

如上所述，我们有各种各样的数据类型。在开始建模之前，我们需要对数据进行预处理。让我们从数值特征开始。首先，我们应用启发式方法，将所有缺失值替换为相应要素的均值。然后，要将所有要素放在一个通用比例上，我们通过将要素重新缩放为零均值和单位方差来对数据进行标准化：

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

要验证这是否确实转换了我们的特征（变量），使其具有零均值和单位方差，请注意 $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ 和 $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$。直观地说，我们对数据进行标准化有两个原因。首先，它证明方便优化。其次，因为我们不知道 * 优先级 * 哪些要素将相关，所以我们不希望对分配给一个要素的系数比任何其他要特征的处罚多。

```{.python .input}
#@tab all
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

接下来我们处理离散值。这包括诸如 “消息分区” 之类的功能。我们用一个热编码替换它们，就像我们之前将多类标签转换为矢量一样（见 :numref:`subsec_classification-problem`）。实例，“消息分区” 假定值为 “RL” 和 “RM”。删除 “消息分区” 特征，创建两个新的指标功能 “Mszoning_rl” 和 “Mszoning_rm”，其值为 0 或 1。根据一个热编码，如果 “MSZing_rl” 的原始值为 “RL”，则 “Mszoning_rl” 为 1，“Mszoning_rm” 为 0。`pandas` 软件包自动为我们执行此操作。

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

您可以看到，此转换会将要素数量从 79 增加到 331。最后，通过 `values` 属性，我们可以从 `pandas` 格式中提取 NumPy 格式，并将其转换为用于训练的张量表示。

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## 培训

为了开始，我们训练一个带有平方损耗的线性模型。毫不奇怪，我们的线性模型不会导致竞争获胜的提交，但它提供了一个完整性检查，以查看数据中是否有意义的信息。如果我们不能比随机猜测更好，那么我们可能有一个很好的机会，我们有一个数据处理错误。如果事情发挥作用，线性模型将作为一个基准，让我们了解简单模型与最佳报告模型的接近程度，让我们感觉到我们应该从更奇特的模型中获得多少收益。

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
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

对于房价，与股票价格一样，我们关心的相对数量超过绝对数量。因此，我们倾向于更关心相对误差 $\frac{y - \hat{y}}{y}$ 而不是绝对误差 $y - \hat{y}$。实例如，如果我们的预测是 10 万美元，当估计俄亥俄州一栋房子的价格时，一个典型的房子的价值是 125,000 美元，那么我们可能正在做一个可怕的工作。另一方面，如果我们在洛斯阿尔托斯山，加利福尼亚州，这可能代表一个惊人的准确预测（在那里，房价中位数超过 400 万美元）。

解决这一问题的一个方法是衡量价格估计数的差异。事实上，这也是比赛用来评估报名质量的官方误差衡量标准。毕竟，对于 $|\log y - \log \hat{y}| \leq \delta$ 来说，一个小值 $\delta$ 转换为 $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$。这会导致预测价格对数与标签价格对数之间出现以下平方根误差：

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),
                                       torch.log(labels))))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

与前面的章节不同，我们的训练功能将依赖于 Adam 优化器（稍后我们将更详细地介绍它）。这个优化器的主要吸引力是，尽管没有为超参数优化提供更好的（有时甚至更糟），但人们往往发现它对初始学习率的敏感性要低得多。

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
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
    # The Adam optimization algorithm is used here
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
    # The Adam optimization algorithm is used here
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

## 折叠交叉验证

您可能还记得，我们在讨论如何处理模型选择 (:numref:`sec_model_selection`) 的一节中引入了 $K$ 倍交叉验证。我们将很好地利用它来选择模型设计和调整超参数。我们首先需要一个函数，在 $K$ 倍的交叉验证过程中返回数据的 $i^\mathrm{th}$ 倍。它将 $i^\mathrm{th}$ 段切割为验证数据，然后将其余部分作为训练数据返回。请注意，这不是处理数据的最有效方式，如果我们的数据集大得多，我们肯定会做更聪明的事情。但是，这种增加的复杂性可能会不必要地混淆我们的代码，因此由于我们的问题的简单性，我们可以在这里安全地省略它。

```{.python .input}
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

当我们在 $K$ 倍交叉验证中训练 $K$ 次时，会返回训练和验证误差平均值。

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## 型号选择

在此样本中，我们选择一组未经调整的超参数，并将其留给读者来改进模型。找到一个好的选择可能需要时间，具体取决于一个优化的变量数量。对于足够大的数据集和正常的超参数排序，$K$ 倍交叉验证往往对多次测试具有合理的弹性。但是，如果我们尝试了不合理的大量选项，我们可能会很幸运，并发现我们的验证性能不再代表真正的误差。

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

请注意，有时候，一组超参数的训练错误数量可能非常低，即使 $K$ 倍交叉验证上的错误数量要高得多。这表明我们是过于拟合。在整个训练过程中，您将希望监控这两个数字。较少过拟合可能表明我们的数据可以支持更强大的模型。巨大的过拟合可能表明，我们可以通过采用正则化技术获得好处。

##  提交关于 Kaggle 的预测

现在我们知道超参数的好选择应该是什么，我们可以使用所有数据来训练它（而不是仅仅是交叉验证切片中使用的数据的 $1-1/K$）。然后，我们通过这种方式获得的模型可以应用到测试集。将预测值保存在 csv 文件中将简化将结果上传到 Kaggle。

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

一个很好的完整性检查是查看测试集上的预测是否类似于 $K$ 倍交叉验证过程的预测。如果他们这样做，现在是时候将它们上传到 Kaggle。下面的代码将生成一个名为 `submission.csv` 的文件。

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

接下来，如 :numref:`fig_kaggle_submit2` 所示，我们可以在 Kaggle 上提交我们的预测，看看它们如何与测试集上的实际房价（标签）进行比较。这些步骤非常简单：

* 登录 Kaggle 网站并访问房价预测竞赛页面。
* 点击 “提交预测” 或 “延迟提交” 按钮（自本文起，该按钮位于右侧）。
* 点击页面底部虚线框中的 “上传提交文件” 按钮，然后选择要上传的预测文件。
* 点击页面底部的 “提交” 按钮查看您的结果。

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 摘要

* 真实数据通常包含不同数据类型的混合，需要进行预处理。
* 将实值数据重新调整为零均值和单位方差是一个很好的默认值。所以用它们的平均值替换缺失的值。
* 将类别要素转换为指标要素，使我们能够将它们像一个热门向量一样处理。
* 我们可以使用 $K$ 倍交叉验证来选择模型并调整超参数。
* 对数对于相对错误非常有用。

## 练习

1. 将您对此部分的预测提交给 Kaggle。你的预测有多好？
1. 您能否通过直接最小化价格对数来改进您的模型？如果您尝试预测价格而不是价格的对数会发生什么？
1. 用他们的平均值替换缺失值是否总是一个好主意？提示：你可以构造一个值不会随机丢失的情况吗？
1. 通过 $K$ 倍交叉验证来调整超参数，提高 Kaggle 上的分数。
1. 通过改进模型（例如，层、体重衰减和丢弃法）来提高分数。
1. 如果我们不像本节所做的那样对连续数值特征进行标准化，会发生什么情况？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
