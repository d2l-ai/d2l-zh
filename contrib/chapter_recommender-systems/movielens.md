# MovieLens数据集

用于推荐系统研究的数据集有很多，而其中[MovieLens](https://movielens.org/)数据集可能是最受欢迎的一个。1997年，为了收集评分数据用于研究目的，明尼苏达大学的GroupLens研究实验室创建了本数据集。MovieLens数据集在包括个性化推荐和社会心理学在内的多个研究领域起到了关键作用。

## 获取数据

MovieLens数据集托管在了[GroupLens](https://grouplens.org/datasets/movielens/)网站上。它包括多个可用版本。此处我们将使用其中的100K版本:cite:`Herlocker.Konstan.Borchers.ea.1999`。该数据集的10万条评分（从一星到五星）来自于943名用户对于1682部电影的评价。该数据集已经经过清洗处理，每个用户都至少有二十条评分数据。该数据集还提供了简单的人口统计信息，例如年龄、性别、风格和物品等。下载压缩包[ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)后解压得到`u.data`文件，其中包含了csv格式的10万条评分。文件夹中还有许多其他的文件，关于这些文件的详细说明可以在数据集的[README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)中找到。

在开始之前，让我们先导入运行本节试验所必须的模块。

```python
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

之后，我们下载MovieLens-100k数据集，以`DataFrame`格式加载交互数据。

```python
#@save
d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    """读取MovieLens-100k数据集"""
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 数据集统计

我们加载一下数据，然后手动检查一下前五条记录。如此一来，我们可以有效地了解数据结构并确保它们已经正确加载。

```python
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

每行数据由四列组成，其中包括用户id（1-943）、物品id（1-1682）、评分（1-5）和时间戳。我们可以据此构造一个大小为$n \times m$的矩阵，$n$和$m$分别代表用户和物品的数量。该数据集仅记录了已有的评分，因此我们可以把它叫作评分矩阵。由于该矩阵的数值可能用于表示精确的评分，因此我们将会互换地使用交互矩阵和评分矩阵。因为用户尚未评价大部分电影，因此评分矩阵中的大部分数值都是未知的。我们将会展示该矩阵的稀疏性。此处稀疏度的定义为`1 - 非零实体的数量 / ( 用户数量 * 物品数量)`。显然，该矩阵非常稀疏（稀疏度为93.695%）。现实世界中的系数矩阵可能会面临更严重的稀疏问题，该问题也一直是构建推荐系统所面临的长期挑战。一个可行的解决方案是，使用额外的辅助信息，例如用户和物品特征，来消除这种稀疏性。

接下来，我们绘制评分计数的分布情况。正如预期的一样，该分布看起来像是一个正态分布，大部分评分数据都集中在3-4之间。

```python
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 分割数据集

我们将数据集切分为训练集和测试集两部分。下面的函数提供了`随机`和`序列感知`两种分割模式。在`随机`模式下，该函数将忽略时间戳，然后随机切分100k的交互数据。在默认情况下，其中90%的数据将作为训练样本，剩余的10%用作测试样本。在`序列感知`模式下，我们利用时间戳排序用户的历史评分，然后将用户的最新评分用于测试，将其余的评分用作训练集。这一模式会用在序列感知推荐的小节中。

```python
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """以随机模式或者序列感知模式分割数据集"""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time) # 最新的评分
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()] 
        train_data = [item for item in train_list if item not in test_data] # 移除测试数据集中已有的评分
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio] # np.random.uniform(0,1,len(data)<1-test_ratio).tolist()
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

请注意，在日常实践中，除了测试集最好还要有验证集。但是简洁起见，我们在这里忽略了验证集。在这种情况下，我们的测试集可以视作保留的验证集。

## 加载数据

分割数据集后，为了方面使用，我们将训练集和测试集转化为了列表和字典（或者矩阵）。下面的函数按行读取dataframe中数据，并且从0开始枚举用户和物品的索引。该函数的返回值为用户、物品和评分列表，以及一个记录了交互数据的字典或者矩阵。我们可以将返回的类型指定为`显式`或者`隐式`。

```python
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    """加载MovieLens-100k数据集"""
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

接下来，为了在之后的章节使用数据集，我们整合上述步骤。这里得到的结果将会封装到`Dataset`和`DataLoader`之中。请注意，训练数据的`DataLoader`的`last_batch`选项被设置为了`rollover`（剩余样本将滚动到下一周期），而且数据的顺序也是打乱的。

```python
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    """分割并加载MovieLens-100k数据集"""
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 小结

* MovieLens广泛用于推荐系统研究。它是免费且公开可用的。
* 为了能在后续章节中使用，我们定义了一些函数用来下载和预处理MovieLens-100k数据集。

## 练习

* 你可以找到其他类似的推荐数据集吗？
* 你可以在[https://movielens.org/](https://movielens.org/)上浏览到关于MovieLens数据集更多的信息。

[Discussions](https://discuss.d2l.ai/t/399)

