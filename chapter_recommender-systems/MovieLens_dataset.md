

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-02 08:29:52
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-02 08:57:52
 * @Description:translate
 * @TODO::
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1092/chapter_recommender-systems/movielens.html
-->

# MovieLens数据集

有许多可用于推荐研究的数据集。其中，MovieLens数据集可能是最受欢迎的数据之一。MovieLens是一个基于web的非商业电影推荐系统。它创建于1997年，由明尼苏达大学(University of Minnesota)的研究实验室GroupLens运营，目的是为研究目的收集电影评级数据。MovieLens的数据，对包括个性化推荐和社会心理学在内的几项研究都至关重要。

## 获取数据

MovieLens数据集由[GroupLens](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)网站托管。有几个版本可供选择。我们将使用MovieLens的100K数据集[Herlocker et al.， 1999](http://preview.d2l.ai/d2l-zh/PR-1092/chapter_references/zreferences.html#herlocker-konstan-borchers-ea-1999)。这个数据集包括从1到5星，从943个用户的1682部电影的$10,000$个评分，。它已经被清理干净，每个用户至少已经给20部电影评分。一些简单的人口统计信息，如用户和项目的年龄、性别、类型。我们可以下载[ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)并解压缩u.data，其中包含所有的csv格式的$100,000$评级。该文件夹中还有许多其他文件，每个文件的详细描述可以在数据集的[README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt)文件中找到。

TODO:CODE

首先，让我们导入运行本节实验所需的包。

TODO:CODE

然后，我们下载MovieLens 100k数据集，并以DataFrame的形式加载交互。

## 数据集的统计学

让我们加载数据并手动检查前五个记录。 这是学习数据结构并验证是否已正确加载它们的有效方法。

TODO：CODE

我们可以看到每一行包含四列，包括“user id” 1-943，“项目ID” 1-1682，“rating” 1-5和“timestamp”。 我们可以构造一个大小为$n×m$的交互矩阵，其中n和m分别是用户数和项目数。 此数据集仅记录现有的评分，因此我们也可以将其称为评分矩阵，并且在此矩阵的值表示准确的评分的情况下，我们将互换使用交互矩阵和评分矩阵。评分矩阵中的大多数值都是未知的，因为用户尚未对大多数电影进行评分。 我们还显示了该数据集的稀疏性。 稀疏度定义为 $1 - number of nonzero entries / ( number of users * number of items)$。 显然，交互矩阵非常稀疏（即稀疏度sparsity = 93.695％）。 现实世界的数据集可能会遭受更大程度的稀疏性，并且一直是构建推荐系统的长期挑战。 可行的解决方案是使用其他辅助信息（例如用户/项目功能）来减轻稀疏性。

然后，我们绘制不同评级的数量的分布。 正如预期的那样，它似乎是正态分布，大多数评分集中在3-4。

TODO：CODE

# 分割数据集

我们将数据集分为训练集和测试集。下面的函数提供了两种分割模式，包括随机（$random$)和顺序感知（$seq-aware$）。在随机模式下，函数在不考虑时间戳的情况下对100k个交互进行随机分割，默认使用90%的数据作为训练样本，其余10%作为测试样本。在seq-aware模式中，我们会忽略用户最近评分的条目进行测试，并将用户历史交互作为训练集。用户历史交互将根据时间戳从最早到最新进行排序。此模式将在支持序列的推荐中使用。

TODO:CODE

请注意，除了测试集之外，在实践中使用验证集是一个好的实践。但是，为了简洁起见，我们省略了它。在这种情况下，我们的测试集可以看作是我们的外置验证集。

TODO:CODE

## 加载数据

分割数据集后，为了方便起见，我们将训练集和测试集转换为列表和字典/矩阵。 以下函数逐行读取数据帧，并枚举从零开始的用户/项索引。 然后，该函数返回用户列表，项目，等级和记录交互的字典/矩阵。我们可以将反馈的类型指定为显性或隐性。

然后，我们将上述步骤放在一起，将在下一节中使用。结果用Dataset和DataLoader包装。注意，用于训练数据的最后一批DataLoader被设置为翻转模式(剩余的样本被滚动到下一个epoch)，并且顺序被重新排序。

TODO:CODE


## 小结

- MovieLens数据集广泛用于搜索推荐。它是公开的，可以免费使用。
- 我们定义了下载和预处理MovieLens 100k数据集的函数，以便在后面的章节中进一步使用。

## 练习

1. 你还找到什么其他的推荐系统
1. 去[https://movielens.org/](https://movielens.org/)查找更多关于MovieLens的信息。
