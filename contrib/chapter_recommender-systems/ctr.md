# 特征丰富的推荐系统

交互数据是表明用户偏好和兴趣的最基本指标。它在之前介绍的模型中起到了关键性作用。但是，交互数据通常比较稀疏，有时也会包含一些噪音。为了解决这一问题，我们可以将物品特征、用户资料和交互发生时的上下文等辅助信息集成到推荐模型中。利用这些特征可以帮助做出更好的推荐，这是因为，当交互数据较为匮乏时，这些特征能够很好地预测用户的兴趣。因此，推荐模型需要具备处理这类特征的能力，并且能够捕捉到内容和上下文中的信息。为了演示这一模型的原理，我们将介绍一个在线广告点击率（click-through rate，CTR）预测任务:cite:`McMahan.Holt.Sculley.ea.2013`，并给出一个匿名的广告数据集。定向广告服务已在业界引起了广泛关注，它通常被设计成推荐引擎的形式。对于提高点击率来说，推荐匹配用户品味和兴趣的广告非常重要。

数字营销人员利用在线广告向客户展示广告信息。点击率是一种测量指标，它用于衡量客户的广告被点击的比例大小。点击率由以下公式计算得到：

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

点击率是预示算法有效性的重要指标，而点击率预测则是计算网站上的某些内容的点击概率的任务。点击率预测模型不仅能够用在定向广告系统中，它也能用在常规物品（电影、新闻和产品等）推荐系统、电子邮件广告系统和搜索引擎中。它还和用户满意度以及转化率有着紧密的关系。在设定营销目标时它也能有所帮助，因为它可以让广告商的预期切合实际。

```python
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## 在线广告数据集

由于互联网和移动计算技术的巨大进步，在线广告已经成为互联网行业的一项重要收入来源，并为其带来了绝大部分营收。所以，展示相关广告或者激发用户兴趣的广告，进而将普通用户转化为付费用户就变得非常重要。接下来我们将介绍一个在线广告数据集。它由34个字段组成，其中第一列表示广告是否被点击（1表示点击，0表示未点击）的目标变量。其他列都是标签化特征。这些列可能表示广告ID、站点ID、应用ID、设备ID、时间戳和用户信息等等。出于匿名和隐私保护的目的，这些列的真实语义并未公布。

下面的代码将从我们的服务器上下载该数据集，然后保存到本地文件夹中。

```python
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

训练集和测试集中分别包含了15000条和3000条数据。

## 数据集包装器

为了方便地从csv文件中加载广告数据，我们实现了一个名为`CTRDataset`的类，它可以被`DataLoader`调用。

```python
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1 # feature_cnts[feature_dim]->{value1:cnts1,...}
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()} # feat_mapper[feature_dim]-> set(v1,...)
            self.feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} # feat_map[feature_dim]-> {v1:idx1,...}
                                for i, cnt in feat_mapper.items()}
            self.defaults = {i: len(cnt) for i, cnt in feat_mapper.items()} # default index for feature[dim][value]
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy() # offset for feature in dim with value X
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

下面的例子将会加载训练数据，然后输出第一条记录。

```python
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

如你所见，这34个字段都是分类特征。每个数值都代表了对应条目的独热索引，标签$0$表示没有被点击。这里的`CTRDataset`也可以用来加载其他的数据集，例如Criteo展示广告挑战赛[数据集](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)和Avazu点击率预测[数据集](https://www.kaggle.com/c/avazu-ctr-prediction) 。

## 小结

* 点击率是一项很重要的指标，它能用于评估广告系统和推荐系统的性能。
* 点击率预测经常被转化为二分类问题。该问题的目标是，在给定特征后，预测广告或物品是否会被点击。

## 练习

* 你能使用`CTRDataset`加载Criteo和Avazu数据集吗？需要注意的是，Criteo包含了实数值特征，因此你可能需要稍微修改一下代码。

[讨论](https://discuss.d2l.ai/t/405)
