# word2vec 实现

前两节描述了不同的词嵌入模型和近似训练法。本节中，我们将以[“词嵌入：word2vec”](word2vec.md)一节中的跳字模型和负采样为例，介绍在语料库上训练词嵌入模型的实现。我们还会介绍一些实现中的技巧，例如二次采样（subsampling）和掩码（mask）变量。

首先导入实验所需的包或模块。

```{.python .input  n=263}
import sys
sys.path.insert(0, '..')

import collections
import gluonbook as gb
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile
```

## 处理数据集

Penn Tree Bank（PTB）是一个常用的小型语料库 [1]。它采样自华尔街日报的文章，训练集、验证集和测试集。我们将在PTB的训练集上训练词嵌入模型。该数据集的每一行为一个句子。句子中的每个词由空格隔开。

```{.python .input  n=264}
with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    with zin.open('ptb/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        # st 是 sentence 在循环中的缩写。
        raw_dataset = [st.split() for st in lines]  
'# sentences: %d' % len(raw_dataset)
```

对于数据集的前三个句子，打印每个句子的词数和前五个词。这个数据集中句尾符被表示成`<eos>`，生僻词表示成`<unk>`，数字则被替换成了`N`。

```{.python .input  n=265}
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])
```

### 建立词语索引

为了计算简单，我们只保留在数据集中至少出现5次的词。

```{.python .input  n=266}
# tk 是 token 的在循环中的缩写。
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
```

然后将词映射到整数索引。

```{.python .input  n=267}
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] 
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
'#tokens: %d' % num_tokens
```

### 二次采样

本文数据中通常会出现高频词，例如英文中的“the”、“a”和“in”。通常来说，一个词，例如“China”，和较低频词，例如“Beijing”，同时出现比和较高频词，例如“the”，同时出现对训练词嵌入更加有帮助。但由于高频出现机会更高，它们对模型训练影响更大。因此训练词嵌入模型时常对高频词进行采样，来降低它们的影响度。word2vec中使用的一个方法是按照一定概率丢弃每个词。对于词$w_i$，它的丢弃概率为：

$$ \mathbb{P}(w_i) = \max\left(1 - \sqrt{\frac{tN}{c_i}}, 0\right),$$ 

这里$N$是数据集的总词数（`num_tokens`），$c_i$是词$w_i$在数据集中出现的次数，常数$t$是一个超参数。可以看到，当$c_i>tN$时，我们将对其进行丢弃，而且出现次数越多，丢弃概率越大。这里我们设$t=10^{-4}$，那么$tN=88.71$。

```{.python .input  n=268}
discard = lambda idx: random.uniform(0, 1) < 1 - math.sqrt(
    1e-4 / counter[idx_to_token[idx]] * num_tokens)  # 判断是否保留。
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
'#tokens: %d' % sum([len(st) for st in subsampled_dataset])
```

可以看到二次采样后我们去掉了几乎一半的词。下面比较一个词采样前后出现的次数，可以看到高频词“the”采样到了1/20一下，

```{.python .input  n=269}
get_count = lambda token: '%s: before=%d, after=%d' % (
    token, sum([st.count(token_to_idx[token]) for st in dataset]), 
    sum([st.count(token_to_idx[token]) for st in subsampled_dataset]))
get_count(b'the')
```

但低频词则完整保留了下来。

```{.python .input  n=270}
get_count(b'join')
```

### 提取中心词和背景词

在跳字模型中，我们将每个离中心词不超过固定距离的词作为它的背景词。下面定义获取所有中心词——背景词对的函数。它每次在整数1和`max_window_size`之间均匀随机采样一个整数作为上下文窗口大小。

```{.python .input  n=271}
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有 2 个词才可能组成一对“中心词-背景词”。
            continue
        centers += st
        for centor in range(len(st)):
            # 均匀随机生成窗口大小（包括 1 和 max_window_size）。
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, centor - window_size),
                                 min(len(st), centor + 1 + window_size)))
            indices.remove(centor)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts
```

下面我们生成一个人工数据集，其中含有词数分别为3和2的两个句子。设最大上下文窗口为2，然后打印所有中心词和它们的背景词。

```{.python .input  n=272}
tiny_dataset = [list(range(4)), list(range(5, 7))]
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

在本节的实验中，我们设最大上下文窗口大小为5。下面提取数据集中所有的中心词及其背景词。

```{.python .input  n=273}
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
```

## 负采样

我们将使用上节介绍的负采样来进行近似训练。对每个上下文窗口中的中心词，我们随机采样$K$乘以这个窗口中背景词个数个噪音词。噪声词采样概率$\mathbb{P}(w)$设为word2vec中使用的$w$词频与总词频的比的3/4次方。且保证每个噪音词不在这个上下文窗口中出现。然后使用$K=5$来生产噪音词。

```{.python .input  n=274}
sampling_weights = [counter[w]**0.75 for w in idx_to_token]

def get_negatives(all_centers, all_contexts, sampling_weights, K):
    all_negatives, candidates, i = [], [], 0
    for center, contexts in zip(all_centers, all_contexts):
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(candidates):
                # 一次采样多个词，可以使得计算更快。
                population = list(range(len(sampling_weights)))
                i, candidates = 0, random.choices(
                    population, sampling_weights, k=K*len(all_centers))                        
            neg, i = candidates[i], i + 1
            # 确保噪音词既不是中心词也不是背景词。
            if neg != center and neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives
    
all_negatives = get_negatives(all_centers, all_contexts, sampling_weights, 5)
```

## 数据读取

我们从数据集中提取了所有中心词`all_centers`，和每个中心词对应的背景词`all_contexts`和噪音词`all_negatives`。接下来我们使用随机小批量来读取它们。

在一个小批量数据中，第$i$个样本包括一个中心词和它对应的$n_i$个背景词和$m_i$个噪声词。但由于每个样本的上下文窗口大小可能不一样，这样背景词与噪声词数量和$n_i+m_i$也会不同。在构造小批量时，我们将每个样本的背景词和噪声词连结在一起，并将长度固定为$l=\max_i n_i+m_i$，也就是形状为（批量大小，$l$）。如果某个样本的长度不够，我们添加0来补齐长度。同时我们构造同样形状的标号：1表示背景词，0表示其他。以及同样形状的掩码，1表示背景词或噪声词，0表示填充。

下面实现给定一个长度为批量大小的序列，其中每个元素为（中心词，背景词，噪声词），返回我们需要的小批量数据格式。

```{.python .input  n=277}
def batchify(data):
    l = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_l = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (l - cur_l)]
        masks += [[1] * cur_l + [0] * (l - cur_l)]
        labels += [[1] * len(context) + [0] * (l - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))
```

下面创建小批量读取迭代器，并打印读取的第一个批量中的数据形状。

```{.python .input  n=278}
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 
                           'masks', 'labels'], batch):
        print(name, ':', data.shape)
    break
```

## 跳字模型

给定中心词$w_c$，和其对应的$n$个背景词$w_{o_1}, \ldots, w_{o_n}$，以及采样得到的$m$个噪音词$w_{o_{n+1}}, \ldots, w_{o_{n+m}}$。在跳字模型我们首先要得到它们对应的词嵌入$\boldsymbol{v}_c$, $\boldsymbol{u}_{o_1}, \ldots, \boldsymbol{u}_{o_n}$, $\boldsymbol{u}_{o_{n+1}}, \ldots, \boldsymbol{u}_{o_{n+m}}$。

### 嵌入层

获取词嵌入的层被称为嵌入层。嵌入层的权重是一个（词典大小，嵌入向量长度$d$）的矩阵。输入一个词的索引$i$，它返回权重的第$i$行作为它的词嵌入向量。在Gluon中可以通过`nn.Embedding`类来得到嵌入层。

```{.python .input  n=188}
embed = nn.Embedding(input_dim=20, output_dim=2)
embed.initialize()
embed.weight
```

嵌入层输入为词索引。假设输入形状为（批量大小，$l$），那么输出的形状为（批量大小，$l$，$d$），最后一个维度用来放置嵌入向量。

```{.python .input  n=184}
x = nd.array([[1,2,3]])
embed(x)
```

### 小批量乘法

我们可以将背景词向量和噪音词向量合并起来，然后使用一次矩阵乘法来计算中心词向量来计算它们的内积$\boldsymbol{v}_{c}^\top\left[\boldsymbol{u}_{o_1},\ldots,\boldsymbol{u}_{o_{n+m}}\right]$。我们需要对小批量里的每个中心词逐一做此运算，虽然可以用for循环来实现，但使用`batch_dot`通常可以得到更好的性能。假设$\boldsymbol{X}=\left[\boldsymbol{X}_1,\ldots,\boldsymbol{X}_n\right]$且$\boldsymbol{Y}=\left[\boldsymbol{Y}_1,\ldots,\boldsymbol{Y}_n\right]$，如果$\boldsymbol{Z}=\text{batch_dot}(\boldsymbol{X},\boldsymbol{Y})$，那么$\boldsymbol{Z}=\left[\boldsymbol{X}_1\boldsymbol{Y}_1,\ldots,\boldsymbol{X}_n\boldsymbol{Y}_n\right]$。

```{.python .input  n=259}
X = nd.ones((2, 3, 4))
Y = nd.ones((2, 4, 6))
nd.batch_dot(X, Y).shape
```

### 跳字模型前向计算

在前向计算中，跳字模型的输入是形状为（批量大小，1）的中心词索引，形状为（批量大小，$n+m$）的背景词和噪音词索引，以及对应的嵌入层。输出形状为（批量大小，1，$n+m$），每个元素是一对嵌入向量的内积。

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

## 模型训练

### 二元交叉熵损失函数

我们知道负采样的损失函数为：

$$\sum_{k=1}^n\log\,\sigma(\boldsymbol{u}_{o_k}^\top\boldsymbol{v}_c) + \sum_{k=n+1}^{n+m}\log\,\sigma(-\boldsymbol{u}_{o_k}^\top\boldsymbol{v}_c),$$

这里$\sigma(x)=1/(1+\exp(-x))$是sigmoid函数。这个等价于一个使用叉熵损失函数的两类softmax回归，又称logistic回归。如果构造标签$y_1,\ldots, y_{n+m}$，使得 $y_1=\ldots=y_n=1$，$y_{n+1}=\ldots=y_{n+m}=0$，那么我们可以重写上面损失函数为标准的logistic回归目标函数：

$$\sum_{k=1}^{n+m}y_k\log\,\sigma(\boldsymbol{u}_{o_k}^\top\boldsymbol{v}_c) + (1-y_k)\log\,\sigma(-\boldsymbol{u}_{o_k}^\top\boldsymbol{v}_c)$$

从而我们可以直接使用Gluon提供的`SigmoidBinaryCrossEntropyLoss`。

```{.python .input}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

### 初始化模型参数

我们构造中心词和背景词对应的嵌入层，并将超参数嵌入向量长度设置成100。

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))
```

### 训练

下面定义训练函数。注意由于填充的关系，在计算损失处跟之前的训练函数略微不同。

```{.python .input  n=228}
def train(net, lr, num_epochs):
    ctx = gb.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(),
                            'adam', {'learning_rate': lr})
    for epoch in range(num_epochs):
        start_time, train_l = time.time(), 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 将 mask 通过 loss 的权重选项来过滤掉填充。
                l = loss(pred.reshape(label.shape), label, mask)
                # l[i] 是第 i 个样本背景词和噪音词损失的均值。但由于填充关系，这个均值
                # 不正确。这里我们修改 l 为对小批量中所有背景词和噪音词的平均损失。
                l = l.sum() * mask.shape[1] / mask.sum()
            l.backward()
            trainer.step(1)  # l 已经做过了平均，这里不需要传入 batch_size。
            train_l += l.asscalar()
        print('epoch %d, train loss %.2f, time %.2fs' % (
            epoch + 1, train_l / len(data_iter), time.time() - start_time))
```

现在我们可以训练使用负采样的跳字模型了。可以看到，使用训练得到的词嵌入模型时，与词“chip”语义最接近的词大多与芯片有关。

```{.python .input  n=229}
train(net, 0.005, 8)
```

## 模型预测

当我们训练好词嵌入模型后，我们可以根据两个词向量的余弦相似度表示词与词之间在语义上的相似度。

```{.python .input  n=235}
def get_similar_tokens(query_token, k, embed):
    w = embed.weight.data()
    x = w[token_to_idx[query_token]]
    cos = nd.dot(w, x) / nd.sum(w*w, axis=1).sqrt() / nd.sum(x*x).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词。
        print('similarity=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))
get_similar_tokens(b'chip', 3, net[0])
```

## 小结

* 我们使用Gluon通过负采样训练了跳字模型。
* 二次采样试图尽可能减轻高频词对训练词嵌入模型的影响。
* 我们可以将长度不同的样本填充至长度相同的小批量，并通过掩码变量区分非填充和填充，然后只令非填充参与损失函数的计算。


## 练习

* 调一调模型和训练超参数。
* 试着找出其他词的近义词。
* 当数据集较大时，我们通常在迭代模型参数时才对当前小批量里的中心词采样背景词和噪声词。也就是说，同一个中心词在不同的迭代周期可能会有不同的背景词或噪声词。这样训练有哪些好处？尝试实现上述训练方法。
* 事实上`SigmoidBinaryCrossEntropyLoss`的实现不是本节给出的目标函数。找出它的实现，并分析为什么要这么做。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7761)

![](../img/qr_embedding-training.svg)


## 参考文献

[1] Penn Tree Bank. https://catalog.ldc.upenn.edu/ldc99t42

[2] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
