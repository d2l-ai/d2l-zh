# 训练词嵌入模型

前两节描述了不同的词嵌入模型和近似训练法。本节中，我们将以[“词嵌入：word2vec”](word2vec.md)一节中的跳字模型和负采样为例，介绍在语料库上训练词嵌入模型的实现。我们还会介绍一些实现中的技巧，例如二次采样（subsampling）和掩码（mask）变量。

首先导入实验所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import collections
import functools
import gluonbook as gb
import itertools
import math
import mxnet as mx
from mxnet import autograd, nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile
```

## 处理数据集

我们将在[“循环神经网络——使用Gluon”](../chapter_recurrent-neural-networks/rnn-gluon.md)一节中介绍的Penn Tree Bank数据集（训练集）上训练词嵌入模型。该数据集的每一行为一个句子。句子中的每个词由空格隔开。

```{.python .input  n=2}
with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    dataset = f.readlines()
    dataset = [sentence.split() for sentence in dataset]
```

查看该数据集中句子个数。

```{.python .input  n=3}
print('# sentences:', len(dataset))
```

对于数据集的前三个句子，打印每个句子的词数和前五个词。

```{.python .input  n=4}
for sentence in dataset[:3]:
    print('# tokens:', len(sentence), sentence[:5])
```

### 建立词语索引

我们为数据集中至少出现5次的词建立整数索引。处理后的数据集只包含被索引词的索引。

```{.python .input  n=5}
min_count = 5

# 将 dataset 中所有词拼接起来，统计 dataset 中各个词出现的频率。
counter = collections.Counter(itertools.chain.from_iterable(dataset))
# 只为词频不低于 min_count 的词建立索引。
idx_to_token = list(token_count[0] for token_count in 
                    filter(lambda token_count: token_count[1] >= min_count,
                           counter.items()))
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

# coded_dataset 只包含被索引词的索引。
coded_dataset = [[token_to_idx[token] for token in sentence
                  if token in token_to_idx] for sentence in dataset]
```

### 二次采样

在一般的文本数据集中，有些词的词频可能过高，例如英文中的“the”、“a”和“in”。通常来说，一个句子中，词“China”和较低频词“Beijing”同时出现比和较高频词“the”同时出现对训练词嵌入更加有帮助。这是因为，绝大多数词都和词“the”同时出现在一个句子里。因此，训练词嵌入模型时可以对词进行二次采样 [1]。具体来说，数据集中每个被索引词$w_i$将有一定概率被丢弃：该概率为

$$ \mathbb{P}(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$ 

其中 $f(w_i)$ 是数据集中词$w_i$的个数与总词数之比，常数$t$是一个超参数：我们在此将它设为$10^{-4}$。可见，越高频的词在二次采样中被丢弃的概率越大。

```{.python .input  n=6}
idx_to_count = [counter[w] for w in idx_to_token]
total_count =  sum(idx_to_count)
idx_to_pdiscard = [1 - math.sqrt(1e-4 / (count / total_count))
                   for count in idx_to_count]

subsampled_dataset = [[
    t for t in s if random.uniform(0, 1) > idx_to_pdiscard[t]]
    for s in coded_dataset]
```

### 提取中心词和背景词

让我们先回顾一下跳字模型。在跳字模型中，我们用一个词（中心词）来预测它在文本序列周围的词，即与该中心词在相同时间窗口内的背景词。设最大时间窗口大小为`max_window_size`。我们先在整数1和`max_window_size`之间均匀随机采样一个正整数$h$作为时间窗口大小。在同一个句子中，与中心词的词间距不超过$h$的词均为该中心词的背景词。举个例子，考虑句子“the man loves his son a lot”。假设中心词为“loves”且$h$为2，那么该中心词的背景词为“the”、“man”、“his”和“son”。假设中心词为“lot”且$h$为3，那么该中心词的背景词为“his”、“son”和“a”。

下面的`get_center_context_arrays`函数根据最大时间窗口`max_window_size`从数据集`coded_sentences`中提取出全部中心词及其背景词。这是通过调用辅助函数`get_one_context`实现的。

```{.python .input  n=7}
def get_center_context_arrays(coded_sentences, max_window_size):
    centers = []
    contexts = []
    for sentence in coded_sentences:
        # 每个句子至少要有 2 个词才可能组成一对“中心词-背景词”。
        if len(sentence) < 2:
            continue
        centers += sentence
        context = [get_one_context(sentence, i, max_window_size)
                   for i in range(len(sentence))]
        contexts += context
    return centers, contexts

def get_one_context(sentence, word_idx, max_window_size):
    # 从 1 和 max_window_size 之间均匀随机生成整数（包括 1 和 max_window_size）。
    window_size = random.randint(1, max_window_size)
    start_idx = max(0, word_idx - window_size)
    # 加 1 是为了将中心词排除在背景词之外。
    end_idx = min(len(sentence), word_idx + 1 + window_size)
    context = []
    # 添加中心词左边的背景词。
    if start_idx != word_idx:
        context += sentence[start_idx:word_idx]
    # 添加中心词右边的背景词。
    if word_idx + 1 != end_idx: 
        context += sentence[word_idx + 1 : end_idx]
    return context
```

下面我们生成一个人工数据集，其中含有词数分别为10和3的两个句子。设最大时间窗口为2。打印所有中心词和它们的背景词。

```{.python .input  n=8}
my_subsampled_dataset = [list(range(10)), list(range(10, 13))]

my_max_window_size = 2
my_all_centers, my_all_contexts = get_center_context_arrays(
    my_subsampled_dataset, my_max_window_size)

print(my_subsampled_dataset)
for i in range(13):
    print('center', my_all_centers[i], 'has contexts', my_all_contexts[i])
```

在本节的实验中，我们设最大时间窗口为5。下面提取数据集中所有的中心词及其背景词。

```{.python .input  n=9}
max_window_size = 5
all_centers, all_contexts = get_center_context_arrays(subsampled_dataset,
                                                      max_window_size)
```

## 负采样

在负采样中，对于同一时间窗口中的一对中心词$w_c$和背景词$w_o$，我们假设存在$K+1$个相互独立事件：中心词$w_c$和背景词$w_o$同时出现在时间窗口、中心词$w_c$和第1个噪声词$w_1$不同时出现在该时间窗口、……、中心词$w_c$和第$K$个噪声词$w_K$不同时出现在该时间窗口。这里设$K=5$。根据[“词嵌入：word2vec”](word2vec.md)一节中练习的建议，噪声词采样概率$\mathbb{P}(w)$可设为$w$词频与总词频的比的3/4次方。由于我们假设噪声词和中心词不同时出现在时间窗口，如果采样的噪声词恰好是当前中心词的某个背景词，该噪声词将被丢弃。

```{.python .input  n=10}
def get_negatives(negatives_shape, all_contexts, negatives_weights):
    population = list(range(len(negatives_weights)))
    k = negatives_shape[0] * negatives_shape[1]
    # 根据每个词的权重（negatives_weights）随机生成 k 个词的索引。
    negatives = random.choices(population, weights=negatives_weights, k=k)
    negatives = [negatives[i : i + negatives_shape[1]]
                 for i in range(0, k, negatives_shape[1])]
    # 如果噪声词是当前中心词的某个背景词，丢弃该噪声词。
    negatives = [
        [negative for negative in negatives_batch
        if negative not in set(all_contexts[i])]
        for i, negatives_batch in enumerate(negatives)
    ]
    return negatives

num_negatives = 5
negatives_weights = [counter[w]**0.75 for w in idx_to_token]
negatives_shape = (len(all_contexts), max_window_size * 2 * num_negatives)
all_negatives = get_negatives(negatives_shape, all_contexts,
                              negatives_weights)
```

## 训练跳字模型

当我们训练好词嵌入模型后，我们可以根据两个词向量的余弦相似度表示词与词之间在语义上的相似度。由此，给定一个词，我们可以通过`get_k_closest_tokens`函数从所有被索引词中找出与该词最接近的`k`个词。

```{.python .input  n=12}
def norm_vecs_by_row(x):
    # 分母中添加的 1e-10 是为了数值稳定性。
    return x / (nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))

def get_k_closest_tokens(token_to_idx, idx_to_token, embedding, k, word):
    word_vec = embedding(nd.array(
        [token_to_idx[word]], ctx=ctx)).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(embedding.weight.data())
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(idx_to_token), )), k=k+1,
                      ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # 除去输入词。
    result = [idx_to_token[i] for i in indices[1:]]
    print('closest tokens to "%s": %s' % (word, ", ".join(result)))
```

### 初始化词嵌入模型

我们设每个词向量的长度为300，并对它们随机初始化。变量`embedding`和`embedding_out`分别代表中心词和背景词的向量。

```{.python .input  n=13}
ctx = gb.try_gpu()
print('running on:', ctx)

embedding_size = 300
embedding = nn.Embedding(input_dim=len(idx_to_token),
                         output_dim=embedding_size)
embedding_out = nn.Embedding(input_dim=len(idx_to_token),
                             output_dim=embedding_size)
embedding.initialize(ctx=ctx)
embedding_out.initialize(ctx=ctx)
embedding.hybridize()
embedding_out.hybridize()

params = list(embedding.collect_params().values()) + list(
    embedding_out.collect_params().values())
trainer = gluon.Trainer(params, 'adam', {'learning_rate': 0.01})
```

使用随机初始化的词向量，与词“chip”最接近的10个词看上去只是随机产生的几个词。

```{.python .input}
example_token = 'chip'
get_k_closest_tokens(token_to_idx, idx_to_token, embedding, 10, example_token)
```

### 掩码变量

根据[“词嵌入：word2vec”](word2vec.md)一节中负采样损失函数的定义，我们使用Gluon的二元交叉熵损失函数`SigmoidBinaryCrossEntropyLoss`。

```{.python .input}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

值得一提的是，我们可以通过掩码变量指定小批量中参与损失函数计算的部分预测值和标签：当掩码为1时，相应位置的预测值和标签将参与损失函数的计算；当掩码为0时，相应位置的预测值和标签则不参与损失函数的计算。

```{.python .input  n=14}
my_pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签中的 1 和 0 分别代表背景词和噪声词。
my_label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
# 掩码变量。
my_mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])
loss(my_pred, my_label, my_mask) * my_mask.shape[1] / my_mask.sum(axis=1)
```

作为比较，我们从零开始实现二元交叉熵损失函数的计算，并根据掩码变量`my_mask`计算掩码为1的预测值和标签的损失函数。

```{.python .input}
sigmoid = lambda x : -math.log(1 / (1 + math.exp(-x)))
printfloat = lambda x : print('%.7f' % (x))
printfloat((sigmoid(1.5) + sigmoid(-0.3) + sigmoid(1) + sigmoid(-2)) / 4)
printfloat((sigmoid(1.1) + sigmoid(-0.6) + sigmoid(-2.2)) / 3)
```

### 读取小批量

在一个小批量数据中，每个样本包括一个中心词、若干个背景词和噪声词。既然每个样本中背景词与噪声词数量之和不同，我们不妨将这些背景词和噪声词连结在一起，并填充0至相同长度。我们通过掩码变量区分非填充（背景词和噪声词）和填充。填充不会参与损失函数的计算。

```{.python .input  n=15}
def batchify_fn(data):
    # data 是一个长度为 batch_size 的 list，其中每个元素为 (中心词, 背景词, 噪声词)。
    centers, contexts, negatives = zip(*data)
    batch_size = len(centers)

    # 每个中心词的背景词 contexts 和噪声词 negatives 的长度不一，我们将当前批量每个中
    # 心词的背景词和噪声词连结在一起，并填充 0 至相同长度 max_len。
    max_len = max(len(c) + len(n) for c, n in zip(contexts, negatives))    
    contexts_negatives = []
    masks = []
    labels = []
    for context, negative in zip(contexts, negatives):
        cur_len = len(context) + len(negative)
        context_negative = context + negative + [0] * (max_len - cur_len)
        # mask 用来区分非填充（背景词和噪声词）和填充。
        mask = [1] * cur_len + [0] * (max_len - cur_len)
        # label 用来区分背景词和非背景词。
        label = [1] * len(context) + [0] * (max_len - len(context))
        contexts_negatives.append(context_negative)
        masks.append(mask)
        labels.append(label)

    centers_nd = nd.array(centers).reshape((batch_size, 1))
    contexts_negatives_nd = nd.array(contexts_negatives)
    masks_nd = nd.array(masks)
    labels_nd = nd.array(labels)
    return centers_nd, contexts_negatives_nd, masks_nd, labels_nd

batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             batchify_fn=batchify_fn, num_workers=num_workers)
```

### 训练模型

下面我们定义跳字模型的训练函数。我们使用掩码变量避免填充对损失函数计算的影响。

```{.python .input  n=16}
def train_embedding(num_epochs):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_l_sum = 0
        for center, context_and_negative, mask, label in data_iter:
            center = center.as_in_context(ctx)
            context_and_negative = context_and_negative.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                # emb_in 形状：（batch_size, 1, embedding_size）。
                emb_in = embedding(center)
                # embedding_out(context_and_negative) 形状：
                #（batch_size, max_len, embedding_size）。
                emb_out = embedding_out(context_and_negative)
                # pred 形状：（batch_size, 1, max_len）。
                pred = nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                # mask 和 label 形状：（batch_size, max_len）。
                # 避免填充对损失函数计算的影响。
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
        print('epoch %d, time %.2fs, train loss %.2f' 
              % (epoch, time.time() - start_time,
                 train_l_sum / len(data_iter)))
        get_k_closest_tokens(token_to_idx, idx_to_token, embedding, 10,
                             example_token)
```

现在我们可以训练使用负采样的跳字模型了。可以看到，使用训练得到的词嵌入模型时，与词“chip”近似的词大多与芯片有关。

```{.python .input  n=17}
train_embedding(num_epochs=5)
```

## 小结

* 我们使用Gluon通过负采样训练了跳字模型。
* 二次采样试图尽可能减轻高频词对训练词嵌入模型的影响。
* 我们可以将长度不同的样本填充至长度相同的小批量，并通过掩码变量区分非填充和填充，然后只令非填充参与损失函数的计算。


## 练习

* 调一调超参数，试着找出其他词的近似词，观察并分析实验结果。

* 当数据集较大时，我们通常在迭代模型参数时才对当前小批量里的中心词采样背景词和噪声词。也就是说，同一个中心词在不同的迭代周期可能会有不同的背景词或噪声词。这样训练有哪些好处？

* 实现上述训练方法。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7761)

![](../img/qr_embedding-training.svg)


## 参考文献

[1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
