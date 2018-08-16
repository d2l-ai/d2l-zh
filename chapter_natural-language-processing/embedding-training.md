# 训练词嵌入模型

前两节描述了不同的词嵌入模型和近似训练法。本节中，我们将以[“词嵌入：word2vec”](word2vec.md)一节中的跳字模型和负采样为例，介绍在语料库上训练词嵌入模型的实现。我们还会介绍一些实现中的技巧，例如二次采样（subsampling）。

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

## 读取和处理数据集

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

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "# sentences: 42068\n"
 }
]
```

对于数据集的前三个句子，打印每个句子的词数和前五个词。

```{.python .input  n=4}
for sentence in dataset[:3]:
    print('# tokens:', len(sentence), sentence[:5])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "# tokens: 24 ['aer', 'banknote', 'berlitz', 'calloway', 'centrust']\n# tokens: 15 ['pierre', '<unk>', 'N', 'years', 'old']\n# tokens: 11 ['mr.', '<unk>', 'is', 'chairman', 'of']\n"
 }
]
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

其中 $f(w_i)$ 是数据集中词$w_i$的个数与总词数之比，常数$t$是一个超参数：我们在此将它设为$10^{-4}$。

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

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]]\ncenter 0 has contexts [1, 2]\ncenter 1 has contexts [0, 2, 3]\ncenter 2 has contexts [0, 1, 3, 4]\ncenter 3 has contexts [1, 2, 4, 5]\ncenter 4 has contexts [2, 3, 5, 6]\ncenter 5 has contexts [4, 6]\ncenter 6 has contexts [5, 7]\ncenter 7 has contexts [5, 6, 8, 9]\ncenter 8 has contexts [6, 7, 9]\ncenter 9 has contexts [8]\ncenter 10 has contexts [11]\ncenter 11 has contexts [10, 12]\ncenter 12 has contexts [10, 11]\n"
 }
]
```

在本节的实验中，我们设最大时间窗口为5。下面提取数据集中所有的中心词及其背景词。

```{.python .input  n=9}
max_window_size = 5
all_centers, all_contexts = get_center_context_arrays(subsampled_dataset,
                                                      max_window_size)
```

## 负采样

Remember that the loss function for negative sampling is defined as

$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \mathbf{v}_c)}. $$

Consequently for training the model we need to sample negatives from the unigram
token frequency distribution. The distribution is typically distorted by raising it elementwise to the  
power 0.75.

Note that while sampling from the unigram distribution is simple, we may
accidentally sample a word as a negative that is actually in the current context
of the center word. To improve training stability, we mask such accidental hits.

Here we directly sample negatives for every context precomputed before.

```{.python .input  n=10}
def get_negatives(negatives_shape, all_contexts, negatives_weights):
    population = list(range(len(negatives_weights)))
    k = negatives_shape[0] * negatives_shape[1]
    # 根据每个词的权重（negatives_weights）随机生成 k 个词的索引。
    negatives = random.choices(population, weights=negatives_weights, k=k)
    negatives = [negatives[i : i + negatives_shape[1]]
                 for i in range(0, k, negatives_shape[1])]
    # 如果负采样的词是当前中心词的背景词之一，丢弃该负采样的词。
    negatives = [
        [negative for negative in negatives_batch
        if negative not in set(all_contexts[i])]
        for i, negatives_batch in enumerate(negatives)
    ]
    return negatives
```

```{.python .input  n=11}
num_negatives = 5
negatives_weights = [counter[w]**0.75 for w in idx_to_token]
negatives_shape = (len(all_contexts), max_window_size * 2 * num_negatives)
all_negatives = get_negatives(negatives_shape, all_contexts,
                              negatives_weights)
```

## 训练和评价跳字模型

First we define a helper function `get_knn` to obtain the k closest words to for
a given word according to our trained word embedding model to evaluate if it
learned successfully.

考虑将norm_vecs_by_row放进gb，供“求近似词和类比词”调用。

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

We then define the model and initialize it randomly. Here we denote the model containing the weights $\mathbf{v}$ as `embedding` and respectively the model for $\mathbf{u}$ as `embedding_out`.

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

example_token = 'chip'
get_k_closest_tokens(token_to_idx, idx_to_token, embedding, 10, example_token)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "running on: gpu(0)\nclosest tokens to \"chip\": operational, triggering, resolved, market-share, bar, paul, iron, exploded, lies, wastewater\n"
 }
]
```

The gluon `SigmoidBinaryCrossEntropyLoss` corresponds to the loss function introduced above.

```{.python .input  n=14}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

Finally we train the word2vec model. We first shuffle our dataset

```{.python .input  n=15}
def batchify_fn(data):
    # data 是一个长度为 batch_size 的 list，其中每个元素为 (中心词, 背景词, 负采样词)。
    centers, contexts, negatives = zip(*data)
    batch_size = len(centers)

    # 每个中心词的背景词 contexts 和负采样词 negatives 的长度不一，我们将当前批量每个中
    # 心词的背景词和负采样词连结在一起，并填充 0 至相同长度 max_len。
    max_len = max(len(c) + len(n) for c, n in zip(contexts, negatives))    
    contexts_negatives = []
    masks = []
    labels = []
    for context, negative in zip(contexts, negatives):
        cur_len = len(context) + len(negative)
        context_negative = context + negative + [0] * (max_len - cur_len)
        # mask 用来区分非填充（背景词和负采样词）和填充。
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

```{.python .input  n=17}
train_embedding(num_epochs=5)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 1, time 4.73s, train loss 0.30\nclosest tokens to \"chip\": intel, computers, computer, microprocessor, audio, compaq, bugs, tandem, apple, mips\nepoch 2, time 4.65s, train loss 0.25\nclosest tokens to \"chip\": intel, microprocessor, microprocessors, computer, risc, manufactures, computers, memory, newest, flaws\nepoch 3, time 4.47s, train loss 0.21\nclosest tokens to \"chip\": intel, microprocessor, microprocessors, computer, chips, user, computers, robots, micro, risc\nepoch 4, time 4.47s, train loss 0.20\nclosest tokens to \"chip\": intel, microprocessors, microprocessor, computer, user, instructions, computers, chips, printers, computing\nepoch 5, time 4.96s, train loss 0.19\nclosest tokens to \"chip\": intel, microprocessors, computers, microprocessor, computer, user, target, robots, mips, printers\n"
 }
]
```

## 小结

* 


## 练习

* 


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7761)

![](../img/qr_embedding-training.svg)


## 参考文献

[1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
