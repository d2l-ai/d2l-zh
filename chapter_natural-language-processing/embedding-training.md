# 训练词嵌入模型

#TODO(@astonzhang) Need edit.

我们在[“词向量：word2vec”](./word2vec.md)introduced the word2vec word embedding model. In this notebook we will show how to train a word2vec model with Gluon. We will introduce training the model with the skip-gram objective and negative sampling. Besides mxnet Gluon we will only use standard Python language features but note that specific  toolkits for Natural Language Processing, such as the Gluon-NLP toolkit exist.

首先导入实验所需的包或模块，并抽取数据集。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')


from mxnet import autograd, nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import collections
import functools
import gluonbook as gb
import itertools
import math
import mxnet as mx
import random
import sys
import time
```

## Data

We then load a corpus for training the word embedding model. Like for training the language model in [“循环神经网络——使用Gluon”](../chapter_recurrent-neural-networks/rnn-gluon.md), we use the Penn Tree Bank（PTB）[1]。它包括训练集、验证集和测试集 。We directly split the datasets into sentences and tokens, considering newlines as paragraph delimeters and whitespace as token delimiter. We print the first five words of the first three sentences of the dataset.

```{.python .input  n=3}
import zipfile

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    dataset = f.readlines()
    dataset = [sentence.split() for sentence in dataset]

for sentence in dataset[:3]:
    print(sentence[:5] + ['...'])
```

### 建立词语索引

下面定义了`Dictionary`类来映射词语和整数索引。We first count all tokens in the dataset and assign integer indices to all tokens that occur more than five times in the corpus. We also construct the reverse mapping token to integer index `token_to_idx` and finally replace all tokens in the dataset with their respective indices.

```{.python .input  n=3}
min_count = 5

# 将 dataset 中所有词拼接起来，统计 dataset 中各个词出现的频率。
counter = collections.Counter(itertools.chain.from_iterable(dataset))
# 只为词频不小于 min_count 的词建立索引。
idx_to_token = list(token_count[0] for token_count in 
                    filter(lambda token_count: token_count[1] >= min_count,
                           counter.items()))
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

# coded_dataset 只包含被索引词的索引。
coded_dataset = [[token_to_idx[token] for token in sentence
                  if token in token_to_idx] for sentence in dataset]
```

### Dataset subsampling

One important trick applied when training word2vec is to subsample the dataset
according to the token frequencies. [2] proposes to discards individual
occurences of words from the dataset with probability $$ P(w_i) = 1 -
\sqrt{\frac{t}{f(w_i)}}$$ where $f(w_i)$ is the frequency ratio with which a word is
observed in a dataset and $t$ is a subsampling constant typically chosen around
$10^{-5}$. We are using a very small dataset here and found results in this case to be better with
$10^{-4}$.

```{.python .input  n=4}
idx_to_count = [counter[w] for w in idx_to_token]
total_count =  sum(idx_to_count)
idx_to_pdiscard = [1 - math.sqrt(1e-4 / (count / total_count))
                   for count in idx_to_count]

pruned_dataset = [[t for t in s if random.uniform(0, 1) > idx_to_pdiscard[t]]
                  for s in coded_dataset]
```

### Transformation of data

The skip-gram objective with negative sampling is based on sampled center,
context and negative data. 在跳字模型中，我们用一个词来预测它在文本序列周围的词。
举个例子，假设文本序列是“the”、“man”、“hit”、“his”和“son”。跳字模型所关心的是，
给定“hit”生成邻近词“the”、“man”、“his”和“son”的条件概率。在这个例子中，“hit”叫中
心词，“the”、“man”、“his”和“son”叫背景词。由于“hit”只生成与它距离不超过2的背景词，
该时间窗口的大小为2。

In general it is common to chose a maximum context size of for example 5 and to
uniformly sample a smaller context size from the interval [1, 5] independently
for each center word. So if we sample a random reduced context size of 1 在这个
例子中，只“man”和“his”叫背景词。

To train our Word2Vec model with batches of data we need to make sure that all
elements of a batch have the same shape, ie. the same context length. However
due to sampling a random reduced context size and as it is not guaranteed that a
sufficient number of words precedes or follows a given center word (as it may be
at the beginning or end of a sentence) the number of context words for a given
center word is not constant. Consequently we pad the context arrays and
introduce a mask that tells the model which of the words in the context array
are real context words and which are just padding.

For big datasets it is important to sample center and context words in a
streaming manner. Here for simplicity and as we use a small dataset we transform
the whole dataset at once into center words with respective contexts so that
during training we only need to iterate over the pre-computed arrays.


我们先随机得到不大于 `max_window_size` 的正整数作为窗口大小，设为$h$。中心词的每个背景词与该中心词在同一个句子中的词间距不超过$h$。

```{.python .input  n=5}
def get_center_context_arrays(coded_sentences, max_window_size):
    centers = []
    contexts = []
    for sentence in coded_sentences:
        # 每句话至少要有 2 个词才可能组成一对“中心词-背景词”。
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

max_window_size = 5
all_centers, all_contexts = get_center_context_arrays(pruned_dataset,
                                                      max_window_size)
```

```{.python .input}
my_pruned_dataset = [list(range(10)), list(range(10, 13))]

my_max_window_size = 2
my_all_centers, my_all_contexts = get_center_context_arrays(
    my_pruned_dataset, my_max_window_size)

print(my_pruned_dataset)
for i in range(13):
    print('center', my_all_centers[i], 'has contexts', my_all_contexts[i])
```

### 负采样

Remember that the loss function for negative sampling is defined as

$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \mathbf{v}_c)}. $$

Consequently for training the model we need to sample negatives from the unigram
token frequency distribution. The distribution is typically distorted by raising it elementwise to the  
power 0.75.

Note that while sampling from the unigram distribution is simple, we may
accidentally sample a word as a negative that is actually in the current context
of the center word. To improve training stability, we mask such accidental hits.

Here we directly sample negatives for every context precomputed before.

```{.python .input  n=6}
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

```{.python .input  n=7}
num_negatives = 5
negatives_weights = [counter[w]**0.75 for w in idx_to_token]
negatives_shape = (len(all_contexts), max_window_size * 2 * num_negatives)
all_negatives = get_negatives(negatives_shape, all_contexts,
                              negatives_weights)
```

## Model

First we define a helper function `get_knn` to obtain the k closest words to for
a given word according to our trained word embedding model to evaluate if it
learned successfully.

考虑将norm_vecs_by_row放进gb，供“求近似词和类比词”调用。

```{.python .input  n=8}
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

```{.python .input  n=9}
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

The gluon `SigmoidBinaryCrossEntropyLoss` corresponds to the loss function introduced above.

```{.python .input  n=10}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

Finally we train the word2vec model. We first shuffle our dataset

```{.python .input  n=11}
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

```{.python .input  n=12}
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

```{.python .input  n=13}
train_embedding(num_epochs=5)
```

[1] word2vec工具. https://code.google.com/archive/p/word2vec/

[2] Mikolov, Tomas, et al. “Distributed representations of words and phrases and their compositionality.” Advances in neural information processing systems. 2013.
