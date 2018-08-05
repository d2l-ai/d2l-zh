# 训练词嵌入模型

#TODO(@astonzhang) Need edit.

我们在[“词向量：word2vec”](./word2vec.md)introduced the word2vec word embedding model. In this notebook we will show how to train a word2vec model with Gluon. We will introduce training the model with the skip-gram objective and negative sampling. Besides mxnet Gluon we will only use standard Python language features but note that specific  toolkits for Natural Language Processing, such as the Gluon-NLP toolkit exist.

首先导入实验所需的包或模块，并抽取数据集。

```{.python .input  n=1}
from mxnet import nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import collections
import functools
import itertools
import math
import mxnet as mx
import operator
import random
import time
```

## Data

We then load a corpus for training the word embedding model. Like for training the language model in [“循环神经网络——使用Gluon”](../chapter_recurrent-neural-networks/rnn-gluon.md), we use the Penn Tree Bank（PTB）[1]。它包括训练集、验证集和测试集 。We directly split the datasets into sentences and tokens, considering newlines as paragraph delimeters and whitespace as token delimiter. We print the first five words of the first three sentences of the dataset.

```{.python .input  n=2}
import zipfile

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    dataset = f.readlines()
    dataset = [sentence.split() for sentence in dataset]

for sentence in dataset[:3]:
    print(sentence[:5] + ['...'])
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['aer', 'banknote', 'berlitz', 'calloway', 'centrust', '...']\n['pierre', '<unk>', 'N', 'years', 'old', '...']\n['mr.', '<unk>', 'is', 'chairman', 'of', '...']\n"
 }
]
```

### 建立词语索引

下面定义了`Dictionary`类来映射词语和整数索引。We first count all tokens in the dataset and assign integer indices to all tokens that occur more than five times in the corpus. We also construct the reverse mapping token to integer index `token_to_idx` and finally replace all tokens in the dataset with their respective indices.

```{.python .input  n=3}
min_token_occurence = 5

# 统计 dataset 中的词频。
counter = collections.Counter(itertools.chain.from_iterable(dataset))
idx_to_token = list(token_count[0] for token_count in 
                    filter(lambda token_count: token_count[1] >= min_token_occurence,
                           counter.items()))
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

coded_dataset = [[token_to_idx[token] for token in sentence if token in token_to_idx] for sentence in dataset]
```

### Dataset subsampling

One important trick applied when training word2vec is to subsample the dataset
according to the token frequencies. [2] proposes to discards individual
occurences of words from the dataset with probability $$ P(w_i) = 1 -
\sqrt{\frac{t}{f(w_i)}}$$ where $f(w_i)$ is the frequency with which a word is
observed in a dataset and $t$ is a subsampling constant typically chosen around
$10^{-5}$. We are using a very small dataset here and found results in this case to be better with
$10^{-4}$.

```{.python .input  n=4}
idx_to_counts = [counter[w] for w in idx_to_token]
frequent_tokens_subsampling_constant = 1e-4
sum_counts =  sum(idx_to_counts)
idx_to_pdiscard = [1 - math.sqrt(frequent_tokens_subsampling_constant / (count / sum_counts))
                   for count in idx_to_counts]

pruned_dataset = [[t for t in s if random.uniform(0, 1) > idx_to_pdiscard[t]] for s in coded_dataset]
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

```{.python .input  n=5}
def get_center_context_arrays(coded_sentences, window_size):
    centers = []
    contexts = []
    for sentence in coded_sentences:
        # We need at least  2  words  to form a source, context pair
        if len(sentence) < 2:
            continue
        centers += sentence
        context = [get_one_context(sentence, i, window_size)
                   for i in range(len(sentence))]
        contexts += context
    return centers, contexts


def get_one_context(sentence, word_index, window_size):
    # A random reduced window size is drawn.
    random_window_size = random.randint(1, window_size)

    start_idx = max(0, word_index - random_window_size)
    # First index outside of the window
    end_idx = min(len(sentence), word_index + random_window_size + 1)

    context = []
    # Get contexts left of center word
    if start_idx != word_index:
        context += sentence[start_idx:word_index]
    # Get contexts right of center word
    if word_index + 1 != end_idx: 
        context += sentence[word_index + 1:end_idx]
    return context

window_size = 5
all_centers, all_contexts = get_center_context_arrays(pruned_dataset, window_size)
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
def get_negatives(shape, true_samples, negatives_weights):
    population = list(range(len(negatives_weights)))
    k = functools.reduce(operator.mul, shape)
    assert len(shape) == 2
    assert len(true_samples) == shape[0]
    
    negatives = random.choices(population, weights=negatives_weights, k=k)
    negatives = [negatives[i:i+shape[1]] for i in range(0, k, shape[1])]
    negatives = [
        [negative for negative in negatives_batch
        if negative not in true_samples[i]]
        for i, negatives_batch in enumerate(negatives)
    ]
    return negatives
```

```{.python .input  n=7}
# This may take around 20 seconds
num_negatives = 5
negatives_weights = [counter[w]**0.75 for w in idx_to_token]
negatives_shape = (len(all_contexts), window_size * 2 * num_negatives)
all_negatives = get_negatives(negatives_shape, all_contexts, negatives_weights)
```

## Model

First we define a helper function `get_knn` to obtain the k closest words to for
a given word according to our trained word embedding model to evaluate if it
learned successfully.

```{.python .input  n=8}
def norm_vecs_by_row(x):
    # 分母中添加的 1e-10 是为了数值稳定性。
    return x / (nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))

def get_knn(token_to_idx, idx_to_token, embedding, k, word):
    word_vec = embedding(nd.array([token_to_idx[word]], ctx=context)).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(embedding.weight.data())
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(idx_to_token), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    result = [idx_to_token[i] for i in indices[1:]]
    print('Closest tokens to "%s": %s' % (word, ", ".join(result)))
    return result
```

We then define the model and initialize it randomly. Here we denote the model containing the weights $\mathbf{v}$ as `embedding` and respectively the model for $\mathbf{u}$ as `embedding_out`.

```{.python .input  n=9}
context = mx.gpu(0)
# context = mx.cpu()

embedding_size = 300
embedding = nn.Embedding(input_dim=len(idx_to_token), output_dim=embedding_size)
embedding_out = nn.Embedding(input_dim=len(idx_to_token), output_dim=embedding_size)

embedding.initialize(ctx=context)
embedding_out.initialize(ctx=context)
embedding.hybridize()
embedding_out.hybridize()

params = list(embedding.collect_params().values()) + list(embedding_out.collect_params().values())
trainer = gluon.Trainer(params, 'adagrad', dict(learning_rate=0.1))

example_token = 'president'
knn = get_knn(token_to_idx, idx_to_token, embedding, 5, example_token)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Closest tokens to \"president\": hoping, tailored, upheld, manila, borough\n"
 }
]
```

The gluon `SigmoidBinaryCrossEntropyLoss` corresponds to the loss function introduced above.

```{.python .input  n=10}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

Finally we train the word2vec model. We first shuffle our dataset

```{.python .input  n=11}
class Dataset(gdata.SimpleDataset):
    def __init__(self, centers, contexts, negatives):
        data = list(zip(centers, contexts, negatives))
        super().__init__(data)

def batchify_fn(data):
    # data is a list with batch_size elements
    # each element is of form (center, context, negative)
    centers, contexts, negatives = zip(*data)
    batch_size = len(centers)  # == len(contexts) == len(negatives)
    
    # contexts and negatives are of variable length
    # we pad them to a fixed length and introduce a mask
    length = max(len(c) + len(n) for c, n in zip(contexts, negatives))
    contexts_negatives = []
    masks = []
    labels = []
    for i, (context, negative) in enumerate(zip(contexts, negatives)):
        len_context_negative = len(context) + len(negative)
        context_negative = context + negative + [0] * (length - len_context_negative)
        mask = [1] * len_context_negative + [0] * (length - len_context_negative)
        label = [1] * len(context) + [0] * (length - len(context))
        contexts_negatives.append(context_negative)
        masks.append(mask)
        labels.append(label)
        
    centers_nd = nd.array(centers).reshape((batch_size, 1))
    contexts_negatives_nd = nd.array(contexts_negatives)
    masks_nd = nd.array(masks)
    labels_nd = nd.array(labels)
    return centers_nd, contexts_negatives_nd, masks_nd, labels_nd

batch_size = 512

data = Dataset(all_centers, all_contexts, all_negatives)
batches = gdata.DataLoader(data, batch_size=batch_size,
                                shuffle=True, batchify_fn=batchify_fn,
                                num_workers=1)
```

```{.python .input  n=12}
def train_embedding(num_epochs=3, eval_period=100):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_l_sum = 0
        for batch_i, (center, context_and_negative, mask, label) in enumerate(batches):
            center = center.as_in_context(context)
            context_and_negative = context_and_negative.as_in_context(context)
            mask = mask.as_in_context(context)
            label = label.as_in_context(context)
            with mx.autograd.record():
                # 1. Compute the embedding of the center words
                emb_in = embedding(center)
                # 2. Compute the context embedding
                emb_out = embedding_out(context_and_negative) * mask.expand_dims(-1)
                # 3. Compute the prediction
                pred = nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                # 4. Compute the Loss function (SigmoidBinaryCrossEntropyLoss)
                l = loss(pred, label)
            # Compute the gradient
            l.backward()
            # Update the parameters
            trainer.step(batch_size=1)
            train_l_sum += l.mean()
            if batch_i % eval_period == 0 and batch_i != 0 :
                cur_l = train_l_sum / eval_period
                print('epoch %d, batch %d, time %.2fs, train loss %.2f'
                      % (epoch, batch_i, time.time() - start_time, cur_l.asscalar()))
                train_l_sum = 0
                get_knn(token_to_idx, idx_to_token, embedding, 5, example_token)
```

```{.python .input  n=13}
train_embedding()
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 1, batch 100, time 1.43s, train loss 0.40\nClosest tokens to \"president\": named, chairman, succeeding, formerly, stoltzman\nepoch 1, batch 200, time 2.42s, train loss 0.33\nClosest tokens to \"president\": succeeding, e., named, officer, sanford\nepoch 1, batch 300, time 3.42s, train loss 0.32\nClosest tokens to \"president\": treasurer, deputy, e., f., succeeding\nepoch 1, batch 400, time 4.41s, train loss 0.31\nClosest tokens to \"president\": treasurer, vice, resigned, formerly, jordan\nepoch 1, batch 500, time 5.40s, train loss 0.31\nClosest tokens to \"president\": treasurer, deputy, formerly, p., named\nepoch 1, batch 600, time 6.39s, train loss 0.31\nClosest tokens to \"president\": treasurer, katz, chairman, deputy, p.\nepoch 1, batch 700, time 7.38s, train loss 0.31\nClosest tokens to \"president\": treasurer, p., chief, chairman, katz\nepoch 2, batch 100, time 1.43s, train loss 0.28\nClosest tokens to \"president\": treasurer, succeeding, formerly, roger, vice\nepoch 2, batch 200, time 2.43s, train loss 0.27\nClosest tokens to \"president\": succeeding, treasurer, formerly, vice, p.\nepoch 2, batch 300, time 3.41s, train loss 0.27\nClosest tokens to \"president\": succeeding, treasurer, dover, vice, formerly\nepoch 2, batch 400, time 4.40s, train loss 0.27\nClosest tokens to \"president\": treasurer, vice, succeeding, resigned, katz\nepoch 2, batch 500, time 5.39s, train loss 0.27\nClosest tokens to \"president\": treasurer, p., vice, ehrlich, succeeding\nepoch 2, batch 600, time 6.38s, train loss 0.27\nClosest tokens to \"president\": vice, treasurer, ehrlich, p., formerly\nepoch 2, batch 700, time 7.37s, train loss 0.27\nClosest tokens to \"president\": vice, succeeding, treasurer, p., formerly\nepoch 3, batch 100, time 1.42s, train loss 0.25\nClosest tokens to \"president\": succeeding, vice, ehrlich, formerly, treasurer\nepoch 3, batch 200, time 2.41s, train loss 0.25\nClosest tokens to \"president\": succeeding, vice, p., treasurer, formerly\nepoch 3, batch 300, time 3.41s, train loss 0.25\nClosest tokens to \"president\": vice, succeeding, p., dover, ehrlich\nepoch 3, batch 400, time 4.40s, train loss 0.25\nClosest tokens to \"president\": vice, succeeding, formerly, ehrlich, gerald\nepoch 3, batch 500, time 5.40s, train loss 0.25\nClosest tokens to \"president\": vice, formerly, gerald, succeeding, p.\nepoch 3, batch 600, time 6.41s, train loss 0.25\nClosest tokens to \"president\": vice, p., dover, formerly, gerald\nepoch 3, batch 700, time 7.42s, train loss 0.25\nClosest tokens to \"president\": vice, p., succeeding, formerly, dover\n"
 }
]
```

[1] word2vec工具. https://code.google.com/archive/p/word2vec/

[2] Mikolov, Tomas, et al. “Distributed representations of words and phrases and their compositionality.” Advances in neural information processing systems. 2013.[2] Mikolov, Tomas, et al. “Distributed representations of words and phrases and their compositionality.” Advances in neural information processing systems. 2013.
