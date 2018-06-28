# Training 词向量：word2vec

我们在[“词向量：word2vec”](./word2vec.md)introduced the word2vec word embedding model. In this notebook we will show how to train a word2vec model with Gluon. We will introduce training the model with the skip-gram objective and negative sampling. Besides mxnet Gluon and numpy we will only use standard Python language features but note that specific  toolkits for Natural Language Processing, such as the Gluon-NLP toolkit exist.

首先导入实验所需的包或模块，并抽取数据集。

```{.python .input  n=1}
import collections
import itertools
import functools
import random

import numpy as np
import mxnet as mx
from mxnet import nd, gluon

from mxnet.gluon import nn
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

### 建立词语索引

下面定义了`Dictionary`类来映射词语和整数索引。We first count all tokens in the dataset and assign integer indices to all tokens that occur more than five times in the corpus. We also construct the reverse mapping token to integer index `token_to_idx` and finally replace all tokens in the dataset with their respective indices.

```{.python .input  n=3}
min_token_occurence = 5

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
$10^{-5}$.

```{.python .input}
idx_to_counts = np.array([counter[w] for w in idx_to_token])
frequent_tokens_subsampling_constant = 1e-5
f = idx_to_counts / np.sum(idx_to_counts)
idx_to_pdiscard = 1 - np.sqrt(frequent_tokens_subsampling_constant / f)

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
def get_center_context_arrays(coded_sentences, window):
    contexts = []
    masks = []
    for sentence in coded_sentences:
        if not sentence:  # Sentence with no words
            continue
            
        context, mask = zip(*[get_one_context_mask(sentence, i, window) for i in range(len(sentence))])
        contexts += context
        masks += mask
        
    centers = mx.nd.array(np.concatenate(coded_sentences))
    contexts = mx.nd.array(np.stack(contexts))
    masks = mx.nd.array(np.stack(masks))
    
    return centers, contexts, masks


def get_one_context_mask(sentence, word_index, window):
    # A random reduced window size is drawn.
    random_window_size = random.randint(1, window)

    start_idx = max(0, word_index - random_window_size)
    # First index outside of the window
    end_idx = min(len(sentence), word_index + random_window_size + 1)

    context = np.zeros(window * 2, dtype=np.float32)
    # Get contexts left of center word
    next_context_idx = 0
    context[:word_index - start_idx] = sentence[start_idx:word_index]
    next_context_idx += word_index - start_idx
    # Get contexts right of center word
    context[next_context_idx:next_context_idx + end_idx - (word_index + 1)] = \
        sentence[word_index + 1:end_idx]
    next_context_idx += end_idx - (word_index + 1)

    # The mask masks entries that fall inside the window
    # but outside random the reduced window size.
    # It is necessary as all context arrays must have the same shape for batching.
    mask = np.ones(window * 2, dtype=np.float32)
    mask[next_context_idx:] = 0

    return context, mask
    

all_centers, all_contexts, all_masks = get_center_context_arrays(pruned_dataset, 5)
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

```{.python .input  n=5}
def get_negatives(shape, true_samples, true_samples_mask, token_frequencies, distortion=0.75):
        token_frequencies = np.array(token_frequencies)
        smoothed_token_freq_cumsum = np.cumsum((token_frequencies**distortion).astype(np.int))

        negatives = np.searchsorted(smoothed_token_freq_cumsum,
            np.random.randint(smoothed_token_freq_cumsum[-1], size=shape))
        
        # Mask accidental hits
        true_samples, true_samples_mask = true_samples.asnumpy(), true_samples_mask.asnumpy()
        negatives_mask = np.array([
            [negatives[i, j] not in true_samples[i] for j in range(negatives.shape[1])]
            for i in range(negatives.shape[0])
        ])

        return mx.nd.array(negatives), mx.nd.array(negatives_mask)
        
        
num_negatives = 5
negatives_weights = [counter[w] for w in idx_to_token]
negatives_shape = (all_contexts.shape[0], all_contexts.shape[1] * num_negatives)
all_negatives, all_negatives_masks = get_negatives(negatives_shape, all_contexts, all_masks, negatives_weights)
```

## Model

First we define a helper function `get_knn` to obtain the k closest words to for
a given word according to our trained word embedding model to evaluate if it
learned successfully.

```{.python .input  n=7}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(token_to_idx, idx_to_token, embedding, k, word):
    word_vec = embedding(mx.nd.array([token_to_idx[word]], ctx=context)).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(embedding.weight.data())
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(idx_to_token), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    result = [idx_to_token[i] for i in indices[1:]]
    print(f'Closest tokens to "{example_token}": {", ".join(result)}')
    return result
```

We then define the model and initialize it randomly. Here we denote the model containing the weights $\mathbf{v}$ as `embedding` and respectively the model for $\mathbf{u}$ as `embedding_out`.

```{.python .input  n=8}
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
trainer = gluon.Trainer(params, 'adagrad', dict(learning_rate=0.05))

example_token = 'president'
knn = get_knn(token_to_idx, idx_to_token, embedding, 5, example_token)
```

The gluon `SigmoidBinaryCrossEntropyLoss` corresponds to the loss function introduced above.

```{.python .input  n=9}
loss_function = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

Finally we train the word2vec model. We first shuffle our dataset 

```{.python .input  n=11}
import time
start = time.time()

batch_size = 256
batch_id = 0

# Shuffle the dataset
random_indices = mx.nd.random.shuffle(mx.nd.arange(len(all_centers)))
all_centers = all_centers[random_indices]
all_contexts = all_contexts[random_indices]
all_masks = all_masks[random_indices]
all_negatives = all_negatives[random_indices]
all_negatives_masks = all_negatives_masks[random_indices]

batches = (
    gluon.data.DataLoader(all_centers, batch_size=batch_size),
    gluon.data.DataLoader(all_contexts, batch_size=batch_size),
    gluon.data.DataLoader(all_masks, batch_size=batch_size),
    gluon.data.DataLoader(all_negatives, batch_size=batch_size),
    gluon.data.DataLoader(all_negatives_masks, batch_size=batch_size),
)

for batch in zip(*batches):
    # Each batch from the context_sampler includes
    # a batch of center words, their contexts as well
    # as a mask as the contexts can be of varying lengths
    (center, word_context, word_context_mask, negatives, negatives_mask) = batch

    # We copy all data to the GPU
    center = center.as_in_context(context)
    word_context = word_context.as_in_context(context)
    word_context_mask = word_context_mask.as_in_context(context)
    negatives = negatives.as_in_context(context)
    negatives_mask = negatives_mask.as_in_context(context)

    # We concatenate the positive context words and negatives
    # to a single ndarray 
    word_context_negatives = mx.nd.concat(word_context, negatives, dim=1)
    word_context_negatives_mask = mx.nd.concat(word_context_mask, negatives_mask, dim=1)

    # We record the gradient of one forward pass
    with mx.autograd.record():
        # 1. Compute the embedding of the center words
        emb_in = embedding(center)

        # 2. Compute the context embedding  and apply mask
        emb_out = embedding_out(word_context_negatives)
        emb_out = emb_out * word_context_negatives_mask.expand_dims(-1)

        # 3. Compute the prediction
        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
        pred = pred.squeeze() * word_context_negatives_mask
        label = mx.nd.concat(word_context_mask, mx.nd.zeros_like(negatives), dim=1)

        # 4. Compute the Loss function (SigmoidBinaryCrossEntropyLoss)
        loss = loss_function(pred, label)

    # Compute the gradient
    loss.backward()

    # Update the parameters
    trainer.step(batch_size=1)

    if batch_id % 500 == 0:
        print(f'Batch {batch_id}: loss = {loss.mean().asscalar():.2f}\t(Took {time.time()-start:.2f} seconds)')
        get_knn(token_to_idx, idx_to_token, embedding, 5, "president")
    batch_id += 1


print(f'Batch {batch_id}: loss = {loss.mean().asscalar():.2f}\t(After {time.time()-start:.2f} seconds)')
knn = get_knn(token_to_idx, idx_to_token, embedding, 5, "president")
```

[1] word2vec工具. https://code.google.com/archive/p/word2vec/

[2] Mikolov, Tomas, et al. “Distributed representations of words and phrases and their compositionality.” Advances in neural information processing systems. 2013.[2] Mikolov, Tomas, et al. “Distributed representations of words and phrases and their compositionality.” Advances in neural information processing systems. 2013.
