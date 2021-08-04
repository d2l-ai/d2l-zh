# Sentiment Analysis and the Dataset
:label:`sec_sentiment`


With the proliferation of online social media
and review platforms,
a plethora of
opinionated data
have been logged,
bearing great potential for
supporting decision making processes.
*Sentiment analysis*
studies people's sentiments
in their produced text,
such as product reviews,
blog comments,
and
forum discussions.
It enjoys wide applications
to fields as diverse as 
politics (e.g., analysis of public sentiments towards policies),
finance (e.g., analysis of sentiments of the market),
and 
marketing (e.g., product research and brand management).

Since sentiments
can be categorized
as discrete polarities or scales (e.g., positive and negative),
we can consider 
sentiment analysis 
as a text classification task,
which transforms a varying-length text sequence
into a fixed-length text category.
In this chapter,
we will use Stanford's [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
for sentiment analysis. 
It consists of a training set and a testing set, 
either containing 25000 movie reviews downloaded from IMDb.
In both datasets, 
there are equal number of 
"positive" and "negative" labels,
indicating different sentiment polarities.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  Reading the Dataset

First, download and extract this IMDb review dataset
in the path `../data/aclImdb`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Next, read the training and test datasets. Each example is a review and its label: 1 for "positive" and 0 for "negative".

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

## Preprocessing the Dataset

Treating each word as a token
and filtering out words that appear less than 5 times,
we create a vocabulary out of the training dataset.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

After tokenization,
let's plot the histogram of
review lengths in tokens.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

As we expected,
the reviews have varying lengths.
To process
a minibatch of such reviews at each time,
we set the length of each review to 500 with truncation and padding,
which is similar to 
the preprocessing step 
for the machine translation dataset
in :numref:`sec_machine_translation`.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Creating Data Iterators

Now we can create data iterators.
At each iteration, a minibatch of examples are returned.

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Putting All Things Together

Last, we wrap up the above steps into the `load_data_imdb` function.
It returns training and test data iterators and the vocabulary of the IMDb review dataset.

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Summary

* Sentiment analysis studies people's sentiments in their produced text, which is considered as a text classification problem that transforms a varying-length text sequence
into a fixed-length text category.
* After preprocessing, we can load Stanford's large movie review dataset (IMDb review dataset) into data iterators with a vocabulary.


## Exercises


1. What hyperparameters in this section can we modify to accelerate training sentiment analysis models?
1. Can you implement a function to load the dataset of [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) into data iterators and labels for sentiment analysis?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:
