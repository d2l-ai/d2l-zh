# Word Similarity and Analogy
:label:`sec_synonyms`

In :numref:`sec_word2vec_pretraining`, 
we trained a word2vec model on a small dataset, 
and applied it
to find semantically similar words 
for an input word.
In practice,
word vectors that are pretrained
on large corpora can be
applied to downstream
natural language processing tasks,
which will be covered later
in :numref:`chap_nlp_app`.
To demonstrate 
semantics of pretrained word vectors
from large corpora in a straightforward way,
let us apply them
in the word similarity and analogy tasks.

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

## Loading Pretrained Word Vectors

Below lists pretrained GloVe embeddings of dimension 50, 100, and 300,
which can be downloaded from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
The pretrained fastText embeddings are available in multiple languages.
Here we consider one English version (300-dimensional "wiki.en") that can be downloaded from the
[fastText website](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

To load these pretrained GloVe and fastText embeddings, we define the following `TokenEmbedding` class.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Below we load the
50-dimensional GloVe embeddings
(pretrained on a Wikipedia subset).
When creating the `TokenEmbedding` instance,
the specified embedding file has to be downloaded if it
was not yet.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Output the vocabulary size. The vocabulary contains 400000 words (tokens) and a special unknown token.

```{.python .input}
#@tab all
len(glove_6b50d)
```

We can get the index of a word in the vocabulary, and vice versa.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Applying Pretrained Word Vectors

Using the loaded GloVe vectors,
we will demonstrate their semantics
by applying them
in the following word similarity and analogy tasks.


### Word Similarity

Similar to :numref:`subsec_apply-word-embed`,
in order to find semantically similar words
for an input word
based on cosine similarities between
word vectors,
we implement the following `knn`
($k$-nearest neighbors) function.

```{.python .input}
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Then, we 
search for similar words
using the pretrained word vectors 
from the `TokenEmbedding` instance `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

The vocabulary of the pretrained word vectors
in `glove_6b50d` contains 400000 words and a special unknown token. 
Excluding the input word and unknown token,
among this vocabulary
let us find 
three most semantically similar words
to word "chip".

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Below outputs similar words
to "baby" and "beautiful".

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Word Analogy

Besides finding similar words,
we can also apply word vectors
to word analogy tasks.
For example,
“man”:“woman”::“son”:“daughter”
is the form of a word analogy:
“man” is to “woman” as “son” is to “daughter”.
Specifically,
the word analogy completion task
can be defined as:
for a word analogy 
$a : b :: c : d$, given the first three words $a$, $b$ and $c$, find $d$. 
Denote the vector of word $w$ by $\text{vec}(w)$. 
To complete the analogy,
we will find the word 
whose vector is most similar
to the result of $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Let us verify the "male-female" analogy using the loaded word vectors.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Below completes a
“capital-country” analogy: 
“beijing”:“china”::“tokyo”:“japan”.
This demonstrates 
semantics in the pretrained word vectors.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

For the
“adjective-superlative adjective” analogy
such as 
“bad”:“worst”::“big”:“biggest”,
we can see that the pretrained word vectors
may capture the syntactic information.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

To show the captured notion
of past tense in the pretrained word vectors,
we can test the syntax using the
"present tense-past tense" analogy: “do”:“did”::“go”:“went”.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Summary

* In practice, word vectors that are pretrained on large corpora can be applied to downstream natural language processing tasks.
* Pretrained word vectors can be applied to the word similarity and analogy tasks.


## Exercises

1. Test the fastText results using `TokenEmbedding('wiki.en')`.
1. When the vocabulary is extremely large, how can we find similar words or complete a word analogy faster?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
