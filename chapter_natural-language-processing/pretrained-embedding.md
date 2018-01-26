```{.python .input  n=1}
from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import collections
```

```{.python .input  n=34}
text.embedding.get_pretrained_file_names('fasttext')[:5]
```

```{.python .input  n=3}
text_data = " hello world \n hello nice world \n hi world \n"
counter = text.utils.count_tokens_from_str(text_data)
```

```{.python .input  n=4}
my_vocab = text.vocab.Vocabulary(counter)
my_embedding = text.embedding.create(
    'fasttext', pretrained_file_name='wiki.simple.vec', vocabulary=my_vocab)
```

```{.python .input  n=5}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

```{.python .input  n=6}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

```{.python .input  n=7}
my_embedding.to_indices(['hello', 'world'])
```

```{.python .input  n=8}
layer = gluon.nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
layer(nd.array([2, 1])).shape
```

```{.python .input  n=9}
layer(nd.array([2, 1]))[:, :5]
```

```{.python .input  n=35}
text.embedding.get_pretrained_file_names('glove')[:5]
```

```{.python .input  n=11}
glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')
```

```{.python .input  n=12}
print(glove_6b50d.token_to_idx['nice'])
print(glove_6b50d.idx_to_token[2586])
print(glove_6b50d.vec_len)
print(len(glove_6b50d))
```

```{.python .input  n=13}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

```{.python .input  n=14}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

cos_sim(x, y)
```

```{.python .input  n=15}
cos_sim(x, z)
```

```{.python .input  n=16}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1)).reshape((-1,1))
```

```{.python .input  n=17}
def get_top_k_by_analogy(token_embedding, k, word1, word2, word3):
    word_vecs = token_embedding.get_vecs_by_tokens([word1, word2, word3])
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))

    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)

    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k+1, ret_typ='indices')

    indices = [int(i.asscalar()) for i in indices]

    # 不考虑未知词为可能的类比词。
    if token_embedding.to_tokens(indices[0]) == token_embedding.unknown_token:
        return token_embedding.to_tokens(indices[1:])
    else:
        return token_embedding.to_tokens(indices[:-1])
```

```{.python .input  n=18}
get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
```

```{.python .input  n=19}
get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
```

```{.python .input  n=20}
get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
```

```{.python .input  n=21}
get_top_k_by_analogy(glove_6b50d, 1, 'do', 'did', 'go')
```

```{.python .input  n=22}
def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])
```

```{.python .input  n=23}
cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
```

```{.python .input  n=24}
cos_sim_word_analogy(glove_6b50d, 'beijing', 'china', 'tokyo', 'japan')
```

```{.python .input  n=33}
cos_sim_word_analogy(glove_6b50d, 'bad', 'worst', 'big', 'biggest')
```

```{.python .input  n=26}
cos_sim_word_analogy(glove_6b50d, 'do', 'did', 'go', 'went')
```
