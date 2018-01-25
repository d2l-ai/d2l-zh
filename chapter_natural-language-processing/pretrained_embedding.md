```{.python .input  n=10}
from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import collections
```

```{.python .input}
text.embedding.get_pretrained_file_names('fasttext')[:10]
```

```{.python .input}
text_data = " hello world \n hello nice world \n hi world \n"
counter = text.utils.count_tokens_from_str(text_data)
```

```{.python .input}
my_vocab = text.vocab.Vocabulary(counter)
my_embedding = text.embedding.create(
    'fasttext', pretrained_file_name='wiki.simple.vec', vocabulary=my_vocab)
```

```{.python .input}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

```{.python .input}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

```{.python .input}
my_embedding.to_indices(['hello', 'world'])
```

```{.python .input}
layer = gluon.nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
layer(nd.array([2, 1])).shape
```

```{.python .input}
layer(nd.array([2, 1]))[:, :5]
```

```{.python .input}
text.embedding.get_pretrained_file_names('glove')
```

```{.python .input}
glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')
```

```{.python .input}
print(glove_6b50d.token_to_idx['nice'])
print(glove_6b50d.idx_to_token[2586])
print(glove_6b50d.vec_len)
print(len(glove_6b50d))
```

```{.python .input}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

```{.python .input}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

cos_sim(x, y)
```

```{.python .input}
cos_sim(x, z)
```

```{.python .input}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1)).reshape((-1,1))
```

```{.python .input}
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

```{.python .input}
get_top_k_by_analogy(glove_6b50d, 10, 'man', 'woman', 'son')
```

```{.python .input}
get_top_k_by_analogy(glove_6b50d, 10, 'beijing', 'china', 'tokyo')
```

```{.python .input}
get_top_k_by_analogy(glove_6b50d, 10, 'bad', 'worst', 'big')
```

```{.python .input}
get_top_k_by_analogy(glove_6b50d, 10, 'do', 'did', 'go')
```

```{.python .input}
def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])
```

```{.python .input}
cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
```

```{.python .input}
cos_sim_word_analogy(glove_6b50d, 'beijing', 'china', 'tokyo', 'japan')
```

```{.python .input}
cos_sim_word_analogy(glove_6b50d, 'bad', 'worst', 'big', 'biggest')
```

```{.python .input}
cos_sim_word_analogy(glove_6b50d, 'do', 'did', 'go', 'went')
```
