```{.python .input  n=1}
from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import collections
```

```{.python .input  n=34}
text.embedding.get_pretrained_file_names('fasttext')[:5]
```

```{.json .output n=34}
[
 {
  "data": {
   "text/plain": "['wiki.ab.vec', 'wiki.ace.vec', 'wiki.ady.vec', 'wiki.aa.vec', 'wiki.af.vec']"
  },
  "execution_count": 34,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/ubuntu/miniconda3/lib/python3.6/site-packages/mxnet/contrib/text/embedding.py:278: UserWarning: At line 1 of the pre-trained text embedding file: token 111051 with 1-dimensional vector [300.0] is likely a header and is skipped.\n  'skipped.' % (line_num, token, elems))\n"
 }
]
```

```{.python .input  n=5}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(2, 300)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=6}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567     0.21454    -0.035389   -0.24299    -0.095645  ]\n [ 0.10444    -0.10858     0.27212     0.13299    -0.33164999]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=7}
my_embedding.to_indices(['hello', 'world'])
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "[2, 1]"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
layer = gluon.nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
layer(nd.array([2, 1])).shape
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(2, 300)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=9}
layer(nd.array([2, 1]))[:, :5]
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567     0.21454    -0.035389   -0.24299    -0.095645  ]\n [ 0.10444    -0.10858     0.27212     0.13299    -0.33164999]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=35}
text.embedding.get_pretrained_file_names('glove')[:5]
```

```{.json .output n=35}
[
 {
  "data": {
   "text/plain": "['glove.42B.300d.txt',\n 'glove.6B.50d.txt',\n 'glove.6B.100d.txt',\n 'glove.6B.200d.txt',\n 'glove.6B.300d.txt']"
  },
  "execution_count": 35,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=11}
glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading /home/ubuntu/.mxnet/embeddings/glove/glove.6B.zip from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/embeddings/glove/glove.6B.zip...\n"
 }
]
```

```{.python .input  n=12}
print(glove_6b50d.token_to_idx['nice'])
print(glove_6b50d.idx_to_token[2586])
print(glove_6b50d.vec_len)
print(len(glove_6b50d))
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "3083\nduty\n50\n400001\n"
 }
]
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

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "\n[ 1.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=15}
cos_sim(x, z)
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "\n[-1.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "['daughter']"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=19}
get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "['japan']"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=20}
get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "['biggest']"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=21}
get_top_k_by_analogy(glove_6b50d, 1, 'do', 'did', 'go')
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "['went']"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "\n[ 0.96583432]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=24}
cos_sim_word_analogy(glove_6b50d, 'beijing', 'china', 'tokyo', 'japan')
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "\n[ 0.90540648]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=33}
cos_sim_word_analogy(glove_6b50d, 'bad', 'worst', 'big', 'biggest')
```

```{.json .output n=33}
[
 {
  "data": {
   "text/plain": "\n[ 0.80596256]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 33,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=26}
cos_sim_word_analogy(glove_6b50d, 'do', 'did', 'go', 'went')
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "\n[ 0.92422962]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
