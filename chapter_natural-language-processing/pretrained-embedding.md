# 使用预训练的词向量

本节介绍如何通过`mxnet.contrib.text`使用预训练的词向量。需要注意的是，`mxnet.contrib.text`正在测试中并可能在未来有改动。如有改动，本节内容会作相应更新。

本节使用的预训练的GloVe和fastText词向量分别来自：

* GloVe项目网站：https://nlp.stanford.edu/projects/glove/
* fastText项目网站：https://fasttext.cc/

我们先载入需要的包。

```{.python .input  n=1}
from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
```

## 由数据集建立词典和载入词向量——以fastText为例

看一下fastText前五个预训练的词向量。它们分别从不同语言的Wikipedia数据集训练得到。

```{.python .input  n=2}
text.embedding.get_pretrained_file_names('fasttext')[:5]
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "['wiki.ab.vec', 'wiki.ace.vec', 'wiki.ady.vec', 'wiki.aa.vec', 'wiki.af.vec']"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 访问词向量

为了演示方便，我们创建一个很小的文本数据集，并计算词频。

```{.python .input  n=3}
text_data = " hello world \n hello nice world \n hi world \n"
counter = text.utils.count_tokens_from_str(text_data)
```

我们先根据数据集建立词典，并为该词典中的词载入fastText词向量。这里使用Simple English的预训练词向量。

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

词典除了包括数据集中四个不同的词语，还包括一个特殊的未知词符号。看一下词典大小。

```{.python .input  n=5}
len(my_embedding)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "5"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

任意一个词典以外词的词向量默认为零向量。

```{.python .input  n=6}
my_embedding.get_vecs_by_tokens('beautiful')[:10]
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "\n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n<NDArray 10 @cpu(0)>"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

看一下数据集中两个词"hello"和"world"词向量的形状。fastText中每个词均使用300维的词向量。

```{.python .input  n=7}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(2, 300)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

打印"hello"和"world"词向量前五个元素。

```{.python .input  n=8}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567     0.21454    -0.035389   -0.24299    -0.095645  ]\n [ 0.10444    -0.10858     0.27212     0.13299    -0.33164999]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

看一下"hello"和"world"在词典中的索引。

```{.python .input  n=9}
my_embedding.to_indices(['hello', 'world'])
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "[2, 1]"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 使用预训练词向量初始化gluon.nn.Embedding

我们可以使用预训练的词向量初始化`gluon.nn.Embedding`。

```{.python .input  n=10}
layer = gluon.nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
```

使用词典中"hello"和"world"两个词在词典中的索引，我们可以通过`gluon.nn.Embedding`得到它们的词向量，并向神经网络的下一层传递。

```{.python .input  n=11}
layer(nd.array([2, 1]))[:, :5]
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567     0.21454    -0.035389   -0.24299    -0.095645  ]\n [ 0.10444    -0.10858     0.27212     0.13299    -0.33164999]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 由预训练词向量建立词典——以GloVe为例

看一下GloVe前五个预训练的词向量。

```{.python .input  n=12}
text.embedding.get_pretrained_file_names('glove')[:5]
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "['glove.42B.300d.txt',\n 'glove.6B.50d.txt',\n 'glove.6B.100d.txt',\n 'glove.6B.200d.txt',\n 'glove.6B.300d.txt']"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

为了演示简便，我们使用小一点的词向量，例如50维。这里不再传入根据数据集建立的词典，而是直接使用预训练词向量中的词典。

```{.python .input  n=13}
glove_6b50d = text.embedding.create('glove', 
                                    pretrained_file_name='glove.6B.50d.txt')
```

看一下这个词典多大。注意其中包含一个特殊的未知词符号。

```{.python .input  n=14}
print(len(glove_6b50d))
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "400001\n"
 }
]
```

我们可以访问词向量的属性。

```{.python .input  n=15}
# 词到索引。
print(glove_6b50d.token_to_idx['beautiful'])
# 索引到词。
print(glove_6b50d.idx_to_token[3367])
# 词向量长度。
print(glove_6b50d.vec_len)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "3367\nbeautiful\n50\n"
 }
]
```

## 使用预训练词向量解类比题——以GloVe为例

余弦相似度可以比较两个向量之间的相似度。我们定义该相似度。

```{.python .input  n=16}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

```{.python .input  n=17}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

cos_sim(x, y)
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "\n[ 1.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=18}
cos_sim(x, z)
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[-1.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=19}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1)).reshape((-1,1))
```

```{.python .input  n=20}
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

```{.python .input  n=21}
get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "['daughter']"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=22}
get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
```

```{.json .output n=22}
[
 {
  "data": {
   "text/plain": "['japan']"
  },
  "execution_count": 22,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=23}
get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "['biggest']"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=24}
get_top_k_by_analogy(glove_6b50d, 1, 'do', 'did', 'go')
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "['went']"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=25}
def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])
```

```{.python .input  n=26}
cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "\n[ 0.96583432]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=27}
cos_sim_word_analogy(glove_6b50d, 'beijing', 'china', 'tokyo', 'japan')
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "\n[ 0.90540648]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=28}
cos_sim_word_analogy(glove_6b50d, 'bad', 'worst', 'big', 'biggest')
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "\n[ 0.80596256]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=29}
cos_sim_word_analogy(glove_6b50d, 'do', 'did', 'go', 'went')
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "\n[ 0.92422962]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 结论

* 使用mxnet.contrib.text可以轻松载入预训练的词向量。
* 使用预训练的词向量可以执行词汇类比任务。


## 练习

* 测试一下fastText的中文词向量：text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec')
* 如果在[使用循环神经网络的语言模型](../chapter_recurrent-neural-networks/rnn-gluon.md)中将Embedding层初始化为预训练的词向量，效果如何？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4372)
