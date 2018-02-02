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

```{.python .input  n=34}
text.embedding.get_pretrained_file_names('fasttext')[:5]
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

词典除了包括数据集中四个不同的词语，还包括一个特殊的未知词符号。看一下词典大小。

```{.python .input}
len(my_embedding)
```

任意一个词典以外词的词向量默认为零向量。

```{.python .input}
my_embedding.get_vecs_by_tokens('beautiful')[:10]
```

看一下数据集中两个词“hello”和“world”词向量的形状。fastText中每个词均使用300维的词向量。

```{.python .input  n=5}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

打印“hello”和“world”词向量前五个元素。

```{.python .input  n=6}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

看一下“hello”和“world”在词典中的索引。

```{.python .input  n=7}
my_embedding.to_indices(['hello', 'world'])
```

### 使用预训练词向量初始化gluon.nn.Embedding

我们可以使用预训练的词向量初始化`gluon.nn.Embedding`。

```{.python .input  n=8}
layer = gluon.nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
```

使用词典中“hello”和“world”两个词在词典中的索引，我们可以通过`gluon.nn.Embedding`得到它们的词向量，并向神经网络的下一层传递。

```{.python .input  n=9}
layer(nd.array([2, 1]))[:, :5]
```

## 由预训练词向量建立词典——以GloVe为例

看一下GloVe前五个预训练的词向量。

```{.python .input  n=35}
text.embedding.get_pretrained_file_names('glove')[:5]
```

为了演示简便，我们使用小一点的词向量，例如50维。这里不再传入根据数据集建立的词典，而是直接使用预训练词向量中的词典。

```{.python .input  n=11}
glove_6b50d = text.embedding.create('glove', 
                                    pretrained_file_name='glove.6B.50d.txt')
```

看一下这个词典多大。注意其中包含一个特殊的未知词符号。

```{.python .input}
print(len(glove_6b50d))
```

我们可以访问词向量的属性。

```{.python .input  n=12}
# 词到索引。
print(glove_6b50d.token_to_idx['beautiful'])
# 索引到词。
print(glove_6b50d.idx_to_token[3367])
# 词向量长度。
print(glove_6b50d.vec_len)
```

## 预训练词向量的应用——以GloVe为例

为了应用预训练词向量，我们需要定义余弦相似度。它可以比较两个向量之间的相似度。

```{.python .input  n=13}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

余弦相似度的值域在-1到1之间。余弦相似度值越大，两个向量越接近。

```{.python .input  n=14}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

print(cos_sim(x, y))
print(cos_sim(x, z))
```

### 求近似词

给定任意词，我们可以从整个词典（大小40万，不含未知词符号）中找出与它最接近的$k$个词（$k$ nearest neighbors）。词与词之间的相似度可以用两个词向量的余弦相似度表示。

```{.python .input}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1)).reshape((-1,1))

def get_knn(token_embedding, k, word):
    word_vec = token_embedding.get_vecs_by_tokens([word]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k+2,
                      ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # 除去未知词符号和输入词。
    return token_embedding.to_tokens(indices[2:])
```

查找词典中与“baby”最接近的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'baby')
```

验证一下“baby”和“babies”两个词向量之间的余弦相似度。

```{.python .input}
cos_sim(glove_6b50d.get_vecs_by_tokens('baby'),
        glove_6b50d.get_vecs_by_tokens('babies'))
```

查找词典中与“computers”最接近的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'computers')
```

查找词典中与“run”最接近的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'run')
```

查找词典中与“beautiful”最接近的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'beautiful')
```

### 求类比词

我们可以使用预训练词向量求词与词之间的类比关系。例如，man : woman :: son : daughter 是一个类比例子：“man”之于“woman”相当于“son”之于“daughter”。求类比词问题可以定义为：对于类比关系中的四个词 a : b :: c : d，给定前三个词a, b, c，求d。解类比词的思路是，找到和c+(b-a)的结果词向量最相似的词向量。

本例中，我们将从整个词典（大小40万，不含未知词符号）中找类比词。

```{.python .input  n=17}
def get_top_k_by_analogy(token_embedding, k, word1, word2, word3):
    word_vecs = token_embedding.get_vecs_by_tokens([word1, word2, word3])
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k+1,
                      ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]

    # 不考虑未知词为可能的类比词。
    if token_embedding.to_tokens(indices[0]) == token_embedding.unknown_token:
        return token_embedding.to_tokens(indices[1:])
    else:
        return token_embedding.to_tokens(indices[:-1])
```

“男-女”类比：“man”之于“woman”相当于“son”之于什么？

```{.python .input  n=18}
get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
```

验证一下vec(“son”)+vec(“woman”)-vec(“man”)与vec(“daughter”)两个向量之间的余弦相似度。

```{.python .input}
def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
```

“首都-国家”类比：“beijing”之于“china”相当于“tokyo”之于什么？

```{.python .input  n=19}
get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
```

“形容词-形容词最高级”类比：“bad”之于“worst”相当于“big”之于什么？

```{.python .input  n=20}
get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
```

“动词一般时-动词过去时”类比：“do”之于“did”相当于“go”之于什么？

```{.python .input  n=21}
get_top_k_by_analogy(glove_6b50d, 1, 'do', 'did', 'go')
```

## 结论

* 使用`mxnet.contrib.text`可以轻松载入预训练的词向量。
* 我们可以应用预训练的词向量求相似词和类比词。


## 练习

* 将近似词和类比词应用中的$k$调大一些，观察结果。
* 测试一下fastText的中文词向量（pretrained_file_name='wiki.zh.vec'）。
* 如果在[使用循环神经网络的语言模型](../chapter_recurrent-neural-networks/rnn-gluon.md)中将Embedding层初始化为预训练的词向量，效果如何？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4373)
