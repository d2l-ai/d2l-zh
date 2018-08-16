# 求近似词和类比词

上一节中，我们在一个较小的语料上训练词嵌入模型并求近似词。对于很多自然语言处理任务，我们还可以直接使用在大规模语料上预训练的词向量。

本节主要介绍如何获取并应用在大规模语料上预训练的词向量。我们还将进一步探究求近似词问题，并实验求类比词的方法。本节使用的预训练的GloVe和fastText词向量分别来自它们的项目网站 [1,2]。

首先导入实验所需的包或模块。

```{.python .input  n=1}
from mxnet import nd
from mxnet.contrib import text
from mxnet.gluon import nn
```

## 由数据集建立词典和载入词向量

下面，我们以fastText为例，由数据集建立词典并载入词向量。fastText提供了基于不同语言的多套预训练的词向量。这些词向量是在大规模语料上训练得到的，例如维基百科语料。以下打印了其中的10种。

```{.python .input  n=34}
print(text.embedding.get_pretrained_file_names('fasttext')[:10])
```

### 访问词向量

为了演示方便，我们创建一个很小的文本数据集，并计算词频。

```{.python .input  n=3}
text_data = ' hello world \n hello nice world \n hi world \n'
counter = text.utils.count_tokens_from_str(text_data)
```

我们先根据数据集建立词典，并为该词典中的词载入fastText词向量。这里使用Simple English的预训练词向量。

```{.python .input  n=4}
my_vocab = text.vocab.Vocabulary(counter)
my_embedding = text.embedding.create(
    'fasttext', pretrained_file_name='wiki.simple.vec', vocabulary=my_vocab)
```

词典除了包括数据集中四个不同的词语，还包括一个特殊的未知词符号。打印词典大小。

```{.python .input}
len(my_embedding)
```

默认情况下，任意一个词典以外词的词向量为零向量。

```{.python .input}
my_embedding.get_vecs_by_tokens('beautiful')[:10]
```

fastText中每个词均使用300维的词向量。打印数据集中两个词“hello”和“world”词向量的形状。

```{.python .input  n=5}
my_embedding.get_vecs_by_tokens(['hello', 'world']).shape
```

打印“hello”和“world”词向量前五个元素。

```{.python .input  n=6}
my_embedding.get_vecs_by_tokens(['hello', 'world'])[:, :5]
```

打印“hello”和“world”在词典中的索引。

```{.python .input  n=7}
my_embedding.to_indices(['hello', 'world'])
```

### 使用预训练词向量初始化Embedding实例

我们在[“循环神经网络——使用Gluon”](../chapter_recurrent-neural-networks/rnn-gluon.md)一节中介绍了Gluon中的Embedding实例，并对其中每个词的向量做了随机初始化。实际上，我们还可以使用预训练的词向量初始化Embedding实例。

```{.python .input  n=8}
layer = nn.Embedding(len(my_embedding), my_embedding.vec_len)
layer.initialize()
layer.weight.set_data(my_embedding.idx_to_vec)
```

使用词典中“hello”和“world”两个词在词典中的索引，我们可以通过Embedding实例得到它们的预训练词向量，并向神经网络的下一层传递。

```{.python .input  n=9}
layer(nd.array([2, 1]))[:, :5]
```

## 由预训练词向量建立词典——以GloVe为例

除了使用数据集建立词典外，我们还可以直接由预训练词向量建立词典。

这一次我们使用GloVe的预训练词向量。以下打印了GloVe提供的各套预训练词向量。这些词向量是在大规模语料上训练得到的，例如维基百科语料和推特语料。

```{.python .input  n=35}
print(text.embedding.get_pretrained_file_names('glove'))
```

我们使用50维的词向量。和之前不同，这里不再传入根据数据集建立的词典，而是直接使用预训练词向量中的词建立词典。

```{.python .input  n=11}
glove_6b50d = text.embedding.create('glove', 
                                    pretrained_file_name='glove.6B.50d.txt')
```

打印词典大小。注意其中包含一个特殊的未知词符号。

```{.python .input}
print(len(glove_6b50d))
```

我们可以访问词向量的属性。

```{.python .input  n=12}
# 词到索引，索引到词。
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 应用预训练词向量

下面我们以GloVe为例，展示预训练词向量的应用。首先，我们定义余弦相似度，并用它表示两个向量之间的相似度。

```{.python .input  n=13}
def cos_sim(x, y):
    return nd.dot(x, y) / (x.norm() * y.norm())
```

余弦相似度的值域在-1到1之间。两个余弦相似度越大的向量越相似。

```{.python .input  n=14}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])
cos_sim(x, y), cos_sim(x, z)
```

### 求近似词

给定任意词，我们可以从GloVe的整个词典（大小40万，不含未知词符号）中找出与它最接近的$k$个词。[“词嵌入：word2vec”](word2vec.md)一节中已经提到，词与词之间的相似度可以用两个词向量的余弦相似度表示。

```{.python .input}
def norm_vecs_by_row(x):
    # 分母中添加的 1e-10 是为了数值稳定性。
    return x / (nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))

def get_knn(token_embedding, k, word):
    word_vec = token_embedding.get_vecs_by_tokens([word]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k+1,
                      ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # 除去输入词。
    return token_embedding.to_tokens(indices[1:])
```

查找词典中与“baby”最近似的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'baby')
```

验证一下“baby”和“babies”两个词向量之间的余弦相似度。

```{.python .input}
cos_sim(glove_6b50d.get_vecs_by_tokens('baby'),
        glove_6b50d.get_vecs_by_tokens('babies'))
```

查找词典中与“computers”最近似的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'computers')
```

查找词典中与“run”最近似的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'run')
```

查找词典中与“beautiful”最近似的5个词。

```{.python .input}
get_knn(glove_6b50d, 5, 'beautiful')
```

### 求类比词

除近似词以外，我们还可以使用预训练词向量求词与词之间的类比关系。例如，man（男人）: woman（女人）:: son（儿子） : daughter（女儿）是一个类比例子：“man”之于“woman”相当于“son”之于“daughter”。求类比词问题可以定义为：对于类比关系中的四个词 $a : b :: c : d$，给定前三个词$a$、$b$和$c$，求$d$。设词$w$的词向量为$\text{vec}(w)$。而解类比词的思路是，找到和$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的结果向量最相似的词向量。

本例中，我们将从整个词典（大小40万，不含未知词符号）中搜索类比词。

```{.python .input  n=17}
def get_top_k_by_analogy(token_embedding, k, word1, word2, word3):
    word_vecs = token_embedding.get_vecs_by_tokens([word1, word2, word3])
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k,
                      ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    return token_embedding.to_tokens(indices)
```

“男-女”类比：“man”之于“woman”相当于“son”之于什么？

```{.python .input  n=18}
get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
```

验证一下$\text{vec(son)+vec(woman)-vec(man)}$与$\text{vec(daughter)}$两个向量之间的余弦相似度。

```{.python .input}
def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
```

“首都-国家”类比：“beijing”（北京）之于“china”（中国）相当于“tokyo”（东京）之于什么？答案应该是“japan”（日本）。

```{.python .input  n=19}
get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
```

“形容词-形容词最高级”类比：“bad”（坏的）之于“worst”（最坏的）相当于“big”（大的）之于什么？答案应该是“biggest”（最大的）。

```{.python .input  n=20}
get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
```

“动词一般时-动词过去时”类比：“do”（做）之于“did”（做过）相当于“go”（去）之于什么？答案应该是“went”（去过）。

```{.python .input  n=21}
get_top_k_by_analogy(glove_6b50d, 1, 'do', 'did', 'go')
```

## 小结


* 我们可以应用预训练的词向量求近似词和类比词。


## 练习

* 将近似词和类比词应用中的$k$调大一些，观察结果。
* 测试一下fastText的中文词向量（pretrained_file_name='wiki.zh.vec'）。
* 如果在[“循环神经网络的Gluon实现”](../chapter_recurrent-neural-networks/rnn-gluon.md)一节中将Embedding实例里的参数初始化为预训练的词向量，效果如何？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4373)

![](../img/qr_similarity-analogy.svg)

## 参考文献

[1] GloVe项目网站. https://nlp.stanford.edu/projects/glove/

[2] fastText项目网站. https://fasttext.cc/
