# 求近义词和类比词

在[“word2vec 的实现”](./word2vec_gluon.md)一节中我们实现了如果使用余弦相似度来寻找近义词，并测试了在一个小规模数据集PTB上训练好的word2vec模型的效果。这一节我们将使用在大规模语料上预训练的词向量模型，并同时演示如何用它们来寻找类比词。首先导入实验所需的包或模块。

```{.python .input  n=1}
from mxnet import nd
from mxnet.contrib import text
from mxnet.gluon import nn
```

## 使用预训练的词向量

MXNet的`contrib.text`提供了跟自然语言处理相关的函数和类。下面查看它目前提供的有预训练模型的词向量模型：

```{.python .input}
text.embedding.get_pretrained_file_names().keys()
```

给定一个模型，我们可以查看它提供了在哪些数据集上训练好的模型。

```{.python .input  n=35}
print(text.embedding.get_pretrained_file_names('glove'))
```

这里的命名规范大致是“模型.数据集词数.词向量长度.txt”，更多信息可以参考GloVe [1]和fastText [2]项目信息。下面我们使用数据集`glove.6B.50d.txt`，它是基于维基百科的一个子集。使用模型名和数据集名可以创建一个词向量实例，创建时会自动下载对应的预训练模型。

```{.python .input  n=11}
glove_6b50d = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.50d.txt')
```

打印词典大小。其中含有40万个词，和一个特殊的未知词符号。

```{.python .input}
len(glove_6b50d)
```

我们可以通过词来得到它在词典中的索引，反之也可以。

```{.python .input  n=12}
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

### 求近义词

这里重新实现[“word2vec 的实现”](./word2vec_gluon.md)一节中介绍过的使用余弦相似度来寻找近义词。首先定义一个通过余弦相似度来求$k$近邻。

```{.python .input}
def knn(W, x, k):
    cos = nd.dot(W, x.reshape((-1,))) / (
        nd.sum(W*W, axis=1).sqrt() * nd.sum(x*x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]
```

然后通过预训练好的模型寻找近义词。

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, 
                    embed.get_vecs_by_tokens([query_token]), k+2)
    for i, c in zip(topk[2:], cos[2:]):  # 除去输入词和未知词。
        print('similarity=%.3f: %s' % (c, (embed.idx_to_token[i])))
        

```

先试一下“chip”的近义词。

```{.python .input}
get_similar_tokens('chip', 3, glove_6b50d)
```

查找“baby”和“beautiful”的近义词个词。

```{.python .input}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 求类比词

除了求近义词以外，我们还可以使用预训练词向量求词与词之间的类比关系。例如，man（男人）: woman（女人）:: son（儿子） : daughter（女儿）是一个类比例子：“man”之于“woman”相当于“son”之于“daughter”。求类比词问题可以定义为：对于类比关系中的四个词 $a : b :: c : d$，给定前三个词$a$、$b$和$c$，求$d$。设词$w$的词向量为$\text{vec}(w)$。而解类比词的思路是，找到和$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的结果向量最相似的词向量。

```{.python .input}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 2)
    return embed.idx_to_token[topk[1]] # 除去未知词。
```

验证下“男-女”类比：

```{.python .input  n=18}
get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
```

“首都-国家”类比：“beijing”（北京）之于“china”（中国）相当于“tokyo”（东京）之于什么？答案应该是“japan”（日本）。

```{.python .input  n=19}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

“形容词-形容词最高级”类比：“bad”（坏的）之于“worst”（最坏的）相当于“big”（大的）之于什么？答案应该是“biggest”（最大的）。

```{.python .input  n=20}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

“动词一般时-动词过去时”类比：“do”（做）之于“did”（做过）相当于“go”（去）之于什么？答案应该是“went”（去过）。

```{.python .input  n=21}
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 小结


* 我们可以应用预训练的词向量求近义词和类比词。


## 练习

* 测试一下fastText的结果。特别的，fastText有中文词向量（pretrained_file_name='wiki.zh.vec'）。
* 如果词典大小特别大，如何提升寻找速度？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4373)

![](../img/qr_similarity-analogy.svg)

## 参考文献

[1] GloVe项目网站. https://nlp.stanford.edu/projects/glove/

[2] fastText项目网站. https://fasttext.cc/
