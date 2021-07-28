# 词相似性和类比
:label:`sec_synonyms`

在 :numref:`sec_word2vec_pretraining` 中，我们在一个小数据集上训练了一个 word2vec 模型，并应用它来查找输入词语义上相似的词。实际上，在大型语言上预训练的单词向量可以应用于下游自然语言处理任务，后面将在 :numref:`chap_nlp_app` 中介绍这些任务。为了以直接的方式展示来自大型语言的预训练单词矢量的语义，让我们在单词相似性和类比任务中应用它们。

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

## 加载预训练的词向量

下面列出了维度 50、100 和 300 的预训练 GLOVE 嵌入，可以从 [GloVe website](https://nlp.stanford.edu/projects/glove/) 下载。经过预训练的 FastText 嵌入提供多种语言版本。这里我们考虑一个英文版本（300 维 “wiki Ien”），它可以从 [fastText website](https://fasttext.cc/) 下载。

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

为了加载这些预训练的 Glove 和 FastText 嵌入，我们定义了以下 `TokenEmbedding` 类。

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

下面我们加载 50 维 GloVE 嵌入物（在维基百科子集上预训练）。创建 `TokenEmbedding` 实例时，如果尚未下载指定的嵌入文件，则必须下载该文件。

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

输出词汇量大小。词汇包含 40 万个单词（令牌）和一个特殊的未知词元。

```{.python .input}
#@tab all
len(glove_6b50d)
```

我们可以在词汇中获得单词的索引，反之亦然。

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 应用预训练的词向量

使用加载的 Glove 向量，我们将通过在以下单词相似性和类比任务中应用它们来演示它们的语义。 

### 词相似性

与 :numref:`subsec_apply-word-embed` 类似，为了根据词矢量之间的余弦相似性为输入词找到语义上相似的词，我们实现了以下 `knn`（$k$-最近邻）函数。

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

然后，我们使用 `TokenEmbedding` 实例 `embed` 实例 `embed` 中的预先训练的词向量来搜索类似的单词。

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d` 中的预训练单词矢量的词汇包含 40 万个单词和一个特殊的未知词元。不包括输入单词和未知词元，在这个词汇中，我们可以找到三个与单词 “芯片” 在语义上最相似的单词。

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

下面输出了与 “宝贝” 和 “美丽” 类似的词语。

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 单词类比

除了找到类似的单词之外，我们还可以将单词矢量应用于单词类比任务。例如，“男人”：“女人”። “儿子”：“女儿” 是一个词类比的形式：“男人” 是 “女人”，因为 “儿子” 就是 “女儿”。具体来说，“类比完成任务” 这个词可以定义为：对于单词类比 $a : b :: c : d$，前三个词 $a$、$b$ 和 $c$，找 $d$。用 $\text{vec}(w)$ 表示单词 $w$ 的矢量。为了完成这个比喻，我们将找到矢量与 $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$ 结果最相似的单词。

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

让我们使用加载的单词矢量来验证 “男-女” 类比。

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

以下是 “首都国” 类比：“北京”：“中国”። “东京”：“日本”。这演示了预训练的单词矢量中的语义。

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

对于 “形容词-超级形容词” 类比，如 “坏”：“最坏”። “大”：“最大”，我们可以看到，预训练的单词向量可能会捕获句法信息。

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

为了在预先训练的单词矢量中显示捕获的过去时概念，我们可以使用 “现在十-过去时” 类比来测试语法：“do”: “do”። “Go”: “走了”。

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 摘要

* 实际上，在大型语言上预训练的单词向量可以应用于下游自然语言处理任务。
* 预训练的词向量可以应用于单词相似性和类比任务。

## 练习

1. 使用 `TokenEmbedding('wiki.en')` 测试 FastText 结果。
1. 当词汇量极大时，我们怎样才能更快地找到类似的单词或完成单词类比？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
