# 文本预处理
:label:`sec_text_preprocessing`

对于序列数据处理问题，我们回顾和评估了使用的统计工具和预测时面临的挑战。这样的数据存在许多种形式，正如我们将在本书的许多章节中特别关注的那样，文本是序列数据最常见例子之一。
例如，一篇文章可以简单地看作是一个单词序列，甚至是一个字符序列。
为了将来在实验中使用序列数据的方便，我们将在本节中专门解释文本的常见预处理步骤。这些步骤通常包括：

1. 将文本作为字符串加载到内存中。
1. 将字符串拆分为标记（如，单词和字符）。
1. 建立一个词汇表，将拆分的标记映射到数字索引。
1. 将文本转换为数字索引序列，方便模型操作。

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## 读取数据集

首先，我们从 H.G.Well 的[时光机器](http://www.gutenberg.org/ebooks/35)中加载文本。这是一个相当小的语料库，只有30000多个单词，但足够实现我们的目标，即介绍文本预处理。现实中的文档集合可能会包含数十亿个单词。下面的函数将数据集读取到由多条文本行组成的列表中，其中每条文本行都是一个字符串。为简单起见，我们在这里忽略了标点符号和字母大写。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## 标记化

下面的 `tokenize` 函数将文本行列表作为输入，列表中的每个元素是一个文本序列（如，一条文本行）。每个文本序列又被拆分成一个标记列表，*标记*（token）是文本的基本单位。最后，返回一个由标记列表组成的列表，其中的每个标记都是一个字符串（string）。

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符标记。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知令牌类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## 词汇表

标记的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。现在，让我们构建一个字典，通常也叫做 *词汇表*（vocabulary），用来将字符串类型的标记映射到从 $0$ 开始的数字索引中。为此，我们先将训练集中的所有文档合并在一起，对它们的唯一标记进行统计，得到的统计结果称之为  *语料*（corpus），然后根据每个唯一标记的出现频率为其分配一个数字索引。很少出现的标记通常被移除，这可以降低复杂性。语料库中不存在或已删除的任何标记都将映射到一个特定的未知标记 “&lt;unk&gt;” 。我们可以选择增加一个列表，用于保存那些被保留的标记，例如：填充标记（“&lt;pad&gt;”）；序列开始标记（“&lt;bos&gt;”）；序列结束标记（“&lt;eos&gt;”）。

```{.python .input}
#@tab all
class Vocab:  #@save
    """文本词汇表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # 未知标记的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  #@save
    """统计标记的频率。"""
    # 这里的 `tokens` 是 1D 列表或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将标记列表展平成使用标记填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

我们使用时光机器数据集作为语料库来构建词汇表。然后，我们打印前几个高频标记及其索引。

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

现在，我们可以将每一条文本行转换成一个数字索引列表。

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## 整合所有功能

在使用上述函数时，我们将所有功能打包到 `load_corpus_time_machine` 函数中，该函数返回 `corpus`（标记索引列表）和 `vocab`（时光机器语料库的词汇表）。我们在这里所做的改变是：
- 1、为了简化后面章节中的训练，我们使用字符而不是单词实现文本标记化；
- 2、`corpus` 是单个列表，而不是使用标记列表构成的一个列表，因为时光机器数据集中的每个文本行不一定是一个句子或一个段落。

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的标记索引列表和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每一个文本行，不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## 小结

* 文本是序列数据的一种重要形式。
* 为了对文本进行预处理，我们通常将文本拆分为标记，构建词汇表将标记字符串映射为数字索引，并将文本数据转换为标记索引以供模型操作。

## 练习

1. 标记化是一个关键的预处理步骤，它因语言而异，尝试找到另外三种常用的标记化文本的方法。
1. 在本节的实验中，将文本标记化为单词和更改 `Vocab` 实例的 `min_freq` 参数。这对词汇量有何影响？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2093)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2095)
:end_tab: