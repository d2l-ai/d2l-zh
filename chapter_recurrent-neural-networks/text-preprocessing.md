# 文本预处理
:label:`sec_text_preprocessing`

我们回顾和评估了序列数据的统计工具和预测挑战。这些数据可以有多种形式。具体来说，正如我们将在本书的许多章节中重点介绍的那样，文本是序列数据最常见例子。例如，一篇文章可以简单地看作是一个单词序列，甚至是一个字符序列。为了方便我们将来对序列数据的实验，我们将在本节中专门解释文本的常见预处理步骤。通常，这些步骤包括：

1. 将文本作为字符串加载到内存中。
1. 将字符串拆分为标记（如，单词和字符）。
1. 建立一个词汇表，将拆分的标记映射到数字索引。
1. 将文本转换为数字索引序列，以便模型可以轻松地对其进行操作。

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

为了开始，我们从H.G.Well的[*时光机器*](http://www.gutenberg.org/ebooks/35)中加载文本。这是一个相当小的语料库，只有30000多个单词，但对于我们想要说明的目标来说，这足够了。现实中的文档集合可能会包含数十亿个单词。下面的函数将数据集读取到文本行组成的列表中，其中每行都是一个字符串。为简单起见，这里我们忽略标点符号和大写。

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

以下 `tokenize` 函数将列表作为输入，列表中的每个元素是文本序列（如，文本行）。每个文本序列被拆分成一个标记列表。*标记*（token）是文本的基本单位。最后返回一个标记列表，其中每个标记都是一个字符串（string）。

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

## 词汇

标记的字符串类型不方便模型使用，因为模型需要输入数字。现在，让我们构建一个字典，通常也叫做*词表*（Vocabulary）来将字符串标记映射到从0开始的数字索引中。为此，我们首先统计训练集中所有文档中的唯一标记，即*语料*（corpus），然后根据每个唯一标记的出现频率为其分配一个数字索引。很少出现的标记通常被移除，这可以降低复杂性。语料库中不存在或已删除的任何标记都将映射到一个特殊的未知标记 “&lt;unk&gt;” 。我们可以选择添加保留令牌的列表，例如“&lt;pad&gt;”表示填充；“&lt;bos&gt;”表示序列的开始；“&lt;eos&gt;”表示序列的结束。

```{.python .input}
#@tab all
class Vocab:  #@save
    """文本词表"""
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
    """Count token frequencies."""
    # 这里的 `tokens` 是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将令牌列表展平
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

我们使用时光机器数据集作为语料库来构建词汇表。然后，我们打印前几个常见标记及其索引。

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

现在我们可以将每一行文本转换成一个数字索引列表。

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## 把所有的东西放在一起

使用上述函数，我们将所有内容打包到 `load_corpus_time_machine` 函数中，该函数返回 `corpus`（标记索引列表）和 `vocab`（时光机器语料库的词汇表）。我们在这里所做的修改是：
- 1、我们将文本 标记化为字符，而不是单词，以简化后面部分中的训练；
- 2、`corpus`是单个列表，而不是标记列表嵌套，因为时光机器数据集中的每个文本行不一定是句子或段落。

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的令牌索引和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每一个文本行不一定是一个句子或段落，
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

1. 标记化是一个关键的预处理步骤。它因语言而异。尝试找到另外三种常用的文本标记方法。
1. 在本节的实验中，将文本标记为单词，并更改 `Vocab` 实例的 `min_freq` 参数。这对词汇量有何影响？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2093)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2095)
:end_tab: