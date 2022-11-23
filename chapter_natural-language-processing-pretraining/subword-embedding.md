# 子词嵌入
:label:`sec_fasttext`

在英语中，“helps”“helped”和“helping”等单词都是同一个词“help”的变形形式。“dog”和“dogs”之间的关系与“cat”和“cats”之间的关系相同，“boy”和“boyfriend”之间的关系与“girl”和“girlfriend”之间的关系相同。在法语和西班牙语等其他语言中，许多动词有40多种变形形式，而在芬兰语中，名词最多可能有15种变形。在语言学中，形态学研究单词形成和词汇关系。但是，word2vec和GloVe都没有对词的内部结构进行探讨。

## fastText模型

回想一下词在word2vec中是如何表示的。在跳元模型和连续词袋模型中，同一词的不同变形形式直接由不同的向量表示，不需要共享参数。为了使用形态信息，*fastText模型*提出了一种*子词嵌入*方法，其中子词是一个字符$n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017`。fastText可以被认为是子词级跳元模型，而非学习词级向量表示，其中每个*中心词*由其子词级向量之和表示。

让我们来说明如何以单词“where”为例获得fastText中每个中心词的子词。首先，在词的开头和末尾添加特殊字符“&lt;”和“&gt;”，以将前缀和后缀与其他子词区分开来。
然后，从词中提取字符$n$-gram。
例如，值$n=3$时，我们将获得长度为3的所有子词：
“&lt;wh”“whe”“her”“ere”“re&gt;”和特殊子词“&lt;where&gt;”。

在fastText中，对于任意词$w$，用$\mathcal{G}_w$表示其长度在3和6之间的所有子词与其特殊子词的并集。词表是所有词的子词的集合。假设$\mathbf{z}_g$是词典中的子词$g$的向量，则跳元模型中作为中心词的词$w$的向量$\mathbf{v}_w$是其子词向量的和：

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText的其余部分与跳元模型相同。与跳元模型相比，fastText的词量更大，模型参数也更多。此外，为了计算一个词的表示，它的所有子词向量都必须求和，这导致了更高的计算复杂度。然而，由于具有相似结构的词之间共享来自子词的参数，罕见词甚至词表外的词在fastText中可能获得更好的向量表示。

## 字节对编码（Byte Pair Encoding）
:label:`subsec_Byte_Pair_Encoding`

在fastText中，所有提取的子词都必须是指定的长度，例如$3$到$6$，因此词表大小不能预定义。为了在固定大小的词表中允许可变长度的子词，我们可以应用一种称为*字节对编码*（Byte Pair Encoding，BPE）的压缩算法来提取子词 :cite:`Sennrich.Haddow.Birch.2015`。

字节对编码执行训练数据集的统计分析，以发现单词内的公共符号，诸如任意长度的连续字符。从长度为1的符号开始，字节对编码迭代地合并最频繁的连续符号对以产生新的更长的符号。请注意，为提高效率，不考虑跨越单词边界的对。最后，我们可以使用像子词这样的符号来切分单词。字节对编码及其变体已经用于诸如GPT-2 :cite:`Radford.Wu.Child.ea.2019`和RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`等自然语言处理预训练模型中的输入表示。在下面，我们将说明字节对编码是如何工作的。

首先，我们将符号词表初始化为所有英文小写字符、特殊的词尾符号`'_'`和特殊的未知符号`'[UNK]'`。

```{.python .input}
#@tab mxnet, pytorch
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
           
```

```{.python .input}
#@tab paddle
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
           
```

因为我们不考虑跨越词边界的符号对，所以我们只需要一个字典`raw_token_freqs`将词映射到数据集中的频率（出现次数）。注意，特殊符号`'_'`被附加到每个词的尾部，以便我们可以容易地从输出符号序列（例如，“a_all er_man”）恢复单词序列（例如，“a_all er_man”）。由于我们仅从单个字符和特殊符号的词开始合并处理，所以在每个词（词典`token_freqs`的键）内的每对连续字符之间插入空格。换句话说，空格是词中符号之间的分隔符。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

我们定义以下`get_max_freq_pair`函数，其返回词内最频繁的连续符号对，其中词来自输入词典`token_freqs`的键。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # “pairs”的键是两个连续符号的元组
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # 具有最大值的“pairs”键
```

作为基于连续符号频率的贪心方法，字节对编码将使用以下`merge_symbols`函数来合并最频繁的连续符号对以产生新符号。

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

现在，我们对词典`token_freqs`的键迭代地执行字节对编码算法。在第一次迭代中，最频繁的连续符号对是`'t'`和`'a'`，因此字节对编码将它们合并以产生新符号`'ta'`。在第二次迭代中，字节对编码继续合并`'ta'`和`'l'`以产生另一个新符号`'tal'`。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'合并# {i+1}:',max_freq_pair)
```

在字节对编码的10次迭代之后，我们可以看到列表`symbols`现在又包含10个从其他符号迭代合并而来的符号。

```{.python .input}
#@tab all
print(symbols)
```

对于在词典`raw_token_freqs`的键中指定的同一数据集，作为字节对编码算法的结果，数据集中的每个词现在被子词“fast_”“fast”“er_”“tall_”和“tall”分割。例如，单词“fast er_”和“tall er_”分别被分割为“fast er_”和“tall er_”。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

请注意，字节对编码的结果取决于正在使用的数据集。我们还可以使用从一个数据集学习的子词来切分另一个数据集的单词。作为一种贪心方法，下面的`segment_BPE`函数尝试将单词从输入参数`symbols`分成可能最长的子词。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 具有符号中可能最长子字的词元段
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

我们使用列表`symbols`中的子词（从前面提到的数据集学习）来表示另一个数据集的`tokens`。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 小结

* fastText模型提出了一种子词嵌入方法：基于word2vec中的跳元模型，它将中心词表示为其子词向量之和。
* 字节对编码执行训练数据集的统计分析，以发现词内的公共符号。作为一种贪心方法，字节对编码迭代地合并最频繁的连续符号对。
* 子词嵌入可以提高稀有词和词典外词的表示质量。

## 练习

1. 例如，英语中大约有$3\times 10^8$种可能的$6$-元组。子词太多会有什么问题呢？如何解决这个问题？提示:请参阅fastText论文第3.2节末尾 :cite:`Bojanowski.Grave.Joulin.ea.2017`。
1. 如何在连续词袋模型的基础上设计一个子词嵌入模型？
1. 要获得大小为$m$的词表，当初始符号词表大小为$n$时，需要多少合并操作？
1. 如何扩展字节对编码的思想来提取短语？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5747)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5748)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11818)
:end_tab:
