# 子词嵌入
:label:`sec_fasttext`

在英语中，“帮助”、“帮助” 和 “帮助” 等单词都是同一个词 “帮助” 的变形形式。“狗” 和 “狗” 之间的关系与 “猫” 和 “猫” 之间的关系相同，“男孩” 和 “男朋友” 之间的关系与 “女孩” 和 “女朋友” 之间的关系相同。在法语和西班牙语等其他语言中，许多动词有 40 多种变形形式，而在芬兰语中，名词最多可能有 15 种情况。在语言学中，形态学研究单词形成和词汇关系。但是，在 word2vec 和 Glove 中都没有探索单词的内部结构。 

## FastText 模型

回想一下 Word2vec 中单词是如何表示的。在跳过图模型和连续字包模型中，同一个单词的不同变形形式直接由不同的矢量表示，没有共享参数。为了使用形态学信息，*FastText* 模型提出了一种 * 子词嵌入 * 方法，其中子字是字符 $n$ 克 :cite:`Bojanowski.Grave.Joulin.ea.2017`。FastText 不是学习单词级矢量表示形式，而是可以将 FastText 视为副词级跳过图，其中每个 * 中心单词 * 由其子词矢量的总和表示。 

让我们说明如何使用 “在哪里” 一词为 FastText 中的每个中心单词获取子词。首先，<” and “> 在单词的开头和结尾添加特殊字符 “”，以区分其他子词的前缀和后缀。然后，从单词中提取字符 $n$ 克。例如，当 $n=3$ 时，我们获得长度为 3 的所有子词：“<wh”, “whe”, “her”, “ere”, “re>” 和特殊的子词 “<where>”。 

在 FastText 中，对于任何一个单词 $w$，用 $\mathcal{G}_w$ 表示其长度介于 3 到 6 之间的所有子词及其特殊子词的并集。词汇是所有单词的子词的结合。让 $\mathbf{z}_g$ 成为字典中子词 $g$ 的矢量，而字 $w$ 的矢量 $w$ 作为跳过图模型中的中心词是其子词矢量的总和： 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

FastText 的其余部分与跳过图模型相同。与跳过图模型相比，FastText 中的词汇量更大，导致更多的模型参数。此外，为了计算单词的表示形式，必须将其所有子词向量求和，从而导致更高的计算复杂性。但是，由于结构相似的单词之间的子词共享参数，稀有单词甚至是词汇不足的单词可以在 FastText 中获得更好的矢量表示形式。 

## 字节对编码
:label:`subsec_Byte_Pair_Encoding`

在 FastText 中，所有提取的子词必须是指定的长度，例如 $3$ 到 $6$，因此不能预定义词汇大小。为了允许在固定大小的词汇中使用可变长度的子词，我们可以应用名为 * 字节对编码 * (BPE) 的压缩算法来提取子词 :cite:`Sennrich.Haddow.Birch.2015`。 

字节对编码对训练数据集执行统计分析，以发现单词中的常见符号，例如任意长度的连续字符。从长度为 1 的符号开始，字节对编码以迭代方式合并最常用的一对连续符号，以生成新的更长的符号。请注意，为了提高效率，不考虑跨越词界的货币对。最后，我们可以使用这样的符号作为子词来对单词进行分段。字节对编码及其变体已用于流行的自然语言处理预训练模型中的输入表示，例如 GPT-2 :cite:`Radford.Wu.Child.ea.2019` 和 Roberta :cite:`Liu.Ott.Goyal.ea.2019`。在下面，我们将说明字节对编码的工作原理。 

首先，我们将符号的词汇初始化为所有英文小写字符、一个特殊的词尾符号 `'_'` 和一个特殊的未知符号 `'[UNK]'`。

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

由于我们不考虑跨越单词边界的符号对，因此我们只需要将字典 `raw_token_freqs` 映射到数据集中的频率（出现次数）。请注意，每个单词附加了特殊符号 `'_'`，以便我们可以轻松地从一系列输出符号（例如 “a_ tall er_ man”）中恢复一个单词序列（例如，“更高的人”）。由于我们从只包含单个字符和特殊符号的词汇开始合并过程，因此每个单词中的每对连续字符（字典 `token_freqs` 的键）之间都会插入空格。换句话说，空格是单词中符号之间的分隔符。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

我们定义了以下 `get_max_freq_pair` 函数，该函数返回单词中最常见的一对连续符号，其中单词来自输入字典 `token_freqs` 的键。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

作为基于连续符号频率的贪婪方法，字节对编码将使用以下 `merge_symbols` 函数合并最常见的连续符号对以生成新的符号。

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

现在我们对字典 `token_freqs` 的密钥迭代执行字节对编码算法。在第一次迭代中，最常见的连续符号对是 `'t'` 和 `'a'`，因此字节对编码将它们合并以生成一个新的符号 `'ta'`。在第二次迭代中，字节对编码继续合并 `'ta'` 和 `'l'`，从而产生另一个新的符号 `'tal'`。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

在对字节对编码进行了 10 次迭代之后，我们可以看到列表 `symbols` 现在包含了另外 10 个与其他符号迭代合并的符号。

```{.python .input}
#@tab all
print(symbols)
```

对于字典 `raw_token_freqs` 键中指定的同一数据集，由于字节对编码算法的结果，数据集中的每个单词现在都被子词 “fast_”、“fast”、“er_”、“tall_” 和 “tall” 分割。例如，单词 “faster_” 和 “taller_” 分别分为 “快速 er_” 和 “高尔 _”。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

请注意，字节对编码的结果取决于正在使用的数据集。我们还可以使用从一个数据集中学到的子词来对另一个数据集的单词进行分段。作为一种贪婪的方法，以下 `segment_BPE` 函数试图将输入参数 `symbols` 中的单词分成尽可能长的子词。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
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

在下面，我们使用从上述数据集中学习的列表 `symbols` 中的子词对表示另一个数据集的 `tokens` 进行细分。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 摘要

* FastText 模型提出了一种子词嵌入方法。基于 word2vec 中的跳过图模型，它表示一个中心词作为其子词矢量的总和。
* 字节对编码对训练数据集执行统计分析，以发现单词中的常见符号。作为一种贪婪的方法，字节对编码以迭代方式合并最常见的一对连续符号。
* 子词嵌入可能会提高稀有单词和字典外单词的表示质量。

## 练习

1. 例如，英语中约有 $3\times 10^8$ 克可能有 $6$ 克。当子词太多时，问题是什么？如何解决这个问题？Hint: refer to the end of Section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`。
1. 如何基于连续词袋模型设计子词嵌入模型？
1. 要获得大小为 $m$ 的词汇，当初符号词汇量大小为 $n$ 时，需要多少合并操作？
1. 如何扩展字节对编码的想法来提取短语？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:
