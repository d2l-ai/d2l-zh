# 词片模型（WordPiece）

之前章节介绍的模型使用的词汇表往往是定长的，定长词汇表会产生一个问题，即会产生词汇表未登录的词，这称为表外词汇（out of vocabulary，OOV）问题。解决这个问题的一个大类方法是使用子词单元（sub-word units）。子词单元会把词汇表未登录的词以子词单元序列来进行表示。另外子词单元也可以从子词表达中学习到组词等能力。词片模型[1]就是子词单元这类方法的典型代表，词片模型可以为任意可能的字符序列生成确定性分段，即词片序列。词片模型可以在字符的灵活性以及单词的效率之间达到一种平衡。

下面来看一个原始文本和词片序列的示例：

原始文本：Jet makers feud over seat width with big orders at stake

词片序列： J et_ makers_ fe ud_ over_ seat _ width_ with_ big_ orders_ at_ stake_

在这个示例中，单词Jet被分成两个单词“J“和“et\_“，单词feud被分成两个单词“fe“ 和 ”ud\_” 。其中“\_”是一个特殊字符，用于标记单词的结尾。

## 字节对编码（Byte-Pair Encoding）

词片模型的一种主要的实现方式叫做字节对编码（Byte-Pair Encoding，BPE）算法[2]。这个算法采用贪心策略。

首先我们将词汇表初始化为符号表。其中特殊字符”\_“作为单词的结尾，"[UNK]"表示未知字符。

```{.python .input  n=1}
import collections

vocabs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
          'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]']
```

我们给出所有待处理单词及其在语料中出现频率计数。。然后，我们将每个单词表示为字符序列形式。

```{.python .input}
original_words = {'low_' : 5, 'lower_' : 2, 'newest_' : 6, 'widest_' : 3}
words = {}
for word, freq in original_words.items():
    new_word = ' '.join(list(word))
    words[new_word] = original_words[word]
```

然后计算所有的相邻符号对，并找到最频繁的符号（“A”，“B”）。

```{.python .input  n=2}
def get_max_freq_pair(words):
    pairs = collections.defaultdict(int)
    for word, freq in words.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # 统计每个相邻单元的出现次数
            pairs[symbols[i], symbols[i + 1]] += freq
    max_freq_pair = max(pairs, key = pairs.get)
    return max_freq_pair
```

并用新符号”AB“替换最频繁的符号（“A”，“B”），每次合并操作都会产生一个代表字符组合的新符号，频繁的字符组合（或者整个单词）可以合并为单个符号。在合并的过程中，不考虑跨越单词边界。

```{.python .input  n=3}
def merge_vocab(max_freq_pair, words, vocabs):
    bigram = ' '.join(max_freq_pair)
    vocabs.append(''.join(max_freq_pair))
    words_out = {}
    for word, freq in words.items():
        new_word = word.replace(bigram, ''.join(max_freq_pair))
        words_out[new_word] = words[word]
    return words_out
```

最终符号词表大小=初始大小+合并操作次数。操作次数是算法唯一的超参数。下面我们来运行一下这个算法。

```{.python .input  n=4}
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(words)
    words = merge_vocab(max_freq_pair, words, vocabs)
    print("Merge #%d:" % (i + 1), max_freq_pair)
```

然后我们打印原始单词和词片序列。

```{.python .input}
print("Words:", list(original_words.keys()))
print("Wordpieces:", list(words.keys()))
```

如上面这个例子中，“widest_”将被分割成“wi“、”d“和”est_“。

然后我们打印词表。

```{.python .input}
print("Vocabs:", vocabs)
```

可以看到之前合并的新符号都加入了词表。

我们可以通过字节对编码得到更加合适的词表，这个词表可能会出现一些不是单词的组合。而由于英语自身的特点，比如在英语中广泛存在的前缀和后缀。所以这些不是单词的组合本身也是有意义的一种形式，通过这些组合可以使模型有效处理近乎于无限的词汇。

## 应用过程
在上一步中，我们已经得到了词表。对于一个待处理的单词，我们设置两个下标start和end，初始分别指向单词的开始和结束位置。我们判断单词下标从start到end的子字符串是否存在于词表中，如果有则代表这个词是当前单词的一部分。然后将start指向end当前位置，将end继续指向单词结束位置。迭代这一个过程，直到单词被遍历完。当我们在遍历完这个单词后仍然有子字符串没有被替换，则将剩余子字符串替换为特殊词，如“[UNK]”。

下面我们来看一个例子，我们首先给定两个单词“slow”和“slowest”，然后使用上一步中得到的词表对这两个单词进行切分。

```{.python .input}
inputs = ['slow_', 'slowest_']
outputs = []
for word in inputs:
    start, end = 0, len(word)
    cur_output = []
    while start < len(word) and start < end:
        if word[start : end] in vocabs:
            cur_output.append(word[start : end])
            start = end
            end = len(word)
        else:
            end -= 1
    if start < len(word):
        cur_output.append('[UNK]')
    outputs.append(' '.join(cur_output))
print('Words:', inputs)
print('Wordpieces:', outputs)
```

## 小结
- 使用定长词汇表会产生词汇表未登录的词，这称为表外词汇问题。
- 可以使用子词单元来解决表外词汇问题，词片模型是子词单元方法的典型代表。
- 词片模型的一种主要的实现方式是字节对编码。

## 参考文献

[1] Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Klingner, J. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

[2] Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.
