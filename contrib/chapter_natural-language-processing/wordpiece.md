# 词片模型(Wordpiece)

之前章节介绍的模型使用的词汇表往往是定长的，但实际上大多数任务基本是开放的词汇表。如果使用定长词汇表会产生表外词汇（OOV,即词汇表未登录的词）的问题。解决这个问题的一个大类方法是使用sub-word units，即子词单元，如字符、单词字符的混合或者更加智能的分词方法。它会把稀有和未知词汇以子词单元序列来进行编码，更简单更有效。另外模型也可以从子词表达中学习到组词等能力。词片模型就是这类方法的典型代表，词片模型可以为任意可能的字符序列生成确定性分段。它可以在字符的灵活性以及单词的效率之间达到一种平衡。

给出原始文本和词片序列的示例：
原始文本：Jet makers feud over seat width with big orders at stake
词片序列： J et_ makers_ fe ud_ over_ seat _ width_ with_ big_ orders_ at_ stake

在这个示例中，单词Jet被分成两个单词“J“和“et\_“，单词feud被分成两个单词“fe“ 和 ”ud\_” 。其中“\_”是一个特殊字符，用于标记单词的结尾。

## 字节对编码(Byte-Pair Encoding)

词片模型的一种主要的实现方式叫做字节对编码（Byte-Pair Encoding， BPE）。这个算法采用贪心策略，具体来说是首先将词汇表初始化为符号表，并将每个单词表示为一个字符序列，并加入特殊字符”\_“结尾。然后计算所有的相邻符号对，并用新符号”AB“替换最频繁的（“A”，“B”），每次合并操作都会产生一个代表字符n-gram的新符号，频繁的字符n-garm（或者整个单词）可以合并为单个符号。在合并的过程中，不考虑跨越单词边界。最终符号词表大小=初始大小+合并操作次数。操作次数是算法唯一的超参数。

下面看一下算法的实现代码。

```{.python .input  n=2}
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            #统计每个相邻单元的出现次数
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    # v_out是使用'a b'替换为'ab'的新词汇表
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
         'n e w e s t </w>':6, 'w i d e s t </w>':3}
         
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    #找到出现次数最高的词对
    best = max(pairs, key=pairs.get)
    #合并词对
    vocab = merge_vocab(best, vocab)
    print("第%d次合并:" % (i + 1), best)
    
print(vocab)
```


如上面的结果，这个例子中，“lower”将被分割成“low”和“er_”。

我们可以通过字节对编码得到更加合适的词表，这个词表可能会出现一些不是单词的组合。由于英语自身的特点，比如在英语中的前缀和后缀。所以这本身也是有意义的一种形式，可以使模型有效处理近乎于无限的词汇。

## 小结
- 