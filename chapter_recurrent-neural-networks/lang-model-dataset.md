# 语言模型数据集（周杰伦专辑歌词）

本节我们将介绍如何预处理一个语言模型数据集，并转换成字符级循环神经网络需要的输入格式。为此，我们收集了周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词，并在后面章节里应用循环神经网络来训练一个语言模型。当模型训练好后，我们可以用这个模型来创作歌词。

## 读取数据集

首先导入本节所需的包和模块。

```{.python .input  n=1}
from mxnet import nd
import random
import zipfile
```

然后读取这个数据集，看看前50个字符是什么样的。

```{.python .input  n=20}
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars[0:40]
```

这个数据集有五万多个字符。为了打印方便，我们把换行符替换成空格。然后使用前一万个字符来训练模型，这样可以使得训练更快一些。

```{.python .input  n=14}
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
```

## 建立字符索引

首先我们将每个字符映射成一个从0开始的整数，或者叫做索引，来方便之后的处理。为了得到索引，我们将数据集里面所有不同的字符取出来，然后将其逐一映射到索引来构造词典，接着打印`vocab_size`，即词典中不同字符的个数。

```{.python .input  n=9}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size
```

之后将训练数据集中每个字符转成从索引，并打印前20个字符和其对应的索引。

```{.python .input  n=18}
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

我们将以上代码封装在`gluonbook`包里的`load_data_jay_lyrics`函数中以方便后面章节调用。调用该函数后会依次得到`corpus_indices`、`char_to_idx`、`idx_to_char`和`vocab_size`四个变量。

## 时序数据的采样

在训练中我们需要每次随机读取小批量样本和标签。这里不同的是时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符：“想”、“要”、“有”、“直”、“升”。且该样本的标签序列为这些字符分别在训练集中的下一个字符：“要”、“有”、“直”、“升”、“机”。我们有两种方式对时序数据采样，分别是随机采样和相邻采样。

### 随机采样

下面代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`指每个小批量的样本数，`num_steps`为每个样本所包含的时间步数。
在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

```{.python .input  n=25}
# 本函数已保存在 gluonbook 包中方便以后使用。
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为输出的索引是相应输入的索引加一。
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    # 返回从 pos 开始的长为 num_steps 的序列
    _data = lambda pos: corpus_indices[pos: pos + num_steps]
    for i in range(epoch_size):
        # 每次读取 batch_size 个随机样本。
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
```

让我们输入一个从0到29的人工序列，设批量大小和时间步数分别为2和6，打印随机采样每次读取的小批量样本的输入`X`和标签`Y`。可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

```{.python .input  n=31}
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

### 相邻采样

除了对原始序列做随机采样之外，我们还可以使相邻的两个随机小批量在原始序列上的位置相毗邻。这时候，我们就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量输入，并如此循环下去。这对实现循环神经网络造成了两方面影响。一方面，
在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态。
另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。
为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图分离出来。

```{.python .input  n=32}
# 本函数已保存在 gluonbook 包中方便以后使用。
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```

同样一样的设置下打印相邻采样每次读取的小批量样本的输入`X`和标签`Y`。相邻的两个随机小批量在原始序列上的位置相毗邻。

```{.python .input  n=33}
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

## 小结

* 时序数据采样方式包括随机采样和相邻采样。使用这两种方式的循环神经网络训练略有不同。

## 练习

* 你还能想到哪些采样小批量数据的办法？
* 如果我们想让一个序列样本就是一个完整的句子，这会给小批量采样带来什么样的问题？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7876)

![](../img/qr_lang-model-dataset.svg)
