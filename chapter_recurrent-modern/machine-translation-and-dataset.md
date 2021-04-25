# 机器翻译与数据集
:label:`sec_machine_translation`

我们已经使用循环神经网络来设计语言模型，这是自然语言处理的关键。另一个旗舰基准测试是“机器翻译”，这是将输入序列转换成输出序列的序列转换模型的的核心问题。序列转换模型在各种现代人工智能应用中发挥着至关重要的作用，将成为本章剩余部分和 :numref:`chap_attention` 的重点。为此，本节介绍机器翻译问题及其稍后将使用的数据集。

*机器翻译*指的是将序列从一种语言自动翻译成另一种语言。事实上，这个领域可能可以追溯到数字计算机发明后不久的20世纪40年代，在第二次世界大战中就使用计算机破解语言编码。几十年来，在使用神经网络进行端到端学习的兴起之前，统计学方法在这一领域一直占据主导地位 :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990` 。基于神经网络的方法通常被称为*神经机器翻译*从而将自己与*统计机器翻译*区分开。
这涉及翻译模型和语言模型等组成部分的统计分析。

这本书强调端到端的学习，将重点放在神经机器翻译方法上。与 :numref:`sec_language_model` 中的语言模型问题（语料库是单一语言的）不同，机器翻译数据集是由源语言和目标语言的文本序列对组成的。因此，我们需要一种不同的方法来预处理机器翻译数据集，而不是复用语言模型的预处理程序。在下面，我们将展示如何将预处理后的数据加载到小批量中进行训练。

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## 下载和预处理数据集

首先，我们下载一个由[Tatoeba项目的双语句子对](http://www.manythings.org/anki/)组成的英-法数据集。数据集中的每一行都是一对制表符分隔的英文文本序列和翻译后的法语文本序列。请注意，每个文本序列可以是一个句子，也可以是包含多个句子的一段。在这个将英语翻译成法语的机器翻译问题中，英语是“源语言”（source language），法语是“目标语言”（target language）。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

下载数据集后，我们对原始文本数据进行几个预处理步骤。例如，我们用单个空格代替连续多个空格，将大写字母转换为小写字母，并在单词和标点符号之间插入空格。

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## 标记化

与 :numref:`sec_language_model` 中的字符级标记化不同，对于机器翻译，我们更喜欢词级标记化（最先进的模型可能使用更高级的标记化技术）。下面的`tokenize_nmt`函数对前`num_examples`个文本序列对进行标记，其中每个标记要么是一个单词，要么是一个标点符号。此函数返回两个标记列表：`source`和`target`。具体地说，`source[i]`是源语言（这里是英语）第$i$个文本序列的标记列表，`target[i]`是目标语言（这里是法语）的标记。

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

让我们绘制每个文本序列的标记数量的直方图。在这个简单的英法数据集中，大多数文本序列的标记少于20个。

```{.python .input}
#@tab all
d2l.set_figsize()
_, _, patches = d2l.plt.hist(
    [[len(l) for l in source], [len(l) for l in target]],
    label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right');
```

## 词表

由于机器翻译数据集由语言对组成，因此我们可以分别为源语言和目标语言构建两个词表。使用词级标记化时，词汇量将明显大于使用字符级标记化时的词汇量。为了缓解这一问题，这里我们将出现次数少于2次的低标记牌视为相同的未知（“&lt;unk&gt;”）令牌。除此之外，我们还指定了额外的特殊标记，例如用于小批量时填充相同长度的序列（“&lt;pad&gt;”），以及序列的开始标记（“&lt;bos&gt;”）和结束标记（“&lt;eos&gt;”）。这样的特殊标记在自然语言处理任务中比较常用。

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## 加载数据集
:label:`subsec_mt_data_loading`

回想一下，在语言模型中，每个序列样本，一个句子的一段或多个句子的跨度，都有一个固定的长度。都有固定的长度。这是由 :numref:`sec_language_model` 中的`num_steps`（时间步数或标记数）参数指定的。在机器翻译中，每个样本都是一对源和目标文本序列，其中每个文本序列可以具有不同的长度。

为了提高计算效率，我们仍然可以通过*截断*和*填充*一次处理一小批量文本序列。假设同一小批量中的每个序列应该具有相同的长度`num_steps`。如果文本序列的标记少于`num_steps`个，我们将继续在其末尾附加特殊的“&lt;pad&gt;”标记，直到其长度达到`num_steps`。否则，我们将截断文本序列，只取其前`num_steps`个令牌，并丢弃其余的标记。这样，每个文本序列将具有相同的长度，以便以相同形状的小批量加载。

以下`truncate_pad`函数如前所述截断或填充文本序列。

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

现在我们定义一个函数，将文本序列转换成小批量进行训练。我们将特殊的“&lt;eos&gt;”标记附加到每个序列的末尾，以指示序列的结束。当模型通过一个接一个地生成序列令牌进行预测时，“&lt;eos&gt;”令牌的生成可以暗示输出序列是完整的。此外，我们还记录了不包括填充标记的每个文本序列的长度。我们稍后将介绍的一些模型将需要此信息。

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量。"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## 训练模型

最后，我们定义`load_data_nmt`函数来返回数据迭代器，以及源语言和目标语言的词汇表。

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词汇表。"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

让我们读出英语-法语数据集中的第一个小批量数据。

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## 小结

* 机器翻译是指将文本序列从一种语言自动翻译成另一种语言。
* 使用词级标记化时的词汇量，将明显大于使用字符级标记化时的词汇量。为了缓解这一问题，我们可以将低频标记视为相同的未知标记。
* 我们可以截断和填充文本序列，以便所有文本序列都具有相同的长度，以便以小批量方式加载。

## 练习

1. 在`load_data_nmt`函数中尝试`num_examples`参数的不同值。这对源语言和目标语言的词汇量有何影响？
1. 某些语言（例如中文和日语）的文本没有单词边界指示符（例如，空格）。对于这种情况，词级标记化仍然是个好主意吗？为什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
