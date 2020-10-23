# 机器翻译和数据集
:label:`sec_machine_translation`

我们使用 RNS 来设计语言模型，这是自然语言处理的关键。另一个旗舰基准是 * 机器翻译 *，这是将输入序列转换为输出序列的 * 序列转换 * 模型的核心问题领域。序列转导模型在各种现代人工智能应用中扮演着至关重要的角色，将成为本章剩余部分和 :numref:`chap_attention` 的重点。为此，本节介绍了机器翻译问题及其将在稍后使用的数据集。

*机器翻译 * 是指
自动翻译序列从一种语言到另一种语言。事实上，这个领域可能可以追溯到 20 世纪 40 年代数字计算机发明后不久，尤其是考虑在二战中使用计算机破解语言代码。几十年来，在使用神经网络的端到端学习兴起之前，统计方法在这一领域占主导地位 :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`。后者通常被称为
*神经机器翻译 *
区分自己与
*统计机器翻译 *
，包括对翻译模型和语言模型等组件进行统计分析。

本书强调端到端学习，重点介绍神经机器翻译方法。与我们在 :numref:`sec_language_model` 中的语言模型问题（语料库为单一语言）不同，机器翻译数据集分别由源语言和目标语言中的文本序列对组成。因此，我们需要一种不同的方法来预处理机器翻译数据集，而不是重复使用预处理语言建模的预处理例程。在下面，我们展示了如何将预处理的数据加载到微型批次中进行训练。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
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

首先，我们下载一个由 [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/) 组成的英法数据集。数据集中的每一行都是一对英文文本序列和翻译后的法文文本序列的制表符分隔对。请注意，每个文本序列只能是一个句子或多个句子的段落。在此机器翻译问题中，英语被翻译成法语，英语是 * 源语言 *，法语是 * 目标语言 *。

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

下载数据集后，我们将继续对原始文本数据执行几个预处理步骤。例如，我们将非中断空格替换为空格，将大写字母转换为小写字母，并在单词和标点符号之间插入空格。

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

与 :numref:`sec_language_model` 中的字符级标记不同，对于机器翻译，我们更喜欢在这里使用字级标记化（最先进的模型可能使用更先进的标记化技术）。以下 `tokenize_nmt` 函数标记了第一个 `num_examples` 文本序列对，其中每个标记都是单词或标点符号。此函数返回两个令牌列表：`source` 和 `target`。具体来说，`source[i]` 是源语言中的 $i^\mathrm{th}$ 文本序列中的标记列表（此处为英语），`target[i]` 是目标语言（此处是法语）。

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

让我们绘制每个文本序列的标记数量的直方图。在这个简单的英语-法语数据集中，大多数文本序列具有少于 20 个标记。

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

## 词汇

由于机器翻译数据集由一对语言组成，因此我们可以分别为源语言和目标语言构建两个词汇。使用字级标记化，词汇大小将显著大于使用字符级标记化的大小。为了缓解这一点，在这里我们将出现小于 2 倍的不经常令牌视为相同的未知（” <unk> “）令牌。除此之外，我们还指定额外的特殊标记，例如用于 <pad> 小批次中相同长度的填充（” “）序列，以及标记 <bos> <eos> 序列的开始（” “）或结束（” “）。这些特殊令牌通常用于自然语言处理任务。

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## 加载数据集
:label:`subsec_mt_data_loading`

回想一下，在语言建模中，每个序列示例中，一个句子的段落或跨越多个句子的跨度都具有固定的长度。这是由 `num_steps`（时间步长或令牌数）参数在 :numref:`sec_language_model` 中指定的。在机器翻译中，每个示例都是一对源文本序列和目标文本序列，其中每个文本序列可能具有不同的长度。

为了提高计算效率，我们仍然可以通过 * 截断 * 和 * 填充 * 同时处理小批文本序列。假设同一小批次中的每个序列都应具有相同的长度 `num_steps`。如果文本序列的标记少于 `num_steps`，我们将继续将特殊 “<pad>” 标记附加到其末尾，直到其长度达到 `num_steps`。否则，我们将通过仅采用第一个 `num_steps` 标记并丢弃其余的标记来截断文本序列。通过这种方式，每个文本序列将具有相同的长度，以相同形状的微型批次加载。

以下 `truncate_pad` 函数会截断或填充文本序列，如前所述。

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

现在我们定义了一个函数，将文本序列转换为小批次进行训练。我们将特殊的 “<eos>” 标记附加到每个序列的末尾，以指示序列的结尾。当模型通过在标记后生成序列标记进行预测时，“<eos>” 标记的生成可以建议输出序列已完成。此外，我们还记录每个文本序列的长度，不包括填充标记。我们稍后将介绍的某些模型将需要这些信息。

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## 把所有东西放在一起

最后，我们定义了 `load_data_nmt` 函数来返回数据迭代器，以及源语言和目标语言的词汇。

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
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

让我们从英法数据集中读取第一个小批次。

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

## 摘要

* 机器翻译是指序列从一种语言自动翻译到另一种语言。
* 使用字级标记化，词汇大小将显著大于使用字符级标记化的大小。为了缓解这一点，我们可以将不经常的令牌视为相同的未知令牌。
* 我们可以截断和填充文本序列，以便所有这些序列都将具有相同的长度以小批量加载。

## 练习

1. 在 `load_data_nmt` 函数中尝试使用 `num_examples` 参数的不同值。这对源语言和目标语言的词汇量有何影响？
1. 某些语言（如中文和日文）的文本没有单词边界指示符（例如空格）。字级标记化对于这种情况是否仍然是一个好主意？为什么还是为什么不呢？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
