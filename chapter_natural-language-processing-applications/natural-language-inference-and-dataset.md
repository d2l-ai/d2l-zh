# 自然语言推理和数据集
:label:`sec_natural-language-inference-and-dataset`

在 :numref:`sec_sentiment` 中，我们讨论了情绪分析的问题。此任务旨在将单个文本序列分为预定义的类别，例如一组情绪极性。但是，当需要决定是否可以从另一句话推断出另一句话，还是通过确定语义上等同的句子来消除冗余时，知道如何对一个文本序列进行分类是不够的。相反，我们需要能够对文本序列成对进行推理。 

## 自然语言推理

*自然语言推理*研究是否有*假设*
可以从*前提*推断出来，其中两者都是文本序列。换句话说，自然语言推理决定了一对文本序列之间的逻辑关系。这种关系通常分为三种类型： 

* *蕴含（entailment）*：假设可以从前提中推断出来。
* *矛盾（contradiction）*：可以从前提中推断假设的否定。
* *中立（neutral）*：所有其他情况。

自然语言推理也被称为识别文本蕴含任务。例如，下面的词元对将被标记为*蕴含（entailment）*，因为假设中的 “显示亲情” 可以从前提中的 “相互拥抱” 推断出来。 

> 前提：两个女人互相拥抱。 

> 假设：两个女人表现出亲情。 

以下是 *矛盾* 的例子，因为 “运行编码示例” 表示 “没有睡觉” 而不是 “睡觉”。 

> 前提：一个男人正在运行从 “潜入深度学习” 中的编码示例。 

> 假设：那个男人在睡觉。 

第三个例子显示了 * 中立性 * 关系，因为 “正在为我们表演” 这一事实既不能推断 “著名的” 也不是 “不出名”。  

> 前提：音乐家们正在为我们表演。 

> 假设：音乐家很有名。 

自然语言推理一直是理解自然语言的核心话题。它拥有广泛的应用程序，从信息检索到开放域问答。为了研究这个问题，我们将首先研究一个流行的自然语言推理基准数据集。 

## 斯坦福大学自然语言推理(SNLI)数据集

[**斯坦福大学自然语言推理(SNLI)语料库**]是超过 50 万个标记为英语句子对 :cite:`Bowman.Angeli.Potts.ea.2015` 的集合。我们将提取的 SNLI 数据集下载并存储在路径 `../data/snli_1.0` 中。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**阅读数据集**]

原始 SNLI 数据集包含的信息比我们实验中真正需要的信息丰富得多。因此，我们定义了一个函数 `read_snli` 来仅提取部分数据集，然后返回前提、假设及其标签的列表。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

现在让我们[**打印前3对前提和假设**]，以及它们的标签（“0”、“1” 和 “2” 分别对应于 “蕴含”、“矛盾” 和 “中性”）。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

该训练套装有大约 550,000 对，测试套装有约 10000 对。以下表明，在培训套装和测试套装中，[**“蕴含（entailment）”、“矛盾（contradiction）”和“中性（neutral）”三个标签都是平衡的**]。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**定义用于加载数据集的类**]

下面我们定义一个类，通过从 Gluon 中的 `Dataset` 类继承来加载 SNLI 数据集。类构造函数中的参数 `num_steps` 指定了文本序列的长度，以便序列的每个小块都具有相同的形状。换句话说，在较长顺序中的第一个 `num_steps` 个代币之后的令牌被修剪，而特殊令牌 “<pad>” 将附加到较短的序列，直到它们的长度变为 `num_steps`。通过实现 `__getitem__` 函数，我们可以使用指数 `idx` 任意访问前提、假设和标签。

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**统筹合一**]

现在我们可以调用 `read_snli` 函数和 `SNLIDataset` 类来下载 SNLI 数据集，并返回 `DataLoader` 实例用于训练和测试集合，以及训练集的词汇表。值得注意的是，我们必须使用从训练集中构建的词汇和测试集合的词汇。因此，在训练集中训练的模型将不知道测试集中的任何新令牌。

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

在这里，我们将批处理大小设置为 128，序列长度为 50，然后调用 `load_data_snli` 函数来获取数据迭代器和词汇。然后我们打印词汇量大小。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

现在我们打印第一个迷你手表的形状。与情绪分析相反，我们有两个输入 `X[0]` 和 `X[1]`，代表对前提和假设。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 摘要

* 自然语言推理研究是否可以从前提中推断假设，两者都是文本序列。
* 在自然语言推断中，前提和假设之间的关系包括内涵、矛盾和中立。
* 斯坦福自然语言推理 (SNLI) 语料库是一个受欢迎的自然语言推断基准数据集。

## 练习

1. 长期以来，机器翻译的评估基于输出翻译和基础真实翻译之间的肤浅 $n$ 克匹配。你能否通过自然语言推断来设计评估机器翻译结果的衡量标准？
1. 我们如何更改超参数来减少词汇量？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:
