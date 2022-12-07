# 自然语言推断与数据集
:label:`sec_natural-language-inference-and-dataset`

在 :numref:`sec_sentiment`中，我们讨论了情感分析问题。这个任务的目的是将单个文本序列分类到预定义的类别中，例如一组情感极性中。然而，当需要决定一个句子是否可以从另一个句子推断出来，或者需要通过识别语义等价的句子来消除句子间冗余时，知道如何对一个文本序列进行分类是不够的。相反，我们需要能够对成对的文本序列进行推断。

## 自然语言推断

*自然语言推断*（natural language inference）主要研究
*假设*（hypothesis）是否可以从*前提*（premise）中推断出来，
其中两者都是文本序列。
换言之，自然语言推断决定了一对文本序列之间的逻辑关系。这类关系通常分为三种类型：

* *蕴涵*（entailment）：假设可以从前提中推断出来。
* *矛盾*（contradiction）：假设的否定可以从前提中推断出来。
* *中性*（neutral）：所有其他情况。

自然语言推断也被称为识别文本蕴涵任务。
例如，下面的一个文本对将被贴上“蕴涵”的标签，因为假设中的“表白”可以从前提中的“拥抱”中推断出来。

>前提：两个女人拥抱在一起。

>假设：两个女人在示爱。

下面是一个“矛盾”的例子，因为“运行编码示例”表示“不睡觉”，而不是“睡觉”。

>前提：一名男子正在运行Dive Into Deep Learning的编码示例。

>假设：该男子正在睡觉。

第三个例子显示了一种“中性”关系，因为“正在为我们表演”这一事实无法推断出“出名”或“不出名”。

>前提：音乐家们正在为我们表演。

>假设：音乐家很有名。

自然语言推断一直是理解自然语言的中心话题。它有着广泛的应用，从信息检索到开放领域的问答。为了研究这个问题，我们将首先研究一个流行的自然语言推断基准数据集。

## 斯坦福自然语言推断（SNLI）数据集

[**斯坦福自然语言推断语料库（Stanford Natural Language Inference，SNLI）**]是由500000多个带标签的英语句子对组成的集合 :cite:`Bowman.Angeli.Potts.ea.2015`。我们在路径`../data/snli_1.0`中下载并存储提取的SNLI数据集。

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

```{.python .input}
#@tab paddle
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**读取数据集**]

原始的SNLI数据集包含的信息比我们在实验中真正需要的信息丰富得多。因此，我们定义函数`read_snli`以仅提取数据集的一部分，然后返回前提、假设及其标签的列表。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签"""
    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

现在让我们[**打印前3对**]前提和假设，以及它们的标签（“0”“1”和“2”分别对应于“蕴涵”“矛盾”和“中性”）。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提：', x0)
    print('假设：', x1)
    print('标签：', y)
```

训练集约有550000对，测试集约有10000对。下面显示了训练集和测试集中的三个[**标签“蕴涵”“矛盾”和“中性”是平衡的**]。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**定义用于加载数据集的类**]

下面我们来定义一个用于加载SNLI数据集的类。类构造函数中的变量`num_steps`指定文本序列的长度，使得每个小批量序列将具有相同的形状。换句话说，在较长序列中的前`num_steps`个标记之后的标记被截断，而特殊标记“&lt;pad&gt;”将被附加到较短的序列后，直到它们的长度变为`num_steps`。通过实现`__getitem__`功能，我们可以任意访问带有索引`idx`的前提、假设和标签。

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
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
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
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

```{.python .input}
#@tab paddle
#@save
class SNLIDataset(paddle.io.Dataset):
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = paddle.to_tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return paddle.to_tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.premises)
```

### [**整合代码**]

现在，我们可以调用`read_snli`函数和`SNLIDataset`类来下载SNLI数据集，并返回训练集和测试集的`DataLoader`实例，以及训练集的词表。值得注意的是，我们必须使用从训练集构造的词表作为测试集的词表。因此，在训练集中训练的模型将不知道来自测试集的任何新词元。

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
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
    """下载SNLI数据集并返回数据迭代器和词表"""
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

```{.python .input}
#@tab paddle
#@save
def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = paddle.io.DataLoader(train_set,batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      return_list=True)
                                             
    test_iter = paddle.io.DataLoader(test_set, batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     return_list=True)
    return train_iter, test_iter, train_set.vocab
```

在这里，我们将批量大小设置为128时，将序列长度设置为50，并调用`load_data_snli`函数来获取数据迭代器和词表。然后我们打印词表大小。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

现在我们打印第一个小批量的形状。与情感分析相反，我们有分别代表前提和假设的两个输入`X[0]`和`X[1]`。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 小结

* 自然语言推断研究“假设”是否可以从“前提”推断出来，其中两者都是文本序列。
* 在自然语言推断中，前提和假设之间的关系包括蕴涵关系、矛盾关系和中性关系。
* 斯坦福自然语言推断（SNLI）语料库是一个比较流行的自然语言推断基准数据集。

## 练习

1. 机器翻译长期以来一直是基于翻译输出和翻译真实值之间的表面$n$元语法匹配来进行评估的。可以设计一种用自然语言推断来评价机器翻译结果的方法吗？
1. 我们如何更改超参数以减小词表大小？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/5721)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/5722)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11828)
:end_tab:
