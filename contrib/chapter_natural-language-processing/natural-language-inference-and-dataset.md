# 自然语言推理及数据集

在前面的一些章节中，我们介绍了诸多用于文本分类的模型。具体来说，是给定单个句子，然后识别该句子的类别。现在我们再来介绍一个句对分类任务。

## 自然语言推理

自然语言推理（Natural language inference）也称文本蕴含（Text Entailment），是一个重要的NLP问题，是指给定一个前提句 (Premise) ，根据这个前提去判断假设句 (Hypothesis) 与前提句的推理关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。

例如：

> 前提：A soccer game with multiple males playing.
> 假设：Some men are playing a sport.
> 关系：蕴含

> 前提：A man inspects the uniform of a figure in some East Asian country.
> 假设：The man is sleeping.
> 关系：矛盾

> 前提：An older and younger man smiling.
> 假设：Two men are smiling and laughing at the cats playing on the floor
> 关系：中性

所以本质上，自然语言推理可以说是一个句对分类任务。

## 斯坦福自然语言推理（SNLI）数据集

自然语言推理任务常用的数据集包括斯坦福大学自然语言推理（SNLI）数据集和多类型自然语言推理（MultiNLI）数据集 。

斯坦福自然语言推理（SNLI）数据集包含57万人工书写英语句子对，手动标记为蕴涵，矛盾和中立。

多类型自然语言推理（MultiNLI）数据集类似于SNLI数据集，但不同之处在于涵盖了一系列口语和书面文本，并支持独特的跨类型泛化评估。

在本书中，我们将使用斯坦福大学自然语言推理（SNLI）数据集。

为了更好地了解这个数据集，我们先导入实验所需的包或模块。

```{.python .input  n=2}
import d2lzh as d2l
from mxnet import gluon, init, np, npx
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
import zipfile

npx.set_np()
```

我们下载这个数据集的压缩包到../data路径下。压缩包大小是100 MB左右。解压之后的数据集将会放置在../data/snli_1.0路径下。

```{.python .input  n=2}
# Save to the d2l package.
def download_snli(data_dir='../data/'):
    url = ('https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
    sha1 = '9fcde07509c7e87ec61c640c1b2753d9041758e4'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall(data_dir)
        
download_snli()
```

进入../data/snli_1.0路径后，我们可以获取数据集的不同组成部分。包含了分割好的训练集、验证集和测试集。

每个集合主要包含下面的字段。

- sentence1：原始前提句。

- sentence2：原始假设句。

- sentence{1,2} _parse：由Penn Treebank格式的Stanford Parser产生的句法解析。

- sentence{1,2} _binary_parse：与句子{1,2} _parse相同的解析，但没有一元节点且没有节点标签。

- annotator_labels：这些是人工标注时的所有单个标签。少数情况下，标签之一可能是空白的，表示注释者看到了句子但无法对其进行标注。

- gold_label：这是大多数人工标注者选择的标签。如果不存在多数，则为“ - ”。



接下来，我们读取训练数据集、验证数据集和测试数据集。且只保留包含有效gold_label的样本。

```{.python .input  n=3}
def read_file(filename):
    label_set = set(["entailment", "contradiction", "neutral"])
    def tokenized(text):
        return text.replace("(", "").replace(")", "").strip().split()
    with open(os.path.join('../data/snli_1.0/', filename), 'r') as f:
        examples = [row.split('\t') for row in f.readlines()[1:]]
    return [(tokenized(row[1]), tokenized(row[2]), row[0]) for row in examples if row[0] in label_set]

train_data, dev_data, test_data = [read_file('snli_1.0_'+ split + '.txt') for split in ["train", "dev", "test"]]
```

我们输出前5个句对和它们的标签。

```{.python .input  n=3}
train_data[:5]
```

我们做一下基本的统计。

```{.python .input  n=3}
print("Training pairs: %d" % len(train_data))
print("Dev pairs: %d" % len(dev_data))
print("Test pairs: %d" % len(test_data))

print("Train labels: {'entailment': %d, 'contradiction': %d, 'neutral': %d}" % ([row[2] for row in train_data].count('entailment'), [row[2] for row in train_data].count('contradiction'), [row[2] for row in train_data].count('neutral')))
print("Dev labels: {'entailment': %d, 'contradiction': %d, 'neutral': %d}" % ([row[2] for row in dev_data].count('entailment'), [row[2] for row in dev_data].count('contradiction'), [row[2] for row in dev_data].count('entailment')))
print("Test labels: {'entailment': %d, 'contradiction': %d, 'neutral': %d}" % ([row[2] for row in test_data].count('entailment'), [row[2] for row in test_data].count('contradiction'), [row[2] for row in test_data].count('neutral')))
```

### 自定义数据集类

我们通过继承Gluon提供的`Dataset`类自定义了一个自然语言推理数据集类`SNLIDataset`。通过实现`__getitem__`函数，我们可以任意访问数据集中索引为idx的输入句对及其对应类别。由于标签是字符串形式，我们需要将其转换为数字索引形式。

```{.python .input  n=3}
# Save to the d2l package.
class SNLIDataset(gdata.Dataset):
    def __init__(self, dataset, max_length, vocab=None):
        self.dataset = dataset
        self.max_length = max_length
        self.data = read_file('snli_1.0_'+ dataset + '.txt')
        if vocab is None:
            self.vocab = self.get_vocab_snli(self.data)
        else:
            self.vocab = vocab
        self.premise, self.hypothesis, self.labels = self.preprocess_snli(self.data, self.vocab)
        print('read ' + str(len(self.premise)) + ' examples')

    def get_vocab_snli(self, data):
        # 过滤出现频度小于5的词
        counter = collections.Counter([tk for s in data for st in s[:2] for tk in st])
        return text.vocab.Vocabulary(counter, min_freq=5)

    def preprocess_snli(self, data, vocab):
        LABEL_TO_IDX = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        max_l = 50  # 将每条评论通过截断或者补0，使得长度变成50

        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

        premise = np.array([pad(vocab.to_indices(x[0])) for x in data])
        hypothesis = np.array([pad(vocab.to_indices(x[1])) for x in data])
        labels = np.array([LABEL_TO_IDX[x[2]] for x in data])
        return premise, hypothesis, labels

    def __getitem__(self, idx):
        return self.premise[idx], self.hypothesis[idx], self.labels[idx]

    def __len__(self):
        return len(self.premise)
```



### 读取数据集

通过自定义的`SNLIDataset`类来分别创建训练集、验证集和测试集的实例。我们指定最大文本长度为50。下面我们可以分别查看训练集、验证集和测试集所保留的样本个数。

```{.python .input  n=3}
train_set = SNLIDataset("train", 50)
dev_set = SNLIDataset("dev", 50, train_set.vocab)
test_set = SNLIDataset("test", 50, train_set.vocab)
```

设批量大小为64，分别定义训练集、验证集和测试集的迭代器。

```{.python .input  n=3}
batch_size = 64
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
dev_iter = gdata.DataLoader(dev_set, batch_size)
test_iter = gdata.DataLoader(test_set, batch_size)
```

打印第一个小批量的形状。不同于文本分类任务，这里的数据是个三元组（句子1，句子2，标签）。

```{.python .input  n=3}
for X1, X2, Y in train_iter:
    print(X1.shape)
    print(X2.shape)
    print(Y.shape)
    break
```

## 小结
- 自然语言推理任务是判断给定的前提句 (Premise)与假设句 (Hypothesis) 间的推理关系
- 自然语言推理任务中句子间的推理关系包含三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。
- 自然语言推理任务一个重要数据集叫作斯坦福自然语言推理（SNLI）数据集。