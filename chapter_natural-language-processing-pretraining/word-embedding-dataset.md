# 用于预训词嵌入的数据集
:label:`sec_word2vec_data`

现在我们已经了解了 word2vec 模型的技术细节和近似训练方法，让我们来看看它们的实现。具体来说，我们将以 :numref:`sec_word2vec` 中的跳跃图模型和 :numref:`sec_approx_train` 中的负取样作为例子。在本节中，我们从用于预训练词嵌入模型的数据集开始：数据的原始格式将转换为可以在训练期间迭代的迷你比赛。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

## 阅读数据集

我们在这里使用的数据集是 [宾夕法尼亚树银行 (PTB)](https://catalog.ldc.upenn.edu/LDC99T42)。此语料库取自《华尔街日报》文章的样本，分为培训、验证和测试集。在原始格式中，文本文件的每一行代表一个用空格分隔的单词句。在这里，我们将每个单词视为标记。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

阅读训练集之后，我们为语料库构建了一个词汇表，其中任何出现少于 10 次的单词都将被 “<unk>” 令牌替换。请注意，原始数据集还包含<unk>代表罕见（未知）单词的 “” 标记。

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## 子采样

文本数据通常具有 “in”、“a” 和 “in” 之类的高频单词：它们甚至可能在非常大的语库中出现数十亿次。但是，这些单词通常与上下文窗口中的许多不同单词共同出现，提供很少有用的信号。例如，考虑上下文窗口中的 “芯片” 一词：直观地说，它与低频单词 “英特尔” 共同出现比与高频单词 “a” 共同出现更有用。此外，使用大量（高频）单词的训练也很慢。因此，在训练单词嵌入模型时，高频单词可以是 * 子样本 * :cite:`Mikolov.Sutskever.Chen.ea.2013`。具体而言，数据集中的每个索引单词 $w_i$ 都将被丢弃概率 

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

其中 $f(w_i)$ 是字数 $w_i$ 与数据集中字数总数的比率，而常数 $t$ 是一个超参数（实验中为 $10^{-4}$）。我们可以看到，只有当相对频率 $f(w_i) > t$ 时（高频）单词 $w_i$ 才能被丢弃，而该词的相对频率越高，被丢弃的可能性就越大。

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

以下代码片段绘制了子采样之前和之后每句话的标记数量的直方图。正如预期的那样，子采样通过删除高频单词来显著缩短句子，这将加快训练速度。

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

对于个别代币，高频单词 “the” 的采样率低于 1/20。

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

相比之下，低频单词 “加入” 被完全保留。

```{.python .input}
#@tab all
compare_counts('join')
```

在进行子采样后，我们将令牌映射到他们的语料库的指数。

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 提取中心词和上下文词

以下 `get_centers_and_contexts` 函数从 `corpus` 中提取所有中心词及其上下文单词。它以上下文窗口大小随机对 1 到 `max_window_size` 之间的整数进行统一采样。对于任何中心单词，与其距离不超过抽样上下文窗口大小的单词就是其上下文单词。

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

接下来，我们创建一个人工数据集，分别包含 7 个词和 3 个词的两个句子。让上下文窗口的最大大小为 2，然后打印所有中心单词及其上下文字。

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

在 PTB 数据集上进行训练时，我们将上下文窗口的最大大小设置为 5。以下提取数据集中的所有中心词及其上下文词。

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 负面采样

我们使用负取样进行近似训练。为了根据预定义的分布对噪声词进行采样，我们定义了以下 `RandomGenerator` 类，其中（可能是非标准化的）采样分布通过参数 `sampling_weights` 传递。

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

例如，我们可以在指数 1、2 和 3 中绘制 10 个随机变量 $X$，采样概率为 $P(X=1)=2/9, P(X=2)=3/9$ 和 $P(X=3)=4/9$，如下所示。

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

对于一对中心词和上下文词，我们随机采样 `K`（实验中有 5 个）噪声词。根据 Word2vec 论文中的建议，噪声词 $w$ 的取样概率 $P(w)$ 设置为字典中的相对频率提高到 0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013` 的功率。

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 在迷你手表中加载训练示例
:label:`subsec_word2vec-minibatch-loading`

在提取所有中心词及其上下文单词和抽样噪声单词之后，它们将被转换为可以在训练期间迭代加载的小型示例。 

在迷你手表中，$i^\mathrm{th}$ 示例包括一个中心词及其 $n_i$ 上下文词和 $m_i$ 个噪声词。由于上下文窗口大小的不同，$n_i+m_i$ 因不同 $i$ 而有所不同。因此，对于每个例子，我们在 `contexts_negatives` 变量中连接其上下文词和噪声单词，然后填入零，直到连接长度达到 $\max_i n_i+m_i$ (`max_len`)。为了在计算损失时排除填充，我们定义了掩码变量 `masks`。`masks` 中的元素与 `contexts_negatives` 中的元素之间存在一对应关系，其中 `masks` 中的零（否则为零）对应于 `contexts_negatives` 中的填充。 

为了区分正面和负面示例，我们通过 `labels` 变量将上下文单词与 `contexts_negatives` 中的噪声单词分开。与 `masks` 类似，`labels` 中的元素与 `contexts_negatives` 中的元素之间也存在一对一的对应关系，其中 `labels` 中的元素（否则为零）对应于 `contexts_negatives` 中的上下文词（正面示例）。 

上述想法在以下 `batchify` 函数中实现。它的输入 `data` 是长度等于批次大小的列表，其中每个元素都是一个示例，包括中心字 `center`、上下文字 `context` 和噪声字 `negative`。此函数返回一个可以在训练期间加载进行计算的迷你比赛，例如包括掩码变量。

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

让我们用两个示例组成的迷你手表来测试这个函数。

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 把所有东西放在一起

最后，我们定义了 `load_data_ptb` 函数，该函数读取 PTB 数据集并返回数据迭代器和词汇。

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

让我们打印数据迭代器的第一个迷你批。

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## 摘要

* 高频单词在训练中可能不那么有用。我们可以对它们进行分样以加快训练速度。
* 为了提高计算效率，我们在迷你手表中加载示例。我们可以定义其他变量来区分填充和非填充，还可以定义积极的例子和负面的例子。

## 练习

1. 如果不使用子采样，本节中的代码运行时间会如何变化？
1. `RandomGenerator` 类缓存 `k` 个随机采样结果。将 `k` 设置为其他值，看看它如何影响数据加载速度。
1. 本节代码中的哪些其他超参数可能会影响数据加载速度？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
