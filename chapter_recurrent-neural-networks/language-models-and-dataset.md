# 语言模型和数据集
:label:`sec_language_model`

在 :numref:`sec_text_preprocessing` 中，我们将看到如何将文本数据映射到令牌中，这些标记可以被视为一系列离散观测值，例如单词或字符。假定长度为 $T$ 的文本序列中的标记反过来是 $x_1, x_2, \ldots, x_T$。然后，在文本序列中，可以将 $x_t$ ($1 \leq t \leq T$) 视为时间步长 $t$ 处的观察值或标签。给定这样的文本序列，* 语言模型 * 的目标是估计序列的联合概率

$$P(x_1, x_2, \ldots, x_T).$$

语言模型非常有用。例如，理想的语言模型将能够单独生成自然文本，只需一次绘制一个标记 $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$。与使用打字机的猴子完全不同，从这种模型中出现的所有文本都会作为自然语言传递，例如英文文本。此外，只需将文本放在以前的对话片段上，即可生成一个有意义的对话框就足够了。显然，我们还远离设计这样一个系统，因为它需要 * 理解 * 文本，而不是仅仅生成语法合理的内容。

尽管如此，语言模式即使形式有限，也有很大的帮助。例如，短语 “识别讲话” 和 “破坏一个漂亮的海滩” 听起来非常相似。这可能会导致语音识别中的歧义，通过拒绝第二次翻译为古怪的语言模型很容易解决这一问题。同样，在文档总结算法中，值得了解的是，“狗咬人” 比 “人咬狗” 更频繁，或者 “我想吃奶奶” 是一个相当令人不安的陈述，而 “我想吃，奶奶” 则更加良性。

## 学习语言模型

显而易见的问题是我们应该如何建模一个文档，甚至是一个令牌序列。假设我们在字级别标记文本数据。我们可以利用我们在 :numref:`sec_sequence` 中应用于序列模型的分析。让我们从应用基本概率规则开始：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

例如，包含四个单词的文本序列的概率如下：

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

为了计算语言模型，我们需要计算单词的概率和给定前几个单词的条件概率。这种概率基本上是语言模型参数。

在这里，我们假设训练数据集是一个较大的文本语料库，例如所有维基百科条目，[Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg)，以及网络上发布的所有文本。单词概率可以根据训练数据集中给定单词的相对字频来计算。例如，估计值 $\hat{P}(\text{deep})$ 可以计算为以单词 “深” 开头的任何句子的概率。稍微不精确的方法是计算 “深度” 一词的所有出现次数，并将其除以语料库中的单词总数。这工作得很好，特别是对于频繁的单词。继续前进，我们可以尝试估计

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

其中 $n(x)$ 和 $n(x, x')$ 分别是单例和连续单词对的出现次数。不幸的是，估计一对单词的概率比较困难，因为 “深度学习” 的发生要少得多。特别是，对于一些不寻常的单词组合来说，找到足够的出现次数以获得准确的估计值可能会很棘手。对于三个词组合和超越的事情，情况会变得更糟。在我们的数据集中可能看不到很多合理的三个词组合。除非我们提供一些解决方案来分配这样的单词组合非零计数，否则我们将无法在语言模型中使用它们。如果数据集很小，或者如果单词非常罕见，我们可能甚至找不到其中的一个。

一个常见的策略是执行某种形式的 * 拉普拉斯平滑 *。解决方案是在所有计数中添加一个小常量。用 $n$ 表示训练集中的单词总数和 $m$ 唯一单词的数量。这个解决方案有助于单例，例如通过

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

这里是超参数。以 $\epsilon_1$ 为例：当 $\epsilon_1 = 0$ 时，不应用平滑；当 $\epsilon_1$ 接近正无穷大时，$\hat{P}(x)$ 接近均匀概率 $1/m$。以上是其他技术可以完成 :cite:`Wood.Gasthaus.Archambeau.ea.2011` 的一个相当原始的变体。

不幸的是，由于以下原因，像这样的模型变得非常笨拙。首先，我们需要存储所有计数。第二，这完全忽略了这个词的含义。例如，"猫" 和 "猫科动物" 应在相关情况下出现。很难根据其他情况调整这些模型，而基于深度学习的语言模型非常适合考虑这一点。最后，长字序列几乎肯定是新颖的，因此简单地计算以前看到的单词序列频率的模型必然在那里表现不佳。

## 马尔科夫模型和 $n$ 克

在讨论涉及深度学习的解决方案之前，我们需要更多的术语和概念。回想一下我们在 :numref:`sec_sequence` 中对马尔科夫模型的讨论。让我们将其应用于语言建模。如果 $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$，则序列上的分布满足第一阶的马尔科夫属性。较高的订单对应于较长的依赖关系。这导致了一些我们可以应用于建模序列的近似值：

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

涉及一个、两个和三个变量的概率公式通常分别称为 * unigram*、* 双重 * 和 *trigram* 模型。在下文中，我们将学习如何设计更好的模型。

## 自然语言统计

让我们看看这是如何在真实数据上工作的。我们基于 :numref:`sec_text_preprocessing` 中引入的时间机器数据集构建一个词汇，并打印前 10 个最常见的单词。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessisarily a sentence or a paragraph, we
# concatenate all text lines 
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

正如我们所看到的，最流行的单词实际上是相当无聊的看法。它们通常被称为 * 停止字 *，因此被过滤掉。尽管如此，它们仍然具有意义，我们仍将使用它们。此外，很清楚，这个词频率衰减相当迅速。$10^{\mathrm{th}}$ 最常见的单词小于 $1/5$，就像最流行的单词一样常见。为了得到一个更好的想法，我们绘制了词频的数字。

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

我们在这里处理了一些非常重要的事情：词频率以明确定义的方式迅速衰减。在将前几个单词作为例外处理之后，所有剩余的单词大致遵循对数对数图上的直线。这意味着单词符合 *Zipf 的定律 *，其中规定 $i^\mathrm{th}$ 最常见单词的频率为：

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

，这等同于

$$\log n_i = -\alpha \log i + c,$$

其中 $\alpha$ 是描述分布特征的指数，$c$ 是一个常数。如果我们想通过计数统计和平滑来建模单词，这应该已经给我们暂停。毕竟，我们将大大高估了尾巴的频率，也被称为不常见的话。但是，其他单词组合，如双字图，三角图，以及更多的内容呢？让我们看看双字图频率的行为方式是否与联合图频率相同。

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

有一件事在这里值得注意。在十个最常见的单词对中，九个由两个停止单词组成，只有一个与实际书籍有关-“时间”。此外，让我们看看三角图频率是否以相同的方式行为。

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

最后，让我们可视化这三种模型中的令牌频率：统一图、双曲图和三角图。

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

这个数字是相当令人兴奋的有几个原因。首先，除了联合词之外，单词序列似乎也遵循 Zipf 定律，尽管 :eqref:`eq_zipf_law` 中的指数较小 $\alpha$，具体取决于序列长度。其次，不同的 $n$ 克的数量并不是那么大。这给我们带来了希望，在语言上有相当多的结构。第三，许多 $n$ 克很少出现，这使得拉普拉斯平滑而不适合语言建模。相反，我们将使用基于深度学习的模型。

## 读取长序列数据

由于序列数据本质上是顺序的，因此我们需要解决处理它的问题。我们在 :numref:`sec_sequence` 号文件中以相当临时的方式这样做。当序列过长而无法由模型一次处理时，我们可能希望拆分这样的序列以供读取。现在让我们来描述一般战略。在介绍模型之前，让我们假设我们将使用神经网络来训练语言模型，其中网络一次处理一批预定义长度的序列（例如 $n$ 时间步长）。现在的问题是如何随机阅读要素和标签的微批次。

首先，由于文本序列可以是任意长度的，例如整个 *Time Machine* 书，我们可以将这样长的序列划分为具有相同数量的时间步长的子序列。在训练我们的神经网络时，这样的子序列的小批量将被输入到模型中。假设网络一次处理 $n$ 时间步长的子序列。:numref:`fig_timemachine_5gram` 显示了从原始文本序列获取子序列的所有不同方法，其中 $n=5$ 和每个时间步的标记对应于一个字符。请注意，我们有相当的自由度，因为我们可以选择一个指示初始位置的任意偏移量。

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

因此，我们应该从 :numref:`fig_timemachine_5gram` 中选择哪一个？事实上，所有这些都是同样好的。但是，如果我们只选择一个偏移量，则所有可能的子序列用于训练我们的网络。因此，我们可以从随机偏移开始对序列进行分区，以获得 * 覆盖 * 和 * 随机 *。在下面，我们描述了如何在
*随机采样 * 和 * 顺序分区 * 策略。

### 随机采样

在随机采样中，每个示例都是在原始长序列上任意捕获的子序列。迭代期间来自两个相邻随机微批次的子序列在原始序列上不一定相邻。对于语言建模，目标是根据迄今为止我们看到的令牌预测下一个令牌，因此标签是原始序列，移动一个令牌。

下面的代码每次都会从数据中随机生成一个小批处理。在这里，参数 `batch_size` 指定了每个小批次中的子序列示例的数量，`num_steps` 是每个子序列中预定义的时间步长数。

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset to partition a sequence
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_subseqs_per_example = num_subseqs // batch_size
    for i in range(0, batch_size * num_subseqs_per_example, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

让我们手动生成一个从 0 到 34 的序列。我们假定批次大小和时间步长的数量分别为 2 和 5。这意味着我们可以生成 $\lfloor (35 - 1) / 5 \rfloor= 6$ 要素标注子序列对。如果小批量为 2，我们只得到 3 个微批次。

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### 顺序分区

除了对原始序列进行随机采样之外，我们还可以确保迭代期间来自两个相邻微批次的子序列在原始序列上相邻。此策略在迭代微批次时保留分割子序列的顺序，因此称为顺序分区。

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

使用相同的设置，让我们为按顺序分区读取的每个小批次的子序列打印功能 `X` 和标签 `Y`。请注意，迭代期间来自两个相邻微批次的子序列在原始序列上确实相邻。

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

现在我们将上述两个采样函数包装到一个类中，以便稍后我们可以将其用作数据迭代器。

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

最后，我们定义了一个函数 `load_data_time_machine`，它同时返回数据迭代器和词汇表，因此我们可以像其他带有 `load_data` 前缀的函数一样使用它，例如 :numref:`sec_fashion_mnist` 中定义的 `d2l.load_data_fashion_mnist`。

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## 摘要

* 语言模型是自然语言处理的关键。
* $n$ 克提供了一个方便的模型，用于通过截断依赖性来处理长序列。
* 长序列遭受的问题，他们很少发生或从来没有。
* Zipf 的定律管辖单词分布不仅适用于单词，而且还适用于其他 $n$ 克。
* 有很多结构，但没有足够的频率来通过拉普拉斯平滑有效地处理不常见的单词组合。
* 读取长序列的主要选择是随机采样和顺序分区。后者可以确保迭代期间来自两个相邻微批次的子序列在原始序列上相邻。

## 练习

1. 假设训练数据集中有 $100,000$ 个单词。四克需要存储多少字频和多字相邻频率？
1. 你会如何模拟对话？
1. 估计统一图、双曲图和三角图的 Zipf 定律的指数。
1. 您还能想到什么其他方法来读取长序列数据？
1. 考虑我们用于读取长序列的随机偏移量。
    1. 为什么有一个随机偏移是一个好主意？
    1. 它是否真的导致文档上的序列完全均匀的分布？
    1. 你要怎么做才能让事情变得更加统一
1. 如果我们希望一个序列示例成为一个完整的句子，这在小批量采样中引入了什么样的问题？我们如何解决这个问题？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
