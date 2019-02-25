# word2vec的实现


本节是对前两节内容的实践。我们以[“词嵌入（word2vec）”](word2vec.md)一节中的跳字模型和[“近似训练”](approx-training.md)一节中的负采样为例，介绍在语料库上训练词嵌入模型的实现。我们还会介绍一些实现中的技巧，如二次采样（subsampling）。

首先导入实验所需的包或模块。

```{.python .input  n=1}
import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile
```

## 处理数据集

PTB（Penn Tree Bank）是一个常用的小型语料库 [1]。它采样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。该数据集的每一行作为一个句子。句子中的每个词由空格隔开。

```{.python .input  n=2}
with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

'# sentences: %d' % len(raw_dataset)
```

对于数据集的前3个句子，打印每个句子的词数和前5个词。这个数据集中句尾符为“&lt;eos&gt;”，生僻词全用“&lt;unk&gt;”表示，数字则被替换成了“N”。

```{.python .input  n=3}
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])
```

### 建立词语索引

为了计算简单，我们只保留在数据集中至少出现5次的词。

```{.python .input  n=4}
# tk是token的缩写
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
```

然后将词映射到整数索引。

```{.python .input  n=5}
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
'# tokens: %d' % num_tokens
```

### 二次采样

文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样 [2]。
具体来说，数据集中每个被索引词$w_i$将有一定概率被丢弃，该丢弃概率为

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$ 

其中 $f(w_i)$ 是数据集中词$w_i$的个数与总词数之比，常数$t$是一个超参数（实验中设为$10^{-4}$）。可见，只有当$f(w_i) > t$时，我们才有可能在二次采样中丢弃词$w_i$，并且越高频的词被丢弃的概率越大。

```{.python .input  n=6}
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
'# tokens: %d' % sum([len(st) for st in subsampled_dataset])
```

可以看到，二次采样后我们去掉了一半左右的词。下面比较一个词在二次采样前后出现在数据集中的次数。可见高频词“the”的采样率不足1/20。

```{.python .input  n=7}
def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

compare_counts('the')
```

但低频词“join”则完整地保留了下来。

```{.python .input  n=8}
compare_counts('join')
```

### 提取中心词和背景词

我们将与中心词距离不超过背景窗口大小的词作为它的背景词。下面定义函数提取出所有中心词和它们的背景词。它每次在整数1和`max_window_size`（最大背景窗口）之间随机均匀采样一个整数作为背景窗口大小。

```{.python .input  n=9}
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts
```

下面我们创建一个人工数据集，其中含有词数分别为7和3的两个句子。设最大背景窗口为2，打印所有中心词和它们的背景词。

```{.python .input  n=10}
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

实验中，我们设最大背景窗口大小为5。下面提取数据集中所有的中心词及其背景词。

```{.python .input  n=11}
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
```

## 负采样

我们使用负采样来进行近似训练。对于一对中心词和背景词，我们随机采样$K$个噪声词（实验中设$K=5$）。根据word2vec论文的建议，噪声词采样概率$P(w)$设为$w$词频与总词频之比的0.75次方 [2]。

```{.python .input  n=12}
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)
```

## 读取数据

我们从数据集中提取所有中心词`all_centers`，以及每个中心词对应的背景词`all_contexts`和噪声词`all_negatives`。我们将通过随机小批量来读取它们。

在一个小批量数据中，第$i$个样本包括一个中心词以及它所对应的$n_i$个背景词和$m_i$个噪声词。由于每个样本的背景窗口大小可能不一样，其中背景词与噪声词个数之和$n_i+m_i$也会不同。在构造小批量时，我们将每个样本的背景词和噪声词连结在一起，并添加填充项0直至连结后的长度相同，即长度均为$\max_i n_i+m_i$（`max_len`变量）。为了避免填充项对损失函数计算的影响，我们构造了掩码变量`masks`，其每一个元素分别与连结后的背景词和噪声词`contexts_negatives`中的元素一一对应。当`contexts_negatives`变量中的某个元素为填充项时，相同位置的掩码变量`masks`中的元素取0，否则取1。为了区分正类和负类，我们还需要将`contexts_negatives`变量中的背景词和噪声词区分开来。依据掩码变量的构造思路，我们只需创建与`contexts_negatives`变量形状相同的标签变量`labels`，并将与背景词（正类）对应的元素设1，其余清0。

下面我们实现这个小批量读取函数`batchify`。它的小批量输入`data`是一个长度为批量大小的列表，其中每个元素分别包含中心词`center`、背景词`context`和噪声词`negative`。该函数返回的小批量数据符合我们需要的格式，例如，包含了掩码变量。

```{.python .input  n=13}
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))
```

我们用刚刚定义的`batchify`函数指定`DataLoader`实例中小批量的读取方式，然后打印读取的第一个批量中各个变量的形状。

```{.python .input  n=14}
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break
```

## 跳字模型

我们将通过使用嵌入层和小批量乘法来实现跳字模型。它们也常常用于实现其他自然语言处理的应用。

### 嵌入层

获取词嵌入的层称为嵌入层，在Gluon中可以通过创建`nn.Embedding`实例得到。嵌入层的权重是一个矩阵，其行数为词典大小（`input_dim`），列数为每个词向量的维度（`output_dim`）。我们设词典大小为20，词向量的维度为4。

```{.python .input  n=15}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

嵌入层的输入为词的索引。输入一个词的索引$i$，嵌入层返回权重矩阵的第$i$行作为它的词向量。下面我们将形状为(2, 3)的索引输入进嵌入层，由于词向量的维度为4，我们得到形状为(2, 3, 4)的词向量。

```{.python .input  n=16}
x = nd.array([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 小批量乘法

我们可以使用小批量乘法运算`batch_dot`对两个小批量中的矩阵一一做乘法。假设第一个小批量中包含$n$个形状为$a\times b$的矩阵$\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n$，第二个小批量中包含$n$个形状为$b\times c$的矩阵$\boldsymbol{Y}_1, \ldots, \boldsymbol{Y}_n$。这两个小批量的矩阵乘法输出为$n$个形状为$a\times c$的矩阵$\boldsymbol{X}_1\boldsymbol{Y}_1, \ldots, \boldsymbol{X}_n\boldsymbol{Y}_n$。因此，给定两个形状分别为($n$, $a$, $b$)和($n$, $b$, $c$)的`NDArray`，小批量乘法输出的形状为($n$, $a$, $c$)。

```{.python .input  n=17}
X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
nd.batch_dot(X, Y).shape
```

### 跳字模型前向计算

在前向计算中，跳字模型的输入包含中心词索引`center`以及连结的背景词与噪声词索引`contexts_and_negatives`。其中`center`变量的形状为(批量大小, 1)，而`contexts_and_negatives`变量的形状为(批量大小, `max_len`)。这两个变量先通过词嵌入层分别由词索引变换为词向量，再通过小批量乘法得到形状为(批量大小, 1, `max_len`)的输出。输出中的每个元素是中心词向量与背景词向量或噪声词向量的内积。

```{.python .input  n=18}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

## 训练模型

在训练词嵌入模型之前，我们需要定义模型的损失函数。


### 二元交叉熵损失函数

根据负采样中损失函数的定义，我们可以直接使用Gluon的二元交叉熵损失函数`SigmoidBinaryCrossEntropyLoss`。

```{.python .input  n=19}
loss = gloss.SigmoidBinaryCrossEntropyLoss()
```

值得一提的是，我们可以通过掩码变量指定小批量中参与损失函数计算的部分预测值和标签：当掩码为1时，相应位置的预测值和标签将参与损失函数的计算；当掩码为0时，相应位置的预测值和标签则不参与损失函数的计算。我们之前提到，掩码变量可用于避免填充项对损失函数计算的影响。

```{.python .input  n=20}
pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

作为比较，下面将从零开始实现二元交叉熵损失函数的计算，并根据掩码变量`mask`计算掩码为1的预测值和标签的损失。

```{.python .input  n=21}
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print('%.7f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))
print('%.7f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))
```

### 初始化模型参数

我们分别构造中心词和背景词的嵌入层，并将超参数词向量维度`embed_size`设置成100。

```{.python .input  n=22}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))
```

### 定义训练函数

下面定义训练函数。由于填充项的存在，与之前的训练函数相比，损失函数的计算稍有不同。

```{.python .input  n=23}
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))
```

现在我们就可以使用负采样训练跳字模型了。

```{.python .input  n=24}
train(net, 0.005, 5)
```

## 应用词嵌入模型

训练好词嵌入模型之后，我们可以根据两个词向量的余弦相似度表示词与词之间在语义上的相似度。可以看到，使用训练得到的词嵌入模型时，与词“chip”语义最接近的词大多与芯片有关。

```{.python .input  n=25}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))

get_similar_tokens('chip', 3, net[0])
```

## 小结

* 可以使用Gluon通过负采样训练跳字模型。
* 二次采样试图尽可能减轻高频词对训练词嵌入模型的影响。
* 可以将长度不同的样本填充至长度相同的小批量，并通过掩码变量区分非填充和填充，然后只令非填充参与损失函数的计算。


## 练习

* 在创建`nn.Embedding`实例时设参数`sparse_grad=True`，训练是否可以加速？查阅MXNet文档，了解该参数的意义。
* 我们用`batchify`函数指定`DataLoader`实例中小批量的读取方式，并打印了读取的第一个批量中各个变量的形状。这些形状该如何计算得到？
* 试着找出其他词的近义词。
* 调一调超参数，观察并分析实验结果。
* 当数据集较大时，我们通常在迭代模型参数时才对当前小批量里的中心词采样背景词和噪声词。也就是说，同一个中心词在不同的迭代周期可能会有不同的背景词或噪声词。这样训练有哪些好处？尝试实现该训练方法。


## 参考文献

[1] Penn Tree Bank. https://catalog.ldc.upenn.edu/LDC99T42

[2] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7761)

![](../img/qr_word2vec-gluon.svg)
