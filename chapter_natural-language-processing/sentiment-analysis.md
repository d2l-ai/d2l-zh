# 文本情感分类：使用循环神经网络

文本分类即把一段不定长的文本序列变换为类别。在这类问题中，文本情感分类（情感分析）是一项重要的自然语言处理任务。例如，Netflix或者IMDb可以对每部电影的评论进行情感分类，从而帮助各个平台改进产品，提升用户体验。

本节介绍如何使用Gluon来创建一个文本情感分类模型。该模型将判断一段不定长的文本序列中包含的是正面还是负面的情绪，也即将文本序列分类为正面或负面。

## 模型设计

在这个模型中，我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络。首先，文本序列的每一个词将以预训练的词向量作为词的特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出。在本节的实验中，我们将双向长短期记忆在最初时间步和最终时间步的隐藏状态连结，作为特征序列的编码信息传递给输出层分类。

在实验开始前，导入所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import collections
import gluonbook as gb
from mxnet import autograd, gluon, init, metric, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
from time import time
import tarfile
```

## 读取IMDb数据集

我们使用Stanford's Large Movie Review Dataset作为文本情感分类的数据集 [1]。它的下载地址是

> http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 。

这个数据集分为训练和测试用的两个数据集，分别有25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”（1）和“负面”（0）的评论数量相等。
我们首先下载这个数据集到“../data”路径下。该数据集的压缩包大小是 81MB，下载解压需要一定时间。解压后的数据集将会放置在“../data/aclImdb”路径下。

```{.python .input  n=4}
def download_imdb(data_dir='../data'):
    """Download the IMDb Dataset."""
    imdb_dir = os.path.join(data_dir, 'aclImdb')
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
    return imdb_dir

imdb_dir = download_imdb()
```

下面，读取训练和测试数据集。

```{.python .input  n=5}
def readIMDB(dir_url, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join('../data/',dir_url, seg, label))
        for file in files:
            with open(os.path.join('../data/',dir_url, seg, label, file), 'r',
                      encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data

train_data = readIMDB('aclImdb', 'train')
test_data = readIMDB('aclImdb', 'test')

random.shuffle(train_data)
random.shuffle(test_data)
```

## 分词

接下来我们对每条评论做分词，从而得到分好词的评论。这里使用最简单的方法：基于空格进行分词。我们将在本节练习中探究其他的分词方法。

```{.python .input  n=6}
def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
test_tokenized = []
for review, score in test_data:
    test_tokenized.append(tokenizer(review))
```

## 创建词典

现在，我们可以根据分好词的训练数据集来创建词典了。这里我们设置了特殊符号“&lt;unk&gt;”（unknown）。它将表示一切不存在于训练数据集词典中的词。

```{.python .input  n=7}
token_counter = collections.Counter()
def count_token(train_tokenized):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1

count_token(train_tokenized)
vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>',
                              reserved_tokens=None)
```

## 预处理数据

下面，我们继续对数据进行预处理。每个不定长的评论将被特殊符号`PAD`补成长度为`maxlen`的序列，并用NDArray表示。

```{.python .input  n=8}
def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)         
    return features

def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen）。
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features

train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 500, 0))
test_features = nd.array(pad_samples(test_features, 500, 0))
train_labels = nd.array([score for _, score in train_data])
test_labels = nd.array([score for _, score in test_data])
```

## 加载预训练的词向量

这里，我们为词典`vocab`中的每个词加载GloVe词向量（每个词向量长度为100）。稍后，我们将用这些词向量作为评论中每个词的特征向量。

```{.python .input  n=9}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

## 定义模型

下面我们根据模型设计里的描述定义情感分类模型。其中的`Embedding`实例即嵌入层，`LSTM`实例即对句子编码信息的隐藏层，`Dense`实例即生成分类结果的输出层。

```{.python .input  n=10}
class SentimentNet(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(len(vocab), embed_size)
            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
```

由于情感分类的训练数据集并不是很大，为应对过拟合现象，我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。在训练中，我们不再更新这些词向量，即不再迭代模型嵌入层中的参数。

```{.python .input  n=11}
num_outputs = 2
lr = 0.8
num_epochs = 5
batch_size = 64
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
ctx = gb.try_all_gpus()

net = SentimentNet(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)
# 设置 embedding 层的 weight 为预训练的词向量。
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
# 训练中不更新词向量（net.embedding 中的模型参数）。
net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练并评价模型

加载完数据以后，我们就可以训练模型了。

```{.python .input  n=40}
train_set = gdata.ArrayDataset(train_features, train_labels)
test_set = gdata.ArrayDataset(test_features, test_labels)
train_loader = gdata.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True)
test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

gb.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs)
```

下面我们使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input  n=18}
def get_sentiment(vocab, sentence):
    sentence = nd.array([vocab.token_to_idx[token] for token in sentence],
                        ctx=gb.try_gpu())
    label = nd.argmax(net(nd.reshape(sentence, shape=(1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'

get_sentiment(vocab, ['i', 'think', 'this', 'movie', 'is', 'great'])
```

```{.python .input}
get_sentiment(vocab, ['the', 'show', 'is', 'terribly', 'boring'])
```

## 小结

* 我们可以应用预训练的词向量和循环神经网络对文本的情感进行分类。


## 练习

* 把迭代周期改大。你的模型能在训练和测试数据集上得到怎样的准确率？通过调节超参数，你能进一步提升分类准确率吗？

* 使用更大的预训练词向量，例如300维的GloVe词向量，能否提升分类准确率？

* 使用spacy分词工具，能否提升分类准确率？。你需要安装spacy：`pip install spacy`，并且安装英文包：`python -m spacy download en`。在代码中，先导入spacy：`import spacy`。然后加载spacy英文包：`spacy_en = spacy.load('en')`。最后定义函数：`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe的词向量对于名词词组的存储方式是用“-”连接各个单词，例如词组“new york”在GloVe中的表示为“new-york”。而使用spacy分词之后“new york”的存储可能是“new york”。

* 通过上面三种方法，你能使模型在测试集上的准确率提高到0.85以上吗？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)


![](../img/qr_sentiment-analysis.svg)


## 参考文献

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.
