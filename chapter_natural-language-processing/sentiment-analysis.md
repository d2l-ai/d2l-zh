# 文本分类：情感分析

文本分类即把一段不定长的文本序列变换为类别。在文本分类问题中，情感分析是一项重要的自然语言处理的任务。例如，Netflix或者IMDb可以对每部电影的评论进行情感分类，从而帮助各个平台改进产品，提升用户体验。

本节介绍如何使用Gluon来创建一个情感分类模型。该模型将判断一段不定长的文本序列中包含的是正面还是负面的情绪，也即将文本序列分类为正面或负面。

## 模型设计

在这个模型中，我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络。首先，文本序列的每一个词将以预训练的词向量作为词的特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出。在本节的实验中，我们将双向长短期记忆在最初时间步和最终时间步的隐藏状态连结，并作为特征序列的编码信息传递给输出层分类。

在实验开始前，导入所需的包或模块。

```{.python .input  n=1}
import sys
sys.path.append('..')
import collections
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.contrib import text
import os
import random
import zipfile
```

## 读取IMDb数据集

我们使用Stanford's Large Movie Review Dataset作为情感分析的数据集 [1]。它的下载地址是

> http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 。

这个数据集分为训练和测试用的两个数据集，分别有25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”（1）和“负面”（0）的评论数量相等。将下载好的数据解压并存放在路径“../data/aclImdb”。

为方便快速上手，我们提供了上述数据集的小规模采样，并存放在路径“../data/aclImdb_tiny.zip”。如果你将使用上述的IMDb完整数据集，还需要把下面`demo`变量改为`False`。

```{.python .input  n=2}
# 如果训练下载的 IMDb 的完整数据集，把下面改 False。
demo = True
if demo:
    with zipfile.ZipFile('../data/aclImdb_tiny.zip', 'r') as zin:
        zin.extractall('../data/')
```

下面，读取训练和测试数据集。

```{.python .input}
def readIMDB(dir_url, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(
            '../data/' + dir_url + '/' + seg + '/' + label + '/')
        for file in files:
            with open('../data/' + dir_url + '/' + seg + '/' + label + '/' 
                      + file, 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data

if demo:
    train_data = readIMDB('aclImdb_tiny/', 'train')
    test_data = readIMDB('aclImdb_tiny/', 'test')
else:
    train_data = readIMDB('aclImdb/', 'train')
    test_data = readIMDB('aclImdb/', 'test')

random.shuffle(train_data)
random.shuffle(test_data)
```

## 分词

接下来我们对每条评论做分词，从而得到分好词的评论。这里使用最简单的方法：基于空格进行分词。我们将在本节练习中探究其他的分词方法。

```{.python .input  n=4}
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

```{.python .input  n=6}
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

下面，我们继续对数据进行预处理。每个不定长的评论将被特殊符号`padding`补成长度为`maxlen`的序列。

```{.python .input  n=7}
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

def pad_samples(features, maxlen=500, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）。
            while len(padded_feature) < maxlen:
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features

ctx = gb.try_gpu()
train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 500, 0), ctx=ctx)
test_features = nd.array(pad_samples(test_features, 500, 0), ctx=ctx)
train_labels = nd.array([score for _, score in train_data], ctx=ctx)
test_labels = nd.array([score for _, score in test_data], ctx=ctx)
```

## 加载预训练的词向量

这里，我们为词典`vocab`中的每个词加载GloVe词向量（每个词向量长度为100）。稍后，我们将用这些词向量作为评论中每个词的特征向量。

```{.python .input  n=11}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

## 定义模型

下面我们根据模型设计里的描述定义情感分类模型。其中的Embedding实例即嵌入层。

```{.python .input}
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

    def forward(self, inputs, begin_state=None):
        embeddings = self.embedding(inputs)
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
```

由于情感分类的训练数据集并不是很大，我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。在训练中，我们不再迭代这些词向量，即模型嵌入层中的参数。当我们使用完整数据集时，我们可以重新调节下面的超参数，例如增加迭代周期。

```{.python .input}
num_outputs = 2
lr = 0.1
num_epochs = 1
batch_size = 10
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
    
net = SentimentNet(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)
# 设置 embedding 层的 weight 为预训练的词向量。
net.embedding.weight.set_data(glove_embedding.idx_to_vec.as_in_context(ctx))
# 训练中不迭代词向量（net.embedding中的模型参数）。
net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练并评价模型

在实验中，我们使用准确率作为评价模型的指标。

```{.python .input  n=13}
def eval_model(features, labels):
    l_sum = 0
    l_n = 0
    accuracy = metric.Accuracy()
    for i in range(features.shape[0] // batch_size):
        X = features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
        y = labels[i*batch_size :(i+1)*batch_size].as_in_context(ctx).T
        output = net(X)
        l = loss(output, y)
        l_sum += l.sum().asscalar()
        l_n += l.size
        accuracy.update(preds=nd.argmax(output, axis=1), labels=y)
    return l_sum / l_n, accuracy.get()[1]
```

下面开始训练模型。

```{.python .input  n=14}
for epoch in range(1, num_epochs + 1):
    for i in range(train_features.shape[0] // batch_size):
        X = train_features[i*batch_size : (i+1)*batch_size].as_in_context(
            ctx).T
        y = train_labels[i*batch_size : (i+1)*batch_size].as_in_context(
            ctx).T
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    train_loss, train_acc = eval_model(train_features, train_labels)
    test_loss, test_acc = eval_model(test_features, test_labels)
    print('epoch %d, train loss %.6f, acc %.2f; test loss %.6f, acc %.2f' 
          % (epoch, train_loss, train_acc, test_loss, test_acc))
```

下面我们试着对一个简单的句子分析情感（1和0分别代表正面和负面）。为了在更复杂的句子上得到较准确的分类，我们需要使用完整数据集训练模型，并适当增大训练周期。

```{.python .input}
review = ['this', 'movie', 'is', 'great']
nd.argmax(net(nd.reshape(
    nd.array([vocab.token_to_idx[token] for token in review], ctx=ctx), 
    shape=(-1, 1))), axis=1).asscalar()
```

## 小结

* 我们可以应用预训练的词向量和循环神经网络对文本进行情感分析。


## 练习

* 使用IMDb完整数据集，并把迭代周期改为3。你的模型能在训练和测试数据集上得到怎样的准确率？通过调节超参数，你能进一步提升模型表现吗？

* 使用更大的预训练词向量，例如300维的GloVe词向量。

* 使用spacy分词工具。你需要安装spacy：`pip install spacy`，并且安装英文包：`python -m spacy download en`。在代码中，先导入spacy：`import spacy`。然后加载spacy英文包：`spacy_en = spacy.load('en')`。最后定义函数：`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe的词向量对于名词词组的存储方式是用“-”连接各个单词，例如词组“new york”在GloVe中的表示为“new-york”。而使用spacy分词之后“new york”的存储可能是“new york”。你能使模型在测试集上的准确率提高到0.85以上吗？




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)


![](../img/qr_sentiment-analysis.svg)


## 参考文献

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.
