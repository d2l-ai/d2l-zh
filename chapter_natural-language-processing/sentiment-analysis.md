# 文本情感分类：使用循环神经网络

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本节关注它的一个子问题：使用文本情感分类（情感分析）来分析文本用户的情绪。这个问题有广泛的应用，例如分析用户对产品的评论可以统计用户的满意度，分析用户对市场行情的情绪能帮助预测接下来的行情。

本节我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络来判断一段不定长的文本序列中包含的是正面还是负面的情绪。在实验开始前，导入所需的包或模块。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

import collections
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
```

## 文本情感分类数据

我们使用Stanford's Large Movie Review Dataset作为文本情感分类的数据集 [1]。这个数据集分为训练和测试用的两个数据集，分别均有25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”（1）和“负面”（0）的评论数量相等。

###  读取数据

我们首先下载这个数据集到“../data”路径下，然后解压至“../data/aclImdb”下。

```{.python .input  n=3}
# 本函数已保存在 gluonbook 包中方便以后使用。
def download_imdb(data_dir='../data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
        
download_imdb()
```

下面，读取训练和测试数据集。每个样本是一条评论和其对应的标号，1表示正面，0表示负面。

```{.python .input  n=13}
def read_imdb(folder='train'):  # 本函数已保存在 gluonbook 包中方便以后使用。
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'r') as f:
                review = f.read().replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
```

### 数据预处理

接下来我们对每条评论做分词，从而得到分好词的评论。这里使用最简单的方法：基于空格进行分词。

```{.python .input  n=14}
def get_tokenized_imdb(data):  # 本函数已保存在 gluonbook 包中方便以后使用。
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
```

现在，我们可以根据分好词的训练数据集来创建词典了。这里我们过滤了出现次数少于5的词。

```{.python .input  n=28}
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

vocab = get_vocab_imdb(train_data)
'# words in vocab:', len(vocab)
```

因为每条评论长度不一致使得不能直接组合成小批量，下面函数将每条评论进行分词，通过词典转换成词索引，然后通过剪裁或者补0（0在词典中代表未知符号）来将每条评论长度固定成500。

```{.python .input  n=44}
def preprocess_imdb(data, vocab):  # 本函数已保存在 gluonbook 包中方便以后使用。
    max_l = 500  # 将每条评论通过截断或者补 0 来使得长固定。
    pad = lambda x: x[:max_l] if len(x) > max_l else x + [0] * (max_l-len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels
```

### 数据迭代器

现在，我们创建数据迭代器来每次返回一个小批量数据。

```{.python .input}
batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

打印第一个小批量数据形状，和训练集中小批量的个数。

```{.python .input}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)
```

## 使用循环神经网络的模型

在这个模型中，我们先将每个词通过嵌入层来得到特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出。下面实现这个模型，其中的`Embedding`实例即嵌入层，`LSTM`实例即对句子编码信息的隐藏层，`Dense`实例即生成分类结果的输出层。

```{.python .input  n=46}
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs 形状是（batch_size, seq_len=500），因为 LSTM 需要将序列作为第一维，
        # 所以将输入转置后再提取词特征，输出形状为（seq_len, batch_size, embed_size）。
        embeddings = self.embedding(inputs.T)
        # states 形状是（seq_len, batch_size, 2*num_hiddens）。
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。
        # 它的形状为（batch_size, 4*num_hiddens）。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
```

创建一个两层的双向循环神经网络。

```{.python .input}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, gb.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
```

### 加载预训练的词向量

由于情感分类的训练数据集并不是很大，为应对过拟合现象，我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。这里，我们为词典`vocab`中的每个词加载GloVe词向量（每个词向量为100维向量）。

```{.python .input  n=45}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

然后我们将用这些词向量作为评论中每个词的特征向量。注意预训练的词向量的长度需要跟创建的模型中的嵌入层输出大小一致。此外，在训练中我们不再更新这些词向量。

```{.python .input  n=47}
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')
```

### 训练并评价模型

这时候我们可以开始训练了。

```{.python .input  n=48}
lr, num_epochs = 0.8, 5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

最后，定义预测函数。

```{.python .input  n=49}
# 本函数已保存在 gluonbook 包中方便以后使用。
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=gb.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
```

```{.python .input}
然后使用训练好的模型对两个简单句子的情感进行分类。
```

```{.python .input  n=50}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## 小结

* 我们可以应用预训练的词向量和循环神经网络对文本的情感进行分类。


## 练习

* 把迭代周期改大。你的模型能在训练和测试数据集上得到怎样的准确率？通过调节超参数，你能进一步提升分类准确率吗？

* 使用更大的预训练词向量，例如300维的GloVe词向量，能否提升分类准确率？

* 使用spaCy分词工具，能否提升分类准确率？。你需要安装spaCy：`pip install spacy`，并且安装英文包：`python -m spacy download en`。在代码中，先导入spacy：`import spacy`。然后加载spacy英文包：`spacy_en = spacy.load('en')`。最后定义函数：`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe的词向量对于名词词组的存储方式是用“-”连接各个单词，例如词组“new york”在GloVe中的表示为“new-york”。而使用spacy分词之后“new york”的存储可能是“new york”。

* 通过上面三种方法，你能使模型在测试集上的准确率提高到0.85以上吗？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)


![](../img/qr_sentiment-analysis.svg)


## 参考文献

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.
