# 文本情感分类：使用循环神经网络

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本节关注它的一个子问题：使用文本情感分类来分析文本作者的情绪。这个问题也叫情感分析（sentiment analysis），并有着广泛的应用。例如，我们可以分析用户对产品的评论并统计用户的满意度，或者分析用户对市场行情的情绪并用以预测接下来的行情。

同求近义词和类比词一样，文本分类也属于词嵌入的下游应用。本节将应用预训练的词向量和含多个隐藏层的双向循环神经网络，来判断一段不定长的文本序列中包含的是正面还是负面的情绪。

在实验开始前，导入所需的包或模块。

```{.python .input  n=1}
import collections
import d2lzh as d2l
from d2lzh import text
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
```

## 文本情感分类数据集

我们使用斯坦福的IMDb数据集（Stanford's Large Movie Review Dataset）作为文本情感分类的数据集 [1]。这个数据集分为训练和测试用的两个数据集，分别包含25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”和“负面”的评论数量相等。

###  读取数据集

首先下载这个数据集到`../data`路径下，然后解压至`../data/aclImdb`路径下。

```{.python .input  n=3}
# 本函数已保存在d2lzh包中方便以后使用
def download_imdb(data_dir='../data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)

download_imdb()
```

接下来，读取训练数据集和测试数据集。每个样本是一条评论及其对应的标签：1表示“正面”，0表示“负面”。

```{.python .input  n=13}
def read_imdb(folder='train'):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
```

### 预处理数据集

我们需要对每条评论做分词，从而得到分好词的评论。这里定义的`get_tokenized_imdb`函数使用最简单的方法：基于空格进行分词。

```{.python .input  n=14}
def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
```

现在，我们可以根据分好词的训练数据集来创建词典了。我们在这里过滤掉了出现次数少于5的词。

```{.python .input  n=28}
def get_vocab_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5,
                                 reserved_tokens=['<pad>'])

vocab = get_vocab_imdb(train_data)
'# words in vocab:', len(vocab)
```

因为每条评论长度不一致所以不能直接组合成小批量，我们定义`preprocess_imdb`函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补“&lt;pad&gt;”（padding）符号来将每条评论长度固定成500。

```{.python .input  n=44}
def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh包中方便以后使用
    max_l = 500  # 将每条评论通过截断或者补'<pad>'，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels
```

### 创建数据迭代器

现在，我们创建数据迭代器。每次迭代将返回一个小批量的数据。

```{.python .input}
batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

打印第一个小批量数据的形状以及训练集中小批量的个数。

```{.python .input}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)
```

## 使用循环神经网络的模型

在这个模型中，每个词先通过嵌入层得到特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出。具体来说，我们可以将双向长短期记忆在最初时间步和最终时间步的隐状态连结，作为特征序列的表征传递给输出层分类。在下面实现的`BiRNN`类中，`Embedding`实例即嵌入层，`LSTM`实例即为序列编码的隐藏层，`Dense`实例即生成分类结果的输出层。

```{.python .input  n=46}
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs
```

创建一个含两个隐藏层的双向循环神经网络。

```{.python .input}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
```

### 加载预训练的词向量

由于情感分类的训练数据集并不是很大，为应对过拟合，我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。这里，我们为词典`vocab`中的每个词加载100维的GloVe词向量。

```{.python .input  n=45}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

然后，我们将用这些词向量作为评论中每个词的特征向量。注意，预训练词向量的维度需要与创建的模型中的嵌入层输出大小`embed_size`一致。此外，在训练中我们不再更新这些词向量。

```{.python .input  n=47}
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')
```

### 训练模型

这时候就可以开始训练模型了。

```{.python .input  n=48}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

最后，定义预测函数。

```{.python .input  n=49}
# 本函数已保存在d2lzh包中方便以后使用
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
```

下面使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input  n=50}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## 小结

* 文本分类把一段不定长的文本序列变换为文本的类别。它属于词嵌入的下游应用。
* 可以应用预训练的词向量和循环神经网络对文本的情感进行分类。


## 练习

* 增加迭代周期。训练后的模型能在训练和测试数据集上得到怎样的精度？再调节其他超参数试试？

* 使用更大的预训练词向量，如300维的GloVe词向量，能否提升分类精度？

* 使用spaCy分词工具，能否提升分类精度？你需要安装spaCy（`pip install spacy`），并且安装英文包（`python -m spacy download en`）。在代码中，先导入spaCy（`import spacy`）。然后加载spaCy英文包（`spacy_en = spacy.load('en')`）。最后定义函数`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`并替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe词向量对于名词词组的存储方式是用“-”连接各个单词，例如，词组“new york”在GloVe词向量中的表示为“new-york”，而使用spaCy分词之后“new york”的存储可能是“new york”。






## 参考文献

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)

![](../img/qr_sentiment-analysis.svg)
