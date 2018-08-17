# 文本情感分类：使用卷积神经网络（textCNN）

TODO(@astonzhang): edits

在之前的章节中介绍了卷积神经网络用于计算机视觉领域。
在本节将介绍如何将卷积神经网络应用于自然语言处理领域。以及参考textCNN模型使用Gluon创建一个卷积神经网络用于文本情感分类。

## 模型设计

卷积神经网络在自然语言处理上的应用，可以类比其图像任务上的应用，即把一个文本用二维图像的方式来表达。每个文本是一个矩阵，将文本中每个词的词向量按顺序纵向排列，即这个矩阵的每一行分别是一个词向量。

在卷积层中，使用不同的卷积核获取不同窗口大小内词的关系；而与计算机视觉中的二维卷积不同的是，自然语言处理任务中一般用的是一维卷积，即卷积核的宽度是词嵌入的维度。因为我们需要获取的是不同窗口内的词所带来的信息。然后，我们应用一个最大池化层，这里采用的是Max-over-time pooling，即对一个feature map选取一个最大值保留，这个最大值可以理解为是这个feature map最重要的特征。将这些取到的最大值连结成一个向量。而由于只取最大值，在做padding时补0，并不会影响结果。

最后，我们将连结得到的向量通过全连接层变换为输出。我们在全连接层前加一个Dropout层，用于减轻过拟合。

![](../img/textcnn.svg)


我们来描述一下这个过程：
1. 我们假设有一个文本，长度 n 为 11 ，词嵌入维度为 7 。此时词嵌入矩阵维度为（11， 7）。
2. 设有三组卷积核，卷积核的宽度为7（词嵌入的维度），卷积宽度f分别是2、3、4，卷积核的数目分别为 4、4、5 。卷积后得到的矩阵维度分别是，（10，4）、（9，4）、（8，5）。即（n-f+1，nums_channels）
3. 再进行 Max-over-time pooling，得到的矩阵维度分别是(4，1)、(4，1)、(5，1)。
4. 压平上述三个矩阵，并连结，得到一个（4+4+5）维度的向量
5. 再通过一个全连接层降低维度。

在实验开始前，导入所需的包或模块。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

import collections
import gluonbook as gb
from mxnet import autograd, gluon, init, metric, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
import os
import random
from time import time
import tarfile
```

## 读取IMDb数据集

我们使用Stanford's Large Movie Review Dataset作为情感分析的数据集 [1]。它的下载地址是

> http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 。

这个数据集分为训练和测试用的两个数据集，分别有25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”（1）和“负面”（0）的评论数量相等。
我们首先下载这个数据集到`../data`下。压缩包大小是 81MB，下载解压需要一定时间。解压之后这个数据集将会放置在`../data/aclImdb`下。

```{.python .input  n=2}
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

```{.python .input  n=3}
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

```{.python .input  n=5}
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

下面，我们继续对数据进行预处理。每个不定长的评论将被特殊符号`PAD`补成长度为`maxlen`的序列，并用NDArray表示。在这里由于模型使用了最大池化层，只取卷积后最大的一个值，所以补0不会对结果产生影响。

```{.python .input  n=6}
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
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）。
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features

train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 1000, 0))
test_features = nd.array(pad_samples(test_features, 1000, 0))
train_labels = nd.array([score for _, score in train_data])
test_labels = nd.array([score for _, score in test_data])
```

## 加载预训练的词向量

这里，我们为词典`vocab`中的每个词加载预训练的GloVe词向量（每个词向量长度为100）。稍后，我们将用这些词向量作为评论中每个词的特征向量。

```{.python .input  n=7}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

## 一维卷积

一维卷积即将一个一维核（kernel）数组作用在一个一维输入数据上来计算一个一维数组输出。下图演示了如何对一个长度为7的输入X作用宽为2的核K来计算输出Y。

![](../img/Conv1D.svg)

可以看到输出Y是一个6维的向量，且第一个元素是由X的最左侧宽为2的子数组与核数组按元素相乘后再相加得来。即Y[0] = (X[0:2] * K).sum()。卷积后输出的数据维度仍遵循n-f+1，即 7-2+1=6

一维卷积常用于序列模型，比如自然语言处理领域中。

下面我们将上述过程实现在corr1d函数里，它接受X和K，输出Y。

```{.python .input  n=8}
def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i : i + w] * K).sum()
    return Y
```

构造上图中的数据来测试实现的正确性。

```{.python .input  n=9}
X = nd.array([0 ,1 ,2, 3, 4 ,5 ,6])
K = nd.array([1 ,2])
corr1d(X, K)
```

一维卷积多通道输入的卷积运算与二维卷积的多通道运算类似。将每个单通道与对应的filter进行卷积运算求和，然后再将多个通道的和相加，得到输出的一个数值。

![](../img/Conv1D-channel.svg)

解释上图，假设存在三个通道$ c_0, c_1, c_2 $，存在一组卷积核$ k_0, k_1, k_2 $

$$ y(i)=\sum_m c_0(i-m)k_0(m) + \sum_m c_1(i-m)k_1(m) + \sum_m c_2(i-m)k_2(m) \\
=\sum_m \sum_{n\in\{0, 1, 2\}} c_n(i-m)k_n(m) $$

![](../img/Conv1D-flatten.svg)

我们将$ c_0, c_1, c_2 $三个向量连结成矩阵C，将$ k_0, k_1, k_2 $连结成矩阵K

$$ y(i)=\sum_m \sum_{n\in\{0, 1, 2\}} C(i-m,j-n)K(m,n) $$

上式与二维卷积的定义等价。

故：多通道一维卷积计算可以等价于单通道二维卷积计算。

类比到图像上，我们可以用一个三维的向量（R, G, B）来表达一个像素点。在做卷积时将R、G、B作为三个通道来进行运算。

在文本任务上，我们可以用一个k维的向量来表达一个词，即词向量。这个k即嵌入层维度embed_size。同样的，在做卷积时也将这k维作为k个通道来进行计算。

所以，对于自然语言处理任务而言，输入的通道数等于嵌入层维度embed_size。

## 定义模型

下面我们根据模型设计里的描述定义情感分类模型。其中的`Embedding`实例即嵌入层，在实验中，我们使用了两个嵌入层。`Conv1D`实例即为卷积层，`GlobalMaxPool1D`实例为池化层，卷积层和池化层用于抽取文本中重要的特征。`Dense`实例即生成分类结果的输出层。

```{.python .input  n=10}
class TextCNN(nn.Block):
    def __init__(self, vocab, embedding_size, ngram_kernel_sizes,
                 nums_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.ngram_kernel_sizes = ngram_kernel_sizes
        self.nums_channels = nums_channels
        self.embedding_static = nn.Embedding(len(vocab), embedding_size)
        self.embedding_non_static = nn.Embedding(len(vocab), embedding_size)
        for i in range(len(ngram_kernel_sizes)):
            conv = nn.Conv1D(nums_channels[i],
                kernel_size=ngram_kernel_sizes[i],
                strides=1,
                activation='relu')
            pool = nn.GlobalMaxPool1D()
            setattr(self, 'conv_{i}', conv)  #将self.conv_{i}置为第i个conv
            setattr(self, 'pool_{i}', pool)  #将self.pool_{i}置为第i个pool
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(num_outputs)
    def forward(self, inputs):
        #inputs 输入的维度为(batch_size, 句子长度) ，转换为(句子长度, batch_size)
        inputs = inputs.T
        #embeddings_static 的维度为（句子长度, batch_size, embed_size） 
        embeddings_static = self.embedding_static(inputs)
        
        #Conv1D 要求的输入是（batch_size, in_channels, width），故需要做transpose((1,2,0))
        embeddings_static = embeddings_static.transpose((1,2,0))  
        
        #embeddings_non_static 的维度同上
        embeddings_non_static = self.embedding_non_static(inputs).transpose((1,2,0))
        
        #按照模型设计，每个卷积核都应用于两个嵌入层，此时卷积核为的核数组。将卷积核在多个嵌入层的运算结果求和，即得到一次卷积结果。
        #这等价于直接连结这两个嵌入层，再将卷积核变成（ngram_kernel_sizes, nums_channels * 2 ），所得结果相同。
        #dim = 1 的意思是从 in_channels 这个维度进行连结。连结后的维度是（batch_size, in_channels*2, width）
        embeddings = nd.concat(embeddings_static, embeddings_non_static, dim=1)
        
        #对于卷积核[i]，在池化之后会形成一个（nums_channels[i],1）的矩阵，需要使用flatten压平成nums_channels[i]维的向量
        encoding = [
            nd.flatten(self.get_pool(i)(self.get_conv(i)(embeddings)))
            for i in range(len(self.ngram_kernel_sizes))]
        
        #在此之前， encoding 有多个元素，每个元素维度是（batch_size, nums_channels[i]）。
        #需要将它连结成一个维度为（batch_size, nums_channels的和）的矩阵
        encoding = nd.concat(*encoding, dim=1)
        
        outputs = self.decoder(self.dropout(encoding))
        return outputs
    #调用self.get_conv(i)，即返回self.conv_{i}
    def get_conv(self, i):
        return getattr(self, 'conv_{i}')
    #调用self.get_pool(i)，即返回self.pool_{i}
    def get_pool(self, i):
        return getattr(self, 'pool_{i}')
```

我们使用在更大规模语料上预训练的词向量作为每个词的特征向量。本实验有两个嵌入层，其中嵌入层`Embedding_non_static`的词向量可以在训练过程中被更新，另一个嵌入层`Embedding_static`的词向量在训练过程中不能被更新。

```{.python .input  n=11}
num_outputs = 2
lr = 0.001
num_epochs = 5
batch_size = 64
embed_size = 100
ngram_kernel_sizes = [3, 4, 5]
nums_channels = [100, 100, 100]
ctx = gb.try_all_gpus()

net = TextCNN(vocab, embed_size, ngram_kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)
# 设置两个embedding 层的 weight 为预训练的词向量。
net.embedding_static.weight.set_data(glove_embedding.idx_to_vec)
net.embedding_non_static.weight.set_data(glove_embedding.idx_to_vec)
# 训练中不更新embedding_static的词向量（net.embedding中的模型参数）。
net.embedding_static.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练并评价模型

使用gluon的DataLoader加载数据

```{.python .input  n=30}
train_set = gdata.ArrayDataset(train_features, train_labels)
test_set = gdata.ArrayDataset(test_features, test_labels)
train_loader = gdata.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True)
test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

下面开始训练模型。

```{.python .input}
gb.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs)
```

下面我们使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input}
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

* 我们可以使用一维卷积来处理时序序列任务，如自然语言处理。

* 多通道一维卷积运算可以等价于单通道二维卷积计算。


## 练习

* 使用IMDb完整数据集，把迭代周期改为 5。你的模型能在训练和测试数据集上得到怎样的准确率？通过调节超参数，你能进一步提升分类准确率吗？

* 使用更大的预训练词向量，例如300维的GloVe词向量，能否提升分类准确率？

* 使用spacy分词工具，能否提升分类准确率？。你需要安装spacy：`pip install spacy`，并且安装英文包：`python -m spacy download en`。在代码中，先导入spacy：`import spacy`。然后加载spacy英文包：`spacy_en = spacy.load('en')`。最后定义函数：`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe的词向量对于名词词组的存储方式是用“-”连接各个单词，例如词组“new york”在GloVe中的表示为“new-york”。而使用spacy分词之后“new york”的存储可能是“new york”。

* 通过上面三种方法，你能使模型在测试集上的准确率提高到0.87以上吗？




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)


![](../img/qr_sentiment-analysis-cnn.svg)


## 参考文献

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
