
# 文本分类：情感分析

在之前的章节中介绍了卷积神经网络用于计算机视觉领域。
在本节将介绍如何将卷积神经网络应用于自然语言处理领域。以及参考textCNN模型使用Gluon创建一个卷积神经网络用于文本情感分类。

## 模型设计

卷积神经网络用于自然语言处理，可以将理解为把一个文本用二维图像的方式来表达，每一行是一个词向量，将每一个词纵向排列。

![](../img/embedding.svg)


在卷积层中，使用不同的卷积核获取不同窗口大小内词的关系；而与计算机视觉中的二维卷积不同的是，自然语言处理任务中一般用的是一维卷积，即卷积核的的宽度是词嵌入的维度。因为我们需要获取的是不同窗口内的词所带来的信息。然后，我们应用一个最大池化层，这里采用的是`Max-over-time pooling`，即对一个feature map选取一个最大值保留，这个最大值可以理解为是这个feature map最重要的特征。将这些取到的最大值拼接成一个向量。

![](../img/textcnn.svg)

最后，我们将拼接得到的向量通过全连接层变换为输出。我们在全连接层前加一个Dropout层，用于减轻过拟合。

在实验开始前，导入所需的包或模块。


```
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

    /home/ubuntu/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py:1067: UserWarning: Duplicate key in file "/home/ubuntu/.config/matplotlib/matplotlibrc", line #2
      (fname, cnt))
    /home/ubuntu/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py:1067: UserWarning: Duplicate key in file "/home/ubuntu/.config/matplotlib/matplotlibrc", line #3
      (fname, cnt))
    /home/ubuntu/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


## 读取IMDb数据集

我们使用Stanford's Large Movie Review Dataset作为情感分析的数据集 [1]。它的下载地址是

> http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 。

这个数据集分为训练和测试用的两个数据集，分别有25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”（1）和“负面”（0）的评论数量相等。将下载好的数据解压并存放在路径“../data/aclImdb”。

为方便快速上手，我们提供了上述数据集的小规模采样，并存放在路径“../data/aclImdb_tiny.zip”。如果你将使用上述的IMDb完整数据集，还需要把下面`demo`变量改为`False`。


```
# 如果使用下载的 IMDb 的完整数据集，把下面改为 False。
demo = True
if demo:
    with zipfile.ZipFile('../data/aclImdb_tiny.zip', 'r') as zin:
        zin.extractall('../data/')
```

下面，读取训练和测试数据集。


```
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


```
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


```
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


```
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

ctx = gb.try_gpu()
train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 1000, 0), ctx=ctx)
test_features = nd.array(pad_samples(test_features, 1000, 0), ctx=ctx)
train_labels = nd.array([score for _, score in train_data], ctx=ctx)
test_labels = nd.array([score for _, score in test_data], ctx=ctx)
```

## 加载预训练的词向量

这里，我们为词典`vocab`中的每个词加载GloVe词向量（每个词向量长度为100）。稍后，我们将用这些词向量作为评论中每个词的特征向量。


```
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

## 定义模型

下面我们根据模型设计里的描述定义情感分类模型。其中的`Embedding`实例即嵌入层，在实验中，我们使用了两个嵌入层。`Conv1D`实例即为卷积层，`GlobalMaxPool1D`实例为池化层，卷积层和池化层用于抽取文本中重要的特征。`Dense`实例即生成分类结果的输出层。


```
class TextCNN(nn.Block):
    def __init__(self, vocab, embedding_size, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.embedding_static = nn.Embedding(len(vocab), embedding_size)
        self.embedding_non_static = nn.Embedding(len(vocab), embedding_size)
        for i in range(len(self.FILTERS)):
            conv = nn.Conv1D(self.FILTER_NUM[i], kernel_size=self.FILTERS[i], strides=1, activation='relu')
            pool = nn.GlobalMaxPool1D()
            setattr(self, f'conv_{i}', conv)
            setattr(self, f'pool_{i}', pool)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(num_outputs)
    def forward(self, inputs):
        embeddings_static = self.embedding_static(inputs).transpose((1,2,0))
        embeddings_non_static = self.embedding_non_static(inputs).transpose((1,2,0))
        embeddings = nd.concat(embeddings_static,embeddings_non_static,dim=1)
        encoding = [
            nd.flatten(self.get_pool(i)(self.get_conv(i)(embeddings)))
            for i in range(len(self.FILTERS))]
        encoding = nd.concat(*encoding, dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
    def get_pool(self, i):
        return getattr(self, f'pool_{i}')
```

我们使用在更大规模语料上预训练的词向量作为每个词的特征向量。本实验有两个嵌入层，其中嵌入层`Embedding_non_static`的词向量可以在训练过程中被更新，另一个嵌入层`Embedding_static`的词向量在训练过程中不能被更新。


```
num_outputs = 2
lr = 0.01
num_epochs = 1
batch_size = 10
embed_size = 100
    
net = TextCNN(vocab, embed_size)
net.initialize(init.Xavier(), ctx=ctx)
# 设置两个embedding 层的 weight 为预训练的词向量。
net.embedding_static.weight.set_data(glove_embedding.idx_to_vec.as_in_context(ctx))
net.embedding_non_static.weight.set_data(glove_embedding.idx_to_vec.as_in_context(ctx))
# 训练中不更新embedding_static的词向量（net.embedding中的模型参数）。
net.embedding_non_static.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练并评价模型

在实验中，我们使用准确率作为评价模型的指标。


```
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


```
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

    epoch 1, train loss 1.291998, acc 0.75; test loss 5.334179, acc 0.50


## 小结

* 我们可以应用卷积神经网络对文本进行情感分析。


## 练习

* 使用IMDb完整数据集，并把迭代周期改为10。你的模型能在训练和测试数据集上得到怎样的准确率？通过调节超参数，你能进一步提升分类准确率吗？

* 使用更大的预训练词向量，例如300维的GloVe词向量，能否提升分类准确率？

* 使用spacy分词工具，能否提升分类准确率？。你需要安装spacy：`pip install spacy`，并且安装英文包：`python -m spacy download en`。在代码中，先导入spacy：`import spacy`。然后加载spacy英文包：`spacy_en = spacy.load('en')`。最后定义函数：`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe的词向量对于名词词组的存储方式是用“-”连接各个单词，例如词组“new york”在GloVe中的表示为“new-york”。而使用spacy分词之后“new york”的存储可能是“new york”。

* 通过上面三种方法，你能使模型在测试集上的准确率提高到0.87以上吗？




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)


![](../img/qr_sentiment-analysis.svg)


## 参考文献

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. Eprint Arxiv.
