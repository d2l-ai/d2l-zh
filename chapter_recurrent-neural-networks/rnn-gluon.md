# 循环神经网络的Gluon实现

本节介绍如何使用Gluon训练循环神经网络。


## Penn Tree Bank数据集

我们以英文单词为单元来训练基于循环神经网络的语言模型。Penn Tree Bank（PTB）是一个标准的文本序列数据集 [1]。它包括训练集、验证集和测试集 。

首先导入实验所需的包或模块，并抽取数据集。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn, utils as gutils
import numpy as np
import time
import zipfile

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')
```

## 建立词语索引

下面定义了`Dictionary`类来映射词语和整数索引。

```{.python .input  n=2}
class Dictionary(object):
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.idx_to_word.append(word)
            self.word_to_idx[word] = len(self.idx_to_word) - 1
        return self.word_to_idx[word]

    def __len__(self):
        return len(self.idx_to_word)
```

以下的`Corpus`类按照读取的文本数据集建立映射词语和索引的词典，并将文本转换成词语索引的序列。这样，每个文本数据集就变成了NDArray格式的整数序列。

```{.python .input  n=3}
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        # 将词语添加至词典。
        with open(path, 'r') as f:
            num_words = 0
            for line in f:
                words = line.split() + ['<eos>']
                num_words += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # 将文本转换成词语索引的序列（ NDArray 格式）。
        with open(path, 'r') as f:
            indices = np.zeros((num_words,), dtype='int32')
            idx = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
        return nd.array(indices, dtype='int32')
```

看一下词典的大小。

```{.python .input  n=4}
data = '../data/ptb/ptb.'
corpus = Corpus(data)
vocab_size = len(corpus.dictionary)
vocab_size
```

## 定义循环神经网络模型库

我们可以定义一个循环神经网络模型库。这样我们就可以使用以ReLU或tanh函数为激活函数的循环神经网络，以及长短期记忆和门控循环单元。和本章中其他实验不同，这里使用了Embedding实例将每个词索引变换成一个长度为`embed_size`的词向量。这些词向量实际上也是模型参数。在随机初始化后，它们会在模型训练结束时被学到。此外，我们使用了丢弃法来应对过拟合。

这里的Embedding实例也叫嵌入层。我们会在“自然语言处理”篇章介绍如何用在大规模语料上预训练的词向量初始化嵌入层参数。

```{.python .input  n=5}
class RNNModel(nn.Block):
    def __init__(self, mode, vocab_size, embed_size, num_hiddens,
                 num_layers, drop_prob=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = nn.Dropout(drop_prob)
            # 将词索引变换成词向量。这些词向量也是模型参数。
            self.embedding = nn.Embedding(
                vocab_size, embed_size, weight_initializer=init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hiddens, num_layers, activation='relu',
                                   dropout=drop_prob, input_size=embed_size)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hiddens, num_layers, activation='tanh',
                                   dropout=drop_prob, input_size=embed_size)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hiddens, num_layers,
                                    dropout=drop_prob, input_size=embed_size)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                                   input_size=embed_size)
            else:
                raise ValueError('Invalid mode %s. Options are rnn_relu, '
                                 'rnn_tanh, lstm, and gru' % mode)
            self.dense = nn.Dense(vocab_size, in_units=num_hiddens)
            self.num_hiddens = num_hiddens

    def forward(self, inputs, state):
        embedding = self.dropout(self.embedding(inputs))
        output, state = self.rnn(embedding, state)
        output = self.dropout(output)
        output = self.dense(output.reshape((-1, self.num_hiddens)))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## 设置超参数

我们接着设置超参数。这里选择使用以ReLU为激活函数的循环神经网络。它包含2个隐藏层。为了得到更好的实验结果，这些超参数还需要重新设置。

```{.python .input  n=6}
model_name = 'rnn_relu'
embed_size = 100
num_hiddens = 100
num_layers = 2
lr = 0.5
clipping_theta = 0.2
num_epochs = 2
batch_size = 32
num_steps = 5
drop_prob = 0.2
eval_period = 1000

ctx = gb.try_gpu()
model = RNNModel(model_name, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob)
model.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 相邻采样

我们将在实验中使用相邻采样。

```{.python .input  n=7}
def batchify(data, batch_size):
    num_batches = data.shape[0] // batch_size
    data = data[: num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

train_data = batchify(corpus.train, batch_size).as_in_context(ctx)
val_data = batchify(corpus.valid, batch_size).as_in_context(ctx)
test_data = batchify(corpus.test, batch_size).as_in_context(ctx)

def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 - i)
    X = source[i : i + seq_len]
    Y = source[i + 1 : i + 1 + seq_len]
    return X, Y.reshape((-1,))
```

[“循环神经网络”](rnn.md)一节里已经解释了，相邻采样应在每次读取小批量前将隐藏状态从计算图分离出来。

```{.python .input  n=8}
def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state
```

## 训练和评价模型

以下定义了模型评价函数。

```{.python .input  n=9}
def eval_rnn(data_source):
    l_sum = nd.array([0], ctx=ctx)
    n = 0
    state = model.begin_state(func=nd.zeros, batch_size=batch_size, ctx=ctx)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        X, y = get_batch(data_source, i)
        output, state = model(X, state)
        l = loss(output, y)
        l_sum += l.sum()
        n += l.size
    return l_sum / n
```

下面的`train_rnn`函数将训练模型并在每个迭代周期结束时评价模型在验证集上的表现。我们可以参考验证集上的结果调节超参数。

```{.python .input  n=10}
def train_rnn():
    for epoch in range(1, num_epochs + 1):
        train_l_sum = nd.array([0], ctx=ctx)
        start_time = time.time()
        state = model.begin_state(func=nd.zeros, batch_size=batch_size,
                                   ctx=ctx)
        for batch_i, idx in enumerate(range(0, train_data.shape[0] - 1,
                                          num_steps)):
            X, y = get_batch(train_data, idx)
            # 从计算图分离隐藏状态变量（包括 LSTM 的记忆细胞）。
            state = detach(state)
            with autograd.record():
                output, state = model(X, state)
                # l 形状：(batch_size * num_steps,)。
                l = loss(output, y).sum() / (batch_size * num_steps)
            l.backward()
            grads = [p.grad(ctx) for p in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。 因此我们将
            # clipping_theta 乘以 num_steps 和 batch_size。
            gutils.clip_global_norm(
                grads, clipping_theta * num_steps * batch_size)
            trainer.step(1)
            train_l_sum += l
            if batch_i % eval_period == 0 and batch_i > 0:
                cur_l = train_l_sum / eval_period
                print('epoch %d, batch %d, train loss %.2f, perplexity %.2f'
                      % (epoch, batch_i, cur_l.asscalar(),
                         cur_l.exp().asscalar()))
                train_l_sum = nd.array([0], ctx=ctx)
        val_l = eval_rnn(val_data)
        print('epoch %d, time %.2fs, valid loss %.2f, perplexity %.2f'
              % (epoch, time.time() - start_time, val_l.asscalar(),
                 val_l.exp().asscalar()))
```

训练完模型以后，我们就可以在测试集上评价模型了。

```{.python .input  n=11}
train_rnn()
test_l = eval_rnn(test_data)
print('test loss %.2f, perplexity %.2f'
      % (test_l.asscalar(), test_l.exp().asscalar()))
```

## 小结

* 我们可以使用Gluon训练循环神经网络。它更简洁，例如无需我们手动实现含有多个隐藏层的复杂模型。
* 在训练语言模型时，我们可以将词索引变换成词向量，并将这些词向量视为模型参数。


## 练习

* 回忆[“模型参数的访问、初始化和共享”](../chapter_deep-learning-computation/parameters.md)一节中有关共享模型参数的描述。将本节中RNNModel类里的`self.dense`的定义改为`nn.Dense(vocab_size, in_units = num_hiddens, params=self.embedding.params)`并运行本节实验。这里为什么可以共享词向量参数？有哪些好处？

* 调调超参数，观察并分析对运行时间以及训练集、验证集和测试集上困惑度的影响。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4089)

![](../img/qr_rnn-gluon.svg)

## 参考文献

[1] Penn Tree Bank. https://catalog.ldc.upenn.edu/ldc99t42
