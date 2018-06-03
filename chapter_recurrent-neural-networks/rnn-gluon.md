# 循环神经网络——使用Gluon

本节介绍如何使用`Gluon`训练循环神经网络。


## Penn Tree Bank (PTB) 数据集

我们以单词为基本元素来训练语言模型。[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42)（PTB）是一个标准的文本序列数据集。它包括训练集、验证集和测试集。

下面我们载入数据集。

```{.python .input  n=1}
import sys
sys.path.append('..')
import gluonbook as gb
import math
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn, utils as gutils
import numpy as np
import time
import zipfile

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')
```

## 建立词语索引

下面定义了`Dictionary`类来映射词语和索引。

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

以下的`Corpus`类按照读取的文本数据集建立映射词语和索引的词典，并将文本转换成词语索引的序列。这样，每个文本数据集就变成了`NDArray`格式的整数序列。

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

## 循环神经网络模型库

我们可以定义一个循环神经网络模型库。这样就可以支持各种不同的循环神经网络模型了。

```{.python .input  n=5}
class RNNModel(nn.Block):
    def __init__(self, mode, vocab_size, embed_size, num_hiddens,
                 num_layers, drop_prob=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = nn.Dropout(drop_prob)
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
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)

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

## 定义参数

我们接着定义模型参数。我们选择使用ReLU为激活函数的循环神经网络为例。这里我们把`epochs`设为1是为了演示方便。


## 多层循环神经网络

我们通过`num_layers`设置循环神经网络隐含层的层数，例如2。

对于一个多层循环神经网络，当前时刻隐含层的输入来自同一时刻输入层（如果有）或上一隐含层的输出。每一层的隐含状态只沿着同一层传递。

把[单层循环神经网络](rnn-scratch.md)中隐含层的每个单元当做一个函数$f$，这个函数在$t$时刻的输入是$\boldsymbol{X}_t, \boldsymbol{H}_{t-1}$，输出是$\boldsymbol{H}_t$：

$$f(\boldsymbol{X}_t, \boldsymbol{H}_{t-1}) = \boldsymbol{H}_t$$

假设输入为第0层，输出为第$L+1$层，在一共$L$个隐含层的循环神经网络中，上式中可以拓展成以下的函数:

$$f(\boldsymbol{H}_t^{(l-1)}, \boldsymbol{H}_{t-1}^{(l)}) = \boldsymbol{H}_t^{(l)}$$

如下图所示。

![](../img/multi-layer-rnn.svg)

```{.python .input  n=6}
model_name = 'rnn_relu'
embed_size = 100
num_hiddens = 100
num_layers = 2
lr = 0.5
clipping_theta = 0.2
num_epochs = 1
batch_size = 32
num_steps = 5
drop_prob = 0.2
eval_period = 500
```

## 批量采样

我们将数据进一步处理为便于相邻批量采样的格式。

```{.python .input  n=7}
ctx = gb.try_gpu()

def batchify(data, batch_size):
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches*batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

train_data = batchify(corpus.train, batch_size).as_in_context(ctx)
val_data = batchify(corpus.valid, batch_size).as_in_context(ctx)
test_data = batchify(corpus.test, batch_size).as_in_context(ctx)

model = RNNModel(model_name, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob)
model.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gloss.SoftmaxCrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 - i)
    X = source[i:i+seq_len]
    Y = source[i+1:i+1+seq_len]
    return X, Y.reshape((-1,))
```

## 从计算图分离隐含状态

在模型训练的每次迭代中，当前批量序列的初始隐含状态来自上一个相邻批量序列的输出隐含状态。为了使模型参数的梯度计算只依赖当前的批量序列，从而减小每次迭代的计算开销，我们可以使用`detach`函数来将隐含状态从计算图分离出来。

```{.python .input  n=8}
def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state
```

## 训练和评价模型

和之前一样，我们定义模型评价函数。

```{.python .input  n=9}
def eval_rnn(data_source):
    l_sum = nd.array([0], ctx=ctx)
    n = 0
    hidden = model.begin_state(func=nd.zeros, batch_size=batch_size, ctx=ctx)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        X, y = get_batch(data_source, i)
        output, hidden = model(X, hidden)
        l = loss(output, y)
        l_sum += l.sum()
        n += l.size
    return l_sum / n
```

最后，我们可以训练模型并在每个epoch评价模型在验证集上的结果。我们可以参考验证集上的结果调参。

```{.python .input  n=10}
def train():
    for epoch in range(1, num_epochs + 1):
        train_l_sum = nd.array([0], ctx=ctx)
        start_time = time.time()
        hidden = model.begin_state(func=nd.zeros, batch_size=batch_size,
                                   ctx=ctx)
        for batch_i, idx in enumerate(range(0, train_data.shape[0] - 1,
                                          num_steps)):
            X, y = get_batch(train_data, idx)
            # 从计算图分离隐藏状态。
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(X, hidden)
                # l 形状：(batch_size * num_steps,)。
                l = loss(output, y).sum() / (batch_size * num_steps)
            l.backward()
            grads = [p.grad(ctx) for p in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
            # 因此我们将 clipping_theta 乘以 num_steps 和 batch_size。
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
train()
test_l = eval_rnn(test_data)
print('test loss %.2f, perplexity %.2f'
      % (test_l.asscalar(), test_l.exp().asscalar()))
```

## 小结

* 我们可以使用Gluon轻松训练各种不同的循环神经网络，并设置网络参数，例如网络的层数。
* 训练迭代中需要将隐含状态从计算图中分离，使模型参数梯度计算只依赖当前的时序数据批量采样。


## 练习

* 调调参数（例如epochs、隐含层的层数、序列长度、隐含状态长度和学习率），看看对运行时间、训练集、验证集和测试集上perplexity造成的影响。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4089)

![](../img/qr_rnn-gluon.svg)
