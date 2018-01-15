# 循环神经网络 --- 使用Gluon

本节介绍如何使用`Gluon`训练循环神经网络。


## Penn Tree Bank (PTB) 数据集

我们以单词为基本元素来训练语言模型。[Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42)（PTB）是一个标准的文本序列数据集。它包括训练集、验证集和测试集。

下面我们载入数据集。

```{.python .input  n=1}
import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

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
        assert os.path.exists(path)
        # 将词语添加至词典。
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # 将文本转换成词语索引的序列（NDArray格式）。
        with open(path, 'r') as f:
            indices = np.zeros((tokens,), dtype='int32')
            idx = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
        return mx.nd.array(indices, dtype='int32')
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
class RNNModel(gluon.Block):
    """循环神经网络模型库"""
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, embed_dim,
                                        weight_initializer=mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='relu',
                                   dropout=dropout, input_size=embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim, num_layers, dropout=dropout,
                                   input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim, num_layers, dropout=dropout,
                                    input_size=embed_dim)
            elif mode == 'gru':
                self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=dropout,
                                   input_size=embed_dim)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)

            self.decoder = nn.Dense(vocab_size, in_units=hidden_dim)
            self.hidden_dim = hidden_dim

    def forward(self, inputs, state):
        emb = self.drop(self.encoder(inputs))
        output, state = self.rnn(emb, state)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## 定义参数

我们接着定义模型参数。我们选择使用ReLU为激活函数的循环神经网络为例。这里我们把`epochs`设为1是为了演示方便。


## 多层循环神经网络

我们通过`num_layers`设置循环神经网络隐含层的层数，例如2。

对于一个多层循环神经网络，当前时刻隐含层的输入来自同一时刻输入层（如果有）或上一隐含层的输出。每一层的隐含状态只沿着同一层传递。

把[单层循环神经网络](rnn-scratch.md)中隐含层的每个单元当做一个函数$f$，这个函数在$t$时刻的输入是$\mathbf{X}_t, \mathbf{H}_{t-1}$，输出是$\mathbf{H}_t$：

$$f(\mathbf{X}_t, \mathbf{H}_{t-1}) = \mathbf{H}_t$$

假设输入为第0层，输出为第$L+1$层，在一共$L$个隐含层的循环神经网络中，上式中可以拓展成以下的函数:

$$f(\mathbf{H}_t^{(l-1)}, \mathbf{H}_{t-1}^{(l)}) = \mathbf{H}_t^{(l)}$$

如下图所示。

![](../img/multi-layer-rnn.svg)

```{.python .input  n=6}
model_name = 'rnn_relu'

embed_dim = 100
hidden_dim = 100
num_layers = 2
lr = 1.0
clipping_norm = 0.2
epochs = 1
batch_size = 32
num_steps = 5
dropout_rate = 0.2
eval_period = 500
```

## 批量采样

我们将数据进一步处理为便于相邻批量采样的格式。

```{.python .input  n=7}
# 尝试使用GPU
import sys
sys.path.append('..')
import utils
context = utils.try_gpu()

def batchify(data, batch_size):
    """数据形状 (num_batches, batch_size)"""
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

train_data = batchify(corpus.train, batch_size).as_in_context(context)
val_data = batchify(corpus.valid, batch_size).as_in_context(context)
test_data = batchify(corpus.test, batch_size).as_in_context(context)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim,
                       num_layers, dropout_rate)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))
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
def model_eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size,
                               ctx=context)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal
```

最后，我们可以训练模型并在每个epoch评价模型在验证集上的结果。我们可以参考验证集上的结果调参。

```{.python .input  n=10}
def train():
    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size,
                                   ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, num_steps)):
            data, target = get_batch(train_data, i)
            # 从计算图分离隐含状态。
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
            # 因此我们将clipping_norm乘以num_steps和batch_size。
            gluon.utils.clip_global_norm(grads,
                                         clipping_norm * num_steps * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = model_eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation ' 
              'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
                                   math.exp(val_L)))
```

训练完模型以后，我们就可以在测试集上评价模型了。

```{.python .input  n=11}
train()
test_L = model_eval(test_data)
print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
```

## 结论

* 我们可以使用Gluon轻松训练各种不同的循环神经网络，并设置网络参数，例如网络的层数。
* 训练迭代中需要将隐含状态从计算图中分离，使模型参数梯度计算只依赖当前的时序数据批量采样。


## 练习

* 调调参数（例如epochs、隐含层的层数、序列长度、隐含状态长度和学习率），看看对运行时间、训练集、验证集和测试集上perplexity造成的影响。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4089)
