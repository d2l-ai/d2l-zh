# 神经机器翻译

本节介绍[编码器—解码器和注意力机制](seq2seq-attention.md)的应用。我们以神经机器翻译（neural machine translation）为例，介绍如何使用Gluon实现一个简单的编码器—解码器和注意力机制模型。


## 使用Gluon实现编码器—解码器和注意力机制

我们先载入需要的包。

```{.python .input}
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, rnn, Block
from mxnet.contrib import text

from io import open
import collections
import datetime
```

下面定义一些特殊字符。其中PAD (padding)符号使每个序列等长；BOS (beginning of sequence)符号表示序列的开始；而EOS (end of sequence)符号表示序列的结束。

```{.python .input}
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
```

以下是一些可以调节的模型参数。我们在编码器和解码器中分别使用了一层和两层的循环神经网络。

```{.python .input}
epochs = 50
epoch_period = 10

learning_rate = 0.005
# 输入或输出序列的最大长度（含句末添加的EOS字符）。
max_seq_len = 5

encoder_num_layers = 1
decoder_num_layers = 2

encoder_drop_prob = 0.1
decoder_drop_prob = 0.1

encoder_hidden_dim = 256
decoder_hidden_dim = 256
alignment_dim = 25

ctx = mx.cpu(0)
```

### 读取数据

我们定义函数读取训练数据集。为了减少运行时间，我们使用一个很小的法语——英语数据集。

这里使用了[之前章节](pretrained-embedding.md)介绍的`mxnet.contrib.text`来创建法语和英语的词典。需要注意的是，我们会在句末附上EOS符号，并可能通过添加PAD符号使每个序列等长。

```{.python .input}
def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []

    with open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:
            input_seq, output_seq = line.rstrip().split('\t')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')

            if len(cur_input_tokens) < max_seq_len and \
                            len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                # 句末附上EOS符号。
                cur_input_tokens.append(EOS)
                # 添加PAD符号使每个序列等长（长度为max_seq_len）。
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)
                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(PAD)
                output_seqs.append(cur_output_tokens)

        fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
        en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab, en_vocab, input_seqs, output_seqs
```

以下创建训练数据集。每一个样本包含法语的输入序列和英语的输出序列。

```{.python .input}
input_vocab, output_vocab, input_seqs, output_seqs = read_data(max_seq_len)
X = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
Y = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    X[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    Y[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)

dataset = gluon.data.ArrayDataset(X, Y)
```

### 编码器、含注意力机制的解码器和解码器初始状态

以下定义了基于[GRU](../chapter_recurrent-neural-networks/gru-scratch.md)的编码器。

```{.python .input}
class Encoder(Block):
    """编码器"""
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=drop_prob,
                               input_size=hidden_dim)

    def forward(self, inputs, state):
        # inputs尺寸: (1, num_steps)，emb尺寸: (num_steps, 1, 256)
        emb = self.embedding(inputs).swapaxes(0, 1)
        emb = self.dropout(emb)
        output, state = self.rnn(emb, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

以下定义了基于[GRU](../chapter_recurrent-neural-networks/gru-scratch.md)的解码器。它包含[上一节里介绍的注意力机制](seq2seq-attention.md)的实现。

```{.python .input}
class Decoder(Block):
    """含注意力机制的解码器"""
    def __init__(self, hidden_dim, output_dim, num_layers, max_seq_len,
                 drop_prob, alignment_dim, encoder_hidden_dim):
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        with self.name_scope():
            self.embedding = nn.Embedding(output_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)
            # 注意力机制。
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(nn.Dense(
                    alignment_dim, in_units=hidden_dim + encoder_hidden_dim,
                    activation="tanh", flatten=False))
                self.attention.add(nn.Dense(1, in_units=alignment_dim,
                                            flatten=False))

            self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=drop_prob,
                               input_size=hidden_dim)
            self.out = nn.Dense(output_dim, in_units=hidden_dim)
            self.rnn_concat_input = nn.Dense(
                hidden_dim, in_units=hidden_dim + encoder_hidden_dim,
                flatten=False)

    def forward(self, cur_input, state, encoder_outputs):
        # 当RNN为多层时，取最靠近输出层的单层隐含状态。
        single_layer_state = [state[0][-1].expand_dims(0)]
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, 1,
                                                   self.encoder_hidden_dim))
        # hidden尺寸: [(1, 1, decoder_hidden_dim)]
        # hidden_broadcast尺寸: (max_seq_len, 1, decoder_hidden_dim)
        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)

        # encoder_outputs_and_hiddens尺寸:
        # (max_seq_len, 1, encoder_hidden_dim + decoder_hidden_dim)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)

        # energy尺寸: (max_seq_len, 1, 1)
        energy = self.attention(encoder_outputs_and_hiddens)

        batch_attention = nd.softmax(energy, axis=0).reshape(
            (1, 1, self.max_seq_len))

        # batch_encoder_outputs尺寸: (1, max_seq_len, encoder_hidden_dim)
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)

        # decoder_context尺寸: (1, 1, encoder_hidden_dim)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)

        # input_and_context尺寸: (1, 1, encoder_hidden_dim + decoder_hidden_dim)
        input_and_context = nd.concat(self.embedding(cur_input).reshape(
            (1, 1, self.hidden_size)), decoder_context, dim=2)
        # concat_input尺寸: (1, 1, decoder_hidden_dim)
        concat_input = self.rnn_concat_input(input_and_context)
        concat_input = self.dropout(concat_input)

        # 当RNN为多层时，用单层隐含状态初始化各个层的隐含状态。
        state = [nd.broadcast_axis(single_layer_state[0], axis=0,
                                   size=self.num_layers)]

        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        output = self.out(output)
        # output尺寸: (1, output_size)，hidden尺寸: [(1, 1, decoder_hidden_dim)]
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

为了初始化解码器的隐含状态，我们通过一层全连接网络来转化编码器的输出隐含状态。

```{.python .input}
class DecoderInitState(Block):
    """解码器隐含状态的初始化"""
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(DecoderInitState, self).__init__()
        with self.name_scope():
            self.dense = nn.Dense(decoder_hidden_dim,
                                  in_units=encoder_hidden_dim,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]
```

### 训练和应用模型

我们定义`translate`函数来应用训练好的模型。这些模型通过该函数的前三个参数传递。解码器的最初时刻输入来自BOS字符。当任一时刻的输出为EOS字符时，输出序列即完成。

```{.python .input}
def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('Input :', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]
        # 添加PAD符号使每个序列等长（长度为max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=mx.nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),
                                                 encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # 编码器的第一个输入为BOS字符。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for i in range(max_seq_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy())
            # 当任一时刻的输出为EOS字符时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)

        print('Output:', ' '.join(output_tokens))
        print('Expect:', fr_en[1], '\n')
```

下面定义模型训练函数。为了初始化解码器的隐含状态，我们通过一层全连接网络来转化编码器最早时刻的输出隐含状态。

```{.python .input}
def train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens):
    # 对于三个网络，分别初始化它们的模型参数并定义它们的优化器。
    encoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder_init_state.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam',
        {'learning_rate': learning_rate})

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    prev_time = datetime.datetime.now()
    data_iter = gluon.data.DataLoader(dataset, 1, shuffle=True)

    total_loss = 0.0
    for epoch in range(1, epochs + 1):
        for x, y in data_iter:
            with autograd.record():
                loss = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=mx.nd.zeros, batch_size=1, ctx=ctx)
                encoder_outputs, encoder_state = encoder(x, encoder_state)

                # encoder_outputs尺寸: (max_seq_len, encoder_hidden_dim)
                encoder_outputs = encoder_outputs.flatten()
                # 编码器的第一个输入为BOS字符。
                decoder_input = nd.array([output_vocab.token_to_idx[BOS]],
                                         ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 使用当前时刻的预测结果作为下一时刻的编码器输入。
                    decoder_input = nd.array(
                        [decoder_output.argmax(axis=1).asscalar()], ctx=ctx)
                    loss = loss + softmax_cross_entropy(decoder_output, y[0][i])
                    if y[0][i].asscalar() == output_vocab.token_to_idx[EOS]:
                        break

            loss.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            total_loss += loss.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            cur_time = datetime.datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = 'Time %02d:%02d:%02d' % (h, m, s)
            print_loss_avg = total_loss / epoch_period / len(data_iter)
            loss_str = 'Epoch %d, Loss %f, ' % (epoch, print_loss_avg)
            print(loss_str + time_str)
            total_loss = 0.0
            prev_time = cur_time

            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,
                      max_seq_len)
```

以下分别实例化编码器、解码器和解码器初始隐含状态网络。

```{.python .input}
encoder = Encoder(len(input_vocab), encoder_hidden_dim, encoder_num_layers,
                  encoder_drop_prob)
decoder = Decoder(decoder_hidden_dim, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_dim, encoder_hidden_dim)
decoder_init_state = DecoderInitState(encoder_hidden_dim, decoder_hidden_dim)
```

给定简单的法语和英语序列，我们可以观察模型的训练结果。打印的结果中，Input、Output和Expect分别代表输入序列、输出序列和正确序列。

```{.python .input}
eval_fr_ens =[['elle est japonaise .', 'she is japanese .'],
              ['ils regardent .', 'they are watching .']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)
```

## 结论

* 我们可以将编码器—解码器和注意力机制应用于神经机器翻译中。


## 练习

* 试着使用更大的翻译数据集来训练模型，例如[WMT](http://www.statmt.org/wmt14/translation-task.html)和[Tatoeba Project](http://www.manythings.org/anki/)。调一调不同参数并观察实验结果。


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4689)
