# 机器翻译

本节介绍[编码器—解码器和注意力机制](seq2seq-attention.md)的应用。我们以神经机器翻译（neural machine translation）为例，介绍如何使用Gluon实现一个简单的编码器—解码器和注意力机制模型。


## 实现编码器—解码器和注意力机制

我们先载入需要的包。

```{.python .input}
import collections
import io
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
```

下面定义一些特殊字符。其中PAD (padding)符号使每个序列等长；BOS (beginning of sequence)符号表示序列的开始；而EOS (end of sequence)符号表示序列的结束。

```{.python .input}
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
```

以下是一些可以调节的模型参数。我们在编码器和解码器中分别使用了一层和两层的循环神经网络。

```{.python .input}
epochs = 40
epoch_period = 10
lr = 0.005
batch_size = 2
# 序列最大长度（含句末添加的EOS字符）。
max_seq_len = 5
# 测试时输出序列的最大长度（含句末添加的EOS字符）。
max_test_output_len = 20
encoder_num_layers = 1
decoder_num_layers = 2
encoder_drop_prob = 0.1
decoder_drop_prob = 0.1
encoder_embed_size = 256
encoder_num_hiddens = 256
decoder_num_hiddens = 256
alignment_size = 25
ctx = mx.cpu(0)
```

### 读取数据

我们定义函数读取训练数据集。为了减少运行时间，我们使用一个很小的法语——英语数据集。

这里使用了[之前章节](pretrained-embedding.md)介绍的`mxnet.contrib.text`来创建法语和英语的词典。需要注意的是，我们会在句末附上EOS符号，并可能通过添加PAD符号使每个序列等长。


我们通常会在输入序列和输出序列后面分别附上一个特殊字符“&lt;eos&gt;”（end of sequence）表示序列的终止。在测试模型时，一旦输出“&lt;eos&gt;”就终止当前的输出序列。

```{.python .input}
def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:
            input_seq, output_seq = line.rstrip().split('\t')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')
            if len(cur_input_tokens) < max_seq_len and \
                    len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                # 句末附上 EOS 符号。
                cur_input_tokens.append(EOS)
                # 添加 PAD 符号使每个序列等长（长度为 max_seq_len ）。
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
fr = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
en = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    fr[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    en[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)
dataset = gdata.ArrayDataset(fr, en)
```

### 编码器、含注意力机制的解码器和解码器初始状态

以下定义了基于[GRU](../chapter_recurrent-neural-networks/gru-scratch.md)的编码器。

```{.python .input}
class Encoder(nn.Block):
    """编码器。"""
    def __init__(self, num_inputs, embed_size, num_hiddens, num_layers,
                 drop_prob, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(num_inputs, embed_size)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=embed_size)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs).swapaxes(0, 1)
        embedding = self.dropout(embedding)
        output, state = self.rnn(embedding, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

以下定义了基于[GRU](../chapter_recurrent-neural-networks/gru-scratch.md)的解码器。它包含[上一节里介绍的注意力机制](seq2seq-attention.md)的实现。

```{.python .input}
class Decoder(nn.Block):
    """含注意力机制的解码器。"""
    def __init__(self, num_hiddens, num_outputs, num_layers, max_seq_len,
                 drop_prob, alignment_size, encoder_num_hiddens, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.encoder_num_hiddens = encoder_num_hiddens
        self.hidden_size = num_hiddens
        self.num_layers = num_layers
        with self.name_scope():
            self.embedding = nn.Embedding(num_outputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)
            # 注意力机制。
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(
                    nn.Dense(alignment_size,
                             in_units=num_hiddens + encoder_num_hiddens,
                             activation="tanh", flatten=False))
                self.attention.add(nn.Dense(1, in_units=alignment_size,
                                            flatten=False))

            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)
            self.out = nn.Dense(num_outputs, in_units=num_hiddens,
                                flatten=False)
            self.rnn_concat_input = nn.Dense(
                num_hiddens, in_units=num_hiddens + encoder_num_hiddens,
                flatten=False)

    def forward(self, cur_input, state, encoder_outputs):
        # 当 RNN 为多层时，取最靠近输出层的单层隐藏状态。
        single_layer_state = [state[0][-1].expand_dims(0)]
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_num_hiddens))
        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)
        energy = self.attention(encoder_outputs_and_hiddens)
        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)
        input_and_context = nd.concat(
            nd.expand_dims(self.embedding(cur_input), axis=1),
            decoder_context, dim=2)
        concat_input = self.rnn_concat_input(input_and_context).reshape(
            (1, -1, 0))
        concat_input = self.dropout(concat_input)
        state = [nd.broadcast_axis(single_layer_state[0], axis=0,
                                   size=self.num_layers)]
        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        output = self.out(output).reshape((-3, -1))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

为了初始化解码器的隐藏状态，我们通过一层全连接网络来转化编码器的输出隐藏状态。

```{.python .input}
class DecoderInitState(nn.Block):
    """解码器隐藏状态的初始化。"""
    def __init__(self, encoder_num_hiddens, decoder_num_hiddens, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(decoder_num_hiddens,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]
```

### 训练和应用模型

我们定义`translate`函数来应用训练好的模型。这些模型通过该函数的前三个参数传递。解码器的最初时刻输入来自BOS字符。当任一时刻的输出为EOS字符时，输出序列即完成。

```{.python .input}
def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('[input] ', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]
        # 添加 PAD 符号使每个序列等长（长度为 max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),
                                                 encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为 BOS 字符。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_test_output_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # 当任一时刻的输出为 EOS 字符时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)
        print('[output]', ' '.join(output_tokens))
        print('[expect]', fr_en[1], '\n')
```

下面定义模型训练函数。为了初始化解码器的隐藏状态，我们通过一层全连接网络来转化编码器最早时刻的输出隐藏状态。这里的解码器使用当前时刻的预测结果作为下一时刻的输入。

```{.python .input}
loss = gloss.SoftmaxCrossEntropyLoss()
eos_id = output_vocab.token_to_idx[EOS]

def train(encoder, decoder, decoder_init_state, max_seq_len, ctx,
          eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    decoder_init_state.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam', {'learning_rate': lr})

    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    l_sum = 0
    for epoch in range(1, epochs + 1):
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                encoder_outputs, encoder_state = encoder(x, encoder_state)
                encoder_outputs = encoder_outputs.flatten()
                # 解码器的第一个输入为 BOS 字符。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时刻的预测结果作为下一时刻的输入。
                    decoder_input = decoder_output.argmax(axis=1)
                    valid_length = valid_length + mask.sum()
                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    mask = mask * (y[:, i] != eos_id)
                l = l / valid_length
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            l_sum += l.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, ' 
                      % (epoch, l_sum / epoch_period / len(data_iter)))
            if epoch != 1:
                l_sum = 0
            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,
                      max_seq_len)
```

以下分别实例化编码器、解码器和解码器初始隐藏状态网络。

```{.python .input}
encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,
                                      decoder_num_hiddens)
```

给定简单的法语和英语序列，我们可以观察模型的训练结果。打印的结果中，Input、Output和Expect分别代表输入序列、输出序列和正确序列。

```{.python .input}
eval_fr_ens =[['elle est japonaise .', 'she is japanese .'],
              ['ils regardent .', 'they are watching .']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)
```

## 评价翻译结果

2002年，IBM团队提出了一种评价翻译结果的指标，叫做[BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) （Bilingual Evaluation Understudy）。

设$k$为我们希望评价的n-gram的最大长度，例如$k=4$。n-gram的精度$p_n$为模型输出中的n-gram匹配参考输出的数量与模型输出中的n-gram的数量的比值。例如，参考输出（真实值）为ABCDEF，模型输出为ABBCD。那么$p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0$。设$len_{ref}$和$len_{MT}$分别为参考输出和模型输出的词数。那么，BLEU的定义为

$$ \exp(\min(0, 1 - \frac{len_{ref}}{len_{MT}})) \prod_{i=1}^k p_n^{1/2^n}$$

需要注意的是，随着$n$的提高，n-gram的精度的权值随着$p_n^{1/2^n}$中的指数减小而提高。例如$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。换句话说，匹配4-gram比匹配1-gram应该得到更多奖励。另外，模型输出越短往往越容易得到较高的n-gram的精度。因此，BLEU公式里连乘项前面的系数为了惩罚较短的输出。例如当$k=2$时，参考输出为ABCDEF，而模型输出为AB，此时的$p_1 = p_2 = 1$，而$\exp(1-6/2) \approx 0.14$，因此BLEU=0.14。当模型输出也为ABCDEF时，BLEU=1。

## 小结

* 我们可以将编码器—解码器和注意力机制应用于神经机器翻译中。
* BLEU可以用来评价翻译结果。


## 练习

* 试着使用更大的翻译数据集来训练模型，例如[WMT](http://www.statmt.org/wmt14/translation-task.html)和[Tatoeba Project](http://www.manythings.org/anki/)。调一调不同参数并观察实验结果。
* Teacher forcing：在模型训练中，试着让解码器使用当前时刻的正确结果（而不是预测结果）作为下一时刻的输入。结果会怎么样？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4689)

![](../img/qr_nmt.svg)
