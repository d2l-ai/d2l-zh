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
        # single_layer_state尺寸: [(1, 1, decoder_hidden_dim)]
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
        # 解码器的第一个输入为BOS字符。
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

下面定义模型训练函数。为了初始化解码器的隐含状态，我们通过一层全连接网络来转化编码器最早时刻的输出隐含状态。这里的解码器使用当前时刻的预测结果作为下一时刻的输入。

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
                # 解码器的第一个输入为BOS字符。
                decoder_input = nd.array([output_vocab.token_to_idx[BOS]],
                                         ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时刻的预测结果作为下一时刻的输入。
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
            if epoch == 1:
                print_loss_avg = total_loss / len(data_iter)
            else:
                print_loss_avg = total_loss / epoch_period / len(data_iter)
            loss_str = 'Epoch %d, Loss %f, ' % (epoch, print_loss_avg)
            print(loss_str + time_str)
            if epoch != 1:
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

## 束搜索

在上一节里，我们提到编码器最终输出了一个背景向量$\mathbf{c}$，该背景向量编码了输入序列$x_1, x_2, \ldots, x_T$的信息。假设训练数据中的输出序列是$y_1, y_2, \ldots, y_{T^\prime}$，输出序列的生成概率是

$$\mathbb{P}(y_1, \ldots, y_{T^\prime}) = \prod_{t^\prime=1}^{T^\prime} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$$


对于机器翻译的输出来说，如果输出语言的词汇集合$\mathcal{Y}$的大小为$|\mathcal{Y}|$，输出序列的长度为$T^\prime$，那么可能的输出序列种类是$\mathcal{O}(|\mathcal{Y}|^{T^\prime})$。为了找到生成概率最大的输出序列，一种方法是计算所有$\mathcal{O}(|\mathcal{Y}|^{T^\prime})$种可能序列的生成概率，并输出概率最大的序列。我们将该序列称为最优序列。但是这种方法的计算开销过高（例如，$10000^{10} = 1 \times 10^{40}$）。


我们目前所介绍的解码器在每个时刻只输出生成概率最大的一个词汇。对于任一时刻$t^\prime$，我们从$|\mathcal{Y}|$个词中搜索出输出词

$$y_{t^\prime} = \text{argmax}_{y_{t^\prime} \in \mathcal{Y}} \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$$

因此，搜索计算开销（$\mathcal{O}(|\mathcal{Y}| \times {T^\prime})$）显著下降（例如，$10000 \times 10 = 1 \times 10^5$），但这并不能保证一定搜索到最优序列。

束搜索（beam search）介于上面二者之间。我们来看一个例子。

假设输出序列的词典中只包含五个词：$\mathcal{Y} = \{A, B, C, D, E\}$。束搜索的一个超参数叫做束宽（beam width）。以束宽等于2为例，假设输出序列长度为3，假如时刻1生成概率$\mathbb{P}(y_{t^\prime} \mid \mathbf{c})$最大的两个词为$A$和$C$，我们在时刻2对于所有的$y_2 \in \mathcal{Y}$都分别计算$\mathbb{P}(y_2 \mid A, \mathbf{c})$和$\mathbb{P}(y_2 \mid C, \mathbf{c})$，从计算出的10个概率中取最大的两个，假设为$\mathbb{P}(B \mid A, \mathbf{c})$和$\mathbb{P}(E \mid C, \mathbf{c})$。那么，我们在时刻3对于所有的$y_3 \in \mathcal{Y}$都分别计算$\mathbb{P}(y_3 \mid A, B, \mathbf{c})$和$\mathbb{P}(y_3 \mid C, E, \mathbf{c})$，从计算出的10个概率中取最大的两个，假设为$\mathbb{P}(D \mid A, B, \mathbf{c})$和$\mathbb{P}(D \mid C, E, \mathbf{c})$。

接下来，我们可以在输出序列：$A$、$C$、$AB$、$CE$、$ABD$、$CED$中筛选出以特殊字符EOS结尾的候选序列。再在候选序列中取以下分数最高的序列作为最终候选序列：

$$ \frac{1}{L^\alpha} \log \mathbb{P}(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t^\prime=1}^L \log \mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$$

其中$L$为候选序列长度，$\alpha$一般可选为0.75。分母上的$L^\alpha$是为了惩罚较长序列的分数中的对数相加项。

## 评价翻译结果

2002年，IBM团队提出了一种评价翻译结果的指标，叫做[BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) （Bilingual Evaluation Understudy）。

设$k$为我们希望评价的n-gram的最大长度，例如$k=4$。n-gram的精度$p_n$为模型输出中的n-gram匹配参考输出的数量与模型输出中的n-gram的数量的比值。例如，参考输出（真实值）为ABCDEF，模型输出为ABBCD。那么$p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0$。设$len_{ref}$和$len_{MT}$分别为参考输出和模型输出的词数。那么，BLEU的定义为

$$ \exp(\min(0, 1 - \frac{len_{ref}}{len_{MT}})) \prod_{i=1}^k p_n^{1/2^n}$$

需要注意的是，随着$n$的提高，n-gram的精度的权值随着$p_n^{1/2^n}$中的指数减小而提高。例如$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。换句话说，匹配4-gram比匹配1-gram应该得到更多奖励。另外，模型输出越短往往越容易得到较高的n-gram的精度。因此，BLEU公式里连乘项前面的系数为了惩罚较短的输出。例如当$k=2$时，参考输出为ABCDEF，而模型输出为AB，此时的$p_1 = p_2 = 1$，而$\exp(1-6/3) \approx 0.37$，因此BLEU=0.37。当模型输出也为ABCDEF时，BLEU=1。

## 结论

* 我们可以将编码器—解码器和注意力机制应用于神经机器翻译中。
* 束搜索有可能提高输出质量。
* BLEU可以用来评价翻译结果。


## 练习

* 试着使用更大的翻译数据集来训练模型，例如[WMT](http://www.statmt.org/wmt14/translation-task.html)和[Tatoeba Project](http://www.manythings.org/anki/)。调一调不同参数并观察实验结果。
* Teacher forcing：在模型训练中，试着让解码器使用当前时刻的正确结果（而不是预测结果）作为下一时刻的输入。结果会怎么样？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4689)
