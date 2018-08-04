# 机器翻译

本节介绍编码器—解码器和注意力机制的应用。我们以机器翻译为例，使用Gluon实现一个含注意力机制的编码器—解码器。机器翻译的输入与输出都是不定长的文本序列。


## 含注意力机制的编码器—解码器

首先，导入实现所需的包或模块。

```{.python .input}
import collections
import io
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
```

下面定义一些特殊符号。其中“&lt;pad&gt;”（padding）符号使每个序列等长。我们已经在前面几节介绍了，“&lt;bos&gt;”和“&lt;eos&gt;”符号分别表示序列的开始和结束。

```{.python .input}
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
```

以下设置了模型超参数。我们在编码器和解码器中分别使用了一层和两层的循环神经网络。实验中，我们选取长度不超过5的输入和输出序列，并将预测时输出序列的最大长度设为20。这些序列长度考虑了句末添加的“&lt;eos&gt;”符号。

```{.python .input}
num_epochs = 40
eval_interval = 10
lr = 0.005
batch_size = 2
max_seq_len = 5
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

我们定义函数读取训练数据集。为了演示方便，这里使用了一个很小的法语—英语数据集。在读取数据时，我们在句末附上“&lt;eos&gt;”符号，并可能通过添加“&lt;pad&gt;”符号使每个序列等长。

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
                # 添加 PAD 符号使每个序列等长（长度为 max_seq_len）。
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

### 定义编码器

以下定义了基于门控循环单元的编码器。

```{.python .input}
class Encoder(nn.Block):
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

### 定义含注意力机制的解码器

以下定义了基于门控循环单元的解码器。解码器中注意力机制的实现参考了[“注意力机制”](attention.md)一节中的描述。

```{.python .input}
class Decoder(nn.Block):
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
                             activation='tanh', flatten=False))
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
        # 当循环神经网络有多个隐藏层时，取最靠近输出层的单层隐藏状态。
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

### 定义解码器初始状态

为了初始化解码器的隐藏状态，我们通过一层全连接网络来变换编码器的输出隐藏状态。

```{.python .input}
class DecoderInitState(nn.Block):
    def __init__(self, encoder_num_hiddens, decoder_num_hiddens, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(decoder_num_hiddens,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]
```

### 训练模型并输出不定长序列

Sutskever等人发现贪婪搜索也可以在机器翻译中也可以取得不错的结果 [1]。我们定义`translate`函数应用训练好的模型，并通过贪婪搜索输出不定长的翻译文本序列。解码器的最初时间步输入来自“&lt;bos&gt;”符号。对于一个输出中的序列，当解码器在某一时间步搜索出“&lt;eos&gt;”符号时，即完成该输出序列。

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
        # 解码器的第一个输入为 BOS 符号。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_test_output_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # 当任一时间步搜索出 EOS 符号时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)
        print('[output]', ' '.join(output_tokens))
        print('[expect]', fr_en[1], '\n')
```

下面定义模型训练函数。为了初始化解码器的隐藏状态，我们通过一层全连接网络来变换编码器最早时间步的输出隐藏状态。解码器中，当前时间步的预测词将作为下一时间步的输入。其实，我们也可以使用样本输出序列在当前时间步的词作为下一时间步的输入。这叫作强制教学（teacher forcing）。

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
    for epoch in range(1, num_epochs + 1):
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                # encoder_outputs 包含了编码器在每个时间步的隐藏状态。
                encoder_outputs, encoder_state = encoder(x, encoder_state)
                encoder_outputs = encoder_outputs.flatten()
                # 解码器的第一个输入为 BOS 符号。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时间步的预测词作为下一时间步的输入。
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

        if epoch % eval_interval == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, ' 
                      % (epoch, l_sum / eval_interval / len(data_iter)))
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

给定简单的法语和英语序列，我们可以观察模型的训练结果。打印的结果中，input、output和expect分别代表输入序列、输出序列和正确序列。我们可以比较output和expect，观察输出序列是否符合预期。

```{.python .input}
eval_fr_ens =[['elle est japonaise .', 'she is japanese .'],
              ['ils regardent .', 'they are watching .']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)
```

为了使模型能够翻译更复杂的句子，我们需要使用更大的训练数据集、调节超参数并增加训练时间。当然，我们还需要有验证数据集，并依据模型在它上面的表现调参。那么，该如何在验证数据集上评价模型的表现呢？这就需要评价翻译结果的指标。


## 评价翻译结果

2002年，IBM团队提出了一种评价翻译结果的指标，叫BLEU（Bilingual Evaluation Understudy）[2]。

设$k$为我们希望评价的$n$个连续词的最大长度，例如$k=4$。设$n$个连续词的精度为$p_n$。它是模型预测序列与样本标签序列匹配$n$个连续词的数量与模型预测序列中$n$个连续词数量之比。举个例子，假设标签序列为$ABCDEF$，预测序列为$ABBCD$。那么$p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0$。设$len_{\text{label}}$和$len_{\text{pred}}$分别为标签序列和模型预测序列的词数。那么，BLEU的定义为

$$ \exp(\min(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}})) \prod_{i=1}^k p_n^{1/2^n}.$$

需要注意的是，匹配较长连续词比匹配较短连续词更难。因此，一方面，匹配较长连续词应被赋予更大权重。而上式中$p_n^{1/2^n}$的指数相当于权重。随着$n$的提高，$n$个连续词的精度的权重随着$1/2^n$的减小而增大。例如$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。另一方面，模型预测较短序列往往会得到较高的$n$个连续词的精度。因此，上式中连乘项前面的系数是为了惩罚较短的输出。举个例子，当$k=2$时，假设标签序列为$ABCDEF$，而预测序列为$AB$。虽然$p_1 = p_2 = 1$，但惩罚系数$\exp(1-6/2) \approx 0.14$，因此BLEU也接近0.14。当预测序列和标签序列完全一致时，BLEU为1。

## 小结

* 我们可以将编码器—解码器和注意力机制应用于机器翻译中。
* BLEU可以用来评价翻译结果。


## 练习

* 试着使用更大的翻译数据集来训练模型，例如WMT [3] 和Tatoeba Project [4]。
* 在解码器中使用强制教学，观察实现现象。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4689)

![](../img/qr_machine-translation.svg)

## 参考文献

[1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).

[2] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.

[3] WMT. http://www.statmt.org/wmt14/translation-task.html

[4] Tatoeba Project. http://www.manythings.org/anki/
