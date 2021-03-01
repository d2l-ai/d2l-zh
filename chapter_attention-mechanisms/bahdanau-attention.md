# Bahdanau 注意力
:label:`sec_seq2seq_attention`

我们在 :numref:`sec_seq2seq` 中学习了机器翻译问题，我们设计了基于两个 RNN 的编码器解码器结构，用于序列到序列的学习。具体来说，RNN 编码器将可变长度的序列转换为固定形状的背景变量（context variable），然后 RNN 解码器每次根据上一步生成的词符号和背景变量，一个一个的生成输出序列的目标词符号。虽然每个解码步中都使用了 *相同的* 整个输入序列的背景变量，但是并非所有输入（源）词符号都对解码某个特定词符号有用。

在另一个相关的挑战中，为了对给定的文本序列进行手写字符生成，Graves 设计了一种可微的注意力模型，将文本字符与更长的笔迹进行对齐，其只对一个输入方向进行了对齐 :cite:`Graves.2013`。受这种对齐能力的启发，Bahdanau 等人提出了一个没有对齐方向限制的可微的注意力模型 :cite:`Bahdanau.Cho.Bengio.2014`。在预测词符号时，如果不是所有输入词符号都相关，模型将仅对齐（或注意）输入序列中与当前预测相关的部分。这是通过将背景变量视为注意力池化的输出来实现的。

## 模型

在接下来描述用于 RNN 编码器解码器的 Bahdanau 注意力时，我们将遵循 :numref:`sec_seq2seq` 中相同的符号。新的基于注意力的模型与 :numref:`sec_seq2seq` 中的模型相同，只不过 :eqref:`eq_seq2seq_s_t` 中的背景变量 $\mathbf{c}$ 在每一个解码的时间步 $t'$ 时都会被 $\mathbf{c}_{t'}$ 替换。假设输入序列中有 $T$ 个词符号，那么解码的时间步 $t'$ 的背景变量是注意力池化的输出：

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

其中，时间步 $t' - 1$ 时解码器的隐藏状态 $\mathbf{s}_{t' - 1}$ 作为查询，编码器的隐藏状态 $\mathbf{h}_t$ 既作为键，也作为值，注意力权重 $\alpha$ 是通过 :eqref:`eq_attn-scoring-alpha` 所定义的加法注意力评分函数计算的。

与 :numref:`fig_seq2seq_details` 中的简单的 RNN 编码器解码器架构略有不同，:numref:`fig_s2s_attention_details` 描述了应用了 Bahdanau 注意力的架构。

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 定义使用注意力的解码器

要实现应用了 Bahdanau 注意力的 RNN 编码器解码器，我们只需重新定义解码器即可。为了更方便地显示学习到的注意力权重，以下 `AttentionDecoder` 类定义了具有注意机制的解码器的基本接口。

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

现在让我们在下面的 `Seq2SeqAttentionDecoder` 类中使用 Bahdanau 注意力来实现 RNN 解码器。解码器的状态初始化为 i) 编码器在所有时间步的最终层的隐藏状态（作为注意力的键和值）；ii) 最后一个时间步时编码器所有层的隐藏状态（用于初始化解码器的隐藏状态）；和 iii) 编码器输入的有效长度（在注意力池化计算过程中排除用于填充的词符号）。在每个解码时间步中，解码器上一个时间步的最终层隐藏状态将作为注意力的查询。最后，将注意力池化的输出和这一步输入的嵌入结果（input embedding）拼接起来, 作为 RNN 解码器这一步的输入。

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

接下来，我们使用小批量数据来测试刚才实现的应用了 Bahdanau 注意力的解码器，数据包含 4 个输入序列，每个序列有 7 个时间步。

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## 训练

与 :numref:`sec_seq2seq_training` 类似，我们在这里指定超参数并实例化基于 Bahdanau 注意力的编码器解码器，然后用这个模型进行机器翻译。由于新增的注意力机制，这会比没有注意力机制的 :numref:`sec_seq2seq_training` 慢得多。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

模型训练完毕后，我们用它将几个英语句子翻译成法语并计算它们的 BLEU 分数。

```{.python .input}
#@tab all
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

通过将翻译最后一个英语句子时的注意力权重可视化，我们可以看到每个查询都会在键值对上分配不均匀的权重。下图显示，在每个解码时间步中，输入序列的不同部分都会有选择性地被聚合到注意力池化中。

```{.python .input}
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key posistions', ylabel='Query posistions')
```

```{.python .input}
#@tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key posistions', ylabel='Query posistions')
```

## 小结

* 在预测词符号时，如果不是所有输入词符号都是相关的，那么基于 Bahdanau 注意力的 RNN 编码器会有选择地聚合输入序列的不同部分。这是通过将背景变量视为加法注意力池化的输出来实现的。
* 在 RNN 编码器解码器中，Bahdanau 的注意力将上一个时间步的解码器隐藏状态视为查询，将编码器在所有时间步的隐藏状态同时视为键和值。

## 练习

1. 在实验中用 LSTM 替换 GRU。
1. 修改实验，将加法注意力评分函数替换为缩放点积注意力的评分函数。它如何影响训练效率？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:
