# 机器翻译

机器翻译是指将一段文本从一种语言自动翻译到另一种语言。因为一段文本序列在不同语言中长度不一定相同，所以我们使用机器翻译为例来介绍编码器—解码器和注意力机制的应用。

## 读取和预处理数据

我们先定义一些特殊符号。其中“&lt;pad&gt;”（padding）符号用来添加在较短序列后，直到每个序列等长，而“&lt;bos&gt;”和“&lt;eos&gt;”符号分别表示序列的开始和结束。

```{.python .input  n=2}
import collections
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
```

接着定义两个辅助函数对后面读取的数据进行预处理。

```{.python .input}
# 对一个序列，记录所有的词在 all_tokens 中以便之后构造词典，然后将该序列后添加 PAD 直到
# 长度变为 max_seq_len，并记录在 all_seqs 中。
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造 NDArray 实例。
def build_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
                                  reserved_tokens=[PAD, BOS, EOS])
    indicies = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indicies)
```

为了演示方便，我们在这里使用一个很小的法语—英语数据集。这个数据集里，每一行是一对法语句子和它对应的英语句子，中间使用“\t”隔开。在读取数据时，我们在句末附上“&lt;eos&gt;”符号，并可能通过添加“&lt;pad&gt;”符号使每个序列的长度均为`max_seq_len`。我们为法语词和英语词分别创建词典。法语词的索引和英语词的索引相互独立。

```{.python .input  n=31}
def read_data(max_seq_len):
    # in 和 out 分别是 input 和 output 的缩写。
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if (len(in_seq_tokens) > max_seq_len - 1 or
            len(out_seq_tokens) > max_seq_len - 1):
            continue  # 如果加上 EOS 后长于 max_seq_len，则忽略掉此样本。
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)
```

将序列的最大长度设成7，然后查看读取到的第一个样本。该样本分别包含法语词索引序列和英语词索引序列。

```{.python .input  n=181}
max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
dataset[0]
```

## 含注意力机制的编码器—解码器

我们将使用含注意力机制的编码器—解码器来将一段法语翻译成英语。下面我们来介绍模型的实现。

### 编码器

在编码器中，我们将词索引通过词嵌入层得到特征表达后输入到一个多层的循环神经网络里。我们使用Gluon的rnn包中的循环神经网络的实现，它们的前向计算输出是最后一层在每个时间步中的状态，这个不同于“循环神经网络”一章中的实现里输出是输出层的输出。因此我们可以直接使用输入来计算注意力机制权重。

```{.python .input  n=165}
class Encoder(nn.Block):
    def __init__(self, num_inputs, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(num_inputs, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是（批量大小，序列长）。将输出互换批量维和序列维。
        embedding = self.embedding(inputs).swapaxes(0, 1)        
        return self.rnn(embedding, state)
                                 
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

编码器前向计算返回输出和最后时间步隐藏状态。输出形状是（序列长，批量大小，隐藏单元个数）。状态是一个列表，由于这里使用GRU单元，这个列表只有一个元素，其形状为（层数，批量大小，隐藏单元个数）。

```{.python .input  n=166}
encoder = Encoder(num_inputs=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
output.shape, state[0].shape
```

### 注意力机制

由于循环网络中的输入输出有额外的序列维，我们先介绍Dense类的一个选项`flatten=False`， 它将不自动的将输入转成二维矩阵，而是保持输入的维数，并将最后一维当做特征维。例如下例中，对三维输入做前向运算后只改变了最后一维的大小。

```{.python .input}
dense = nn.Dense(2, flatten=False)
dense.initialize()
dense(nd.zeros((3, 5, 7))).shape
```

回忆[“注意力机制”](./attention.md)一节中的公式定义，函数$a$可以看做是连续作用两个全连接层，第一个的输入是解码器的隐藏状态和编码器的所有隐藏状态的拼接，且使用tanh作为激活函数，第二个则输出个数为1。两个全连接层均不使用偏差，而且不将三维输入转换成二维矩阵。这里$a$中$\boldsymbol{v}$的长度是一个超参数。

```{.python .input  n=167}
def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False, 
                       flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model
```

注意力机制的输入包括形状为（序列长，批量大小，隐藏单元大小）的编码器所有时间步的隐藏状态，和形状为（批量大小，隐藏单元大小）的当前时间步的解码器隐藏状态。它返回背景变量，其形状为（批量大小，隐藏单元个数）。

```{.python .input  n=168}
def attention_forward(model, enc_states, dec_state): 
    # 将解码器状态广播到跟编码器状态一样的形状后进行拼接，然后输入到 model 中。
    dec_states = nd.broadcast_axis(
        dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states)  # e 的形状为（序列长，批量大小，1）
    alpha = nd.softmax(e, axis=0)  # 在序列维度做 softmax。
    return (alpha * enc_states).sum(axis=0)  # 返回背景变量。
```

下面具体来看一下注意力机制的输入和输出的形状。

```{.python .input  n=169}
seq_len, batch_size, num_hiddens = 10, 4, 8
model = attention_model(10)
model.initialize()
enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
dec_state = nd.zeros((batch_size, num_hiddens))
attention_forward(model, enc_states, dec_state).shape
```

### 含注意力机制的解码器

解码器和编码器同样先使用词嵌入层对输入词索引进行特征提取，然后使用多层循环神经网络。不同之处在于，在每个时间步里，解码器将使用前一时间的隐藏状态和编码器的所有隐藏状态来计算注意力机制中的背景变量，然后将其拼接上输入进循环神经网络中。这里的实现里，我们只为编码器循环神经网络最上层的隐藏状态和解码器循环神经网络最下层的隐藏状态应用注意力机制。以及我们直接使用编码器最后时间步的状态来作为解码器的初始状态，这样要求编码器和解码器的循环神经网络使用同样的层数和隐藏单元个数。最后使用输出单元个数为输入个数的全连接层得到每个时间步的预测。

下面实现解码器，它的前向计算每次调用会计算一个时间步。其输入包括长为批量大小的当前时间步输入，形状为（批量大小，隐藏单元个数）的前一时间步隐藏状态，和形状为（序列上，批量大小，隐藏单元大小）的编码器所有时间步的隐藏状态。返回的则是形状为（批量大小，输入个数）的预测，和更新后的隐藏状态。

```{.python .input  n=170}
class Decoder(nn.Block):
    def __init__(self, num_inputs, embed_size, num_hiddens, num_layers, 
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(num_inputs, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)        
        self.out = nn.Dense(num_inputs, flatten=False)
        
    def forward(self, cur_input, state, enc_states):
        # 单个时间步的前向计算。使用最下层的隐藏状态来计算注意力权重。
        c = attention_forward(self.attention, enc_states, state[0][0])
        # 将背景向量和词嵌入层的输出在特征维拼接后进入 GRU。
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # 增加大小为 1 的序列维后计算 GRU 的输出和状态。
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # 去掉序列维，输出形状为（批量大小，输入个数）。
        output = self.out(output).squeeze(axis=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接使用编码器最后时间步的状态作为初始状态。
        return enc_state 
```

## 模型训练


首先实现一个小批量的前向计算。这里我们使用样本输出序列在当前时间步的词作为下一时间步的输入，即强制教学（teacher forcing）。此外，同[“word2vec 的实现”](./word2vec-gluon.md)一节一样我们使用掩码来不对填充项计算损失。

```{.python .input}
def batch_forward(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    # 编码
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始解码器状态。它时间步 1 的输入是 BOS。
    dec_state = decoder.begin_state(enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # 我们将使用 mask 来忽略掉拟合 PAD 的损失。
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    batch_loss = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        batch_loss = batch_loss + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学。
        num_not_pad_tokens += mask.sum().asscalar()
        # 当遇到 EOS 时，后面将都是 PAD，对应的 mask 项设成 0。
        mask = mask * (y != out_vocab.token_to_idx[EOS])
    return batch_loss / num_not_pad_tokens
```

在训练函数中，我们需要对编码器和解码器的参数都做更新。

```{.python .input  n=188}
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loss_sum = 0
        for X, Y in data_iter:
            with autograd.record():
                batch_loss = batch_forward(encoder, decoder, X, Y, loss)
            batch_loss.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            loss_sum += batch_loss.asscalar() 
        if (epoch+1) % 10 == 0:
            print("epoch %d, loss %.3f" % (
                epoch + 1, loss_sum / len(data_iter)))
```

接下来创建模型实例。

```{.python .input}
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob = 10, 0.5

encoder = Encoder(len(in_vocab), embed_size, num_hiddens, 
                  num_layers, drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, 
                  num_layers, attention_size, drop_prob)
```

并使用一组训练参数来查看训练效果。

```{.python .input}
lr, batch_size, num_epochs = 0.01, 2, 40
train(encoder, decoder, dataset, lr, batch_size, num_epochs)
```

## 模型预测

在[“束搜索”](./beam-search.md)中我们介绍了三种方法来生成解码器每个时间步的输入。这里我们实现最简单的贪婪搜索。

```{.python .input  n=177}
def translate(encoder, decoder, input_seq, max_seq_len):
    # 编码。
    in_tokens = input_seq.split(' ') 
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    enc_output, enc_state = encoder(enc_input, enc_state)
    # 初始化解码。
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS:  # 当任一时间步搜索出 EOS 符号时，输出序列即完成。
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens
```

简单测试正确性。法语句子“ils regardent.”对应的英语句子为“they are watching.”。

```{.python .input}
input_seq = 'ils regardent .'
translate(encoder, decoder, input_seq, max_seq_len)
```

## 评价翻译结果

机器翻译结果通常使用BLEU（Bilingual Evaluation Understudy）[2]来进行衡量。对于任意一个预测序列中的连续子序列，BLEU考察这个连续子序列是否出现在样本标签序列中。具体开始，假设所有长为$n$的预测序列中的连续词出现在标签序列中的概率为$p_n$。举个例子，假设标签序列为$ABCDEF$，预测序列为$ABBCD$。那么$p_1 = 4/5,\ p_2 = 3/4,\ p_3 = 1/3,\ p_4 = 0$。设$len_{\text{label}}$和$len_{\text{pred}}$分别为标签序列和预测序列的词数。那么，BLEU的定义为

$$ \exp\left(\min\left(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$

这里$k$是最大匹配长度。可以看到当预测序列和标签序列完全一致时，BLEU为1。

因为匹配较长连续词比匹配较短连续词更难，BLEU对匹配较长连续词的成功率赋予了更大权重。上式中$p_n^{1/2^n}$的指数相当于权重。随着$n$的提高，$n$个连续词的精度的权重随着$1/2^n$的减小而增大。例如$0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96$。

模型预测较短序列往往会得到较高的$n$个连续词的精度。因此，上式中连乘项前面的系数是为了惩罚较短的输出。举个例子，当$k=2$时，假设标签序列为$ABCDEF$，而预测序列为$AB$。虽然$p_1 = p_2 = 1$，但惩罚系数$\exp(1-6/2) \approx 0.14$，因此BLEU也接近0.14。

下面实现BLEU。

```{.python .input}
def bleu(pred_tokens, label_tokens, k):
    l_pred, l_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1-l_label/l_pred))
    for n in range(1, k+1):
        p_n = 0
        for i in range(l_pred - n + 1):
            if ' '.join(pred_tokens[i: i+n]) in ' '.join(label_tokens):
                p_n += 1
        score *= p_n / (l_pred - n + 1)
    return score
```

并定义一个辅助打印函数。

```{.python .input}
def score(input_seq, label_seq, k):
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.1f, predict: %s'%(bleu(pred_tokens, label_tokens, k), ' '.join(pred_tokens)))
```

预测正确是分数为1。

```{.python .input}
score('ils regardent .', 'they are watching .', k=2)
```

尝试一个不在训练集中的样本。

```{.python .input}
score('ils sont canadiens', 'they are canadian.', k=2)
```

## 小结

* 我们可以将编码器—解码器和注意力机制应用于机器翻译中。
* BLEU可以用来评价翻译结果。

## 练习

* 如何改进解码器的状态初始函数使得可以处理解码器的隐藏单元个数跟编码器不一样？如果层数也不一样呢？
* 如何让解码器的第$i$层循环神经网络使用编码器中的第$i$层的隐藏状态来计算注意力机制权重？
* 在训练中替换强制教学为同测试一样使用前一时间步里的输出作为输入，观察不同。
* 试着使用更大的翻译数据集来训练模型，例如WMT [3] 和Tatoeba Project [4]。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4689)

![](../img/qr_machine-translation.svg)

## 参考文献

[1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).

[2] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.

[3] WMT. http://www.statmt.org/wmt14/translation-task.html

[4] Tatoeba Project. http://www.manythings.org/anki/
