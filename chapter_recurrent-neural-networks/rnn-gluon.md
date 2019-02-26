# 循环神经网络的简洁实现

本节将使用Gluon来更简洁地实现基于循环神经网络的语言模型。首先，我们读取周杰伦专辑歌词数据集。

```{.python .input  n=1}
import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()
```

## 定义模型

Gluon的`rnn`模块提供了循环神经网络的实现。下面构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层`rnn_layer`，并对权重做初始化。

```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

接下来调用`rnn_layer`的成员函数`begin_state`来返回初始化的隐藏状态列表。它有一个形状为(隐藏层个数, 批量大小, 隐藏单元个数)的元素。

```{.python .input  n=37}
batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
state[0].shape
```

与上一节中实现的循环神经网络不同，这里`rnn_layer`的输入形状为(时间步数, 批量大小, 输入个数)。其中输入个数即one-hot向量长度（词典大小）。此外，`rnn_layer`作为Gluon的`rnn.RNN`实例，在前向计算后会分别返回输出和隐藏状态，其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输入。需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)。而`rnn.RNN`实例在前向计算返回的隐藏状态指的是隐藏层在最后时间步的可用于初始化下一时间步的隐藏状态：当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中；对于像长短期记忆这样的循环神经网络，该变量还会包含其他信息。我们会在本章的后面介绍长短期记忆和深度循环神经网络。

```{.python .input  n=38}
num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

接下来我们继承`Block`类来定义一个完整的循环神经网络。它首先将输入数据使用one-hot向量表示后输入到`rnn_layer`中，然后使用全连接输出层得到输出。输出个数等于词典大小`vocab_size`。

```{.python .input  n=39}
# 本类已保存在d2lzh包中方便以后使用
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## 训练模型

同上一节一样，下面定义一个预测函数。这里的实现区别在于前向计算和初始化隐藏状态的函数接口。

```{.python .input  n=41}
# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    # 使用model的成员函数来初始化隐藏状态
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

让我们使用权重为随机值的模型来预测一次。

```{.python .input  n=42}
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
predict_rnn_gluon('分开', 10, model, vocab_size, ctx, idx_to_char, char_to_idx)
```

接下来实现训练函数。算法同上一节的一样，但这里只使用了相邻采样来读取数据。

```{.python .input  n=18}
# 本函数已保存在d2lzh包中方便以后使用
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # 梯度裁剪
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)  # 因为已经误差取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, ctx, idx_to_char,
                    char_to_idx))
```

使用和上一节实验中一样的超参数来训练模型。

```{.python .input  n=19}
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```

## 小结

* Gluon的`rnn`模块提供了循环神经网络层的实现。
* Gluon的`rnn.RNN`实例在前向计算后会分别返回输出和隐藏状态。该前向计算并不涉及输出层计算。

## 练习

* 与上一节的实现进行比较。看看Gluon的实现是不是运行速度更快？如果你觉得差别明显，试着找找原因。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4089)

![](../img/qr_rnn-gluon.svg)
