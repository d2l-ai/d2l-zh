# 循环神经网络的Gluon实现

本节将使用Gluon来实现基于循环神经网络的语言模型。首先导入本节需要的包和模块，并读取周杰伦专辑歌词数据集。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import math
import time
import gluonbook as gb
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import rnn, nn, loss as gloss

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = gb.load_data_jay_lyrics()
```

## 定义模型

Gluon的`rnn`模块提供了循环神经网络的实现。下面构造一个单隐藏层的且隐藏单元个数为256的循环神经网络层`rnn_layer`，并对权重做初始化。

```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

接下来调用`rnn_layer`的成员函数`begin_state`来返回初始化的隐藏状态列表。它有一个形状为（1，`batch_size`，`num_hiddens`）的元素，其中1表示只有一个隐藏层。

```{.python .input  n=37}
batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
state[0].shape
```

与上一节里我们实现的循环神经网络不同，这里`rnn_layer`的输入格式为（`num_steps`，`batch_size`，`vocab_size`）。

```{.python .input  n=38}
num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape
```

返回的输出是形状为（`num_steps`，`batch_size`，`num_hiddens`）的NDArray（上一节实现是`num_steps`个（`batch_size`，`num_hiddens`）形状的NDArray），它可以之后被输入到输出层里。

接下来我们继承Block类来定义一个完整的循环神经网络，它首先将输入数据使用one-hot表示后输入到`rnn_layer`中，然后使用全连接输出层得到输出。

```{.python .input  n=39}
# 本类已保存在 gluonbook 包中方便以后使用。
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)
        
    def forward(self, inputs, state):
        # 将输入转置成（num_steps，batch_size）后获取 one-hot 表示。
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将 Y 形状变形成（num_steps * batch_size，num_hiddens），
        # 它的输出形状为（num_steps * batch_size，vocab_size）。
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## 模型训练

首先同前一节一样定义一个预测函数，这里的实现区别在于前向计算和初始化隐藏状态的函数接口稍有不同。

```{.python .input  n=41}
# 本函数已保存在 gluonbook 包中方便以后使用。
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    # 使用 model 的成员函数来初始化隐藏状态。
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数。
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

让我们使用权重为随机值的模型来预测一次。

```{.python .input  n=42}
ctx = gb.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
predict_rnn_gluon('分开', 10, model, vocab_size, ctx, idx_to_char, char_to_idx)
```

接下来实现训练函数，它的算法同上一节一样，但这里只使用了随机采样来读取数据。

```{.python .input  n=18}
# 本函数已保存在 gluonbook 包中方便以后使用。
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, 
                                corpus_indices, idx_to_char, char_to_idx, 
                                num_epochs, num_steps, lr, clipping_theta, 
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        loss_sum, start = 0.0, time.time()
        data_iter = gb.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for t, (X, Y) in enumerate(data_iter):
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # 梯度剪裁。
            params = [p.data() for p in model.collect_params().values()]
            gb.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)  # 因为已经误差取过均值，梯度不用再做平均。
            loss_sum += l.asscalar()

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(loss_sum / (t + 1)), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, 
                    ctx, idx_to_char, char_to_idx))
```

使用和上一节一样的超参数来训练模型。

```{.python .input  n=19}
num_epochs = 200
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
prefixes = ['分开', '不分开']
pred_period = 50
pred_len = 50

train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, 
                            corpus_indices, idx_to_char, char_to_idx, 
                            num_epochs, num_steps, lr, clipping_theta, 
                            batch_size, pred_period, pred_len, prefixes)
```

## 小结

* Gluon的`rnn`模块提供了循环神经网络层的实现。

## 练习

* 比较跟前一节的实现，看看Gluon的版本是不是运行速度更快？如果你觉得差别明显，试着找找原因。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4089)

![](../img/qr_rnn-gluon.svg)
