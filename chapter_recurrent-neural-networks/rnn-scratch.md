# 循环神经网络的从零开始实现

在本节中，我们将从零开始实现一个基于字符级循环神经网络的语言模型，并在周杰伦专辑歌词数据集上训练一个模型来进行歌词创作。首先，我们读取周杰伦专辑歌词数据集：

```{.python .input  n=1}
import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()
```

## one-hot向量

为了将词表示成向量输入到神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为$N$（即词典大小`vocab_size`），每个字符已经同一个从0到$N-1$的连续整数值索引一一对应。如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为$N$的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。

```{.python .input  n=2}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

我们每次采样的小批量的形状是(批量大小, 时间步数)。下面的函数将这样的小批量变换成数个可以输入进网络的形状为(批量大小, 词典大小)的矩阵，矩阵个数等于时间步数。也就是说，时间步$t$的输入为$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n$为批量大小，$d$为输入个数，即one-hot向量长度（词典大小）。

```{.python .input  n=3}
def to_onehot(X, size):  # 本函数已保存在d2lzh包中方便以后使用
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

## 初始化模型参数

接下来，我们初始化模型参数。隐藏单元个数 `num_hiddens`是一个超参数。

```{.python .input  n=4}
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

## 定义模型

我们根据循环神经网络的计算表达式实现该模型。首先定义`init_rnn_state`函数来返回初始化的隐藏状态。它返回由一个形状为(批量大小, 隐藏单元个数)的值为0的`NDArray`组成的元组。使用元组是为了更便于处理隐藏状态含有多个`NDArray`的情况。

```{.python .input  n=5}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

下面的`rnn`函数定义了在一个时间步里如何计算隐藏状态和输出。这里的激活函数使用了tanh函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，当元素在实数域上均匀分布时，tanh函数值的均值为0。

```{.python .input  n=6}
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

做个简单的测试来观察输出结果的个数（时间步数），以及第一个时间步的输出层输出的形状和隐藏状态的形状。

```{.python .input  n=7}
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape
```

## 定义预测函数

以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。这个函数稍显复杂，其中我们将循环神经单元`rnn`设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。

```{.python .input  n=8}
# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

我们先测试一下`predict_rnn`函数。我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。因为模型参数为随机值，所以预测结果也是随机的。

```{.python .input  n=9}
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)
```

## 裁剪梯度

循环神经网络中较容易出现梯度衰减或梯度爆炸。我们会在[“通过时间反向传播”](bptt.md)一节中解释原因。为了应对梯度爆炸，我们可以裁剪梯度（clip gradient）。假设我们把所有模型参数梯度的元素拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后的梯度

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

的$L_2$范数不超过$\theta$。

```{.python .input  n=10}
# 本函数已保存在d2lzh包中方便以后使用
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## 困惑度

我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下[“softmax回归”](../chapter_deep-learning-basics/softmax-regression.md)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

* 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
* 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
* 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。

显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小`vocab_size`。

## 定义模型训练函数

跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

1. 使用困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。相关讨论可参考[“语言模型数据集（周杰伦专辑歌词）”](lang-model-dataset.md)一节。

另外，考虑到后面将介绍的其他循环神经网络，为了更通用，这里的函数实现更长一些。

```{.python .input  n=11}
# 本函数已保存在d2lzh包中方便以后使用
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
```

## 训练模型并创作歌词

现在我们可以训练模型了。首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input  n=12}
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
```

下面采用随机采样训练模型并创作歌词。

```{.python .input  n=13}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

接下来采用相邻采样训练模型并创作歌词。

```{.python .input  n=19}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

## 小结

* 可以用基于字符级循环神经网络的语言模型来生成文本序列，例如创作歌词。
* 当训练循环神经网络时，为了应对梯度爆炸，可以裁剪梯度。
* 困惑度是对交叉熵损失函数做指数运算后得到的值。


## 练习

* 调调超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 不裁剪梯度，运行本节中的代码，结果会怎样？
* 将`pred_period`变量设为1，观察未充分训练的模型（困惑度高）是如何创作歌词的。你获得了什么启发？
* 将相邻采样改为不从计算图分离隐藏状态，运行时间有没有变化？
* 将本节中使用的激活函数替换成ReLU，重复本节的实验。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/989)

![](../img/qr_rnn-scratch.svg)
