# 基于循环神经网络的语言模型

前两节介绍了语言模型和循环神经网络的设计。在本节中，我们将从零开始实现一个基于循环神经网络的语言模型，并应用它创作歌词。在这个应用中，我们收集了周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词，并应用循环神经网络来训练一个语言模型。当模型训练好后，我们将以一种简单的方式创作歌词：给定开头的几个词，输出预测概率最大的下一个词；然后将该词附在开头后继续输出预测概率最大的下一个词；如此循环。

## 字符级循环神经网络

本节实验中的循环神经网络将每个字符视作词，该模型被称为字符级循环神经网络（character-level recurrent neural network）。因为不同字符的个数远小于不同词的个数（对于英文尤其如此），所以字符级循环神经网络通常计算更加简单。

设小批量中样本数$n=1$，文本序列为“想”、“要”、“直”、“升”、“机”（《Jay》中第一曲《可爱女人》的第一句，由徐若瑄填词）。图6.2演示了如何使用字符级循环神经网络来给定当前字符预测下一个字符。在训练时，我们对每个时间步的输出作用Softmax，然后使用交叉熵损失函数来计算它与标签的误差。

![基于字符级循环神经网络的语言模型。输入序列和标签序列分别为“想”、“要”、“直”、“升”和“要”、“直”、“升”、“机”。](../img/rnn-train.svg)

## 读取数据集

首先导入本节所需的包和模块。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import random
import zipfile
```

然后读取这个数据集，看看前50个字符是什么样的。

```{.python .input  n=20}
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars[0:40]
```

这个数据集有五万多个字符。为了打印方便，我们把换行符替换成空格。然后使用前两万个字符来训练模型，这样可以使得训练更快一些。

```{.python .input  n=14}
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]
```

## 建立字符索引

首先我们将每个字符映射成一个从0开始的整数，或者叫做索引，来方便之后的处理。为了得到索引，我们将数据集里面所有不同的字符取出来，然后将其逐一映射到索引来构造词典，接着打印`vocab_size`，即词典中不同字符的个数。

```{.python .input  n=9}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size
```

之后将训练数据集中每个字符转成从索引，并打印前20个字符和其对应的索引。

```{.python .input  n=18}
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

## 时序数据的采样

在训练中我们需要每次随机读取小批量样本和标签。这里不同的是时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符：“想”、“要”、“有”、“直”、“升”。且该样本的标签序列为这些字符分别在训练集中的下一个字符：“要”、“有”、“直”、“升”、“机”。我们有两种方式对时序数据采样，分别是随机采样和相邻采样。

### 随机采样

下面代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`指每个小批量的样本数，`num_steps`为每个样本所包含的时间步数。
在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

```{.python .input  n=25}
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为输出的索引是相应输入的索引加一。
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    # 返回从 pos 开始的长为 num_steps 的序列
    _data = lambda pos: corpus_indices[pos: pos + num_steps]
    for i in range(epoch_size):
        # 每次读取 batch_size 个随机样本。
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
```

让我们输入一个从0到29的人工序列，设批量大小和时间步数分别为2和6，打印随机采样每次读取的小批量样本的输入`X`和标签`Y`。可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

```{.python .input  n=31}
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

### 相邻采样

除了对原始序列做随机采样之外，我们还可以使相邻的两个随机小批量在原始序列上的位置相毗邻。这时候，我们就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量输入，并如此循环下去。这对实现循环神经网络造成了两方面影响。一方面，
在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态。
另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。
为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图分离出来。

```{.python .input  n=32}
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```

同样一样的设置下打印相邻采样每次读取的小批量样本的输入`X`和标签`Y`。相邻的两个随机小批量在原始序列上的位置相毗邻。

```{.python .input  n=33}
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

## One-hot向量

为了将词表示成向量来输入进神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为$N$（即`vocab_size`），每个字符已经同一个从0到$N-1$的连续整数值索引一一对应。如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为$N$的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。下面分别展示了索引为0和2的one-hot向量。

```{.python .input  n=21}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

我们每次采样的小批量的形状是（`batch_size`, `num_steps`）。下面这个函数将其转换成`num_steps`个可以输入进网络的形状为（`batch_size`, `vocab_size`）的矩阵。也就是总时间步$T=$`num_steps`，时间步$t$的输入$\boldsymbol{X_t} \in \mathbb{R}^{n \times d}$，其中$n=$`batch_size`，$d=$`vocab_size`（one-hot向量长度）。

```{.python .input  n=35}
def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

## 初始化模型参数

接下来，我们初始化模型参数。隐藏单元个数 `num_hiddens`是一个超参数。

```{.python .input  n=37}
ctx = gb.try_gpu()
print('will use', ctx)

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size

def get_params():
    # 隐藏层参数。
    W_xh = nd.random.normal(
        scale=0.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hh = nd.random.normal(
        scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数。
    W_hy = nd.random.normal(
        scale=0.01, shape=(num_hiddens, num_outputs), ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度。
    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
```

## 定义模型

我们根据循环神经网络的表达式实现该模型。首先定义`init_rnn_state`函数来返回初始化的隐藏状态。这里隐藏状态是一个形状为（`batch_size`, `num_hiddens`）的矩阵。

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, ctx, is_lstm=False):
    state = [nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), ]
    if is_lstm: # 初始化记忆细胞，在后面介绍 LSTM 的章节时使用，本节可以忽略。        
        state.append(nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))
    return state
```

下面定义在一个时间步里如何计算隐藏状态和输出。这里的激活函数使用了tanh函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，当元素在实数域上均匀分布时，tanh函数值的均值为0。

```{.python .input  n=38}
def rnn(inputs, H, *params):
    # inputs 和 outputs 皆为 num_steps 个形状为（batch_size, vocab_size）的矩阵。
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H
```

做个简单的测试来观察输出结果的个数，第一个输出的形状，和新状态的形状。

```{.python .input  n=39}
state = init_rnn_state(X.shape[0], num_hiddens, ctx=ctx)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, H_new = rnn(inputs, *state, *params)
len(outputs), outputs[0].shape, H_new.shape
```

## 定义预测函数

以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。这个函数稍显复杂，主要因为我们将循环神经单元`rnn`和输入索引变换网络输入的函数`get_inputs`设置成了可变项，这样在后面小节介绍其他循环神经网络（例如LSTM）和特征表示方法（例如Embedding）时能重复使用这个函数。

```{.python .input  n=15}
def predict_rnn(rnn, prefix, num_chars, params, num_hiddens, vocab_size, ctx,
                idx_to_char, char_to_idx, get_inputs, is_lstm=False):
    prefix = prefix.lower()
    state = init_rnn_state(1, num_hiddens, ctx, is_lstm)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix)):
        # 将上一时间步的输出作为当前时间步的输入。
        X = get_inputs(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态。
        (Y, *state) = rnn(X, *state, *params)
        # 下一个时间步的输入是 prefix 里的字符或者当前的最好预测字符。
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

验证一下这个函数。因为模型参数为随机值，所以预测结果也是随机的。

```{.python .input}
predict_rnn(rnn, '想要', 10, params, num_hiddens, vocab_size, 
            ctx, idx_to_char, char_to_idx, to_onehot)
```

## 裁剪梯度

循环神经网络中较容易出现梯度衰减或爆炸，其原因我们会在[下一节](bptt.md)解释。为了应对梯度爆炸，我们可以裁剪梯度（clipping gradient）。假设我们把所有模型参数梯度的元素拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后梯度

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

的$L_2$范数不超过$\theta$。

```{.python .input  n=16}
def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## 困惑度

语言模型里我们通常使用困惑度（perplexity）来评价模型的好坏。回忆一下[“Softmax回归”](../chapter_deep-learning-basics/softmax-regression.md)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

* 最佳情况下，模型总是把标签类别的概率预测为1。此时困惑度为1。
* 最坏情况下，模型总是把标签类别的概率预测为0。此时困惑度为正无穷。
* 基线情况下，模型总是预测所有类别的概率都相同。此时困惑度为类别数。

显然，任何一个有效模型的困惑度必须小于类别数。在本例中，困惑度必须小于词典中不同的字符数`vocab_size`。相对于交叉熵损失，困惑度的值更大，使得模型比较时更加清楚。例如“模型一比模型二的困惑度小1”比“模型一比模型二的交叉熵损失小0.01”感官上更加清楚一下。

## 定义模型训练函数

跟之前章节的训练模型函数相比，这里有以下几个不同。

1. 使用困惑度（perplexity）评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。

同样这个函数由于考虑到后面将介绍的循环神经网络，所以实现更长一些。

```{.python .input  n=17}
def train_and_predict_rnn(rnn, is_random_iter, num_epochs, num_steps,
                          num_hiddens, lr, clipping_theta, batch_size,
                          vocab_size, pred_period, pred_len, prefixes,
                          get_params, get_inputs, ctx, corpus_indices,
                          idx_to_char, char_to_idx, is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):        
        if not is_random_iter:  # 如使用相邻采样，在 epoch 开始时初始化隐藏变量。
            state = init_rnn_state(batch_size, num_hiddens, ctx, is_lstm)
        train_l_sum, train_l_cnt = nd.array([0], ctx=ctx), 0
        for X, Y in data_iter(corpus_indices, batch_size, num_steps, ctx):            
            if is_random_iter: # 如使用随机采样，在每个小批量更新前初始化隐藏变量。
                state = init_rnn_state(batch_size, num_hiddens, ctx, is_lstm)
            else:  # 否则需要使用 detach 函数从计算图分离隐藏状态变量。
                for s in state:
                    s.detach()
            with autograd.record():                
                X = get_inputs(X, vocab_size)
                # outputs 有 num_steps 个形状为 (batch_size, vocab_size) 的矩阵。
                (outputs, *state) = rnn(X, *state, *params)
                # 拼接之后形状为 (batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                # Y 的形状是 (batch_size, num_steps)，装置后再变成长 
                # batch * num_steps 的向量，这样跟输出的行一一对应。
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算分类误差。
                l = loss(outputs, y)
            l.backward()
            # 裁剪梯度后使用 SGD 更新权重。
            grad_clipping(params, clipping_theta, ctx)
            gb.sgd(params, lr, 1)
            train_l_sum += l.sum()
            train_l_cnt += l.size
        if (epoch+1) % pred_period == 0:
            print('\nepoch %d, perplexity %f'
                  % (epoch+1, (train_l_sum / train_l_cnt).exp().asscalar()))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    rnn, prefix, pred_len, params, num_hiddens, vocab_size,
                    ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))
```

以上介绍的函数，除去`get_params`外，均定义在`gluonbook`包中供后面章节调用。有了这些函数以后，我们就可以训练模型了。

## 训练模型并创作歌词

首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input}
num_epochs = 200
num_steps = 35
batch_size = 32
lr = 0.2
clipping_theta = 5
prefixes = ['分开', '不分开']
pred_period = 50
pred_len = 50
```

下面采用随机采样训练模型并创作歌词。

```{.python .input  n=18}
train_and_predict_rnn(rnn, True, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, to_onehot, ctx,
                      corpus_indices, idx_to_char, char_to_idx)
```

接下来采用相邻采样训练模型并创作歌词。

```{.python .input  n=19}
train_and_predict_rnn(rnn, False, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, to_onehot, ctx,
                      corpus_indices, idx_to_char, char_to_idx)
```

## 小结

* 我们可以应用基于字符级循环神经网络的语言模型来创作歌词。
* 时序数据采样方式包括随机采样和相邻采样。使用这两种方式的循环神经网络训练略有不同。
* 当训练循环神经网络时，为了应对梯度爆炸，我们可以裁剪梯度。
* 困惑度是对交叉熵损失函数做指数运算后得到的值。

## 练习

* 调调超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 不裁剪梯度，运行本节代码。结果会怎样？
* 将`pred_period`改为1，观察未充分训练的模型（困惑度高）是如何创作歌词的。你获得了什么启发？
* 将相邻采样改为不从计算图分离隐藏状态，运行时间有没有变化？
* 将本节中使用的激活函数替换成ReLU，重复本节的实验。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/989)

![](../img/qr_rnn.svg)
