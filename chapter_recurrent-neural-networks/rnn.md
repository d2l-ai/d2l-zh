# 循环神经网络

前两节介绍了语言模型和循环神经网络的设计。在本节中，我们将从零开始实现一个基于循环神经网络的语言模型，并应用它创作歌词。循环神经网络还有更广泛的应用。我们将在“自然语言处理”篇章中使用循环神经网络对不定长的文本序列分类，或把它翻译成不定长的另一语言的文本序列。


## 基于循环神经网络的语言模型

首先让我们简单回顾一下上一节描述的循环神经网络表达式。给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$），设该时间步隐藏状态为$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$（隐藏单元个数为$h$），输出层变量为$\boldsymbol{O}_t \in \mathbb{R}^{n \times q}$（输出个数为$q$），隐藏层的激活函数为$\phi$。循环神经网络的矢量计算表达式为

$$
\begin{aligned}
\boldsymbol{H}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h),\\
\boldsymbol{O}_t &= \boldsymbol{H}_t \boldsymbol{W}_{hy} + \boldsymbol{b}_y,
\end{aligned}
$$

其中隐藏层的权重$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重$\boldsymbol{W}_{hy} \in \mathbb{R}^{h \times q}$和偏差$\boldsymbol{b}_y \in \mathbb{R}^{1 \times q}$为循环神经网络的模型参数。有些文献所指的循环神经网络只含隐藏状态$\boldsymbol{H}_t$的计算表达式。




在语言模型中，输入个数$x$为任意词的特征向量长度（本节稍后将讨论）；输出个数$y$为语料库中所有可能的词的个数。对循环神经网络的输出做softmax运算，我们可以得到时间步$t$输出所有可能的词的概率分布$\hat{\boldsymbol{Y}}_t \in \mathbb{R}^{n \times q}$：

$$\hat{\boldsymbol{Y}}_t = \text{softmax}(\boldsymbol{O}_t).$$


由于隐藏状态$\boldsymbol{H}_t$捕捉了时间步1到时间步$t$的小批量输入$\boldsymbol{X}_1, \ldots, \boldsymbol{X}_t$的信息，$\hat{\boldsymbol{Y}}_t$可以批量表达语言模型中给定文本序列中过去词生成下一个词的条件概率。有了这些条件概率，语言模型可以计算任意文本序列的概率。


## 字符级循环神经网络


本节实验中的循环神经网络将每个字符视作词。我们有时将该模型称为字符级循环神经网络（character-level recurrent neural network）。
设小批量中样本数$n=1$，文本序列为“你”、“好”、“世”、“界”。为了表达给定文本序列中过去词生成下一个词的条件概率，我们需要把输入序列和标签序列分别设为“你”、“好”、“世”和“好”、“世”、“界”，如图6.2所示。


![基于循环神经网络的语言模型。输入序列和标签序列分别为“你”、“好”、“世”和“好”、“世”、“界”。](../img/rnn-train.svg)


当训练模型时，我们可以使用分类模型中常用的交叉熵损失函数计算各个时间步的损失。
在图6.2中，由于隐藏层中隐藏状态的循环迭代，时间步3的输出$\boldsymbol{O}_3$取决于文本序列“你”、“好”、“世”。
由于训练数据中该序列的下一个词为“界”，时间步3的损失将取决于该时间步基于序列“你好世”生成下一个词的概率分布与该时间步标签“界”。

## 创作歌词

在创作歌词的实验中，我们将应用基于字符级循环神经网络的语言模型。
与图6.2中的例子类似，我们将根据训练数据集的文本序列得到输入序列和标签序列。当模型训练好后，我们将以一种简单的方式创作歌词：根据给定的前缀，输出预测概率最大的下一个词；然后将该词附在前缀后继续输出预测概率最大的下一个词；如此循环。

创作歌词也可用到其他技术。例如，将输入拆分成以词语而不是字符为单位的序列、添加嵌入层（本章后面会介绍）或使用“自然语言处理”篇章中介绍的束搜索。



## 歌词数据集


我们使用周杰伦歌词数据集来训练模型作词。该数据集里包含了著名创作型歌手周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中歌曲的歌词。

首先导入实现所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import random
import zipfile
```

下面我们读取这个数据集，看看前50个字符是什么样的。

```{.python .input  n=1}
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt', encoding='utf-8') as f:
    corpus_chars = f.read()

corpus_chars[0:50]
```

看一下数据集中文本序列的长度。

```{.python .input  n=2}
len(corpus_chars)
```

接着我们稍微处理下数据集。为了打印方便，我们把换行符替换成空格。我们使用序列的前两万个字符训练模型。

```{.python .input  n=3}
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]
```

## 建立字符索引

我们将数据集里面所有不同的字符取出来做成词典。打印`vocab_size`，即词典中不同字符的个数。

```{.python .input  n=4}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size
```

然后，把每个字符转成从0开始的索引从而方便之后的使用。

```{.python .input  n=5}
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:40]
print('chars: \n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices: \n', sample)
```

## 时序数据的采样

同之前的实验一样，我们需要每次随机读取小批量样本和标签。不同的是，时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符：“想”、“要”、“有”、“直”、“升”。那么该样本的标签序列为这些字符分别在训练集中的下一个字符：“要”、“有”、“直”、“升”、“机”。

我们有两种方式对时序数据采样，分别是随机采样和相邻采样。

### 随机采样

下面代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`指每个小批量的样本数，`num_steps`为每个样本所包含的时间步数。
在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

```{.python .input  n=6}
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为输出的索引是相应输入的索引加一。
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    for i in range(epoch_size):
        # 每次读取 batch_size 个随机样本。
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        Y = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield X, Y
```

让我们输入一个从0到29的人工序列，设批量大小和时间步数分别为2和3，打印随机采样每次读取的小批量样本的输入`X`和标签`Y`。可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

```{.python .input  n=7}
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=3):
    print('X: ', X, '\nY:', Y, '\n')
```

### 相邻采样

除了对原始序列做随机采样之外，我们还可以使相邻的两个随机小批量在原始序列上的位置相毗邻。这时候，我们就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量输入，并如此循环下去。这对实现循环神经网络造成了两方面影响。一方面，
在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态。
另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。
为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图分离出来。

```{.python .input  n=8}
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    # 减一是因为输出的索引是相应输入的索引加一。
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```

让我们输入一个从0到29的人工序列，设批量大小和时间步数分别为2和3，打印相邻采样每次读取的小批量样本的输入`X`和标签`Y`。相邻的两个随机小批量在原始序列上的位置相毗邻。

```{.python .input  n=9}
my_seq = list(range(30))
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('X: ', X, '\nY:', Y, '\n')
```

## One-hot向量

为了用向量表示词，一个简单的办法是使用one-hot向量。
假设词典中不同字符的数量为$N$，每个字符可以和从0到$N-1$的连续整数一一对应。这些与字符对应的整数也叫字符的索引。
如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为`vocab_size`的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。因此，本节实验中循环神经网络的输入个数$x$是任意词的特征向量长度`vocab_size`。

下面分别展示了索引为0和2的one-hot向量。

```{.python .input  n=10}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

我们每次采样的小批量的形状是（`batch_size`, `num_steps`）。下面这个函数将其转换成`num_steps`个可以输入进网络的形状为（`batch_size`, `num_steps`）的矩阵。对于一个时间步数为`num_steps`的序列，每个批量输入$\boldsymbol{X} \in \mathbb{R}^{n \times x}$，其中$n=$ `batch_size`，$x=$`vocab_size`（one-hot向量长度）。

```{.python .input  n=11}
def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

get_inputs = to_onehot
inputs = get_inputs(X, vocab_size)
len(inputs), inputs[0].shape
```

## 初始化模型参数

接下来，我们初始化模型参数。隐藏单元个数 `num_hiddens`是一个超参数。

```{.python .input  n=12}
ctx = gb.try_gpu()
print('will use', ctx)

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size

def get_params():
    # 隐藏层参数。
    W_xh = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens),
                            ctx=ctx)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens),
                            ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数。
    W_hy = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs),
                            ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
```

## 定义模型

我们根据循环神经网络的表达式实现该模型。这里的激活函数使用了tanh函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，当元素在实数域上均匀分布时，tanh函数值的均值为0。

假设小批量中样本数为`batch_size`，时间步数为`num_steps`。
以下`rnn`函数的`inputs`和`outputs`皆为`num_steps` 个形状为（`batch_size`, `vocab_size`）的矩阵，隐藏状态`H`是一个形状为（`batch_size`, `num_hiddens`）的矩阵。

```{.python .input  n=13}
def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H
```

做个简单的测试：

```{.python .input  n=14}
state = nd.zeros(shape=(X.shape[0], num_hiddens), ctx=ctx)
params = get_params()
outputs, state_new = rnn(get_inputs(X.as_in_context(ctx), vocab_size), state,
                         *params)
len(outputs), outputs[0].shape, state_new.shape
```

## 定义预测函数

以下函数预测基于前缀`prefix`接下来的`num_chars`个字符。我们将用它根据训练得到的循环神经网络`rnn`来创作歌词。

```{.python .input  n=15}
def predict_rnn(rnn, prefix, num_chars, params, num_hiddens, vocab_size, ctx,
                idx_to_char, char_to_idx, get_inputs, is_lstm=False):
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    if is_lstm:
        # 当 RNN 使用 LSTM 时才会用到（后面章节会介绍），本节可以忽略。
        state_c = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        # 在序列中循环迭代隐藏状态。
        if is_lstm:
            # 当 RNN 使用 LSTM 时才会用到（后面章节会介绍），本节可以忽略。
            Y, state_h, state_c = rnn(get_inputs(X, vocab_size), state_h,
                                      state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X, vocab_size), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])
```

## 裁剪梯度

循环神经网络中较容易出现梯度衰减或爆炸。我们会在[下一节](bptt.md)中解释原因。为了应对梯度爆炸，我们可以裁剪梯度（clipping gradient）。假设我们把所有模型参数梯度的元素拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后梯度

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

的$L_2$范数不超过$\theta$。

```{.python .input  n=16}
def grad_clipping(params, state_h, Y, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
```

## 定义模型训练函数

跟之前章节的训练模型函数相比，这里有以下几个不同。

1. 使用困惑度（perplexity）评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。

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

    for epoch in range(1, num_epochs + 1):
        # 如使用相邻采样，隐藏变量只需在该 epoch 开始时初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
        train_l_sum = nd.array([0], ctx=ctx)
        train_l_cnt = 0
        for X, Y in data_iter(corpus_indices, batch_size, num_steps, ctx):
            # 如使用随机采样，读取每个随机小批量前都需要初始化隐藏变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, num_hiddens),
                                       ctx=ctx)
            # 如使用相邻采样，需要使用 detach 函数从计算图分离隐藏状态变量。
            else:
                state_h = state_h.detach()
                if is_lstm:
                    state_c = state_c.detach()       
            with autograd.record():
                # outputs 形状：(batch_size, vocab_size)。
                if is_lstm:
                    outputs, state_h, state_c = rnn(
                        get_inputs(X, vocab_size), state_h, state_c, *params) 
                else:
                    outputs, state_h = rnn(
                        get_inputs(X, vocab_size), state_h, *params)
                # 设 t_ib_j 为时间步 i 批量中的元素 j：
                # y 形状：（batch_size * num_steps,）
                # y = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]。
                y = Y.T.reshape((-1,))
                # 拼接 outputs，形状：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                l = loss(outputs, y)
            l.backward()
            # 裁剪梯度。
            grad_clipping(params, state_h, Y, clipping_theta, ctx)
            gb.sgd(params, lr, 1)
            train_l_sum = train_l_sum + l.sum()
            train_l_cnt += l.size
        if epoch % pred_period == 0:
            print('\nepoch %d, perplexity %f'
                  % (epoch, (train_l_sum / train_l_cnt).exp().asscalar()))
            for prefix in prefixes:
                print(' - ', predict_rnn(
                    rnn, prefix, pred_len, params, num_hiddens, vocab_size,
                    ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))
```

### 困惑度

回忆一下[“Softmax回归”](../chapter_deep-learning-basics/softmax-regression.md)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

* 最佳情况下，模型总是把标签类别的概率预测为1。此时困惑度为1。
* 最坏情况下，模型总是把标签类别的概率预测为0。此时困惑度为正无穷。
* 基线情况下，模型总是预测所有类别的概率都相同。此时困惑度为类别数。

显然，任何一个有效模型的困惑度必须小于类别数。在本例中，困惑度必须小于词典中不同的字符数`vocab_size`。


## 训练模型并创作歌词

以上介绍的`to_onehot`、`data_iter_random`、`data_iter_consecutive`、`grad_clipping`、`predict_rnn`和`train_and_predict_rnn`函数均定义在`gluonbook`包中供后面章节调用。有了这些函数以后，我们就可以训练模型了。

首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为100个字符的一段歌词。我们每过40个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input}
num_epochs = 200
num_steps = 35
batch_size = 32
lr = 0.2
clipping_theta = 5
prefixes = ['分开', '不分开']
pred_period = 40
pred_len = 100
```

下面采用随机采样训练模型并创作歌词。

```{.python .input  n=18}
train_and_predict_rnn(rnn, True, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, get_inputs, ctx,
                      corpus_indices, idx_to_char, char_to_idx)
```

接下来采用相邻采样训练模型并创作歌词。

```{.python .input  n=19}
train_and_predict_rnn(rnn, False, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, get_inputs, ctx,
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
