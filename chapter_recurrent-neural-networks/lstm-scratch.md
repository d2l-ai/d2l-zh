# 长短期记忆（LSTM）--- 从0开始

[上一节](bptt.md)中，我们介绍了循环神经网络中的梯度计算方法。我们发现，循环神经网络的隐含层变量梯度可能会出现衰减或爆炸。虽然[梯度裁剪](rnn-scratch.md)可以应对梯度爆炸，但无法解决梯度衰减的问题。因此，给定一个时间序列，例如文本序列，循环神经网络在实际中其实较难捕捉两个时刻距离较大的文本元素（字或词）之间的依赖关系。

为了更好地捕捉时序数据中间隔较大的依赖关系，我们介绍了一种常用的门控循环神经网络，叫做[门控循环单元](gru-scratch.md)。本节将介绍另一种常用的门控循环神经网络，长短期记忆（long short-term memory，简称LSTM）。它由Hochreiter和Schmidhuber在1997年被提出。事实上，它比门控循环单元的结构稍微更复杂一点。


## 长短期记忆

我们先介绍长短期记忆的构造。长短期记忆的隐含状态包括隐含层变量$\mathbf{H}$和细胞$\mathbf{C}$（也称记忆细胞）。它们形状相同。


### 输入门、遗忘门和输出门


假定隐含状态长度为$h$，给定时刻$t$的一个样本数为$n$特征向量维度为$x$的批量数据$\mathbf{X}_t \in \mathbb{R}^{n \times x}$和上一时刻隐含状态$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$，输入门（input gate）$\mathbf{I}_t \in \mathbb{R}^{n \times h}$、遗忘门（forget gate）$\mathbf{F}_t \in \mathbb{R}^{n \times h}$和输出门（output gate）$\mathbf{O}_t \in \mathbb{R}^{n \times h}$的定义如下：

$$\mathbf{I}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)$$

$$\mathbf{F}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)$$

$$\mathbf{O}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)$$

其中的$\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。函数$\sigma$自变量中的三项相加使用了[广播](../chapter_crashcourse/ndarray.md)。

和[门控循环单元](gru-scratch.md)中的重置门和更新门一样，这里的输入门、遗忘门和输出门中每个元素的值域都是$[0, 1]$。


### 候选细胞

和[门控循环单元](gru-scratch.md)中的候选隐含状态一样，长短期记忆中的候选细胞$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$也使用了值域在$[-1, 1]$的双曲正切函数tanh做激活函数：

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)$$

其中的$\mathbf{W}_{xc} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_c \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。


### 细胞

我们可以通过元素值域在$[0, 1]$的输入门、遗忘门和输出门来控制隐含状态中信息的流动：这通常可以应用按元素乘法符$\odot$。当前时刻细胞$\mathbf{C}_t \in \mathbb{R}^{n \times h}$的计算组合了上一时刻细胞和当前时刻候选细胞的信息，并通过遗忘门和输入门来控制信息的流动：

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t$$

需要注意的是，如果遗忘门一直近似1且输入门一直近似0，过去的细胞将一直通过时间保存并传递至当前时刻。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时序数据中间隔较大的依赖关系。


### 隐含状态

有了细胞以后，接下来我们还可以通过输出门来控制从细胞到隐含层变量$\mathbf{H}_t \in \mathbb{R}^{n \times h}$的信息的流动：

$$\mathbf{H}_t = \mathbf{O}_t \odot \text{tanh}(\mathbf{C}_t)$$

需要注意的是，当输出门近似1，细胞信息将传递到隐含层变量；当输出门近似0，细胞信息只自己保留。



输出层的设计可参照[循环神经网络](rnn-scratch.md)中的描述。


## 实验


为了实现并展示门控循环单元，我们依然使用周杰伦歌词数据集来训练模型作词。这里除长短期记忆以外的实现已在[循环神经网络](rnn-scratch.md)中介绍。


### 数据处理

我们先读取并对数据集做简单处理。

```{.python .input  n=1}
import zipfile
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
corpus_indices = [char_to_idx[char] for char in corpus_chars]

vocab_size = len(char_to_idx)
print('vocab size:', vocab_size)
```

我们使用onehot来将字符索引表示成向量。

```{.python .input  n=2}
def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]
```

### 初始化模型参数

以下部分对模型参数进行初始化。参数`hidden_dim`定义了隐含状态的长度。

```{.python .input  n=3}
import mxnet as mx

# 尝试使用GPU
import sys
sys.path.append('..')
from mxnet import nd
import utils
ctx = utils.try_gpu()
print('Will use', ctx)

input_dim = vocab_size
# 隐含状态长度
hidden_dim = 256
output_dim = vocab_size
std = .01

def get_params():
    # 输入门参数
    W_xi = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hi = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_i = nd.zeros(hidden_dim, ctx=ctx)
    
    # 遗忘门参数
    W_xf = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hf = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_f = nd.zeros(hidden_dim, ctx=ctx)
    
    # 输出门参数
    W_xo = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_ho = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_o = nd.zeros(hidden_dim, ctx=ctx)

    # 候选细胞参数
    W_xc = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hc = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_c = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
```

## 定义模型

我们将前面的模型公式翻译成代码。

```{.python .input  n=4}
def lstm_rnn(inputs, state_h, state_c, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hy, b_y] = params

    H = state_h
    C = state_c
    outputs = []
    for X in inputs:        
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H, C)
```

### 训练模型

下面我们开始训练模型。我们假定谱写歌词的前缀分别为“分开”、“不分开”和“战争中部队”。这里采用的是相邻批量采样实验门控循环单元谱写歌词。

```{.python .input  n=5}
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

utils.train_and_predict_rnn(rnn=lstm_rnn, is_random_iter=False, epochs=200,
                            num_steps=35, hidden_dim=hidden_dim, 
                            learning_rate=0.2, clipping_norm=5,
                            batch_size=32, pred_period=20, pred_len=100,
                            seqs=seqs, get_params=get_params,
                            get_inputs=get_inputs, ctx=ctx,
                            corpus_indices=corpus_indices,
                            idx_to_char=idx_to_char, char_to_idx=char_to_idx,
                            is_lstm=True)
```

可以看到一开始学到简单的字符，然后简单的词，接着是复杂点的词，然后看上去似乎像个句子了。

## 结论

* 长短期记忆的提出是为了更好地捕捉时序数据中间隔较大的依赖关系。
* 长短期记忆的结构比门控循环单元的结构较复杂。


## 练习

* 调调参数（例如数据集大小、序列长度、隐含状态长度和学习率），看看对运行时间、perplexity和预测的结果造成的影响。
* 在相同条件下，比较长短期记忆和门控循环单元以及循环神经网络的运行效率。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4042)
