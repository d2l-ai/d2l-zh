# 门控循环单元（GRU）--- 从0开始

[上一节](bptt.md)中，我们介绍了循环神经网络中的梯度计算方法。我们发现，循环神经网络的隐含层变量梯度可能会出现衰减或爆炸。虽然[梯度裁剪](rnn-scratch.md)可以应对梯度爆炸，但无法解决梯度衰减的问题。因此，给定一个时间序列，例如文本序列，循环神经网络在实际中其实较难捕捉两个时刻距离较大的文本元素（字或词）之间的依赖关系。

门控循环神经网络（gated recurrent neural networks）的提出，是为了更好地捕捉时序数据中间隔较大的依赖关系。其中，门控循环单元（gated recurrent unit，简称GRU）是一种常用的门控循环神经网络。它由Cho、van Merrienboer、 Bahdanau和Bengio在2014年被提出。


## 门控循环单元

我们先介绍门控循环单元的构造。它比循环神经网络中的隐含层构造稍复杂一点。

### 重置门和更新门

门控循环单元的隐含状态只包含隐含层变量$\mathbf{H}$。假定隐含状态长度为$h$，给定时刻$t$的一个样本数为$n$特征向量维度为$x$的批量数据$\mathbf{X}_t \in \mathbb{R}^{n \times x}$和上一时刻隐含状态$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$，重置门（reset gate）$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和更新门（update gate）$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$的定义如下：

$$\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r)$$

$$\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z)$$

其中的$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。函数$\sigma$自变量中的三项相加使用了[广播](../chapter_crashcourse/ndarray.md)。

需要注意的是，重置门和更新门使用了值域为$[0, 1]$的函数$\sigma(x) = 1/(1+\text{exp}(-x))$。因此，重置门$\mathbf{R}_t$和更新门$\mathbf{Z}_t$中每个元素的值域都是$[0, 1]$。


### 候选隐含状态

我们可以通过元素值域在$[0, 1]$的更新门和重置门来控制隐含状态中信息的流动：这通常可以应用按元素乘法符$\odot$。门控循环单元中的候选隐含状态$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$使用了值域在$[-1, 1]$的双曲正切函数tanh做激活函数：

$$\tilde{\mathbf{H}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{R}_t \odot \mathbf{H}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)$$

其中的$\mathbf{W}_{xh} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。

需要注意的是，候选隐含状态使用了重置门来控制包含过去时刻信息的上一个隐含状态的流入。如果重置门近似0，上一个隐含状态将被丢弃。因此，重置门提供了丢弃与未来无关的过去隐含状态的机制。


### 隐含状态

隐含状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$的计算使用更新门$\mathbf{Z}_t$来对上一时刻的隐含状态$\mathbf{H}_{t-1}$和当前时刻的候选隐含状态$\tilde{\mathbf{H}}_t$做组合，公式如下：

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t$$

需要注意的是，更新门可以控制过去的隐含状态在当前时刻的重要性。如果更新门一直近似1，过去的隐含状态将一直通过时间保存并传递至当前时刻。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时序数据中间隔较大的依赖关系。

我们对门控循环单元的设计稍作总结：

* 重置门有助于捕捉时序数据中短期的依赖关系。
* 更新门有助于捕捉时序数据中长期的依赖关系。


输出层的设计可参照[循环神经网络](rnn-scratch.md)中的描述。


## 实验


为了实现并展示门控循环单元，我们依然使用周杰伦歌词数据集来训练模型作词。这里除门控循环单元以外的实现已在[循环神经网络](rnn-scratch.md)中介绍。


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
    # 隐含层
    W_xz = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hz = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_z = nd.zeros(hidden_dim, ctx=ctx)
    
    W_xr = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hr = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_r = nd.zeros(hidden_dim, ctx=ctx)

    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
```

## 定义模型

我们将前面的模型公式翻译成代码。

```{.python .input  n=4}
def gru_rnn(inputs, H, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:        
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + R * nd.dot(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)
```

### 训练模型

下面我们开始训练模型。我们假定谱写歌词的前缀分别为“分开”、“不分开”和“战争中部队”。这里采用的是相邻批量采样实验门控循环单元谱写歌词。

```{.python .input  n=5}
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

utils.train_and_predict_rnn(rnn=gru_rnn, is_random_iter=False, epochs=200,
                            num_steps=35, hidden_dim=hidden_dim, 
                            learning_rate=0.2, clipping_norm=5,
                            batch_size=32, pred_period=20, pred_len=100,
                            seqs=seqs, get_params=get_params,
                            get_inputs=get_inputs, ctx=ctx,
                            corpus_indices=corpus_indices,
                            idx_to_char=idx_to_char, char_to_idx=char_to_idx)
```

可以看到一开始学到简单的字符，然后简单的词，接着是复杂点的词，然后看上去似乎像个句子了。

## 结论

* 门控循环单元的提出是为了更好地捕捉时序数据中间隔较大的依赖关系。
* 重置门有助于捕捉时序数据中短期的依赖关系。
* 更新门有助于捕捉时序数据中长期的依赖关系。


## 练习

* 调调参数（例如数据集大小、序列长度、隐含状态长度和学习率），看看对运行时间、perplexity和预测的结果造成的影响。
* 在相同条件下，比较门控循环单元和循环神经网络的运行效率。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4042)
