# 门控循环单元（GRU） --- 从0开始

[上一节](bptt.md)中，我们介绍了循环神经网络中的梯度计算方法。我们发现，循环神经网络的隐含层变量梯度可能会出现衰减或爆炸。虽然[梯度裁剪](rnn-scratch.md)可以应对梯度爆炸，但无法解决梯度衰减的问题。因此，给定一个时间序列，例如文本序列，循环神经网络在实际中较难捕捉两个时刻距离较大的文本元素（字或词）之间的依赖关系。

门控循环神经网络（gated recurrent neural networks）的提出，是为了更好地捕捉时序数据中间隔较大的依赖关系。其中，门控循环单元（gated recurrent unit，简称GRU）是一种常用的门控循环神经网络。它由Cho等于2014年被提出。


## 门控循环单元

我们先介绍门控循环单元的构造。它比循环神经网络中的隐含层构造稍复杂一点。

### 重置门和更新门

门控循环单元的隐含状态只包含隐含层变量$\mathbf{H}$。假定隐含状态长度为$h$，给定时刻$t$的一个样本数为$n$特征向量维度为$x$的批量数据$\mathbf{X}_t \in \mathbb{R}^{n \times x}$和上一时刻隐含状态$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$，重置门（reset gate）$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和更新门（update gate）$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$的定义如下：

$$\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r)$$

$$\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z)$$

其中的$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。函数$\sigma$自变量中的三项相加使用了[广播](../chapter_crashcourse/ndarray.md)。

需要注意的是，重置门和更新门使用了值域为$[0, 1]$的函数$\sigma(x) = 1/(1+\text{exp}(-x))$。因此，重置门$\mathbf{R}_t$和更新门$\mathbf{Z}_t$中每个元素的值域都是$[0, 1]$。


### 候选隐含状态

我们可以通过元素值域在$[0, 1]$的更新门和重置门来控制隐含状态中信息的流动：这通常可以应用按元素乘法符$\odot$。门控循环单元中的候选隐含状态$\tilde{\mathbf{H}_t} \in \mathbb{R}^{n \times h}$使用了值域在$[-1, 1]$的双曲正切函数tanh做激活函数：

$$\tilde{\mathbf{H}_t} = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{R}_t \odot \mathbf{H}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)$$

其中的$\mathbf{W}_{xh} \in \mathbb{R}^{x \times h}$和$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$是可学习的权重参数，$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$是可学习的偏移参数。

需要注意的是，候选隐含状态使用了重置门来控制包含过去时刻信息的上一个隐含状态的流入。如果重置门近似0，上一个隐含状态将被丢弃。因此，重置门提供了丢弃与未来无关的过去隐含状态的机制。


### 隐含状态

隐含状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$的计算使用更新门$\mathbf{Z}_t$来对上一时刻的隐含状态$\mathbf{H}_{t-1}$和当前时刻的候选隐含状态$\tilde{\mathbf{H}_t}$做线性组合，公式如下：

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}_t}$$

需要注意的是，更新门可以控制过去的隐含状态在当前时刻究竟有多重要。如果更新门近似1，




$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h)$$

假定隐含层长度为$h$，那么其中的权重参数的尺寸为$\mathbf{W}_{xh} \in \mathbb{R}^{x \times h}$。偏移参数 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$在与前一项$\mathbf{X} \mathbf{W}_{xh} \in \mathbb{R}^{n \times h}$ 相加时使用了[广播](../chapter_crashcourse/ndarray.md)。这个隐含层的输出的尺寸为$\mathbf{H} \in \mathbb{R}^{n \times h}$。

把隐含层的输出$\mathbf{H}$作为输出层的输入，最终的输出

$$\hat{\mathbf{Y}} = \text{softmax}(\mathbf{H} \mathbf{W}_{hy} + \mathbf{b}_y)$$

假定每个样本对应的输出向量维度为$y$，其中 $\hat{\mathbf{Y}} \in \mathbb{R}^{n \times y}, \mathbf{W}_{hy} \in \mathbb{R}^{h \times y}, \mathbf{b}_y \in \mathbb{R}^{1 \times y}$且两项相加使用了[广播](../chapter_crashcourse/ndarray.md)。


将上面网络改成循环神经网络，我们首先对输入输出加上时间戳$t$。假设$\mathbf{X}_t \in \mathbb{R}^{n \times x}$是序列中的第$t$个批量输入（样本数为$n$，每个样本的特征向量维度为$x$），对应的隐含层输出是隐含状态$\mathbf{H}_t  \in \mathbb{R}^{n \times h}$（隐含层长度为$h$），而对应的最终输出是$\hat{\mathbf{Y}}_t \in \mathbb{R}^{n \times y}$（每个样本对应的输出向量维度为$y$）。在计算隐含层的输出的时候，循环神经网络只需要在前馈神经网络基础上加上跟前一时间$t-1$输入隐含层$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$的加权和。为此，我们引入一个新的可学习的权重$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h)$$

输出的计算跟前面一致：

$$\hat{\mathbf{Y}}_t = \text{softmax}(\mathbf{H}_t \mathbf{W}_{hy}  + \mathbf{b}_y)$$

一开始我们提到过，隐含状态可以认为是这个网络的记忆。该网络中，时刻$t$的隐含状态就是该时刻的隐含层变量$\mathbf{H}_t$。它存储前面时间里面的信息。我们的输出是只基于这个状态。最开始的隐含状态里的元素通常会被初始化为0。


## 周杰伦歌词数据集


为了实现并展示循环神经网络，我们使用周杰伦歌词数据集来训练模型作词。该数据集里包含了著名创作型歌手周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》所有歌曲的歌词。


下面我们读取这个数据并看看前面49个字符（char）是什么样的：

我们看一下数据集里的字符数。

接着我们稍微处理下数据集。为了打印方便，我们把换行符替换成空格，然后截去后面一段使得接下来的训练会快一点。

```{.python .input  n=1}
import zipfile
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]
```

## 字符的数值表示

先把数据里面所有不同的字符拿出来做成一个字典：

```{.python .input  n=2}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

vocab_size = len(char_to_idx)

print('vocab size:', vocab_size)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "vocab size: 1465\n"
 }
]
```

然后可以把每个字符转成从0开始的索引(index)来方便之后的使用。

```{.python .input  n=3}
corpus_indices = [char_to_idx[char] for char in corpus_chars]
```

## 时序数据的批量采样

同之前一样我们需要每次随机读取一些（`batch_size`个）样本和其对用的标号。这里的样本跟前面有点不一样，这里一个样本通常包含一系列连续的字符（前馈神经网络里可能每个字符作为一个样本）。

如果我们把序列长度（`num_steps`）设成5，那么一个可能的样本是“想要有直升”。其对应的标号仍然是长为5的序列，每个字符是对应的样本里字符的后面那个。例如前面样本的标号就是“要有直升机”。


### 随机批量采样

下面代码每次从数据里随机采样一个批量。

为了便于理解时序数据上的随机批量采样，让我们输入一个从0到29的人工序列，看下读出来长什么样：

由于各个采样在原始序列上的位置是随机的时序长度为`num_steps`的连续数据点，相邻的两个随机批量在原始序列上的位置不一定相毗邻。因此，在训练模型时，读取每个随机时序批量前需要重新初始化隐含状态。


### 相邻批量采样

除了对原序列做随机批量采样之外，我们还可以使相邻的两个随机批量在原始序列上的位置相毗邻。

相同地，为了便于理解时序数据上的相邻批量采样，让我们输入一个从0到29的人工序列，看下读出来长什么样：

由于各个采样在原始序列上的位置是毗邻的时序长度为`num_steps`的连续数据点，因此，使用相邻批量采样训练模型时，读取每个时序批量前，我们需要将该批量最开始的隐含状态设为上个批量最终输出的隐含状态。在同一个epoch中，隐含状态只需要在该epoch开始的时候初始化。


## Onehot编码

注意到每个字符现在是用一个整数来表示，而输入进网络我们需要一个定长的向量。一个常用的办法是使用onehot来将其表示成向量。也就是说，如果一个字符的整数值是$i$, 那么我们创建一个全0的长为`vocab_size`的向量，并将其第$i$位设成1。该向量就是对原字符的onehot编码。

记得前面我们每次得到的数据是一个`batch_size * num_steps`的批量。下面这个函数将其转换成`num_steps`个可以输入进网络的`batch_size * vocab_size`的矩阵。对于一个长度为`num_steps`的序列，每个批量输入$\mathbf{X} \in \mathbb{R}^{n \times x}$，其中$n=$ `batch_size`，而$x=$`vocab_size`（onehot编码向量维度）。

```{.python .input  n=4}
def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]
```

## 初始化模型参数

对于序列中任意一个时间戳，一个字符的输入是维度为`vocab_size`的onehot编码向量，对应输出是预测下一个时间戳为词典中任意字符的概率，因而该输出是维度为`vocab_size`的向量。

当序列中某一个时间戳的输入为一个样本数为`batch_size`（对应模型定义中的$n$）的批量，每个时间戳上的输入和输出皆为尺寸`batch_size * vocab_size`（对应模型定义中的$n \times x$）的矩阵。假设每个样本对应的隐含状态的长度为`hidden_dim`（对应模型定义中隐含层长度$h$），根据矩阵乘法定义，我们可以推断出模型隐含层和输出层中各个参数的尺寸。

```{.python .input  n=5}
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

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Will use gpu(0)\n"
 }
]
```

## 定义模型

当序列中某一个时间戳的输入为一个样本数为`batch_size`的批量，而整个序列长度为`num_steps`时，以下`rnn`函数的`inputs`和`outputs`皆为`num_steps` 个尺寸为`batch_size * vocab_size`的矩阵，隐含变量$\mathbf{H}$是一个尺寸为`batch_size * hidden_dim`的矩阵。该隐含变量$\mathbf{H}$也是循环神经网络的隐含状态`state`。

我们将前面的模型公式翻译成代码。这里的激活函数使用了按元素操作的双曲正切函数

$$\text{tanh}(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

```{.python .input  n=6}
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

做个简单的测试：

## 预测序列

在做预测时我们只需要给定时间0的输入和起始隐含变量。然后我们每次将上一个时间的输出作为下一个时间的输入。

![](../img/rnn_3.png)

## 梯度剪裁

我们在[正向传播和反向传播](../chapter_supervised-learning/backprop.md)中提到，
训练神经网络往往需要依赖梯度计算的优化算法，例如我们之前介绍的[随机梯度下降](../chapter_supervised-learning/linear-regression-scratch.md)。
而在循环神经网络的训练中，当每个时序训练数据样本的时序长度`num_steps`较大或者时刻$t$较小，目标函数有关$t$时刻的隐含层变量梯度较容易出现衰减（valishing）或爆炸（explosion）。我们会在[下一节](bptt.md)详细介绍出现该现象的原因。

为了应对梯度爆炸，一个常用的做法是如果梯度特别大，那么就投影到一个比较小的尺度上。假设我们把所有梯度接成一个向量 $\boldsymbol{g}$，假设剪裁的阈值是$\theta$，那么我们这样剪裁使得$\|\boldsymbol{g}\|$不会超过$\theta$：

$$ \boldsymbol{g} = \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

## 训练模型

下面我们可以还是训练模型。跟前面前置网络的教程比，这里有以下几个不同。

1. 通常我们使用困惑度（Perplexity）这个指标。
2. 在更新前我们对梯度做剪裁。
3. 在训练模型时，对时序数据采用不同批量采样方法将导致隐含变量初始化的不同。

### 困惑度（Perplexity）

回忆以下我们之前介绍的[交叉熵损失函数](../chapter_supervised-learning/softmax-regression-scratch.md)。在语言模型中，该损失函数即被预测字符的对数似然平均值的相反数：

$$\text{loss} = -\frac{1}{N} \sum_{i=1}^N \log p_{\text{target}_i}$$

其中$N$是预测的字符总数，$p_{\text{target}_i}$是在第$i$个预测中真实的下个字符被预测的概率。

而这里的困惑度可以简单的认为就是对交叉熵做exp运算使得数值更好读。

为了解释困惑度的意义，我们先考虑一个完美结果：模型总是把真实的下个字符的概率预测为1。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 1$。这种完美情况下，困惑度值为1。

我们再考虑一个基线结果：给定不重复的字符集合$W$及其字符总数$|W|$，模型总是预测下个字符为集合$W$中任一字符的概率都相同。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 1/|W|$。这种基线情况下，困惑度值为$|W|$。

最后，我们可以考虑一个最坏结果：模型总是把真实的下个字符的概率预测为0。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 0$。这种最坏情况下，困惑度值为正无穷。

任何一个有效模型的困惑度值必须小于预测集中元素的数量。在本例中，困惑度必须小于字典中的字符数$|W|$。如果一个模型可以取得较低的困惑度的值（更靠近1），通常情况下，该模型预测更加准确。

我们先采用随机批量采样实验循环神经网络谱写歌词。我们假定谱写歌词的前缀分别为“分开”、“不分开”和“战争中部队”。

我们再采用相邻批量采样实验循环神经网络谱写歌词。

```{.python .input  n=7}
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

utils.train_and_predict_rnn(rnn=gru_rnn, is_random_iter=False, epochs=200,
                            num_steps=35, hidden_dim=hidden_dim, 
                            learning_rate=0.2, clipping_theta=5,
                            batch_size=32, pred_period=20, pred_len=100,
                            seqs=seqs, get_params=get_params,
                            get_inputs=get_inputs, ctx=ctx,
                            corpus_indices=corpus_indices,
                            idx_to_char=idx_to_char, char_to_idx=char_to_idx)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 20. Perplexity 274.044136\n -  \u5206\u5f00 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\n -  \u4e0d\u5206\u5f00 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\u7684\u6211\u7684 \u6211\u4e0d\u7684\u6211\n\nEpoch 40. Perplexity 106.089353\n -  \u5206\u5f00 \u6211\u60f3\u8981\u4f60\u7684\u7231\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u8981\u4f60\u7684\u7231\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u7684\u7231\u4f60\u5728\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u75af\u72c2\u7684\u53ef\u7231\n\nEpoch 60. Perplexity 29.577557\n -  \u5206\u5f00 \u4e00\u76f4\u4e00\u76f4\u70ed \u4e09\u54fc\u54c8\u516e \u4f60\u5728\u6211\u7528 \u8bf4\u4e0d\u53bb \u522b\u4e0d\u662f \u522b\u4e0d\u662f \u4e0d\u60f3\u518d\u8fd9\u6837\u7740\u6211 \u522b\u4e0d\u518d\u518d\u60f3 \u6211\u4e0d\u80fd\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\n -  \u4e0d\u5206\u5f00 \u4e3a\u4ec0\u4e48\u6211\u4eec \u8bf4\u4e0d\u662f \u4e0d\u60f3\u518d\u8fd9\u6837\u7740\u6211 \u4e0d\u8981\u6211 \u4f60\u7231\u6211 \u60f3\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u4ed6\u4e0d\u4f11 \u522b\u6c89\u9ed8 \u662f\u4e0d\u662f \u522b\u4e0d\u53bb \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \u7684\u7075\u9b42 \n\nEpoch 80. Perplexity 8.342536\n -  \u5206\u5f00 \u8fd9\u6837\u65f6\u7684\u592a\u6d3b \u8ba9\u6211\u60f3\u8981\u4f60 \u662f\u6211\u7684\u5047\u4e0d\u5012 \u6211\u60f3\u4f60 \u4f60\u7231\u6211 \u60f3 \u7b80\uff01\u7b80\uff01\u5355\uff01\uff01\uff01 \u7231~~~~~~~~~~ \u6211\u60f3\u8981\u4f60\u60f3\u56de\u5230 \u4f46\u8fd9\u6837\u7684\u592a\u6d3b \u6211\u7231\u4f60 \u4f60\u7231\u6211 \u60f3 \u7b80\uff01\u7b80\uff01\u5355\uff01\u5355\uff01 \u7231~~~~~~~~~~ \u6211\u60f3\u8981\n -  \u4e0d\u5206\u5f00  \u7ecf\u5728\u4f60\u7684\u592a\u5feb \u6211\u5f88\u60f3\u8981\u7684\u56fd\u6a21 \u60f3\u8981\u4f60\u8bf4\u7684\u753b\u9762 \u6ca1\u6709\u60f3\u8bf4\u4f60\u7684\u753b\u9762 \u60f3\u8981\u4f60\u7684\u8138 \u5728\u4e00\u79cd\u5473\u9053\u53eb\u505a\u5bb6 \u9646\u7fbd\u6ce1\u7684\u8336 \u542c\u8bf4\u540d\u548c\u5229\u90fd\u4e0d\u62ff \u9646\u7fbd\u6ce1\u7684\u8336 \u6709\u4e00\u79cd\u5473\u9053\u53eb\u505a\u5bb6 \u9646\u7fbd\u6ce1\u7684\u8336 \u50cf\u5e45\u6cfc\u58a8\u7684\u5c71\u6c34\u753b \u4e00\u671d\u53c8\u91cd\u6765\u7684\u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u8fd9\u65f6 \u4e00\u9635\u4e24\u9897\u4e09\u9897\u56db\u9897 \u8fde\u6210\u7ebf\u80cc\u8457\u80cc\u9ed8\u9ed8\u9ed8\u8bb8\u5fc3\u613f \u770b\u8fdc\u661f \u4e00\u9897\u4e24\u9897\u4e09\u9897\u56db\u9897 \u8fde\u6210\u7ebf\u80cc\u8457\u80cc\u9ed8\u9ed8\u9ed8\u8bb8\u5fc3\u613f \u770b\u8fdc\u661f \u4e00\u9897\u4e24\u9897\u4e09\u9897\u56db\u9897 \u8fde\u6210\u7ebf\u80cc\u8457\u80cc\u9ed8\u9ed8\u9ed8\u8bb8\u5fc3\u613f \u770b\u8fdc\u661f \u4e00\u9897\u4e24\u9897\u4e09\u9897\u56db\u9897 \u8fde\u6210\u7ebf\u80cc\u8457\u80cc\u9ed8\u9ed8\u9ed8\u8bb8\n\nEpoch 100. Perplexity 3.207470\n -  \u5206\u5f00 \u8fd9\u6837\u5b50 \u4e00\u76f4\u8d70 \u6211\u60f3\u5c31\u8fd9\u6837\u7275\u7740\u4f60\u7684\u624b\u4e0d\u653e\u5f00 \u7231\u53ef\u4e0d\u53ef\u4ee5\u7b80\u7b80\u5355\u5355\u6ca1\u6709\u4f24\u5bb3 \u4f60 \u9760\u7740\u6211\u7684\u80a9\u8180 \u4f60 \u5728\u6211\u80f8\u53e3\u7761\u8457 \u50cf\u8fd9\u6837\u7684\u751f\u6d3b \u6211\u7231\u4f60 \u4f60\u7231\u6211 \u60f3 \u7b80\uff01\u7b80\uff01\u5355\uff01\u5355\uff01 \u7231~~~~~~~~~~ \u6211\u60f3\u8981\u4f60\u7684\u5fae\u7b11\u6bcf\n -  \u4e0d\u5206\u5f00 \u6cea\u8fc7\u53bb\u79cd \u5982\u679c\u4e86\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u4e3a\u8fd9\u6837) \u8fd9\u4e0d\u5230\u8fc7\u53bb \u8bd5\u7740\u4e0d\u6491\u4e0d\u61c2 \u4e00\u8eab\u6b63\u6c14 \u4ed6\u4eec\u513f\u5b50\u6211\u4e60\u60ef \u4ece\u5c0f\u5c31\u8033\u6fe1\u76ee\u67d3 \u4ec0\u4e48\u5200\u67aa\u8ddf\u68cd\u68d2 \u6211\u90fd\u800d\u7684\u6709\u6a21\u6709\u6837 \u4ec0\u4e48\u5175\u5668\u6700\u4e0d\u4f1a \u60f3\u4f60\u90a3\u5168\u662f\u5f00\u4e86\u8fd9\u6837\u638f \u6cea\u8fc7\u4e86\u5927\u975e\u4e8b\u5927\u6211\u7684\u611f\u8df3\u666f\u8c61 \u6c89\u6ca6\u5047\u8c61 \u4f60\u53ea\u4f1a\u611f\u5230\u66f4\u52a0\n\nEpoch 120. Perplexity 1.749342\n -  \u5206\u5f00 \u8fd9\u6837\u5bf9 \u544a\u8bc9\u540e \u4e00\u4e5d\u56db\u4e09 \u56de\u5934\u770b \u7684\u7247\u6bb5 \u6709\u4e00\u4e9b\u98ce\u971c \u8001\u5531\u76d8 \u65e7\u76ae\u7bb1 \u88c5\u6ee1\u4e86\u660e\u4fe1\u7247\u7684\u94c1\u76d2\u91cc\u85cf\u8457\u4e00\u7247\u73ab\u7470\u82b1\u74e3 \u5bf9\u4e0d\u8d77 \u5e7f\u8ba9\u6211\u7684\u5730\u7403\u53d8\u6697 (\u8857\u89d2\u7684\u6d88\u9632\u6813\u4e0a\u7684\u7ea2\u8272\u6cb9\u6f06 \u53cd\u5c04\u51fa\u513f\u65f6\u5929\u771f\u7684\u5b09\u620f\u6a21\u6837) \u88ab\u671f\u5f85 \u88ab\n -  \u4e0d\u5206\u5f00 \u6cea\u5f71\u65f6\u95f4 \u8bf4\u4f60\u7231\u7a7a \u8bf4\u6ca1\u6709\u4e9b\u70e6 \u6709\u4e9b\u4e8b\u4e0d\u51c6\u6211\u518d\u63d0 \u6211\u7559\u7740\u966a\u4f60 \u6700\u540e\u7684\u8ddd\u79bb \u662f\u4f60\u7684\u4fa7\u8138\u5012\u5728\u6211\u7684\u6000\u91cc \u4f60\u6162\u6162\u7761\u53bb \u6211\u6447\u4e0d\u9192\u4f60 \u6cea\u6c34\u5728\u6218\u58d5\u91cc\u51b3\u4e86\u5824 \u5728\u4eba\u7684\u5e8f\u5728\u5e72\u67af \u4e00\u6b21\u6765\u5b83\u5730\u5a18 \u6211\u7ffb\u9605\u5730\u62c6\u5c01\u5c06\u957f\u6c5f\u6c34\u638f\u7a7a \u4eba\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u5f53\u5e74 \u4e00\u76f4\u4e24\u9897\u4e09\u9897\u56db\u9897 \u8fde\u6210\u7ebf\u80cc\u8457\u80cc\u9ed8\u9ed8\u8bb8\u4e0b\u5fc3\u613f \u770b\u8fdc\u65b9\u7684\u661f\u5982\u679c\u542c\u7684\u89c1 \u5b83\u4e00\u5b9a\u5b9e\u73b0\u5b83\u4e00\u5b9a\u5b9e\u73b0 \u8f7d\u8457\u4f60 \u5f77\u5f7f\u8f7d\u8457\u9633\u5149 \u4e0d\u7ba1\u5230\u54ea\u91cc\u90fd\u662f\u6674\u5929 \u8774\u5fc3\u5728\u5927\u5b57\u523b\u5927\u4e86\u6c38\u8fdc \u90a3\u5df2\u98ce\u5316\u5343\u5e74\u7684\u8a93\u8a00 \u4e00\u5207\u53c8\u91cd\u6f14 \u6211\u770b\u5230\u8fd9\u91cc\n\nEpoch 140. Perplexity 1.262575\n -  \u5206\u5f00 \u8fd9\u6837\u4e8b\u7684\u8bdd\u53eb\u6211 \u4f55\u6094\u626b\u52a8 \u88c5\u87d1\u8782\u8718\u51c9 \u767d\u8272\u8721\u70db \u6e29\u6696\u4e86\u7a7a\u5c4b\u85e4\u8513\u690d\u7269 \u722c\u6ee1\u4e86\u4f2f\u7235\u7684\u575f\u5893 \u53e4\u5821\u91cc\u4e00\u7247\u8352\u829c \u957f\u6ee1\u6742\u8349\u7684\u6ce5\u571f \u4e0d\u4f1a\u9a91\u626b\u628a\u7684\u80d6\u5973\u5deb \u7528\u62c9\u4e01\u6587\u5ff5\u5492\u8bed\u5566\u5566\u545c \u5979\u517b\u7684\u9ed1\u732b\u7b11\u8d77\u6765\u50cf\u54ed \u5566\u5566\u5566\u545c \u7528\u6c34\u6676\u7403\n -  \u4e0d\u5206\u5f00 \u6cea\u6c34\u5728\u6218\u58d5\u91cc\u9762\u4e86\u5824 \u6cea\u6c34\u5728\u6218\u58d5\u91cc\u51b3 \u6572\u5f80\u5c11\u5409\u4ed6\u624d\u80fd\u4e70\u5f97\u5230) \u4ed6\u771f\u7684\u771f\u7684\u60f3\u77e5\u9053 (\u4ed6\u4e0d\u77e5\u9053\u600e\u4e48\u529e\u624d\u597d \u7761\u4e0d\u7740) \u4ed6\u5c06\u90a3\u6251\u6ee1\u6253\u7834\u4e86 (\u5c0f\u5c0f\u613f\u671b\u5c31\u5feb\u5b9e\u73b0\u4e86 \u4ed6\u5728\u7b11) \u8dd1\u904d\u4e86\u7267\u573a \u53c8\u7ed5\u8fc7\u4e86\u6751\u5e84 \u4ed6\u5c31\u7ad9\u5728\u8857\u89d2\u7684\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u7684\u4f60\u8eab\u91cc\u91cc\u6f02\u6cca\u5fc3\u4f24\u900f \u5a18\u5b50\u5979\u4eba\u5728\u6c5f\u5357\u7b49\u6211 \u6cea\u4e0d\u4f11 \u8bed\u6c89\u9ed8 \u4e00\u58f6\u597d\u9152 \u518d\u6765\u4e00\u7897\u70ed\u7ca5 \u914d\u4e0a\u51e0\u65a4\u7684\u725b\u8089 \u6211\u8bf4\u5e97\u5c0f\u4e8c \u4e09\u4e24\u94f6\u591f\u4e0d\u591f \u666f\u8272\u5165\u79cb \u6f2b\u5929\u9ec4\u6c99\u51c9\u8fc7 \u585e\u5317\u7684\u5ba2\u6808\u4eba\u591a \u7267\u8349\u6709\u6ca1\u6709 \u6211\u9a6c\u513f\u6709\u4e9b\u7626 \u5929\u6daf\u5c3d\u5934 \u6ee1\n\nEpoch 160. Perplexity 1.110391\n -  \u5206\u5f00 \u8fd9\u6837\u65f6\u95f4 \u5168\u6e56\u4e86\u65e5\u573a \u53c8\u8f9b\u82e6\u82e6 \u5168\u5bb6\u6015\u65e5\u51fa \u767d\u8272\u8721\u70db \u6e29\u6696\u4e86\u7a7a\u5c4b \u767d\u8272\u8721\u70db \u6e29\u6696\u4e86\u7a7a\u5c4b\u85e4\u8513\u690d\u7269 \u722c\u6ee1\u4e86\u4f2f\u7235\u7684\u575f\u5893 \u53e4\u5821\u91cc\u4e00\u7247\u8352\u829c \u957f\u6ee1\u6742\u8349\u7684\u6ce5\u571f \u4e0d\u4f1a\u9a91\u626b\u628a\u7684\u80d6\u5973\u5deb \u7528\u62c9\u4e01\u6587\u5ff5\u5492\u8bed\u5566\u5566\u545c \u5979\u517b\u7684\u9ed1\u732b\u7b11\n -  \u4e0d\u5206\u5f00 \u6211\u53ea\u80fd\u518d\u653e \u6211\u4e0d\u80fd\u8fd9\u6837 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u80fd \u7231\u60c5\u8d70\u7684\u592a\u5feb\u5c31\u50cf\u9f99\u5377\u98ce \u4e0d\u80fd\u627f\u53d7\u6211\u5df2\u65e0\u5904\u53ef\u8eb2 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u60f3\u4f60 \u7231\u60c5\u6765\u7684\u592a\u5feb\u5c31\u50cf\u9f99\u5377\u98ce \u79bb\u4e0d\u5f00\u66b4\u98ce\u5708\u6765\u4e0d\u53ca\u9003 \u6211\u4e0d\u80fd\u518d\u60f3 \u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u71ac\u591a \u8fd9\u8336\u684c\u6a1f\u6728\u8d70\u7684\u8a93\u9986 \u6ce2\u5f26\u68ee\u7565\u90fd\u4e3a\u4e3a\u4eba\u6cbc\u6cfd \u6211\u7528\u53e4\u8001\u7684\u5492\u8bed\u91cd\u6e29 \u541f\u5531\u7075\u9b42\u5e8f\u66f2\u5bfb\u6839 \u9762\u5bf9\u9b54\u754c\u7684\u90aa\u543b \u4e0d\u88ab\u6c61\u67d3\u7684\u8f6c\u751f \u7ef4\u6301\u7eaf\u767d\u7684\u8c61\u5f81 \u7136\u540e\u8fd8\u539f\u4e3a\u4eba \u8ba9\u6211\u4eec \u534a\u517d\u4eba \u7684\u7075\u9b42 \u7ffb\u6eda \u6536\u8d77\u6b8b\u5fcd \u56de\u5fc6\u517d\u5316\u7684\u8fc7\u7a0b\n\nEpoch 180. Perplexity 1.070912\n -  \u5206\u5f00 \u8fd9\u6837\u9762\u7684\u5473\u53eb\u6851 \u6211\u6709\u4f60\u7684\u8138 \u6211\u6709\u6094\u5b9a\u53ef\u53ef\u4ee5 \u4eba\u540e\u7684\u542c\u6211\u9762\u7ea2\u7684\u53ef\u7231\u5973\u4eba \u6e29\u67d4\u7684\u8ba9\u6211\u9762\u7ea2\u7684\u53ef\u7231\u5973\u4eba \u6e29\u67d4\u7684\u8ba9\u6211\u9762\u7ea2\u7684\u53ef\u7231\u5973\u4eba \u6e29\u67d4\u7684\u8ba9\u4f60\u5fc3\u75bc\u7684\u53ef\u7231\u5973\u4eba \u900f\u67d4\u7684\u8ba9\u6211\u9762\u7ea2\u7684\u53ef\u7231\u5973\u4eba \u6e29\u67d4\u7684\u8ba9\u4f60\u5fc3\u7ea2\u7684\u53ef\u7231\u5973\u4eba \n -  \u4e0d\u5206\u5f00\u5c31\u8d70 \u628a\u624b\u6162\u6162\u4ea4\u7ed9\u6211 \u653e\u4e0b\u5fc3\u4e2d\u7684\u56f0\u60d1 \u96e8\u70b9\u4ece\u4e24\u65c1\u5212\u8fc7 \u5272\u5f00\u4e24\u79cd\u7cbe\u795e\u7684\u6211 \u7ecf\u8fc7\u8001\u4f2f\u7684\u5bb6 \u7bee\u6846\u53d8\u5f97\u597d\u9ad8 \u722c\u8fc7\u7684\u90a3\u68f5\u6811 \u53c8\u4f55\u65f6\u53d8\u5f97\u6e3a\u5c0f \u8fd9\u6837\u4e5f\u597d \u5f00\u59cb\u6ca1\u4eba\u6ce8\u610f\u5230\u4f60\u6211 \u7b49\u96e8\u53d8\u5f3a\u4e4b\u524d \u6211\u4eec\u5c06\u4f1a\u5206\u5316\u8f6f\u5f31 \u8d81\u65f6\u95f4\u6ca1\u53d1\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u71ac\u591a \u5728\u785d\u70df\u4e2d\u8fdb\u8d77\u51b0\u68d2\u8fdc \u90a3\u662f\u98ce\u5316\u7684\u90aa\u543b \u4e0d\u88ab\u6c61\u67d3\u7684\u8f6c\u751f \u7ef4\u6301\u7eaf\u767d\u7684\u8c61\u5f81 \u7136\u540e\u8fd8\u539f\u4e3a\u4eba \u8ba9\u6211\u4eec \u534a\u517d\u4eba \u7684\u7075\u9b42 \u7ffb\u6eda \u6536\u8d77\u6b8b\u5fcd \u56de\u5fc6\u517d\u5316\u7684\u8fc7\u7a0b \u8ba9\u6211\u4eec \u534a\u517d\u4eba \u7684\u773c\u795e \u5355\u7eaf \u800c\u975e\u8d2a\u5a6a\u7740\u6c38\u6052 \u53ea\u5bf9\u66b4\u529b\u5fe0\u8bda\n\nEpoch 200. Perplexity 1.049616\n -  \u5206\u5f00 \u8fd9\u6837\u9762\u7684\u8bdd \u6ca1\u6709\u4e9b\u529b\u5bf9 \u5bb6\u8c61\u653e\u8d77\u7684\u753b\u9762 \u56de\u5fc6\u6fc0\u6d3b \u662f\u662f\u90a3\u98ce\u6ee1\u4e86\u7834\u4e86\u6211 \u592a\u96e8\u9762\u7684\u6d88\u8ff9\u6210\u4e0a\u7684\u6670 \u6ca1\u6709\u56de\u5fc6 \u5728\u5c0f\u9547\u5728\u80cc\u8fb9 \u5e72\u4ec0\u4e48(\u5ba2) \u5e72\u4ec0\u4e48(\u5ba2) \u6211\u6253\u5f00\u4efb\u7763\u4e8c\u8109 \u5e72\u4ec0\u4e48(\u5ba2) \u5e72\u4ec0\u4e48(\u5ba2) \u4e1c\u4e9a\u75c5\u592b\u7684\u62db\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u8981\u8fd9\u79cd\u7275\u4efd \u5305\u5bb9\u4e3a\u9f99 \u4f60\u5c71\u311f\u53cb\u662f\u8fc7\u4e86 \u6211 \u60f3\u548c\u4f60\u770b\u68d2\u7403 \u60f3\u8fd9\u6837\u6ca1\u62c5\u5fe7 \u5531\u7740\u6b4c \u4e00\u76f4\u8d70 \u6211\u60f3\u5c31\u8fd9\u6837\u7275\u7740\u4f60\u7684\u624b\u4e0d\u653e\u5f00 \u7231\u53ef\u4e0d\u53ef\u4ee5\u7b80\u7b80\u5355\u5355\u6ca1\u6709\u4f24\u5bb3 \u4f60 \u9760\u7740\u6211\u7684\u80a9\u8180 \u4f60 \u5728\u6211\u80f8\u53e3\u7761\u8457 \u50cf\u8fd9\u6837\u7684\u751f\u6d3b \u6211\u7231\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u71ac\u591a \u5df2\u5728\u89d2\u4f26\u6cb3\u91cc\u9082\u4e86\u6c38\u8fdc \u90a3\u5df2\u98ce\u5316\u5343\u5e74\u7684\u8a93\u8a00 \u4e00\u5207\u53c8\u91cd\u6f14 \u6211\u611f\u5230\u5f88\u75b2\u5026\u79bb\u5bb6\u4e61\u8fd8\u662f\u5f88\u8fdc \u5bb3\u6015\u518d\u4e5f\u4e0d\u80fd\u56de\u5230\u4f60\u8eab\u8fb9 \u6211\u7ed9\u4f60\u7684\u7231\u5199\u5728\u897f\u5143\u524d \u6df1\u57cb\u5728\u7f8e\u7d22\u4e0d\u8fbe\u7c73\u4e9a\u5e73\u539f \u51e0\u5341\u4e2a\u4e16\u7eaa\u540e\u51fa\u571f\u53d1\u73b0 \u6ce5\u677f\u4e0a\u7684\u5b57\u8ff9\u4f9d\u7136\u6e05\u6670\u53ef\n\n"
 }
]
```

可以看到一开始学到简单的字符，然后简单的词，接着是复杂点的词，然后看上去似乎像个句子了。

## 结论

* 通过隐含状态，循环神经网络适合处理前后有依赖关系时序数据样本。
* 对前后有依赖关系时序数据样本批量采样时，我们可以使用随机批量采样和相邻批量采样。
* 循环神经网络较容易出现梯度衰减和爆炸。


## 练习

* 调调参数（例如数据集大小、序列长度和学习率），看看对perplexity和预测的结果造成的区别。
* 在随机批量采样中，如果在同一个epoch中只把隐含变量在该epoch开始的时候初始化会怎么样？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/989)
