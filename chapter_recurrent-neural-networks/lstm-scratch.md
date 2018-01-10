# 长短期记忆（LSTM） --- 从0开始

前面的教程里我们使用的网络都属于**前馈神经网络**。之所以叫前馈，是因为整个网络是一条链（回想下`gluon.nn.Sequential`），每一层的结果都是反馈给下一层。这一节我们介绍**循环神经网络**，这里每一层不仅输出给下一层，同时还输出一个**隐藏状态**，给当前层在处理下一个样本时使用。下图展示这两种网络的区别。

![](../img/rnn_1.png)

循环神经网络的这种结构使得它适合处理前后有依赖关系的样本。我们拿语言模型举个例子来解释这个是怎么工作的。语言模型的任务是给定句子的前*t*个字符，然后预测第*t+1*个字符。假设我们的句子是“你好世界”，使用前馈神经网络来预测的一个做法是，在时间1输入“你”，预测”好“，时间2向同一个网络输入“好”预测“世”。下图左边展示了这个过程。

![](../img/rnn_2.png)

注意到一个问题是，当我们预测“世”的时候只给了“好”这个输入，而完全忽略了“你”。直觉上“你”这个词应该对这次的预测比较重要。虽然这个问题通常可以通过**n-gram**来缓解，就是说预测第*t+1*个字符的时候，我们输入前*n*个字符。如果*n=1*，那就是我们这里用的。我们可以增大*n*来使得输入含有更多信息。但我们不能任意增大*n*，因为这样通常带来模型复杂度的增加从而导致需要大量数据和计算来训练模型。

循环神经网络使用一个隐藏状态来记录前面看到的数据来帮助当前预测。上图右边展示了这个过程。在预测“好”的时候，我们输出一个隐藏状态。我们用这个状态和新的输入“好”来一起预测“世”，然后同时输出一个更新过的隐藏状态。我们希望前面的信息能够保存在这个隐藏状态里，从而提升预测效果。

## 循环神经网络

在对输入输出数据有了解后，我们来正式介绍循环神经网络。

首先回忆一下单隐层的前馈神经网络的定义，例如[多层感知机](../chapter_supervised-learning/mlp-scratch.md)。假设隐层的激活函数是$\phi$，对于一个样本数为$n$特征向量维度为$x$的批量数据$\mathbf{X} \in \mathbb{R}^{n \times x}$（$\mathbf{X}$是一个$n$行$x$列的实数矩阵）来说，那么这个隐层的输出就是

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h)$$

假定隐层宽度为$h$，那么其中的权重参数的尺寸为$\mathbf{W}_{xh} \in \mathbb{R}^{x \times h}$。偏移参数 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$在与前一项$\mathbf{X} \mathbf{W}_{xh} \in \mathbb{R}^{n \times h}$ 相加时使用了[广播](../chapter_crashcourse/ndarray.md)。这个隐层的输出的尺寸为$\mathbf{H} \in \mathbb{R}^{n \times h}$。

把隐层的输出$\mathbf{H}$作为输出层的输入，最终的输出

$$\hat{\mathbf{Y}} = \text{softmax}(\mathbf{H} \mathbf{W}_{hy} + \mathbf{b}_y)$$

假定每个样本对应的输出向量维度为$y$，其中 $\hat{\mathbf{Y}} \in \mathbb{R}^{n \times y}, \mathbf{W}_{hy} \in \mathbb{R}^{h \times y}, \mathbf{b}_y \in \mathbb{R}^{1 \times y}$且两项相加使用了[广播](../chapter_crashcourse/ndarray.md)。


将上面网络改成循环神经网络，我们首先对输入输出加上时间戳$t$。假设$\mathbf{X}_t \in \mathbb{R}^{n \times x}$是序列中的第$t$个批量输入（样本数为$n$，每个样本的特征向量维度为$x$），对应的隐层输出是隐藏状态$\mathbf{H}_t  \in \mathbb{R}^{n \times h}$（隐层宽度为$h$），而对应的最终输出是$\hat{\mathbf{Y}}_t \in \mathbb{R}^{n \times y}$（每个样本对应的输出向量维度为$y$）。在计算隐层的输出的时候，循环神经网络只需要在前馈神经网络基础上加上跟前一时间$t-1$输入隐层$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$的加权和。为此，我们引入一个新的可学习的权重$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h)$$

输出的计算跟前面一致：

$$\hat{\mathbf{Y}}_t = \text{softmax}(\mathbf{H}_t \mathbf{W}_{hy}  + \mathbf{b}_y)$$

一开始我们提到过，隐层输出（又叫隐藏状态）可以认为是这个网络的记忆。它存储前面时间里面的信息。我们的输出是只基于这个状态。最开始的隐藏状态里的元素通常会被初始化为0。


## 周杰伦歌词数据集


为了实现并展示循环神经网络，我们使用周杰伦歌词数据集来训练模型作词。该数据集里包含了著名创作型歌手周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》所有歌曲的歌词。

![](../img/jay.jpg)


下面我们读取这个数据并看看前面49个字符（char）是什么样的：

```{.python .input  n=1}
import zipfile
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
print(corpus_chars[0:49])
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\u60f3\u8981\u6709\u76f4\u5347\u673a\n\u60f3\u8981\u548c\u4f60\u98de\u5230\u5b87\u5b99\u53bb\n\u60f3\u8981\u548c\u4f60\u878d\u5316\u5728\u4e00\u8d77\n\u878d\u5316\u5728\u5b87\u5b99\u91cc\n\u6211\u6bcf\u5929\u6bcf\u5929\u6bcf\u5929\u5728\u60f3\u60f3\u60f3\u60f3\u8457\u4f60\n\n"
 }
]
```

我们看一下数据集里的字符数。

```{.python .input  n=2}
len(corpus_chars)
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "64925"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

接着我们稍微处理下数据集。为了打印方便，我们把换行符替换成空格，然后截去后面一段使得接下来的训练会快一点。

```{.python .input  n=3}
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]
```

## 字符的数值表示

先把数据里面所有不同的字符拿出来做成一个字典：

```{.python .input  n=4}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

vocab_size = len(char_to_idx)

print('vocab size:', vocab_size)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "vocab size: 1465\n"
 }
]
```

然后可以把每个字符转成从0开始的索引(index)来方便之后的使用。

```{.python .input  n=5}
corpus_indices = [char_to_idx[char] for char in corpus_chars]

sample = corpus_indices[:40]

print('chars: \n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices: \n', sample)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "chars: \n \u60f3\u8981\u6709\u76f4\u5347\u673a \u60f3\u8981\u548c\u4f60\u98de\u5230\u5b87\u5b99\u53bb \u60f3\u8981\u548c\u4f60\u878d\u5316\u5728\u4e00\u8d77 \u878d\u5316\u5728\u5b87\u5b99\u91cc \u6211\u6bcf\u5929\u6bcf\u5929\u6bcf\n\nindices: \n [316, 1007, 935, 305, 248, 224, 50, 316, 1007, 485, 556, 366, 1048, 120, 263, 1362, 50, 316, 1007, 485, 556, 598, 519, 510, 513, 108, 50, 598, 519, 510, 120, 263, 1088, 50, 1097, 1110, 480, 1110, 480, 1110]\n"
 }
]
```

## 时序数据的批量采样

同之前一样我们需要每次随机读取一些（`batch_size`个）样本和其对用的标号。这里的样本跟前面有点不一样，这里一个样本通常包含一系列连续的字符（前馈神经网络里可能每个字符作为一个样本）。

如果我们把序列长度（`num_steps`）设成5，那么一个可能的样本是“想要有直升”。其对应的标号仍然是长为5的序列，每个字符是对应的样本里字符的后面那个。例如前面样本的标号就是“要有直升机”。


### 随机批量采样

下面代码每次从数据里随机采样一个批量。

```{.python .input  n=6}
import random
from mxnet import nd

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为label的索引是相应data的索引加一
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个数据
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label
```

为了便于理解时序数据上的随机批量采样，让我们输入一个从0到29的人工序列，看下读出来长什么样：

```{.python .input  n=7}
my_seq = list(range(30))

for data, label in data_iter_random(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "data:  \n[[  9.  10.  11.]\n [ 21.  22.  23.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[ 10.  11.  12.]\n [ 22.  23.  24.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[  3.   4.   5.]\n [ 12.  13.  14.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[  4.   5.   6.]\n [ 13.  14.  15.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[ 15.  16.  17.]\n [  6.   7.   8.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[ 16.  17.  18.]\n [  7.   8.   9.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[ 24.  25.  26.]\n [  0.   1.   2.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[ 25.  26.  27.]\n [  1.   2.   3.]]\n<NDArray 2x3 @cpu(0)> \n\n"
 }
]
```

由于各个采样在原始序列上的位置是随机的时序长度为`num_steps`的连续数据点，相邻的两个随机批量在原始序列上的位置不一定相毗邻。因此，在训练模型时，读取每个随机时序批量前需要重新初始化隐藏状态。


### 相邻批量采样

除了对原序列做随机批量采样之外，我们还可以使相邻的两个随机批量在原始序列上的位置相毗邻。

```{.python .input  n=8}
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    # 减一是因为label的索引是相应data的索引加一
    epoch_size = (batch_len - 1) // num_steps
    
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label
```

相同地，为了便于理解时序数据上的相邻批量采样，让我们输入一个从0到29的人工序列，看下读出来长什么样：

```{.python .input  n=9}
my_seq = list(range(30))

for data, label in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "data:  \n[[  0.   1.   2.]\n [ 15.  16.  17.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[  1.   2.   3.]\n [ 16.  17.  18.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[  3.   4.   5.]\n [ 18.  19.  20.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[  4.   5.   6.]\n [ 19.  20.  21.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[  6.   7.   8.]\n [ 21.  22.  23.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[  7.   8.   9.]\n [ 22.  23.  24.]]\n<NDArray 2x3 @cpu(0)> \n\ndata:  \n[[  9.  10.  11.]\n [ 24.  25.  26.]]\n<NDArray 2x3 @cpu(0)> \nlabel: \n[[ 10.  11.  12.]\n [ 25.  26.  27.]]\n<NDArray 2x3 @cpu(0)> \n\n"
 }
]
```

由于各个采样在原始序列上的位置是毗邻的时序长度为`num_steps`的连续数据点，因此，使用相邻批量采样训练模型时，读取每个时序批量前，我们需要将该批量最开始的隐藏状态设为上个批量最终输出的隐藏状态。在同一个epoch中，隐藏状态只需要在该epoch开始的时候初始化。


## Onehot编码

注意到每个字符现在是用一个整数来表示，而输入进网络我们需要一个定长的向量。一个常用的办法是使用onehot来将其表示成向量。也就是说，如果一个字符的整数值是$i$, 那么我们创建一个全0的长为`vocab_size`的向量，并将其第$i$位设成1。该向量就是对原字符的onehot编码。

```{.python .input  n=10}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  0.  0. ...,  0.  0.  0.]\n [ 0.  0.  1. ...,  0.  0.  0.]]\n<NDArray 2x1465 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

记得前面我们每次得到的数据是一个`batch_size * num_steps`的批量。下面这个函数将其转换成`num_steps`个可以输入进网络的`batch_size * vocab_size`的矩阵。对于一个长度为`num_steps`的序列，每个批量输入$\mathbf{X} \in \mathbb{R}^{n \times x}$，其中$n=$ `batch_size`，而$x=$`vocab_size`（onehot编码向量维度）。

```{.python .input  n=11}
def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]

inputs = get_inputs(data)

print('input length: ', len(inputs))
print('input[0] shape: ', inputs[0].shape)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "input length:  3\ninput[0] shape:  (2, 1465)\n"
 }
]
```

## 初始化模型参数

对于序列中任意一个时间戳，一个字符的输入是维度为`vocab_size`的onehot编码向量，对应输出是预测下一个时间戳为词典中任意字符的概率，因而该输出是维度为`vocab_size`的向量。

当序列中某一个时间戳的输入为一个样本数为`batch_size`（对应模型定义中的$n$）的批量，每个时间戳上的输入和输出皆为尺寸`batch_size * vocab_size`（对应模型定义中的$n \times x$）的矩阵。假设每个样本对应的隐藏状态的长度为`hidden_dim`（对应模型定义中的$h$），根据矩阵乘法定义，我们可以推断出模型隐含层和输出层中各个参数的尺寸。

```{.python .input  n=12}
import mxnet as mx

# 尝试使用GPU
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
print('Will use', ctx)

# 隐藏状态长度
input_dim = vocab_size
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

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Will use gpu(0)\n"
 }
]
```

## 定义模型

当序列中某一个时间戳的输入为一个样本数为`batch_size`的批量，而整个序列长度为`num_steps`时，以下`rnn`函数的`inputs`和`outputs`皆为`num_steps` 个尺寸`batch_size * vocab_size`的矩阵，隐藏状态变量$\mathbf{H}$是一个尺寸为`batch_size * hidden_dim`的矩阵。

我们将前面的模型公式翻译成代码。这里的激活函数使用了按元素操作的双曲正切函数

$$\text{tanh}(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

```{.python .input  n=13}
def lstm_rnn(inputs, H, C, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hy, b_y] = params

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

## 预测序列

在做预测时我们只需要给定时间0的输入和起始隐藏状态。然后我们每次将上一个时间的输出作为下一个时间的输入。

![](../img/rnn_3.png)

```{.python .input  n=14}
def predict(prefix, num_chars, params):
    # 预测以 prefix 开始的接下来的 num_chars 个字符
    prefix = prefix.lower()
    state = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    C = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        # 在序列中循环迭代隐藏状态
        Y, state, C = lstm_rnn(get_inputs(X), state, C, *params)
        if i < len(prefix)-1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])
```

## 梯度剪裁

我们在[正向传播和反向传播](../chapter_supervised-learning/backprop.md)中提到，
训练神经网络往往需要依赖梯度计算的优化算法，例如我们之前介绍的[随机梯度下降](../chapter_supervised-learning/linear-regression-scratch.md)。
而在循环神经网络的训练中，当每个时序训练数据样本的时序长度`num_steps`较大或者时刻$t$较小，目标函数有关$t$时刻的隐含层变量梯度较容易出现衰减（valishing）或爆炸（explosion）。我们会在[下一节](bptt.md)详细介绍出现该现象的原因。

为了应对梯度爆炸，一个常用的做法是如果梯度特别大，那么就投影到一个比较小的尺度上。假设我们把所有梯度接成一个向量 $\boldsymbol{g}$，假设剪裁的阈值是$\theta$，那么我们这样剪裁使得$\|\boldsymbol{g}\|$不会超过$\theta$：

$$ \boldsymbol{g} = \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

```{.python .input  n=15}
def grad_clipping(params, theta):
    norm = nd.array([0.0], ctx)
    for p in params:
        norm += nd.sum(p.grad ** 2)
    norm = nd.sqrt(norm).asscalar()
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm
```

## 训练模型

下面我们可以还是训练模型。跟前面前置网络的教程比，这里有以下几个不同。

1. 通常我们使用困惑度（Perplexity）这个指标。
2. 在更新前我们对梯度做剪裁。
3. 在训练模型时，对时序数据采用不同批量采样方法将导致隐藏状态初始化的不同。

### 困惑度（Perplexity）

回忆以下我们之前介绍的[交叉熵损失函数](../chapter_supervised-learning/softmax-regression-scratch.md)。在语言模型中，该损失函数即被预测字符的对数似然平均值的相反数：

$$\text{loss} = -\frac{1}{N} \sum_{i=1}^N \log p_{\text{target}_i}$$

其中$N$是预测的字符总数，$p_{\text{target}_i}$是在第$i$个预测中真实的下个字符被预测的概率。

而这里的困惑度可以简单的认为就是对交叉熵做exp运算使得数值更好读。

为了解释困惑度的意义，我们先考虑一个完美结果：模型总是把真实的下个字符的概率预测为1。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 1$。这种完美情况下，困惑度值为1。

我们再考虑一个基线结果：给定不重复的字符集合$W$及其字符总数$|W|$，模型总是预测下个字符为集合$W$中任一字符的概率都相同。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 1/|W|$。这种基线情况下，困惑度值为$|W|$。

最后，我们可以考虑一个最坏结果：模型总是把真实的下个字符的概率预测为0。也就是说，对任意的$i$来说，$p_{\text{target}_i} = 0$。这种最坏情况下，困惑度值为正无穷。

任何一个有效模型的困惑度值必须小于预测集中元素的数量。在本例中，困惑度必须小于字典中的字符数$|W|$。如果一个模型可以取得较低的困惑度的值（更靠近1），通常情况下，该模型预测更加准确。

```{.python .input  n=16}
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'

from mxnet import autograd
from mxnet import gluon
from math import exp

epochs = 200
seq_len = 35
learning_rate = .1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def train_and_predict(is_random_iter):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()

    for e in range(1, epochs + 1):
        # 如使用相邻批量采样，在同一个epoch中，隐藏状态只需要在该epoch开始的时候初始化。
        if not is_random_iter:
            state = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            C = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)

        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, seq_len, ctx):
            # 如使用随机批量采样，处理每个随机小批量前都需要初始化隐藏状态。
            if is_random_iter:
                state = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                C = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs 尺寸：(batch_size, vocab_size)
                outputs, state, C = lstm_rnn(get_inputs(data), state, C, *params)
                # label 尺寸：（batch_size * seq_len,）设t_ib_j为i时间批量中的j元素）
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                label = label.T.reshape((-1,))
                # 拼接outputs，尺寸：(batch_size * seq_len, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                # 经上述操作，outputs和label已对齐。
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, 5)
            utils.SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % 20 == 0:
            print("Epoch %d. Perplexity %f" % (e, exp(train_loss/num_examples)))
            print(' - ', predict(seq1, 100, params))
            print(' - ', predict(seq2, 100, params))
            print(' - ', predict(seq3, 100, params), '\n')
```

我们先采用随机批量采样实验循环神经网络谱写歌词。我们假定谱写歌词的前缀分别为“分开”、“不分开”和“战争中部队”。

```{.python .input  n=17}
train_and_predict(is_random_iter=True)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 20. Perplexity 347.668014\n -  \u5206\u5f00                                                                                                     \n -  \u4e0d\u5206\u5f00                                                                                                     \n -  \u6218\u4e89\u4e2d\u90e8\u961f                                                                                                      \n\nEpoch 40. Perplexity 315.137527\n -  \u5206\u5f00 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \n -  \u4e0d\u5206\u5f00 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211  \n\nEpoch 60. Perplexity 260.790779\n -  \u5206\u5f00 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\n -  \u4e0d\u5206\u5f00 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u7684 \u6211\u4e0d \u6211 \n\nEpoch 80. Perplexity 189.881298\n -  \u5206\u5f00 \u6211\u4e0d\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u4f60\u4f60\u60f3\u4f60 \u4f60\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u4e0d\u7684\u4f60 \u6211\u4e0d\u4e0d\u7684\u4f60 \u6211\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e0d \u4f60\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211  \n\nEpoch 100. Perplexity 137.223110\n -  \u5206\u5f00 \u6211\u60f3\u60f3\u4f60\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u4f60\u4f60\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u4f60\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d\u8981\u6211\u60f3\u4f60 \u4f60\u4e0d \n\nEpoch 120. Perplexity 95.942211\n -  \u5206\u5f00 \u6211\u60f3\u7740\u4f60\u7684\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\n -  \u4e0d\u5206\u5f00 \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e0d \u6211\u4e0d\u4e0d \u4f60\u4e0d\u4e0d \u4f60\u4e0d\u4e48 \u6211\u4e0d\u4e48 \u4f60\u4e0d\u4e48 \u6211\u4e0d\u4e48\u518d\u4f60 \u4f60\u4e0d\u4e0d\u89c9 \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e48 \u6211\u4e0d\u4e48\u518d\u4e0d \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e48 \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e48 \u6211\u4e0d\u4e48\u518d\u4f60 \u4f60\u4e0d\u4e0d\u89c9 \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e48 \u6211\u4e0d\u4e48\u518d\u4e0d \u6211\u4e0d\u8981 \u4f60\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u8fd9\u4f60 \u4f60\u4e0d\u4e0d\u518d\u6211 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u4f60\u4e0d\u4e0d\u518d \n\nEpoch 140. Perplexity 67.862265\n -  \u5206\u5f00 \u6211\u60f3\u4f60\u7684\u7231\u6211 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\n -  \u4e0d\u5206\u5f00 \u6211\u4e0d\u8981\u8fd9\u4e0d\u5f00 \u6211\u4e0d\u8981\u4f60\u60f3\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u8fd9\u60f3\u4f60 \u4f60\u4e0d\u8981\u4f60\u60f3\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d \n\nEpoch 160. Perplexity 49.742422\n -  \u5206\u5f00 \u6211\u60f3\u7740\u4f60\u7684\u60f3\u754c \u6211\u60f3\u8981\u4f60\u7684\u79bb\u5f00 \u4f60\u4f1a\u4f1a\u4f60 \u4f60\u4e0d\u5f00 \u522b\u4e0d\u8981 \u60f3\u4e0d\u7684\u8fd9\u6837\u9f99 \u4f60\u8bf4 \u4f60\u60f3\u518d\u4e45\u6211 \u4f60\u8bf4\u4f60 \u4f60\u7231\u6211 \u60f3\u4e0d\u7684\u8fd9\uff01\u4f60 \u8bf4\u8bf4 \u4f60\u60f3\u60f3\u518d\u60f3\u8981 \u4f60\u8bf4 \u4f60\u60f3\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3\u4f60 \u8bf4\u8bf4 \u4f60\u60f3\u60f3 \u6211\u4e0d\u4e86\u8fd9\u60f3\u4f60 \u8bf4\u8bf4\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u4f60\u8fd9\u60f3\u4f60 \u4f60\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u7684\u5929  \u60f3\u5728\u4f60\u7684\u7231\u5199 \u4f60\u7684\u8ba9\u6211\u6709\u4f60\u7684\u53ef\u7231 \u6211\u60f3\u8bf4\u4f60\u7684\u60f3 \u4e00\u4eba\u5473\u9053\u7684\u753b\u9762 \u6211\u60f3\u8bf4\u4f60\u7684\u60f3\u754c \u6ca1\u60f3\u4f60\u4f60\u7684\u4f60 \u6211\u8981\u80fd\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d \n\nEpoch 180. Perplexity 35.618232\n -  \u5206\u5f00 \u6211\u60f3\u7740\u4f60\u7684\u624b \u6211\u77e5\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u60f3\u6211 \u7231\u4f60\u7684\u6211\u7231\u6211 \u8bf4\u4f60\u7684\u8ba9\u6211\u7684\u7ea2 \u6211\u60f3\u60f3\u4f60\u7684\u60f3\u6211 \u60f3\u4e0d\u518d\u6211 \u4f60\u7ecf\u6211\u6709\u60f3\u60f3 \u4f60\u8bf4\u554a \u662f\u4e0d\u5f00 \u662f\u4f60\u7684\u592a\u7231\u4f60 \u50cf\u6697\u4f26 \u662f\u662f\u6211 \u662f\u4f60 \u8fd9\u662f\u6b62 xi xi xi xi xi \n -  \u4e0d\u5206\u5f00 \u6211 \u60f3\u6211\u4e0d\u60f3\u4f60 \u4f60\u4e0d\u4e0d\u8fd9\u6211 \u6211\u4e0d\u8981 \u4f60\u4e0d\u6211 \u60f3\u4f60\u7684\u592a\u592a\u60c5\u50cf\u9f99\u5377\u5377 \u4e0d\u80fd\u518d\u98ce \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u4e0d\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u60f3\u6211 \u7231\u4f60\u7684\u6211\u7231\u6211 \u6211\u60f3\u4f60 \u4f60\u7231\u6211 \u60f3\u7b80\u7b80\u7684\u592a\u60c5 \u50cf\u4e0d\u662f \u53c8\u662f\u6211 \u7684\u7075\u60c5 \u4e00\u4e48\u53e4 xi xi\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u7684\u5929  \u6709\u6709\u4f60 \u534a\u517d\u4eba \u7684\u7075\u9b42 \u7ffb\u6eda \u4e00\u76f4\u6b62 \u7684i  \u4e00i xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi xi \n\nEpoch 200. Perplexity 26.431425\n -  \u5206\u5f00 \u6211\u60f3\u7740\u4f60\u7684\u4e16\u754c \u8ba9\u4f60\u4f1a\u4f60\u7684\u79bb\u5f00 \u6ca1\u4ec0\u4e48\u8fd9\u6837\u7684\u5929 \u6211\u60f3\u4f60\u8bf4\u4f60\u7684\u7ecf \u4e00\u77e5\u6837\u89c9 \u6211\u5df2\u7ecf\u7ecf\u79bb\u5f00 \u6211\u77e5\u4e0d\u89c9 \u4f60\u4e0d\u4e0d\u518d\u5f00\u6d3b \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u7ecf\u79bb\u5f00\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u7ecf\u8fd9\u8282\u9f99 \u6211\u77e5\u77e5\u89c9 \u4f60\u5df2\u4e0d\u518d\u5f00\u6d3b \u4e0d\u77e5\u540e\u89c9 \u6211\u5df2\u7ecf\n -  \u4e0d\u5206\u5f00 \u6211 \u60f3\u4f60\u4e0d\u4e0d\u60f3\u4f60 \u6211 \u4e0d\u4e0d\u518d\u60f3\u6211 \u4e0d\u4e0d\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u662f\u6211\u5f00\u4e86\u4f60 \u8bf4\u6563 \u4f60\u7684\u60f3\u4e45\u4e86\u6211 \u4f60\u60f3\u554a \u4f60\u60f3\u4e0d\u4e45\u6211\u60f3\u8981 \u8bf4\u8bf4\u4f60 \u4f60\u60f3\u4e0d\u4e0d\u5f00 \u4f60\u8bf4\u4e0d\u89c9\u6253\u6211\u4e0d\u77e5\u4e0d\u4e0d\u8981 \u6211\u8bf4\u4f60 \u522b\u4e0d\u5f00 \u522b\u4f60\u7684\u8ba9\u6211\u7684\u7ea2\u9053 \u6211\u60f3\u8bf4\u4f60\u5df2\u7ecf\u7ecf\u4e45\n -  \u6218\u4e89\u4e2d\u90e8\u961f\u7684\u5929  \u6709\u6709\u4f60\u7684\u7231\u5199 \u8ba9\u4f60\u7684\u90a3\u91cc\u7684\u753b\u5bb6 \u6211\u60f3\u60f3\u4f60\u7684\u60f3\u9762 \u6ca1\u60f3\u60f3\u4f60 \u4f60\u60f3\u518d\u518d\u60f3\u6211 \u4e0d\u4e0d\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u7ecf\u4e0d\u5f00 \u6211\u77e5\u77e5\u89c9 \u4f60\u5df2\u7ecf\u5f00\u5f00\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u7ecf\u8fd9\u8282\u9f99 \u6211\u77e5\u77e5\u89c9 \u4f60\u4e0d\u662f\u518d\u5f00\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u7ecf\u8fd9\u8282\u9f99 \u6211 \n\n"
 }
]
```

我们再采用相邻批量采样实验循环神经网络谱写歌词。

```{.python .input  n=18}
train_and_predict(is_random_iter=False)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 20. Perplexity 346.843838\n -  \u5206\u5f00        \u6211\u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684\n -  \u4e0d\u5206\u5f00       \u6211 \u6211\u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \n -  \u6218\u4e89\u4e2d\u90e8\u961f     \u6211 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684 \u7684 \u7684 \u6211\u7684  \n\nEpoch 40. Perplexity 321.479765\n -  \u5206\u5f00 \u6211 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \n -  \u4e0d\u5206\u5f00 \u6211\u6211 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211\u7684 \u6211\u6211\u7684 \u6211\u6211 \u6211\u6211 \n\nEpoch 60. Perplexity 266.029243\n -  \u5206\u5f00 \u6211\u4e0d \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\n -  \u4e0d\u5206\u5f00 \u6211\u4e0d\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u4e0d\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684 \u6211\u7684  \n\nEpoch 80. Perplexity 189.023529\n -  \u5206\u5f00 \u6211\u60f3\u7684\u6211 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u4f60\u4f60\u4f60\u4f60 \u6211\u4e0d\u4e0d\u4f60 \u6211\u4e0d\u6211 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u6211 \u6211\u4e0d\u4e0d \u6211\u4e0d\u4e86 \u6211\u4e0d \u6211\u4e0d \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \u6211 \n\nEpoch 100. Perplexity 124.826569\n -  \u5206\u5f00 \u6211\u60f3\u7684\u4f60\u60f3\u6211 \u4f60\u4e0d\u7684\u4f60 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d \u8fd9\u662f \u6211\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u7684\u4f60\u60f3\u6211 \u6211\u4e0d\u7684\u4f60 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u7684\u4f60 \u6211\u60f3\u7684\u8fd9\u6211 \u4f60\u4e0d\u7684\u4f60 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u7684\u6211 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d\u4e86 \u6211\u4e0d \u8fd9\u662f \u6211\u60f3 \u8fd9\u7075 xi xi xi xi \n\nEpoch 120. Perplexity 77.126213\n -  \u5206\u5f00 \u6211\u60f3\u60f3\u7684\u60f3 \u6211\u4e0d\u8981\u8fd9\u4f60\u7684\u6211\u4e0d \u6211\u60f3\u8981\u4f60 \u4f60\u4e0d\u4f1a\u4f60 \u6211\u4e0d\u8981\u4e0d\u4f60 \u6211\u4e0d\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u4e0d \u7231\u4e0d\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u7231 \u6211\u4e0d\u4e0d\u4e0d\u7231 \u6211\u4e0d\u4e0d\u4e0d\u7231 \u6211\u4e0d\u4e0d\u4e0d\u4f60 \u7231\u7231\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u4f60 \u7231\u7231\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d\u4e0d\u4f60\n -  \u4e0d\u5206\u5f00 \u6211\u60f3\u4f60\u7684\u4f60 \u6211\u4e0d\u8981\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u7684\u4f60 \u7231\u7231\u4e0d\u4e0d\u7684\u7231 \u6211\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d\u4f60 \u6211\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u7684\u7231\u7231 \u7231\u4e0d\u4e0d\u4e0d \u4f60\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u7684\u7231\u7231 \u7231\u4e0d\u4e0d\u4e0d \u4f60\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u7684\u7231\u7231 \u7231\u4e0d\u4e0d\u4e0d\u4e0d\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u7684\u7231 \u6211\u7684\u5929\u5473 \u6211\u4e0d\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d \u6211\u4e0d\u4e0d\u4e0d\u4f60 \u4f60\u4e0d\u4e0d\u4e0d\u8d70 \u4f60\u4e0d\u4e0d\u4e0d\u8d70 \u4f60\u4e0d\u4e0d\u597d\u4f60 \u4f60\u4e0d\u4e0d\u597d\u4f60 \u4f60\u4e0d\u4e0d\u597d\u4f60 \u4f60\u4e0d\u4e0d\u597d\u4f60 \u4e0d\u4e0d\u4e0d\u597d\u8d70 \u4e0d\u4e0d\u4e0d\u597d \u6211\u4e0d\u4e0d\u597d\u4f60 \u4e0d\u4e0d\u4e0d\u597d \u6211\u4e0d\u4e0d\u597d\u4f60 \u4e0d\u4e0d\u4e0d\u597d \u6211\u4e0d\u4e0d \n\nEpoch 140. Perplexity 48.387817\n -  \u5206\u5f00 \u4f60\u60f3\u4e86\u4f60\u7684\u5feb\u4f60 \u6211\u60f3\u4f60\u7684\u7231\u4f60 \u6211\u60f3\u60f3 \u4e0d\u4e0d \u6211\u60f3 \u4f60\u4e0d \u4f60\u4e0d\u7684\u4f60 \u7231\u4e0d\u4e0d \u4f60\u4e0d\u4e86\u4f60\u7684\u60f3 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u4e0d\u4e86\u4f60\u7684\u5feb \u540e\u77e5\u540e\u89c9 \u6211\u4e0d\u4e86\u597d\u751f \u540e\u77e5\u540e\u89c9 \u6211\u77e5\u4e86\u597d\u751f \u540e\u77e5\u540e\u89c9 \u6211\u77e5\u4e86\u597d\u751f \u4f60\u77e5\u4e0d\u89c9 \u6211\u77e5\u4e86\u597d\u751f \n -  \u4e0d\u5206\u5f00 \u4f60\u60f3\u4f60\u7684\u4f60\u7684\u6211 \u8fd9\u6837 \u4f60\u7684\u4f60 \u6211\u60f3\u7684\u8fd9\uff01 \u6211\u4e0d\u8981 \u4f60\u4e0d\u4e86\u4f60\u7684\u5feb\u77e5 \u522b\u602a\u6211 \u522b\u602a\u6211\u7684\u60f3 \u4f60\u4e0d\u4e86\u4f60\u7684\u60f3 \u6211\u60f3 \u4f60\u60f3 \u6211\u60f3\u7684\u60f3\u7684\u60f3 \u522b\u4e0d\u4e0d \u522b\u60f3\u4e86\u6211\u7684\u60f3 \u4f60\u4e0d\u4e0d \u4f60\u60f3\u4e86\u4f60\u7684\u60f3 \u4f60\u4e0d\u4e0d \u4f60\u60f3\u4e86\u4f60\u7684\u60f3 \u4f60\u4e0d\u4e0d \n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u4f60\u6ca1\u4f60 \u6211\u7684\u5929\u754c \u4e0d\u4e0d \u4f60\u4e0d\u4e86 \u6211\u77e5\u4e86\u4f60 \u4f60\u77e5\u4e86\u4f60 \u4f60\u4e0d\u4e86 \u4f60\u4e0d\u4e86 \u4f60\u4e86\u4e86\u8fd9\u6211 \u4f60\u77e5\u4e86 \u4f60\u4e0d\u4e86 \u4f60\u77e5\u4e86\u8fd9\u6211 \u4f60\u77e5\u597d \u6211\u4e0d\u4e86 \u4f60\u77e5\u4e86\u8fd9\u6211 \u4f60\u77e5\u597d\u89c9 \u5feb\u77e5\u4e86\u89c9 \u6211\u77e5\u4e86\u89c9 \u6211\u77e5\u4e86\u89c9 \u6211\u77e5\u4e86\u4f60 \u4f60\u4e0d\u4e86\u4f60  \n\nEpoch 160. Perplexity 30.004391\n -  \u5206\u5f00 \u4f60\u60f3\u4e86\u4f60\u7684\u60f3\u8981 \u8fd9\u6837\u7684\u6ca1\u6837 \u6211\u7684\u4f60\u4e86\u4f60\u7684\u5feb\u4f60 \u522b\u4e0d\u4e0d\u4f60 \u4f60\u5df2\u6211\u7684\u60f3 \u4f60\u4e0d\u8981\u4f60 \u6211\u60f3\u518d\u4f60\u7684\u60f3\u77e5 \u522b\u7231\u4e0d\u7684\u4f60 \u6211\u4e0d\u4e0d\u79bb \u6211\u7684\u8be5\u7b80 \u6211\u77e5\u4e86\u505c\u751f \u53c8\u77e5\u540e\u89c9 \u6211\u77e5\u4e86\u597d\u751f \u53c8\u77e5\u4e86\u89c9\u8d70 \u4f60\u77e5\u4e0d\u89c9 \u4f60\u4e0d\u4e86\u597d\u751f \u4f60\u77e5\u4e0d\n -  \u4e0d\u5206\u5f00 \u4f60\u8bf4\u4f60\u7684\u7231\u6211\u7684\u5929  \u4e0d\u8981 \u4f60\u60f3\u60f3 \u60f3\u60f3 \u7b80\uff01 \u5355\uff01 \u5bf9\u5bf9 \u505c\u5bf9\u4e86\u505c\u7740\u7740\u7740  \u4e0d\u5f00\u6211 \u6211\u60f3\u7684\u60f3 \u60f3\u60f3\u7684\u8fd9\u6837 \u5355\u77e5\u5728\u5bf9\u4f60 \u8ba9\u77e5\u4e86\u6211\u7684\u4f60 \u4f60\u77e5\u4e0d\u89c9 \u4f60\u4e0d\u4e86\u597d\u79bb\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u4e0d\u4e86\u597d\u751f\u6d3b \u540e\u77e5\u540e\u89c9 \u6211\u4e0d\u4e86\u597d\u751f\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u4f60\u5728\u4f60 \u522b\u60f3\u7684 \u60f3\u60f3\u7684\u8fd9 \u5355\u5728\u6211\u7684\u5728\u6211\u5728 \u6211\u4eec \u8ba9\u6211\u4eba\u4eba\u7684\u9c7c  \u6709\u53bb \u4f60\u60f3\u311f\u8fd9\u4f60 \u8fd9\u62b9 \u4f60\u53bb \u8fd9\u311f\u311f\u8fd9\u5355 \u8fd9\u53bb \u4f60\u8ff7\u4eba\u311f\u311f\u6c14 \u8fd9\u62b9 \u4f60\u60f3\u311f\u311f\u311f\u9c7c \u8fd9\u62b9 \u4f60\u60f3\u4eba\u311f\u311f\u9c7c \u8fd9\u62b9 \u6211\u60f3\u79bb\u311f\u311f\u9c7c \u8fd9\u62b9 \u6211\u60f3\u79bb\u311f \n\nEpoch 180. Perplexity 19.038801\n -  \u5206\u5f00 \u4f60\u60f3\u4e86\u4f60\u7684\u60f3\u8981\u4f60 \u6211\u60f3\u80fd\u4f60\u60f3\u5df2 \u4e00\u4e00\u4e00\u8d77\u5728\u5728\u6211\u4e0d\u7684\u4e0d\u4e0d  \u60f3\u4e0d\u4e48\u4f60\u60f3\u5f88 \u4f60\u8981\u4f60 \u8bf4\u4e0d\u6211 \u6211\u8981\u4e86\u8fd9\u6837\u4e48\u6211\u6709\u9519 \u7231\u4e0d\u4e0d\u8fd9\u4f60 \u6211\u60f3\u4e0d\u518d\u60f3 \u6211\u4e0d\u518d\u518d\u4e0d \u6211\u4e0d\u8981\u4f60 \u4f60\u4e0d\u4e86\u8d70\u592a\u5feb \u50cf\u4e0d\u540e \u5feb\u602a\u6211\u7684\u60f3\u4e86\u6211 \u522b\u4e0d\u4e0d \n -  \u4e0d\u5206\u5f00 \u4f60\u8bf4\u4f60\u7684\u592a\u6211\u7684\u90a3\u671b  \u60f3\u4f60\u8fd9\u60f3\u60f3\u5f88\u4f60  \u60f3\u4f60\u8bf4 \u4f60\u4e0d\u773c \u6211\u8981 \u8fd9\uff01\uff01 \u5355\uff01\uff01 \u5355\u8be5\u4e86\u505c\u505c\u7740\u4f60\u7684\u7231 \u7231\u4e0d\u4e0d\u53ef\u6211 \u60f3\u60f3\u60f3\u7684\u60f3 \u6211\u60f3\u518d\u4f60 \u6211\u4e0d\u518d\u518d\u60f3\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u4e0d\u4e86\u79bb\u79bb\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u4e0d\u4e86\u8fd9\u751f\u6d3b \u540e\u77e5\u540e\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u5728\u4f60\u7684\u60f3\u8702 \u4e00\u4e00\u79cd\u5728\u4e0d\u90fd \u5e72\u5f7b \u6211\u60f3\u60f3\u79bb \u6ca1\u6ca1\u6ca1\u8fd9\u4f60 \u53c8\u77e5\u5728\u505c\u843d \u53c8\u77e5\u4e86\u6211\u4eec\u5f00 \u4f60\u4e0d\u4e86\u6211\u7684\u4f60 \u4f60\u8bf4\u4e86\u6211\u4e0d\u5f00 \u4f60\u8bf4\u4e86\u6211 \u5206\u4f60\u4e86 \u4e0d\u4eba\u4e48\u505c\u505c\u7559 \u4e00\u76f4\u597d\u505c\u7740\u6211 \u8ba9\u6211\u4eec \u4e00\u517d\u4eba \u7684\u7075\u9b42 \u5355\u6eda\u6b62\u505c\u7740\u7740\u4f60  \u8ba9 \n\nEpoch 200. Perplexity 12.733728\n -  \u5206\u5f00 \u4f60\u60f3\u662f\u4f60\u7684\u662f\u6211 \u6211\u60f3 \u4f60\u60f3\u4f60\u5f88\u6211 \u4e0d\u4e0d\u8fd9\u4f60\u4e86\u6211\u7684\u4f60\u7684\u4f60 \u4e0d\u60f3\u4e0d\u8fd9\u4f60\u5df2\u5f88\u4e45 \u4e0d\u60f3\u8eb2 \u8bf4\u4f60\u773c\u775b\u770b\u6211 \u8bf4\u8bdd \u4f60\u60f3\u518d\u4f60\u60f3\u6211\u7684\u8bf4 \u522b\u6563 \u4f60\u60f3\u6211\u5f88\u60f3\u6211 \u8bf4\u8bdd \u4f60\u4e0d\u5f88\u4f60\u4e86\u6211 \u8bf4\u8bf4 \u4f60\u60f3\u5f88\u4e45\u4e86\u6211 \u8bf4\u8bf4 \u4f60\u60f3\u5f88\u4e45\u4e86\u6211 \n -  \u4e0d\u5206\u5f00 \u4f60\u7ecf\u662f\u4e0d\u4e0d\u4e86\u6211\u4e86\u4e86\u4e0d\u4e0d\u4e0d\u80fd  \u53ef\u6211\u4e0d\u6765\u4f60\u4e0d\u4e0d\u4e0d\u4e0d \u4e0d\u4e0d\u7684\u4f60\u4e0d\u4e0d\u5c31\u5c31\u4e0d\u4e0d \u6211\u4e0d\u60f3\u518d\u4f60 \u6211\u4e0d\u4e0d\u518d\u60f3 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u80fd \u7231\u4e0d\u8d70\u7684\u592a\u5feb\u5c31\u50cf\u9f99\u5377\u98ce \u4e0d\u4e0d\u5f00\u4e0d\u6211\u5df2\u65e0\u4e0d\u53ef\u8eb2 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u518d\u518d\u60f3 \u6211\u4e0d\u518d\u518d\u60f3 \u6211\n -  \u6218\u4e89\u4e2d\u90e8\u961f \u6211\u60f3\u7684\u90a3\u8fdc \u6ca1\u9a6c\u7684\u591f\u591f \u666f\u666f\u5728\u8fc7 \u8fc7\u5230\u6211\u53bb\u8fc7\u4f60 \u662f\u4e0d\u5c31  \u5b50\u4eba\u311f\u5728\u6709 \u4e00\u5728\u5728\u5728\u5728\u4eba \u6709\u8eab\u4e3a\u6211\u7684\u591a\u5929 \u8fd9\u8eab\u4e3a\u9f99 \u5168\u4f7f\u5730\u53cc\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u6211\u311f\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u76f4\u622a\u68cd \u54fc\u54fc\u54c8\u516e \u5feb\u4f7f\u7528\u76f4\u622a\u68cd \u54fc\u54fc \n\n"
 }
]
```

可以看到一开始学到简单的字符，然后简单的词，接着是复杂点的词，然后看上去似乎像个句子了。

## 结论

通过隐藏状态，循环神经网络很够更好的使用数据里的时序信息。

## 练习

* 调调参数（例如数据集大小、序列长度和学习率），看看对Perplexity和预测的结果造成的区别。
* 在随机批量采样中，如果在同一个epoch中只把隐藏状态在该epoch开始的时候初始化会怎么样？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/989)
