# 自注意力和位置编码
:label:`sec_self-attention-and-positional-encoding`

在深度学习中，我们经常使用 CNN 或 RNN 对序列进行编码。现在尝试基于注意力机制，通过将一个令牌序列输入到注意力池化模块中，以便查询、键和值使用的是同一组令牌。每个查询都会关注所有的“键－值”对，并生成一个注意力输出。由于查询、键和值来自同一个集合，因此执行的
*自注意力 * :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`，也称为 * 内部注意力 * :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`。
在本节中，我们将讨论使用自注意力的序列编码和序列顺序的附加信息。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## 自注意力

给定一个令牌输入序列 $\mathbf{x}_1, \ldots, \mathbf{x}_n$，其中任何 $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$)，它的自注意力输出是一个长度相同的序列 $\mathbf{y}_1, \ldots, \mathbf{y}_n$，其中

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

根据 :eqref:`eq_attn-pooling` 中注意力池化函数 $f$ 的定义。下面的代码片段使用多头注意力计算张量的自注意力，输入张量与输出张量的形状都是（批量大小、时间步长或令牌中的序列长度，令牌维度 $d$）。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab all
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

## 比较 CNN、RNN 和自注意力
:label:`subsec_cnn-rnn-self-attention`

试着比较能够将令牌长度为 $n$ 的序列映射到另一个长度相等的序列的不同架构，架构中的每个输入令牌或者每个输出令牌都是由 $d$ 维矢量表示。具体就是对比 CNN、RNN 和自注意力三种架构，比较它们的计算复杂性、是否顺序操作以及最大路径长度。请注意，顺序操作会阻止并行计算，而任意序列位置组合之间的路径越短，则学习序列中的远距离依赖关系就越容易 :cite:`Hochreiter.Bengio.Frasconi.ea.2001`。

![Comparing CNN (padding tokens are omitted), RNN, and self-attention architectures.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

考虑一个内核大小为 $k$ 的卷积层。我们将在后面的章节中提供有关使用 CNN 处理序列的更多详细信息。目前，我们只需要知道，由于序列长度是 $n$，输入通道和输出通道的数量都是 $d$，因此卷积层的计算复杂度为 $\mathcal{O}(knd^2)$。如 :numref:`fig_cnn-rnn-self-attention` 所示，CNN 是分层的，因此有 $\mathcal{O}(1)$ 个顺序操作，最大路径长度为 $\mathcal{O}(n/k)$。例如，$\mathbf{x}_1$ 和 $\mathbf{x}_5$ 处于 :numref:`fig_cnn-rnn-self-attention` 中内核大小为 3 的双层 CNN 的感受野范围内。

当更新 RNN 的隐藏状态时，$d \times d$ 权重矩阵和 $d$ 维隐藏状态的乘法计算复杂度为 $\mathcal{O}(d^2)$。由于序列长度为 $n$，因此循环层的计算复杂度为 $\mathcal{O}(nd^2)$。根据 :numref:`fig_cnn-rnn-self-attention`，有 $\mathcal{O}(n)$ 个顺序操作导致无法并行化计算，最大路径长度也是 $\mathcal{O}(n)$。

在自注意力中，查询、键和值都是 $n \times d$ 矩阵。考虑 :eqref:`eq_softmax_QK_V` 中的缩放的“点－积”注意力，其中 $n \times d$ 矩阵乘以 $d \times n$ 矩阵，然后输出 $n \times n$ 矩阵乘以 $n \times d$ 矩阵。因此，自注意力具有 $\mathcal{O}(n^2d)$ 计算复杂性。正如我们在 :numref:`fig_cnn-rnn-self-attention` 中看到的那样，每个令牌都通过自注意力直接连接到任何其他令牌。因此，由于 $\mathcal{O}(1)$ 顺序操作导致计算可以并行，而且最大路径长度也是 $\mathcal{O}(1)$。

总而言之，CNN 和自注意力都可以享受并行计算，而且自注意力的最大路径长度最短。但是，自注意力的计算复杂度为序列长度的二次方，因此在很长的序列中其计算速度非常缓慢。

## 位置编码
:label:`subsec_positional-encoding`

与逐个重复处理序列令牌的 RNN 不同，自注意力放弃了顺序操作，而倾向于并行计算。为了使用序列的顺序信息，可以通过在输入表示中添加 * 位置编码 * 来注入绝对或相对位置信息。位置编码可以固定的值，也可以是可学习的参数。下面的基于正弦和余弦函数的位置编码函数就能输出固定值 :cite:`Vaswani.Shazeer.Parmar.ea.2017`。

假设输入表示 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 包含一个序列，令牌序列长度为 $n$，每个令牌是 $d$ 维的嵌入表示。位置编码使用相同形状的位置嵌入矩阵 $\mathbf{P} \in \mathbb{R}^{n \times d}$ ，最终输出为 $\mathbf{X} + \mathbf{P}$，该矩阵在 $i^\mathrm{th}$ 行和 $(2j)^\mathrm{th}$ 或 $(2j + 1)^\mathrm{th}$ 列上的元素为

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

乍一看，这种三角函数的设计很奇怪。在解释之前，让我们先实现 `PositionalEncoding` 类。

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

在位置嵌入矩阵 $\mathbf{P}$ 中，行对应于序列中的位置，列表示不同的位置编码维度。在下面的示例中，可以看到位置嵌入矩阵的 $6^{\mathrm{th}}$ 和 $7^{\mathrm{th}}$ 列的频率高于 $8^{\mathrm{th}}$ 和 $9^{\mathrm{th}}$ 列。$6^{\mathrm{th}}$ 和 $7^{\mathrm{th}}$ 列之间的偏移量是由于正弦函数和余弦函数的交替，$8^{\mathrm{th}}$ 和 $9^{\mathrm{th}}$ 列也是如此。

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### 绝对位置信息

为了了解沿着编码维度单调递减的频率与绝对位置信息的相关性，让我们打印出 $0, 1, \ldots, 7$ 的二进制表示形式。正如我们所看到的，每个数字、每两个数字和每四个数字上的最低位、第二位和第三位最低位分别交替。

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

在二进制表示中，较高位的比特相比于较低位的比特的变换频率更低。同样，如下面的热图所示，位置编码通过使用三角函数沿着编码维度降低频率。由于输出是浮点数，因此这样的连续表示比二进制表示法更节省空间。

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 相对位置信息

除了捕获绝对位置信息之外，上述的位置编码还允许模型通过学习加入相对位置信息。这是因为对于任何固定的位置偏移量 $\delta$，位置 $i + \delta$ 处的位置编码可以用位置 $i$ 的线性投影来表示。

这种投影的数学解释为：代表 $\omega_j = 1/10000^{2j/d}$，对于任何固定的偏移量 $\delta$，:eqref:`eq_positional-encoding-def` 中的任何一对 $(p_{i, 2j}, p_{i, 2j+1})$ 都可以线性投影到 $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$：

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

其中，$2\times 2$ 投影矩阵不依赖于任何位置索引 $i$。

## 摘要

* 在自注意力中，查询、键和值都来自同一个地方。
* CNN 和自注意力都能得到并行计算的好处，并且自注意力的最大路径长度最短。但是，因为自注意力的计算复杂度为序列长度的二次方，因此在很长的序列中其计算速度非常缓慢。
* 为了使用序列的顺序信息，可以通过在输入表示中添加位置编码来注入绝对或相对位置信息。

## 练习

1. 假设使用深层架构来表示序列，这种通过堆叠包含了位置编码的自注意力层实现的架构可能存在什么问题？
1. 尝试设计一种可以学习得到的位置编码方法？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:
