# 多头注意力

:label:`sec_multihead-attention`

在实践中，给定相同集合的查询、键和值，我们可能希望模型将来自同一注意力机制不同行为的知识结合起来，例如：捕获序列内各种范围的依赖关系（例如，短范围与长距离）。因此，允许注意力机制联合使用查询、键和值的不同表示子空间可能是有益的。

为此，可以使用 $h$ 个独立学习的线性投影来转换查询、键和值，而不是执行单一的注意力池化。然后，这些 $h$ 个投影后的查询、键和值将并行地输入注意力池化。最后，$h$ 个注意力池化输出被连接在一起，使用另一个学习后的线性投影进行转换，以产生最终输出。这种设计被称为 * 多头注意力 *，其中 $h$ 个注意力池化的每一个输出称为一个 * 头 * :cite:`Vaswani.Shazeer.Parmar.ea.2017`。:numref:`fig_multi-head-attention` 使用全连接层来执行可学习的线性变换，描述多头注意力。

![Multi-head attention, where multiple heads are concatenated then linearly transformed.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`

## 模型

在实现多头注意力之前，先以数学方式将这个模型正规化。给定一个查询 $\mathbf{q} \in \mathbb{R}^{d_q}$、一个键 $\mathbf{k} \in \mathbb{R}^{d_k}$ 和一个值 $\mathbf{v} \in \mathbb{R}^{d_v}$，每个注意力的头 $\mathbf{h}_i$ ($i = 1, \ldots, h$) 的计算方法为

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

其中，可以学习的参数 $\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$ 和 $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$ 以及 $f$ 是注意力池化，例如 :numref:`sec_attention-scoring-functions` 中的可加性注意力和缩放的“点－积”注意力。多头注意力输出是另一种线性转换，通过可以学习的参数 $\mathbf W_o\in\mathbb R^{p_o\times h p_v}$ 将 $h$ 个头连接在一起：

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

基于这种设计，每个头会将注意力放在输入的不同部分。因此，可以表示比简单的加权平均值更复杂的函数。

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

## 实现

在实现过程中，我们为多头注意力的每个头选择使用缩放的“点－积”注意力。为避免计算成本和参数化成本的显著增长，我们设置了 $p_q = p_k = p_v = p_o / h$。如果我们将查询、键和值的线性变换的输出数量设置为 $p_q h = p_k h = p_v h = p_o$，则可以并行计算 $h$ 头。在下面的实现中，$p_o$ 是通过参数 `num_hiddens` 指定的。

```{.python .input}
#@save
class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # 'queries' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_hiddens')
        # 'valid_lens' 的形状：('batch_size',) 或者 ('batch_size', 查询的个数)
        # 变换后，输出的 'queries', 'keys', 'values' 的形状：
        # ('batch_size'*'num_heads', 查询或者“键－值”对的个数, 'num_hiddens'/'num_heads')
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在 axis=0，拷贝第一项（标题或者失量）'num_heads' 次；然后拷贝下一项；等等
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # 'output' 的形状：('batch_size'*'num_heads', 查询的个数, 'num_hiddens'/'num_heads')
        output = self.attention(queries, keys, values, valid_lens)
        
        # 'output_concat' 的形状：('batch_size', 查询的个数, 'num_hiddens')
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab pytorch
#@save
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 'queries' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_hiddens')
        # 'valid_lens' 的形状：('batch_size',) 或者 ('batch_size', 查询的个数)
        # 变换后，输出的 'queries', 'keys', 'values' 的形状：
        # ('batch_size'*'num_heads', 查询或者“键－值”对的个数, 'num_hiddens'/'num_heads')
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在 axis=0，拷贝第一项（标题或者失量）'num_heads' 次；然后拷贝下一项；等等
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 'output' 的形状：('batch_size'*'num_heads', 查询的个数, 'num_hiddens'/'num_heads')
        output = self.attention(queries, keys, values, valid_lens)

        # 'output_concat' 的形状：('batch_size', 查询的个数, 'num_hiddens')
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

为了允许多个头的并行计算，上面的 `MultiHeadAttention` 类使用了下面定义的两个转置函数。特别说明，`transpose_output` 函数是 `transpose_qkv` 函数的逆操作。

```{.python .input}
#@save
def transpose_qkv(X, num_heads):
    # 输入 'X' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_hiddens')
    # 输出 'X' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_heads')
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 'X' 的形状：
    # ('batch_size', 'num_heads', 查询或者“键－值”对的个数, 'num_hiddens' / 'num_heads')
    X = X.transpose(0, 2, 1, 3)

    # `output` 的形状：
    # (`batch_size` * `num_heads`, 查询或者“键－值”对的个数, `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """ `transpose_qkv` 的逆运算"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab pytorch
#@save
def transpose_qkv(X, num_heads):
    # 输入 'X' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_hiddens')
    # 输出 'X' 的形状：('batch_size', 查询或者“键－值”对的个数, 'num_heads')
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 'X' 的形状：
    # ('batch_size', 'num_heads', 查询或者“键－值”对的个数, 'num_hiddens' / 'num_heads')
    X = X.permute(0, 2, 1, 3)

    # `output` 的形状：
    # (`batch_size` * `num_heads`, 查询或者“键－值”对的个数, `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """ `transpose_qkv` 的逆运算"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

试着使用一个小例子来测试实现的 `MultiHeadAttention` 类，例子中的键和值相同的。因此，多头注意力输出的形状是（`batch_size`、`num_queries`、`num_hiddens`）。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab all
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
```

## 摘要

* 多头注意力通过查询、键和值的不同表示子空间将同一注意力池化的知识结合在一起。
* 为了并行计算多头注意力的那些头，需要使用恰当的张量操作。

## 练习

1. 将这个实验中的多个头的注意力权重可视化。
1. 假设有一个基于多头注意力的训练好的模型，我们希望裁剪不重要的注意力头以提高预测速度。我们应该如何设计实验来衡量注意力的头的重要性？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1635)
:end_tab:
