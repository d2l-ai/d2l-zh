# 循环神经网络
:label:`sec_rnn`

在 :numref:`sec_language_model` 中，我们推出了 $n$ 克模型，其中，在时间步长 $x_t$ 的条件概率仅取决于前面的单词 $n-1$。如果我们想在 $x_t$ 上纳入早于时间步长 $t-(n-1)$ 的可能影响，我们需要增加 $n$。然而，模型参数的数量也会随之呈指数增加，因为我们需要为词汇集 $\mathcal{V}$ 存储 $|\mathcal{V}|^n$ 数字。因此，最好使用潜在变量模型，而不是建模 $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$：

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

其中 $h_{t-1}$ 是一个 * 隐藏状态 *（也称为隐藏变量），用于存储时间步骤 $t-1$ 之前的序列信息。通常，可以根据当前输入 $x_{t}$ 和之前的隐藏状态 $h_{t-1}$ 计算任何时间步骤 $t$ 的隐藏状态：

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

对于 :eqref:`eq_ht_xt` 中一个足够强大的函数 $f$，潜变量模型不是一个近似值。毕竟，$h_t$ 可以简单地存储到目前为止观察到的所有数据。但是，它可能会使计算和存储成本高昂。

回想一下，我们已经讨论了隐藏的图层与隐藏的单位在 :numref:`chap_perceptrons`。值得注意的是，隐藏层和隐藏状态指的是两个非常不同的概念。如所述，隐藏图层是从输入到输出的路径上的视图中隐藏的图层。隐藏状态在技术上是对我们在给定步骤执行的任何操作的 * 输入 *，并且它们只能通过查看以前时间步长的数据来计算。

*循环神经网络 * (RNS) 是具有隐藏状态的神经网络。在介绍 RNN 模型之前，我们首先重新介绍了 :numref:`sec_mlp` 中引入的 MLP 模型。

## 无隐藏状态的神经网络

让我们来看看具有单个隐藏层的 MLP。让隐藏层的激活函数为 $\phi$。给定一个小批处理大小为 $n$ 和 $d$ 输入的示例，隐藏图层的输出 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 计算为

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

在 :eqref:`rnn_h_without_state` 中，我们具有隐藏层的权重参数 $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$、偏置参数 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ 和隐藏单位的数量。接下来，隐藏变量 $\mathbf{H}$ 用作输出图层的输入。输出图层由

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中 $\mathbf{O} \in \mathbb{R}^{n \times q}$ 是输出变量，$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ 是权重参数，$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ 是输出层的偏置参数。如果这是一个分类问题，我们可以使用 $\text{softmax}(\mathbf{O})$ 来计算输出类别的概率分布。

这完全类似于我们之前在 :numref:`sec_sequence` 中解决的回归问题，因此我们省略了细节。我们可以随机选择特征标签对，并通过自动分化和随机梯度下降来了解我们网络的参数。

## 具有隐藏状态的循环神经网络
:label:`subsec_rnn_w_hidden_states`

事情是完全不同的，当我们有隐藏的状态。让我们更详细地看一下结构。

假设我们在时间步骤 $t$ 时有一个小批量的输入。换句话说，对于 $n$ 序列示例的微型批次，$\mathbf{X}_t$ 的每一行对应于序列中的时间步骤 $t$ 的一个示例。接下来，用 $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ 表示时间步长 $t$ 的隐藏变量。与 MLP 不同，这里我们保存了前一个时间步长的隐藏变量 $\mathbf{H}_{t-1}$，并引入了一个新的权重参数 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 来描述如何在当前时间步长中使用前一时间步长的隐藏变量。具体而言，当前时间步长的隐藏变量的计算取决于当前时间步长的输入以及前一个时间步长的隐藏变量：

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

与 :eqref:`rnn_h_without_state` 相比，:eqref:`rnn_h_with_state` 又增加了一个术语，从而实例化了 :eqref:`eq_ht_xt`。从相邻时间步长的隐藏变量 $\mathbf{H}_t$ 和 $\mathbf{H}_{t-1}$ 之间的关系中，我们知道这些变量捕获并保留了序列的历史信息直到其当前时间步长，就像神经网络当前时间步长的状态或内存一样。因此，这样的隐藏变量称为 * 隐藏状态 *。由于隐藏状态使用与当前时间步长中上一个时间步长相同的定义，因此 :eqref:`rnn_h_with_state` 的计算是 * 重复出现 *。因此，基于循环计算的隐藏状态的神经网络被命名为
*复发神经网络 *。
在 RNS 中执行 :eqref:`rnn_h_with_state` 计算的图层称为 * 循环图层 *。

构建 RNS 有很多不同的方法。具有 :eqref:`rnn_h_with_state` 定义的隐藏状态的 RNS 非常常见。对于时间步长 $t$，输出层的输出与 MLP 中的计算相似：

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

RNN 的参数包括权重 $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 和隐藏层的偏置 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重 $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ 和偏置 $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$。值得一提的是，即使在不同的时间步骤，RNS 也始终使用这些模型参数。因此，RNN 的参数化成本不会随着时间步长的增加而增加。

:numref:`fig_rnn` 说明了 RNN 在三个相邻时间步长的计算逻辑。在任何时间步骤 $t$ 中，隐藏状态的计算可视为：i) 将当前时间步长 $t$ 的输入 $\mathbf{X}_t$ 和前一时间步骤 $t-1$ 的隐藏状态连接到一个完全连接的层中；ii) 通过激活函数。这样一个完全连接的层的输出是当前时间步长 $t$ 的隐藏状态 $\mathbf{H}_t$。在这种情况下，模型参数是 $\mathbf{W}_{xh}$ 和 $\mathbf{W}_{hh}$ 的连接，以及偏置为 $\mathbf{b}_h$，所有这些参数都是从 :eqref:`rnn_h_with_state` 开始的。当前时间步骤 $t$，$\mathbf{H}_t$ 的隐藏状态将参与计算下一个时间步骤 $t+1$ 的隐藏状态。此外，$\mathbf{H}_t$ 还将被馈入完全连接的输出层，以计算当前时间步长 $t$ 的输出 $\mathbf{O}_t$。

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

我们刚才提到的是，隐藏状态的 $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ 计算相当于矩阵乘法，以及 $\mathbf{X}_t$ 和 $\mathbf{H}_{t-1}$ 的串联和 $\mathbf{W}_{xh}$ 和 $\mathbf{W}_{hh}$ 的串联。虽然这可以在数学中证明，但在下面我们只是使用一个简单的代码片段来显示这一点。首先，我们定义了矩阵 `X`、`W_xh`、`H` 和 `W_hh`，这些矩阵的形状分别为（3、1）、（1、4）、（3、4）和（4、4）。乘以 `X` 乘以 `W_xh` 和 `H`，分别乘以 `H`，然后再加上这两个乘法，我们得到了一个形状矩阵（3，4）。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

现在，我们将矩阵 `X` 和 `H` 沿列（轴 1）连接起来，以及沿行（轴 0）的矩阵 `W_xh` 和 `W_hh`。这两个连接分别产生形状（3、5）和形状（5、4）的矩阵。乘以这两个串联矩阵，我们得到了与上述相同的形状（3，4）输出矩阵。

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## 基于 RNN 的字符级语言模型

回想一下，对于 :numref:`sec_language_model` 中的语言建模，我们的目标是根据当前和过去的令牌预测下一个令牌，因此我们将原始序列移动一个标记作为标签。现在我们说明如何使用 RNS 来构建语言模型。让小批量大小为 1，文本的序列是 “机器”。为了简化后续部分中的训练，我们将文本标记为字符而不是单词，并考虑使用 * 字符级语言模型 *。:numref:`fig_rnn_train` 演示了如何通过 RNN 预测字符级语言建模的基于当前和前一个字符的下一个字符。

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

在训练过程中，我们针对每个时间步长对输出层的输出运行 softmax 运算，然后使用交叉熵损耗来计算模型输出和标签之间的误差。由于隐藏层中隐藏状态的循环计算，:numref:`fig_rnn_train`、$\mathbf{O}_3$ 中的时间步骤 3 的输出由文本序列 “m”、“a” 和 “c” 确定。由于训练数据中序列的下一个字符是 “h”，所以时间步长 3 的损失将取决于基于特征序列 “m”、“a”、“c” 和此时间步长的标签 “h” 生成的下一个字符的概率分布。

实际上，每个令牌都由一个 $d$ 维向量表示，我们使用批量大小 $n>1$。因此，输入 $\mathbf X_t$ 在时间步长 $t$ 将是一个 $n\times d$ 矩阵，这与我们在 :numref:`subsec_rnn_w_hidden_states` 中讨论的内容相同。

## 困惑
:label:`subsec_perplexity`

最后，让我们讨论如何测量语言模型质量，这将用于评估我们基于 RN 的模型。一种方法是检查文本有多令人惊讶。一个好的语言模型能够预测高精度令牌，我们将看到什么。考虑不同语言模式提出的 “正在下雨” 短语的以下延续：

1. “外面正在下雨”
1. “下雨香蕉树”
1. “这是下雨的皮乌; KCJ 普维波伊特”

就质量而言，示例 1 显然是最好的。这些词是明智的，逻辑上连贯的。虽然它可能不太准确地反映哪个单词在语义上跟随（“旧金山” 和 “冬季” 本来是完全合理的扩展），但模型能够捕获下面哪种单词。例 2 通过产生一个无意义的扩展来更糟糕。尽管如此，至少该模型已经学会了如何拼写单词和单词之间的某种程度的相关性。最后，示例 3 表示训练不良的模型未正确拟合数据。

我们可以通过计算序列的可能性来衡量模型的质量。不幸的是，这是一个难以理解和难以比较的数字。毕竟，较短的序列比较长的序列更容易发生，因此评估托尔斯泰的巨大作品上的模型
*战争与和平 * 将不可避免地产生比圣艾修佩里的小说 * 小王子 * 的可能性要小得多。缺少的是平均值。

信息理论在这里派上用场。在引入软最大回归 (:numref:`subsec_info_theory_basics`) 时，我们已经定义了熵、意外和交叉熵，并在 [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) 中讨论了更多的信息理论。如果我们想压缩文本，我们可以询问如何预测给定当前令牌集合的下一个令牌。更好的语言模型应该使我们能够更准确地预测下一个标记。因此，它应该允许我们在压缩序列时花费更少的位。所以我们可以通过一个序列的所有 $n$ 令牌的平均交叉熵损失来测量它：

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

其中 $P$ 由语言模型给出，$x_t$ 是从序列中时间步骤 $t$ 观察到的实际标记。这使得在不同长度的文档上的性能具有可比性。由于历史原因，从事自然语言处理的科学家更愿意使用称为 * 困惑 * 的数量。简而言之，它是 :eqref:`eq_avg_ce_for_lm` 的指数：

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

困惑可以最好地理解为我们在决定接下来选择哪个令牌时所拥有的真实选择数量的谐波平均值。让我们来看看一些情况：

* 在最佳情况下，模型始终将标签标记的概率完美估计为 1。在这种情况下，模型的困惑是 1。
* 在最坏的情况下，模型始终将标签标记的概率预测为 0。在这种情况下，困惑是正无穷大。
* 在基线上，模型预测词汇的所有可用代币的均匀分布。在这种情况下，困惑等于词汇的唯一标记的数量。事实上，如果我们在没有任何压缩的情况下存储序列，这将是我们对其进行编码的最好方法。因此，这提供了一个非平凡的上限，任何有用的模型都必须击败。

在下面的章节中，我们将为字符级语言模型实现 RNS，并使用困惑来评估这些模型。

## 摘要

* 使用循环计算隐藏状态的神经网络称为循环神经网络 (RNN)。
* RNN 的隐藏状态可以捕获序列的历史信息，直到当前时间步长。
* 随着时间步长数的增加，RNN 模型参数的数量不会增加。
* 我们可以使用 RNN 创建字符级语言模型。
* 我们可以使用困惑来评估语言模型的质量。

## 练习

1. 如果我们使用 RNN 来预测文本序列中的下一个字符，那么任何输出的所需维度是什么？
1. 为什么 RNS 可以根据文本序列中的所有先前令牌在某个时间步骤表达令牌的条件概率？
1. 如果您通过长序列反向传播，渐变会发生什么情况？
1. 与本节中描述的语言模型相关的一些问题是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
