# 数值稳定性和初始化
:label:`sec_numerical_stability`

到目前为止，我们实现的每个模型都要求我们根据一些预先指定的分布初始化其参数。到目前为止，我们认为初始化方案是理所当然的，掩盖了这些选择是如何做出的细节。你甚至可能会得到这些选择并不是特别重要的印象。相反，初始化方案的选择在神经网络学习中起着重要作用，对于保持数值稳定性至关重要。此外，这些选择可以与非线性激活函数的选择相结合。我们选择哪个函数以及我们如何初始化参数可以决定我们的优化算法收敛的速度。这里的选择不佳可能会导致我们在训练过程中遇到爆炸或消失的梯度。在本节中，我们深入探讨这些主题，并讨论一些有用的启发式学习方法，你会发现在你的职业生涯中对深度学习有用。

## 消失和爆炸渐变

考虑一个具有 $L$ 图层、输入 $\mathbf{x}$ 和输出 $\mathbf{o}$ 的深度网络。每层 $l$ 由变换定义 $f_l$ 由权重 $\mathbf{W}^{(l)}$ 参数化，其隐藏变量为 $\mathbf{h}^{(l)}$（让 $\mathbf{h}^{(0)} = \mathbf{x}$），我们的网络可以表示为：

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

如果所有隐藏的变量和输入都是向量，我们可以写 $\mathbf{o}$ 相对于任何一组参数 $\mathbf{W}^{(l)}$ 的梯度，如下所示：

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

换句话说，这个梯度是 $L-l$ 矩阵和梯度向量 $\mathbf{v}^{(l)}$ 的乘积。因此，我们很容易遇到同样的数值下流问题，当相乘的概率过多时，往往会出现这些问题。在处理概率时，一个常见的技巧是切换到对数空间，即将压力从尾数转移到数值表示的指数。不幸的是，我们上面的问题更严重：最初矩阵 $\mathbf{M}^{(l)}$ 可能有各种各样的特征值。它们可能是小型或大型，并且它们的商品可能是 * 非常大 * 或 * 非常小 *。

梯度不稳定所构成的风险超出了数字表示范围。不可预测的幅度梯度也威胁到我们优化算法的稳定性。我们可能面临的参数更新可能是（i）过大，破坏我们的模型（* 爆炸梯度 * 问题）；或（ii）过小（* 消失梯度 * 问题），使得学习不可能，因为参数在每次更新中很难移动。

### 消失渐变

导致梯度问题消失的一个常见的罪魁祸首是选择激活函数 $\sigma$，该函数在每个图层的线性运算之后附加。从历史上看，号格模函数 $1/(1 + \exp(-x))$（在 :numref:`sec_mlp` 中引入）很受欢迎，因为它类似于阈值函数。由于早期的人工神经网络受到生物神经网络的启发，因此激发 * 完全 * 或 * 根本不 *（如生物神经元）的神经元的想法似乎有吸引力。让我们仔细看一下西格莫图，看看它为什么会导致渐变消失。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

正如你所看到的，当它的输入很大和它们很小时，sigmoid 的梯度都会消失。此外，当通过许多层反向传播时，除非我们处于 Goldilocks 区域，其中许多西格莫体的输入接近零，否则整体积的渐变可能会消失。当我们的网络拥有许多图层时，除非我们小心，否则梯度可能会在某个图层被切断。事实上，这个问题曾经困扰着深度网络培训。因此，RelU 更稳定（但神经上不太合理），已成为从业人员的默认选择。

### 分解渐变

相反的问题，当梯度爆炸时，也可能同样令人烦恼。为了更好地说明这一点，我们绘制了 100 个高斯随机矩阵，并将它们与一些初始矩阵相乘。对于我们选择的尺度（方差 $\sigma^2=1$ 的选择），矩阵积爆炸。当由于深度网络的初始化而发生这种情况时，我们没有机会获得梯度下降优化器收敛。

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### 打破对称性

神经网络设计中的另一个问题是它们的参数化固有的对称性。假设我们有一个简单的 MLP 与一个隐藏层和两个单位。在这种情况下，我们可以排列第一个图层的权重 $\mathbf{W}^{(1)}$，并同样排列输出图层的权重以获得相同的功能。没有什么特别的区分第一个隐藏单元与第二个隐藏单元。换句话说，我们在每个图层的隐藏单位之间具有排列对称性。

这不仅仅是一种理论上的滋扰。考虑上述具有两个隐藏单位的单层 MLP。为了举例说明，假设输出图层将两个隐藏单位转换为仅一个输出单位。想象一下，如果我们将隐藏层的所有参数初始化为 $\mathbf{W}^{(1)} = c$，则会发生什么情况。在这种情况下，在正向传播过程中，任何隐藏单元都需要相同的输入和参数，从而产生相同的激活，这是馈送到输出单元。在反向传播期间，相对于参数 $\mathbf{W}^{(1)}$ 来区分输出单位会给出一个梯度，其元素都具有相同的值。因此，在基于梯度的迭代（例如，微型批次随机梯度下降）之后，$\mathbf{W}^{(1)}$ 的所有元素仍然采用相同的值。这样的迭代永远不会打破对称 *，我们可能永远无法实现网络的表现力。隐藏层的行为就好像它只有一个单元。请注意，虽然迷你批次随机梯度下降不会破坏这种对称性，但丢弃法正则化会！

## 参数初始化

解决上面提出的问题（或至少缓解）的一种方法是通过仔细的初始化。优化过程中的额外注意和适当的正则化可以进一步提高稳定性。

### 默认初始化

在前面的章节中，例如，在 :numref:`sec_linear_concise` 中，我们使用正态分布来初始化我们的权重值。如果我们不指定初始化方法，框架将使用默认的随机初始化方法，这通常适用于中等问题大小。

### 泽维尔初始化

让我们来看一下输出（例如隐藏变量）$o_{i}$ 对于某些完全连接层的比例分布
*没有非线性 *。
对于该图层的输入 $x_j$ 及其相关权重 $w_{ij}$，输出由

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

权重 $w_{ij}$ 都是独立于同一分布绘制的。此外，让我们假设此分布具有零均值和方差 $\sigma^2$。请注意，这并不意均值分布必须是高斯，只是平均值和方差需要存在。现在，让我们假设图层 $x_j$ 的输入也具有零均值和方差 $\gamma^2$，并且它们独立于 $w_{ij}$，并且彼此独立。在这种情况下，我们可以按如下方式计算 $o_i$ 的均值和方差：

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

保持方差固定的一种方法是设置 $n_\mathrm{in} \sigma^2 = 1$。现在考虑反向传播。在那里，我们面临类似的问题，尽管渐变从更接近输出的图层传播。使用与正向传播相同的推理，我们看到梯度的方差可能会爆炸，除非 $n_\mathrm{out} \sigma^2 = 1$，其中 $n_\mathrm{out}$ 是该层的输出数量。这使我们处于两难境地：我们不可能同时满足这两个条件。相反，我们只是试图满足：

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

这是现在标准和实际上有益的 *Xavier 初始化 * 的基础推理，以其创造者 :cite:`Glorot.Bengio.2010` 的第一作者命名。通常，Xavier 初始化从平均值为零且方差为 $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$ 的高斯分布中采样权重。我们还可以调整 Xavier 的直觉来选择从均匀分布取样权重时的方差。请注意，均匀分布 $U(-a, a)$ 具有方差。在 $\sigma^2$ 上将 $\frac{a^2}{3}$ 插入我们的条件中会产生根据

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

虽然上述数学推理中不存在非线性的假设在神经网络中很容易被违反，但 Xavier 初始化方法在实践中效果良好。

### 超越

上面的推理几乎没有划痕现代参数初始化方法的表面。深度学习框架通常实现十几种不同的启发式方法。此外，参数初始化仍然是深度学习基础研究的一个热点领域。其中包括专门用于绑定（共享）参数、超分辨率、序列模型和其他情况的启发式方法。实例，Xiao 等人利用精心设计的初始化方法 :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`，展示了在没有架构技巧的情况下训练 1000 层神经网络的可能性。

如果您感兴趣的话题，我们建议您深入了解本模块的产品，阅读提出并分析了每个启发式方法的论文，然后探索关于该主题的最新出版物。也许你会偶然发现，甚至发明一个聪明的想法，并为深度学习框架做出贡献。

## 摘要

* 渐变消失和爆炸是深度网络中常见的问题。在参数初始化过程中需要非常小心，以确保渐变和参数保持良好的控制。
* 需要初始化启发式方法来确保初始渐变既不太大也不太小。
* RelU 激活功能可缓解梯度问题。这可以加速收敛。
* 随机初始化是确保优化之前破坏对称性的关键。
* Xavier 初始化表明，对于每个图层，任何输出的方差不受输入数量的影响，任何梯度的方差都不受输出数量的影响。

## 练习

1. 你能否设计其他情况，神经网络可能表现出对称性，除了 MLP 层中的排列对称性之外，还需要打破？
1. 我们可以将线性回归或 softmax 回归中的所有权重参数初始化为相同的值吗？
1. 查找两个矩阵积的特征值的分析界限。这告诉你什么确保渐变条件良好？
1. 如果我们知道某些术语有发散，我们可以在事后解决这个问题吗？查看关于分层自适应速率缩放的论文，了解灵感 :cite:`You.Gitman.Ginsburg.2017`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
