# 通过时间反向传播
:label:`sec_bptt`

到目前为止，我们已经反复提到像
*爆炸梯度 *,
*消失的梯度 *,
以及需要
*分离 RNS 的渐变 *。
例如，在 :numref:`sec_rnn_scratch` 中，我们在序列上调用了 `detach` 函数。所有这些都没有得到真正的解释，以便能够快速构建模型并了解它是如何工作的。在本节中，我们将更深入地探讨序列模型反向传播的细节，以及为什么（以及如何）数学的工作原理。

当我们首次实施 RNS (:numref:`sec_rnn_scratch`) 时，我们遇到了梯度爆炸的一些影响。特别是，如果你解决了练习，你会看到梯度裁剪对于确保适当收敛至关重要。为了更好地理解此问题，本节将回顾如何计算序列模型的梯度。请注意，它的工作原理上没有什么新概念。毕竟，我们仍然只是应用链规则来计算梯度。尽管如此，还是值得重新审查反向传播（:numref:`sec_backprop`）。

我们在 :numref:`sec_backprop` 中描述了 MLP 中的前向和向后传播和计算图。RNN 中的正向传播相对简单。
*通过时间的反向传播 * 实际上是一个特定的
反向传播技术在 R神经网络中的应用它要求我们一次扩展 RNN 的计算图，以获取模型变量和参数之间的依赖关系。然后，根据链规则，我们应用反向传播来计算和存储梯度。由于序列可能相当长，因此依赖关系可能相当冗长。例如，对于 1000 个字符的序列，第一个令牌可能会对最终位置的令牌产生重大影响。这在计算上并不是真正可行的（它需要太长时间，需要太多的内存），并且在我们得到这个非常难以捉摸的梯度之前，它需要超过 1000 个矩阵产品。这是一个充满计算和统计不确定性的过程。在下文中，我们将阐明会发生什么情况以及如何在实践中解决这一问题。

## RNs 中的梯度分析
:label:`subsec_bptt_analysis`

我们从 RNN 工作原理的简化模型开始。此模型忽略有关隐藏状态的详细信息及其更新方式的详细信息。这里的数学表示法没有像过去那样明确区分标量，向量和矩阵。这些细节对于分析并不重要，只会使本小节中的符号变得混乱。

在此简化模型中，我们将 $h_t$ 表示为隐藏状态，将 $x_t$ 表示为输入，将 $o_t$ 表示为时间步长 $t$ 的输出。回想一下我们在 :numref:`subsec_rnn_w_hidden_states` 中的讨论，即输入和隐藏状态可以连接为隐藏图层中的一个权重变量。因此，我们分别使用 $w_h$ 和 $w_o$ 来指示隐藏图层和输出图层的权重。因此，每次步骤的隐藏状态和输出都可以解释为

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

其中 $f$ 和 $g$ 分别是隐藏图层和输出图层的变换。因此，我们有一个值链 $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$，它们通过循环计算彼此依赖。正向传播相当简单。我们所需要的只是一次一步循环 $(x_t, h_t, o_t)$ 三倍。然后通过一个客观函数在所有 $T$ 时间步长中评估输出 $o_t$ 和所需标签 $y_t$ 之间的差异，

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

对于反向传播，问题有点棘手，特别是当我们计算相对于目标函数 $L$ 的参数 $w_h$ 的梯度时。具体来说，按照连锁规则，

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_h)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh` 中产品的第一个和第二个因素易于计算。第三个因素 $\partial h_t/\partial w_h$ 是事情变得棘手的地方，因为我们需要重复计算参数 $w_h$ 对 $h_t$ 的影响。根据第 :eqref:`eq_bptt_ht_ot` 号文件中的经常性计算结果，$h_t$ 号文件的计算结果也取决于 $h_{t-1}$ 号文件和 $w_h$ 号文件的计算结果。因此，使用链规则产生

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

为了得出上述梯度，假设我们有三个序列满足 $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ 和 $a_{t}=b_{t}+c_{t}a_{t-1}$ 的顺序。然后对于 $t\geq 1$，它很容易显示

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

通过以下方式取代了

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

的梯度计算符合 $a_{t}=b_{t}+c_{t}a_{t-1}$ 的要求。因此，根据 :eqref:`eq_bptt_at`，我们可以删除

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

虽然我们可以使用链规则递归地计算 $\partial h_t/\partial w_h$，但只要 $t$ 很大，这个链就会变得很长。让我们讨论处理这一问题的若干战略.

### 完整计算 ##

显然，我们可以计算 :eqref:`eq_bptt_partial_ht_wh_gen` 中的全部总和。然而，这非常缓慢，渐变可能会爆炸，因为初始条件的微妙变化可能会对结果产生很大影响。也就是说，我们可以看到类似于蝴蝶效应的东西，初始条件的最小变化导致结果的不成比例的变化。就我们要估计的模型而言，这实际上是相当不可取的。毕竟，我们正在寻找能够很好地概括出来的可靠估计数。因此，这种战略几乎从未在实践中使用过。

### 截断时间步骤 ##

或者，我们可以在 $\tau$ 步骤后截断总和。这就是我们到目前为止一直在讨论的内容，例如当我们在 :numref:`sec_rnn_scratch` 中分离渐变时。这导致真实梯度的 * 近似 *，只需将总和终止为 $\partial h_{t-\tau}/\partial w_h$。在实践中，这工作得很好。它通常被称为经过时间的截断反向推进 :cite:`Jaeger.2002`。这样做的后果之一是，该模式主要侧重于短期影响，而不是长期后果。这实际上是 * 可取的 *，因为它会将估计值偏向更简单和更稳定的模型。

### 随机截断 ##

最后，我们可以用预期正确但截断序列的随机变量替换 $\partial h_t/\partial w_h$。这是通过使用预定义的 $\xi_t$ 序列来实现的，其中包括 $\xi_t$ 和 $P(\xi_t = \pi_t^{-1}) = \pi_t$，因此是 $E[\xi_t] = 1$。我们使用它来替换 :eqref:`eq_bptt_partial_ht_wh_recur` 中的渐变 $\partial h_t/\partial w_h$

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

根据第 $\xi_t$ 号决议的定义，可以得出这样的结论。每当 $\xi_t = 0$ 时，循环计算在该时间步骤 $t$ 终止。这会导致长度不同的序列的加权总和，其中长序列很少见，但过重适当。这个想法是由塔莱克和奥利维尔 :cite:`Tallec.Ollivier.2017` 提出的。

### 比较策略

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt` 说明了在分析 * 时间机器 * 书的前几个字符时使用反向传播为 RNs 的 RNs 时间机器 * 书中的三种策略：

* 第一行是将文本划分为不同长度的段的随机截断。
* 第二行是将文本分解为相同长度的子序列的常规截断。这就是我们在 RNN 实验中一直在做的。
* 第三行是通过时间的完全反向传播，导致计算上不可行的表达式。

遗憾的是，虽然理论上具有吸引力，但随机截断并不比常规截断更好，很可能是由于多种因素。首先，经过一系列反向传播步骤后的观测结果足以捕获实际依赖关系。其次，方差增加抵消了随着步骤越多，渐变更精确的事实。第三，我们实际上 * 希望 * 只有短范围交互的模型。因此，经常截断的反向传播随着时间的推移具有轻微的正则化效果，这是可取的。

## 通过时间的反向传播详细信息

在讨论一般原则之后，让我们详细讨论反向传播问题。与 :numref:`subsec_bptt_analysis` 中的分析不同，下面我们将展示如何计算目标函数相对于所有分解模型参数的梯度。为了保持简单，我们考虑一个没有偏差参数的 RNN，其在隐藏层中的激活函数使用身份映射 ($\phi(x)=x$)。对于时间步长 $t$，请将单个示例输入和标签分别为 $\mathbf{x}_t \in \mathbb{R}^d$ 和 $y_t$。隐藏状态和输出 $\mathbf{o}_t \in \mathbb{R}^q$ 的计算方法是

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

其中重量参数为 $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 和 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 和 $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。以 $l(\mathbf{o}_t, y_t)$ 表示在时间步骤 $t$ 的损失。我们的客观函数, 损失超过 $T$ 从序列开始的时间步骤是因此

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

为了在 RNN 计算过程中可视化模型变量和参数之间的依赖关系，我们可以为模型绘制一个计算图，如 :numref:`fig_rnn_bptt` 所示。例如，时间步长 3、$\mathbf{h}_3$ 的隐藏状态的计算取决于模型参数 $\mathbf{W}_{hx}$ 和 $\mathbf{W}_{hh}$、最后一个时间步长 $\mathbf{h}_2$ 的隐藏状态以及当前时间步长 $\mathbf{x}_3$ 的输入。

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

如上所述，在 :numref:`fig_rnn_bptt` 中的模型参数分别为 $\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$ 和 $\mathbf{W}_{qh}$。通常，训练该模型需要对这些参数 $\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$ 和 $\partial L/\partial \mathbf{W}_{qh}$ 进行梯度计算。根据 :numref:`fig_rnn_bptt` 中的依赖关系，我们可以沿箭头的相反方向遍历来计算和存储渐变。为了灵活地表达链规则中不同形状的矩阵、向量和标量的乘法，我们继续使用 $\text{prod}$ 运算符，如 :numref:`sec_backprop` 所述。

首先，随时区分模型输出的目标函数相对于步骤 $t$ 非常简单：

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

现在，我们可以根据输出层中的参数 $\mathbf{W}_{qh}$ 计算目标函数的梯度：$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$。根据 :numref:`fig_rnn_bptt` 计算，目标函数取决于使用链规则产生

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

其中提供的资金是以 :eqref:`eq_bptt_partial_L_ot` 为单位的。

接下来，如 :numref:`fig_rnn_bptt` 所示，在最后一个时间步骤 $T$ 目标函数依赖于隐藏状态 $\mathbf{h}_T$，只有通过 $\mathbf{o}_T$。因此，我们可以使用链规则轻松找到渐变 $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$：

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

在任何时间步骤中，它都会变得更加棘手，在这种情况下，目标功能 $L$ 取决于 $\mathbf{h}_t$ 和 $\mathbf{o}_t$。根据链规则，隐藏状态 $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ 在任何时间步骤 $t < T$ 的梯度可以重复计算为：

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

为了进行分析，扩展任何时间步骤 $1 \leq t \leq T$ 的循环计算

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

我们可以从 :eqref:`eq_bptt_partial_L_ht` 中看到，这个简单的线性例子已经展现了长序列模型的一些关键问题：它涉及到 $\mathbf{W}_{hh}^\top$ 的潜在非常大的功率。在其中，小于 1 的特征值消失，大于 1 的特征值发散。这在数字上是不稳定的，表现为消失和爆炸的梯度。解决这个问题的一种方法是以计算方便的大小截断时间步长，如 :numref:`subsec_bptt_analysis` 中所述。实际上，这种截断是通过在给定数量的时间步长后分离渐变来实现的。稍后我们将看到更复杂的序列模型如何进一步缓解这种情况。

最后，:numref:`fig_rnn_bptt` 表明，目标函数 $L$ 依赖于隐藏层中的模型参数 $\mathbf{W}_{hx}$ 和 $\mathbf{W}_{hh}$。为了计算相对于这些参数 $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ 和 $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 的梯度，我们应用了

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

其中通过 :eqref:`eq_bptt_partial_L_hT_final_step` 和 :eqref:`eq_bptt_partial_L_ht_recur` 重复计算的 $\partial L/\partial \mathbf{h}_t$ 是影响数值稳定性的关键数量。

正如我们在 :numref:`sec_backprop` 中所解释的那样，随着时间的反向传播是反向传播在 RNs 中的应用，训练 RNS 将随着时间的推移与反向传播交替。此外，通过时间的反向传播依次计算和存储上述梯度。具体而言，存储的中间值会被重复使用，以避免重复计算，例如存储 $\partial L/\partial \mathbf{h}_t$，以便在计算 $\partial L / \partial \mathbf{W}_{hx}$ 和 $\partial L / \partial \mathbf{W}_{hh}$ 时使用。

## 摘要

* 经过时间的反向传播仅仅是一种反向传播应用于具有隐藏状态的序列模型。
* 为了计算方便性和数值稳定性，例如常规截断和随机截断，需要截断。
* 矩阵的高功率可能导致特征值发散或消失。这表现为爆炸或消失的梯度。
* 为了有效计算，在随时间推移的反向传播期间缓存中间值。

## 练习

1. 假设我们有一个具有特征值 $\lambda_i$ 的对称矩阵，其对应的特征向量是 $\mathbf{v}_i$ ($i = 1, \ldots, n$)。在没有一般性损失的情况下，假设它们是按顺序 $|\lambda_i| \geq |\lambda_{i+1}|$ 订购的。
   1. 显示该系统具有特征值。
   1. 证明对于随机向量 $\mathbf{x} \in \mathbb{R}^n$，具有高概率 $\mathbf{M}^k \mathbf{x}$ 将与特征向量 $\mathbf{v}_1$ 非常一致
共计 $\mathbf{M}$。将此声明正式化。
   1. 上述结果对于 RNS 中的梯度意味着什么？
1. 除了梯度裁剪之外，你能想到任何其他方法来应对循环神经网络中的梯度爆炸吗？

[Discussions](https://discuss.d2l.ai/t/334)
