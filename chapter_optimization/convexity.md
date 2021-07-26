# Conexity
:label:`sec_convexity`

Conexity 在优化算法的设计中起着至关重要的作用。这在很大程度上是因为在这种情况下分析和测试算法要容易得多。换句话说，如果算法即使在凸设置中也表现不佳，那么通常我们不应该希望看到很好的结果。此外，尽管深度学习中的优化问题通常是非凸出的，但它们往往表现出接近局部最小值的凸问题的一些特性。这可能会导致令人兴奋的新优化变体，例如 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## 定义

在凸分析之前，我们需要定义 * 凸集 * 和 * 凸函数 *。它们导致了通常应用于机器学习的数学工具。 

### 凸集

套装是凸度的基础。简而言之，如果对于任何 $a, b \in \mathcal{X}$，连接 $a$ 和 $b$ 的线段也是 $\mathcal{X}$，则矢量空间中的一组 $\mathcal{X}$ 为 * 凸 X *。从数学角度来说，这意味着对于所有 $\lambda \in [0, 1]$ 我们都有 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

这听起来有点抽象。考虑 :numref:`fig_pacman`。第一组不是凸面的，因为存在不包含在其中的线段。另外两套没有遇到这样的问题。 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

除非你能用它们做点什么，否则自己的定义并不是特别有用。在这种情况下，我们可以查看 :numref:`fig_convex_intersect` 所示的十字路口。假设 $\mathcal{X}$ 和 $\mathcal{Y}$ 是凸集。然后 $\mathcal{X} \cap \mathcal{Y}$ 也是凸起来的。要看到这一点，请考虑任何 $a, b \in \mathcal{X} \cap \mathcal{Y}$。由于 $\mathcal{X}$ 和 $\mathcal{Y}$ 是凸的，所以连接 $a$ 和 $b$ 的线段都包含在 $\mathcal{X}$ 和 $\mathcal{Y}$ 中。鉴于这一点，它们也需要包含在 $\mathcal{X} \cap \mathcal{Y}$ 中，从而证明我们的定理。 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我们可以很少努力加强这一结果：鉴于凸集 $\mathcal{X}_i$，它们的交叉点 $\cap_{i} \mathcal{X}_i$ 是凸的。要看看相反的情况是不正确的，请考虑两套不相交的 $\mathcal{X} \cap \mathcal{Y} = \emptyset$。现在选择 $a \in \mathcal{X}$ 和 $b \in \mathcal{Y}$。:numref:`fig_nonconvex` 中连接 $a$ 和 $a$ 和 $b$ 的线段需要包含一些既不在 $\mathcal{X}$ 中也不是 $\mathcal{Y}$ 中的部分，因为我们假设为 $\mathcal{X} \cap \mathcal{Y} = \emptyset$。因此，直线段也不在 $\mathcal{X} \cup \mathcal{Y}$ 中，从而证明了一般来说凸集的并集不需要凸起。 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

通常，深度学习中的问题是在凸集上定义的。例如，$\mathbb{R}^d$ 是一组 $d$ 维矢量的实数，是一个凸集（毕竟，$\mathbb{R}^d$ 中任意两个点之间的线保留在 $\mathbb{R}^d$ 中）。在某些情况下，我们使用有限长度的变量，例如 $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$ 定义的半径为 $r$ 的球。 

### 凸函数

现在我们已经有了凸集，我们可以引入 * 凸函数 * $f$。给定凸集 $\mathcal{X}$，函数 $f: \mathcal{X} \to \mathbb{R}$ 是 * 凸面 * 如果对于所有 $x, x' \in \mathcal{X}$ 和所有 $\lambda \in [0, 1]$ 我们都有 

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

为了说明这一点，让我们绘制一些函数并检查哪些功能符合要求。下面我们定义了一些函数，包括凸和非凸。

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

正如预期的那样，余弦函数是 * nonconvex*，而抛物线和指数函数是。请注意，要使条件有意义，需要 $\mathcal{X}$ 是凸集的要求。否则，$f(\lambda x + (1-\lambda) x')$ 的结果可能没有很好的界定。 

### Jensen 的不平等

鉴于凸函数 $f$，最有用的数学工具之一是 * Jensen 的不平等性 *。这相当于对凸度定义的概括： 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

其中 $\alpha_i$ 是非负实数，因此 $\sum_i \alpha_i = 1$ 和 $X$ 是一个随机变量。换句话说，对凸函数的期望不低于期望的凸函数，后者通常是一个更简单的表达式。为了证明第一个不平等，我们一次将凸度的定义应用于总和中的一个术语。 

延森不平等的常见应用之一是用一个更简单的表达来限制一个更复杂的表达方式。例如，它的应用可以是部分观察到的随机变量的对数可能性。也就是说，我们使用 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

自 $\int P(Y) P(X \mid Y) dY = P(X)$ 以来。这可以在变分方法中使用。这里 $Y$ 通常是未观察到的随机变量，$P(Y)$ 是对它可能如何分布的最佳猜测，$P(X)$ 是集成了 $Y$ 的分布。例如，在群集中，$Y$ 可能是集群标签，$P(X \mid Y)$ 是应用集群标签时的生成模型。 

## 属性

凸函数有许多有用的属性。我们在下面介绍一些常用的。 

### 本地 Minima 是全球最小值

首先，凸函数的局部最小值也是全局最小值。我们可以通过矛盾来证明这一点，如下所示。 

考虑在凸集 $\mathcal{X}$ 上定义的凸函数 $f$。假设 $x^{\ast} \in \mathcal{X}$ 是局部最低值：存在一个小的正值 $p$，所以对于 $x \in \mathcal{X}$ 满足 $0 < |x - x^{\ast}| \leq p$，我们有 $f(x^{\ast}) < f(x)$。 

假设本地最低位 $x^{\ast}$ 不是 $f$ 的全球最低值：存在 $x' \in \mathcal{X}$，其中 $f(x') < f(x^{\ast})$。还存在着 $\lambda \in [0, 1)$，例如 $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$，所以 $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$。  

但是，根据凸函数的定义，我们有 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

这与我们关于 $x^{\ast}$ 是当地最低限度的说法相矛盾.因此，不存在 $x' \in \mathcal{X}$，其中 $f(x') < f(x^{\ast})$。当地最低值 $x^{\ast}$ 也是全球最低水平。 

例如，凸函数 $f(x) = (x-1)^2$ 的局部最小值为 $x=1$，这也是全局最小值。

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函数的局部最小值也是全局最小值这一事实非常方便。这意味着，如果我们尽量减少功能，我们就不能 “卡住”。但是请注意，这并不意味着不能有一个以上的全局最低值，或者甚至可能存在一个。例如，函数 $f(x) = \mathrm{max}(|x|-1, 0)$ 在时间间隔 $[-1, 1]$ 内获得了最小值。相反，函数 $f(x) = \exp(x)$ 在 $\mathbb{R}$ 上没有达到最低值：对于 $x \to -\infty$，它渐近到 $0$，但没有 $x$，其中 $x$，其中 $f(x) = 0$。 

### 下面的凸函数集是凸

我们可以通过凸函数的 * 下面的集合 * 来方便地定义凸集。具体来说，给定在凸集 $\mathcal{X}$ 上定义的凸函数 $f$，下面的任何一组 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

是凸的。  

让我们快速证明这一点。回想一下，对于任何 $x, x' \in \mathcal{S}_b$，我们都需要展示 $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ 只要 $\lambda \in [0, 1]$。自 $f(x) \leq b$ 和 $f(x') \leq b$ 以来，根据凸度的定义，我们有  

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### 凸度和第二衍生品

只要函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 的第二个导数存在，就很容易检查 $f$ 是否凸。我们所需要做的就是检查 $f$ 的黑森州是否为正半定性：$\nabla^2f \succeq 0$，即，表示黑森州矩阵 $\nabla^2f$ 乘 $\mathbf{H}$，$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ 表示所有 $\mathbf{x} \in \mathbb{R}^n$。例如，函数 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ 自 $\nabla^2 f = \mathbf{1}$ 以来就是凸的，也就是说，它的黑森语是一个身份矩阵。 

从形式上来说，两次可分的一维函数 $f: \mathbb{R} \rightarrow \mathbb{R}$ 如果而且只有在其第二个导数 $f'' \geq 0$ 时是凸的。对于任何两次可分化的多维函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$，如果而且仅当黑森州 $\nabla^2f \succeq 0$ 时，它是凸的。 

首先，我们需要证明一维的情况。为了看到 $f$ 的凸度意味着 $f'' \geq 0$ 我们使用了这样一个事实： 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

由于第二个衍生物是由有限差异的限制给出的，因此 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

为了看到 $f'' \geq 0$ 意味着 $f$ 是凸的，我们使用的事实是 $f'' \geq 0$ 意味着 $f'$ 是一个单调的非递减函数。让 $a < x < b$ 成为 $\mathbb{R}$ 中的三点，其中 $x = (1-\lambda)a + \lambda b$ 和 $\lambda \in (0, 1)$。根据平均值定理，存在 $\alpha \in [a, x]$ 和 $\beta \in [x, b]$ 这样 

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

因此，通过单调性 $f'(\beta) \geq f'(\alpha)$ 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

自 $x = (1-\lambda)a + \lambda b$ 以来，我们有 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

从而证明了凸度。 

其次，在证明多维情况之前，我们需要一个词语：$f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是凸的，如果且只有在所有 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

是凸的。 

为了证明 $f$ 的凸度意味着 $g$ 是凸的，我们可以证明对于所有 $a, b, \lambda \in [0, 1]$（因此 $0 \leq \lambda a + (1-\lambda) b \leq 1$） 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

为了证明情况，我们可以证明对于所有 $\lambda \in [0, 1]$  

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

最后，使用上述词语和一维案例的结果，可以按如下方式证明多维情况。如果而且仅当所有 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$（其中 $z \in [0,1]$）都是凸的情况下，多维函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是凸的。根据一维情况，只有在 $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) 对于所有 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$（$\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$）的情况下，这相当于 $\mathbf{H} \succeq 0$，根据正半定矩阵的定义，这相当于 $\mathbf{H} \succeq 0$。 

## 限制

凸优化的一个不错的特性是它使我们能够有效地处理约束条件。也就是说，它使我们能够解决 * 受限的优化 * 形式的问题： 

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

其中 $f$ 是目标，函数 $c_i$ 是约束函数。看看这确实考虑了 $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$ 的情况。在这种情况下，参数 $\mathbf{x}$ 受限于单位球。如果第二个约束是 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$，那么这对应于放在半空间上的所有 $\mathbf{x}$。同时满足这两个限制条件等于选择一个球的一片。 

### 拉格朗日

一般来说，解决受限的优化问题很困难。解决这个问题的一种方法来自于具有相当简单的直觉的物理学。想象一下盒子里面有一个球。球将滚到最低的地方，重力将与盒子侧面可以强加于球的力量平衡。简而言之，目标函数（即重力）的梯度将被约束函数的梯度所抵消（由于墙壁 “向后推”，球需要留在盒子内）。请注意，一些限制可能不活跃：球未触及的墙壁将无法对球施加任何力量。 

跳过 * 拉格朗日 * $L$ 的推导，上述推理可以通过以下鞍点优化问题来表达： 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

这里的变量 $\alpha_i$ ($i=1,\ldots,n$) 是所谓的 * 拉格朗日乘数 *，可以确保约束条件得到正确实施。选择它们足够大，以确保 $c_i(\mathbf{x}) \leq 0$ 适用于所有 $i$。例如，对于任何 $\mathbf{x}$，其中 $c_i(\mathbf{x}) < 0$ 当然，我们最终会选择 $\alpha_i = 0$。此外，这是一个鞍点优化问题，人们希望与所有 $\alpha_i$ 相比 * 最大化 * $L$，同时 * 最小化 * 相对于 $\mathbf{x}$。有丰富的文献解释了如何到达函数 $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$。就我们的目的而言，只要知道 $L$ 的鞍点就足够了，是最好地解决最初的约束优化问题的地方。 

### 处罚

至少 * 近似 * 满足受限优化问题的一种方法是调整拉格朗日 $L$。我们只需在目标功能 $f(x)$ 中添加 $\alpha_i c_i(\mathbf{x})$，而不是满足 $c_i(\mathbf{x}) \leq 0$。这可以确保限制条件不会受到太严重的违反。 

事实上，我们一直在使用这个技巧。考虑 :numref:`sec_weight_decay` 中的体重衰减。在其中，我们将 $\frac{\lambda}{2} \|\mathbf{w}\|^2$ 添加到目标函数中，以确保 $\mathbf{w}$ 不会变得太大。从受限的优化角度来看，我们可以看出，这将确保部分半径为 $r$ 的 $\|\mathbf{w}\|^2 - r^2 \leq 0$。调整 $\lambda$ 的值可以让我们改变 $\mathbf{w}$ 的大小。 

一般来说，增加罚款是确保大致满足约束条件的好方法。实际上，事实证明，这比确切的满意度要强劲得多。此外，对于非凸问题，许多在凸情况下使精确方法如此吸引力的属性（例如，最佳性）已不再存在。 

### 预测

满足制约因素的另一种策略是预测。再次，我们之前遇到了它们，例如，在 :numref:`sec_rnn_scratch` 中处理渐变剪切时。在那里，我们确保渐变的长度以 $\theta$ 为限。 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

事实证明，这是在 $\theta$ 半径 $\theta$ 的球上的 * 投影 * $\mathbf{g}$。更一般地说，在凸集上的投影 $\mathcal{X}$ 被定义为 

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

这是 $\mathcal{X}$ 至 $\mathbf{x}$ 中的最接近点。  

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

预测的数学定义可能听起来有点抽象。:numref:`fig_projections` 更清楚地解释了这一点。里面我们有两个凸集，一个圆圈和一个钻石。在预测期间，两个集中的点（黄色）保持不变。两个集之外的积分（黑色）将投影到集合内部的点（红色），这些点与原始积分（黑色）相近。虽然对于 $L_2$ 球而言，这使方向保持不变，但一般情况并不一定如此，就像钻石的情况可以看出的那样。 

凸投影的用途之一是计算稀疏权重矢量。在这种情况下，我们将重量矢量投射到 $L_1$ 球上，这是 :numref:`fig_projections` 中钻石表壳的通用版本。 

## 摘要

在深度学习的背景下，凸函数的主要目的是激励优化算法并帮助我们详细了解它们。在下面我们将看到如何相应地得出梯度下降和随机梯度下降。 

* 凸集的交叉点是凸的。工会不是。
* 对凸函数的期望不低于期望的凸函数（詹森的不平等性）。
* 如果而且只有当其 Hessian（第二衍生物的矩阵）为正半定值时，两次可分化的函数才是凸的。
* 凸约束可以通过拉格朗日添加。在实践中，我们可以简单地将它们添加到客观功能中加一个惩罚。
* 投影映射到最接近原始点的凸集中的点。

## 练习

1. 假设我们想通过绘制集合中点之间的所有线并检查线是否包含来验证集合的凸度。
    1. 证明只检查边界上的点就足够了。
    1. 证明只检查集合的顶点就足够了。
1. 用 $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ 表示使用 $p$ 标准的半径 $r$ 的球。证明所有 $p \geq 1$ 对于所有 $p \geq 1$ 来说，$\mathcal{B}_p[r]$ 都是凸的。
1. 给定凸函数 $f$ 和 $g$，表明 $\mathrm{max}(f, g)$ 也是凸的。证明 $\mathrm{min}(f, g)$ 不是凸起的。
1. 证明 softmax 函数的标准化是凸的。更具体地说，证明了 $f(x) = \log \sum_i \exp(x_i)$ 的凸度。
1. 证明线性子空间，即 $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$，是凸集。
1. 证明，对于 $\mathbf{b} = \mathbf{0}$ 的线性子空间，对于某些矩阵 $\mathbf{M}$，投影 $\mathrm{Proj}_\mathcal{X}$ 可以写为 $\mathbf{M} \mathbf{x}$。
1. 显示，对于两次可分的凸函数 $f$，我们可以为大约 $\xi \in [0, \epsilon]$ 写 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$。
1. 给定向量 $\mathbf{w} \in \mathbb{R}^d$ 和 $\|\mathbf{w}\|_1 > 1$，计算 $L_1$ 单位球上的投影。
    1. 作为中间步骤，写出受惩的目标 $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$ 并计算给定 $\lambda > 0$ 的解决方案。
    1. 你能找到 $\lambda$ 的 “正确” 值没有经过很多试验和错误吗？
1. 鉴于凸集 $\mathcal{X}$ 和两个向量 $\mathbf{x}$ 和 $\mathbf{y}$，证明预测永远不会增加距离，即 $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$。

[Discussions](https://discuss.d2l.ai/t/350)
