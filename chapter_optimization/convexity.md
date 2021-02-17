# 凸性
:label:`sec_convexity`

*凸性* （convexity）在优化算法的设计中起到至关重要的作用。这主要是由于在这种情况下对算法进行分析和测试要容易得多。换言之，如果该算法甚至在凸性条件设定下的效果很差，通常我们很难在其他条件下看到好的结果。
此外，即使深度学习中的优化问题通常是非凸的，它们也经常在局部极小值附近表现出一些凸性。 这可能会产生令人兴奋的新的优化变体，例如 :cite: `Izmailov.Podoprikhin.Garipov.ea.2018` 。 

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

在进行凸分析之前，我们需要定义*凸集*（convex sets）和*凸函数*（convex functions）。 它们产生普遍用于机器学习的数学工具。 

### 凸集

*凸集*（convex set）是凸性的基础。简单地说，如果对于任何 $a，b \in \mathcal{X}$ ，连接 $a$ 和 $b$ 的线段也位于 $\mathcal{X}$ 中，则向量空间中的一个集合 $\mathcal{X}$ 是*凸*（convex）的。 在数学术语上，这意味着对于所有 $\lambda \in [0, 1]$ ，我们得到

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$


这听起来有点抽象。思考 :numref:`fig_pacman` 。第一组不是凸的，因为存在线段不包含在集合的内部。另外两组没有这样的问题。 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

定义本身并不是特别有用，除非您可以使用定义做一些事情。在这种情况下，我们可以看一下交集 :numref:`fig_convex_intersect` 。假设 $\mathcal{X}$ 和 $\mathcal{Y}$ 是凸集。 那么 $\mathcal {X} \cap \mathcal{Y}$ 也是凸集的。 看到这一点，考虑任意 $a, b \in \mathcal{X} \cap \mathcal{Y}$ 。因为 $\mathcal{X}$ 和 $\mathcal{Y}$ 是凸集，所以连接 $a$ 和 $b$ 的线段包含在 $\mathcal{X}$ 和 $\mathcal{Y}$ 中。 鉴于此，它们也需要包含在 $\mathcal{X} \cap \mathcal{Y}$ 中，从而证明我们的定理。 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我们可以毫不费力地进一步得到这样的结果：给定凸集 $\mathcal{X}_i$ ，它们的交集 $\cap_{i} \mathcal{X}_i$ 是凸的。看到相反的情况是不正确的，考虑两个不相交的集合 $\mathcal{X} \cap \mathcal{Y} = \emptyset$ 。 现在，取 $a \in \mathcal{X}$ 和 $b \in \mathcal{Y}$ 。 因为我们假设 $\mathcal{X} \cap \mathcal{Y} = \emptyset$ ，在 :numref:`fig_nonconvex` 中连接 $a$ 和 $b$ 的线段需要包含一部分既不在 $\mathcal{X}$ 也不在 $\mathcal{Y}$ 中。 因此线段也不在 $\mathcal{X} \cup \mathcal{Y}$ 中，因此证明了凸集的并集不需要是凸的。 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

通常，深度学习中的问题是在凸集上定义的。 例如， $\mathbb{R}^d$ ，即实数的 $d$-维向量的集合是凸集（毕竟 $\mathbb{R}^d$ 中任意两点之间的线存在 $\mathbb{R}^d$ ）中。 在某些情况下，我们使用有界长度的变量，例如球的半径定义为 $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \| \mathbf{x} \| \leq r\}$ 。 

### 凸函数

现在我们有了凸集，我们可以引入*凸函数*（convex function） $f$ 。给定一个凸集 $\mathcal{X}$ ，如果对于所有 $x, x' \in \mathcal{X}$ 和所有 $\lambda \in [0, 1]$ ，一个函数 $f: \mathcal{X} \to \mathbb{R}$ 是凸的，我们可以得到

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

为了说明这一点，让我们绘制一些函数并检查哪些函数满足要求。下面我们定义一些函数，包括凸函数和非凸函数。 


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

不出所料，余弦函数为*非凸*（nonconvex）的，而抛物线函数和指数函数为凸的。 请注意， 为使该条件有意义， $\mathcal{X}$ 是凸集的要求是必要的。 否则可能无法很好地界定 $f(\lambda x + (1-\lambda) x')$ 的结果。 

### 詹森不等式

给定一个凸函数 $f$ ，最有用的数学工具之一就是*詹森不等式*（Jensen's inequality）。它是凸性定义的一种推广：

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right) \text{ and } E_X[f(X)] \geq f\left(E_X[X]\right),$$

其中 $\alpha_i$ 是非负实数，因此 $\sum_i \alpha_i = 1$ 且 $X$ 是随机变量。换句话说，凸函数的期望不小于期望的凸函数，其中后者通常是一个更简单的表达式。为了证明第一个不等式，我们多次将凸性的定义应用于一次求和中的一项。

詹森不等式的一个常见应用是用一个较简单的表达式约束一个较复杂的表达式。例如，它可以应用于部分观察到的随机变量的对数似然。也就是说，我们用

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

由 $\int P(Y) P(X \mid Y) dY = P(X)$ 。这可以用在变分法中。在这里， $Y$ 是典型的未观察到的随机变量， $P(Y)$ 是它可能如何分布的最佳猜测， $P(X)$ 是将 $Y$ 积分后的分布。例如，在聚类中 $Y$ 可能是簇标签，而在应用簇标签时， $P(X \mid Y)$ 是生成模型。
 
## 性质

凸函数有一些有用的性质。我们这样描述它们。

### 局部极小值是全局极小值

首先凸函数的局部极小值也是全局极小值。我们作出相反的假设并证明它是错误的。如果 $ x^{\ast} \in \mathcal{X}$ 是一个局部最小值，使得有一个很小的正值 $p$ ，使得 $x \in \mathcal{X}$ 满足 $0 < |x - x^{\ast}| \leq p$ 有 $f(x^{\ast}) < f(x)$ 。假设存在 $x' \in \mathcal{X}$ ，其中 $f(x') < f(x^{\ast})$ 。根据凸性的性质，

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &< f(x^{\ast}) \\
\end{aligned}$$

存在 $\lambda \in [0, 1)$ ， $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ 为例，因此 $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$ 。然而，因为 $f(\lambda x^{\ast} + (1-\lambda) x') < f(x^{\ast})$ ，这违反了我们的局部最小值前提声明。因此，对于 $f(x') < f(x^{\ast})$ 不存在 $x' \in \mathcal{X}$ 。局部最小值 $x^{\ast}$ 也是全局最小值。

例如，对于函数 $f(x) = (x-1)^2$ ，有一个局部最小值 $x=1$ ， 它也是全局最小值。

```{.python .input}
#@tab all
f = lambda x: (x-1)**2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函数的局部极小值同时也是全局极小值这一性质是很方便的。这意味着，如果我们最小化函数，我们就不会“卡住”。但是，请注意，这并不意味着不能有一个以上的全局最小值，甚至可能存在一个。例如，函数 $f(x) = \mathrm{max}(|x|-1, 0)$ 在 $[-1,1]$ 区间上取得其最小值。相反，函数 $f(x) = \exp(x)$ 在 $\mathbb{R}$ 上没有取得最小值。对于 $x \to -\infty$ ，它趋近于 $0$ ，但是没有 $f(x) = 0$ 的 $x$ 。

### 凸函数与凸集

凸函数将凸集定义为*降维集合*（below-sets）。它们定义为：

$$S_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}.$$

这样的集合是凸的。让我们快速证明一下。记住，对于任何 $x, x' \in \mathcal{S}_b$ ，我们需要证明 $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ 中，只要 $\lambda \in [0, 1]$ 。但是这直接遵循了凸性的定义，因为 $f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$。

看看下面的函数 $f(x, y) = 0.5 x^2 + \cos(2 \pi y)$ 。它显然是非凸的。水平集相应地是非凸的。 实际上，它们通常由不相交的集合组成。

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 + 0.5 * d2l.cos(2 * np.pi * y)
# Plot the 3D surface
d2l.set_figsize((6, 4))
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.contour(x, y, z, offset=-1)
ax.set_zlim(-1, 1.5)
# Adjust labels
for func in [d2l.plt.xticks, d2l.plt.yticks, ax.set_zticks]:
    func([-1, 0, 1])
```

### 导数和凸性

当一个函数的二阶导数存在时，很容易检查它的凸性。我们需要做的就是检查 $\partial_x^2 f(x) \succeq 0$ ，即它的所有特征值是否都是非负的。例如，函数 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ 是凸的，因为 $\partial_{\mathbf{x}}^2 f = \mathbf{1}$ ， 即其导数是单位矩阵。

首先要意识到的是我们只需要证明一维函数的这个性质。毕竟，我们通常总是可以定义一些函数 $g(z) = f(\mathbf{x} + z \cdot \mathbf{v})$ 。这个函数的一阶导数和二阶导数 $g' = (\partial_{\mathbf{x}} f)^\top \mathbf{v}$ 和 $g'' = \mathbf{v}^\top (\partial^2_{\mathbf{x}} f) \mathbf{v}$ 。特别地，无论 $f$ 的海赛矩阵是否为正半定时，即无论它的所有特征值是否都大于等于零时，对所有 $\mathbf{v}$ 有 $g'' \geq 0$ 。回到标量情况。

为了看到 $f''(x) \geq 0$ 对于凸函数我们使用的事实

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

因为二阶导数是由有限差分的极限给出的，所以遵循

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

为了证明相反的情况是正确的，我们使用 $f'' \geq 0$ 这一事实表明 $f'$ 是一个单调递增函数。设 $a < x < b$ 是 $\mathbb{R}$ 中的三个点。我们用中值定理来表示

$$\begin{aligned}
f(x) - f(a) & = (x-a) f'(\alpha) \text{ for some } \alpha \in [a, x] \text{ and } \\
f(b) - f(x) & = (b-x) f'(\beta) \text{ for some } \beta \in [x, b].
\end{aligned}$$

通过单调性 $f'(\beta) \geq f'(\alpha)$ ，因此

$$\begin{aligned}
    f(b) - f(a) & = f(b) - f(x) + f(x) - f(a) \\
    & = (b-x) f'(\beta) + (x-a) f'(\alpha) \\
    & \geq (b-a) f'(\alpha).
\end{aligned}$$

从几何上看， $f(x)$ 低于 $f(a)$ 和 $f(b)$ 的连接线，从而证明了它的凸性。我们用下面的图表忽略了更正式的推导。 

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2
x = d2l.arange(-2, 2, 0.01)
axb, ab = d2l.tensor([-1.5, -0.5, 1]), d2l.tensor([-1.5, 1])
d2l.set_figsize()
d2l.plot([x, axb, ab], [f(x) for x in [x, axb, ab]], 'x', 'f(x)')
d2l.annotate('a', (-1.5, f(-1.5)), (-1.5, 1.5))
d2l.annotate('b', (1, f(1)), (1, 1.5))
d2l.annotate('x', (-0.5, f(-0.5)), (-1.5, f(-0.5)))
```


## 约束

凸优化的一个很好的特性是能够让我们有效地处理约束。也就是说，它使我们能够解决以下形式的问题：

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

这里 $f$ 是目标函数， $c_i$ 是约束函数。为了解这是什么，考虑这样的情况 $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$ 。在这种情况下，参数 $\mathbf{x}$ 被限制为单位球。如果第二个约束是 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$ ，那么这对应于半空间上所有的 $\mathbf{x}$ 。同时满足这两个约束等于选择一个球的切片作为约束集。


### 拉格朗日函数

通常，求解一个有约束的优化问题是困难的。解决这个问题的一种方法来自物理中相当简单的直觉。想象一个球在一个盒子里。球会滚到最低的地方，重力将与盒子两侧对球施加的力平衡。简而言之，目标函数(即重力)的梯度将被约束函数的梯度所抵消（由于墙壁的“推回”作用，需要保持在盒子内）。请注意，任何不起作用的约束（即球不接触壁）都将无法对球施加任何力。 

跳过拉格朗日函数 $L$ 的推导（详见Boyd和Vandenberghe的著作 :cite:`Boyd.Vandenberghe.2004` ），上述推理可以通过以下鞍点优化问题来表示：

$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

这里的变量 $\alpha_i$ 是所谓的*拉格朗日乘数*（Lagrange Multipliers），它确保约束被正确地执行。选择它们的大小足以确保所有 $i$ 的 $c_i(\mathbf{x}) \leq 0$ 。例如，自然地对于 $c_i(\mathbf{x}) < 0$ 中任意 $\mathbf{x}$ ，我们最终会选择 $\alpha_i = 0$ 。 此外，这是一个*鞍点*（saddlepoint）优化问题，在这个问题中，我们想要使 $L$ 相对于 $\alpha$ *最大化*（maximize），同时使它相对于 $\mathbf{x}$ *最小化*（minimize）。有大量的文献解释如何得出函数 $L(\mathbf{x}, \alpha)$ 。 对于我们的目标，知道 $L$ 的鞍点是原始约束优化问题的最优解就足够了。

### 惩罚

一种至少近似地满足约束优化问题的方法是采用拉格朗日函数 $L$ 。 除了满足 $c_i(\mathbf{x}) \leq 0$ 之外，我们只需将 $\alpha_i c_i(\mathbf{x})$ 添加到目标函数 $f(x)$ 。这样可以确保不会严重违反约束。 

事实上，我们一直在使用这个技巧。考虑权重衰减 :numref:`sec_weight_decay` 。在目标函数中加入 $\frac{\lambda}{2} |\mathbf{w}|^2$ ，以确保 $\mathbf{w}$ 不会增长太大。使用约束优化的观点，我们可以看到，对于若干半径 $r$ ，这将确保 $|\mathbf{w}|^2 - r^2 \leq 0$ 。通过调整 $\lambda$ 的值，我们可以改变 $\mathbf{w}$ 的大小。

通常，添加惩罚是确保近似满足约束的一种好方法。在实践中，这被证明比精确的满意度更可靠。此外，对于非凸问题，许多使精确方法在凸情况下如此吸引人的性质（例如，最优性）不再成立。

### 投影

满足约束条件的另一种策略是投影。同样，我们之前也遇到过，例如在处理梯度裁剪 :numref:`sec_rnn_scratch` 时。在这里，我们确保梯度的长度以 $c$ 为界限，通过

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$

这就是 $g$ 在半径为 $c$ 的球上的*投影*（projection）。更一般地，在凸集 $\mathcal{X}$ 上的投影被定义为

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|_2.$$

因此，它是 $\mathcal{X}$ 中离 $\mathbf{X}$ 最近的点。这听起来有点抽象，:numref:`fig_projections` 在某种程度上解释得更清楚。其中有两个凸集，一个圆和一个菱形。集合内的点（黄色）保持不变。集合外的点（黑色）被映射到集合内最近的点（红色）。而对于 $L_2$ 的球，这将使方向保持不变，但一般情况下并不需要如此，就像在菱形的例子中所看到的那样。

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

凸投影的用途之一是计算稀疏权重向量。在本例中，我们将 $\mathbf{w}$ 投射到 $L_1$ 球上（后者是上图中菱形的一般化版本）。

## 小结

在深度学习的背景下，凸函数的主要目的是促进优化算法，并帮助我们详细了解它们。下面我们将看到梯度下降法和随机梯度下降法是如何相应推导出来的。

* 凸集的交点是凸的，并集不是。
* 凸函数的期望大于期望的凸函数（詹森不等式）。
* 一个二次可微函数是凸函数当且仅当它的二阶导数始终只有非负特征值。
* 凸约束可以通过拉格朗日函数来添加。在实践中，只需在目标函数中加上一个惩罚就可以了。
* 投影映射到凸集中最接近原始点的点。

## 练习题 

1. 假设我们想要通过绘制集合内点之间的所有直线并检查这些直线是否包含来验证集合的凸性。
    i. 证明只检查边界上的点是充分的。
    ii. 证明只检查集合的顶点是充分的。
2. 用 $p$ -范数表示半径为 $r$ 的球，证明 $\mathcal{B}_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ ， $\mathcal{B}_p[r]$ 对于所有 $p \geq 1$ 是凸的。

3. 已知凸函数 $f$ 和 $g$ 表明 $\mathrm{max}(f, g)$ 也是凸函数。证明 $\mathrm{min}(f, g)$ 不是凸的。

4. 证明Softmax函数的归一化是凸的。更具体地证明 $f(x) = \log \sum_i \exp(x_i)$ 的凸性。

5. 证明线性子空间是凸集，即 $\mathcal{X} = {\mathbf{X} | \mathbf{W} \mathbf{X} = \mathbf{b}}$ 。

6. 证明在线性子空间 $\mathbf{b} = 0$ 的情况下，对于矩阵 $\mathbf{M}$ 的投影 $\mathrm {Proj} \mathcal{X}$ 可以写成$\mathbf{M} \mathbf{X}$。

7. 证明对于凸二次可微函数 $f$ ，对于 $\xi \in [0, \epsilon]$ ，我们可以写成 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ 。

8. 给定一个向量 $\mathbf{w} \in \mathbb{R}^d$ 与 $|\mathbf{w}| 1 > 1$ 计算在 $\ell_1$ 单位球上的投影。
    i. 作为中间步骤，写出惩罚目标 $|\mathbf{w} - \mathbf{w}'|_2^2 + \lambda |\mathbf{w}'|_1$ ，计算给定 $\lambda > 0$ 的解。
    ii. 你能找到 $\lambda$ 的“正确”值而无需大量的试验和错误吗？

9. 给定一个凸集 $\mathcal{X}$ 和两个向量 $\mathbf{X}$ 和 $\mathbf{y}$ 证明了投影不会增加距离，即$\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$。


[讨论区](https://discuss.d2l.ai/t/350)
