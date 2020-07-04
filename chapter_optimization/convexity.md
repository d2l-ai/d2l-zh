# 凸性
:label:`sec_convexity`

凸性在优化算法设计中起着至关重要的作用。这很大程度上是因为在这种环境下分析和测试算法要容易得多。换句话说，如果算法即使在凸性的集合上也表现不佳，我们就不应该希望看到好的结果。此外，尽管深度学习中的优化问题通常是非凸问题，但它们往往在局部极小点附近表现出凸问题的一些性质。这可能导致令人兴奋的新的优化变体，如:cite:`Izmailov.Podoprikhin.Garipov.ea.2018`。

## 基础

让我们从基础开始。

## 集合

集合是凸性的基础。 简而言之，如果对于$X$中的任何$a, b \in X$，连接$a$和$b$的线段也位于$X$中，则向量空间中的集合$X$是凸的。 用数学术语来说，这意味着对于所有$\lambda \in [0, 1]$中，我们有

$$\lambda \cdot a + (1-\lambda) \cdot b \in X \text{ whenever } a, b \in X.$$

这听起来有点抽象。考虑 :numref:`fig_pacman`所示的图片。第一个集合不是凸的，因为它不包含线段。另外两组则没有这样的问题。

![Three shapes, the left one is nonconvex, the others are convex](../img/pacman.svg)
:label:`fig_pacman`

除非您可以对它们进行某些操作，否则它们本身的定义并不是特别有用。 在这种情况下，我们可以查看并集和交集，如：numref：`fig_convex_intersect`中所示。 假设$X$和$Y$是凸集。 那么$X \cap Y$也是凸的。 要看到这一点，请考虑$a, b \in X \cap Y$中的任何$a, b \in X \cap Y$。 由于$X$和$Y$是凸的，因此连接$a$和$b$的线段包含在$X$和$Y$中。鉴于此，它们也需要包含在$X \cap Y$中，从而证明我们的第一个定理。

![The intersection between two convex sets is convex](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

我们可以毫不费力地增强此结果：给定凸集$X_i$，它们的交点$\cap_{i} X_i$是凸的。
若要查看相反的结果，请考虑两个不相交的集合$X \cap Y = \emptyset$。 现在选择$a \in X$和$b \in Y$。 连接$a$和$b$的：numref：`fig_nonconvex`中的线段需要包含不在$X$或$Y$中的部分，因为我们假定$X \cap Y = \emptyset$。 因此线段也不在$X \cup Y$中，因此证明了凸集的并集一般不必是凸的。

![The union of two convex sets need not be convex](../img/nonconvex.svg)
:label:`fig_nonconvex`

通常，深度学习中的问题是在凸域上定义的。 例如，$\mathbb{R}^d$是一个凸集（毕竟$\mathbb{R}^d$中任意两点之间的线仍然在$\mathb{R}^d$中。在某些情况下，我们使用有界长度的变量，例如由$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_2 \leq r\}$。

### 函数

既然我们有了凸集，我们可以引入凸函数$f$。 给定凸集$X$，在其上定义了一个函数$f$：如果对于所有$x, x' \in X$，$f: X \to \mathbb{R}$是凸的，且所有$\lambda \in [0, 1]$我们有

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

为了说明这一点，让我们绘制一些函数，并检查哪些满足需求。我们需要导入一些库。

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

让我们定义几个函数，凸函数和非凸函数。

```{.python .input}
def f(x):
    return 0.5 * x**2  # Convex

def g(x):
    return np.cos(np.pi * x)  # Nonconvex

def h(x):
    return np.exp(0.5 * x)  # Convex

x, segment = np.arange(-2, 2, 0.01), np.array([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))

for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

正如预期的那样，余弦函数是非凸的，而抛物线和指数函数是非凸的。注意，要使条件有意义，$X$是凸集这一要求是必要的。否则， $f(\lambda x + (1-\lambda) x')$ 的结果可能没有很好的定义。凸函数有许多令人满意的性质。

## Jensen不等式

Jensen不等式是最有用的工具之一。它相当于凸性定义的一般化：

$$\begin{aligned}
    \sum_i \alpha_i f(x_i) & \geq f\left(\sum_i \alpha_i x_i\right)
    \text{ and }
    E_x[f(x)] & \geq f\left(E_x[x]\right),
\end{aligned}$$

其中，$\alpha_i$是非负的实数，例如 $\sum_i \alpha_i = 1$ 。换句话说，凸函数的期望比期望的凸函数要大。为了证明第一个不等式，我们将凸性的定义反复地应用到和式中的一项上。期望可以通过取有限段的极限来证明。

Jensen不等式的常见应用之一是关于部分观察到的随机变量的对数似然性。 也就是说，我们使用

$$E_{y \sim P(y)}[-\log P(x \mid y)] \geq -\log P(x).$$

这是因为$\int P(y) P(x \mid y) dy = P(x)$。
这用在变分方法中。 在这里，$y$通常是不可观察的随机变量，$P(y)$是对其分布方式的最佳猜测，$Px$是集成了$y$的分布。 例如，在聚类中，$y$可能是聚类标签，而$P(x \mid y)$是应用聚类标签时的生成模型。

## 性质

凸函数有一些有用的性质。我们如下这么描述它们。

### 没有局部最小值

特别是，凸函数没有局部极小值。让我们假设相反的情况，并证明它是错的。如果 x∈X 是局部最小值，则存在x的某个邻域，其中 $f(x)$ 是其最小值。由于 x 只是一个局部极小值，所以必须有另一个x′∈X ，使 $f(x') < f(x)$ 。但是，通过凸性，整个直线 在 $\lambda \in [0, 1)$ 范围的 $\lambda x + (1-\lambda) x'$ 上的函数值必须少于 $f(x')$，如:

$$f(x) > \lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

这与 $f(x)$ 是局部最小值的假设相矛盾。例如，函数 $f(x) = (x+1) (x-1)^2$ 在 $x=1$ 时具有局部最小值。然而，它不是一个全局最小值。

```{.python .input}
def f(x):
    return (x-1)**2 * (x+1)

d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸函数没有局部极小值这一事实是非常方便的。这意味着，如果我们最小化函数，就不会陷入困境。但是请注意，这并不意味着不能有一个以上的全局最小值，或者甚至可能存在一个全局最小值。例如，函数 $f(x) = \mathrm{max}(|x|-1, 0)$ 在区间 $[-1, 1]$ 内达到其最小值。相反，函数 $f(x) = \exp(x)$ 在 $\mathbb{R}$ 上没有得到最小值。对于 $x \to -\infty$  ，它趋近于 $0$，但是没有 x 使得 $f(x) = 0$。

### 凸函数与凸集

凸函数将凸集定义为 *Below-sets*。 它们被定义为

$$S_b := \{x | x \in X \text{ and } f(x) \leq b\}.$$

这样的集合是凸的。 让我们迅速证明这一点。 请记住，对于任何$x, x' \in S_b$，我们需要证明$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$，只要$\lambda \in [0, 1]$$\lambda \in [0, 1]$。但这直接来自于凸度的定义，因为$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$。

看看下面的函数$f(x, y) = 0.5 x^2 + \cos(2 \pi y)$。 显然是非凸的。 水平集相应地是非凸的。 实际上，它们通常由不相交的集合组成。

```{.python .input}
x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101),
                   indexing='ij')

z = x**2 + 0.5 * np.cos(2 * np.pi * y)

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

### 导数与凸性

只要存在函数的二阶导数，就很容易检查凸性。 我们需要做的只是检查$\partial_x^x f(x) \succeq 0$，即其所有特征值是否均为非负值。 例如，函数$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2_2$是凸的，因为$\partial_{\mathbf{x}}^2 f = \mathbf{1}$，即其导数为单位矩阵。

首先要意识到的是，我们只需要证明一维函数的此属性。 毕竟，总的来说，我们总是可以定义一些函数$ g（z）= f（\ mathbf {x} + z \ cdot \ mathbf {v}）$。 此函数具有一阶和二阶导数$ g'=（\ partial _ {\ mathbf {x}} f）^ \ top \ mathbf {v} $和$ g''= \ mathbf {v} ^ \ top（\ partial  ^ 2 _ {\ mathbf {x}} f）\ mathbf {v} $。 特别是，每当$f$的Hessian为正半定数时，即每当其所有特征值均大于零时，对于所有$ \ mathbf {v} $，$ g''\ geq 0 $。 因此回到标量情况。

为了看到凸函数$f''(x) \geq 0$，我们使用了

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

由于二阶导数由有限差分的极限给出，因此

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

为了看到相反的结论，我们使用了一个事实，即$f'' \geq 0$表示$f'$是单调递增的函数。 令$a < x < b$为$\mathbb{R}$中的三分。 我们用平均值定理来表达

$$\begin{aligned}
f(x) - f(a) & = (x-a) f'(\alpha) \text{ for some } \alpha \in [a, x] \text{ and } \\
f(b) - f(x) & = (b-x) f'(\beta) \text{ for some } \beta \in [x, b].
\end{aligned}$$

通过单调性 $f'(\beta) \geq f'(\alpha)$, 因此

$$\begin{aligned}
    f(b) - f(a) & = f(b) - f(x) + f(x) - f(a) \\
    & = (b-x) f'(\beta) + (x-a) f'(\alpha) \\
    & \geq (b-a) f'(\alpha).
\end{aligned}$$

根据几何形状，得出$ f(x)$低于连接$ f(a)$和$ f(b)$的线，从而证明了凸性。 我们忽略了更正式的推导，以支持下面的图表。

```{.python .input}
def f(x):
    return 0.5 * x**2

x = np.arange(-2, 2, 0.01)
axb, ab = np.array([-1.5, -0.5, 1]), np.array([-1.5, 1])

d2l.set_figsize()
d2l.plot([x, axb, ab], [f(x) for x in [x, axb, ab]], 'x', 'f(x)')
d2l.annotate('a', (-1.5, f(-1.5)), (-1.5, 1.5))
d2l.annotate('b', (1, f(1)), (1, 1.5))
d2l.annotate('x', (-0.5, f(-0.5)), (-1.5, f(-0.5)))
```

## 约束

凸优化的一个很好的特性是它使我们能够有效地处理约束。 也就是说，它使我们能够解决以下形式的问题：

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

这里 $f$ 是目标，函数 $c_i$ 是约束函数。 要了解这是什么，请考虑 $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$的情况。 在这种情况下，参数 $\mathbf{x}$。 如果第二个约束是 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$，则这对应于所有位于半空间的 $\mathbf{x}$。 同时满足两个约束就等于选择一个球的切片作为约束集。

### 拉格朗日函数

至少近似满足约束优化问题的一种方法是采用拉格朗日函数。与其满足 ci(x) 0 ，我们只是简单地将侧重的c_i(x)≤0侧重的  α_ic_i(x) 添加到目标函数 f(x) 中。这确保了约束不会被严重违反

一般来说，解决一个约束优化问题是困难的。解决这个问题的一种方法来自物理学，有一种相当简单的直觉。想象一个球在一个盒子里。这个球会滚到最低的地方，重力会被盒子的两边施加在球上的力平衡掉。简而言之，目标函数(即。将被约束函数的梯度所抵消(由于墙的“向后推”的特性，需要保持在盒子内)。请注意，任何无效的约束（即，球不接触墙壁）都将无法对球施加任何力。

跳过拉格朗日函数 L 的推导(详见Boyd和Vandenberghe的书:cite:`Boyd.Vandenberghe.2004`)，上述推理可以通过以下鞍点优化问题来表达:

$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

这里的变量 $\alpha_i$ 就是所谓的拉格朗日乘数，它保证了正确强制了约束。它们被选择的足够大，以确保所有 $i$ 的 $c_i(\mathbf{x}) \leq 0$。例如，对于 $c_i(\mathbf{x}) < 0$ 自然地 $c_i(\mathbf{x}) < 0$ 的任何 $\mathbf{x}$ ，我们最终会选择的是 $\alpha_i = 0$。此外，这是一个鞍点优化问题，在这个问题中，我们希望*最大化*关于外置的LL，同时*最小化*关于 x 的 $\mathbf{x}$ 。有大量的文献解释如何得到函数 $L(\mathbf{x}, \alpha)$。为了我们的目的，知道 $L$ 的鞍点就足够了。

### 惩罚项

至少近似地满足约束优化问题的一种方法是适配拉格朗日函数 $L$ 。 除了满足 $c_i(\mathbf{x}) \leq 0$之外，我们只需将 $\alpha_i c_i(\mathbf{x})$ 添加到目标函数 $f(x)$。这样可以确保不会严重违反约束。

实际上，我们一直在用这个技巧。考虑:numref:`sec_weight_decay`中的重量衰减。在该方法中，我们在 $\frac{\lambda}{2} \|\mathbf{w}\|^2$  到目标函数中，以确保 $\mathbf{w}$ 不会变得太大。使用约束优化的观点，我们可以看到，这将确保 $\|\mathbf{w}\|^2 - r^2 \leq 0$ 为一些半径 $r$。通过调整 $\lambda$ 的值，我们可以改变 $\mathbf{w}$ 的大小。

通常，添加惩罚是确保近似约束满足的一种好方法。在实践中，这比精确的满足更加可靠。此外，对于非凸问题，许多使精确方法在凸情况下如此吸引人的特性(例如最优性)不再成立。

### 投影

满足约束条件的另一种策略是投影。同样，我们以前遇到过它们，比如在:numref:`sec_rnn_scratch`中处理渐变剪辑时。在这里，我们确保了梯度的长度以 $c$ 为界通过

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$

这就是$g$在半径为$c$的球上的投影。更一般地，(凸)集$X$上的投影定义为

$$\mathrm{Proj}_X(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in X} \|\mathbf{x} - \mathbf{x}'\|_2.$$

因此，它是$X$到$\mathbf{x}$的最近点。这听起来有点抽象。:numref:``fig_projections``更清楚地说明了这一点。其中有两个凸集，一个圆和一个菱形。集合内的点(黄色)保持不变。集合外的点(黑色)映射到集合内最近的点(红色)。而对于 $\ell_2$ 球，这使方向不变，这不必是一般情况下，可以看到菱形的情况。

![Convex Projections](../img/projections.svg)
:label:`fig_projections`

凸投影的一个用途是计算稀疏权值向量。在这种情况下，我们将 $\mathbf{w}$ 投影到一个 $\ell_1$ 的球上(后者是上图中菱形的广义版本)。

## 小结

在深度学习的背景下，凸函数的主要目的是驱使优化算法，并帮助我们了解它们的细节。下面我们将看到如何相应地推导梯度下降和随机梯度下降。

* 凸集的交集是凸的。 并集不是。
* 凸函数的期望大于期望的凸函数（詹森不等式）。
* 当且仅当其二阶导数在整个过程中仅具有非负特征值时，二次可微函数才是凸函数。
* 可以通过拉格朗日函数添加凸约束。 实际上，只需将它们加到目标函数中即可。
* 投影映射到（凸）集中最接近原始点的点。

## 练习

1. 假设我们要通过绘制集​​合中点之间的所有线并检查是否包含这些线来验证集合的凸性。
    * 证明：仅检查边界上的点就足够了。
    * 证明：仅检查集合的顶点就足够了。
2. 用 $B_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ 使用 $p$-norm半径为 $r$的球。 证明 $B_p [r]$ 对于所有 $p \ geq 1$ 是凸的。
3. 给定凸函数 $f$ 和 $g$ 显示 $\mathrm{max}(f, g)$ 也是凸的。 证明：对于所有 $p \geq 1$， $B_p[r]$是凸的。
4. 证明softmax函数的归一化是凸的。 更具体地说，证明：$\mathrm{min}(f, g)$的凸性。
5. 证明线性子空间是凸集，如 $X = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$。
6. 证明在$ \ mathbf {b} = 0 $的线性子空间的情况下，对于某些矩阵$，投影$ \ mathrm {Proj} _X $可以写成$ \ mathbf {M} \ mathbf {x} $  \ mathbf {M} $。
7. 证明对于凸二次微分函数 $f$，对于某些$\xi \in [0, \epsilon]$，我们可以写 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$。
8. 给定向量 $\mathbf{w} \in \mathbb{R}^d$ ，其中 $\|\mathbf{w}\|_1 > 1$计算 $\ell_1$ 单位球上的投影。
    * 作为中间步骤，写出惩罚项 $\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$ ，并对于一个已给定的$\lambda > 0$计算出结果。
    * 您是否可以找到 $\lambda$ 的“正确”值，而无需经过反复试验？
9. 给定凸集t $X$ 和两个向量 $\mathbf{x}$ 和 $\mathbf{y}$证明投影永远不会增加距离，即 $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_X(\mathbf{x}) - \mathrm{Proj}_X(\mathbf{y})\|$。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/350)
:end_tab:
