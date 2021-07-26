# 优化和深度学习

在本节中，我们将讨论优化与深度学习之间的关系以及在深度学习中使用优化的挑战。对于深度学习问题，我们通常会先定义 * 损失函数 *。一旦我们有了损失函数，我们就可以使用优化算法来尽量减少损失。在优化中，损失函数通常被称为优化问题的 * 目标函数 *。按照传统和惯则，大多数优化算法都关注的是 * 最小化 *。如果我们需要最大限度地实现目标，那么有一个简单的解决方案：只需翻转目标上的标志即可。 

## 优化的目标

尽管优化提供了一种最大限度地减少深度学习损失功能的方法，但实质上，优化和深度学习的目标是根本不同的。前者主要关注的是尽量减少一个目标，而鉴于数据量有限，后者则关注寻找合适的模型。在 :numref:`sec_model_selection` 中，我们详细讨论了这两个目标之间的区别。例如，训练错误和泛化错误通常不同：由于优化算法的客观函数通常是基于训练数据集的损失函数，因此优化的目标是减少训练错误。但是，深度学习（或更广义地说，统计推断）的目标是减少概括错误。为了完成后者，除了使用优化算法来减少训练错误之外，我们还需要注意过度拟合。

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

为了说明上述不同的目标，让我们考虑经验风险和风险。如 :numref:`subsec_empirical-risk-and-risk` 所述，经验风险是训练数据集的平均损失，而风险则是整个数据群的预期损失。下面我们定义了两个函数：风险函数 `f` 和经验风险函数 `g`。假设我们只有有限量的训练数据。因此，这里的 `g` 不如 `f` 平滑。

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

下图说明，训练数据集的最低经验风险可能与最低风险（概括错误）不同。

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## 深度学习中的优化挑战

在本章中，我们将特别关注优化算法在最小化目标函数方面的性能，而不是模型的泛化错误。在 :numref:`sec_linear_regression` 中，我们区分了优化问题中的分析解和数值解。在深度学习中，大多数客观的功能都很复杂，没有分析解决方案。相反，我们必须使用数值优化算法。本章中的优化算法都属于此类别。 

深度学习优化存在许多挑战。其中一些最令人恼人的是局部最小值、鞍点和消失的渐变。让我们来看看它们。 

### 本地迷你

对于任何客观函数 $f(x)$，如果 $f(x)$ 的值 $f(x)$ 在 $x$ 附近的任何其他点小于 $f(x)$ 的值，那么 $f(x)$ 在 $x$ 附近的任何其他点的值小于 $f(x)$，那么 $f(x)$ 可能是局部最低值。如果 $f(x)$ 的值为 $f(x)$，为整个域的目标函数的最小值，那么 $f(x)$ 是全局最小值。 

例如，给定函数 

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

我们可以接近该函数的局部最小值和全局最小值。

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

深度学习模型的客观功能通常有许多局部最佳值。当优化问题的数值解近于局部最佳值时，最终迭代获得的数值解可能只能最小化目标函数 * 本地 *，而不是随着目标函数解的梯度接近或变为零而不是 * 全局 *。只有一定程度的噪音可能会使参数从当地的最低值中排除出来。事实上，这是迷你批随机梯度下降的有益特性之一，在这种情况下，迷你匹配的渐变的自然变化能够从局部最小值中移除参数。 

### 鞍积分

除了局部最小值之外，鞍点也是梯度消失的另一个原因。* 鞍点 * 是指函数的所有渐变都消失但既不是全局也不是局部最小值的任何位置。考虑这个函数 $f(x) = x^3$。它的第一个和第二个衍生品消失了 $x=0$。这时优化可能会停顿，尽管它不是最低限度。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

如下例所示，较高尺寸的鞍点甚至更加阴险。考虑这个函数 $f(x, y) = x^2 - y^2$。它的鞍点为 $(0, 0)$。这是相对于 $y$ 的最高值，最低为 $x$。此外，它 * 看起来像马鞍，这就是这个数学属性的名字的地方。

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

我们假设函数的输入是 $k$ 维矢量，其输出是标量，因此其黑森州矩阵将有 $k$ 特征值（参考 [online appendix on eigendecompositions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html)）。函数的解决方案可以是局部最小值、局部最大值或函数梯度为零的位置的鞍点： 

* 当函数在零梯度位置处的 Hessian 矩阵的特征值全部为正值时，我们有该函数的局部最小值。
* 当函数在零梯度位置处的 Hessian 矩阵的特征值全部为负值时，我们有该函数的局部最大值。
* 当函数在零梯度位置处的 Hessian 矩阵的特征值为负值和正值时，我们对函数有一个鞍点。

对于高维度问题，至少 * 部分 * 特征值为负的可能性相当高。这使得马鞍点比本地最小值更有可能。介绍凸体时，我们将在下一节中讨论这种情况的一些例外情况。简而言之，凸函数是黑森人的特征值永远不是负值的函数。但是，可悲的是，大多数深度学习问题并不属于这个类别。尽管如此，这是研究优化算法的好工具。 

### 消失渐变

可能遇到的最阴险的问题是渐变消失。回想一下我们在 :numref:`subsec_activation-functions` 中常用的激活函数及其衍生品。例如，假设我们想尽量减少函数 $f(x) = \tanh(x)$，然后我们恰好从 $x = 4$ 开始。正如我们所看到的那样，$f$ 的梯度接近零。更具体地说，$f'(x) = 1 - \tanh^2(x)$，因此是 $f'(4) = 0.0013$。因此，在我们取得进展之前，优化将会停滞很长一段时间。事实证明，这是在引入 RELU 激活功能之前训练深度学习模型相当棘手的原因之一。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

正如我们所看到的那样，深度学习的优化充满挑战。幸运的是，有一系列强大的算法表现良好，即使对于初学者也很容易使用。此外，没有必要找到 * 最佳解决方案。本地最佳甚至其近似的解决方案仍然非常有用。 

## 摘要

* 尽量减少训练错误并不能 * 保证我们找到最佳的参数集来最大限度地减少泛化错误。
* 优化问题可能有许多局部最低限度。
* 问题可能有更多的马鞍点，因为通常问题不是凸起的。
* 渐变消失可能会导致优化停滞。重新参数化问题通常会有所帮助。对参数进行良好的初始化也可能是有益的。

## 练习

1. 考虑一个简单的 MLP，隐藏层中有一个隐藏层（例如，）$d$ 维度和单个输出。表明对于任何本地最低限度来说，至少有 $d！$ 行为相同的等效解决方案。
1. 假设我们有一个对称随机矩阵 $\mathbf{M}$，其中条目 $M_{ij} = M_{ji}$ 各自从某种概率分布 $p_{ij}$ 中提取。此外，假设 $p_{ij}(x) = p_{ij}(-x)$，即分布是对称的（详情请参见 :cite:`Wigner.1958`）。
    1. 证明特征值的分布也是对称的。也就是说，对于任何特征向量 $\mathbf{v}$，关联的特征值 $\lambda$ 满足 $P(\lambda > 0) = P(\lambda < 0)$ 的概率为 $P(\lambda > 0) = P(\lambda < 0)$。
    1. 为什么以上 * 不 * 暗示 $P(\lambda > 0) = 0.5$？
1. 你能想到深度学习优化还涉及哪些其他挑战？
1. 假设你想在（真实的）鞍上平衡一个（真实的）球。
    1. 为什么这很难？
    1. 你能也利用这种效果进行优化算法吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
