# 渐变下降
:label:`sec_gd`

在本节中，我们将介绍 * 梯度下降 * 的基本概念。尽管很少在深度学习中直接使用，但了解梯度下降是了解随机梯度下降算法的关键。例如，由于学习率过高，优化问题可能会分歧。这种现象已经可以从梯度下降中看出来。同样，预处理是梯度下降的常见技术，可以继续使用更高级的算法。让我们从一个简单的特殊情况开始。 

## 一维梯度下降

一个维度的梯度下降是一个很好的例子，可以解释为什么梯度下降算法可能会降低目标函数的值。考虑一些持续差异的实值函数 $f: \mathbb{R} \rightarrow \mathbb{R}$。使用泰勒扩张我们获得 

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

也就是说，一阶近似 $f(x+\epsilon)$ 是由函数值 $f(x)$ 和第一个导数 $f'(x)$ 给出的，为 $x$。假设对于小 $\epsilon$ 而言，朝负梯度方向移动将减少 $f$ 并非不合理。为了简单起见，我们选择固定的步长 $\eta > 0$ 然后选择 $\epsilon = -\eta f'(x)$。把这个插入上面的泰勒扩张我们得到了 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

如果衍生品 $f'(x) \neq 0$ 没有消失，我们会从 $\eta f'^2(x)>0$ 开始取得进展。此外，我们总是可以选择足够小的 $\eta$ 以使高阶条款变得无关紧要。因此我们到达 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

这意味着，如果我们使用 

$$x \leftarrow x - \eta f'(x)$$

为了迭代 $x$，函数 $f(x)$ 的值可能会下降。因此，在梯度下降中，我们首先选择初始值 $x$ 和一个常数 $\eta > 0$，然后使用它们连续迭代 $x$ 直到达到停止条件，例如，当梯度 $|f'(x)|$ 的幅度足够小或迭代次数已达到一定价值。 

为简单起见，我们选择目标函数 $f(x)=x^2$ 来说明如何实现梯度下降。尽管我们知道 $x=0$ 是最小化 $f(x)$ 的解决方案，但我们仍然使用这个简单的函数来观察 $x$ 如何变化。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

接下来，我们使用 $x=10$ 作为初始值，并假设 $\eta=0.2$。使用梯度下降对 $x$ 进行 10 次迭代，我们可以看到，最终，$x$ 的值接近最佳解决方案。

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

如下所示，优化超过 $x$ 的进展情况。

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 学习率
:label:`subsec_gd-learningrate`

学习率 $\eta$ 可以由算法设计师设置。如果我们使用的学习率太低，将导致 $x$ 的更新速度非常缓慢，需要更多的迭代才能获得更好的解决方案。要显示在这种情况下会发生什么，请考虑 $\eta = 0.05$ 的同一优化问题的进展情况。正如我们所看到的那样，即使在 10 个步骤之后，我们还远离最佳解决方案。

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

相反，如果我们使用过高的学习率，$\left|\eta f'(x)\right|$ 对于一阶泰勒扩张公式来说可能太大了。也就是说，:eqref:`gd-taylor-2` 中的术语 $\mathcal{O}(\eta^2 f'^2(x))$ 可能会变得重要。在这种情况下，我们无法保证 $x$ 的迭代能够降低 $f(x)$ 的值。例如，当我们将学习率设置为 $\eta=1.1$ 时，$x$ 超出了最佳解决方案 $x=0$ 并逐渐发散。

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 本地迷你

为了说明非凸函数会发生什么情况，请考虑 $f(x) = x \cdot \cos(cx)$ 对于某个常数 $c$ 的情况。这个函数有无限多个本地最小值。根据我们对学习率的选择以及问题的条件有多好，我们最终可能会找到许多解决方案之一。下面的例子说明了（不现实的）高学习率如何导致较差的本地最低水平。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 多变量渐变下降

现在我们对单变量案有了更好的直觉，让我们来考虑 $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$ 的情况。也就是说，目标函数 $f: \mathbb{R}^d \to \mathbb{R}$ 将向量映射为标量。相应地，它的渐变也是多变量的。它是由 $d$ 部分衍生品组成的向量： 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

梯度中的每个部分衍生元素 $\partial f(\mathbf{x})/\partial x_i$ 表示相对于输入 $x_i$ 的 $f$ 的变化率为 $\mathbf{x}$，为 $\mathbf{x}$。和以前一样，在单变量的情况下，我们可以使用相应的泰勒近似值作为多变量函数来了解我们应该做什么。特别是，我们有 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

换句话说，$\boldsymbol{\epsilon}$ 中的二阶术语，最陡的下降方向是由负梯度 $-\nabla f(\mathbf{x})$ 给出的。选择合适的学习率 $\eta > 0$ 可以产生原型的梯度下降算法： 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

要了解算法在实践中的行为，让我们构建一个目标函数 $f(\mathbf{x})=x_1^2+2x_2^2$，其中二维矢量 $\mathbf{x} = [x_1, x_2]^\top$ 作为输入，标量作为输出。梯度由 $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ 给出。我们将通过从初始位置 $[-5, -2]$ 的梯度下降观察 $\mathbf{x}$ 的轨迹。  

首先，我们还需要两个辅助函数。第一个使用更新函数并将其应用于初始值 20 次。第二个助手可视化了 $\mathbf{x}$ 的轨迹。

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

接下来，我们观察学习率 $\eta = 0.1$ 的优化变量 $\mathbf{x}$ 的轨迹。我们可以看到，经过 20 个步骤，$\mathbf{x}$ 的价值接近 $[0, 0]$ 的最低水平。进展情况相当不错，尽管相当缓慢。

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 自适应方法

正如我们在 :numref:`subsec_gd-learningrate` 中看到的那样，获得 $\eta$ “恰到好处” 的学习率是棘手的。如果我们选择太小，我们就没有什么进展。如果我们选择太大，解决方案就会振荡，在最坏的情况下，它甚至可能会分歧。如果我们可以自动确定 $\eta$ 或者根本不必选择学习率，该怎么办？二阶方法不仅看目标函数的价值和梯度，而且还查看其 *curvature* 在这种情况下可以有所帮助。虽然这些方法由于计算成本不能直接应用于深度学习，但它们为如何设计模仿下面概述的算法的许多理想属性的高级优化算法提供了有用的直觉。 

### 牛顿的方法

回顾泰勒对某些职能 $f: \mathbb{R}^d \rightarrow \mathbb{R}$ 的扩张，在第一个任期之后没有必要停止。事实上，我们可以把它写成 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

为了避免繁琐的符号，我们将 $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$ 定义为 $f$ 的黑森语，这是一个 $d \times d$ 矩阵。对于小型 $d$ 和简单的问题，$\mathbf{H}$ 很容易计算。另一方面，对于深度神经网络而言，由于存储 $\mathcal{O}(d^2)$ 条条目的成本，$\mathbf{H}$ 可能太大。此外，通过反向传播进行计算可能太昂贵。现在让我们忽略这些考虑因素，看看我们会得到什么算法。 

毕竟，最低的 $f$ 满足 $\nabla f = 0$。遵循 :numref:`subsec_calculus-grad` 中的微积分规则，采取 :eqref:`gd-hot-taylor` 的衍生品，对 $\boldsymbol{\epsilon}$ 的衍生品，忽略了我们得出的高阶条款 

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

也就是说，作为优化问题的一部分，我们需要反转黑森州 $\mathbf{H}$。 

作为一个简单的例子，对于 $f(x) = \frac{1}{2} x^2$，我们有 $\nabla f(x) = x$ 和 $\mathbf{H} = 1$。因此，对于任何 $x$，我们获得了 $\epsilon = -x$。换句话说，*Single* 步骤足以完美收敛而无需进行任何调整！唉，我们在这里有点幸运：泰勒的扩张自 $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$ 以来就是准确的。  

让我们看看其他问题会发生什么。给定一些常数 $c$ 的凸双余弦函数 $f(x) = \cosh(cx)$，我们可以看到，经过几次迭代后达到了 $x=0$ 的全局最低值。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

现在让我们考虑一个 * 非凸 * 函数，例如 $f(x) = x \cos(c x)$ 对于某些常数 $c$。毕竟，请注意，在牛顿的方法中，我们最终被黑森人划分。这意味着，如果第二个衍生品为 * 负 * 我们可能会走向 * 增加 * 值 $f$ 的方向。这是算法的一个致命缺陷。让我们看看实际中会发生什么。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

这出现了极大的错误。我们怎么能修复它？一种方法是通过取代其绝对值来 “修复” 黑森人。另一种策略是恢复学习率。这似乎破坏了目的，但并非完全。拥有二阶信息可以让我们在曲率较大时保持谨慎态度，并在客观功能更平坦的情况下采取更长的步骤。比如 $\eta = 0.5$，让我们来看看这是如何在稍低的学习率下工作的。正如我们所看到的那样，我们有一个非常有效的算法。

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 收敛性分析

我们只分析牛顿方法的收敛率为一些凸和三倍可差分目标函数 $f$，其中第二个导数为非零值，即 $f'' > 0$。多变量证明是下面一维论点的直接延伸，省略了，因为它在直觉方面没有太大帮助。 

用 $x^{(k)}$ 表示 $k^\mathrm{th}$ 迭代时 $x$ 的值，让 $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ 成为 $k^\mathrm{th}$ 迭代时与最优性的距离。通过泰勒扩张我们有条件 $f'(x^*) = 0$ 可以写成 

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

这支持了大约 $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$。将上述扩张除以 $f''(x^{(k)})$ 收益率 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

回想一下，我们有更新 $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$。插入这个更新方程式，并且考虑双方的绝对价值，我们有 

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

因此，每当我们处于 $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ 的边界区域时，我们都会出现二次递减的误差  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

顺便说一句，优化研究人员称之为 * 线性 * 收敛，而像 $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ 这样的条件将被称为 * 恒定 * 收敛率。请注意，此分析附带了一些注意事项。首先，我们实际上没有太多的保证，我们何时能够到达迅速趋同的区域。相反，我们只知道一旦我们达到这一目标，趋同将非常快。其次，这项分析要求 $f$ 在高阶衍生品之前表现良好。归结为确保 $f$ 在如何改变其价值方面没有任何 “令人惊讶的” 属性。 

### 预处理

毫不奇怪，计算和存储完整的 Hessian 是非常昂贵的。因此，寻找替代办法是可取的。改善问题的一种方法是 * 先决条件 *。它避免了完整计算黑森语，但只计算 * 对角 * 条目。这会导致更新表单的算法 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

尽管这不如完整的牛顿方法那么好，但它仍然比不使用它好得多。要了解为什么这可能是个好主意，请考虑一个变量表示以毫米为单位的高度，另一个变量表示高度（以千米为单位）。假设两种自然尺度都以米为单位，那么我们在参数化方面存在严重的不匹配。幸运的是，使用预处理消除了这一点使用梯度下降进行有效预处理等于为每个变量（矢量 $\mathbf{x}$ 的坐标）选择不同的学习率。正如我们稍后将看到的那样，预处理推动了随机梯度下降优化算法的一些创新。  

### 使用线搜索进行渐变下降

梯度下降的关键问题之一是，我们可能会超过目标或进展不足。问题的一个简单解决方法是将行搜索结合梯度下降结合使用。也就是说，我们使用 $\nabla f(\mathbf{x})$ 给出的方向，然后对学习率 $\eta$ 最小化 $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$ 进行二进制搜索。 

该算法迅速收敛（有关分析和证明，请参见 :cite:`Boyd.Vandenberghe.2004`）。但是，为了深度学习的目的，这并不是那么可行，因为行搜索的每一步都要求我们评估整个数据集的目标函数。这太昂贵了，难以完成。 

## 摘要

* 学习率很重要。太大而且我们分歧，太小了，我们没有取得进展。
* 渐变下降可能会陷入局部最小值。
* 在高维度上，调整学习率很复杂。
* 预处理可以帮助调整比例。
* 牛顿的方法一旦开始在凸出的问题中正常工作，就会快得多。
* 小心使用牛顿的方法而不对非凸问题进行任何调整。

## 练习

1. 尝试不同的学习率和客观函数来实现梯度下降。
1. 在 $[a, b]$ 的时间间隔内实施行搜索以最大限度地减少凸函数。
    1. 你是否需要衍生品进行二进制搜索，即决定是选择 $[a, (a+b)/2]$ 还是 $[(a+b)/2, b]$。
    1. 算法的收敛速度有多快？
    1. 实施算法并将其应用到最小化 $\log (\exp(x) + \exp(-2x -3))$。
1. 设计 $\mathbb{R}^2$ 上定义的客观函数，其中梯度下降速度非常缓慢。提示：不同的缩放不同的坐标。
1. 使用预处理实现牛顿方法的轻量级版本：
    1. 使用对角线 Hessian 作为预调器。
    1. 使用该值的绝对值，而不是实际（可能有符号）值。
    1. 将此应用于上述问题。
1. 将上述算法应用于许多客观函数（凸与否）。如果你将坐标旋转 $45$ 度会发生什么？

[Discussions](https://discuss.d2l.ai/t/351)
