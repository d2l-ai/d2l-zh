# 微积分
:label:`sec_calculus`

找到多边形的区域一直保持神秘，直到至少 2500 年前，当时古希腊人将多边形划分成三角形，并将他们的面积加在一起。要找到区域的弯曲形状，如一个圆，古希腊人刻在这样的形状的多边形。如:numrefe: `fig_circle_area` 所示，一个具有较多边长相等长度的刻字面更好地接近圆。此过程也称为 * 用尽方法 *。

![Find the area of a circle with the method of exhaustion.](../img/polygon_circle.svg)
:label:`fig_circle_area`

事实上，用尽的方法是 * 积分计算 *（将在：numrefe：`sec_integral_calculus` 中描述）的起源地。2000 多年后，微积分的另一个分支，* 微分微分 *，被发明。在微分微分微分最关键的应用中，优化问题考虑如何做一些 * 最好的 *。正如在：numref:`subsec_norms_and_objectives` 中讨论的那样，这些问题在深度学习中无处不在。

在深度学习中，我们 * 训练 * 模型，连续更新它们，以便随着他们看到越来越多的数据而变得越来越好。通常，获得更好意味着最小化 * 损失函数 *，这是一个回答 “我们的模型如何 * 坏 *” 这个问题的得分？这个问题比它看起来更微妙。最终，我们真正关心的是生成一个在我们从未见过的数据上表现良好的模型。但是我们只能将模型拟合到我们可以实际看到的数据。因此，我们可以将拟合模型的任务分解为两个关键问题：i) * 优化 *：将模型拟合到观测数据的过程；ii) * 泛化 *：指导如何生成超出精确数据集的模型的数学原理和从业人员的智慧点用于训练他们。

为了帮助您了解后面的章节中的优化问题和方法，这里我们给出了一个非常简短的关于深度学习中常用的差分微积分的入门介绍。

## 衍生品和差异化

我们首先解决衍生物的计算问题，这是几乎所有深度学习优化算法的关键一步。在深度学习中，我们通常选择相对于模型参数可区分的损失函数。简而言之，这意味着，对于每个参数，我们可以确定损失增加或减少的速度，如果我们增加 * 或 * 减少 * 该参数无限小的数量。

假设我们有一个函数 $f: \mathbb{R} \rightarrow \mathbb{R}$，其输入和输出都是标量。$f$ 的 * 衍生物 * 被定义为

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$
:eqlabel:`eq_derivative`

如果存在此限制。如果存在 $f'(a)$，则说 $f$ 是可差分的 *。如果 $f$ 在一个时间间隔的每个数字都是可区分的，则此函数在此时间间隔中是可区分的。我们可以将导数 $f'(x)$ 解释为相对于 $x$ 的 * 瞬时 * 变化率。所谓的瞬时变化速率是基于 $x$ 中的变化 $h$，该变化接近 $0$。

为了说明衍生品，让我们试验一个样本。定义。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

通过设定 $x=1$ 并让 $h$ 接近 $0$，计算结果表明：$\frac{f(x+h) - f(x)}{h}$ 的数值结果接近 $0$。虽然这个实验不是一个数学证明，但我们稍后会看到，导数 $u'$ 是 $2$ 当 $x=1$。

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

让我们熟悉一些衍生品的等效符号。给定 $y = f(x)$，其中 $x$ 和 $y$ 分别是函数 $f$ 的独立变量和相关变量。以下表达式是等效的：

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

其中符号 $\frac{d}{dx}$ 和 $D$ 是表示 * 差分运算符 * 的操作。我们可以使用以下规则来区分常见函数：

* ($C$ 是一个常数)，
* $Dx^n = nx^{n-1}$（电源规则 *，$n$ 是任何实数），
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

为了区分由几个简单的函数（如上述常见函数）形成的函数，以下规则对我们来说非常方便。假设函数 $f$ 和 $g$ 都是可区分的，$C$ 是常量，我们有 * 常量多规则 *

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

* 总和规则 *

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

* 产品规则 *

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

和 * 商规则 *

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

现在我们可以应用上述几个规则来查找 $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$。因此，通过设置 $x = 1$，我们有 $u' = 2$：这得到了我们早期在本节中的实验的支持，其中数值结果接近 $2$。这种导数也是切线的斜率曲线 $u = f(x)$ 当 $x = 1$.

为了可视化对衍生工具的这种解释，我们将使用 `matplotlib`，这是 Python 中流行的绘图库。要配置 `matplotlib` 生成的数字的属性，我们需要定义一些函数。在下面，`use_svg_display` 函数指定了 `matplotlib` 软件包来输出 svg 数字以获得更清晰的图像。

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

我们定义 `set_figsize` 函数来指定图形大小。请注意，这里我们直接使用 `d2l.plt`，因为导入语句 `from matplotlib import pyplot as plt` 已标记为保存在序言中的 `d2l` 软件包中。

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

下面的 `set_axes` 函数设置由 `matplotlib` 产生的数字轴的属性。

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

通过这三个用于图形配置的函数，我们定义了 `plot` 函数，以简洁地绘制多条曲线，因为我们需要在整个书中可视化许多曲线。

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

现在我们可以在 $x=1$ 处绘制函数 $u = f(x)$ 及其切线 $y = 2x - 3$，其中系数 $2$ 是切线的斜率。

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 部分衍生品

到目前为止，我们已经处理了只有一个变量函数的区分。在深度学习中，函数通常取决于 * 多个 * 变量。因此，我们需要将差异化的想法扩展到这些 * 多变量 * 函数。

让 $y = f(x_1, x_2, \ldots, x_n)$ 成为一个具有 $n$ 变量的函数。对于其参数 $i^\mathrm{th}$ 参数 $y$ 的部分导数 * 是

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

为了计算 $\frac{\partial y}{\partial x_i}$，我们可以简单地将 $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ 作为常数处理，并计算 $y$ 相对于 $x_i$ 的导数。对于部分衍生品的符号，以下是等价的：

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 渐变

我们可以将多变量函数的部分导数与其所有变量连结起来，以获得函数的 * 梯度 * 向量。假设函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 的输入是一个 $n$ 维向量 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$，并且输出是一个标量。函数 $f(\mathbf{x})$ 相对于 $\mathbf{x}$ 的梯度是部分导数 $n$ 的向量：

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

其中 $\nabla_{\mathbf{x}} f(\mathbf{x})$ 通常在没有模棱两可的情况下被 $\nabla f(\mathbf{x})$ 取代。

让 $\mathbf{x}$ 成为一个 $n$ 维向量，在区分多变量函数时通常使用以下规则：

* 对于所有这些国家来说，
* 对于所有这些国家来说，
* 对于所有这些国家来说，
* $\ nabla_ {\ 麦斯比夫 {x}}\ |\ 麦斯比夫 {x}\ |^2 =\ 纳布拉 _ {\ 麦斯比夫 {x}}\ 麦斯比夫 {x} ^\ 顶部\ 麦斯比夫 {x} = 2\ 麦斯比夫 {x} $。

同样，对于任何矩阵 $\mathbf{X}$，我们都有 $\ nabla_ {\\ 数字符 {X}}\ |\ 数字符 {X}\ |_F^2 = 2\ 数字符 {X} $。正如我们稍后将看到的，渐变对于在深度学习中设计优化算法非常有用。

## 链规则

然而，这种渐变可能很难找到。这是因为深度学习中的多元函数通常是 * 合成 *，所以我们可能不会应用任何上述规则来区分这些函数。幸运的是，* 链规则 * 使我们能够区分复合函数。

让我们首先考虑一个变量的函数。假设函数 $y=f(u)$ 和 $u=g(x)$ 都是可区分的，那么链规则指出

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

现在让我们把注意力转向一个更一般的场景，其中函数具有任意数量的变量。假设可差分函数 $y$ 具有变量 $u_1, u_2, \ldots, u_m$，其中每个可差分函数 $u_i$ 都有变量 $x_1, x_2, \ldots, x_n$。请注意，$y$ 是一个函数。然后链规则给出

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

对于任何 $i = 1, 2, \ldots, n$ 号决定。

## 摘要

* 微分微分和积分微分是微分的两个分支，其中前者可以应用于深度学习中无处不在的优化问题。
* 衍生物可以被解释为函数相对于其变量的瞬时变化率。它也是相切线与函数曲线的斜率。
* 梯度是一个向量，其分量是多变量函数相对于其所有变量的部分导数。
* 链规则使我们能够区分复合函数。

## 练习

1. 在 $x = 1$ 时绘制函数及其切线。
1. 找到功能的梯度 $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. 函数 $f (\ 麦斯比夫 {x}) =\ |\ 麦斯比夫 {x}\ |_2$ 的梯度是什么？
1. 你可以写出一个连锁规则的情况下，其中 $u = f(x, y, z)$ 和 $x = x(a, b)$，$y = y(a, b)$ 和 $z = z(a, b)$？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
