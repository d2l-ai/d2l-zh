# 线性回归
:label:`sec_linear_regression`

*回归 * 是指一组用于建模的方法
一个或多个独立变量与依赖变量之间的关系。在自然科学和社会科学中，回归的目的最常见是
*表征 * 输入和输出之间的关系。
另一方面，机器学习最常关心的是 * 预测 *。

每当我们想要预测数值时，回归问题就会弹出。常见的例子包括预测价格（住房、股票等）、预测住宿时间（住院患者）、需求预测（零售销售）等等。并非每个预测问题都是一个典型的回归问题。在随后的章节中，我们将介绍分类问题，其中的目标是预测一组类别中的成员资格。

## 线性回归的基本元素

*线性回归 * 可能是最简单的
和最流行的标准工具回归。可以追溯到 19 世纪的黎明，线性回归源于几个简单的假设。首先，我们假设独立变量 $\mathbf{x}$ 和相关变量 $y$ 之间的关系是线性的，也就是说，$y$ 可以表示为 $\mathbf{x}$ 中元素的加权和，考虑到观测值的一些噪声。其次，我们假设任何噪声都表现良好（遵循高斯分布）。

为了激励这种方法，让我们从一个运行的样本开始。假设我们希望根据房屋面积（平方英尺）和年龄（以年为单位）估算房屋价格（以美元为单位）。为了实际匹配预测房价的模型，我们需要掌握一个销售数据集，其中包括我们知道每个家庭的销售价格、面积和年龄。在机器学习术语中，数据集称为 * 训练数据集 * 或 * 训练集 *，每行（这里是与一次销售相对应的数据）称为 * 示例 *（或 * 数据点 *、* 数据实例 *、* 样本 *）。我们试图预测的东西（价格）被称为 * 标签 *（或 * 目标 *）。预测所依据的自变量（年龄和面积）称为 * 要素 *（或 * 协变量 *）。

通常，我们将使用 $n$ 来表示数据集中的示例数量。我们通过 $i$ 对数据点进行索引，表示每个输入为 $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，相应的标签为 $y^{(i)}$。

### 线性模型
:label:`subsec_linear_model`

线性假设只是说，目标（价格）可以表示为要素（面积和年龄）的加权总和：

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

在 :eqref:`eq_price-area` 中，$w_{\mathrm{area}}$ 和 $w_{\mathrm{age}}$ 被称为 * 重量 *，$b$ 被称为 * 偏移 * 或 * 截获 *）。权重决定了每个要特征对我们的预测的影响，而偏差只是说，当所有要素都取值 0 时，预测价格应该采取什么值。即使我们永远不会看到任何家庭的零面积，或者是正是零年年龄，我们仍然需要偏差，否则我们将限制我们的模型的表现力。严格来说，:eqref:`eq_price-area` 是输入要素的 * 仿射变换 *，其特点是通过加权总和对要素进行 * 线性变换 *，并通过增加的偏差结合 * 平移 *。

给定一个数据集，我们的目标是选择权重 $\mathbf{w}$ 和偏差 $b$，以便平均而言，根据我们的模型做出的预测最适合数据中观察到的真实价格。输出预测由输入要素的仿射变换确定的模型为 * 线性模型 *，其中仿射变换由所选权重和偏差指定。

在通常专注于只有少数特征的数据集的学科中，显式表达像这样长形式的模型很常见。在机器学习中，我们通常使用高维数据集，因此使用线性代数表法更加方便。当我们的输入包含 $d$ 功能时，我们将我们的预测 $\hat{y}$（一般来说，“帽子” 符号表示估计值）表示为

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

将所有要素收集到矢量 $\mathbf{x} \in \mathbb{R}^d$ 中，并将所有权重收集到矢量 $\mathbf{w} \in \mathbb{R}^d$ 中，我们可以使用点积紧凑地表达我们的模型：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

在 :eqref:`eq_linreg-y` 中，矢量 $\mathbf{x}$ 对应于单个数据点的要素。我们通常会发现，通过 * 设计矩阵 * $\mathbf{X} \in \mathbb{R}^{n \times d}$ 引用我们整个数据集 $n$ 示例的功能很方便。此处，$\mathbf{X}$ 包含每个样本一行，每个特征一列。

对于特征 $\mathbf{X}$ 的集合，预测值 $\hat{\mathbf{y}} \in \mathbb{R}^n$ 可以通过矩阵矢量积来表示：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

在求和期间应用广播 (见 :numref:`subsec_broadcasting`).给定训练数据集 $\mathbf{X}$ 和相应的（已知）标签 $\mathbf{y}$ 的特征，线性回归的目标是找到权重向量 $\mathbf{w}$ 和偏差项 $b$，给出了从 $\mathbf{X}$ 相同分布取样的新数据点的特征，新数据点的标签将（在期望值）以最小的误差进行预测。

即使我们相信 $\mathbf{x}$ 给定的 $y$ 预测的最佳模型是线性的，我们也不希望找到一个 $n$ 示例的真实世界数据集，其中 $y^{(i)}$ 完全等于 $\mathbf{w}^\top \mathbf{x}^{(i)}+b$。样本，无论我们使用什么仪器来观察特征 $\mathbf{X}$ 和标签 $\mathbf{y}$ 都可能会出现少量的测量误差。因此，即使我们确信底层关系是线性的，我们也会加入一个噪声项来解决这些错误。

在我们开始寻找最好的 * 参数 *（或 * 模型参数 *）$\mathbf{w}$ 和 $b$ 之前，我们还需要两件事：(i) 某些给定模型的质量测量；(ii) 更新模型以提高其质量的程序。

### 损失函数

在我们开始考虑如何 * 拟合 * 我们的模型之前，我们需要确定一个 * 适合性 * 的度量。* 损耗函数 * 量化目标的 * 实际 * 和 * 预测 * 值之间的距离。亏损通常为非负数，其中较小的值越好，完美的预测会导致损失 0。回归问题中最常用的损失函数是平方误差。当我们对样本 $i$ 的预测为 $\hat{y}^{(i)}$，相应的真实标签为 $y^{(i)}$ 时，平方误差由以下公式给出：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

常数 $\frac{1}{2}$ 没有真正的区别，但会证明在公证上方便，当我们采取损失的衍生物时取消。由于训练数据集是给我们的，因此超出了我们的控制范围，经验误差只是模型参数的函数。为了使事情更具体，请考虑下面的样本，我们为一维情况绘制回归问题，如 :numref:`fig_fit_linreg` 所示。

![Fit data with a linear model.](../img/fit_linreg.svg)
:label:`fig_fit_linreg`

请注意，由于二次依赖，估计值 $\hat{y}^{(i)}$ 和观测值 $y^{(i)}$ 之间的巨大差异导致损失的更大贡献。为了测量 $n$ 示例的整个数据集上的模型质量，我们只需平均（或等价的总和）训练集上的损失。

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

在训练模型时，我们希望找到能够最大限度地减少所有训练示例中总损耗的参数 ($\mathbf{w}^*, b^*$)：

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 分析解决方案

线性回归恰好是一个异常简单的优化问题。与我们将在本书中遇到的大多数其他模型不同，线性回归可以通过应用一个简单的公式来解决。首先，我们可以将偏差 $b$ 归入参数 $\mathbf{w}$ 中，方法是在设计矩阵中附加一列，由所有矩阵组成。然后我们的预测问题是最小化 $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。损耗表面上只有一个临界点，它对应于整个域上的最小损耗。以 $\mathbf{w}$ 损失的衍生物并将其设置为零，产生分析（封闭形式）解决方案：

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

虽然像线性回归这样的简单问题可能会接受分析解决方案，但你不应该习惯这样的好运。虽然分析解决方案允许进行良好的数学分析，但对分析解决方案的要求非常严格，以至于排除所有深度学习。

### 迷你批次随机梯度下降

即使在我们无法分析模型的情况下，事实证明，我们仍然可以在实践中有效地训练模型。此外，对于许多任务，那些难以优化的模型变得更好，以至于弄清楚如何训练它们最终是非常值得的麻烦。

优化几乎所有深度学习模型的关键技术，我们将在本书中提到这一技术，包括通过更新参数以递增降低损失函数的方向来迭代减少误差。此算法称为 * 渐变降序 *。

梯度下降的最天真的应用是采用损失函数的导数，这是在数据集中每个样本中计算的损失的平均值。在实践中，这可能非常缓慢：我们必须在进行单次更新之前传递整个数据集。因此，每次我们需要计算更新时，我们经常会解决采样随机的小批例子，这是一个名为 * minibatch batch 随机梯度降序 * 的变量。

在每次迭代中，我们首先随机抽样一个由固定数量的训练示例组成的小批 $\mathcal{B}$。然后，我们根据模型参数计算微型批次平均损耗的导数（梯度）。最后，我们将梯度乘以预定的正值 $\eta$，并从当前参数值中减去生成的项。

我们可以用数学方式表示更新如下（$\partial$ 表示偏导数）：

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

总而言之，算法的步骤如下：(i) 我们初始化模型参数的值，通常是随机的；(ii) 我们从数据中迭代抽样随机小批，按负梯度的方向更新参数。对于二次损耗和仿射变换，我们可以明确写出来，如下所示：

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

请注意，$\mathbf{w}$ 和 $\mathbf{x}$ 都是向量。在这里，更优雅的矢量符号使得数学比以系数表达事物更具可读性，比如 $w_1, w_2, \ldots, w_d$。集合的基数 $|\mathcal{B}|$ 表示每个小批次（* 批次大小 *）中的示例数，$\eta$ 表示 * 学习率 *。我们强调，批量大小和学习率的值是手动预先指定的，通常不是通过模型培训学习的。这些可调整但不能在训练循环中更新的参数称为 * 超参数 *。
*超参数调整 * 是选择超参数的过程，
，通常要求我们根据训练循环的结果进行调整，这些结果是在单独的 * 验证数据集 *（或 * 验证集 *）中评估的。

在训练了一些预先确定的迭代次数（或者直到满足某些其他停止条件）之后，我们记录估计的模型参数，表示 $\hat{\mathbf{w}}, \hat{b}$。请注意，即使我们的函数是真正的线性和无噪音，这些参数也不会是损失的确切最小化因素，因为尽管算法向最小化缓慢收敛，但它无法在有限的步骤中完全实现它。

线性回归恰好是一个学习问题，整个域只有一个最小值。但是，对于更复杂的模型（如深度网络），损耗表面包含许多最小值。幸运的是，由于尚未完全理解的原因，深度学习从业者很少难找到能够最大限度地减少训练设置 * 损失的参数。更艰巨的任务是找到可以实现我们之前从未见过的低数据损失的参数，这是一个称为 * 泛化 * 的挑战。我们在整本书中回到这些主题。

### 使用学习模型进行预测

鉴于所学到的线性回归模型 $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$，我们现在可以估计一个新房子的价格（未包含在训练数据中），因为它的面积 $x_1$ 和年龄 $x_2$。估计给定特征的目标通常称为 * 预测 * 或 * 推断 *。

我们将尝试坚持 * 预测 *，因为调用这个步骤 * 推断 * 尽管作为深度学习的标准术语出现，但是有些用词不当。在统计数据中，* 推断 * 通常表示基于数据集估计参数。在深度学习从业人员与统计人员交谈时，这种滥用术语是一种常见的混淆根源。

## 速度矢量化

在训练我们的模型时，我们通常希望同时处理整个小批量的示例。高效地做到这一点需要我们对计算进行矢量化并利用快速线性代数库，而不是在 Python 中编写昂贵的 for 循环。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

为了说明为什么这么重要，我们可以考虑两种添加矢量的方法。首先，我们实例化两个包含所有矢量的 1000 维矢量。在一种方法中，我们将使用 Python for 循环遍历向量。在另一种方法中，我们将依赖于对 `+` 的单个调用。

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

由于我们将在这本书中经常对运行时间进行基准测试，所以让我们定义一个计时器。

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

现在，我们可以对工作负载进行基准测试。首先，我们使用 for 循环添加它们，一次一个坐标。

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

或者，我们依靠重新加载的 `+` 运算符来计算单元总和。

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

你可能注意到，第二种方法比第一种方法快得多。矢量化代码通常会产生数量级的加速。此外，我们将更多的数学推送到图书馆，并且不需要自己编写尽可能多的计算，从而降低了错误的可能性。

## 正态分布与平方损耗
:label:`subsec_normal_distribution_and_squared_loss`

虽然你已经可以得到你的手脏只使用上述信息，但在下面我们可以通过有关噪声分布的假设更正式地激励平方损失目标。

线性回归是高斯于 1795 年发明的，他也发现了正态分布（也称为 * 高斯 *）。事实证明，正态分布和线性回归之间的连接比常见子关系更深。要刷新内存，平均值 $\mu$ 和方差 $\sigma^2$（标准差 $\sigma$）的正态分布的概率密度如下：

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

下面我们定义了一个 Python 函数来计算正态分布。

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

我们现在可以显示正态分布。

```{.python .input}
#@tab all
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

正如我们所看到的，改变均值对应于沿 $x$ 轴的偏移，增加方差会将分布扩散，从而降低其峰值。

使用均方误差损失函数（或简单的平方损失）激励线性回归的一种方法是正式假定观测值来自嘈杂的观测值，其中噪声的正态分布如下：

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

因此，我们现在可以通过

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

现在，根据最大似然原理，参数 $\mathbf{w}$ 和 $b$ 的最佳值是最大化整个数据集的 * * 的最佳值：

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

根据最大似然原理选择的估计数称为 * 最大似然估计数 *。虽然, 最大化许多指数函数的产品, 可能看起来很困难, 我们可以大大简化事情, 而不改变目标, 通过最大化的可能性日志，而不是.由于历史原因，优化通常表示为最小化而不是最大化。所以，在不改变任何东西的情况下，我们可以最大限度地减少 * 负对数 * $-\log P(\mathbf y \mid \mathbf X)$。制定数学给我们：

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

现在我们只需要一个假设，即 $\sigma$ 是一些固定的常量。因此，我们可以忽略第一个术语，因为它不依赖于 $\mathbf{w}$ 或 $b$。现在，第二个项与前面引入的平方误差损耗相同，除了乘法常数 $\frac{1}{\sigma^2}$。幸运的是，解决方案并不依赖于 $\sigma$。因此，最小化均方误差等同于在加法高斯噪声假设下对线性模型的最大似然估计。

## 从线性回归到深度网络

到目前为止，我们只谈论线性模型。虽然神经网络覆盖了更丰富的模型系列，但我们可以通过用神经网络的语言来将线性模型视为神经网络。首先，让我们先用 “层” 符号重写事物。

### 神经网络图

深度学习从业者喜欢绘制图表来可视化模型中发生的事情。在 :numref:`fig_single_neuron` 中，我们将线性回归模型描述为神经网络。请注意，这些逻辑示意图突出显示连通性模式，例如每个输入连接到输出的方式，但不突出显示权重或偏差所获得的值。

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

对于 :numref:`fig_single_neuron` 所示的神经网络，输入值为 $x_1, \ldots, x_d$，因此输入图层中的 * 输入数 *（或 * 特征维度 *）为 $d$。网络在 :numref:`fig_single_neuron` 中的输出是 $o_1$，因此输出层中的 * 输出数 * 是 1。请注意，输入值都是 * 给 *，只有一个 * 计算 * 神经元。重点放在计算发生的地方，通常我们在计算图层时不考虑输入图层。也就是说，:numref:`fig_single_neuron` 中神经网络的 * 层数 * 是 1。我们可以将线性回归模型视为仅由一个人工神经元组成的神经网络，或者是单层神经网络。

由于对于线性回归，每个输入都连接到每个输出（在本例中只有一个输出），我们可以将此变换（:numref:`fig_single_neuron` 中的输出图层）视为 * 完全连接图层 * 或 * 密集图层 *。我们将在下一章中更多地讨论由这些层组成的网络。

### 生物学

由于线性回归（1795 年发明）早于计算神经科学，因此将线性回归描述为神经网络似乎不合时宜。要了解为什么线性模型是一个自然的地方，当控制神经学家/神经生理学家沃伦·麦卡洛克和沃尔特·皮茨开始开发模型的人工神经元，考虑一个生物神经元在 :numref:`fig_Neuron`，包括
*树突 *（输入端子），
* 核 * (CPU)、* 轴突 * (输出线) 和 * 轴突终端 * (输出端子)，通过 * 突触 * 实现与其他神经元的连接。

![The real neuron.](../img/Neuron.svg)
:label:`fig_Neuron`

来自其他神经元（或环境传感器，如视网膜）的信息 $x_i$ 被收到在树突中。特别是，该信息通过 * 突触重量 * $w_i$ 加权来确定输入的效果（例如，通过产品 $x_i w_i$ 激活或抑制）。来自多个来源的加权输入以加权总和 $y = \sum_i x_i w_i + b$ 的形式聚合在核中，然后将这些信息发送到轴突 $y$ 中进一步处理，通常在通过 $\sigma(y)$ 进行一些非线性处理之后。从那里，它要么到达目的地（例如肌肉），要么通过树突进入另一个神经元。

当然，很多这样的单位可以用正确的连通性和正确的学习算法拼凑在一起，产生比任何一个神经元更有趣和复杂的行为，可以表达归功于我们研究真实的生物神经系统。

与此同时，今天的大多数深度学习研究在神经科学方面几乎没有直接的灵感。我们援引斯图尔特·罗素和彼得·诺维格谁，在他们的经典 AI 教科书
*人工智能系统 : A Modern Approach* :cite:`Russell.Norvig.2016`,
指出，虽然飞机可能受到鸟类的启发，但几个世纪以来鸟类学并不是航空创新的主要驱动力。同样，当今深度学习的灵感来自于数学、统计和计算机科学。

## 摘要

* 机器学习模型中的关键要素是训练数据，损失函数，优化算法，很明显，模型本身。
* 矢量化使一切更好（主要是数学）和更快（主要是代码）。
* 最小化目标函数并执行最大似然估计可能均值同样的事情。
* 线性回归模型也是神经网络。

## 练习

1. 假设我们有一些数据 $x_1, \ldots, x_n \in \mathbb{R}$.我们的目标是找到一个恒定的 $b$，以便最小化 $\sum_i (x_i - b)^2$。
    * 找到最佳值 $b$ 的分析解决方案。
    * 这个问题及其解决方案与正态分布有何关联？
1. 推导出具有平方误差的线性回归优化问题的解析解。为了保持简单，您可以从问题中省略偏差 $b$（我们可以通过向 $\mathbf X$ 添加一列，包括所有列）来实现这一点。
    * 在矩阵和矢量表示法中写出优化问题（将所有数据视为单个矩阵，将所有目标值视为单个向量）。
    * 计算相对于 $w$ 的损失梯度。
    * 通过将梯度设置为零并求解矩阵方程来查找分析解。
    * 这什么时候可能比使用随机梯度下降更好？这种方法什么时候可能会中断？
1. 假定管理附加噪声 $\epsilon$ 的噪声模型是指数分布。也就是说，
    * 写出模型 $-\log P(\mathbf y \mid \mathbf X)$ 下数据的负对数似然。
    * 你能找到一个封闭的形式解决方案吗？
    * 建议一个随机梯度下降算法来解决这个问题。什么可能会出错（提示：当我们不断更新参数时，静止点附近会发生什么）？你能解决这个问题吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
