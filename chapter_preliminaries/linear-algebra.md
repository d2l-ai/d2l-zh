# 线性代数
:label:`sec_linear-algebra`

现在您可以存储和操作数据，让我们简要地回顾一下基本线性代数的子集，您需要了解和实现本书中介绍的大多数模型。下面我们介绍线性代数中的基本数学对象、算术和运算，通过数学表示法和代码中的相应实现来表达每个对象。

## 标量

如果你从来没有学过线性代数或机器学习，那么你过去的数学经验可能包括一次思考一个数字。而且，如果你曾经平衡支票簿，甚至在餐厅支付晚餐费用，那么你已经知道如何做基本的事情，如添加和乘以数字对。样本，帕罗阿尔托的温度为 $52$ 华氏度。形式上，我们调用仅包含一个数值 * 标量 * 组成的值。如果要将此值转换为摄氏度（指标制系统更合理的温度刻度），则可以计算表达式 $c = \frac{5}{9}(f - 32)$，将 $f$ 设置为 $52$。在此等式中，每个术项（$5$、$9$ 和 $32$）都是标量值。占位符 $c$ 和 $f$ 称为 * 变量 *，它们表示未知的标量值。

在本书中，我们采用了数学表示法，其中标量变量由普通小写字母表示（例如，$x$、$y$ 和 $z$）。我们用 $\mathbb{R}$ 表示所有（连续）* 实值 * 标量的空间。为了便宜起见，我们将严格定义确切 * 空间 * 是什么，但现在只要记住，表达式 $x \in \mathbb{R}$ 是一种正式的方式来说，$x$ 是一个实值标量。符号 $\in$ 可以发音为 “在”，只是表示集合中的会员资格。类似地，我们可以写 $x, y \in \{0, 1\}$ 来说明 $x$ 和 $y$ 是数字，其值只能是 $0$ 或 $1$。

标量由只有一个元素的张量表示。在下一个代码段中，我们实例化两个标量，并使用它们执行一些熟悉的算术运算，即加法，乘法，除法和指数。

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant([3.0])
y = tf.constant([2.0])

x + y, x * y, x / y, x**y
```

## 矢量

您可以将矢量视为简单的标量值列表。我们将这些值称为矢量的 * 元素 *（* 条目 * 或 * 组件 *）。当我们的矢量代表数据集中的示例时，它们的值具有一些真实世界的意义。样本如，如果我们正在培训一个模型来预测贷款违约风险，我们可能会将每个申请人与一个向量相关联，其组成部分与其收入、雇用期限、以前违约次数和其他因素相对应。如果我们正在研究医院患者可能面临的心脏病发作的风险，我们可能会用一个载体来表示每个患者，其成分捕获最近的生命体征、胆固醇水平、每天运动分钟等。在数学表示法中，我们通常将载体表示为粗面、较低的载体信件（例如，第 $\mathbf{x}$ 号、第 $\mathbf{y}$ 号和第 $\mathbf{z})$ 号和第 $\mathbf{z})$ 号）。

我们通过一维张量处理矢量。一般来说，张量可以具有任意长度，取决于机器的内存限制。

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

我们可以通过使用下标来引用矢量的任何元素。样本，我们可以通过 $x_i$ 来参考 $i^\mathrm{th}$ 元素。请注意，元素 $x_i$ 是一个标量，所以我们在引用它时不会粗体字体。大量文献认为柱向量是矢量的默认方向，本书也是如此。在数学中，矢量 $\mathbf{x}$ 可以写为

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

其中 $x_1, \ldots, x_n$ 是矢量的元素。在代码中，我们通过索引到张量来访问任何元素。

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### 长度、维度和形状

让我们重新探讨 :numref:`sec_ndarray` 中的一些概念。向量只是一个数字数组。就像每个数组都有一个长度一样，每个向量也是如此。在数学表示法中，如果我们想说一个向量 $\mathbf{x}$ 由 $n$ 实值标量组成，我们可以将其表示为 $\mathbf{x} \in \mathbb{R}^n$。矢量的长度通常称为矢量的 * 维度 *。

与普通的 Python 数组一样，我们可以通过调用 Python 的内置 `len()` 函数来访问张量的长度。

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

当张量代表一个向量（精确地使用一个轴）时，我们也可以通过 `.shape` 属性访问它的长度。形状是一个元组，沿着张量的每个轴列出长度（维度）。对于只有一个轴的张量，形状只有一个元素。

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

请注意，在这些上下文中，“维度” 一词往往会超载，这往往会混淆人们。为了澄清，我们使用 * 矢量 * 或 * 轴 * 的维度来表示其长度，即矢量或轴的元素数量。但是，我们使用张量的维度来表示张量的轴数。在这个意义上，张量的某个轴的维度将是该轴的长度。

## 矩阵

正如向量将标量从阶零概化为第一阶，矩阵将向量从第一阶推广到第二阶。矩阵，我们通常用粗体表示大写字母（例如，$\mathbf{X}$、$\mathbf{Y}$ 和 $\mathbf{Z}$），在代码中表示为具有两个轴的张量。

在数学表示法中，我们使用 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 来表示矩阵 $\mathbf{A}$ 由实值标量的 $m$ 行和 $n$ 列组成。在视觉上，我们可以说明任何矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 作为一个表格，其中每个元素 $a_{ij}$ 属于 $i^{\mathrm{th}}$ 行和 $j^{\mathrm{th}}$ 列：

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

对于任何 $\mathbf{A} \in \mathbb{R}^{m \times n}$，其形状是（$\mathbf{A}$）或 $m \times n$。具体而言，当矩阵具有相同数量的行和列时，其形状将变为正方形；因此，它被称为 * 方矩阵 *。

当调用我们最喜欢的函数来实例化张量时，我们可以通过指定一个具有两个分量 $m$ 和 $n$ 的形状来创建一个 $m \times n$ 矩阵。

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

我们可以通过指定行（$i$）和列（$j$）的索引来访问矩阵中的标量元素 $a_{ij}$，例如 $[\mathbf{A}]_{ij}$。如果没有给出矩阵 $\mathbf{A}$ 的标量元素，例如在 :eqref:`eq_matrix_def` 中，我们可以简单地使用矩阵 $\mathbf{A}$ 的小写字母和索引下标 $a_{ij}$，来参考 $[\mathbf{A}]_{ij}$。为了保持符号简单，只有在必要时才会将逗号插入到单独的索引中，例如 $a_{2, 3j}$ 和 $[\mathbf{A}]_{2i-1, 3}$。

有时候，我们想翻转轴。当我们交换矩阵的行和列时，结果称为矩阵的 * 转换 *。在形式上，我们用 $\mathbf{A}^\top$ 和 $\mathbf{B} = \mathbf{A}^\top$ 来表示一个矩阵的转置，如果是 $\mathbf{B} = \mathbf{A}^\top$，则是 $i$ 和 $j$。因此，在 :eqref:`eq_matrix_def` 中的转置是一个 $n \times m$ 矩阵：

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

现在我们在代码中访问矩阵的转置。

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

作为方矩阵的一种特殊类型，* 对称矩阵 * $\mathbf{A}$ 等于其转置：$\mathbf{A} = \mathbf{A}^\top$。这里我们定义了一个对称矩阵 `B`。

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

现在我们将 `B` 与它的转置进行比较。

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

矩阵是有用的数据结构：它们允许我们组织具有不同变异模式的数据。样本，我们矩阵中的行可能对应于不同的房屋（数据点），而列可能对应于不同的属性。如果您曾经使用过电子表格软件或已阅读过 :numref:`sec_pandas`，这应该听起来很熟悉。因此，尽管单个向量的默认方向是列向量，但在表示表格数据集的矩阵中，将每个数据点视为矩阵中的行向量更为常规。而且，正如我们将在后面的章节中看到的那样，这个约定将启用常见的深度学习实践。样本如，沿着张量的最外轴，我们可以访问或枚举数据点的小批次，或者只是数据点，如果不存在微型批次。

## 张量

就像向量概化标量一样，矩阵概括向量一样，我们可以构建具有更多轴的数据结构。张量（本小节中的 “张量” 指代数对象）为我们提供了描述具有任意数量轴的 $n$ 维数组的通用方法。样本，向量是一阶张量，矩阵是二阶张量。张量用特殊字体字体的大写字母（例如，$\mathsf{X}$、$\mathsf{Y}$ 和 $\mathsf{Z}$）表示，它们的索引机制（例如 $x_{ijk}$ 和 $[\mathsf{X}]_{1, 2i-1, 3}$）与矩阵类似。

当我们开始处理图像时，张量将变得更加重要，图像作为 $n$ 维数组到达，其中 3 个轴对应于高度，宽度，以及用于堆叠颜色通道（红色，绿色和蓝色）的 * 通道 * 轴。现在，我们将跳过更高阶张量，并专注于基础知识。

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## 张量算术的基本性质

任意数量的轴的标量，向量，矩阵和张量（本小节中的 “张量” 指代数对象）有一些很好的属性，通常会派上用场。样本，您可能已经从元素操作的定义中注意到，任何元素一元运算都不会改变其操作数的形状。同样，给定具有相同形状的任何两个张量，任何二进制元素运算的结果都将是相同形状的张量。样本，添加两个相同形状的矩阵对这两个矩阵执行元素加法。

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

具体而言，两个矩阵的元素乘法称为它们的 * 哈达玛产品 *（数学符号 $\odot$）。考虑一下矩阵 $\mathbf{B} \in \mathbb{R}^{m \times n}$，其中第 $i$ 行和第 $j$ 列的元素是 $b_{ij}$。哈达玛产品的总体值（在 :eqref:`eq_matrix_def` 中定义）和 $\mathbf{B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

将张量乘以或添加标量也不会改变张量的形状，其中操作数张量的每个元素都将被添加或乘以标量。

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## 减少
:label:`subseq_lin-alg-reduction`

我们可以使用任意张量执行的一个有用的操作是计算它们的元素的总和。在数学表示法中，我们使用 $\sum$ 符号表示总和。为了表达元素在长度为 $d$ 的矢量中的总和，我们写了 $\sum_{i=1}^d x_i$。在代码中，我们可以调用计算总和的函数。

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

我们可以在任意形状张量的元素上表示总和。样本，可以写入一个矩阵 $\mathbf{A}$ 的元素的总和。

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

默认情况下，调用用于计算总和的函数
*减少 * 张量沿其所有轴转换为标量。
我们还可以指定通过求和降低张量的轴。以矩阵为样本。要通过汇总所有行的元素来减小行维度（轴 0），我们在调用函数时指定 `axis=0`。由于输入矩阵沿轴 0 减小以生成输出向量，因此输入轴 0 的尺寸会在输出形状中丢失。

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

指定 `axis=1` 将通过汇总所有列的元素来减小列尺寸（轴 1）。因此，输入轴 1 的尺寸在输出形状中丢失。

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

通过求和减少沿行和列的矩阵相当于总结矩阵的所有元素。

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

相关数量是 * 均值 *，也称为 * 平均值 *。我们通过将总和除以元素总数来计算平均值。在代码中，我们可以调用计算任意形状张量上的均值的函数。

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同样，计算均值的函数也可以沿指定轴减少张量。

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### 不减少总和
:label:`subseq_lin-alg-non-reduction`

但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用。

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

实例，由于 `sum_A` 在对每行进行总和后仍保持两个轴，我们可以将 `A` 除以 `sum_A` 与广播。

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

如果我们想沿某个轴计算 `A` 元素的累积总和，比如 `axis=0`（一行一行），我们可以调用 `cumsum` 函数。此函数不会沿任何轴降低输入张量。

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## 点产品

到目前为止，我们只执行了元素操作、总和和平均值。如果这是我们所能做的，线性代数可能不值得自己的部分。但是，最基本的操作之一是点积。给定两个向量 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$，它们的 * 点积 * $\mathbf{x}^\top \mathbf{y}$（或 $\langle \mathbf{x}, \mathbf{y}  \rangle$）是相同位置的元素积的总和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

请注意，我们可以通过执行元素乘法，然后进行总和来表达两个向量的点积：

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

点产品在广泛的环境中非常有用。样本，给定一组值（由矢量 $\mathbf{x}  \in \mathbb{R}^d$ 表示）和一组由 $\mathbf{w} \in \mathbb{R}^d$ 表示的权重表示，$\mathbf{x}$ 中的值的加权总和根据权重 $\mathbf{w}$ 可以表示为点积 $\mathbf{x}^\top \mathbf{w}$。当权重为非负数且总和为 1（即 $\left(\sum_{i=1}^{d} {w_i} = 1\right)$）时，点积表示 * 加权平均值 *。标准化两个向量以具有单位长度后，点积表示它们之间角度的余弦。我们将在本节后面正式介绍这个 * 长度 * 的概念。

## 矩阵矢量产品

现在我们知道如何计算点积，我们可以开始理解 * 矩阵矢量产品 *。回顾分别在 :eqref:`eq_matrix_def` 和 :eqref:`eq_vec_def` 中定义和可视化的基质 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和载体 $\mathbf{x} \in \mathbb{R}^n$。让我们从矩阵 $\mathbf{A}$ 的行向量可视化开始

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每个 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ 都是一个行向量，表示矩阵中的 $i^\mathrm{th}$ 行。矩阵向量积 $\mathbf{A}\mathbf{x}$ 只是一个长度为 $m$ 的柱向量，其 $i^\mathrm{th}$ 元素是点积 $\mathbf{a}^\top_i \mathbf{x}$：

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

我们可以把一个矩阵 $\mathbf{A}\in \mathbb{R}^{m \times n}$ 乘法看作是一个从 $\mathbb{R}^{n}$ 到 $\mathbb{R}^{m}$ 的向量的转换。这些转换证明是非常有用的。样本，我们可以将旋转表示为乘法方矩阵。正如我们将在后续章节中看到的那样，我们还可以使用矩阵矢量产品来描述计算神经网络中的每个图层时所需要的最密集的计算方法。

用张量代码表示矩阵矢量积，我们使用与点积相同的 `dot` 函数。当我们用矩阵 `A` 和矢量 `x` 调用 `np.dot(A, x)` 时，执行矩阵矢量积。请注意，`A` 的列尺寸（沿轴 1 的长度）必须与 `x` 的尺寸（其长度）相同。

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## 矩阵矩阵乘法

如果你已经得到了点积和矩阵矢量产品的挂起，那么 * 矩阵矩阵乘法 * 应该很简单。

假设我们有两个矩阵：

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

用 $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ 表示行向量代表矩阵 $i^\mathrm{th}$ 行的行向量，并让 $\mathbf{b}_{j} \in \mathbb{R}^k$ 成为从矩阵 $j^\mathrm{th}$ 列的列向量。为了生成矩阵积 $\mathbf{C} = \mathbf{A}\mathbf{B}$，最简单的是从其行向量和 $\mathbf{B}$ 列向量的角度来考虑：

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

然后生成矩阵积 $\mathbf{C} \in \mathbb{R}^{n \times m}$，因为我们简单地计算每个元素 $c_{ij}$ 作为点积 $\mathbf{a}^\top_i \mathbf{b}_j$：

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

我们可以认为矩阵矩阵乘法 $\mathbf{AB}$ 简单地执行 $m$ 矩阵矢量积，并将结果拼接在一起，形成一个 $n \times m$ 矩阵。在下面的代码段中，我们在 `A` 和 `B` 上执行矩阵乘法。这里，`A` 是一个包含 5 行和 4 列的矩阵，`B` 是一个包含 4 行和 3 列的矩阵。乘法后，我们得到了一个包含 5 行和 3 列的矩阵。

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

矩阵矩阵乘法可以简单地称为 * 矩阵乘法 *，不应与 Hadamard 产品混淆。

## 规范
:label:`subsec_lin-algebra-norms`

线性代数中一些最有用的运算符是 *norms*。非正式地，矢量的规范告诉我们 * big* 矢量是如何。这里考虑的 * 大小 * 概念不涉及维度，而是涉及组件的大小。

在线性代数中，矢量范数是将向量映射到标量的函数 $f$，满足一些属性。给定任何向量 $\mathbf{x}$，第一个属性说，如果我们按常数因子 $\alpha$ 缩放矢量的所有元素，其范数也会按相同常数因子的 * 绝对值 * 缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个属性是熟悉的三角形不平等：

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个属性只是说，规范必须是非负面的：

$$f(\mathbf{x}) \geq 0.$$

这是有道理的，因为在大多数情况下，任何东西的最小 * size * 是 0。最终属性要求实现最小范数，并且只能通过由全零组成的向量来实现。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

你可能会注意到，规范听起来很像距离度量。如果你还记得欧几里得距离（考虑毕达哥拉斯定理），那么非消极和三角形不平等的概念可能会响铃。事实上，欧几里得距离是一个规范：具体而言，它是 $L_2$ 规范。假设在维矢量 $n$ 中的元素是 $\mathbf{x}$。$\mathbf{x}$ 的 $L_2$ * 规范 * 是矢量元素平方和的平方根：

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

其中，在 $L_2$ 规范中常常省略下标 $2$，也就是说，$\|\mathbf{x}\|$ 等同于 $\|\mathbf{x}\|_2$。在代码中，我们可以按如下方式计算向量的 $L_2$ 范数。

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

在深度学习中，我们更经常地使用平方 $L_2$ 规范。您还会经常遇到 $L_1$ * 规范 *，它表示为矢量元素的绝对值之和：

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

与 $L_2$ 范数相比，它受异常值的影响较小。为了计算 $L_1$ 范数，我们使用元素的总和来组合绝对值函数。

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

$L_2$ 规范和 $L_1$ 规范都是比较普遍的规范 * 的特殊情况：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

类似于 $L_2$ 规范的矢量，矩阵 $\mathbf{X} \in \mathbb{R}^{m \times n}$ 的 * 弗罗比尼斯法则 * 是矩阵元素的平方和的平方根：

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

Frobenius 范数满足矢量规范的所有属性。它的行为就好像它是矩阵形向量的 $L_2$ 范数。调用以下函数将计算矩阵的 Frobenius 范数。

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### 规范和目标
:label:`subsec_norms_and_objectives`

虽然我们不希望得到太远的领先自己，但我们可以植物一些直觉已经有关为什么这些概念是有用的。在深度学习中，我们经常试图解决优化问题：
*最大化 * 分配给观测数据的概率;
*最小化 * 预测之间的距离
以及地面真相观察.将矢量表示分配给项目（如文字、产品或新闻文章），以便最小化相似项目之间的距离，并最大化不同项目之间的距离。通常，目标，也许是深度学习算法的最重要组成部分（除了数据），被表示为规范。

## 关于线性代数的更多信息

在本节中，我们教你所有的线性代数，你需要了解一大块现代深度学习。线性代数还有很多，其中很多数学对于机器学习非常有用。样本，矩阵可以分解为因子，这些分解可以显示真实世界数据集中的低维结构。机器学习的整个子领域都侧重于使用矩阵分解及其向高阶张量的概化来发现数据集中的结构并解决预测问题。但这本书的重点是深度学习。我们相信，一旦你把手弄脏了，在真实数据集上部署有用的机器学习模型，你会更倾向于学习更多数学。因此，虽然我们保留稍后介绍更多数学的权利，但我们将在这里总结这一部分。

如果您渴望了解有关线性代数的更多信息，您可以参考 [online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) 或其他优秀资源 :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`。

## 摘要

* 标量、向量、矩阵和张量是线性代数中的基本数学对象。
* 向量概化标量，矩阵概化向量。
* 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。
* 一个张量可以沿指定的轴减少 `sum` 和 `mean`。
* 两个矩阵的元素乘法被称为他们的哈达玛乘积。它与矩阵乘法不同。
* 在深度学习中，我们经常使用规范，如 $L_1$ 规范、$L_2$ 规范和弗罗本纽斯规范。
* 我们可以对标量、向量、矩阵和张量执行各种操作。

## 练习

1. 证明一个矩阵 $\mathbf{A}$ 的转置是 $\mathbf{A}$ 的转置。
1. 给出两个矩阵 $\mathbf{A}$ 和 $\mathbf{B}$, 显示转置的总和等于一个总和的转置:$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. 给定任何方矩阵 $\mathbf{A}$，是否始终是对称的？为什么？
1. 我们在本节中定义了形状（2,3,4）的张量 `X`。什么是输出的结果？
1. 对于任意形状的张量 `X`，`len(X)` 总是对应于 `X` 特定轴的长度吗？那是什么轴？
1. 运行 `A / A.sum(axis=1)`，看看会发生什么。你能分析原因吗？
1. 当在曼哈顿的两个点之间行驶时，您需要在坐标方面（即通道和街道方面）覆盖多少距离？你可以沿对角线旅行吗？
1. 考虑一个形状（2，3，4）的张量。沿轴 0、1 和 2 的求和输出的形状是什么？
1. 向 `linalg.norm` 函数提供 3 个或更多轴的张量，并观察其输出。这个函数计算任意形状的张量是什么？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
