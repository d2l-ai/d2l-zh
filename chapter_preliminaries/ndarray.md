# 数据操作
:label:`sec_ndarray`

为了完成任何操作，我们需要一些方法来存储和操作数据。一般来说，我们需要处理两件重要的事情：（一）获取数据；（二）在计算机内部处理数据。没有某种方式来存储数据是没有意义的，所以让我们先通过玩合成数据来弄脏我们的手。首先，我们介绍了 $n$ 维数组，也称为 * 张量 *。

如果您使用过 Python 中最广泛使用的科学计算软件包 NumPy，那么您会发现本部分很熟悉。无论您使用哪个框架，它的 * 张量类 *（在 MxNet 中为 `ndarray`，在 Pytorch 和张量流中为 `Tensor`）都与 Numpy 的 `ndarray` 类似，具有一些杀手功能。首先，GPU 支持加速计算，而 NumPy 仅支持 CPU 计算。其次，张量类支持自动差分。这些属性使得张量类适合深度学习。在整个书中，当我们说张量时，我们指的是张量类的实例，除非另有说明。

## 入门

在本节中，我们的目标是帮助您启动和运行，为您提供基本的数学和数字计算工具，在您完成本书的过程中，您将在此基础上构建这些工具。如果你努力研究一些数学概念或库函数，不要担心。以下各节将在实际例子的背景下重新讨论这些材料，并将下沉。另一方面，如果你已经有一些背景，并且想要深入了解数学内容，只需跳过这部分。

:begin_tab:`mxnet`
首先，我们从 MxNet 导入 `np` (`numpy`) 和 `npx` (`numpy_extension`) 模块。在这里，`np` 模块包含 NumPy 支持的函数，而 `npx` 模块包含一组扩展，用于在类似 NumPy 的环境中实现深度学习。当使用张量时，我们几乎总是调用 `set_np` 函数：这是为了兼容 MxNet 其他组件的张量处理。
:end_tab:

:begin_tab:`pytorch`
首先，我们导入 `torch`。请注意，虽然它被称为皮托尔赫，但我们应该导入 `torch` 而不是 `pytorch`。
:end_tab:

:begin_tab:`tensorflow`
首先，我们导入 `tensorflow`。由于名称有点长，我们通常使用短别名 `tf` 导入它。
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

张量表示一个（可能是多维）的数值数组。对于一个轴，张量对应（在数学中）与 * 矢量 *。对于两个轴，张量对应于一个 * 矩阵 *。具有两个以上轴的张量没有特殊的数学名称。

首先，我们可以使用 `arange` 创建一个行向量 `x`，其中包含以 0 开始的前 12 个整数，尽管它们是默认创建为浮点数。张量中的每个值都称为张量的 * 元素 *。实例，张量 `x` 中有 12 个元素。除非另有规定，否则新张量将存储在主存储器中并指定用于基于 CPU 的计算。

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

我们可以通过检查张量的 `shape` 属性来访问张量的 * 形状 *（沿每个轴的长度）。

```{.python .input}
#@tab all
x.shape
```

如果我们只想知道张量中元素的总数，即所有形状元素的乘积，我们可以检查它的大小。因为我们在这里处理一个向量，所以它的 `shape` 的单个元素与它的大小相同。

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

要在不改变元素数量或其值的情况下改变张量的形状，我们可以调用 `reshape` 函数。样本，我们可以变换张量 `x` 从具有形状 (12,) 的行向量转换为具有形状 (3, 4) 的矩阵。这个新张量包含完全相同的值，但将它们视为组织为 3 行和 4 列的矩阵。重申，尽管形状已经改变，但 `x` 中的元素没有。请注意，通过改变形状，大小不会改变。

```{.python .input}
#@tab mxnet, pytorch
x = x.reshape(3, 4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(x, (3, 4))
x
```

通过手动指定每个尺寸进行重塑是不必要的。如果我们的目标形状是一个具有形状（高度，宽度）的矩阵，那么在我们知道宽度之后，高度会隐式给出。我们为什么要自己执行分裂？在上面的样本中，为了获得包含 3 行的矩阵，我们指定了它应该有 3 行和 4 列。幸运的是，张量可以自动计算出一个维度，给出其余部分。我们通过为我们希望张量自动推断的维度放置 `-1` 来调用此功能。在我们的例子中，我们可以等效地称为 `x.reshape(-1, 4)` 或 `x.reshape(3, -1)`，而不是调用 `x.reshape(3, 4)`。

通常，我们希望我们的矩阵初始化为零，一，一些其他常量，或者从特定分布中随机采样的数字。我们可以创建一个表示张量的张量，所有元素都设置为 0，形状为 (2,3,4)，如下所示：

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

同样，我们可以创建每个元素设置为 1 的张量，如下所示：

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

通常，我们想从某个概率分布中随机抽样张量中每个元素的值。样本，当我们构造数组作为神经网络中的参数时，我们通常会随机初始化它们的值。以下代码片段创建一个形状（3，4）的张量。其每个元素都从平均值为 0，标准差为 1 的标准高斯（正态）分布中随机采样。

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

我们还可以通过提供包含数值的 Python 列表（或列表列表）来为所需张量中的每个元素指定精确值。在这里，最外层的列表对应于轴 0，内部列表对应于轴 1。

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 操作

这本书不是关于软件工程。我们的兴趣不仅仅限于从数组读取和写入数据。我们想在这些数组上执行数学运算。一些最简单和最有用的操作是 * 元素 * 操作。它们将标准标量运算应用于数组的每个元素。对于将两个数组作为输入的函数，元素运算将一些标准的二进制运算符应用于两个数组中的每对相应元素。我们可以从任何从标量映射到标量的函数创建一个元素函数。

在数学表示法中，我们将通过签名 $f: \mathbb{R} \rightarrow \mathbb{R}$ 来表示这样的 * 一个 * 标量运算符（获取一个输入）。这只是意味着该函数从任何实数（$\mathbb{R}$）映射到另一个实数。同样，我们通过签名 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ 表示一个 * 二进制 * 标量运算符（取两个实际输入，并产生一个输出）。给定任意两个向量 $\mathbf{u}$ 和 $c_i \gets f(u_i, v_i)$2 * 具有相同形状 *，以及二进制运算符 $c_i \gets f(u_i, v_i)$0，我们可以通过为所有 $i$ 设置 $c_i \gets f(u_i, v_i)$ 来生成一个向量 $\mathbf{c} = F(\mathbf{u},\mathbf{v})$，其中 $c_i, u_i$ 和 $c_i \gets f(u_i, v_i)$3 是矢量的 $i^\mathrm{th}$ 元素。在这里，我们通过 * 将标量函数提升到元素向量操作来生成矢量值 $c_i \gets f(u_i, v_i)$1。

对于任意形状的相同形状张量，常见的标准算术运算符（`+`、`-`、`*`、`/` 和 `**`）都已 * 提升 * 为元素运算。我们可以在同一形状的任何两个张量上调用元素操作。在下面的样本中，我们使用逗号来制定一个 5 元素元组，其中每个元素都是元素操作的结果。

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

可以按元素方式应用更多的操作，包括像指数这样的一元运算符。

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

除了元素计算外，我们还可以执行线性代数运算，包括向量点积和矩阵乘法。我们将在:numref:`sec_linear-algebra` 中解释线性代数的关键位（没有假定事先知识）。

我们也可以 * 连接 * 多张张量在一起，将它们端到端堆叠以形成更大的张量。我们只需要提供张量列表，并告诉系统沿哪个轴连结。下面的样本显示了当我们沿行（轴 0，形状的第一个元素）与列（轴 1，形状的第二个元素）连结两个矩阵时会发生什么情况。我们可以看到，第一个输出张量的轴-0 长度 ($6$) 是两个输入张量轴-0 长度 ($3 + 3$) 的总和；第二个输出张量的轴-1 长度 ($8$) 是两个输入张量轴-1 长度 ($4 + 4$) 的总和。

```{.python .input}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([x, y], axis=0), tf.concat([x, y], axis=1)
```

有时，我们想通过 * 逻辑语句 * 构建二进制张量。以 `x == y` 为样本。对于每个位置，如果 `x` 和 `y` 在该位置处相等，则新张量中的相应条目采用值 1，这意味着逻辑语句 `x == y` 在该位置处为真；否则该位置为 0。

```{.python .input}
#@tab all
x == y
```

对张量中的所有元素进行求和会产生一个只有一个元素的张量。

```{.python .input}
#@tab mxnet, pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```

## 广播机制
:label:`subsec_broadcasting`

在上面的部分中，我们看到了如何在相同形状的两个张量上执行元素操作。在某些情况下，即使形状不同，我们仍然可以通过调用 * 广播机制 * 来执行单元操作。这种机制的工作方式如下：首先，通过适当复制元素来展开一个或两个数组，以便在转换之后，两个张量具有相同的形状。其次，对生成的数组执行元素操作。

在大多数情况下，我们沿着一个数组最初只有长度 1 的轴进行广播，如下样本：

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

由于 `a` 和 `b` 分别是 $3\times1$ 和 $1\times2$ 矩阵，因此如果我们想添加它们，它们的形状不匹配。我们 * 将两个矩阵的条目广播为一个更大的 $3\times2$ 矩阵，如下所示：对于矩阵 `a`，它复制列，对于矩阵 `b`，它会复制行，然后再加上两个元素。

```{.python .input}
#@tab all
a + b
```

## 索引和切片

就像在任何其他 Python 数组中一样，张量中的元素可以通过索引访问。与任何 Python 数组一样，第一个元素具有索引 0，并且指定范围以包含第一个元素，但在 * 之前 * 最后一个元素。与标准 Python 列表一样，我们可以通过使用负索引根据元素到列表末尾的相对位置访问元素。

因此，`[-1]` 选择最后一个元素，`[1:3]` 选择第二个元素和第三个元素，如下所示：

```{.python .input}
#@tab all
x[-1], x[1:3]
```

:begin_tab:`mxnet, pytorch`
除了阅读之外，我们还可以通过指定索引来编写矩阵的元素。
:end_tab:

:begin_tab:`tensorflow`
张量流中的 `Tensors` 是不可变的，不能分配给。张量流中的 `Variables` 是支持分配的状态的可变容器。请记住，张量流中的渐变不会通过 `Variable` 分配向后流动。

除了为整个 `Variable` 分配一个值之外，我们还可以通过指定索引来编写 `Variable` 的元素。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
x[1, 2] = 9
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[1, 2].assign(9)
x_var
```

如果我们想为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们分配值。实例，`[0:2, :]` 访问第一行和第二行，其中 `:` 采用沿轴 1（列）的所有元素。虽然我们讨论了矩阵的索引，但这显然也适用于向量和超过 2 个维度的张量。

```{.python .input}
#@tab mxnet, pytorch
x[0:2, :] = 12
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[0:2,:].assign(tf.ones(x_var[0:2,:].shape, dtype = tf.float32)*12)
x_var
```

## 节省内存

运行操作可能会导致将新内存分配给主机结果。样本如，如果我们编写 `y = x + y`，我们将取消引用 `y` 用来指向的张量，而是指向新分配的内存处的 `y`。在下面的样本中，我们使用 Python 的 `id()` 函数演示了这一点，该函数给了我们内存中引用对象的确切地址。运行 `y = y + x` 后，我们会发现 `id(y)` 指向另一个位置。这是因为 Python 首先评估 `y + x`，为结果分配新的内存，然后使 `y` 指向内存中的这个新位置。

```{.python .input}
#@tab all
before = id(y)
y = y + x
id(y) == before
```

这可能是不可取的，原因有两个。首先，我们不想一直不必要地分配内存。在机器学习中，我们可能有数百兆字节的参数，并且每秒多次更新所有参数。通常情况下，我们希望执行这些更新 * 到位。其次，我们可以指向来自多个变量的相同参数。如果我们不更新，其他引用仍然会指向旧的内存位置，这样我们的代码中的一部分可能会无意中引用陈旧的参数。

:begin_tab:`mxnet, pytorch`
幸运的是，执行就地操作非常简单。我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如 `y [:] = <expression> `。为了说明这一概念，我们首先创建一个新的矩阵 `z`，其形状与另一个 `y` 相同，使用 `zeros_like` 来分配一个 $0$ 条目的块。
:end_tab:

:begin_tab:`tensorflow`
`Variables` 是张量流中状态的可变容器。它们提供了一种存储模型参数的方法。我们可以将操作的结果分配给 `assign` 的 `Variable`。为了说明这个概念，我们创建了一个 `Variable` `z`，其形状与另一个张量 `y` 相同，使用 `zeros_like` 来分配一个 $0$ 个条目的块。
:end_tab:

```{.python .input}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab pytorch
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab tensorflow
z = tf.Variable(tf.zeros_like(y))
print('id(z):', id(z))
z.assign(x + y)
print('id(z):', id(z))
```

:begin_tab:`mxnet, pytorch`
如果在后续计算中未重复使用值 `x`，我们也可以使用 `x[:] = x + y` 或 `x += y` 来减少操作的内存开销。
:end_tab:

:begin_tab:`tensorflow`
即使您将状态持久存储在 `Variable` 中，您也可能希望通过避免为不是模型参数的张量分配额来进一步减少内存使用量。

由于 TensorFlow `Tensors` 是不可变的，且梯度不会通过 `Variable` 分配流动，因此 TensorFlow 不会提供明确的方式来就地运行单个操作。

但是，TensorFlow 提供了 `tf.function` 装饰器来将计算包装在 TensorFlow 图中，该图在运行之前进行编译和优化。这允许 TensorFlow 修剪未使用的值，并重复使用不再需要的先前分配。这样可以最大限度地减少 TensorFlow 计算的内存开销。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(x, y):
  z = tf.zeros_like(y)  # This unused value will be pruned out.
  a = x + y  # Allocations will be re-used when no longer needed.
  b = a + y
  c = b + y
  return c + y

computation(x, y)
```

## 转换为其他 Python 对象

转换为 NumPy 张量很容易，反之亦然。转换的结果不共享内存。这个小的不便实际上是非常重要的：当您在 CPU 或 GPU 上执行操作时，您不希望停止计算，等待查看 Python 的 NumPy 包是否希望使用相同的内存块执行其他操作。

```{.python .input}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

```{.python .input}
#@tab pytorch
a = x.numpy()
b = torch.tensor(a)
type(a), type(b)
```

```{.python .input}
#@tab tensorflow
a = x.numpy()
b = tf.constant(a)
type(a), type(b)
```

要将大小 1 张量转换为 Python 标量，我们可以调用 `item` 函数或 Python 的内置函数。

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## 摘要

* 为深度学习存储和操作数据的主要接口是张量（$n$ 维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换为其他 Python 对象。

## 练习

1.运行本节中的代码。将本节中的条件语句 `x == y` 更改为 `x < y` or `x > y`，然后看看你可以得到什么样的张量。1.用其他形状（例如三维张量）替换广播机构中按元件操作的两个张量。结果是否与预期相同？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/187)
:end_tab:
