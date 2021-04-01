# 数据操作
:label:`sec_ndarray`

为了能够完成各种操作，我们需要某种方法来存储和操作数据。一般来说，我们需要做两件重要的事情：（1）获取数据；（2）在数据读入计算机后对其进行处理。如果没有某种方法来存储数据，那么获取数据是没有意义的。我们先尝试一个合成数据。首先，我们介绍$n$维数组，也称为 *张量*（tensor）。

如果你使用过 Python 中最广泛使用的科学计算包 NumPy，那么你会感觉对本部分很熟悉。无论你使用哪个框架，它的 *张量类*（在 MXNet 中为 `ndarray`，在 PyTorch 和TensorFlow中为 `Tensor`）与 Numpy 的 `ndarray` 类似，但都比Numpy 的 `ndarray`多一些重要功能。首先，GPU 很好地支持加速计算，而 NumPy 仅支持 CPU 计算。其次，张量类支持自动微分。这些功能使得张量类更适合深度学习。除非另有说明，在整本书中所说的张量指的是张量类的实例。


## 入门

在本节中，我们的目标是帮助你开始了解并运行一些基本数值计算工具。在你阅读本书的过程中，将用到这些工具。如果你很难理解一些数学概念或库函数，请不要担心。在后面的章节将通过一些实际的例子来回顾这些内容。如果你已经有了一些背景知识，想要深入学习数学内容，可以就跳过这一节。

:begin_tab:`mxnet`
首先，我们从MXNet导入 `np`（`numpy`）模块和 `npx`（`numpy_extension`）模块。`np` 模块包含了 NumPy 支持的函数。而 `npx` 模块包含了一组扩展函数，用来在类似 NumPy 的环境中实现深度学习开发。当使用张量时，我们几乎总是会调用 `set_np` 函数：这是为了兼容 MXNet 其他的张量处理组件。
:end_tab:

:begin_tab:`pytorch`
(**首先，我们导入 `torch`。请注意，虽然它被称为PyTorch，但我们应该导入 `torch` 而不是 `pytorch`。**)
:end_tab:

:begin_tab:`tensorflow`
首先，我们导入 `tensorflow`。由于名称有点长，我们经常使用短别名 `tf` 导入它。
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

[**张量表示一个数值组成的数组，这个数组可能有多个维度**]。具有一个轴的张量对应于数学上的 *向量*（vector）。具有两个轴的张量对应于数学上的 *矩阵*（matrix）。具有两个轴以上的张量没有特殊的数学名称。

首先，我们可以使用 `arange` 创建一个行向量 `x`。这个行向量包含以0开始的前12个整数，它们默认创建为浮点数。张量中的每个值都称为张量的 *元素*（element）。例如，张量 `x` 中有 12 个元素。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。



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

[**我们可以通过张量的 `shape` 属性来访问张量的 *形状***] (~~和张量中元素的总数~~)（沿每个轴的长度）。

```{.python .input}
#@tab all
x.shape
```

如果我们只想知道张量中元素的总数，即形状的所有元素乘积，我们可以检查它的大小（size）。
因为这里在处理的是一个向量，所以它的 `shape` 与它的 `size` 相同。

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

[**要改变一个张量的形状而不改变元素数量和元素值，我们可以调用 `reshape` 函数。**]
例如，我们可以把张量 `x` 从形状为 (12, ) 的行向量转换为形状 (3, 4) 的矩阵。这个新的张量包含与转换前相同的值，但是把它们看成一个三行四列的矩阵。要重点说明一下，虽然形状发生了改变，但元素值没有变。注意，通过改变张量的形状，张量的大小不会改变。


```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

不需要通过手动指定每个维度来改变形状。
如果我们的目标形状是 (高度, 宽度) ，那么在知道宽度后，高度应当会隐式得出，我们不必自己做除法。在上面的例子中，要获得一个有3行的矩阵，我们手动指定了它有3行和4列。幸运的是，张量在给出其他部分后可以自动计算出一个维度。我们可以通过将希望张量自动推断的维度放置 `-1` 来调用此功能。在上面的例子中，我们可以用 `x.reshape(-1, 4)` 或 `x.reshape(3, -1)`来取代`x.reshape(3, 4)`。


有时，我们希望[**使用全0、全1、其他常量或者从特定分布中随机采样的数字**]，来初始化矩阵。我们可以创建一个形状为 (2, 3, 4) 的张量，其中所有元素都设置为0。代码如下：


```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

同样的，我们可以创建一个张量，其中所有元素都设置为1。代码如下：

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

有时我们想从某个概率分布中随机采样来得到张量中每个元素的值。例如，当我们构造数组来作为神经网络中的参数时，我们通常会随机初始化参数的值。以下代码创建一个形状为 (3, 4) 的张量。其中的每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样。


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

我们还可以[**通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值**]。在这里，最外层的列表对应于轴 0，内层的列表对应于轴 1。


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

## 运算

这本书不是关于软件工程的。我们的兴趣不仅仅限于从数组读取和写入数据。我们想在这些数组上执行数学运算。一些最简单且最有用的操作是 *按元素*（elementwise） 操作。它们将标准标量运算符应用于数组的每个元素。对于将两个数组作为输入的函数，按元素运算将二元运算符应用于两个数组中的每对位置对应的元素。我们可以基于任何从标量到标量的函数来创建按元素函数。

在数学表示法中，我们将通过符号 $f: \mathbb{R} \rightarrow \mathbb{R}$ 来表示 *一元* 标量运算符（只接收一个输入）。这意味着该函数从任何实数（$\mathbb{R}$）映射到另一个实数。同样，我们通过符号 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ 表示 *二元* 标量运算符，这意味着该函数接收两个输入，并产生一个输出。给定同一形状的任意两个向量$\mathbf{u}$和$\mathbf{v}$ 和二元运算符 $f$，我们可以得到向量$\mathbf{c} = F(\mathbf{u},\mathbf{v})$。具体计算方法是$c_i \gets f(u_i, v_i)$ ，其中 $c_i$、$u_i$ 和 $v_i$ 分别是向量$\mathbf{c}$、$\mathbf{u}$ 和 $\mathbf{v}$中的元素。在这里，我们通过将标量函数升级为按元素向量运算来生成向量值 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$。

对于任意具有相同形状的张量，[**常见的标准算术运算符（`+`、`-`、`*`、`/` 和 `**`）都可以被升级为按元素运算**]。我们可以在同一形状的任意两个张量上调用按元素操作。在下面的例子中，我们使用逗号来表示一个具有5个元素的元组，其中每个元素都是按元素操作的结果。



```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

可以(**按按元素方式应用更多的计算**)，包括像求幂这样的一元运算符。

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

除了按元素计算外，我们还可以执行线性代数运算，包括向量点积和矩阵乘法。我们将在 :numref:`sec_linear-algebra` 中解释线性代数的重点内容（不需要先修知识）。

[**我们也可以把多个张量 *连结*（concatenate） 在一起**]，把它们端对端地叠起来形成一个更大的张量。
我们只需要提供张量列表，并给出沿哪个轴连结。下面的例子分别演示了当我们沿行（轴-0，形状的第一个元素）和按列（轴-1，形状的第二个元素）连结两个矩阵时会发生什么情况。我们可以看到，第一个输出张量的轴-0长度 ($6$) 是两个输入张量轴-0长度的总和 ($3 + 3$)；第二个输出张量的轴-1长度 ($8$) 是两个输入张量轴-1长度的总和 ($4 + 4$)。

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

有时，我们想[**通过 *逻辑运算符* 构建二元张量**]。以 `X == Y` 为例子。
对于每个位置，如果 `X` 和 `Y` 在该位置相等，则新张量中相应项的值为1，这意味着逻辑语句 `X == Y` 在该位置处为真，否则该位置为 0。

```{.python .input}
#@tab all
X == Y
```

[**对张量中的所有元素进行求和会产生一个只有一个元素的张量。**]

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## 广播机制
:label:`subsec_broadcasting`

在上面的部分中，我们看到了如何在相同形状的两个张量上执行按元素操作。在某些情况下，[**即使形状不同，我们仍然可以通过调用 *广播机制* （broadcasting mechanism） 来执行按元素操作**]。这种机制的工作方式如下：首先，通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状。其次，对生成的数组执行按元素操作。

在大多数情况下，我们将沿着数组中长度为1的轴进行广播，如下例子：


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

由于 `a` 和 `b` 分别是 $3\times1$ 和 $1\times2$ 矩阵，如果我们让它们相加，它们的形状不匹配。我们将两个矩阵*广播*为一个更大的 $3\times2$ 矩阵，如下所示：矩阵 `a`将复制列，矩阵 `b`将复制行，然后再按元素相加。

```{.python .input}
#@tab all
a + b
```

## 索引和切片

就像在任何其他 Python 数组中一样，张量中的元素可以通过索引访问。与任何 Python 数组一样：第一个元素的索引是 0；可以指定范围以包含第一个元素和最后一个之前的元素。与标准 Python 列表一样，我们可以通过使用负索引根据元素到列表尾部的相对位置访问元素。

因此，我们[**可以用 `[-1]` 选择最后一个元素，可以用 `[1:3]` 选择第二个和第三个元素**]，如下所示：

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
[**除读取外，我们还可以通过指定索引来将元素写入矩阵。**]
:end_tab:

:begin_tab:`tensorflow`
TensorFlow中的 `Tensors` 是不可变的，也不能被赋值。TensorFlow中的 `Variables` 是支持赋值的可变容器。请记住，TensorFlow中的梯度不会通过 `Variable` 反向传播。

除了为整个 `Variable` 分配一个值之外，我们还可以通过索引来写入 `Variable` 的元素。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

如果我们想[**为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值。**]
例如，`[0:2, :]` 访问第1行和第2行，其中 “:” 代表沿轴 1（列）的所有元素。虽然我们讨论的是矩阵的索引，但这也适用于向量和超过2个维度的张量。

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## 节省内存

[**运行一些操作可能会导致为新结果分配内存**]。例如，如果我们用 `Y = X + Y`，我们将取消引用 `Y` 指向的张量，而是指向新分配的内存处的张量。

在下面的例子中，我们用 Python 的 `id()` 函数演示了这一点，它给我们提供了内存中引用对象的确切地址。运行 `Y = Y + X` 后，我们会发现 `id(Y)` 指向另一个位置。这是因为 Python 首先计算 `Y + X`，为结果分配新的内存，然后使 `Y` 指向内存中的这个新位置。

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

这可能是不可取的，原因有两个：首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新。其次，我们可能通过多个变量指向相同参数。如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

:begin_tab:`mxnet, pytorch`
幸运的是，(**执行原地操作**)非常简单。我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如 `Y[:] = <expression>`。为了说明这一点，我们首先创建一个新的矩阵 `Z`，其形状与另一个 `Y` 相同，使用 `zeros_like` 来分配一个全$0$的块。
:end_tab:

:begin_tab:`tensorflow`
`Variables` 是TensorFlow中的可变容器。它们提供了一种存储模型参数的方法。我们可以通过`assign`将一个操作的结果分配给一个 `Variable`。为了说明这一点，我们创建了一个与另一个张量 `Y` 相同的形状的 `Z`，使用 `zeros_like` 来分配一个全$0$的块。
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**如果在后续计算中没有重复使用 `X`，我们也可以使用 `X[:] = X + Y` 或 `X += Y` 来减少操作的内存开销。**]
:end_tab:

:begin_tab:`tensorflow`
即使你将状态持久存储在 `Variable` 中，你也可能希望避免为不是模型参数的张量过度分配内存，从而进一步减少内存使用量。

由于 TensorFlow的 `Tensors` 是不可变的，而且梯度不会通过 `Variable` 流动，因此 TensorFlow 没有提供一种明确的方式来原地运行单个操作。

但是，TensorFlow提供了`tf.function`修饰符，将计算封装在TensorFlow图中，该图在运行前经过编译和优化。这允许TensorFlow删除未使用的值，并复用先前分配的且不再需要的值。这样可以最大限度地减少 TensorFlow 计算的内存开销。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # 这个未使用的值将被删除
    A = X + Y  # 当不再需要时，分配将被复用
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 转换为其他 Python 对象

[**转换为 NumPy 张量**]很容易，反之也很容易。转换后的结果不共享内存。
这个小的不便实际上是非常重要的：当你在 CPU 或 GPU 上执行操作的时候，如果 Python 的 NumPy 包也希望使用相同的内存块执行其他操作，你不希望停下计算来等它。


```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```
要(**将大小为1的张量转换为 Python 标量**)，我们可以调用 `item` 函数或 Python 的内置函数。

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

## 小结

* 深度学习存储和操作数据的主要接口是张量（$n$维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他 Python 对象。

## 练习

1. 运行本节中的代码。将本节中的条件语句 `X == Y` 更改为 `X < Y` 或 `X > Y`，然后看看你可以得到什么样的张量。
1. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1745)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1747)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1746)
:end_tab:
