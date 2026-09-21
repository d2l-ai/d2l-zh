# 特征分解
:label:`sec_eigendecompositions`

特征值往往是学习线性代数时遇到的最有用的概念之一，
但初学者很容易忽略它的重要性。
下面我们将介绍特征分解，
并尝试说明它为何如此重要。

假设矩阵$A$如下：

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

将$A$作用于任意向量$\mathbf{v}=[x,y]^\top$，
可以得到向量$\mathbf{A}\mathbf{v}=[2x,-y]^\top$。
它有一个直观的解释：
先将向量沿$x$方向拉伸至原来的两倍，
再沿$y$方向翻转。

然而，对于*某些*向量，总有一些性质保持不变。
例如，$[1,0]^\top$被映射为$[2,0]^\top$，
而$[0,1]^\top$被映射为$[0,-1]^\top$。
变换后的向量仍在原来的直线上，
矩阵所做的唯一改变就是分别将它们伸缩$2$倍和$-1$倍。
我们把这样的向量称为*特征向量*（eigenvector），
把相应的伸缩倍数称为*特征值*（eigenvalue）。

一般而言，如果能找到数$\lambda$
和向量$\mathbf{v}$，使得

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

那么称$\mathbf{v}$是$A$的特征向量，$\lambda$是对应的特征值。

## 寻找特征值
下面研究如何求出特征值。等式两边同时减去$\lambda\mathbf{v}$，
再将向量提取出来，
可知上式等价于：

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

要使 :eqref:`eq_eigvalue_der` 成立，$(\mathbf{A}-\lambda\mathbf{I})$
必须把某个方向压缩为零，
因此它不可逆，其行列式也为零。
所以，只需找出使$\det(\mathbf{A}-\lambda\mathbf{I})=0$成立的$\lambda$，
就能得到*特征值*。
求得特征值后，再求解
$\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$，
即可找到对应的*特征向量*。

### 一个例子
下面考虑一个更复杂的矩阵：

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

考虑$\det(\mathbf{A}-\lambda\mathbf{I})=0$，
它等价于多项式方程
$0=(2-\lambda)(3-\lambda)-2=(4-\lambda)(1-\lambda)$。
因此，两个特征值分别为$4$和$1$。
为了求出对应的向量，需要求解

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{和} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

可以分别求得向量$[1,-1]^\top$和$[1,2]^\top$。

可以使用内置的`numpy.linalg.eig`函数在代码中验证结果。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

注意，`numpy`会把特征向量归一化为长度$1$，
而我们手算的特征向量可以取任意长度。
此外，符号的选择也是任意的。
尽管如此，计算得到的向量仍与我们针对相同特征值手算出的向量平行。

## 分解矩阵
继续研究前面的例子。令

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

其各列为矩阵$\mathbf{A}$的特征向量。再令

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

其对角线元素为对应的特征值。
根据特征值和特征向量的定义，有

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

矩阵$W$可逆，因此可以在等式两边右乘$W^{-1}$，得到

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

下一节将介绍这种分解带来的一些良好结果。
目前只需知道，只要能找到一组完备的线性无关特征向量
（从而使$W$可逆），这样的分解就存在。

## 特征分解上的运算
特征分解 :eqref:`eq_eig_decomp` 的一个优点是，
许多常见运算都可以用特征分解简洁地表示。首先考虑：

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{n\text{次}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{n\text{次}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{n\text{次}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

这说明，对于矩阵的任意正整数次幂，
只需将其特征值提升到相同次幂，就能得到相应的特征分解。
负整数次幂也可以得到同样的结论，
因此要计算矩阵的逆，只需考虑

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

换言之，只需对每个特征值取倒数。
只要所有特征值都非零，这种方法就成立，
因此矩阵可逆等价于没有零特征值。

进一步可以证明，如果$\lambda_1,\ldots,\lambda_n$
是一个矩阵的特征值，那么该矩阵的行列式为

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

也就是所有特征值的乘积。
这在直观上也很合理：无论$\mathbf{W}$进行了怎样的伸缩，
$W^{-1}$都会将其抵消，所以最终真正保留下来的伸缩
只来自与对角矩阵$\boldsymbol{\Sigma}$相乘，
而它对体积的伸缩倍数正是所有对角元素的乘积。

最后，回顾矩阵的秩是其线性无关列的最大数目。
仔细考察特征分解可以发现，
矩阵的秩等于$\mathbf{A}$的非零特征值个数。

还可以继续列举例子，但核心思想已经很清楚：
特征分解可以简化许多线性代数计算，
它既是许多数值算法的基础运算，
也是大量线性代数分析的基础。

## 对称矩阵的特征分解
我们并不总能找到足够多的线性无关特征向量
来完成上述过程。例如，矩阵

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

只有一个特征向量，即$(1,0)^\top$。
要处理这样的矩阵，需要使用本节无法涵盖的更高级技术，
例如若尔当标准形或奇异值分解。
我们通常需要把注意力限制在能够保证存在一组完备特征向量的矩阵上。

最常见的一类是*对称矩阵*（symmetric matrix），
即满足$\mathbf{A}=\mathbf{A}^\top$的矩阵。
此时可以令$W$为*正交矩阵*（orthogonal matrix）：它的每一列都是长度为$1$的向量，且各列彼此垂直，并满足
$\mathbf{W}^\top=\mathbf{W}^{-1}$；此外，所有特征值都是实数。
因此在这种特殊情形下， :eqref:`eq_eig_decomp` 可以写成

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Gershgorin圆盘定理
特征值通常很难通过直觉来判断。
面对任意矩阵，如果不实际计算，往往很难说明其特征值是多少。
不过，如果矩阵中最大的元素位于对角线上，
有一个定理可以方便地给出较好的近似。

设$\mathbf{A}=(a_{ij})$为任意$n\times n$方阵，
定义$r_i=\sum_{j\neq i}|a_{ij}|$。
令$\mathcal{D}_i$表示复平面上以$a_{ii}$为圆心、$r_i$为半径的圆盘。
那么，$\mathbf{A}$的每个特征值都包含在某个$\mathcal{D}_i$中。

这个表述包含的信息较多，下面通过一个例子来理解它。考虑矩阵：

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

这里$r_1=0.3$、$r_2=0.6$、$r_3=0.8$且$r_4=0.9$。
该矩阵是对称矩阵，所以所有特征值都是实数。
这意味着所有特征值都位于下列某个区间内：

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


数值计算表明，其特征值近似为$0.99$、$2.97$、$4.95$和$9.08$，
全都落在上述区间内。

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

这样便可以近似估计特征值。
当对角线元素明显大于其他元素时，
这种近似通常相当准确。

这看似只是一个小结论，
但特征分解是一个复杂而微妙的主题，
任何直观认识都很有帮助。

## 一个有用的应用：迭代映射的增长

现在我们已经从原理上理解了特征向量，
下面来看它们如何帮助我们深入理解神经网络行为中的一个核心问题：
恰当的权重初始化。

### 用特征向量描述长期行为

对深度神经网络初始化进行完整的数学研究超出了本书的范围，
但可以通过一个简化模型理解特征值如何帮助我们分析模型的工作方式。
我们知道，神经网络由线性变换层和非线性运算交替组成。
为简单起见，这里假设不存在非线性，
而且变换只是重复应用同一个矩阵$A$，
于是模型的输出为

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

初始化这类模型时，$A$通常取元素服从高斯分布的随机矩阵，
因此我们构造一个这样的矩阵。
具体而言，从一个元素服从均值为$0$、方差为$1$的高斯分布的$5\times5$矩阵开始。

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### 随机数据上的行为
为简化这个模型，
假设输入数据向量$\mathbf{v}_{in}$是一个五维高斯随机向量。
先思考我们期望发生什么。
考虑一个一般的机器学习问题：
我们尝试把图像等输入数据转化为预测，
例如图像中是猫的概率。
如果反复应用$\mathbf{A}$会把随机向量拉伸得非常长，
那么输入中的微小变化就会被放大为输出中的巨大变化——
仅对输入图像做细微修改，就会产生截然不同的预测。
这显然不合理。

反过来，如果$\mathbf{A}$不断缩短随机向量，
那么经过许多层后，向量实际上会收缩为零，
输出将不再依赖输入。这显然也不合理。

我们需要在增长和衰减之间取得微妙的平衡，
确保输出随输入变化，但变化不能过大。

下面反复将矩阵$\mathbf{A}$乘以随机输入向量，并记录向量的范数，看看会发生什么。

```{.python .input}
#@tab mxnet
# 计算反复应用`A`后的范数序列
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, '迭代', '值')
```

```{.python .input}
#@tab pytorch
# 计算反复应用`A`后的范数序列
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, '迭代', '值')
```

```{.python .input}
#@tab tensorflow
# 计算反复应用`A`后的范数序列
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, '迭代', '值')
```

范数正在不受控制地增长！
如果计算相邻范数的比值，就能发现一种规律。

```{.python .input}
#@tab mxnet
# 计算范数的伸缩因子
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, '迭代', '比值')
```

```{.python .input}
#@tab pytorch
# 计算范数的伸缩因子
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, '迭代', '比值')
```

```{.python .input}
#@tab tensorflow
# 计算范数的伸缩因子
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, '迭代', '比值')
```

观察上述计算的最后一部分可以发现，
随机向量的伸缩因子为`1.974459321485[...]`。
末尾几位略有变化，
但伸缩因子已经稳定下来。

### 回到特征向量

我们已经看到，特征向量和特征值对应于对象被伸缩的程度，
但前面的讨论针对的是特定向量和特定伸缩。
下面看看$\mathbf{A}$的情形。
这里需要说明一点：为了看到全部特征值，
我们需要使用复数。
可以把复数特征值理解为伸缩和旋转。
通过计算复数的模
（实部与虚部的平方和再开平方），
可以度量伸缩因子。我们还将对这些值排序。

```{.python .input}
#@tab mxnet
# 计算特征值
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'特征值的模：{norm_eigs}')
```

```{.python .input}
#@tab pytorch
# 计算特征值
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'特征值的模：{norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# 计算特征值
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'特征值的模：{norm_eigs}')
```

### 一个观察结果

这里出现了一个有些出人意料的现象：
前面得到的矩阵$\mathbf{A}$作用于随机向量时的长期伸缩因子，
*恰好*就是$\mathbf{A}$的最大特征值
（精确到小数点后十三位）。
这显然不是巧合。

从几何角度思考正在发生的事情，这一现象就容易理解了。考虑一个随机向量，
它在每个方向上都含有一定分量，
因而至少有一小部分与$\mathbf{A}$的最大特征值所对应的特征向量同向。
这个特征值和特征向量非常重要，分别称为*主特征值*（principal eigenvalue）
和*主特征向量*（principal eigenvector）。
应用$\mathbf{A}$后，随机向量会沿各个特征向量所对应的方向伸缩，
但沿主特征向量方向的伸缩最大。
这意味着应用$A$后，随机向量变得更长，
其方向也更接近主特征向量。
反复应用矩阵后，它会与主特征向量越来越对齐，
直到在实际意义上，随机向量已经变成主特征向量。
事实上，这一过程正是*幂迭代*（power iteration）的基础，
该算法用于求矩阵的最大特征值和对应的特征向量。有关详细内容，可参见 :cite:`Golub.Van-Loan.1996`。

### 修正归一化

根据以上讨论，我们不希望随机向量被持续拉伸或压缩，
而希望它在整个过程中大致保持相同大小。
为此，用主特征值重新缩放矩阵，
使最大的特征值变为$1$。
下面观察此时会发生什么。

```{.python .input}
#@tab mxnet
# 重新缩放矩阵`A`
A /= norm_eigs[-1]

# 再次执行相同实验
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, '迭代', '值')
```

```{.python .input}
#@tab pytorch
# 重新缩放矩阵`A`
A /= norm_eigs[-1]

# 再次执行相同实验
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, '迭代', '值')
```

```{.python .input}
#@tab tensorflow
# 重新缩放矩阵`A`
A /= norm_eigs[-1]

# 再次执行相同实验
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, '迭代', '值')
```

还可以像前面一样绘制相邻范数之比，并观察到它确实趋于稳定。

```{.python .input}
#@tab mxnet
# 同时绘制比值
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, '迭代', '比值')
```

```{.python .input}
#@tab pytorch
# 同时绘制比值
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, '迭代', '比值')
```

```{.python .input}
#@tab tensorflow
# 同时绘制比值
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, '迭代', '比值')
```

## 讨论

现在得到的结果正是我们所希望的。
使用主特征值对矩阵归一化后，
随机数据不再像之前那样爆炸，
而是最终稳定在某个特定值附近。
如果能从基本原理出发完成这种处理当然更好。
深入研究相关数学可以发现：对于一个大型随机矩阵，
若其元素相互独立且服从均值为$0$、方差为$1$的高斯分布，
其最大特征值平均约为$\sqrt{n}$；
在本例中即$\sqrt{5}\approx2.2$。
这源于一个称为*圆律*（circular law）的迷人结论 :cite:`Ginibre.1965`。
研究表明，随机矩阵的特征值（以及一个相关对象——奇异值）
与神经网络的恰当初始化存在深刻联系，相关讨论可见 :citet:`Pennington.Schoenholz.Ganguli.2017` 及后续工作。

## 小结
* 特征向量是被矩阵伸缩后方向保持不变的向量。
* 特征值表示矩阵作用后特征向量的伸缩倍数。
* 矩阵的特征分解可以把许多运算简化为对特征值的运算。
* Gershgorin圆盘定理可以给出矩阵特征值的近似范围。
* 矩阵幂的迭代行为主要取决于最大特征值的大小。这一认识在神经网络初始化理论中有许多应用。

## 练习
1. 求下列矩阵的特征值和特征向量：
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1. 求下列矩阵的特征值和特征向量。与上一个例子相比，这个例子有什么奇特之处？
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. 不计算特征值，下列矩阵的最小特征值是否可能小于$0.5$？*提示*：这道题可以心算完成。
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1087)
:end_tab:
