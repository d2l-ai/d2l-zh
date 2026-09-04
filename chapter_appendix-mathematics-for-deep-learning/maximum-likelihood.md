# 最大似然
:label:`sec_maximum_likelihood`

最大似然是机器学习中最常见的思考方式之一。其基本思想是：使用参数未知的概率模型时，使数据出现概率最大的参数就是最有可能的参数。

## 最大似然原理

这一思想可以用贝叶斯方法解释，有助于我们理解它。假设模型的参数为$\boldsymbol{\theta}$，数据样本的集合为$X$。具体而言，可以设想$\boldsymbol{\theta}$是一个标量，表示抛硬币时正面朝上的概率，而$X$是一系列相互独立的抛硬币结果。稍后我们会深入研究这个例子。

如果要找出模型参数最可能的取值，就需要求

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

根据贝叶斯定理，这等价于

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

表达式$P(X)$是在不考虑参数的情况下生成数据的概率，它完全不依赖$\boldsymbol{\theta}$，因此去掉这一项不会改变$\boldsymbol{\theta}$的最优选择。类似地，可以假设我们事先并不认为某组参数优于其他参数，也就是说$P(\boldsymbol{\theta})$同样不依赖$\boldsymbol{\theta}$。例如，在抛硬币的例子中，正面朝上的概率可以是$[0,1]$内的任意值，而我们事先并不认为硬币一定公平或不公平，这通常称为*无信息先验*（uninformative prior）。由此可见，应用贝叶斯定理后，$\boldsymbol{\theta}$的最佳选择就是其最大似然估计：

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

按照常用术语，给定参数时数据出现的概率$P(X \mid \boldsymbol{\theta})$称为*似然*（likelihood）。

### 一个具体例子

下面通过一个具体例子说明其工作原理。假设只有一个参数$\theta$，表示抛硬币得到正面的概率，那么得到反面的概率就是$1-\theta$。如果观测数据$X$是一个序列，其中包含$n_H$次正面和$n_T$次反面，利用独立事件的概率相乘可得

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

如果抛硬币$13$次，得到序列“HHHTHTTHHHHHT”，其中$n_H=9$且$n_T=4$，那么

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

这个例子的一个优点是，我们一开始就知道答案。事实上，如果问：“我抛了$13$次硬币，其中$9$次正面朝上，那么对硬币正面朝上概率的最佳猜测是什么？”大家都会正确猜出$9/13$。最大似然方法将从基本原理出发得到这个数值，而且这种方法可以推广到复杂得多的情形。

在本例中，$P(X \mid \theta)$的图像如下：

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

它在我们预期的$9/13\approx 0.7\ldots$附近取得最大值。为了判断最大值是否恰好位于此处，可以借助微积分。函数在最大值处的切线斜率为零。因此，可以先找出使导数为零的$\theta$，再从中选出使概率最大的取值，从而求得 :eqref:`eq_max_like` 中的最大似然估计。计算如下：

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

这个方程有三个解：$0$、$1$和$9/13$。前两个解给观测序列赋予的概率为$0$，显然是最小值而不是最大值。最后一个取值给观测序列赋予的概率*不为零*，因此必然是最大似然估计$\hat \theta=9/13$。

## 数值优化与负对数似然

前面的例子很简单，但如果有数十亿个参数和数据样本，该怎么办？

首先，如果假设所有数据样本相互独立，那么似然就是许多概率的乘积，实际计算时无法直接处理。每个概率都在$[0,1]$内，例如其典型值约为$1/2$，而$(1/2)^{1000000000}$远低于机器精度，无法直接进行计算。

然而，对数可以把乘积转化为求和，此时

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

即使使用$32$位单精度浮点数，也完全可以表示这个数。因此，我们应考虑*对数似然*（log-likelihood），即

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

由于函数$x\mapsto\log(x)$单调递增，最大化似然等价于最大化对数似然。在 :numref:`sec_naive_bayes` 中讨论朴素贝叶斯分类器时，我们会看到这一推理的具体应用。

我们通常使用损失函数，并希望将损失最小化。取$-\log(P(X \mid \boldsymbol{\theta}))$，即可把最大似然问题转化为损失最小化问题；这个量称为*负对数似然*（negative log-likelihood）。

为说明这一点，再考虑前面的抛硬币问题，并假装我们不知道其闭式解。可以计算出

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

这个表达式可以写成代码；即使面对数十亿次抛硬币结果，也可以顺利进行优化。

```{.python .input}
#@tab mxnet
# 设置数据
n_H = 8675309
n_T = 256245

# 初始化参数
theta = np.array(0.5)
theta.attach_grad()

# 执行梯度下降
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# 检查输出
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# 设置数据
n_H = 8675309
n_T = 256245

# 初始化参数
theta = torch.tensor(0.5, requires_grad=True)

# 执行梯度下降
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# 检查输出
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# 设置数据
n_H = 8675309
n_T = 256245

# 初始化参数
theta = tf.Variable(tf.constant(0.5))

# 执行梯度下降
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# 检查输出
theta, n_H / (n_H + n_T)
```

数值计算方便并不是人们喜欢使用负对数似然的唯一原因。它还有其他几个优势。



考虑对数似然的第二个原因是它简化了微积分法则的应用。如上所述，由于独立性假设，机器学习中遇到的大多数概率都是各个概率的乘积。

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

这意味着，如果直接应用乘积法则求导，将得到

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

这需要$n(n-1)$次乘法和$(n-1)$次加法，因此计算时间与输入规模的平方成正比。通过巧妙地组合各项，可以将其降至线性时间，但这需要额外思考。对于负对数似然，则有

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

由此得到

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

这只需要$n$次除法和$n-1$次加法，因此计算时间与输入规模呈线性关系。

考虑负对数似然的第三个也是最后一个原因，是它与信息论之间的关系，我们将在 :numref:`sec_information_theory` 中详细讨论。信息论是一套严谨的数学理论，提供了度量随机变量所含信息或随机程度的方法。该领域的核心研究对象是熵：

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

它度量信息源的随机性。注意，这不过是$-\log$概率的平均值。因此，将负对数似然除以数据样本数，就会得到熵的一个近亲——交叉熵。仅凭这一理论解释，就足以说明为什么可以报告整个数据集上的平均负对数似然来衡量模型性能。

## 连续变量的最大似然

到目前为止，我们所做的一切都假设处理的是离散随机变量，但如果要处理连续随机变量，该怎么办？

简而言之，除了把所有概率替换为概率密度之外，其他内容完全不变。回顾一下，我们用小写$p$表示密度，因此现在可以写成

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

问题变成了：“为什么这样做是合理的？”毕竟，我们引入密度的原因正是连续随机变量取得某个特定结果的概率为零。那么，对于任意一组参数，生成观测数据的概率不都是零吗？

事实确实如此。要理解为什么可以转而使用密度，需要追踪$\epsilon$在推导中的变化。

首先重新定义目标。对于连续随机变量，我们不再计算恰好取得正确数值的概率，而是计算落在某个宽度为$\epsilon$的范围内的概率。为简单起见，假设数据是同分布随机变量$X_1,\ldots,X_N$的重复观测$x_1,\ldots,x_N$。根据前面的讨论，可以写成

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

因此，对它取负对数可得

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

考察这个表达式可以发现，$\epsilon$只出现在加法常数$-N\log(\epsilon)$中。该项完全不依赖参数$\boldsymbol{\theta}$，所以$\boldsymbol{\theta}$的最优选择也不依赖$\epsilon$。无论要求四位精度还是四百位精度，$\boldsymbol{\theta}$的最佳选择都保持不变。因此可以去掉含$\epsilon$的项，得到需要优化的目标：

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

由此可见，只需用概率密度替换概率，最大似然方法就能像处理离散随机变量一样处理连续随机变量。

## 小结
* 最大似然原理告诉我们：对于给定的数据集，拟合得最好的模型是以最高概率生成这些数据的模型。
* 人们通常改用负对数似然，原因包括数值稳定性、将乘积转化为求和（从而简化梯度计算），以及它与信息论之间的理论联系。
* 最大似然最容易在离散情形下得到解释，但通过最大化数据点上的概率密度，也可以直接推广到连续情形。

## 练习
1. 已知某个非负随机变量的密度为$\alpha e^{-\alpha x}$，其中$\alpha>0$。从该随机变量得到一个观测值$3$。$\alpha$的最大似然估计是多少？
2. 假设数据集中的样本$\{x_i\}_{i=1}^N$来自均值未知、方差为$1$的高斯分布。均值的最大似然估计是多少？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1097)
:end_tab:
