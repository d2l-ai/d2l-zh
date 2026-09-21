# 概率分布
:label:`sec_distributions`

现在我们已经学习了如何处理离散和连续情形下的概率，下面来认识一些常见的概率分布。根据所研究的机器学习领域，我们可能需要熟悉更多分布；而在深度学习的某些领域，也可能完全用不到它们。不过，下面列出的分布都是值得掌握的基础知识。首先导入一些常用库。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # 在torch中定义圆周率
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # 在TensorFlow中定义圆周率
```

## 伯努利分布

这是通常会遇到的最简单的随机变量。它描述一次抛硬币：以概率$p$取值$1$，以概率$1-p$取值$0$。若随机变量$X$服从该分布，记作

$$
X \sim \textrm{Bernoulli}(p).
$$

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x \ge 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

其概率质量函数如下图所示。

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

下面绘制累积分布函数 :eqref:`eq_bernoulli_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

若$X\sim\textrm{Bernoulli}(p)$，则：

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

可以按如下方式从伯努利随机变量中采样任意形状的数组。

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 离散均匀分布

下一个常见的随机变量服从离散均匀分布。这里假设其支持集为整数$\{1,2,\ldots,n\}$，当然也可以任意选择其他取值集合。此处“均匀”的含义是每个可能取值出现的概率都相同。对于每个$i\in\{1,2,3,\ldots,n\}$，其概率为$p_i=\frac{1}{n}$。若随机变量$X$服从该分布，记作

$$
X \sim U(n).
$$

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1,\ \text{其中 }1 \le k < n, \\ 1 & x \ge n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

首先绘制概率质量函数。

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

下面绘制累积分布函数 :eqref:`eq_discrete_uniform_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

若$X\sim U(n)$，则：

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

可以按如下方式从离散均匀随机变量中采样任意形状的数组。

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 连续均匀分布

接下来讨论连续均匀分布。其基本思想是：不断增大离散均匀分布中的$n$，再将其缩放到区间$[a,b]$内，就会趋近于一个连续随机变量，它以均等的方式从$[a,b]$中选取任意值。记该分布为

$$
X \sim U(a, b).
$$

其概率密度函数为

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x \ge b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

首先绘制概率密度函数 :eqref:`eq_cont_uniform_pdf`。

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

下面绘制累积分布函数 :eqref:`eq_cont_uniform_cdf`。

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

若$X\sim U(a,b)$，则：

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

可以按如下方式从均匀随机变量中采样任意形状的数组。注意，默认情况下采样来自$U(0,1)$；若需要其他范围，则必须进行缩放。

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 二项分布

下面稍微增加一点复杂度，考察*二项*（binomial）随机变量。它来自$n$次相互独立的试验，每次试验成功的概率均为$p$；我们关心一共会出现多少次成功。

下面用数学语言表述。每次试验都是一个独立随机变量$X_i$，其中用$1$表示成功，用$0$表示失败。每次试验都像一次独立抛硬币，以概率$p$成功，所以$X_i\sim\textrm{Bernoulli}(p)$。二项随机变量为

$$
X = \sum_{i=1}^n X_i.
$$

在这种情况下，记作

$$
X \sim \textrm{Binomial}(n, p).
$$

为了得到累积分布函数，需要注意恰好成功$k$次共有$\binom{n}{k}=\frac{n!}{k!(n-k)!}$种方式，而每种方式发生的概率都是$p^k(1-p)^{n-k}$。因此累积分布函数为

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1,\ \text{其中 }0 \le k < n, \\ 1 & x \ge n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

首先绘制概率质量函数。

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# 计算二项式系数
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# 计算二项式系数
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# 计算二项式系数
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

下面绘制累积分布函数 :eqref:`eq_binomial_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

若$X\sim\textrm{Binomial}(n,p)$，则：

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

这一结论来自期望对$n$个伯努利随机变量之和的线性性，以及相互独立随机变量之和的方差等于各方差之和这一事实。可以按如下方式采样。

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 泊松分布
现在进行一个思想实验。假设我们站在公交车站，想知道下一分钟会有多少辆公交车到站。首先考虑$X^{(1)}\sim\textrm{Bernoulli}(p)$，它表示一分钟内有一辆公交车到站的概率。对于远离市中心的公交车站，这可能是相当合理的近似：一分钟内或许永远不会有一辆以上的公交车到站。

然而，在繁忙地区，两辆公交车到站是可能的，甚至很常见。可以把一分钟分成前$30$秒和后$30$秒，并把随机变量拆成两部分来建模。此时可以写成

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

其中$X^{(2)}$是总和，且$X^{(2)}_i\sim\textrm{Bernoulli}(p/2)$。于是总数服从$X^{(2)}\sim\textrm{Binomial}(2,p/2)$。

还可以继续把一分钟划分为$n$段。根据相同的推理可得

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

考察这些随机变量。根据上一节， :eqref:`eq_eq_poisson_approx` 的均值为$\mu_{X^{(n)}}=n(p/n)=p$，方差为$\sigma_{X^{(n)}}^2=n(p/n)(1-(p/n))=p(1-p/n)$。当$n\rightarrow\infty$时，这两个量分别稳定为$\mu_{X^{(\infty)}}=p$和$\sigma_{X^{(\infty)}}^2=p$。这表明，在无限细分的极限下，*可能*存在一个定义良好的随机变量。

这并不令人意外，因为在现实世界中可以直接数出到站公交车的数量；不过，看到数学模型具有良好定义仍然很有意义。上述讨论可以形式化为*稀有事件定律*（law of rare events）。

仔细完成上述推导，可以得到以下模型。若随机变量$X$以如下概率取值于$\{0,1,2,\ldots\}$，则称$X\sim\textrm{Poisson}(\lambda)$：

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

$\lambda>0$称为*速率*（rate，也称*形状参数*），表示单位时间内预期到达的平均次数。

对概率质量函数求和即可得到累积分布函数。

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1,\ \text{其中 }0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

首先绘制概率质量函数 :eqref:`eq_poisson_mass`。

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

下面绘制累积分布函数 :eqref:`eq_poisson_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

如上所示，其均值和方差格外简洁。若$X\sim\textrm{Poisson}(\lambda)$，则：

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

可以按如下方式采样。

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## 高斯分布
下面尝试另一个有所关联的实验。假设再次进行$n$次相互独立的$\textrm{Bernoulli}(p)$测量$X_i$，它们的和服从$X^{(n)}\sim\textrm{Binomial}(n,p)$。这一次不让$n$增大而$p$减小，而是固定$p$并令$n\rightarrow\infty$。此时$\mu_{X^{(n)}}=np\rightarrow\infty$且$\sigma_{X^{(n)}}^2=np(1-p)\rightarrow\infty$，所以没有理由认为这一极限具有良好定义。

不过，问题仍有解决的希望。定义

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

容易看出，该变量的均值为$0$、方差为$1$，因此有理由相信它会收敛到某个极限分布。绘制这些分布的图像后，这一判断会更加可信。

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

需要注意的是，与泊松分布的情形相比，这里除以了标准差，意味着所有可能结果被压缩到越来越小的区域中。这表明最终得到的极限不再是离散分布，而是连续分布。

完整推导超出了本文的范围，不过*中心极限定理*（central limit theorem）指出，当$n\rightarrow\infty$时，这一过程会得到高斯分布（也称正态分布）。更明确地说，对于任意$a,b$：

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

如果随机变量$X$具有如下密度，则称它服从均值为$\mu$、方差为$\sigma^2$的正态分布，记作$X\sim\mathcal{N}(\mu,\sigma^2)$：

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

首先绘制概率密度函数 :eqref:`eq_gaussian_pdf`。

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

下面绘制累积分布函数。高斯累积分布函数无法用更初等的函数写成闭式公式，对此的讨论超出了本附录的范围。我们将使用`erf`对该积分进行数值计算。

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

细心的读者或许会认出其中一些表达式。我们确实在 :numref:`sec_integral_calculus` 中遇到过这个积分，也正是通过该计算才能看出$p_X(x)$下的总面积为$1$，因而它是有效的概率密度。

选择抛硬币只是为了缩短计算，这一选择并非本质要求。事实上，可以取任意一组独立同分布随机变量$X_i$，并构造

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

Then

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

它将近似服从高斯分布。要使这一结论成立还需要附加条件，最常见的是$E[X^4]<\infty$，但其基本思想已经很清楚。

中心极限定理说明了为什么高斯分布是概率论、统计学和机器学习的基础。只要某个测量量可以看成许多微小、独立贡献之和，就可以认为该测量量近似服从高斯分布。

高斯分布还有许多有趣的性质，这里再讨论一个。高斯分布是一种*最大熵分布*（maximum entropy distribution）。我们将在 :numref:`sec_information_theory` 中深入讨论熵，目前只需知道它是对随机性的一种度量。从严格的数学意义上说，在均值和方差固定的所有随机变量中，高斯分布是*最*随机的选择。因此，如果只知道随机变量的均值和方差，那么从某种意义上说，高斯分布是最保守的分布选择。

最后回顾一下，若$X\sim\mathcal{N}(\mu,\sigma^2)$，则：

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

可以按如下方式从高斯分布（或标准正态分布）中采样。

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 指数族
:label:`subsec_exponential_family`

上述所有分布都有一个共同性质：它们都属于*指数族*（exponential family）。
指数族是一组密度可以表示为如下形式的分布：

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

这个定义有些微妙，下面仔细考察其中各项。

首先，$h(\mathbf{x})$称为*基础测度*（underlying measure或base measure）。
可以把它看成最初选定的测度，我们再用指数权重对其进行修改。

其次，向量$\boldsymbol{\eta}=(\eta_1,\eta_2,...,\eta_l)\in\mathbb{R}^l$
称为*自然参数*（natural parameter）或*规范参数*（canonical parameter），
它定义了如何修改基础测度。将这些参数与
$\mathbf{x}=(x_1,x_2,...,x_n)\in\mathbb{R}^n$的某个函数$T(\cdot)$作点积，
再取指数，自然参数就进入了新的测度。向量
$T(\mathbf{x})=(T_1(\mathbf{x}),T_2(\mathbf{x}),...,T_l(\mathbf{x}))$
称为$\boldsymbol{\eta}$的*充分统计量*（sufficient statistic）。之所以这样命名，
是因为$T(\mathbf{x})$所表示的信息足以计算概率密度，
不再需要样本$\mathbf{x}$中的其他信息。

第三，$A(\boldsymbol{\eta})$称为*累积量函数*（cumulant function），
它保证上述分布 :eqref:`eq_exp_pdf` 的积分为$1$，即

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

具体而言，考虑高斯分布。假设$\mathbf{x}$是单变量，前面已经看到其密度为

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

这与指数族的定义吻合，其中：

* *基础测度*：$h(x) = \frac{1}{\sqrt{2 \pi}}$；
* *自然参数*：$\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$；
* *充分统计量*：$T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$；
* *累积量函数*：$A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

值得注意的是，上述各项的具体选择具有一定任意性。
真正重要的性质是分布能够表示成这种形式，而不是每一项的确切形式。

正如 :numref:`subsec_softmax_and_derivatives` 中所提到的，一种广泛使用的方法是
假设最终输出$\mathbf{y}$服从指数族分布。指数族是机器学习中经常遇到的一类
常见而强大的分布。


## 小结
* 伯努利随机变量可以对结果为“是”或“否”的事件建模。
* 离散均匀分布对从有限个可能值中进行选择建模。
* 连续均匀分布对从一个区间中进行选择建模。
* 二项分布对一系列伯努利随机变量建模，并统计成功次数。
* 泊松随机变量对稀有事件的到达次数建模。
* 高斯随机变量对大量独立随机变量之和建模。
* 上述所有分布都属于指数族。

## 练习

1. 随机变量为两个相互独立的二项随机变量$X,Y\sim\textrm{Binomial}(16,1/2)$之差$X-Y$，求其标准差。
2. 对泊松随机变量$X\sim\textrm{Poisson}(\lambda)$，当$\lambda\rightarrow\infty$时，可以证明$(X-\lambda)/\sqrt{\lambda}$近似服从高斯分布。为什么这一结论是合理的？
3. 两个取值数均为$n$的离散均匀随机变量之和的概率质量函数是什么？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1099)
:end_tab:
