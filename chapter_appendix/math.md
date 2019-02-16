# 数学基础

本节总结了本书中涉及的有关线性代数、微分和概率的基础知识。为避免赘述本书未涉及的数学背景知识，本节中的少数定义稍有简化。


## 线性代数

下面分别概括了向量、矩阵、运算、范数、特征向量和特征值的概念。

### 向量

本书中的向量指的是列向量。一个$n$维向量$\boldsymbol{x}$的表达式可写成

$$
\boldsymbol{x} = 
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{n} 
\end{bmatrix},
$$

其中$x_1, \ldots, x_n$是向量的元素。我们将各元素均为实数的$n$维向量$\boldsymbol{x}$记作$\boldsymbol{x} \in \mathbb{R}^{n}$或$\boldsymbol{x} \in \mathbb{R}^{n \times 1}$。


### 矩阵

一个$m$行$n$列矩阵的表达式可写成

$$
\boldsymbol{X} = 
\begin{bmatrix}
    x_{11} & x_{12}  & \dots  & x_{1n} \\
    x_{21} & x_{22}  & \dots  & x_{2n} \\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} & x_{m2}  & \dots  & x_{mn}
\end{bmatrix},
$$

其中$x_{ij}$是矩阵$\boldsymbol{X}$中第$i$行第$j$列的元素（$1 \leq i \leq m, 1 \leq j \leq n$）。我们将各元素均为实数的$m$行$n$列矩阵$\boldsymbol{X}$记作$\boldsymbol{X} \in \mathbb{R}^{m \times n}$。不难发现，向量是特殊的矩阵。


### 运算

设$n$维向量$\boldsymbol{a}$中的元素为$a_1, \ldots, a_n$，$n$维向量$\boldsymbol{b}$中的元素为$b_1, \ldots, b_n$。向量$\boldsymbol{a}$与$\boldsymbol{b}$的点乘（内积）是一个标量：

$$\boldsymbol{a} \cdot \boldsymbol{b} = a_1 b_1 + \ldots + a_n b_n.$$


设两个$m$行$n$列矩阵

$$
\boldsymbol{A} = 
\begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1n} \\
    a_{21} & a_{22} & \dots  & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \dots  & a_{mn}
\end{bmatrix},\quad
\boldsymbol{B} = 
\begin{bmatrix}
    b_{11} & b_{12} & \dots  & b_{1n} \\
    b_{21} & b_{22} & \dots  & b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    b_{m1} & b_{m2} & \dots  & b_{mn}
\end{bmatrix}.
$$

矩阵$\boldsymbol{A}$的转置是一个$n$行$m$列矩阵，它的每一行其实是原矩阵的每一列：
$$
\boldsymbol{A}^\top = 
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$


两个相同形状的矩阵的加法是将两个矩阵按元素做加法：

$$
\boldsymbol{A} + \boldsymbol{B} = 
\begin{bmatrix}
    a_{11} + b_{11} & a_{12} + b_{12} & \dots  & a_{1n} + b_{1n} \\
    a_{21} + b_{21} & a_{22} + b_{22} & \dots  & a_{2n} + b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} + b_{m1} & a_{m2} + b_{m2} & \dots  & a_{mn} + b_{mn}
\end{bmatrix}.
$$

我们使用符号$\odot$表示两个矩阵按元素做乘法的运算：

$$
\boldsymbol{A} \odot \boldsymbol{B} = 
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

定义一个标量$k$。标量与矩阵的乘法也是按元素做乘法的运算：


$$
k\boldsymbol{A} = 
\begin{bmatrix}
    ka_{11} & ka_{12} & \dots  & ka_{1n} \\
    ka_{21} & ka_{22} & \dots  & ka_{2n} \\
    \vdots & \vdots   & \ddots & \vdots \\
    ka_{m1} & ka_{m2} & \dots  & ka_{mn}
\end{bmatrix}.
$$

其他诸如标量与矩阵按元素相加、相除等运算与上式中的相乘运算类似。矩阵按元素开根号、取对数等运算也就是对矩阵每个元素开根号、取对数等，并得到和原矩阵形状相同的矩阵。

矩阵乘法和按元素的乘法不同。设$\boldsymbol{A}$为$m$行$p$列的矩阵，$\boldsymbol{B}$为$p$行$n$列的矩阵。两个矩阵相乘的结果

$$
\boldsymbol{A} \boldsymbol{B} = 
\begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1p} \\
    a_{21} & a_{22} & \dots  & a_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \dots  & a_{ip} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \dots  & a_{mp}
\end{bmatrix}
\begin{bmatrix}
    b_{11} & b_{12} & \dots  & b_{1j} & \dots & b_{1n} \\
    b_{21} & b_{22} & \dots  & b_{2j} & \dots  & b_{2n} \\
    \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
    b_{p1} & b_{p2} & \dots  & b_{pj} & \dots  & b_{pn}
\end{bmatrix}
$$

是一个$m$行$n$列的矩阵，其中第$i$行第$j$列（$1 \leq i \leq m, 1 \leq j \leq n$）的元素为

$$a_{i1}b_{1j}  + a_{i2}b_{2j} + \ldots + a_{ip}b_{pj} = \sum_{k=1}^p a_{ik}b_{kj}. $$


### 范数

设$n$维向量$\boldsymbol{x}$中的元素为$x_1, \ldots, x_n$。向量$\boldsymbol{x}$的$L_p$范数为

$$\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

例如，$\boldsymbol{x}$的$L_1$范数是该向量元素绝对值之和：

$$\|\boldsymbol{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

而$\boldsymbol{x}$的$L_2$范数是该向量元素平方和的平方根：

$$\|\boldsymbol{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$

我们通常用$\|\boldsymbol{x}\|$指代$\|\boldsymbol{x}\|_2$。

设$\boldsymbol{X}$是一个$m$行$n$列矩阵。矩阵$\boldsymbol{X}$的Frobenius范数为该矩阵元素平方和的平方根：

$$\|\boldsymbol{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2},$$

其中$x_{ij}$为矩阵$\boldsymbol{X}$在第$i$行第$j$列的元素。


### 特征向量和特征值


对于一个$n$行$n$列的矩阵$\boldsymbol{A}$，假设有标量$\lambda$和非零的$n$维向量$\boldsymbol{v}$使

$$\boldsymbol{A} \boldsymbol{v} = \lambda \boldsymbol{v},$$

那么$\boldsymbol{v}$是矩阵$\boldsymbol{A}$的一个特征向量，标量$\lambda$是$\boldsymbol{v}$对应的特征值。



## 微分

我们在这里简要介绍微分的一些基本概念和演算。


### 导数和微分

假设函数$f: \mathbb{R} \rightarrow \mathbb{R}$的输入和输出都是标量。函数$f$的导数

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$

且假定该极限存在。给定$y = f(x)$，其中$x$和$y$分别是函数$f$的自变量和因变量。以下有关导数和微分的表达式等价：

$$f'(x) = y' = \frac{\text{d}y}{\text{d}x} = \frac{\text{d}f}{\text{d}x} = \frac{\text{d}}{\text{d}x} f(x) = \text{D}f(x) = \text{D}_x f(x),$$

其中符号$\text{D}$和$\text{d}/\text{d}x$也叫微分运算符。常见的微分演算有$\text{D}C = 0$（$C$为常数）、$\text{D}x^n = nx^{n-1}$（$n$为常数）、$\text{D}e^x = e^x$、$\text{D}\ln(x) = 1/x$等。

如果函数$f$和$g$都可导，设$C$为常数，那么

$$
\begin{aligned}
\frac{\text{d}}{\text{d}x} [Cf(x)] &= C \frac{\text{d}}{\text{d}x} f(x),\\
\frac{\text{d}}{\text{d}x} [f(x) + g(x)] &= \frac{\text{d}}{\text{d}x} f(x) + \frac{\text{d}}{\text{d}x} g(x),\\ 
\frac{\text{d}}{\text{d}x} [f(x)g(x)] &= f(x) \frac{\text{d}}{\text{d}x} [g(x)] + g(x) \frac{\text{d}}{\text{d}x} [f(x)],\\
\frac{\text{d}}{\text{d}x} \left[\frac{f(x)}{g(x)}\right] &= \frac{g(x) \frac{\text{d}}{\text{d}x} [f(x)] - f(x) \frac{\text{d}}{\text{d}x} [g(x)]}{[g(x)]^2}.
\end{aligned}
$$


如果$y=f(u)$和$u=g(x)$都是可导函数，依据链式法则，

$$\frac{\text{d}y}{\text{d}x} = \frac{\text{d}y}{\text{d}u} \frac{\text{d}u}{\text{d}x}.$$


### 泰勒展开

函数$f$的泰勒展开式是

$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!} (x-a)^n,$$

其中$f^{(n)}$为函数$f$的$n$阶导数（求$n$次导数），$n!$为$n$的阶乘。假设$\epsilon$是一个足够小的数，如果将上式中$x$和$a$分别替换成$x+\epsilon$和$x$，可以得到

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2).$$

由于$\epsilon$足够小，上式也可以简化成

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$



### 偏导数

设$u$为一个有$n$个自变量的函数，$u = f(x_1, x_2, \ldots, x_n)$，它有关第$i$个变量$x_i$的偏导数为

$$ \frac{\partial u}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


以下有关偏导数的表达式等价：

$$\frac{\partial u}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = \text{D}_i f = \text{D}_{x_i} f.$$

为了计算$\partial u/\partial x_i$，只需将$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$视为常数并求$u$有关$x_i$的导数。



### 梯度


假设函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$的输入是一个$n$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_n]^\top$，输出是标量。函数$f(\boldsymbol{x})$有关$\boldsymbol{x}$的梯度是一个由$n$个偏导数组成的向量：

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_n}\bigg]^\top.$$


为表示简洁，我们有时用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。

假设$\boldsymbol{x}$是一个向量，常见的梯度演算包括

$$
\begin{aligned}
\nabla_{\boldsymbol{x}} \boldsymbol{A}^\top \boldsymbol{x} &= \boldsymbol{A}, \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A}  &= \boldsymbol{A}, \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}  &= (\boldsymbol{A} + \boldsymbol{A}^\top)\boldsymbol{x},\\
\nabla_{\boldsymbol{x}} \|\boldsymbol{x} \|^2 &= \nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{x} = 2\boldsymbol{x}.
\end{aligned}
$$

类似地，假设$\boldsymbol{X}$是一个矩阵，那么
$$\nabla_{\boldsymbol{X}} \|\boldsymbol{X} \|_F^2 = 2\boldsymbol{X}.$$




### 海森矩阵

假设函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$的输入是一个$n$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_n]^\top$，输出是标量。假定函数$f$所有的二阶偏导数都存在，$f$的海森矩阵$\boldsymbol{H}$是一个$n$行$n$列的矩阵：

$$
\boldsymbol{H} = 
\begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots  & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix},
$$

其中二阶偏导数

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial }{\partial x_j} \left(\frac{\partial f}{ \partial x_i}\right).$$



## 概率

最后，我们简要介绍条件概率、期望和均匀分布。

### 条件概率

假设事件$A$和事件$B$的概率分别为$P(A)$和$P(B)$，两个事件同时发生的概率记作$P(A \cap B)$或$P(A, B)$。给定事件$B$，事件$A$的条件概率

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}.$$

也就是说，

$$P(A \cap B) = P(B) P(A \mid B) = P(A) P(B \mid A).$$

当满足

$$P(A \cap B) = P(A) P(B)$$

时，事件$A$和事件$B$相互独立。


### 期望

离散的随机变量$X$的期望（或平均值）为

$$E(X) = \sum_{x} x P(X = x).$$



### 均匀分布

假设随机变量$X$服从$[a, b]$上的均匀分布，即$X \sim U(a, b)$。随机变量$X$取$a$和$b$之间任意一个数的概率相等。




## 小结

* 本节总结了本书中涉及的有关线性代数、微分和概率的基础知识。


## 练习

* 求函数$f(\boldsymbol{x}) = 3x_1^2 + 5e^{x_2}$的梯度。




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6966)

![](../img/qr_math.svg)
