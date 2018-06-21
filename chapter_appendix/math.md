# 数学基础

本节总结了本书中涉及到的线性代数、微分和概率统计的基础知识。


## 线性代数

以下概括了向量、矩阵、运算、范数、特征向量和特征值的概念。

### 向量

本书中的向量指的是列向量。设$d$维向量

$$
\boldsymbol{x} = 
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{d} 
\end{bmatrix},
$$

其中$x_1, \ldots, x_d$是向量的元素。我们将各元素均为实数的$d$维向量$\boldsymbol{x}$记作$\boldsymbol{x} \in \mathbb{R}^{d}$或$\boldsymbol{x} \in \mathbb{R}^{d \times 1}$。


### 矩阵

设$m$行$n$列矩阵

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

设$d$维向量$\boldsymbol{a}$中的元素为$a_1, \ldots, a_d$，$d$维向量$\boldsymbol{b}$中的元素为$b_1, \ldots, b_d$。向量$\boldsymbol{a}$与$\boldsymbol{b}$的点乘（内积）是一个标量：

$$\boldsymbol{a} \cdot \boldsymbol{b} = a_1 b_1 + \ldots + a_d b_d.$$


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


两个相同形状的矩阵的加法实际上是按元素做加法：

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
    ka_{11} & ka_{21} & \dots  & ka_{m1} \\
    ka_{12} & ka_{22} & \dots  & ka_{m2} \\
    \vdots & \vdots   & \ddots & \vdots \\
    ka_{1n} & ka_{2n} & \dots  & ka_{mn}
\end{bmatrix}.
$$

其它例如标量与矩阵按元素相加、相除等运算与上式中的运算类似。矩阵按元素开根号、取对数等运算也即对矩阵每个元素开根号、取对数等，并得到和原矩阵形状相同的矩阵。


设$\boldsymbol{A}$为$m$行$p$列的矩阵，$\boldsymbol{B}$为$p$行$n$列的矩阵。两个矩阵相乘的结果

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


需要注意区分矩阵乘法和矩阵的按元素乘法。


TODO(@astonzhang)


### 范数



### 特征向量和特征值




## 微分


### 导数和梯度


### 常用梯度


### 泰勒展开


### 黑塞矩阵




## 概率和统计


### 全概率


### 条件概率


### 最大似然估计







## MOVE




### 向量和$L_p$范数




向量$\boldsymbol{x}$的$L_p$范数为

$$\|\boldsymbol{x}\|_p = (\sum_{i=1}^d |x_i|^p)^{1/p}.$$

例如$\boldsymbol{x}$的$L_1$范数是该向量元素绝对值的和：

$$\|\boldsymbol{x}\|_1 = \sum_{i=1}^d |x_i|.$$

而$\boldsymbol{x}$的$L_2$范数是该向量元素平方和的平方根：

$$\|\boldsymbol{x}\|_2 = \sqrt{\sum_{i=1}^d x_i^2}.$$

我们通常用$\|\boldsymbol{x}\|$指代$\|\boldsymbol{x}\|_2$。




### 矩阵和Frobenius范数



矩阵$\boldsymbol{X}$的Frobenius范数为该矩阵元素平方和的平方根：

$$\|\boldsymbol{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$









### 按元素运算

假设$\boldsymbol{x} = [4, 9]^\top$，以下是一些按元素运算的例子：

* 按元素相加： $\boldsymbol{x} + 1 = [5, 10]^\top$
* 按元素相乘： $\boldsymbol{x} \odot \boldsymbol{x} = [16, 81]^\top$
* 按元素相除： $72 / \boldsymbol{x} = [18, 8]^\top$
* 按元素开方： $\sqrt{\boldsymbol{x}} = [2, 3]^\top$


### 运算







## 导数和梯度

TODO(@astonzhang)<解释导数>。

假设函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的输入是一个$d$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$。函数$f(\boldsymbol{x})$有关$\boldsymbol{x}$的梯度是一个由$d$个偏导数组成的向量：

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_d}\bigg]^\top.$$


为表示简洁，我们有时用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。


### 常用梯度

假设$\boldsymbol{x}$是一个向量，那么

$$
\begin{aligned}
&\nabla_{\boldsymbol{x}} \boldsymbol{A}^\top \boldsymbol{x} = \boldsymbol{A} \\
&\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A}  = \boldsymbol{A} \\
&\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}  = (\boldsymbol{A} + \boldsymbol{A}^\top)\boldsymbol{x}\\
&\nabla_{\boldsymbol{x}} \|\boldsymbol{x} \|^2 = \nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{x} = 2\boldsymbol{x}
\end{aligned}
$$

假设$\boldsymbol{X}$是一个矩阵，那么
$$\nabla_{\boldsymbol{X}} \|\boldsymbol{X} \|_F^2 = 2\boldsymbol{X}.$$


### 泰勒展开

函数f的泰勒展开式是

$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!} (x-a)^n,$$

其中$f^{(n)}$为函数$f$的$n$阶导数。假设$\epsilon$是个足够小的数，如果将上式中$x$和$a$分别替换成$x+\epsilon$和$x$，我们可以得到

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2).$$

由于$\epsilon$足够小，上式也可以简化成

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$


### 黑塞矩阵（Hessian matrix）







## 小结

## 练习

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6966)

![](../img/qr_math.svg)
