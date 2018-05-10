# 数学基础

TODO(@astonzhang)<解释符号一览>。

## 向量

设$d$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$，或

$$
\boldsymbol{x} = 
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{d} 
\end{bmatrix}.
$$

上式中$x_1, \ldots, x_d$是向量的元素。我们用$\boldsymbol{x} \in \mathbb{R}^{d}$或$\boldsymbol{x} \in \mathbb{R}^{d \times 1}$表示，$\boldsymbol{x}$是一个各元素均为实数的$d$维向量。


## 矩阵

设$m$行$n$列矩阵

$$
\boldsymbol{X} = 
\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
\end{bmatrix}.
$$

上式中$x_{11}, \ldots, x_{mn}$是矩阵的元素。我们用$\boldsymbol{X} \in \mathbb{R}^{m \times n}$表示，$\boldsymbol{X}$是一个各元素均为实数的$m$行$n$列矩阵。


## 运算


### 按元素运算

假设$\boldsymbol{x} = [4, 9]^\top$，以下是一些按元素运算的例子：

* 按元素相加： $\boldsymbol{x} + 1 = [5, 10]^\top$
* 按元素相乘： $\boldsymbol{x} \odot \boldsymbol{x} = [16, 81]^\top$
* 按元素相除： $72 / \boldsymbol{x} = [18, 8]^\top$
* 按元素开方： $\sqrt{\boldsymbol{x}} = [2, 3]^\top$


## 导数和梯度

TODO(@astonzhang)<解释导数>。

假设函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的输入是一个$d$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$。函数$f(\boldsymbol{x})$有关$\boldsymbol{x}$的梯度是一个由$d$个偏导数组成的向量：

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_d}\bigg]^\top.$$


为表示简洁，我们有时用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。


### 常用梯度

$$\nabla_{\boldsymbol{x}} \boldsymbol{A}^\top \boldsymbol{x} = \boldsymbol{A} \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A}  = \boldsymbol{A} \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}  = (\boldsymbol{A} + \boldsymbol{A}^\top)\boldsymbol{x}\\
\nabla_{\boldsymbol{x}} \|\boldsymbol{x} \|^2 = \nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{x} = 2\boldsymbol{x}
$$


### 泰勒展开

函数f的泰勒展开式是

$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!} (x-a)^n,$$

其中$f^{(n)}$为函数$f$的$n$阶导数。假设$\epsilon$是个足够小的数，如果将上式中$x$和$a$分别替换成$x+\epsilon$和$x$，我们可以得到

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2).$$

由于$\epsilon$足够小，上式也可以简化成

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$
