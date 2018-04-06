# 数学基础

TODO(@astonzhang)


## 运算


### 按元素运算

假设$\boldsymbol{x} = [4, 9]^\top$，以下是一些按元素运算的例子：

* 按元素相加： $\boldsymbol{x} + 1 = [5, 10]^\top$
* 按元素相乘： $\boldsymbol{x} \odot \boldsymbol{x} = [16, 81]^\top$
* 按元素相除： $72 / \boldsymbol{x} = [18, 8]^\top$
* 按元素开方： $\sqrt{\boldsymbol{x}} = [2, 3]^\top$


## 导数和梯度

假设目标函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的输入是一个$d$维向量$\boldsymbol{x} = [x_1, x_2, \ldots, x_d]^\top$。目标函数$f(\boldsymbol{x})$有关$\boldsymbol{x}$的梯度是一个由$d$个偏导数组成的向量：

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_d}\bigg]^\top.$$


为表示简洁，我们用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。

### 泰勒展开

函数f的泰勒展开式是

$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!} (x-a)^n,$$

其中$f^{(n)}$为函数$f$的$n$阶导数。假设$\epsilon$是个足够小的数，如果将上式中$x$和$a$分别替换成$x+\epsilon$和$x$，我们可以得到

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2).$$

由于$\epsilon$足够小，上式也可以简化成

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$
