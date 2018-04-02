# 数学基础

TODO(@astonzhang)


## 导数和梯度

假设目标函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的输入是一个多维向量$\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$。目标函数$f(\mathbf{x})$有关$\mathbf{x}$的梯度是一个由偏导数组成的向量：

$$\nabla_\mathbf{x} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$


为表示简洁，我们有时用$\nabla f(\mathbf{x})$代替$\nabla_\mathbf{x} f(\mathbf{x})$。

### 泰勒展开


