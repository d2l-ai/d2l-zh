# 正向传播和反向传播


我们在[线性回归 --- 从0开始](linear-regression-scratch.md)中使用了一个叫做随机梯度下降的优化算法来训练模型。在随机梯度下降每一次迭代中，模型参数的当前值将自减学习率与该参数梯度的乘积。注意到这里我们使用了模型参数的梯度。

和线性回归一样，通常情况下，我们需要使用优化算法来训练深度学习模型，而优化算法往往会依赖模型参数梯度的计算。因此，模型参数梯度的计算对模型的训练来说十分重要。然而，在深度学习模型中，由于网络结构的复杂性，模型参数梯度的计算通常并不直观。虽然我们可以通过`MXNet`轻松获取模型参数的梯度，但是了解模型参数梯度的计算将有助于我们进一步了解深度学习模型训练的本质。


## 概念

反向传播（back-propagation）是计算深度学习模型参数梯度的方法。总的来说，反向传播中会依据微积分中的链式法则，按照输出层、靠近输出层的隐含层、靠近输入层的隐含层和输入层的次序，依次计算并存储模型损失函数有关模型各层的中间变量和参数的梯度。

反向传播对于各层中变量和参数的梯度计算可能会依赖各层变量和参数的当前值。对深度学习模型按照输入层、靠近输入层的隐含层、靠近输出层的隐含层和输出层的次序，依次计算并存储模型的中间变量叫做正向传播（forward-propagation）。


## 案例分析——正则化的多层感知机

为了解释正向传播和反向传播，我们以一个简单的$L_2$范数[正则化](reg-scratch.md)的[多层感知机](mlp-scratch.md)为例。


### 模型定义

给定一个输入为$\mathbf{x} \in \mathbb{R}^x$（每个样本输入向量长度为$x$）和真实值为$y \in \mathbb{R}$的训练数据样本，不考虑偏差项，我们可以得到中间变量

$$\mathbf{z} = \mathbf{W}^{(1)} \mathbf{x}$$

其中$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times x}$是模型参数。中间变量$\mathbf{z} \in \mathbb{R}^h$应用按元素操作的激活函数$\phi$后将得到向量长度为$h$的隐含层变量

$$\mathbf{h} = \phi (\mathbf{z})$$

隐含层$\mathbf{h} \in \mathbb{R}^h$也是一个中间变量。通过模型参数$\mathbf{W}^{(2)} \in \mathbb{R}^{y \times h}$可以得到向量长度为$y$输出层变量

$$\mathbf{o} = \mathbf{W}^{(2)} \mathbf{h}$$

假设损失函数为$\ell$，损失项

$$L = \ell(\mathbf{o}, y)$$

根据$L_2$范数[正则化](reg-scratch.md)的定义，带有提前设定的超参数$\lambda$的正则化项

$$s = \frac{\lambda}{2} (\|\mathbf{W}^{(1)}\|_2^2 + \|\mathbf{W}^{(2)}\|_2^2)$$

模型最终需要被优化的目标函数

$$J = L + s$$

### 计算图

为了可视化模型变量和参数之间在计算中的依赖关系，我们可以绘制计算图。

![](../img/backprop.svg)

### 梯度的计算与存储

在上图中，模型的参数是$\mathbf{W}^{(1)}$和$\mathbf{W}^{(2)}$。为了在模型训练中学习这两个参数，以随机梯度下降为例，假设学习率为$\eta$，我们可以通过

$$\mathbf{W}^{(1)} = \mathbf{W}^{(1)} - \eta \frac{\partial J}{\partial \mathbf{W}^{(1)}}$$

$$\mathbf{W}^{(2)} = \mathbf{W}^{(2)} - \eta \frac{\partial J}{\partial \mathbf{W}^{(2)}}$$

来不断迭代模型参数的值。因此我们需要模型参数梯度$\partial J/\partial \mathbf{W}^{(1)}$和$\partial J/\partial \mathbf{W}^{(2)}$。为此，我们可以按照反向传播的次序依次计算并存储梯度。

为了表述方便，对输入输出$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$为任意形状张量的函数$\mathsf{Y}=f(\mathsf{X})$和$\mathsf{Z}=g(\mathsf{Y})$，我们使用

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}})$$

来表达链式法则。以下依次计算得到的梯度将依次被存储。


首先，我们计算目标函数有关损失项和有关正则项的梯度

$$\frac{\partial J}{\partial L} = 1$$

$$\frac{\partial J}{\partial s} = 1$$


其次，我们依据链式法则计算目标函数有关输出层变量的梯度$\partial J/\partial \mathbf{o} \in \mathbb{R}^{y}$。

$$\frac{\partial J}{\partial \mathbf{o}} 
= \text{prod}(\frac{\partial J}{\partial L}， \frac{\partial L}{\partial \mathbf{o}})
= \frac{\partial L}{\partial \mathbf{o}}$$


正则项有关两个参数的梯度可以很直观地计算：

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}$$

$$\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}$$



现在我们可以计算最靠近输出层的模型参数的梯度$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{y \times h}$。在计算图中，$\mathbf{W}^{(2)}$可以经过$\mathbf{o}$和$s$通向$J$，依据链式法则，我们有

$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}} 
= \text{prod}(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}) + \text{prod}(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}})
= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}
$$


沿着输出层向隐含层继续反向传播，隐含层变量的梯度$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$可以这样计算

$$
\frac{\partial J}{\partial \mathbf{h}} 
= \text{prod}(\frac{\partial J}{\partial \mathbf{o}}， \frac{\partial \mathbf{o}}{\partial \mathbf{h}})
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}
$$


注意到激活函数$\phi$是按元素操作的，中间变量$\mathbf{z}$的梯度$\partial J/\partial \mathbf{z} \in \mathbb{R}^h$的计算需要使用按元素乘法符$\odot$

$$
\frac{\partial J}{\partial \mathbf{z}} 
= \text{prod}(\frac{\partial J}{\partial \mathbf{h}}， \frac{\partial \mathbf{h}}{\partial \mathbf{z}})
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi^\prime(\mathbf{z})
$$

最终，我们可以得到最靠近输入层的模型参数的梯度$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times x}$。在计算图中，$\mathbf{W}^{(1)}$可以经过$\mathbf{z}$和$s$通向$J$，依据链式法则，我们有

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}} 
= \text{prod}(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}) + \text{prod}(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}})
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}
$$


需要再次提醒的是，每次迭代中，上述各个依次计算出的梯度会被依次存储或更新。这是为了避免重复计算。例如，由于输出层变量梯度$\partial J/\partial \mathbf{o}$被计算存储，反向传播稍后的参数梯度$\partial J/\partial \mathbf{W}^{(2)}$和隐含层变量梯度$\partial J/\partial \mathbf{h}$的计算可以直接读取输出层变量梯度的值，而无需重复计算。

还有需要注意的是，反向传播对于各层中变量和参数的梯度计算可能会依赖通过正向传播计算出的各层变量和参数的当前值。举例来说，参数梯度$\partial J/\partial \mathbf{W}^{(2)}$的计算需要依赖隐含层变量的当前值$\mathbf{h}$。这个当前值是通过从输入层到输出层的正向传播计算并存储得到的。

## 总结

正向传播和反向传播是深度学习模型训练的基石（目前是）。


## 练习

- 如果模型的层数特别多，梯度的计算会有什么问题？
- 1986年，Rumelhart, Hinton, 和Williams提出了[反向传播](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)。然而2017年[Hinton表示他对反向传播“深刻怀疑”](https://www.axios.com/ai-pioneer-advocates-starting-over-2485537027.html)并发表了[Capsule论文](https://arxiv.org/abs/1710.09829)。你对此有何思考？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/3710)
