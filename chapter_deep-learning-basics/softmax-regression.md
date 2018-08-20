# Softmax回归

前几节介绍的线性回归模型适用于输出为连续值的情景。在另一类情景中，模型输出可以是一个像图片类别这样的离散值。对于这样的离散值预测问题，我们可以使用诸如softmax回归在内的分类模型。和线性回归不同，softmax回归的输出单元从一个变成了多个，且引入了softmax运算使得输出更适合离散值的预测和训练。本节以softmax回归模型为例，介绍神经网络中的分类模型。


## 分类问题

让我们考虑一个简单的图片分类问题，其输入图片高和宽均为两像素，且为色彩为灰度。这样每个像素值都可以用一个标量表示，我们将图片中的四个像素记为$x_1, x_2, x_3, x_4$。假设训练数据集中图片的真实标签为狗、猫或鸡（如同前面假设房价只取决于居住面积和房龄，这里我们假设可以用4个像素表示出这三种动物），这些标签分别对应离散值$y_1, y_2, y_3$。

通常我们使用连续值来表示类别，例如$y_1=1, y_2=2, y_3=3$，那么一张图片的标签为1、2和3这三个数字中的一个。虽然我们仍然可以使用回归模型来进行建模，预测时我们将预测值就近定点化到1、2和3这三个值之一上。但这一连续值到离散值的转化通常会影响到分类质量，因此我们一般使用更加适合离散值输出的模型来解决分类问题。

## Softmax回归模型

Softmax回归跟线性回归一样将输入特征与权重做线性叠加，一个主要不同在于softmax回归的输出值个数等于标签里的类别数。因为我们特征为4，且有3种动物，所以权重里包含12个标量，偏移有3个标量，且对每个输入计算$o_1, o_2, o_3$这三个输出：

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1,\\
o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2,\\
o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3.
\end{aligned}
$$


图3.2用神经网络图描绘了上面的计算，其同线性回归一样，也是一个单层神经网络。因为$o_1, o_2, o_3$的计算都要依赖于$x_1, x_2, x_3, x_4$。所以，softmax回归的输出层也是一个全连接层。

![Softmax回归是一个单层神经网络。](../img/softmaxreg.svg)

### Softmax运算

因为分类问题中我们需要得到离散的预测输出，一个简单的办法是将输出值$o_i$当做预测类别是$i$的置信度，从而值最大的输出所对应的类就是预测输出，即$\operatorname*{argmax}_i o_i$。例如如果$o_1,o_2,o_3$分别为$0.1,10,0.1$，那么预测类别为2，其代表猫。

单直接使用输出层的有两点不方便之处，一方面是由于其输出值的范围不固定，难以直观上判断其值的意义。例如上面例子里10表示很置信图片里是猫，因为其大于其他两类100倍。但如果$o_1=o_3=10^3$，那么10表示图片里是猫的几率很低。另一方面，由于真实标号是离散值，其不容易与输出值计算误差。

Softmax运算（softmax operator）提出正是为了解决这两个问题。它通过下式将输出值变换成值为正且和为1的概率分布：

$$\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3),$$

其中

$$
\begin{aligned}
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\\
\hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\\
\hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
\end{aligned}
$$

容易看出$\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$且$\hat{y}_1 > 0, \hat{y}_2 > 0, \hat{y}_3 > 0$，因此$\hat{y}_1, \hat{y}_2, \hat{y}_3$是一个合法的概率分布。这时候，如果$\hat y_2=0.8$，不管其他两个值多少，我们都知道有80%概率图片里是猫。此外，可以注意到

$$\operatorname*{argmax}_i o_i = \operatorname*{argmax}_i \hat y_i,$$

因此softmax运算不改变预测类别输出。

## 单样本分类的矢量计算表达式

为了提高计算效率，我们可以将单样本分类通过矢量计算来表达。在上面的图片分类问题中，假设softmax回归的权重和偏差参数分别为

$$
\boldsymbol{W} = 
\begin{bmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43}
\end{bmatrix},\quad
\boldsymbol{b} = 
\begin{bmatrix}
    b_1 & b_2 & b_3
\end{bmatrix},
$$



设$2 \times 2$图片样本$i$的特征为

$$\boldsymbol{x}^{(i)} = \begin{bmatrix}x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix},$$

输出层输出为

$$\boldsymbol{o}^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix},$$

预测为狗、猫或鸡的概率分布为

$$\boldsymbol{\hat{y}}^{(i)} = \begin{bmatrix}\hat{y}_1^{(i)} & \hat{y}_2^{(i)} & \hat{y}_3^{(i)}\end{bmatrix}.$$


我们对样本$i$分类的矢量计算表达式为

$$
\begin{aligned}
\boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}).
\end{aligned}
$$

## 小批量样本分类的矢量计算表达式


为了进一步提升计算效率，我们通常对小批量数据做矢量计算。广义上，给定一个小批量样本，其批量大小为$n$，输入个数（特征数）为$d$，输出个数（类别数）为$q$。设批量特征为$\boldsymbol{X} \in \mathbb{R}^{n \times d}$。假设softmax回归的权重和偏差参数分别为$\boldsymbol{W} \in \mathbb{R}^{d \times q}, \boldsymbol{b} \in \mathbb{R}^{1 \times q}$。Softmax回归的矢量计算表达式为

$$
\begin{aligned}
\boldsymbol{O} &= \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{Y}} &= \text{softmax}(\boldsymbol{O}),
\end{aligned}
$$

其中的加法运算使用了广播机制，$\boldsymbol{O}, \boldsymbol{\hat{Y}} \in \mathbb{R}^{n \times q}$且这两个矩阵的第$i$行分别为$\boldsymbol{o}^{(i)}$和$\boldsymbol{\hat{y}}^{(i)}$。


## 交叉熵损失函数

前面我们提到使用softmax运算后可以更方便的与标注计算误差。因为softmax运算将输出转换成一个合法的对类别预测的分布，同时真实标签也可以当做是类别的分布：对于样本$i$，我们构造$\boldsymbol{y}^{(i)}\in \mathbb{R}^{q}$ ，使得其第$y^{(i)}$个元素为1，其余为0。这样我们的训练目标可以设为使得预测概率分布$\boldsymbol{\hat y}^{(i)}$尽可能的接近标注概率分布$\boldsymbol{y}^{(i)}$。

我们可以跟线性回归那样使用平方损失函数$\frac{1}{2}\|\boldsymbol{\hat y}^{(i)}-\boldsymbol{y}^{(i)}\|^2$。但注意到想要预测分类结果正确，我们不需要预测概率完全等于标注概率，例如在图片分类的例子里，如果$y^{(i)}=2$，那么我们只需要$\hat y^{(i)}_2$比其他两个预测值大就行了。即使其值为0.5，不管其他两个值为多少，类别预测均正确。而平方损失则过于严格，例如$\hat y^{(i)}_0=\hat y^{(i)}_1=0.1$比$\hat y^{(i)}_0=0, \hat y^{(i)}_1=.2$的损失要小很多，虽然两者都有同样正确的分类预测结果。

改善这一问题是一个方法是使用更适合衡量两个概率分布不同的测量函数，其中交叉熵（cross entropy）是一个常用的衡量方法：

$$h\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = \sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)}.$$

在这里，我们知道$\boldsymbol y^{(i)}$中只有第$y^{(i)}$个元素为1，其余全为0，于是$h(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = \log \hat y_{y^{(i)}}^{(i)}$。也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，我们就可以确保分类结果正确，这跟其他值无关。当然，遇到一个样本有多个标号时，例如图片里含有不止一个物体，我们不能做这一步简化。但对于这种情况，交叉熵同样只关心对图片中出现的类别的预测概率。



假设训练数据集的样本数为$n$，交叉熵损失函数定义为
$$\ell(\boldsymbol{\Theta}) = -\frac{1}{n} \sum_{i=1}^n h\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),$$

其中$\boldsymbol{\Theta}$代表模型参数。同样的，如果每个样本只有一个标号，那么交叉熵损失可以简写成$\ell(\boldsymbol{\Theta}) = -\frac 1n  \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$。从另一个角度来看，我们知道最小化$\ell(\boldsymbol{\Theta})$，等价于最大化$-e^{\ell(\boldsymbol{\Theta})}=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$，也就是说最小化交叉熵损失函数等价于最大化在对训练数据集所有标签类别的联合预测概率。



## 模型预测及评价

在训练好softmax回归模型后，给定任一样本特征，我们可以预测每个输出类别的概率。通常，我们把预测概率最大的类别作为输出类别。如果它与真实类别（标签）一致，说明这次预测是正确的。在下一节的实验中，我们将使用准确率（accuracy）来评价模型的表现。它等于正确预测数量与总预测数量的比。

## 小结

* Softmax回归适用于分类问题。它使用softmax运算输出类别的概率分布。
* Softmax回归是一个单层神经网络，输出个数等于分类问题中的类别个数。


## 练习

* 如果按本节softmax运算的定义来实现该运算，在计算时可能会有什么问题？提示：想想$\exp(50)$的大小。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6403)

![](../img/qr_softmax-regression.svg)
