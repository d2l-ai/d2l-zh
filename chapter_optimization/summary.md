# 优化算法总结

本章我们介绍并实现了多个深度学习中常使用的优化算法。它们均基于小批量随机梯度下降，但区别在于如何使用梯度来更新自变量。这些改进算法主要基于两个技术：使用指数加权移动平均来平滑时间步之间的变化，和对每个自变量元素使用自适应的学习率。这一节我们将总结这些算法的自变量更新公式，使得你可以更清楚的总结它们之间的区别。

## 小批量随机梯度下降

假设我们的目标是最小化连续可导的目标函数$f(\boldsymbol{x}):\mathbb{R}^d \rightarrow \mathbb{R}$。在更新开始前，即时间步$0$，随机初始化自变量$\boldsymbol{x}_{0}\in \mathbb{R}$。在时间步$t=1,2,\ldots$，首先随机均匀采样由训练数据样本索引所组成的小批量$\mathcal{B}_t$，然后基于它计算目标函数在$\boldsymbol{x}_{t-1}$处的梯度：

$$\boldsymbol{g}_t \leftarrow \frac{1}{|\mathcal{B}_t|}\nabla f_{\mathcal{B}_t}(\boldsymbol{x}_{t-1}) = \frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t} \nabla f_i(\boldsymbol{x}_{t-1}),$$

这里$|\mathcal{B}_t|$是小批量里的样本数，是一个超参数，且一般在时间步之间保持不变。

给定正的学习率$\eta_t$，这是另一个超参数，如下更新自变量：

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \eta_t \boldsymbol{g}_t.$$

过大或者过小的学习率都会带来问题。我们可以根据实验效果将其固定成一个正常数$\eta_t=\eta$，或者随着时间减小，例如$\eta_t=\eta t^\alpha$（通常$\alpha=-1$或者$-0.5$）或者$\eta_t = \eta \alpha^t$（例如$\alpha=0.95$）。


## 动量法

动量法基于指数加权移动平均来平滑小批量随机梯度下降里的自变量更新量$\eta_t \boldsymbol{g}_t$，平滑后的值记录在动量变量$\boldsymbol{v}\in\mathbb{R}^d$里。在时间步$0$，将$\boldsymbol{v}$的元素初始成0。在时间步$t>0$：

$$
\begin{aligned}
\boldsymbol{v}_t &\leftarrow \gamma \boldsymbol{v}_{t-1} + \eta_t \boldsymbol{g}_t \\
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{v}_t,
\end{aligned}
$$

这里$0\le \gamma < 1$是一个超参数，其效果近似于让$\boldsymbol{v}_t$是最近$1/(1-\gamma)$个时间步$\{\eta_{t-i} \boldsymbol{g}_{t-i}/(1-\gamma): i=0,\ldots, 1/(1-\gamma)-1\}$的加权平均。因此，当$\gamma=0$时，动量法等价与小批量随机梯度下降，即没有任何平滑。而$\gamma$越接近1，更新量越平滑。

## Adagrad

Adagrad为每个自变量元素使用一个自适应的且衰减的学习率。它将每个元素在过去时间步的梯度的平方和记录在状态变量$\boldsymbol{s}\in \mathbb{R}^d$中。在时间步$0$，将$\boldsymbol{s}$中元素初始成0。在时间步$t>0$：

$$
\begin{aligned}
\boldsymbol{s}_{t} &\leftarrow \boldsymbol{s}_{t-1} + \boldsymbol{g}_{t} \odot \boldsymbol{g}_{t}\\
\boldsymbol{x}_{t} &\leftarrow \boldsymbol{x}_{t-1} - \frac{\eta_t}{\sqrt{\boldsymbol{s}_{t}+\epsilon}} \odot\boldsymbol{g}_{t}
\end{aligned}
$$

这里$\epsilon$是极小的正的常数（例如$10^{-6}$）来防止除以0。Adagrad通过对学习率除以状态变量$\boldsymbol{s}$使得梯度数值较大的元素的学习率衰减更快。这样自变量各个元素的更新值不会差异过大而导致目标函数值容易发散或者下降缓慢。


## RMSProp

Adagrad的状态变量随着时间非递减，其容易导致训练后期学习率过小。RMSProp使用指数加权移动平均来更新状态变量，通过超参数$0\le\gamma\le 1$来控制作用移动平均的时间窗。

$$
\begin{aligned}
\boldsymbol{s}_{t} &\leftarrow \gamma \boldsymbol{s}_{t-1} + (1-\gamma)\boldsymbol{g}_{t} \odot \boldsymbol{g}_{t}\\
\boldsymbol{x}_{t} &\leftarrow \boldsymbol{x}_{t-1} - \frac{\eta_t}{\sqrt{\boldsymbol{s}_{t}+\epsilon}} \odot\boldsymbol{g}_{t}
\end{aligned}
$$

其效果近似于让$\boldsymbol{s}_t$是最近$1/(1-\gamma)$个时间步$\{\boldsymbol{g}_{t-i} \odot \boldsymbol{g}_{t-i}: i=0,\ldots, 1/(1-\gamma)-1\}$的加权平均，注意跟动量法不同在于这里没有除数$1-\gamma$。当$\gamma=0$时，宽口为1，自变量的更新量的绝对值均变成$\eta_t$，当$\gamma$越靠近$1$时窗口越长。

## Adadelta

Adadelta在RMSProp的基础上引入了新的状态变量$\Delta\boldsymbol{x}$，它用来维护自变量更新平方的指数加权移动平均。它的作用是用来取代学习率$\eta_t$。

$$
\begin{aligned}
\boldsymbol{s}_{t} &\leftarrow \rho\boldsymbol{s}_{t-1} + (1-\rho)\boldsymbol{g}_{t}\odot \boldsymbol{g}_{t}\\
\boldsymbol{g}_{t}' &\leftarrow \sqrt{ \frac{\Delta \boldsymbol{x}_{t-1}+\epsilon}{\boldsymbol{s}_{t}+\epsilon}}\odot\boldsymbol{g}_{t}\\
\boldsymbol{x}_{t} &\leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}_{t}'\\
\Delta\boldsymbol{x}_{t} &\leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1-\rho)\boldsymbol{g}_{t}'\odot\boldsymbol{g}_{t}'
\end{aligned}
$$

## Adam

Adam在RMSProp的基础上对梯度也作用了指数加权移动平均并保存在额外的状态变量$\boldsymbol{v}$里，并且对$\boldsymbol{s}$和$\boldsymbol{v}$做了时间上的校验。

$$
\begin{aligned}
\boldsymbol{v}_{t} &\leftarrow \beta_1 \boldsymbol{v}_{t-1} + (1-\beta_1) \boldsymbol{g}_{t}\\
\boldsymbol{v}_{t}' &\leftarrow \frac{\boldsymbol{v}_{t}}{1-\beta_1^t} \\
\boldsymbol{s}_{t} &\leftarrow \beta_2 \boldsymbol{s}_{t-1} + (1-\beta_2) \boldsymbol{g}_{t} \odot \boldsymbol{g}_{t}\\
\boldsymbol{s}'_{t} &\leftarrow \frac{\boldsymbol{s}_{t}}{1-\beta_2^t} \\
\boldsymbol{x}_{t} &\leftarrow \boldsymbol{x}_{t-1} - \frac{\eta}{\sqrt{\boldsymbol{s}_{t}'+\epsilon}}\odot\boldsymbol{v}_{t}'
\end{aligned}
$$

## 练习

- 深度学习优化算法一直是研究的热点。查阅最近论文了解最新的进展。例如如何在使用特别大的批量大小的时候仍保证靠近解的速度，以及$\eta_t$是关于时间$t$的周期函数而不是非递增函数。

