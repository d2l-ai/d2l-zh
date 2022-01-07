# 多层感知机
:label:`chap_perceptrons`

在本章中，我们将第一次介绍真正的*深度*网络。
最简单的深度网络称为*多层感知机*。多层感知机由多层神经元组成，
每一层与它的上一层相连，从中接收输入；
同时每一层也与它的下一层相连，影响当前层的神经元。
当我们训练容量较大的模型时，我们面临着*过拟合*的风险。
因此，本章将从基本的概念介绍开始讲起，包括*过拟合*、*欠拟合*和模型选择。
为了解决这些问题，本章将介绍*权重衰减*和*暂退法*等正则化技术。
我们还将讨论数值稳定性和参数初始化相关的问题，
这些问题是成功训练深度网络的关键。
在本章的最后，我们将把所介绍的内容应用到一个真实的案例：房价预测。
关于模型计算性能、可伸缩性和效率相关的问题，我们将放在后面的章节中讨论。

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```
