# 多层感知器
:label:`chap_perceptrons`

在本章中，我们将介绍您的第一个真正的深度网络。最简单的深度网络叫做多层感知器，它们由多层神经元组成，每一层都与下面层（从中接收输入）和上面的神经元（反过来影响它们）完全相连。
当我们训练大容量的模型时，我们会有过拟合的风险。因此，我们需要为你提供一个正式的介绍，介绍过拟合、欠拟合和模型选择的概念。
为了帮助你解决这些问题，我们将介绍正则化技术，如权重衰减（Weight decay）和丢弃法（Dropout）。我们还将讨论与数值稳定性和参数初始化相关的问题，这些问题是成功训练深度网络的关键。自始至终，我们的目标不仅是让您牢牢掌握概念，而且还能实际使用深度网络。在本章的最后，我们将所介绍的内容应用到一个房价预测的实际案例中。我们将有关模型的计算性能、可伸缩性和效率的问题放到后面的章节进行讲解。

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
