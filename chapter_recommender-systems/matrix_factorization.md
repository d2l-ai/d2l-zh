

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-02 09:00:13
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-02 09:31:31
 * @Description:translate
 * @TODO::
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1092/chapter_recommender-systems/mf.html
-->

# 矩阵分解

矩阵分解[Koren et al.， 2009](http://preview.d2l.ai/d2l-en/PR-1092/chapter_references/zreferences.html#koren-bell-volinsky-2009)是文献中比较成熟的推荐系统算法。矩阵分解模型的第一个版本是由Simon Funk在一篇著名的[博客文章](https://sifter.org/~simon/journal/20061211.html)中提出的，在文中他描述了相互作用矩阵分解的思想。2006年，Netflix举办了一场竞赛，让它广为人知。当时，媒体流媒体和视频租赁公司Netflix宣布了一项提高推荐系统性能的竞赛。能够在Netflix的基础上改进的最好的团队。随后，大奖由BellKor的Pragmatic Chaos团队获得，这是一个由BellKor、Pragmatic Theory和BigChaos(现在不需要担心这些算法)组成的团队。虽然最后的分数是一个集成方案（即，是许多算法的组合）的结果。）矩阵分解算法在最终的混合中起到了关键作用。Netflix大奖方案的技术报告[Toscher et al.， 2009](http://preview.d2l.ai/d2l-en/PR-1092/chapter_references/zreferences.html#toscher-jahrer-bell-2009)详细介绍了采用的模型。在本节中，我们将深入讨论矩阵分解模型及其实现的细节。

## 矩阵分解模型

矩阵分解是一类协同过滤模型。具体来说，该模型将用户-物品交互矩阵(如等级矩阵)分解为两个低秩矩阵的乘积，得到用户-物品交互的低秩结构。

TODO:all MATH:设R Rm nR Rm n表示mm用户与nn项的交互矩阵，RR的值表示明确的评分。将用户-物品交互分解为用户潜矩阵P Rm kP Rm k和物品潜矩阵Q Rn kQ Rn k，其中k m,nk m,n为潜因子大小。让pupu表示PP的第一行，qiqi表示QQ的第一行。对于给定的第二项，奇奇元素衡量的是该项具有电影的类型和语言等特征的程度。对于给定的用户$u$, $p_u$的元素衡量用户对物品相应特征的兴趣程度。这些潜在的因素可能度量那些例子中提到的明显的维度，或者是完全无法解释的。预测的收视率可由

TODO:MATH

其中TODO:MATH:R ^∈Rm×n是预测评级矩阵，其形状与R相同。 该预测规则的一个主要问题是无法对用户/项目偏差进行建模。 例如，某些用户倾向于给出较高的评分，或者某些项目由于质量较差而总是获得较低的评分。 这些偏差在实际应用中很常见。 为了捕获这些偏差，引入了用户特定偏差和项目特定偏差项。 具体而言，用户$u$对项目$i$给出的预测评分由下式计算：

TODO:MATH

然后，我们通过最小化预测评分和真实评分之间的均方误差来训练矩阵因子分解模型。目标函数定义如下

TODO:MATH


其中，$albert$为正则化率。采用正则化项TODO:MATH(P 2F+ Q 2F+b2u+b2i)对参数的大小进行惩罚，以避免过拟合。已知的$(u,i)$对存储在集合$K={(u,i)|R_ui未知}$（$K={(u,i)|R_ui is known}$）。模型参数可以通过随机梯度下降（Stochastic Gradient Descent）和Adam等优化算法进行学习。

矩阵分解模型的直观说明如下
TODO:PIC


在本节的其余部分，我们将解释矩阵分解的实现，并在MovieLens数据集上训练模型。

TODO:CODE

## 模型实现

首先，我们实现上述矩阵分解模型。 可以使用nn.Embedding创建用户和项目潜在因素。$(input_dim)$是项目/用户数，而$(output_dim)$是潜在因子(k)的维数。 我们还可以使用$nn.Embedding$通过将$output_dim$设置为1来创建用户/项目偏差。 在前向传播$forward%功能中，用户和项目ID用于查找嵌入embeddings。


TODO:CODE

# 评估方法

然后我们实施RMSE(均方根误差)测量，它通常用于测量模型预测的评分和实际观察到的评分(ground truth)之间的差异[Gunawardana & Shani, 2015]。RMSE定义为

TODO:MATH

其中$T$是由您希望对其进行评估的用户和项组成的集合。$|T|$就是这个集合的大小。我们可以使用mx.metric提供的RMSE函数。

TODO:CODE


## 训练和评估模型

在训练函数中，我们采用了重量衰减的$L2$损失。重量衰减机制与$L2$正则化具有相同的效果。

TODO:CODE

最后，让我们把所有东西放在一起，训练这个模型。这里，我们将潜在因子维数设置为30。

TODO:CODE

下面，我们使用训练过的模型来预测用户(ID 20)可能给一个物品(ID 30)的评级。

TODO:CODE


## 总结

- 矩阵分解模型在推荐系统中得到了广泛的应用。它可以用来预测用户可能给一个项目的评分。
- 我们可以实现推荐系统的训练矩阵分解。

## 练习

1. 练习可以改变潜在因素的大小。潜在因素的大小如何影响模型性能?
1. 尝试不同的优化器、学习率和权重衰减率。
1. 检查其他用户对特定电影的预测评分。
