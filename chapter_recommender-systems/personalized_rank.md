

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-02 18:54:00
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-02 19:21:03
 * @Description:translate Personalized Ranking for Recommender Systems
 * @TODO::
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1092/chapter_recommender-systems/ranking.html
-->

# 个性化推荐系统排名

在前一部分，只有显性反馈被考虑，模型被训练和测试的观察评级。 这种方法有两个缺点: 第一，大多数反馈不是显性的，而是在现实场景中的隐性反馈，显性反馈的收集成本可能更高。 其次，完全忽略了可能预测用户兴趣的未观察到的用户-项（user-item）对，这使得这些方法不适用于评分不是随机缺失而是由用户偏好决定的情况。 未观察到的用户-项目对是真正的负反馈（用户对项目不感兴趣）和缺失值（用户将来可能与项目交互）的混合物。 我们只是简单地忽略了矩阵分解和自动记录中的未观察对。 显然，这些模型不能区分观察对和非观察对（observed and non-observed pairs），通常不适合个性化排名任务。

为此，一类以从隐性反馈生成排名推荐列表为目标的推荐模型已经获得了普及。 一般来说，个性化的排名模型可以通过逐点、成对或列表的方法进行优化。 点态方法每次考虑一个单独的交互，并训练一个分类器或回归器来预测个人的偏好。 和 AutoRec 是按照点对点的目标进行优化的矩阵分解。 成对的方法为每个用户考虑一对项目，目的是近似该对项目的最佳排序。 通常，成对的方法更适合于排名任务，因为预测相对顺序会让人联想到排名的性质。 列表方法近似排序的整个列表的项目，例如，直接优化排名措施，如规范化折扣累积收益(NDCG)。 然而，列表方法比逐点或成对的方法更复杂，计算更密集。 在本节中，我们将介绍两个成对的目标 / 损失，TODO:Right?贝叶斯个性化排名损失和铰链损失（Bayesian Personalized Ranking loss 和 Hinge loss），以及它们各自的实现。

## 贝叶斯个性化排序损失及其实现

贝叶斯个性化排序(BPR)[ Rendle 等人，2009](http://preview.d2l.ai/d2l-en/PR-1092/chapter_references/zreferences.html#rendle-freudenthaler-gantner-ea-2009)是一个由最大后验估计量导出的两个个性化排序损失。 它在现有的许多推荐模型中得到了广泛的应用。 Bpr 的训练数据包括正负对(缺失值)。 它假设用户更喜欢正面的项目，而不是其他所有未观察到的项目。

在形式上，训练数据是以$(u，i，j)$的形式由元组构成的，这表示用户$u$更喜欢项$i$而不是项$j$。 以最大化后验概率为目标的业务流程重组的贝叶斯公式如下:

TODO:MATH

其中表示任意推荐模型的参数,$>_u$表示用户$u$所需的所有项目的个性化总排名,我们可以建立最大后验估计量来推导个性化排名任务的一般优化准则。

TODO:MATH

其中 d: = {(u，i，j) i ∈ i + u ∧ j ∈ i something i + u } d: = {(u，i，j) i ∈ Iu + ∧ j ∈ i something Iu + }为训练集，i + u Iu + 表示用户喜欢的项，i 表示所有项，i + u i something Iu + 表示用户喜欢的除外项。 Y ^ ui y ^ ui 和 y ^ uj y ^ uj 分别是用户 u 到 i i 和 j j 的预测得分。 先验 p () p ()是一个具有零均值和方差协方差矩阵的正态分布。 在这里，我们 let = i = i TODO:MATH

TODO:PIC

我们将实现基类 mxnet.gluon.loss。 损失和超越前向方法构造贝叶斯个性化排名损失。 我们首先导入 Loss 类和 np 模块。

TODO:CODE

BPR loss的实现如下：

TODO:CODE

## Hinge loss以及它的实现

用于排序的Hinge loss与在支持向量机等分类器中经常使用的gloun库中提供的Hinge loss形式不同。在推荐系统中用于排名的损失有如下形式。

TODO:MATH

其中$m$为安全边际尺寸。它的目的是把负的项目从正的项目中推开。与BPR类似，它的目标是优化正样本和负样本之间的相关距离，而不是绝对输出，这使得它非常适合于推荐系统。

这两种损失在推荐的个性化排名中是可以互换的。

## 总结

- 推荐系统中用于个性化排序任务的排序损失有三种类型，即点态方法、成对方法和列表方法。
- Bayesian Personalized Ranking loss 和 Hinge loss这两种损失可以互换使用。

## 练习

1. 是否有BPR和hinge loss的变体可用?
1. 你能找到使用BPR或铰链损失的推荐模型吗?
