# 前言

这些年机器学习社区和生态圈进入一个奇怪的状态。二十一世纪早期的时候虽然只有少数一
些问题被攻克了，但我们自认为我们理解这些模型是怎么运行的，以及为什么。而现在机器
学习系统非常强大，但留下一个巨大问题，为什么它们如此有效。

这个新世界提供了巨大的机会，同时也带来了浮躁的投机。现在研究预印本被标题党和肤浅
的内容充斥，人工智能创业公司只需要几个演示就能获得巨大的估值，朋友圈也被不懂技术
的营销人员写的小白文刷屏。在这个混乱的，充斥的快钱和宽松标准的时代，我们坚信我们
需要坚持自己的写作标准和深度。同时为了很好地解释，实现，和可视化我们想讲的模型，
我们需要保证我们作者会认为这是一个挑战，而不是乏味的机械工作。

## 组织

目前我们使用下面这个方式来组织每个教程（除了少数几个背景知识介绍教程外）：

1. 引入一个（或者少数几个）新概念
2. 提供一个使用真实数据的完整样例

This will be interleaved by background material, as needed. That is, we will
often err on the side of making tools available before explaining them fully
(and we will follow up by explaining the background later). For instance, we
will use Stochastic Gradient Descent before fully explaining why it is
useful. This helps with giving practitioners the necessary ammunition to solve
problems quickly, at the expense of requiring the reader to trust us with some
decisions, at least in the short term. Throughout we'll be working with the
MXNet library, which has the rare property of being flexible enough for research
while being fast enough for production. We'll generally be using MXNet's new
high-level imperative interface gluon. Note that this is not the same as
mxnet.module, an older, symbolic interface supported by MXNet.

We'll be teaching deep learning concepts from scratch. Sometimes, we'll want to
delve into fine details about the models that are hidden from the user by
gluon's advanced features. This comes up especially in the basic tutorials,
where we'll want you to understand everything that happens in a given layer. In
these cases, we'll generally present two versions of the tutorial: one where we
implement everything from scratch, relying only on NDArray and automatic
differentiation, and another where we show how to do things succinctly with
gluon. Once we've taught you how a layer works, we can just use the gluon
version in subsequent tutorials.


## 通过动手来学习

Many textbooks teach a series of topics, each in exhaustive detail. For example,
Chris Bishop's excellent textbook, Pattern Recognition and Machine Learning
teaches each topic so thoroughly, that getting to the chapter on linear
regression requires a non-trivial amount of work. When I was first learning
machine learning, this actually limited the book's usefulness as an introductory
text. When I rediscovered it a couple years later, I loved it precisely for its
thoroughness but it's still not how I could imagine learning in the first place.

Instead, in this book, we'll teach most concepts just in time. For the
fundamental preliminaries like linear algebra and probability, we'll provide a
brief crash course from the outset, but we want you to taste the satisfaction of
training your first model before worrying about exotic probability
distributions.
