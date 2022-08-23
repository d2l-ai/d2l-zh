# 注意力机制
:label:`chap_attention`

灵长类动物的视觉系统接受了大量的感官输入，
这些感官输入远远超过了大脑能够完全处理的程度。
然而，并非所有刺激的影响都是相等的。
意识的聚集和专注使灵长类动物能够在复杂的视觉环境中将注意力引向感兴趣的物体，例如猎物和天敌。
只关注一小部分信息的能力对进化更加有意义，使人类得以生存和成功。

自19世纪以来，科学家们一直致力于研究认知神经科学领域的注意力。
本章的很多章节将涉及到一些研究。

首先回顾一个经典注意力框架，解释如何在视觉场景中展开注意力。
受此框架中的*注意力提示*（attention cues）的启发，
我们将设计能够利用这些注意力提示的模型。
1964年的Nadaraya-Waston核回归（kernel regression）正是具有
*注意力机制*（attention mechanism）的机器学习的简单演示。

然后继续介绍的是注意力函数，它们在深度学习的注意力模型设计中被广泛使用。
具体来说，我们将展示如何使用这些函数来设计*Bahdanau注意力*。
Bahdanau注意力是深度学习中的具有突破性价值的注意力模型，它双向对齐并且可以微分。

最后将描述仅仅基于注意力机制的*Transformer*架构，
该架构中使用了*多头注意力*（multi-head attention）
和*自注意力*（self-attention）。
自2017年横空出世，Transformer一直都普遍存在于现代的深度学习应用中，
例如语言、视觉、语音和强化学习领域。

```toc
:maxdepth: 2

attention-cues
nadaraya-waston
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```
