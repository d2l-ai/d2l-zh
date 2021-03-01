# 注意力机制
:label:`chap_attention`

灵长类动物视觉系统的视神经接受大量的感官输入，这些输入信号远远超过了大脑能够完全处理的数量。幸运的是，并非所有的输入信号对大脑的刺激都是平等的。意识的聚集和集中使灵长类动物能够在复杂的视觉环境中将注意力聚集到感兴趣的物体上，例如猎物和捕食者。这种只关注一小部分信息的能力在进化过程中具有重要的意义，得益于此人类能够生存和成功。

自 19 世纪以来，科学家们一直在研究认知神经科学领域的注意力。在本章中，我们将首先回顾一个流行框架，此框架解释了注意力是如何在视觉场景中生效的。受此框架中的注意力提示（attention cues）的启发，我们将设计利用这种注意力提示的模型。值得一提的是，1964 年的 Nadaraya-Waston 核回归可以看做是简单的具有 *注意力机制* 的机器学习方法。

接下来，我们将继续介绍注意力函数，它在深度学习相关的注意力模型设计中被广泛使用。具体来说，我们将展示如何使用这些函数来设计 *Bahdanau 注意力*，这是深度学习中突破性的注意力模型，其可以实现双向对齐并且是可微的（differentiable）。

最后，我们将描述基于注意机制的 *Transformer* 架构，它融合了最近提出的 *多头注意力*（multi-head attention）和 *自注意力*（self-attention）设计。自 2017 年被提出以来，Transformer 一直在现代深度学习应用中普遍存在，广泛应用在例如语言、视觉、语音和强化学习领域。

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
