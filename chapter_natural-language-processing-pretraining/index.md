# 自然语言处理：预训
:label:`chap_nlp_pretrain`

人类需要沟通。出于对人类状况的这种基本需求，每天都会产生大量书面文本。鉴于社交媒体、聊天应用程序、电子邮件、产品评论、新闻文章、研究论文和书籍中的丰富文本，使计算机能够理解它们以人类语言为基础提供帮助或做出决策变得至关重要。 

*自然语言处理 * 使用自然语言研究计算机与人之间的互动。
实际上，使用自然语言处理技术来处理和分析文本（人类自然语言）数据是非常常见的，例如 :numref:`sec_language_model` 中的语言模型和 :numref:`sec_machine_translation` 中的机器翻译模型。 

为了理解文本，我们可以从学习它的表述开始。利用来自大型语句的现有文本序列，
*自我监督学习 *
广泛用于预训文本表示法，例如使用其周围文本的其他部分来预测文本的某些隐藏部分。通过这种方式，模型可以通过监督从 * 大量 * 文本数据中学习，而无需 * 昂贵的 * 标签工作！ 

正如我们将在本章中看到的那样，当将每个单词或子词视为单个标记时，可以使用 word2vec、GloVE 或在大型语库上嵌入子词模型来预训每个令牌的表示形式。训练前后，每个令牌的表示可以是向量，但是，无论上下文是什么，它都保持不变。例如，“银行” 的矢量表示在 “去银行存一些钱” 和 “去银行坐下” 两个方面都是相同的。因此，更多最近的训练前模型使同一标记的表示适应不同的环境。其中包括 BERT，这是一个基于变压器编码器的更深入的自我监督模型。在本章中，我们将重点介绍如何为文本预先训练此类表示形式，正如 :numref:`fig_nlp-map-pretrain` 中所强调的那样。 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on the upstream text representation pretraining.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`

从总体情况来看，:numref:`fig_nlp-map-pretrain` 显示，预训练的文本表示可以被馈送到不同的下游自然语言处理应用程序的各种深度学习架构中。我们将在 :numref:`chap_nlp_app` 中报道它们。

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining
```
