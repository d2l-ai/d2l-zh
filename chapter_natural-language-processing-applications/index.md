# 自然语言处理：应用
:label:`chap_nlp_app`

我们已经看到了如何在文本序列中表示令牌并在 :numref:`chap_nlp_pretrain` 中训练它们的表示形式。这种预训练的文本表示法可以被馈送到不同的模型中，用于不同的下游自然语言处理 

事实上，前面的章节已经讨论过一些自然语言处理应用
*没有预培训 *，
只是为了解释深度学习架构。例如，在 :numref:`chap_rnn` 中，我们依靠 RNN 来设计语言模型来生成类似小说的文本。在 :numref:`chap_modern_rnn` 和 :numref:`chap_attention` 中，我们还设计了基于 RNN 和机器翻译注意力机制的模型。 

但是，本书不打算全面涵盖所有此类应用程序。相反，我们的重点是 * 如何应用语言的（深度）表示学习来解决自然语言处理问题 *。鉴于预先训练的文本表示法，本章将探讨两个常见的和具有代表性的下游自然语言处理任务：情绪分析和自然语言推断，它们分别分析单个文本和文本对的关系。 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

如 :numref:`fig_nlp-map-app` 所述，本章重点介绍使用不同类型的深度学习架构（例如 MLP、CNN、RNN 和注意力）设计自然语言处理模型的基本思想。尽管可以将任何预训练的文本表示形式与 :numref:`fig_nlp-map-app` 中的任何应用程序的任何架构结合起来，但我们选择了一些代表性组合。具体来说，我们将探索基于 RNN 和 CNN 的流行架构以进行情绪分析。对于自然语言推断，我们选择注意力和 MLP 来演示如何分析文本对。最后，我们介绍如何针对各种自然语言处理应用程序微调预训练的 BERT 模型，例如序列级别（单文本分类和文本对分类）和令牌级别（文本标记和问答）。作为一个具体的实证案例，我们将对 BERT 进行微调，以实现自然语言推断。 

正如我们在 :numref:`sec_bert` 中引入的那样，对于各种自然语言处理应用程序，BERT 需要极少的体系结构更改。但是，这一好处的代价是为下游应用程序微调大量 BERT 参数。当空间或时间有限时，那些基于 MLP、CNN、RNN 和注意力精心制作的模型更加可行。在下面，我们从情绪分析应用程序开始，分别说明基于 RNN 和 CNN 的模型设计。

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
