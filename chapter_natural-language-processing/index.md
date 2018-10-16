# 自然语言处理

自然语言处理关注计算机与人类之间的自然语言交互。在实际中，我们常常使用自然语言处理技术，例如“循环神经网络”一章中介绍的语言模型，来处理和分析大量的自然语言数据。

本章中，我们将先介绍如何用向量表示词，并在语料库上训练词向量。我们还将应用在更大语料库上预训练的词向量求近义词和类比词。接着，在文本分类任务中，我们进一步应用词向量分析文本情感，并分别基于循环神经网络和卷积神经网络讲解时序数据分类的两种重要思路。此外，自然语言处理任务中很多输出是不定长的，例如任意长度的句子。我们将描述应对这类问题的编码器—解码器模型、束搜索和注意力机制，并将它们应用于机器翻译中。

```eval_rst

.. toctree::
   :maxdepth: 2

   word2vec
   approx-training
   word2vec-gluon
   fasttext
   glove
   similarity-analogy
   sentiment-analysis-rnn
   sentiment-analysis-cnn
   seq2seq
   beam-search
   attention
   machine-translation
```
