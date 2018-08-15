# 自然语言处理

自然语言处理关注计算机与人类自然语言的交互。在实际中，我们常常使用自然语言处理技术，例如“循环神经网络”一章中介绍的语言模型，来处理和分析大量的自然语言数据。

本章中，我们将先介绍如何用向量表示词，并应用这些词向量寻找近义词和类比词。接着，在文本分类任务中，我们进一步应用词向量分析文本情感。此外，自然语言处理任务中很多输出是不定长的，例如任意长度的句子。我们将描述应对这类问题的编码器—解码器模型以及注意力机制，并将它们应用于机器翻译中。

```eval_rst

.. toctree::
   :maxdepth: 2

   word2vec
   glove-fasttext
   embedding-training
   similarity-analogy
   sentiment-analysis
   sentiment-analysis-cnn
   seq2seq
   beam-search
   attention
   machine-translation
```
