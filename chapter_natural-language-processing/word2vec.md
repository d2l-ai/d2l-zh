# 词向量 — word2vec


自然语言是一套用来表达含义的复杂系统。在这套系统中，词是表义的基本单元。在机器学习中，如何使用向量表示词？

顾名思义，词向量是用来表示词的向量，通常也被认为是词的特征向量。近年来，词向量已逐渐成为自然语言处理的基础知识。


## One-hot向量的局限

我们在[循环神经网络](../chapter_recurrent-neural-networks/rnn-scratch.md)中介绍过one-hot向量来表示词。假设词典中不同词的数量为$N$，每个词可以和从0到$N-1$的连续整数一一对应。假设一个词的相应整数表示为$i$，为了得到该词的one-hot向量表示，我们创建一个全0的长为$N$的向量，并将其第$i$位设成1。

然而，使用one-hot词向量并不是一个好选择。一个主要的原因是，one-hot词向量无法表达不同词之间的相似度。例如，任何一对词的one-hot向量的余弦相似度都为0。


## word2vec

2013年，Google团队发表了[word2vec](https://code.google.com/archive/p/word2vec/)工具。word2vec工具主要包含两个模型：跳字模型（skip-gram）和连续词袋模型（continuous bag of words，简称CBOW），以及两种高效训练的方法：负采样（negative sampling）和层序softmax（hierarchical softmax）。值得一提的是，word2vec词向量可以较好地表达不同词之间的相似和类比关系。

word2vec自提出后被广泛应用在自然语言处理任务中。它的模型和训练方法也启发了很多后续的词向量模型。本节将重点介绍word2vec的模型和训练方法。



## 跳字模型



## 连续词袋模型

## 负采样

## 层序softmax

## 结论

word2vec。


## 练习

* 词组问题。


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4203)
