# 词嵌入：GloVe和fastText


在word2vec被提出以后，很多其他词嵌入模型也陆续发表。本节介绍其中比较有代表性的两个模型。它们分别是由斯坦福团队发表的GloVe和由Facebook团队发表的fastText。


## GloVe

GloVe使用了词与词之间的共现（co-occurrence）信息。我们定义$\boldsymbol{X}$为共现词频矩阵，其中元素$x_{ij}$为词$j$出现在词$i$的背景的次数。这里的“背景”有多种可能的定义。举个例子，在一段文本序列中，如果词$j$出现在词$i$左边或者右边不超过10个词的距离，我们可以认为词$j$出现在词$i$的背景一次。令$x_i = \sum_k x_{ik}$为任意词出现在词$i$的背景的次数。那么，

$$p_{ij} = \mathbb{P}(j \mid i) = \frac{x_{ij}}{x_i}$$

为词$j$在词$i$的背景中出现的条件概率。这一条件概率也称词$i$和词$j$的共现概率。


### 共现概率比值

GloVe论文里展示了以下词对的共现概率与比值 [1]：

* $\mathbb{P}(k \mid \text{ice})$：0.00019（$k$= solid），0.000066（$k$= gas），0.003（$k$= water），0.000017（$k$= fashion）
* $\mathbb{P}(k \mid \text{steam})$：0.000022（$k$= solid），0.00078（$k$= gas），0.0022（$k$= water），0.000018（$k$= fashion）
* $\mathbb{P}(k \mid \text{ice}) / \mathbb{P}(k \mid \text{steam})$：8.9（$k$= solid），0.085（$k$= gas），1.36（$k$= water），0.96（$k$= fashion）


我们可以观察到以下现象：

* 对于与ice（冰）相关而与steam（蒸汽）不相关的词$k$，例如$k=$solid（固体），我们期望共现概率比值$p_{ik}/p_{jk}$较大，例如上面最后一行结果中的值8.9。
* 对于与ice不相关而与steam相关的词$k$，例如$k=$gas（气体），我们期望共现概率比值$p_{ik}/p_{jk}$较小，例如上面最后一行结果中的值0.085。
* 对于与ice和steam都相关的词$k$，例如$k=$water（水），我们期望共现概率比值$p_{ik}/p_{jk}$接近1，例如上面最后一行结果中的值1.36。
* 对于与ice和steam都不相关的词$k$，例如$k=$fashion（时尚），我们期望共现概率比值$p_{ik}/p_{jk}$接近1，例如上面最后一行结果中的值0.96。

由此可见，共现概率比值能比较直观地表达词与词之间的关系。GloVe试图用有关词向量的函数来表达共现概率比值。

### 用词向量表达共现概率比值

GloVe的核心思想在于使用词向量表达共现概率比值。而任意一个这样的比值需要三个词$i$、$j$和$k$的词向量。对于共现概率$p_{ij} = \mathbb{P}(j \mid i)$，我们称词$i$和词$j$分别为中心词和背景词。我们使用$\boldsymbol{v}_i$和$\tilde{\boldsymbol{v}}_i$分别表示词$i$作为中心词和背景词的词向量。

我们可以用有关词向量的函数$f$来表达共现概率比值：

$$f(\boldsymbol{v}_i, \boldsymbol{v}_j, \tilde{\boldsymbol{v}}_k) = \frac{p_{ik}}{p_{jk}}.$$

需要注意的是，函数$f$可能的设计并不唯一。下面我们考虑一种较为合理的可能性。

首先，用向量之差来表达共现概率的比值，并将上式改写成

$$f(\boldsymbol{v}_i - \boldsymbol{v}_j, \tilde{\boldsymbol{v}}_k) = \frac{p_{ik}}{p_{jk}}.$$

由于共现概率比值是一个标量，我们可以使用向量之间的内积把函数$f$的自变量进一步改写，得到

$$f((\boldsymbol{v}_i - \boldsymbol{v}_j)^\top \tilde{\boldsymbol{v}}_k) = \frac{p_{ik}}{p_{jk}}.$$

由于任意一对词共现的对称性，我们希望以下两个性质可以同时被满足：

* 任意词作为中心词和背景词的词向量应该相等：对任意词$i$，$\boldsymbol{v}_i = \tilde{\boldsymbol{v}}_i$。
* 词与词之间共现词频矩阵$\boldsymbol{X}$应该对称：对任意词$i$和$j$，$x_{ij} = x_{ji}$。

为了满足以上两个性质，一方面，我们令

$$f((\boldsymbol{v}_i - \boldsymbol{v}_j)^\top \tilde{\boldsymbol{v}}_k) = \frac{f(\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_k)}{f(\boldsymbol{v}_j^\top \tilde{\boldsymbol{v}}_k)},$$

并得到$f(x) = \text{exp}(x)$。以上两式的右边联立，


$$f(\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_k) = \exp(\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_k) = p_{ik} = \frac{x_{ik}}{x_i}.$$

由上式可得

$$\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_k = \log(p_{ik}) = \log(x_{ik}) - \log(x_i).$$

另一方面，我们可以把上式中$\log(x_i)$替换成两个偏差项之和$b_i + \tilde{b}_k$，得到

$$\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_k = \log(x_{ik}) - b_i - \tilde{b}_k.$$

因此，对于任意一对词$i$和$j$，我们可以用它们的词向量表达共现词频的对数：

$$\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_j + b_i + \tilde{b}_j = \log(x_{ij}).$$


### 损失函数

GloVe中的共现词频是直接在训练数据上统计得到的。为了学习词向量和相应的偏差项，我们希望上式中的左边与右边尽可能接近。给定词典$\mathcal{V}$和权重函数$h(x_{ij})$，GloVe的损失函数为

$$\sum_{i \in \mathcal{V}, j \in \mathcal{V}} h(x_{ij}) \left(\boldsymbol{v}_i^\top \tilde{\boldsymbol{v}}_j + b_i + \tilde{b}_j - \log(x_{ij})\right)^2,$$

其中权重函数$h(x)$的一个建议选择是，当$x < c$（例如$c = 100$），令$h(x) = (x/c)^\alpha$（例如$\alpha = 0.75$），反之令$h(x) = 1$。由于权重函数的存在，损失函数的计算复杂度与共现词频矩阵$\boldsymbol{X}$中非零元素的数目呈正相关。我们可以从$\boldsymbol{X}$中随机采样小批量非零元素，并使用优化算法迭代共现词频相关词的向量和偏差项。

我们提到过，任意词的中心词向量和背景词向量是等价的。但由于初始化值的不同，同一个词最终学习到的两组词向量可能不同。当学习得到所有词向量以后，GloVe使用一个词的中心词向量与背景词向量之和作为该词的最终词向量。




## fastText

我们在上一节介绍了word2vec的跳字模型和负采样。fastText以跳字模型为基础，将每个中心词视为子词（subword）的集合，并使用负采样学习子词的词向量。因此，fastText是一个子词嵌入模型。

举个例子，设子词长度为3个字符，“where”的子词包括“&lt;wh”、“whe”、“her”、“ere”、“re&gt;”和特殊子词（整词）“&lt;where&gt;”。这些子词中的“&lt;”和“&gt;”符号是为了将作为前后缀的子词区分出来。并且，这里的子词“her”与整词“&lt;her&gt;”也可被区分开。给定一个词$w$，我们通常可以把字符长度在3到6之间的所有子词和特殊子词的并集$\mathcal{G}_w$取出。假设词典中任意子词$g$的子词向量为$\boldsymbol{z}_g$，我们可以把使用负采样的跳字模型的损失函数


$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\boldsymbol{u}_o^\top \boldsymbol{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\boldsymbol{u}_{i_k}^\top \boldsymbol{v}_c)} $$

直接替换成

$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\boldsymbol{u}_o^\top \sum_{g \in \mathcal{G}_{w_c}} \boldsymbol{z}_g)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\boldsymbol{u}_{i_k}^\top \sum_{g \in \mathcal{G}_{w_c}} \boldsymbol{z}_g)}. $$

可以看到，原中心词向量被替换成了中心词的子词向量之和。与word2vec和GloVe不同，词典以外的新词的词向量可以使用fastText中相应的子词向量之和。

fastText对于一些语言较重要，例如阿拉伯语、德语和俄语。例如，德语中有很多复合词，例如乒乓球（英文table tennis）在德语中叫“Tischtennis”。fastText可以通过子词表达两个词的相关性，例如“Tischtennis”和“Tennis”。


由于词是自然语言的基本表义单元，词向量广泛应用于各类自然语言处理的场景中。例如，我们可以在之前介绍的语言模型中使用在更大规模语料上预训练的词向量，从而提升语言模型的准确性。


## 小结

* GloVe用词向量表达共现词频的对数。
* fastText用子词向量之和表达整词。它能表示词典以外的新词。


## 练习

* GloVe中，如果一个词出现在另一个词的背景中，是否可以利用它们之间在文本序列的距离重新设计词频计算方式？提示：可参考Glove论文4.2节 [1]。
* 如果丢弃GloVe中的偏差项，是否也可以满足任意一对词共现的对称性？
* 在fastText中，子词过多怎么办（例如，6字英文组合数为$26^6$）？提示：可参考fastText论文3.2节 [2]。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4372)

![](../img/qr_glove-fasttext.svg)

## 参考文献

[1] Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

[2] Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). Enriching word vectors with subword information. arXiv preprint arXiv:1607.04606.
