# 词嵌入（word2vec）


自然语言是一套用来表达含义的复杂系统。在这套系统中，词是表义的基本单元。顾名思义，词向量是用来表示词的向量，也可被认为是词的特征向量。把词映射为实数域上向量的技术也叫词嵌入（word embedding）。近年来，词嵌入已逐渐成为自然语言处理的基础知识。


## 为何不采用one-hot向量

我们在[“循环神经网络”](../chapter_recurrent-neural-networks/rnn.md)一节中使用one-hot向量表示词（字符为词）。回忆一下，假设词典中不同词的数量（词典大小）为$N$，每个词可以和从0到$N-1$的连续整数一一对应。这些与词对应的整数也叫词的索引。
假设一个词的索引为$i$，为了得到该词的one-hot向量表示，我们创建一个全0的长为$N$的向量，并将其第$i$位设成1。这样将每个词表示成了一个长度为$N$的向量，可以直接被神经网络使用。

虽然one-hot词向量构造简单，但通常不是一个好选择。一个主要的原因是它忽略了词与词之间的相关性。我们无法从one-hot词向量本身推测词与词之间的相关性。例如我们通常使用余弦相似度来衡量一对向量的相似度。对于向量$\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$，余弦相似度为

$$\frac{\boldsymbol{x}^\top \boldsymbol{y}}{\|\boldsymbol{x}\| \|\boldsymbol{y}\|} \in [-1, 1],$$

其对应这两个向量之间夹角的余弦值。但对于任何一对词的one-hot向量，它们的余弦相似度都为0，所以我们无法从中得到有用信息。

word2vec [1, 2] 的提出是为了解决上面这个问题。它将每个词表示成一个定长的向量，并使得这些向量能较好地表达不同词之间的相似和类比关系。word2vec里包含了两个模型：跳字模型（skip-gram）[1]和连续词袋模型（continuous bag of words，简称CBOW）[2]。，接下来让我们分别介绍这两个模型以及它们的训练方法。

## 跳字模型


在跳字模型中，我们用一个词来预测它在文本序列周围的词。举个例子，假设文本序列是“the”、“man”、“loves”、“his”和“son”。以“loves”作为中心词，设上下文窗口大小为2。跳字模型所关心的是，给定中心词“loves”，生成与它距离不超过2个词的背景词“the”、“man”、“his”和“son”的条件概率。也就是

$$\mathbb{P}(\textrm{the},\textrm{man},\textrm{his},\textrm{son}\mid\textrm{loves}),$$

假设给定中心词的情况下背景词相互独立，那么上式可以改写成：

$$\mathbb{P}(\textrm{the}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{man}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{his}\mid\textrm{loves})\cdot\mathbb{P}(\textrm{son}\mid\textrm{loves}).$$

![上下文窗口大小为2的跳字模型。](../img/skip-gram.svg)


在跳字模型中，每个词被表示成两个$d$维向量用来计算条件概率。假设这个词在词典中索引为$i$，当它为中心词时表示为$\boldsymbol{v}_i\in\mathbb{R}^d$，而为背景词时表示为$\boldsymbol{u}_i\in\mathbb{R}^d$。设中心词$w_c$在词典中索引为$c$，背景词$w_o$在词典中索引为$o$，给定中心词生成背景词的条件概率可以通过softmax函数定义为

$$\mathbb{P}(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},$$

这里词典索引集$\mathcal{V}$的定义是$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。

一般来说，假设给定一个长度为$T$的文本序列，时间步$t$的词为$w^{(t)}$。当上下文窗口大小为$m$时，且假设给定中心词的情况下背景词相互独立，跳字模型的似然估计是给定任一中心词生成所有背景词的概率：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(w^{(t+j)} \mid w^{(t)}),$$

这里比$0$前和比$T$后的时间步自动忽略了。

### 跳字模型训练

跳字模型的模型参数是每个词对应的中心词向量和背景词向量。训练中我们通过最大化似然估计来学习模型参数。我们知道最大似然估计与最小化它的负对数函数等价，也就是最小化以下损失函数：

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, \mathbb{P}(w^{(t+j)} \mid w^{(t)}).$$


如果使用随机梯度下降，那么在每一个时间步里我们随机采样一个较短的子序列来计算有关该子序列的损失，然后计算梯度来更新模型参数。梯度计算的关键是对数条件概率有关中心词向量和背景词向量的梯度。根据定义，首先看到


$$\log \mathbb{P}(w_o \mid w_c) =
\boldsymbol{u}_o^\top \boldsymbol{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)\right)$$

通过微分，我们可以得到上式中$\boldsymbol{v}_c$的梯度为：

$$
\begin{aligned}
\frac{\partial \text{log}\, \mathbb{P}(w_o \mid w_c)}{\partial \boldsymbol{v}_c} &= \boldsymbol{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\boldsymbol{u}_j^\top \boldsymbol{v}_c)\boldsymbol{u}_j}{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\\& = \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\boldsymbol{u}_j^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\right) \boldsymbol{u}_j = \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} \mathbb{P}(w_j \mid w_c) \boldsymbol{u}_j.
\end{aligned}
$$

它的计算需要字典中所有词以$w_c$为中心词的条件概率。同理，$\boldsymbol{u}_o$的梯度为：

$$\frac{\partial \text{log}\, \mathbb{P}(w_o \mid w_c)}{\partial \boldsymbol{u}_o} = \boldsymbol{v}_c - \frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)\boldsymbol{v}_c}{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)} = \boldsymbol{v}_c - \mathbb{P}(w_o \mid w_c)\boldsymbol{v}_c. $$

训练结束后，对于词典中的任一索引为$i$的词，我们均得到该词作为中心词和背景词的两组词向量$\boldsymbol{v}_i$和$\boldsymbol{u}_i$。我们通常使用中心词向量$\boldsymbol{v}$作为每个词的表征向量用在其他应用里。

## 连续词袋模型

连续词袋模型与跳字模型类似。与跳字模型最大的不同是，连续词袋模型用一个中心词在文本序列前后的背景词来预测该中心词。在同样的文本序列“the”、 “man”、“loves”、“his”和“son”里，以“loves”作为中心词，且上下文窗口大小为2时，连续词袋模型关心的是以背景词“the”、“man”、“his”和“son”为条件的中心词“loves”的条件概率，也就是

$$\mathbb{P}(\textrm{loves}\mid\textrm{the},\textrm{man},\textrm{his},\textrm{son}).$$

![上下文窗口大小为2的连续词袋模型。](../img/cbow.svg)

因为连续词袋模型的背景词有多个，我们将这些背景词向量取平均，然后使用和跳字模型一样的方法来计算条件概率。设$\boldsymbol{v_i}\in\mathbb{R}^d$和$\boldsymbol{u_i}\in\mathbb{R}^d$分别表示词典中索引为$i$的词的作为背景词和中心词的向量（注意符号和跳字模型中是相反的）。设中心词$w_c$在词典中索引为$c$，背景词$w_{o_1}, \ldots, w_{o_{2m}}$在词典中索引为$o_1, \ldots, o_{2m}$，那么条件概率通过如下计算：

$$\mathbb{P}(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\boldsymbol{u}_c^\top (\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\boldsymbol{u}_i^\top (\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}}) \right)}.$$

为了让符号更加简单，我们记$\boldsymbol{w}_o=\{w_{o_1}, \ldots, w_{o_{2m}}\}$，且$\bar{\boldsymbol{v}}_o = \left(\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}} \right)/(2m)$，那么上式可以简写成

$$\mathbb{P}(w_c \mid \boldsymbol{w}_o) = \frac{\exp\left(\boldsymbol{u}_c^\top \bar{\boldsymbol{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\boldsymbol{u}_i^\top \bar{\boldsymbol{v}}_o\right)}.$$

这样条件概率的计算形式同跳字模型一致。最后，给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$，上下文窗口大小为$m$，连续词袋模型的目标是最大化由背景词生成任一中心词的概率

$$ \prod_{t=1}^{T}  \mathbb{P}(w^{(t)} \mid  w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)}).$$

### 连续词袋模型训练

连续词袋模型训练同跳字模型基本一致。首先最大化连续词袋模型的似然估计等价于最小化以下损失函数：

$$  -\sum_{t=1}^T  \text{log}\, \mathbb{P}(w^{(t)} \mid  w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)}).$$

接下来看到

$$\log\,\mathbb{P}(w_c \mid \boldsymbol{w}_o) = \boldsymbol{u}_c^\top \bar{\boldsymbol{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\boldsymbol{u}_i^\top \bar{\boldsymbol{v}}_o\right)\right)$$

通过微分，我们可以计算出上式中条件概率的对数有关任一背景词向量$\boldsymbol{v}_{o_i}$($i = 1, \ldots, 2m$)的梯度为：

$$\frac{\partial \log\, \mathbb{P}(w_c \mid \boldsymbol{w}_o)}{\partial \boldsymbol{v}_o} = \frac{1}{2m} \left(\boldsymbol{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\boldsymbol{u}_j^\top \bar{\boldsymbol{v}}_o)\boldsymbol{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_j^\top \bar{\boldsymbol{v}}_o)} \right) = \frac{1}{2m}\left(\boldsymbol{u}_c - \sum_{j \in \mathcal{V}} \mathbb{P}(w_j \mid \boldsymbol{w}_o) \boldsymbol{u}_j \right).$$

关于中心词向量$\boldsymbol{u}_c$梯度为：

$$\frac{\partial \text{log}\, \mathbb{P}(w_c \mid \boldsymbol{w}_o)}{\partial \boldsymbol{u}_{c}} = \bar{\boldsymbol{v}}_o - \frac{\exp(\boldsymbol{u}_c^\top \bar{\boldsymbol{v}}_o)\boldsymbol{u}_c}{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_j^\top \bar{\boldsymbol{v}}_o)} = \bar{\boldsymbol{v}}_o - \mathbb{P}(w_c \mid \boldsymbol{w}_o)\bar{\boldsymbol{v}}_o.$$

同跳字模型不一样的一点在于，我们通常使用背景词向量$\boldsymbol{v}$作为每个词的表征向量用在其他应用里。

## 小结

* 词向量是用于表示自然语言中词的语义的向量。
* word2vec包含跳字模型和连续词袋模型，它们均用两个向量来表示每个词是中心词和背景词时的情形，并通过最大似然估计来训练得到词的向量表示。


## 练习

* 每次梯度的计算复杂度是多少？如果词典大小很大时，会有什么问题？
* 英语中有些固定短语由多个词组成，例如“new york”，你能想到什么办法来将这些词组纳入到词典中吗？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4203)

![](../img/qr_word2vec.svg)


## 参考文献

[1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).


[2] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation
of word representations in vector space. arXiv:1301.3781.
