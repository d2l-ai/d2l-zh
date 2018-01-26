# 词向量 — GloVe和fastText


本节介绍两种更新一点的词向量。它们分别是2014年由Stanford团队发表的[GloVe](https://nlp.stanford.edu/pubs/glove.pdf)和2017年由Facebook团队发表的[fastText](https://arxiv.org/pdf/1607.04606.pdf)。


## GloVe

GloVe使用了词与词之间的共现（co-occurrence）信息。我们定义$\mathbf{X}$为共现词频矩阵，其中元素$x_{ij}$为词$j$出现在词$i$的环境（context）的次数。这里的“环境”有多种可能的定义。举个例子，在一段文本序列中，如果词$j$出现在词$i$左边或者右边不超过10个词的距离，我们可以认为词$j$出现在词$i$的环境一次。令$x_i = \sum_k x_{ik}$为任意词出现在词$i$的环境的次数。那么，

$$P_{ij} = \mathbb{P}(j \mid i) = \frac{x_{ij}}{x_i}$$

为词$j$出现在词$i$的环境的概率。这一概率也称词$i$和词$j$的共现概率。


### 共现概率比值

[Glove论文](https://nlp.stanford.edu/pubs/glove.pdf)里展示了以下一组词对的共现概率与比值：


|共现概率与比值 |$k$= solid|  $k$= gas | $k$= water |$k$= fashion |
|:-:|:-:|:-:|:-:|:-:|
|$\mathbb{P}(k \mid \text{ice})$|$0.00019$|$0.000066$|$0.003$|$0.000017$|
|$\mathbb{P}(k \mid \text{steam})$|$0.000022$|$0.00078$|$0.0022$|$0.000018$|
|$\mathbb{P}(k \mid \text{ice}) / \mathbb{P}(k \mid \text{steam})$|$8.9$|$0.085$|$1.36$|$0.96$|

我们通过上表可以观察到以下现象：

* 对于与ice相关而与steam不相关的词$k$，例如$k=$solid，我们期望共现概率比值$P_{ik}/P_{jk}$较大，例如上表最后一栏的8.9。
* 对于与ice不相关而与steam相关的词$k$，例如$k=$gas，我们期望共现概率比值$P_{ik}/P_{jk}$较小，例如上表最后一栏的0.085。
* 对于与ice和steam都相关的词$k$，例如$k=$water，我们期望共现概率比值$P_{ik}/P_{jk}$接近1，例如上表最后一栏的1.36。
* 对于与ice和steam都不相关的词$k$，例如$k=$fashion，我们期望共现概率比值$P_{ik}/P_{jk}$接近1，例如上表最后一栏的0.96。

由此可见，共现概率比值能比较直观地表达词之间的关系。GloVe试图用有关词向量的函数来表达共现概率比值。

### 用词向量表达共现概率比值

GloVe的核心在于使用词向量表达共现概率比值。而任意一个这样的比值需要三个词$i$、$j$和$k$的词向量。对于共现概率$P_{ij} = \mathbb{P}(j \mid i)$，我们称词$i$和词$j$分别为中心词和背景词。我们使用$\mathbf{v}$和$\tilde{\mathbf{v}}$分别表示中心词和背景词的词向量。

我们可以用有关词向量的函数$f$来表达共现概率比值：

$$f(\mathbf{v}_i, \mathbf{v}_j, \tilde{\mathbf{v}}_k) = \frac{P_{ik}}{P_{jk}}$$

需要注意的是，函数$f$可能的设计并不唯一。首先，我们用向量之差来表达共现概率的比值，并将上式改写成

$$f(\mathbf{v}_i - \mathbf{v}_j, \tilde{\mathbf{v}}_k) = \frac{P_{ik}}{P_{jk}}$$

由于共现概率比值是一个标量，我们可以使用向量之间的内积把函数$f$的自变量进一步改写。我们可以得到

$$f((\mathbf{v}_i - \mathbf{v}_j)^\top \tilde{\mathbf{v}}_k) = \frac{P_{ik}}{P_{jk}}$$

由于任意一对词共现的对称性，我们希望以下两个性质可以同时被满足：

* 任意词作为中心词和背景词的词向量应该相等：对任意词$i$，$\mathbf{v}_i = \tilde{\mathbf{v}}_i$
* 词与词之间共现次数矩阵$\mathbf{X}$应该对称：对任意词$i$和$j$，$x_{ij} = x_{ji}$

为了满足以上两个性质，一方面，我们令

$$f((\mathbf{v}_i - \mathbf{v}_j)^\top \tilde{\mathbf{v}}_k) = \frac{f(\mathbf{v}_i^\top \tilde{\mathbf{v}}_k)}{f(\mathbf{v}_j^\top \tilde{\mathbf{v}}_k)}$$

并得到$f(x) = \text{exp}(x)$。以上两式的右边联立，


$$\exp(\mathbf{v}_i^\top \tilde{\mathbf{v}}_k) = P_{ik} = \frac{x_{ik}}{x_i}$$

由上式可得

$$\mathbf{v}_i^\top \tilde{\mathbf{v}}_k = \log(x_{ik}) - \log(x_i)$$

另一方面，我们可以把上式中$\log(x_i)$替换成两个偏移项之和$b_i + b_k$，得到

$$\mathbf{v}_i^\top \tilde{\mathbf{v}}_k = \log(x_{ik}) - b_i - b_k$$

将索引$i$和$k$互换，我们可验证对称性的两个性质可以同时被上式满足。

因此，对于任意一对词$i$和$j$，用它们词向量表达共现概率比值最终可以被简化为表达它们共现词频的对数：

$$\mathbf{v}_i^\top \tilde{\mathbf{v}}_j + b_i + b_j = \log(x_{ij})$$


### 损失函数

上式中的共现词频是直接在训练数据上统计得到的，为了学习词向量和相应的偏移项，我们希望上式中的左边与右边越接近越好。给定词典大小$V$和权重函数$f(x_{ij})$，我们定义损失函数为

$$\sum_{i, j = 1}^V f(x_{ij}) (\mathbf{v}_i^\top \tilde{\mathbf{v}}_j + b_i + b_j - \log(x_{ij}))^2$$

对于权重函数$f(x)$，一个建议的选择是，当$x < c$（例如$c = 100$），令$f(x) = (x/c)^\alpha$（例如$\alpha = 0.75$），反之令$f(x) = 1$。需要注意的是，损失函数的计算复杂度与共现词频矩阵$\mathbf{X}$中非零元素的数目呈线性关系。我们可以从$\mathbf{X}$中随机采样小批量非零元素，使用[随机梯度下降](../chapter_optimization/gd-sgd-scratch.md)迭代词向量和偏移项。当所有词向量学习得到后，GloVe使用一个词的中心词向量与背景词向量之和作为该词的最终词向量。

## fastText

我们以跳字模型为例讨论负采样。

词典$\mathcal{V}$大小之所以会在目标函数中出现，是因为中心词$w_c$生成背景词$w_o$的概率$\mathbb{P}(w_o \mid w_c)$使用了softmax，而softmax正是考虑了背景词可能是词典中的任一词，并体现在softmax的分母上。

我们不妨换个角度，假设中心词$w_c$生成背景词$w_o$由以下相互独立事件联合组成来近似

* 中心词$w_c$和背景词$w_o$同时出现在该训练数据窗口
* 中心词$w_c$和第1个噪声词$w_1$不同时出现在该训练数据窗口（噪声词$w_1$按噪声词分布$\mathbb{P}(w)$随机生成，假设一定和$w_c$不同时出现在该训练数据窗口）
* ...
* 中心词$w_c$和第$K$个噪声词$w_K$不同时出现在该训练数据窗口（噪声词$w_K$按噪声词分布$\mathbb{P}(w)$随机生成，假设一定和$w_c$不同时出现在该训练数据窗口）

我们可以使用$\sigma(x) = 1/(1+\text{exp}(-x))$函数来表达中心词$w_c$和背景词$w_o$同时出现在该训练数据窗口的概率：

$$\mathbb{P}(D = 1 \mid w_o, w_c) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c)$$

那么，中心词$w_c$生成背景词$w_o$的对数概率可以近似为

$$ \text{log} \mathbb{P} (w_o \mid w_c) = \text{log} [\mathbb{P}(D = 1 \mid w_o, w_c) \prod_{k=1, w_k \sim \mathbb{P}(w)}^K \mathbb{P}(D = 0 \mid w_k, w_c) ]$$

假设噪声词$w_k$在词典中的索引为$i_k$，上式可改写为

$$ \text{log} \mathbb{P} (w_o \mid w_c) = \text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  + \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} [1-\frac{1}{1+\text{exp}(-\mathbf{u}_{i_k}^\top \mathbf{v}_c)}] $$

因此，有关中心词$w_c$生成背景词$w_o$的损失函数是

$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \mathbf{v}_c)} $$


当我们把$K$取较小值时，每次随机梯度下降的梯度计算开销将由$\mathcal{O}(|\mathcal{V}|)$降为$\mathcal{O}(K)$。

我们也可以对连续词袋模型进行负采样。有关背景词$w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)}$生成中心词$w_c$的损失函数

$$-\text{log} \mathbb{P}(w^{(t)} \mid  w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)})$$

在负采样中可以近似为

$$-\text{log} \frac{1}{1+\text{exp}[-\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) /(2m)]}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}[(\mathbf{u}_{i_k}^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) /(2m)]}$$

同样地，当我们把$K$取较小值时，每次随机梯度下降的梯度计算开销将由$\mathcal{O}(|\mathcal{V}|)$降为$\mathcal{O}(K)$。





## 结论

word2vec工具中的跳字模型和连续词袋模型可以使用两种负采样和层序softmax减小训练开销。


## 练习

* GloVe中，如果一个词出现在另一个词的环境中，是否可以利用它们之间在文本序列的距离重新设计词频计算方式？（可参考[Glove论文](https://nlp.stanford.edu/pubs/glove.pdf)4.2节）
* 如果丢弃GloVe中的偏移项，是否也可以满足任意一对词共现的对称性？



**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4203)
