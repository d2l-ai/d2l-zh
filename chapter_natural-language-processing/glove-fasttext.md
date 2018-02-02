# 词向量 — GloVe和fastText


本节介绍两种更新一点的词向量。它们分别是2014年由Stanford团队发表的[GloVe](https://nlp.stanford.edu/pubs/glove.pdf)和2017年由Facebook团队发表的[fastText](https://arxiv.org/pdf/1607.04606.pdf)。


## GloVe

GloVe使用了词与词之间的共现（co-occurrence）信息。我们定义$\mathbf{X}$为共现词频矩阵，其中元素$x_{ij}$为词$j$出现在词$i$的环境（context）的次数。这里的“环境”有多种可能的定义。举个例子，在一段文本序列中，如果词$j$出现在词$i$左边或者右边不超过10个词的距离，我们可以认为词$j$出现在词$i$的环境一次。令$x_i = \sum_k x_{ik}$为任意词出现在词$i$的环境的次数。那么，

$$P_{ij} = \mathbb{P}(j \mid i) = \frac{x_{ij}}{x_i}$$

为词$j$出现在词$i$的环境的概率。这一概率也称词$i$和词$j$的共现概率。


### 共现概率比值

[GloVe论文](https://nlp.stanford.edu/pubs/glove.pdf)里展示了以下一组词对的共现概率与比值：

* $\mathbb{P}(k \mid \text{ice})$：0.00019（$k$= solid），0.000066（$k$= gas），0.003（$k$= water），0.000017（$k$= fashion）
* $\mathbb{P}(k \mid \text{steam})$：0.000022（$k$= solid），0.00078（$k$= gas），0.0022（$k$= water），0.000018（$k$= fashion）
* $\mathbb{P}(k \mid \text{ice}) / \mathbb{P}(k \mid \text{steam})$：8.9（$k$= solid），0.085（$k$= gas），1.36（$k$= water），0.96（$k$= fashion）


我们通过上表可以观察到以下现象：

* 对于与ice相关而与steam不相关的词$k$，例如$k=$solid，我们期望共现概率比值$P_{ik}/P_{jk}$较大，例如上面最后一栏的8.9。
* 对于与ice不相关而与steam相关的词$k$，例如$k=$gas，我们期望共现概率比值$P_{ik}/P_{jk}$较小，例如上面最后一栏的0.085。
* 对于与ice和steam都相关的词$k$，例如$k=$water，我们期望共现概率比值$P_{ik}/P_{jk}$接近1，例如上面最后一栏的1.36。
* 对于与ice和steam都不相关的词$k$，例如$k=$fashion，我们期望共现概率比值$P_{ik}/P_{jk}$接近1，例如上面最后一栏的0.96。

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

对于权重函数$f(x)$，一个建议的选择是，当$x < c$（例如$c = 100$），令$f(x) = (x/c)^\alpha$（例如$\alpha = 0.75$），反之令$f(x) = 1$。需要注意的是，损失函数的计算复杂度与共现词频矩阵$\mathbf{X}$中非零元素的数目呈线性关系。我们可以从$\mathbf{X}$中随机采样小批量非零元素，使用[随机梯度下降](../chapter_optimization/gd-sgd-scratch.md)迭代词向量和偏移项。

需要注意的是，对于任意一对$i, j$，损失函数中存在以下两项之和

$$f(x_{ij}) (\mathbf{v}_i^\top \tilde{\mathbf{v}}_j + b_i + b_j - \log(x_{ij}))^2 + f(x_{ji}) (\mathbf{v}_j^\top \tilde{\mathbf{v}}_i + b_j + b_i - \log(x_{ji}))^2$$

由于$x_{ij} = x_{ji}$，对调$\mathbf{v}$和$\tilde{\mathbf{v}}$并不改变损失函数中这两项之和的值。也就是说，在损失函数所有项上对调$\mathbf{v}$和$\tilde{\mathbf{v}}$也不改变整个损失函数的值。因此，任意词的中心词向量和背景词向量是等价的。只是由于初始化值的不同，同一个词最终学习到的两组词向量可能不同。当所有词向量学习得到后，GloVe使用一个词的中心词向量与背景词向量之和作为该词的最终词向量。




## fastText


fastText在[使用负采样的跳字模型](word2vec.md)基础上，将每个中心词视为子词（subword）的集合，并学习子词的词向量。


以where这个词为例，设子词为3个字符，它的子词包括“&lt;wh”、“whe”、“her”、“ere”、“re&gt;”和特殊子词（整词）“&lt;where&gt;”。其中的“&lt;”和“&gt;”是为了将作为前后缀的子词区分出来。而且，这里的子词“her”与整词“&lt;her&gt;”也可被区分。给定一个词$w$，我们通常可以把字符长度在3到6之间的所有子词和特殊子词的并集$\mathcal{G}_w$取出。假设词典中任意子词$g$的子词向量为$\mathbf{z}_g$，我们可以把[使用负采样的跳字模型](word2vec.md)的损失函数


$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \mathbf{v}_c)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \mathbf{v}_c)} $$

直接替换成

$$ - \text{log} \mathbb{P} (w_o \mid w_c) = -\text{log} \frac{1}{1+\text{exp}(-\mathbf{u}_o^\top \sum_{g \in \mathcal{G}_{w_c}} \mathbf{z}_g)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}(\mathbf{u}_{i_k}^\top \sum_{g \in \mathcal{G}_{w_c}} \mathbf{z}_g)} $$

我们可以看到，原中心词向量被替换成了中心词的子词向量的和。与整词学习（word2vec和GloVe）不同，词典以外的新词的词向量可以使用fastText中相应的子词向量之和。

fastText对于一些语言较重要，例如阿拉伯语、德语和俄语。例如，德语中有很多复合词，例如乒乓球（英文table tennis）在德语中叫“Tischtennis”。fastText可以通过子词表达两个词的相关性，例如“Tischtennis”和“Tennis”。



## 结论

* GloVe用词向量表达共现词频的对数。
* fastText用子词向量之和表达整词。


## 练习

* GloVe中，如果一个词出现在另一个词的环境中，是否可以利用它们之间在文本序列的距离重新设计词频计算方式？（可参考[Glove论文](https://nlp.stanford.edu/pubs/glove.pdf)4.2节）
* 如果丢弃GloVe中的偏移项，是否也可以满足任意一对词共现的对称性？
* 在fastText中，子词过多怎么办（例如，6字英文组合数为$26^6$）？（可参考[fastText论文](https://arxiv.org/pdf/1607.04606.pdf)3.2节）



**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/4372)
