# 全局向量的词表示（GloVe）

在介绍GloVe模型前让我们重新来看一下word2vec中的跳字模型。记$q_{ij}$是跳字模型中使用softmax函数拟合的条件概率$p_{ij}=\mathbb{P}(w_j\mid w_i)$。也就是$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)}$，这里$\mathbf{v}_i$和$\mathbf{u}_i$分别是词$w_i$作为中心词和背景词时的向量表示。

那么跳字模型的目标函数可以简写成最小化

$$L = -\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{C}_i} \log\,q_{ij},$$

这里$\mathcal{C}_i$是词$w_i$在序列中每次出现的背景窗口中的词索引的合集，且允许重复元素。记条件词频$x_{ij}$是词$w_j$作为背景词出现在$w_i$背景窗口中的次数，那么$x_{ij} = |\{j:j\in\mathcal{C}_i\}|$。同时记词频$x_i=|\mathcal{C}_i|$，那么有$p_{ij} = x_{ij}/x_i$。我们可以改写跳字模型的目标函数为：

$$L = -\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij} =
-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$$

这里，$\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$计算的是以$w_i$为中心词的背景词条件概率分布$p_{ij}$和模型预测的条件概率分布$q_{ij}$的交叉熵。且使用$w_i$作为背景词出现的词频来加权。最小化这个目标函数会使得预测的条件概率分布靠近真实的条件概率分布，特别是对与出现比较频繁的条件中心词。

## GloVe模型

我们知道交叉熵是常用损失函数一种，另一种本书介绍的常用损失是平方损失：$(p_{ij} - q_{ij})^2$。但直接使用时求导比较复杂。GloVe [1]在这个想法的基础上做了三点改动。

1. 使用没有概率化的$p'_{ij}=x_{ij}$和$q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$，并对其做对数。因此平方损失项是$\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$。
2. 为词$w_i$增加了中心词标量偏移$b_i$和背景词标量偏移$c_i$。
3. 因为平方损失里已经包含了词频信息，所以出现次数多的词更受重视。GloVe将频次权重$x_i$替换成更加缓和的$h(x_{ij})$，这里$h(x)$是值域在$[0,1]$的非递减函数。

这样，GloVe的目标是最小化下面目标函数：

$$-\sum_{i\in\mathcal{V},\ j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$

其中权重函数$h(x)$的一个建议选择是：当$x < c$（例如$c = 100$），令$h(x) = (x/c)^\alpha$（例如$\alpha = 0.75$），反之令$h(x) = 1$。因为$h(0)=0$，所以对于$x_{ij}=0$的平方损失项可以直接忽略。当使用小批量随机梯度下降来训练时，每个时间步我们随机采样小批量非零$x_{ij}$，然后计算梯度来迭代模型参数。

注意到如果$x_i$出现在$x_j$的背景窗口里，那么$x_{j}$也会出现在$x_i$的背景窗口里。也就是说$x_{ij}=x_{ji}$。不同于word2vec中拟合的是非对称的$p_{ij}$，GloVe拟合的是对称的$\log\, x_{ij}$。因此，任意词的中心词向量和背景词向量在GloVe中是等价的。但由于初始化值的不同，同一个词最终学习到的两组词向量可能不同。当学习得到所有词向量以后，GloVe使用中心词向量与背景词向量之和作为该词的最终词向量。对偏移同样如此。

## 使用条件概率比值的解释

我们可以从另外一个方向出发来理解GloVe模型。首先我们来看下面几组条件概率（[1]中称为共现概率）以及它们之间的比值：

* $\mathbb{P}(k \mid \text{ice})$：0.00019（$k$= solid），0.000066（$k$= gas），0.003（$k$= water），0.000017（$k$= fashion）
* $\mathbb{P}(k \mid \text{steam})$：0.000022（$k$= solid），0.00078（$k$= gas），0.0022（$k$= water），0.000018（$k$= fashion）
* $\mathbb{P}(k \mid \text{ice}) / \mathbb{P}(k \mid \text{steam})$：8.9（$k$= solid），0.085（$k$= gas），1.36（$k$= water），0.96（$k$= fashion）


我们可以观察到以下现象：

* 对于与ice（冰）相关而与steam（蒸汽）不相关的词$k$，例如$k=$solid（固体），我们期望条件概率比值较大，例如上面最后一行结果中的值8.9。
* 对于与ice不相关而与steam相关的词$k$，例如$k=$gas（气体），我们期望条件概率比值较小，例如上面最后一行结果中的值0.085。
* 对于与ice和steam都相关的词$k$，例如$k=$water（水），我们期望条件概率比值接近1，例如上面最后一行结果中的值1.36。
* 对于与ice和steam都不相关的词$k$，例如$k=$fashion（时尚），我们期望条件概率比值接近1，例如上面最后一行结果中的值0.96。

由此可见，条件概率比值能比较直观地表达词与词之间的关系。我们可以构造一个词向量函数使得它能有效拟合条件概率比值。我们知道，任意一个这样的比值需要三个词$w_i$、$w_j$和$w_k$。以$w_i$作为中心词的条件概率比值为${p_{ij}}/{p_{ik}}$。我们可以找一个函数，它使用词向量来拟合这个条件概率比值：

$$f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$

这里函数$f$可能的设计并不唯一，我们只需考虑一种较为合理的可能性。注意到条件概率比值是一个标量，我们可以将$f$限制为一个标量函数：$f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) = f\left((\boldsymbol{u}_j - \boldsymbol{u}_k)^\top {\boldsymbol{v}}_i\right)$。交换$j$和$k$后可以看到$f$应该满足$f(x)f(-x)=1$，因此一个可能是$f(x)=\exp(x)$，于是

$$f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) = \frac{\exp\left(\boldsymbol{u}_j^\top {\boldsymbol{v}}_i\right)}{\exp\left(\boldsymbol{u}_k^\top {\boldsymbol{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

满足最右边约等号的一个可能是$\exp\left(\boldsymbol{u}_j^\top {\boldsymbol{v}}_i\right) \approx \alpha p_{ij}$，这里$\alpha$是一个常数。考虑到$p_{ij}=x_{ij}/x_i$，取对数后$\boldsymbol{u}_j^\top {\boldsymbol{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$。我们使用额外的偏移来拟合$- \log\,\alpha + \log\,x_k$，为了对称性，我们同时使用中心词和背景词偏移，那么：

$$\boldsymbol{u}_j^\top \tilde{\boldsymbol{v}}_i + b_i + c_j \approx \log(x_{ij}).$$

之后使用加权的平方误差我们可以得到GloVe的目标函数。

## 小结


* GloVe用词向量表达条件词频的对数，并通过加权平方误差来构建目标函数。


## 练习

* GloVe中，如果一个词出现在另一个词的背景中，是否可以利用它们之间在文本序列的距离重新设计词频计算方式？提示：可参考Glove论文4.2节 [1]。
* 如果丢弃GloVe中的偏差项，是否也可以满足任意一对词条件的对称性？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4372)

![](../img/qr_glove.svg)

## 参考文献

[1] Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
