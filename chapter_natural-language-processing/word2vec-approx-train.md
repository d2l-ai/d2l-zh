# word2vec的近似训练

在上一节中我们看到，不论是跳字模型还是连续词袋模型，每一步的梯度计算都需要遍历词典中所有的词来计算条件概率。对于较大的词典，每步梯度计算开销会过大。这一节我们介绍两个技术来降低梯度计算的复杂度。因为跳字模型和连续词袋模型非常类似（互换了中心词和背景词顺序），本节仅以跳字模型为例介绍这两个技术。

回忆跳字模型的核心是如何计算给定中心词$w_c$来生成背景词$w_o$的条件概率：

$$\mathbb{P}(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},$$

由于分母中有字典大小这个项，所以导致计算困难。本节要介绍的两个技术均是简化这个分母项。

## 负采样

负采样（negative sampling）考虑的是一个不一样的目标函数。跟跳字模型中使用给定中心词去生成背景词不同，负采样考虑一对词$w_c$和$w_o$是不是可能组成中心词和背景词对，即$w_o$是否可能出现在$w_c$的背景窗口里。我们假设会出现在窗口的概率为

$$\mathbb{P}(D=1\mid w_c, w_o) = \sigma(\boldsymbol{u_o}^\top \boldsymbol{v_c}), \quad \sigma(x) = \frac{1}{1+\exp(-x)}.$$

那么我们可以考虑最大化所有中心词和背景词出现的概率来训练词向量。具体来说，给定一个长度为$T$的文本序列，当背景窗口大小为$m$时，我们最大化下面的联合概率：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)}).$$

但这个模型里只有正例样本，我们可以让所有词向量相等且值为无穷大，这样上面概率等于最大值1。我们可以加入负例样本来使得目标函数更有意义。在负采样中，我们为每个中心词和背景词对根据分布$\mathbb{P}(w)$采样$K$个没有出现在背景窗口中的词，它们被称为噪音词，并最大化它们不出现的概率。这样，我们得到下面目标函数：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \left(\mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim \mathbb{P}(w)}^K \mathbb{P}(D=0\mid w^{(t)}, w_k)\right).$$

记词$w^{(t)}$的索引为$i_t$。最大化上面目标函数等价于最小化它的负对数值：

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \left(\log\, \sigma(\boldsymbol{u_{i_{t+j}}}^\top \boldsymbol{v_{i_t}}) + \sum_{k=1,\ w_k \sim \mathbb{P}(w)}^K \log\left(1-\sigma(\boldsymbol{u_{k}}^\top \boldsymbol{v_{i_t}})\right)\right).$$

因为$\sigma$中分母只有两项，所以每次的梯度计算不再跟词典大小相关，而跟$K$线性相关。通常$K$是较小的常数。对于采样分布$\mathbb{P}(w)$，word2vec [1] 使用了$P(w_i)=(c_i/N)^{3/4}$，这里$c_i$是$w_i$在数据中出现的次数，$N$是数据的总词数。这样出现频次更多的词更容易被采样成噪音词。


## 层序softmax

层序softmax（hierarchical softmax）是另一种近似训练法。它使用了二叉树这一数据结构，树的每个叶子节点代表着词典$\mathcal{V}$中的每个词。

![层序softmax。树的每个叶子节点代表着词典的每个词。](../img/hi-softmax.svg)


假设$L(w)$为从二叉树的根节点到词$w$的叶子节点的路径（包括根和叶子节点）上的节点数。设$n(w,j)$为该路径上第$j$个节点，并设该节点的背景词向量为$\mathbf{u}_{n(w,j)}$。以图10.3为例，$L(w_3) = 4$。层序softmax将跳字模型中的条件概率改写成：

$$\mathbb{P}(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o,j)) ]\!] \cdot \mathbf{u}_{n(w_o,j)}^\top \mathbf{v}_c\right),$$

其中$\text{leftChild}(n)$是节点$n$的左孩子节点；如果判断$x$为真，$[\![x]\!] = 1$；反之$[\![x]\!] = -1$。
让我们计算图10.3中给定词$w_c$生成词$w_3$的条件概率。我们需要将$w_c$的词向量$\mathbf{v}_c$和根节点到$w_3$路径上的非叶子节点向量一一求内积。由于在二叉树中由根节点到叶子节点$w_3$的路径上需要向左、向右、再向左地遍历（图10.3中加粗的路径），我们得到

$$\mathbb{P}(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3,1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3,2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3,3)}^\top \mathbf{v}_c).$$

所以这里我们将$\mathbb{P}(w_o \mid w_c)$表示成从根节点开始，是如何通过$L(w_o)-1$次走左还是走右判断来达到$w_o$。由于$\sigma(x)+\sigma(-x) = 1$，给定中心词$w_c$生成词典$\mathcal{V}$中任一词的条件概率之和为1这一条件也将满足：

$$\sum_{w_o \in \mathcal{V}} \mathbb{P}(w_o \mid w_c) = 1.$$

此外，$L(w_o)-1 = \text{log}_2|\mathcal{V}|$，这样我们将每次迭代的计算开销由$\mathcal{O}(|\mathcal{V}|)$下降至$\mathcal{O}(\text{log}_2|\mathcal{V}|)$。

## 小结

* 负采样将目标函数转换成判断一对词是否构成中心词和背景词对。
* 层序softmax则将条件概率计算修改成对数个词典大小的二元softmax运算。

## 练习

* 有没有其他的噪音词采样方法？
* 推导为什么最后一个公式成立。
* 负采样和层序softmax如何作用到连续词袋模型。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/8135)

![](../img/qr_word2vec-approx-train.svg)

## 参考文献

[1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
