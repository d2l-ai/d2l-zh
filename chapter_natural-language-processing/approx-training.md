# 近似训练

回忆上节内容。跳字模型的核心在于使用softmax运算得到给定中心词$w_c$来生成背景词$w_o$的条件概率

$$\mathbb{P}(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}。$$

由于softmax运算考虑了背景词可能是词典$\mathcal{V}$中的任一词，所以上式分母中出现了词典项。因此，上式的计算复杂度为$\mathcal{O}(|\mathcal{V}|)$。在上一节中我们看到，不论是跳字模型还是连续词袋模型，由于条件概率使用了softmax运算，每一步的梯度计算都需要遍历词典中所有的词来计算条件概率。对于含几十万或上百万词的较大词典，每次的梯度计算开销可能过大。为了降低该计算复杂度，本节将介绍两个近似训练方法：负采样（negative sampling）或层序softmax（hierarchical softmax）。由于跳字模型和连续词袋模型类似，本节仅以跳字模型为例介绍这两个方法。



## 负采样

负采样修改了原来的目标函数。它将背景词$w_o$出现在中心词$w_c$的一个背景窗口作为一个事件，并将该事件的概率计算为

$$\mathbb{P}(D=1\mid w_c, w_o) = \sigma(\boldsymbol{u_o}^\top \boldsymbol{v_c}),$$

其中的函数

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$

现在我们可以考虑最大化文本序列中所有该事件的联合概率来训练词向量。具体来说，给定一个长度为$T$的文本序列，当背景窗口大小为$m$时，考虑最大化联合概率

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)}).$$

然而，以上模型中包含的事件全是正例样本。结果，当所有词向量相等且值为无穷大时，以上的联合概率将得到最大值1。很明显，这样的词向量毫无意义。负采样通过采样并添加负例样本使目标函数更有意义。对于背景词$w_o$出现在中心词$w_c$的一个背景窗口这样的一个事件（正例样本），我们根据分布$\mathbb{P}(w)$采样$K$个未出现在该背景窗口中的词，即噪音词。每个噪声词不出现在中心词$w_c$的该背景窗口与这个正例样本组成$K + 1$个相互独立事件。


这样，我们得到下面目标函数：

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \left(\mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim \mathbb{P}(w)}^K \mathbb{P}(D=0\mid w^{(t)}, w_k)\right).$$

记词$w^{(t)}$的索引为$i_t$。最大化上面目标函数等价于最小化它的负对数值：

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \left(\log\, \sigma(\boldsymbol{u_{i_{t+j}}}^\top \boldsymbol{v_{i_t}}) + \sum_{k=1,\ w_k \sim \mathbb{P}(w)}^K \log\left(1-\sigma(\boldsymbol{u_{k}}^\top \boldsymbol{v_{i_t}})\right)\right).$$

因为$\sigma$中分母只有两项，所以每次的梯度计算不再跟词典大小相关，而跟$K$线性相关。通常$K$是较小的常数。对于采样分布$\mathbb{P}(w)$，word2vec [1] 使用了$P(w_i)=(c_i/N)^{3/4}$，这里$c_i$是$w_i$在数据中出现的次数，$N$是数据的总词数。这样出现频次更多的词更容易被采样成噪音词。


## 层序softmax

层序softmax是另一种近似训练法。它使用了二叉树这一数据结构，树的每个叶子节点代表着词典$\mathcal{V}$中的每个词。

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
