#  可分解注意力模型用于自然语言推理

在上一节中，我们介绍了自然语言推理任务。给定前提句和假设句，该任务判断这两个句子之间的推理关系。推理关系包含三种，即蕴含关系、矛盾关系和中性关系。

本节将介绍自然语言推理任务的经典工作之一：可分解注意力模型（decomposable attention model）[1]。回想`机器翻译`章节，我们将注意力机制应用在机器翻译任务中。在机器翻译任务中，我们可以通过注意力机制来找到寻找单词间对应关系。可分解注意力模型就是注意力机制的作用带来了启发。可分解注意力模型认为仅依赖于两个句子单词间的对应关系就足够得出两个句子的推理关系，而两个句子单词间的对应关系可以通过注意力机制来找到。该模型当时在斯坦福自然语言推理数据集上取得最高的准确率，且其参数量相比其他复杂模型低了一个数量级。

首先导入实验所需的包和模块。

```{.python .input  n=1}
import d2lzh as d2l
import mxnet as mx

from mxnet import autograd, gluon, init, np, npx
from mxnet.contrib import text
from mxnet.gluon import nn

npx.set_np()
```

## 模型

在介绍模型前我们先来看下面三个句子：

- Bob is in his room, but because of the thunder and lightning outside, he cannot sleep.
- Bob is awake.
- It is sunny outside.

由于第一句中的“cannot sleep”和第二句中的“awake”具有同等含义。即使第一句的结构复杂，我们也可以很容易的得出结论：第二句蕴含在第一句之中。同样地，由于第一句中的“thunder and lightning”与第三句中的“sunny”互斥，我们可以得出第一句和第三句矛盾。

从上面的例子中，我们发现前提句和假设句的单词间往往存在同义或互斥关系，通过这种单词间的关系可以得出前提句和假设句的推理关系。

因此自然可以想到一个办法，我们可以将判断前提句与假设句之间推理关系的任务分解成子问题：首先将前提句中的每个单词和假设句中的每个单词建立对应关系。这种在单词间建立对应关系操作也叫做单词对齐。再比较每个单词对齐关系，通过比较结果来判断前提句与假设句间的推理关系。这也就是可分解注意力模型的基本三个步骤：注意、比较和合并。

### 注意（attend）

前提句的文本序列由$ l_A $个词组成，假设句的文本序列由$ l_B $个词组成，词嵌入维度为 $ d $ 。在注意过程中，我们分别输入前提句$\boldsymbol{A}= (a_1,\ldots,a_{l_A}) \in \mathbb{R}^{l_A \times d}$和假设句$\boldsymbol{B} = (b_1,\ldots,b_{l_B}) \in \mathbb{R}^{l_B \times d}$，其中$a_i$和$b_i$分别是前提句A和假设句B中单词的词嵌入表示。

在之前注意力机制章节提到，注意力机制在seq2seq模型中，可以学习到目标序列中的标记与源序列中的标记之间的密切关系，这种两个序列间标记的密切关系本质上也是一种单词对齐关系，所以我们可以使用注意力机制来学习到单词对齐关系。

首先我们需要分别计算$ {a_1,\ldots,a_{l_A}} $ 与 ${b_1,\ldots,b_{l_B}}$ 任意两个词之间未经过归一化的注意力权重矩阵 $ e $。即分别将 $ a_i $ 和 $ b_j $通过前馈网络变换后，再计算内积注意力。

$$
e_{ij} = F(a_i)^\top F(b_j)
$$
下面我们需要对句子$A$进行操作，此时$ \beta_i $ 为B中与$ a_i $相对应的对齐词表示。通俗来说，是将$ a_i $这个词通过$(b_1,\ldots,b_{l_B})$加权组合得到。这种操作叫做软对齐。同样的，也需要对句子$B$进行软对齐操作。
$$
\beta_i = \sum_{j=1}^{l_B}\frac{\exp(e_{i j})}{ \sum_{k=1}^{l_B} \exp(e_{i k})} b_j,
$$
$$
\alpha_j = \sum_{i=1}^{l_A}\frac{\exp(e_{i j})}{ \sum_{k=1}^{l_A} \exp(e_{k j})} a_i,
$$

```{.python .input  n=2}
# 定义前馈神经网络
def ff_layer(out_units, flatten=True):
    m = nn.Sequential()
    m.add(nn.Dropout(0.2))
    m.add(nn.Dense(out_units, activation='relu', 
                   flatten=flatten))
    m.add(nn.Dropout(0.2))
    m.add(nn.Dense(out_units, in_units=out_units, activation='relu', 
                   flatten=flatten))
    return m
    
class Attend(nn.Block):
    def __init__(self, hidden_size, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = ff_layer(out_units=hidden_size, flatten=False)
            
    def forward(self, a, b):
        # 分别将两个句子通过前馈神经网络，维度为(批量大小, 句子长度, 隐藏单元数目)
        f_a = self.f(a)
        f_b = self.f(b)
        # 计算注意力打分e，维度为(批量大小, 句子1长度, 句子2长度)
        e = npx.batch_dot(f_a, f_b, transpose_b=True)
        # 对句子A进行软对齐操作，将句子B对齐到句子A。
        # beta维度为(批量大小, 句子1长度, 隐藏单元数目)
        beta = npx.batch_dot(npx.softmax(e), b)
        # 对句子B进行软对齐操作，将句子A对齐到句子B。
        # alpha维度为(批量大小, 句子2长度, 隐藏单元数目)
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), a)
        return beta, alpha
```

经过了这一步，我们就将问题转化成了对齐后单词对的比较问题。

### 比较（compare）

在注意过程中，前提句中的每个词都和假设句中的词建立了软对齐关系。在比较过程中，我们需要比较这每一个软对齐关系。要比较这些对齐关系。我们需要拼接每一个词表示$a_i$和其对齐词表示$\beta_i$，然后使用一个前馈网络进行变换。我们将变换后的向量称为比较向量。同样地，也对每一个词$b_i$与其对齐词表示$\alpha_i$进行这样的操作。
$$
v_{1,i} = G([a_i, \beta_i])
$$
$$
v_{2,j} = G([b_i, \alpha_i])
$$

```{.python .input  n=3}
class Compare(nn.Block):
    def __init__(self, hidden_size, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = ff_layer(out_units=hidden_size, flatten=False)

    def forward(self, a, b, beta, alpha):
        # 拼接每一个词和其对齐词表示，并通过前馈神经网络
        v1 = self.g(np.concatenate([a, beta], axis=2))
        v2 = self.g(np.concatenate([b, alpha], axis=2))
        return v1, v2
```

### 合并（aggregate）

现在我们有两个比较向量的集合，在这一步中，我们需要将比较向量的集合转化为句子的表示向量。比较容易的办法是对每个集合中的向量取平均作为句子的表示向量。

$$
v_1 = \sum_{i=1}^{l_A}v_{1,i}
$$

$$
v_2 = \sum_{j=1}^{l_B}v_{2,j}
$$

然后我们拼接两个句子的表示向量，并通过前馈网络进行分类。
$$
\hat y = H([v_1,v_2])
$$

```{.python .input  n=4}
class Aggregate(nn.Block):
    def __init__(self, hidden_size, num_class, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = ff_layer(out_units=hidden_size, flatten=True)
        self.h.add(nn.Dense(num_class, in_units=hidden_size))
            
    def forward(self, feature1, feature2):
        # 对每个集合中的向量取平均作为句子的表示向量
        feature1 = feature1.sum(axis=1)
        feature2 = feature2.sum(axis=1)
        # 拼接每个句子的表示，使用前馈网络进行分类
        yhat = self.h(np.concatenate([feature1, feature2], axis=1))
        return yhat
```

## 使用可分解注意力模型

我们将上面的“注意”、“比较”和“合并”这三个过程结合起来。

```{.python .input  n=5}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, hidden_size, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(hidden_size)
        self.compare = Compare(hidden_size)
        self.aggregate = Aggregate(hidden_size, 3)
        
    def forward(self, X):
        premise, hypothesis = X
        a = self.embedding(premise)
        b = self.embedding(hypothesis)
        # 注意（attend）过程
        beta, alpha = self.attend(a ,b)
        # 比较（compare）过程
        v1, v2 = self.compare(a, b, beta, alpha)
        # 合并（aggregate）过程
        yhat = self.aggregate(v1, v2)
        return yhat
```

### 读取数据集

我们使用上一节讲到的斯坦福自然语言推理数据集分别创建训练集和测试集的实例，并定义训练集和测试集的迭代器。

```{.python .input  n=6}
d2l.download_snli(data_dir='../data')
train_set = d2l.SNLIDataset("train")
test_set = d2l.SNLIDataset("test", train_set.vocab)

batch_size = 256
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(test_set, batch_size)
```

创建一个`DecomposableAttention`实例，并设置嵌入层输出大小为100维，隐藏层输出为100维。

```{.python .input  n=7}
embed_size, hidden_size, ctx = 100, 100, d2l.try_all_gpus()
net = DecomposableAttention(train_set.vocab, embed_size, hidden_size)
net.initialize(init.Xavier(), ctx=ctx)
```

### 训练模型

我们直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。这里，我们为词典vocab中的每个词加载100维的GloVe词向量。注意，预训练词向量的维度需要与创建的模型中的嵌入层输出大小embed_size一致。

```{.python .input}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=train_set.vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
```

我们定义一个`split_batch_multi_input`函数，这个函数将具有多个输入的小批量数据样本batch划分并复制到ctx变量所指定的各个显存上。

```{.python .input}
# Saved in the d2l package for later use
def split_batch_multi_input(X, y, ctx_list):
    """Split X and y into multiple devices specified by ctx"""
    X = list(zip(*[gluon.utils.split_and_load(feature, ctx_list, even_split=False)
                   for feature in X]))
    return (X, gluon.utils.split_and_load(y, ctx_list, even_split=False))
```

现在就可以训练模型了。

```{.python .input}
lr, num_epochs = 0.001, 10
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs,
               ctx, split_batch_multi_input)
```

### 评价模型

最后，定义预测函数。

```{.python .input  n=30}
# Saved in the d2l package for later use
def predict_snli(net, premise, hypothesis):
    premise = np.array(train_set.vocab.to_indices(premise), ctx=d2l.try_gpu())
    hypothesis = np.array(train_set.vocab.to_indices(hypothesis), ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)), hypothesis.reshape((1, -1))]), axis=1)
    return 'neutral' if label == 0 else 'contradiction' if label == 1 else 'entailment'
```

下面使用训练好的模型对两个简单句子间的关系进行推理。

```{.python .input  n=30}
predict_snli(net,
        ['A', 'person', 'on', 'a', 'horse', 'jumps', 'over', 'a', 'broken', 'down', 'airplane', '.'],
        ['A', 'person', 'is', 'training', 'his', 'horse', 'for', 'a', 'competition', '.'])
```

## 计算时间复杂度

设隐藏层维度为$d$，前馈网络的计算时间复杂度为$O(d^2)$。对于注意过程，
共需要$O(l)$次通过前馈网络变换，时间复杂度是$O(ld^2)$。计算内积注意力的时间复杂度是$O(l^2d)$。
比较过程共需要$O(l)$次通过单层全连接层变换，故该过程的时间复杂度是$O(ld^2)$。
在合并过程中，只是通过了一次单隐藏层多层感知机，故时间复杂度是$O(d^2)$。
因此总的时间复杂度是$O(l d^2+ l^2 d)$。


## 小结

* 可以使用注意力机制实现词的软对齐。
* 可分解注意力模型将自然语言推理问题转化成了对齐后单词对的比较问题。


## 参考文献

[1] Parikh, A.P., Täckström, O., Das, D., & Uszkoreit, J. (2016). A Decomposable Attention Model for Natural Language Inference. *EMNLP*.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![](../img/qr_sentiment-analysis-cnn.svg)
