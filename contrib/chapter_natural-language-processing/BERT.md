# 双向语言表征模型（BERT）

在“词嵌入”章节中，我们提到了词向量是用来表示词的向量，也就是说词向量是能够反映出语义的特征。但如Word2Vec这类常用的词嵌入模型在训练完成后，每个单词的词向量就会固定。在之后使用的时候，无论出现单词的上下文是什么，单词的词向量都不会随着上下文发生变化。例如“apple”这个词既有水果的意思，又是一家公司的名字，在不同的上下文中，“apple”这个词的词向量应该不同。这种不会随着上下文发生变化的词向量叫做静态词向量。而我们期待一个好的词向量应该能够随着不同上下文产生变化，这种能够随着上下文语境不同而变化的词向量叫做动态词向量。

既然需要得到随着不同上下文产生变化的动态词向量，我们可以设计一种动态计算词向量的网络。这个网络的输入是每个词的静态词向量，输出是每个词在当前上下文中的词向量。这个网络类似于词嵌入模型，可以预先在大量的语料中进行训练。这种动态计算词向量的网络叫做语言表示模型。

来自Transformer的双向编码器表示（BERT）就是这么一种语言表示模型。该模型首先在大规模语料上来预训练上下文深度双向表示，这一阶段叫做预训练阶段。在适用于下游广泛的任务时，只需要一个额外的输出层，就可以对预训练的 BERT 表示进行微调，而无需对特定于任务进行大量模型结构的修改。

首先导入实验所需的包和模块。

```{.python .input  n=1}
import d2lzh as d2l
import os
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

## 模型结构

BERT的基础模型结构是在“Transformer”章节中描述的多层双向Transformer编码器。原始的Transformer包括编码器和解码器部分。由于BERT是语言表示模型，目标是对文本序列进行编码表示，因此只需要编码器机制。
BERT分为Base和Large两个版本。Base版本包含12层Transformer，有110M的参数。Large版本包含24层Transformer，有340M的参数。

### 输入表示

BERT的输入支持单个句子或一对句子。分别适用于单句任务（如文本分类任务）和句对任务（如自然语言推理任务）。BERT的输入包含三部分，分别是令牌嵌入、片段嵌入、位置嵌入。

令牌嵌入（Token Embeddings）是将各个词转换成固定维度的向量。首先在序列的开始位置加入特殊标记“[CLS]”，在序列的结束位置加入特殊标记“[SEP]”。如果有两个句子，直接拼接在一起，在每个句子序列的结束位置都加入“[SEP]”。在BERT中，每个词会被转换成768维的向量表示。

片段嵌入（Segment Embeddings）是为了使BERT能够处理句对的输入。句子对中的两个句子被简单的拼接在一起作为输入，因为我们需要使模型能够区分一个句子对中的两个句子，这就是片段嵌入的作用。片段嵌入只有两种向量表示，把向量0给第一个句子序列中的每个令牌，把向量1给第二个句子序列中的每个令牌。如果是输入仅仅有一个句子，那序列中的每个令牌的片段嵌入都是向量0。向量0和向量1都是在训练过程中更新得到的。每个向量都是768维，所以片段嵌入层的大小是（2，768）。

位置嵌入（Position Embeddings）。为了解决Transformer无法编码序列的问题，我们引入了位置嵌入。在BERT中的位置嵌入与Transformer里的位置嵌入稍有不同。Transformer里的位置嵌入是通过公式计算得到的，而BERT中的位置嵌入是在各个位置上学习一个向量表示，从而来将顺序的信息编码进来。BERT最长能处理512个令牌的序列，所以位置嵌入层的大小是（512，768）。

![输入表示](../img/bert_inputs.svg)

对于一个长度为n的输入序列，我们将有令牌嵌入（n，768）用来表示句子中的词，片段嵌入（n，768）用来区分两个句子，位置嵌入（n，768）用来学习到顺序信息。将这三种嵌入按元素相加，得到一个（n，768）的表示，这一表示就是BERT的输入。

我们修改“Transformer”中的TransformerEncoder方法，加入BERT所需要的令牌嵌入、片段嵌入、位置嵌入。

```{.python .input  n=14}
# Save to the d2l package.
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, units, hidden_size,
                 num_heads, num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.segment_embed = gluon.nn.Embedding(2, units)
        self.word_embed = gluon.nn.Embedding(vocab_size, units)
        self.pos_encoding = d2l.PositionalEncoding(units, dropout)
        self.blks = gluon.nn.Sequential()
        for i in range(num_layers):
            self.blks.add(d2l.EncoderBlock(units, hidden_size, num_heads, dropout))

    def forward(self, words, segments, mask, *args):
        X = self.word_embed(words) + self.segment_embed(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, mask)
        return X
```

现在我们模拟一个句对数据输入测试这个TransformerEncoder，每个句子对包含8个单词，使用随机整数代表不同的单词。

```{.python .input  n=15}
encoder = BERTEncoder(vocab_size=10000, units=768, hidden_size=1024,
                      num_heads=4, num_layers=2, dropout=0.1)
encoder.initialize()

num_samples, num_words = 2, 8
# 随机生成单词用于测试
words = np.array([[24070, 25855, 17552, 25326, 9637, 19443, 25959, 23623],
 [7129, 24248, 23612, 14431, 1140, 10231, 4587, 11968]])
# 我们使用0来表示对应单词来自第一个句子，使用1表示对应单词第二个句子
segments = np.array([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,1]])

encodings = encoder(words, segments, None)
print(encodings.shape)  # (批量大小, 单词数, 嵌入大小)
```

## 预训练任务

BERT包含两个预训练任务：下一句预测、遮蔽语言模型。


### 遮蔽语言模型（mask-lm）
一般来说语言表示模型只能从左到右或者从右到左的单向训练。因为如果允许双向训练就意味着会使得每个词在多层的网络中间接地“看到自己”。
为了训练深度双向的表示，BERT设计了一种名为遮蔽语言模型的任务。这个任务类似于完形填空的猜词任务。具体来说，就是随机将一定比例的输入标记替换为遮蔽标记“[MASK]”，然后预测这些被遮蔽的标记。即将遮蔽标记对应的输入隐藏向量输入一个单层网络，用softmax计算词汇表中每个单词的概率，以预测遮蔽标记应该对应哪个词。

创建遮蔽语言模型的预测模型，模型需要重建被掩蔽的单词，我们使用gather_nd来选择代表遮蔽位置令牌的向量。然后将遮蔽位置令牌的向量通过一个前馈网络，以预测词汇表中所有单词的概率分布。

![遮蔽语言模型](../img/bert_mlm.svg)

```{.python .input  n=16}
# Save to the d2l package.
class MaskLMDecoder(nn.Block):
    def __init__(self, vocab_size, units, **kwargs):
        super(MaskLMDecoder, self).__init__(**kwargs)
        self.decoder = gluon.nn.Sequential()
        self.decoder.add(gluon.nn.Dense(units, flatten=False, activation='relu'))
        self.decoder.add(gluon.nn.LayerNorm())
        self.decoder.add(gluon.nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions, *args):
        
        batch_size = X.shape[0]
        num_masked_positions = masked_positions.shape[1]
        ctx = masked_positions.context
        dtype = masked_positions.dtype
        batch_idx = np.arange(0, batch_size, dtype=dtype, ctx=ctx)
        batch_idx = np.repeat(batch_idx, num_masked_positions)
        batch_idx = batch_idx.reshape((1, -1))
        masked_positions = masked_positions.reshape((1, -1))
        
        position_idx = np.concatenate([batch_idx, masked_positions], axis=0)
        encoded = X[position_idx[0,:],position_idx[1,:]]
        encoded = encoded.reshape((batch_size, num_masked_positions, X.shape[-1])).as_np_ndarray()
        pred = self.decoder(encoded)
        return pred
```

下面我们生成一些随机单词作为演示标签。 我们使用SoftmaxCrossEntropyLoss作为损失函数。 然后将预测结果和真实标签传递给损失函数。

```{.python .input  n=17}
mlm_decoder = MaskLMDecoder(vocab_size=30000, units=768)
mlm_decoder.initialize()

mlm_positions = np.array([[0,1],[4,8]])
mlm_label = np.array([[100, 200],[100, 200]])
mlm_pred = mlm_decoder(encodings, mlm_positions)  # (批量大小, 遮蔽数目, 词表大小)
mlm_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_loss = mlm_loss_fn(mlm_pred, mlm_label)
print(mlm_pred.shape, mlm_loss.shape)
```

### 下一句预测

在自然语言处理中有很多下游任务是建立在理解两个句子之间关系的基础上。比如自然语言推理任务。这并不是语言模型所能直接学习到的。为了能够学习到句子间关系，BERT设计了一个预测下一句的二分类任务，即预测输入的两个句子是否为连续的文本。具体就是为每个训练样本选择句子A和B时，50%的概率B是A真实的下一句，有一半的概率使用来自语料库的随机句子替换句子B。
> 输入：[CLS] the man went to [MASK] store [SEP]
> he bought a gallon [MASK] milk [SEP]
> 标签：IsNext

> 输入： [CLS] the man [MASK] to the store [SEP]
> penguin [MASK] are flight ##less birds [SEP]
> 标签：NotNext

在训练时，将“[CLS]”标记的输出送入一个单层网络，并使用softmax计算“是下一句”标签的概率，以判断句子是否是当前句子的下一句。使用“[CLS]”是因为Transformer是可以把全局信息编码进每个位置，因此“[CLS]”位置的输出表示可以包含整个输入序列的特征。

我们设计下一句预测任务的模型，我们将编码后的结果传递给多层感知机以获得下一个句子预测。

![下一句预测](../img/bert_nsp.svg)

```{.python .input  n=18}
# Save to the d2l package.
class NextSentenceClassifier(nn.Block):
    def __init__(self, units=768, **kwargs):
        super(NextSentenceClassifier, self).__init__(**kwargs)
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(units=units, flatten=False, activation='tanh'))
        self.classifier.add(gluon.nn.Dense(units=2, flatten=False))

    def forward(self, X, *args):
        X = X[:, 0, :]  # 获取第一个令牌“[CLS]”的编码
        return self.classifier(X)
```

下一句预测是二分类问题，我们依然使用交叉熵作为损失函数。 我们将编码结果传递给`NextSentenceClassifier`以获得下一句预测结果。 我们使用1作为真实下一句的标签，否则使用0。 然后将预测结果和真实标签传递给损失函数。

```{.python .input  n=19}
ns_classifier = NextSentenceClassifier()
ns_classifier.initialize()

ns_pred = ns_classifier(encodings)  # (批量大小, 1)
ns_label = np.array([0, 1])  # 标签1为真实的下一句，标签0为随机的下一句
ns_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
ns_loss = ns_loss_fn(ns_pred, ns_label)
print(ns_pred.shape, ns_loss.shape)
```

## 构建模型

我们将上面的从Transfomer中修改得到的TransformerEncoder，以及下一句任务预测模型、遮蔽语言模型合并到一起，得到BERT模型。

```{.python .input  n=20}
# Save to the d2l package.
class BERTModel(nn.Block):
    def __init__(self, vocab_size=None, embed_size=128, hidden_size=512, num_heads=2, num_layers=4, dropout=0.1):
        super(BERTModel, self).__init__()
        self._vocab_size = vocab_size
        self.encoder = BERTEncoder(vocab_size=vocab_size, units=embed_size, hidden_size=hidden_size,
                      num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        
        self.ns_classifier = NextSentenceClassifier()
        self.mlm_decoder = MaskLMDecoder(vocab_size=vocab_size, units=embed_size)

    def forward(self, inputs, token_types, valid_length=None, masked_positions=None):

        seq_out = self.encoder(inputs, token_types, valid_length)

        next_sentence_classifier_out = self.ns_classifier(seq_out)
        
        mlm_decoder_out = self.mlm_decoder(seq_out, masked_positions)
        
        return seq_out, next_sentence_classifier_out, mlm_decoder_out
```

## 小结

- 随着上下文语境不同而变化的词向量叫做动态词向量。
- BERT旨在通过训练上下文来预训练深度双向表示。
- BERT的基础模型结构是在“Transformer”章节中描述的多层双向Transformer编码器。
- 遮蔽语言模型是随机将输入标记遮蔽，然后预测这些被遮蔽的标记。
- 预测下一句的二分类任务，预测输入的两个句子是否为连续的文本。

## 参考文献

[1] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![]()
