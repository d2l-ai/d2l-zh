# 来自变形金刚（BERT）的双向编码器表示
:label:`sec_bert`

我们引入了几个单词嵌入模型来理解自然语言。训练前后，可以将输出视为矩阵，其中每一行都是表示预定义词汇的一个单词的矢量。事实上，这些单词嵌入模型都是 * 上下文无关 *。让我们首先说明这个财产。 

## 从上下文独立到上下文敏感

回想一下 :numref:`sec_word2vec_pretraining` 和 :numref:`sec_synonyms` 中的实验。例如，word2vec 和 GLOVE 都将相同的预训练向量分配给同一个单词，而不管单词的上下文如何（如果有）。从形式上来说，任何令牌 $x$ 的上下文无关表示是一个函数 $f(x)$，它只需要 $x$ 作为输入。鉴于自然语言中的多聚论和复杂语义的丰富性，与上下文无关的表示有明显的局限性。例如，上下文 “起重机正在飞行” 和 “起重机驾驶员来了” 中的 “起重机” 一词具有完全不同的含义；因此，根据上下文，同一个词可能被分配不同的表示形式。 

这激励了 * 上下文敏感的 * 单词表示形式的开发，其中单词的表示取决于它们的上下文。因此，令牌 $x$ 的上下文相关表示是函数 $f(x, c(x))$，具体取决于 $x$ 及其上下文 $c(x)$。受欢迎的上下文相关表示包括 tagLM（语言模型增强序列词元器）:cite:`Peters.Ammar.Bhagavatula.ea.2017`、Cove（上下文向量）:cite:`McCann.Bradbury.Xiong.ea.2017` 和 elMO（来自语言模型的嵌入）:cite:`Peters.Neumann.Iyyer.ea.2018`。 

例如，通过将整个序列作为输入，elMO 是一个函数，它为输入序列中的每个单词分配一个表示形式。具体来说，elMO 将预训练的双向 LSTM 中的所有中间图层表示法合并为输出表示法。然后，elMO 表示法将作为附加功能添加到下游任务的现有监督模型中，例如连接现有模型中的 elMO 表示法和原始表示法（例如 GLOVE）。一方面，在添加 elMO 表示之后，预训练的双向 LSTM 模型中的所有权重都会被冻结。另一方面，现有的受监督模型是专门针对给定任务定制的。当时利用不同的最佳模型来处理不同的任务，增加 ELMO 改善了六个自然语言处理任务的最新状态：情绪分析、自然语言推断、语义角色词元、共引解析、命名实体识别和问题回答。 

## 从特定于任务到不可知的任务

尽管 elMO 显著改进了针对各种自然语言处理任务的解决方案，但每个解决方案仍然取决于 * 任务特定的 * 架构。但是，为每个自然语言处理任务设计一个特定的架构实际上并不平凡。GPT（生成预训练）模型代表着为上下文相关表示 :cite:`Radford.Narasimhan.Salimans.ea.2018` 设计一个通用 * 任务无关 * 模型的努力。GPT 建立在变压器解码器之上，预先训练将用于表示文本序列的语言模型。将 GPT 应用于下游任务时，语言模型的输出将被输入添加的线性输出图层，以预测任务的标签。与冻结预训练模型参数的 elMO 形成鲜明对比，GPT 在监督学习下游任务期间对预训练的变压器解码器中的 * 所有参数进行了微调。GPT 在自然语言推断、问答、句子相似性和分类等十二项任务上进行了评估，并在对模型架构的改动最小的情况下改善了其中 9 项任务的最新状态。 

但是，由于语言模型的自回归性质，GPT 只是向前（从左到右）。在 “我去银行存款现金” 和 “我去银行坐下来” 的情况下，由于 “银行” 对左边的情境敏感，GPT 将返回 “银行” 的相同表述，尽管它有不同的含义。 

## BERT：结合两全其美

正如我们所看到的那样，elMO 以双向方式对上下文进行编码，但使用特定于任务的架构；虽然 GPT 与任务无关，但是对上下文进行了从左到右编码。BERT（来自变形金刚的双向编码器表示）结合了两全其美的结合，对于范围广泛的自然语言处理任务 :cite:`Devlin.Chang.Lee.ea.2018`，对于上下文的双向编码器表示法，只需最少的体系结构更改。使用预训练的变压器编码器，BERT 能够根据其双向上下文表示任何令牌。在监督下游任务学习期间，BERT 在两个方面与 GPT 类似。首先，BERT 表示将被输入添加的输出层，根据任务的性质对模型架构进行最小的更改，例如对每个令牌的预测与整个序列的预测。其次，预训练的变压器编码器的所有参数都经过微调，而额外的输出层将从头开始训练。:numref:`fig_elmo-gpt-bert` 描述了 elMO、GPT 和 BERT 之间的差异。 

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT 进一步改善了十一项自然语言处理任务的最新状态，这些类别包括：(i) 单一文本分类（例如情绪分析）、（ii）文本对分类（例如自然语言推理）、（iii）问答、（iv）文本词元（例如，指定实体识别）。所有这些都在 2018 年提出，从上下文敏感的 elMO 到与任务无关的 GPT 和 BERT，概念上简单但经验强大的自然语言深度表示预训练，彻底改变了各种自然语言处理任务的解决方案。 

在本章的其余部分，我们将深入研究 BERT 的预培训。当 :numref:`chap_nlp_app` 中解释自然语言处理应用程序时，我们将说明对下游应用程序的 BERT 的微调。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 输入表示法
:label:`subsec_bert_input_rep`

在自然语言处理中，某些任务（例如情绪分析）将单个文本作为输入，而在其他一些任务（例如自然语言推断）中，输入是一对文本序列。BERT 输入序列明确表示单个文本对和文本对。在前者中，BERT 输入序列是特殊分类词元 “<cls>”、文本序列的词元和特殊分隔令牌 “<sep>” 的串联。在后者中，BERT 输入序列是 “<cls>”、第一个文本序列的词元 “<sep>”、第二个文本序列的词元和 “<sep>” 的连接。我们将始终将术语 “BERT 输入序列” 与其他类型的 “序列” 区分开来。例如，一个 *BERT 输入序列 * 可能包含一个 * 文本序列 * 或两个 * 文本序列 *。 

为了区分文本对，学习的细分嵌入 $\mathbf{e}_A$ 和 $\mathbf{e}_B$ 分别添加到第一个序列和第二序列的令牌嵌入中。对于单个文本输入，只使用 $\mathbf{e}_A$。 

以下 `get_tokens_and_segments` 以一句或两句话作为输入，然后返回 BERT 输入序列的词元及其对应的段 ID。

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT 选择变压器编码器作为其双向架构。在变压器编码器中常见，位置嵌入在 BERT 输入序列的每个位置都添加。但是，与原来的变压器编码器不同，BERT 使用 * 可学习 * 位置嵌入。总而言之，:numref:`fig_bert-input` 显示，BERT 输入序列的嵌入是令牌嵌入、区段嵌入和位置嵌入的总和。 

![BERT 输入序列的嵌入是令牌嵌入、区段嵌入和位置嵌入的总和。](../img/bert-input.svg) :label:`fig_bert-input` 

以下 `BERTEncoder` 类与 :numref:`sec_transformer` 中实施的 `TransformerEncoder` 类类似。与 `TransformerEncoder` 不同，`BERTEncoder` 使用细分嵌入和可学习的位置嵌入。

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

假设词汇量大小是 10000。为了演示 `BERTEncoder` 的前向推理，让我们创建它的实例并初始化其参数。

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

我们将 `tokens` 定义为 2 个长度为 8 的 BERT 输入序列，其中每个词元都是词汇的索引。输入 `BERTEncoder` 的 `BERTEncoder` 和输入 `tokens` 返回编码结果，其中每个令牌由超参数 `num_hiddens` 预定义的向量表示，其长度由超参数 `num_hiddens` 预定义。此超参数通常称为变压器编码器的 * 隐藏大小 *（隐藏单位数）。

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 培训前任务
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder` 的前向推断给出了输入文本的每个令牌的 BERT 表示以及插入的特殊词元 “<cls>” 和 “<seq>”。接下来，我们将使用这些表示法来计算预训练 BERT 的损失函数。预培训由以下两项任务组成：蒙版语言建模和下一句话预测。 

### 蒙面语言建模
:label:`subsec_mlm`

如 :numref:`sec_language_model` 所示，语言模型使用左侧的上下文来预测令牌。为了对上下文进行双向编码以表示每个令牌，BERT 会随机掩盖令牌，并使用双向上下文中的令牌以自我监督的方式预测被掩码的令牌。此任务被称为 * 蒙面语言模型 *。 

在此预训任务中，15％ 的代币将随机选择作为预测的蒙面代币。要在不使用标签作弊的情况下预测蒙面的令牌，一种简单的方法是始终在 <mask>BERT 输入序列中用特殊的 “” 令牌替换它。但是，人为的特殊令牌 “<mask>” 永远不会出现在微调中。为避免预训和微调之间的这种不匹配，如果词元被掩盖进行预测（例如，在 “这部电影很棒” 中选择了 “很棒” 来掩盖和预测），则在输入内容中将替换为： 

* <mask>80％ 的时间里，一个特殊的 “” 令牌（例如，“这部电影很棒” 变成 “这部电影是”<mask>）；
* 10％ 的时间内随机令牌（例如，“这部电影很棒” 变成 “这部电影很喝”）；
* 10％ 的时间内不变的标签令牌（例如，“这部电影很棒” 变成 “这部电影很棒”）。

请注意，在 15％ 的时间里，插入随机令牌的 10％。这种偶尔的噪音鼓励 BERT 在双向上下文编码中减少对蒙面令牌的偏见（特别是当标签令牌保持不变时）。 

我们实施了以下 `MaskLM` 课程来预测 BERT 预训的蒙面语言模型任务中的蒙面令牌。该预测使用一个隐藏层 MLP（`self.mlp`）。在前向推断中，它需要两个输入：`BERTEncoder` 的编码结果和用于预测的代币位置。输出是这些仓位的预测结果。

```{.python .input}
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

为了演示 `MaskLM` 的前向推断，我们创建了它的实例 `mlm` 并对其进行初始化。回想一下，`encoded_X` 从前向推断 `BERTEncoder` 代表 2 个 BERT 输入序列。我们将 `mlm_positions` 定义为在 `encoded_X` 的 BERT 输入序列中要预测的 3 个指数。`mlm` 的前瞻推断回报预测结果为 `mlm_Y_hat`，在 `encoded_X` 的所有蒙面仓位 `mlm_positions`。对于每个预测，结果的大小等于词汇量大小。

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

通过掩码下的预测令牌 `mlm_Y_hat` 的地面真相标签 `mlm_Y`，我们可以计算 BERT 预训练中蒙面语言模型任务的交叉熵损失。

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### 下一句预测
:label:`subsec_nsp`

虽然蒙版语言建模能够对表示单词的双向上下文进行编码，但它并没有明确建模文本对之间的逻辑关系。为了帮助理解两个文本序列之间的关系，BERT 在其预训中考虑了二进制分类任务 * 下一句预测 *。在为预训生成句子对时，有一半时间它们确实是带有 “True” 标签的连续句子；而另一半时间，第二个句子是从标有 “False” 标签的语料库中随机抽取的。 

接下来的 `NextSentencePred` 类使用一个隐藏层 MLP 来预测第二句是否是 BERT 输入序列中第一句的下一句。由于变压器编码器中的自我注意力，特殊令牌 “<cls>” 的 BERT 表示对输入的两个句子进行了编码。因此，MLP 分类器的输出层 (`self.output`) 采用 `X` 作为输入，其中 `X` 是 MLP 隐藏层的输出，其输入是编码的 “<cls>” 令牌。

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

我们可以看到，`NextSentencePred` 实例的前向推断返回每个 BERT 输入序列的二进制预测。

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

还可以计算两个二进制分类的交叉熵损失。

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

值得注意的是，上述两项预培训任务中的所有标签都可以在没有人工标签的情况下从培训前语料库中轻而易举地获得。原来的 BERT 已经在 Bookcorpus :cite:`Zhu.Kiros.Zemel.ea.2015` 和英语维基百科的连接方面进行了预培训。这两个文本语句是巨大的：它们分别有 8 亿个单词和 25 亿个单词。 

## 把所有东西放在一起

在预训练 BERT 时，最终损失函数是掩码语言建模的损失函数和下一句预测的线性组合。现在我们可以通过实例化三个类 `BERTEncoder`、`MaskLM` 和 `NextSentencePred` 来定义 `BERTModel` 类。前向推理返回编码的 BERT 表示 `encoded_X`、对蒙面语言建模 `mlm_Y_hat` 的预测以及下一句预测 `nsp_Y_hat`。

```{.python .input}
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 摘要

* Word2vec 和 Glove 等单词嵌入模型与上下文无关。无论单词的上下文如何（如果有），它们都会将相同的预训练向量分配给同一个单词。他们很难以很好地处理自然语言中的多聚结或复杂的语义。
* 对于上下文相关的单词表示（例如 elMO 和 GPT），单词的表示取决于它们的上下文。
* elMO 以双向方式对上下文进行编码，但使用特定于任务的架构（但是，为每个自然语言处理任务设计一个特定的架构实际上并不平凡）；而 GPT 与任务无关，但是从左到右编码上下文。
* BERT 结合了两全其美：它以双向方式编码上下文，对于各种自然语言处理任务，只需最少的体系结构更改。
* BERT 输入序列的嵌入是令牌嵌入、区段嵌入和位置嵌入的总和。
* 培训前 BERT 由两项任务组成：蒙面语言建模和下一句话预测。前者能够对表示单词的双向上下文进行编码，而后者则明确建模文本对之间的逻辑关系。

## 练习

1. 为什么 BERT 会成功？
1. 所有其他事情都相同，蒙面语言模型是否需要比从左到右语言模型需要更多或更少的预训步骤才能收敛？为什么？
1. 在 BERT 的最初实现中，`BERTEncoder`（通过 `d2l.EncoderBlock`）中的定位前馈网络和 `MaskLM` 中的完全连接层都使用高斯误差线性单元 (GELU) :cite:`Hendrycks.Gimpel.2016` 作为激活函数。研究 GELU 和 RELU 之间的区别。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
