# 预训 word 2vec
:label:`sec_word2vec_pretraining`

我们继续实施 :numref:`sec_word2vec` 中定义的跳过图模型。然后我们将在 PTB 数据集上使用负采样来预训练 word2vec。首先，让我们通过调用 `d2l.load_data_ptb` 函数来获取该数据集的数据迭代器和词汇，该函数在 :numref:`sec_word2vec_data` 中描述了

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## 跳过格兰氏模型

我们通过使用嵌入层和批量矩阵乘法来实现跳过图模型。首先，让我们回顾一下嵌入图层的工作原理。 

### 嵌入层

如 :numref:`sec_seq2seq` 所述，嵌入图层将词元的索引映射到其要素矢量。此图层的权重是一个矩阵，其行数等于字典大小 (`input_dim`)，列数等于每个词元的矢量维度 (`output_dim`)。在训练一个词嵌入模型之后，这种权重就是我们所需要的。

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

嵌入图层的输入是词元的索引（单词）。对于任何词元索引 $i$，可以从嵌入图层中权重矩阵的 $i^\mathrm{th}$ 行中获得其矢量表示形式。由于矢量维度 (`output_dim`) 设置为 4，因此嵌入图层将返回形状 (2、3、4) 的向量，以表示形状 (2, 3) 的小批词元索引（2、3）。

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 定义正向传播

在正向传播中，跳过图模型的输入包括形状的中心词索引 `center`（批次大小，1）和形状的连接上下文和噪声词索引 `contexts_and_negatives`（批次大小，`max_len`），其中 `max_len` 在 :numref:`subsec_word2vec-minibatch-loading` 中定义了 `max_len`。这两个变量首先通过嵌入层从词元索引转换为矢量，然后它们的批量矩阵乘法（在 :numref:`subsec_batch_dot` 中描述）返回形状输出（批次大小，1，`max_len`）。输出中的每个元素都是中心单词矢量和上下文或噪声单词矢量的点积。

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

让我们打印这个 `skip_gram` 函数的输出形状以获取一些示例输入。

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 训练

在用负采样训练跳过图模型之前，首先让我们定义它的损失函数。 

### 二进制交叉熵损失

根据 :numref:`subsec_negative-sampling` 中负取样的损失函数的定义，我们将使用二进制交叉熵损失。

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

回想一下我们在 :numref:`subsec_word2vec-minibatch-loading` 中对掩码变量和标签变量的描述。以下计算给定变量的二进制交叉熵损失。

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

下面显示了如何在二进制交叉熵损失中使用 sigmoid 激活函数计算上述结果（以较低效率的方式）。我们可以将这两个输出视为两个标准化损失，平均值超过非蒙面预测。

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### 初始化模型参数

当词汇表中的所有单词分别用作中心词和上下文词时，我们为词汇中的所有单词定义了两个嵌入层。单词矢量维度 `embed_size` 设置为 100。

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 定义训练循环

下面定义了训练循环。由于存在填充，损失函数的计算与之前的训练函数略有不同。

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

现在我们可以使用负取样训练跳过图模型。

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 应用单词嵌入

在训练 word2vec 模型之后，我们可以使用训练模型中单词矢量的余弦相似性，从字典中查找语义上与输入词最相似的单词。

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## 摘要

* 我们可以使用嵌入层和二进制交叉熵损失来训练一个带负采样的跳过图模型。
* 单词嵌入的应用包括根据单词矢量的余弦相似性为给定单词寻找语义上相似的单词。

## 练习

1. 使用训练的模型，为其他输入词找到语义上相似的单词。你能通过调整超参数来改善结果吗？
1. 当训练语料库庞大时，我们经常在更新模型参数 * 时对当前迷你表中的中心单词采样上下文单词和噪声单词。换句话说，同一个中心单词在不同的训练时期可能有不同的上下文词或噪音词。这种方法有什么好处？尝试实施这种训练方法。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
