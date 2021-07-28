# 预训练数据集 BERT
:label:`sec_bert-dataset`

要预训练在 :numref:`sec_bert` 中实施的 BERT 模型，我们需要以理想的格式生成数据集，以便于完成两项预训练任务：蒙版语言建模和下一句预测。一方面，原始的 BERT 模型是在两个巨大的语库 Bookcorpus 和英语维基百科（见 :numref:`subsec_bert_pretraining_tasks`）的连接方面进行了预训练，这使得这本书的大多数读者难以运行。另一方面，现成的预训练 BERT 模型可能不适合来自医学等特定领域的应用。因此，在自定义数据集上对 BERT 进行预训练越来越受欢迎。为了促进 BERT 预训练的演示，我们使用了较小的语料库 Wikitext-2 :cite:`Merity.Xiong.Bradbury.ea.2016`。 

与 :numref:`sec_word2vec_data` 中用于预训 word2vec 的 PTB 数据集相比，WikiText-2 (i) 保留了原始标点符号，使其适合于下一句预测；(ii) 保留原始大小写和数字；(iii) 大两倍以上。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

在 Wikitext-2 数据集中，每行代表一个段落，其中任何标点符号和前面的词元之间插入空格。保留至少有两句话的段落。为了分割句子，为了简单起见，我们只使用句点作为分隔符。我们将在本节末尾的练习中讨论更复杂的句子分割技术。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 为预训任务定义助手函数

在下面，我们首先为两个 BERT 预训练任务实施辅助函数：下一句预测和蒙版语言建模。稍后将原始文本语料库转换为预训练 BERT 的理想格式的数据集时，将调用这些辅助函数。 

### 生成下一句预测任务

根据 :numref:`subsec_nsp` 的描述，`_get_next_sentence` 函数生成了二进制分类任务的训练示例。

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

以下函数通过调用 `_get_next_sentence` 函数生成从输入 `paragraph` 进行下一句预测的训练示例。这里 `paragraph` 是一个句子列表，其中每句都是一个令牌列表。参数 `max_len` 指定了预训期间 BERT 输入序列的最大长度。

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### 生成蒙版语言建模任务
:label:`subsec_prepare_mlm_data`

为了根据 BERT 输入序列生成蒙版语言建模任务的训练示例，我们定义了以下 `_replace_mlm_tokens` 函数。在其输入中，`tokens` 是代表 BERT 输入序列的令牌列表，`candidate_pred_positions` 是 BERT 输入序列的令牌索引列表，不包括特殊令牌的索引（在蒙屏语言建模任务中没有预测特殊令牌），`num_mlm_preds` 表示预测数量（回想 15％随机令牌可以预测）。在 :numref:`subsec_mlm` 中对蒙版语言建模任务的定义之后，在每个预测位置，输入可以被特殊的 “<mask>” 令牌或随机令牌替换，或者保持不变。最后，该函数在可能的替换后返回输入令牌、发生预测的令牌索引以及这些预测的标签。

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

通过调用上述 `_replace_mlm_tokens` 函数，以下函数将 BERT 输入序列 (`tokens`) 作为输入序列 (`tokens`) 并返回输入令牌的索引（在可能的令牌替换后，如 :numref:`subsec_mlm` 所述）、发生预测的令牌指数以及词元这些指数的索引预测。

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## 将文本转换为训练前数据集

现在我们已经准备好自定义 `Dataset` 课程，用于预训 BERT。在此之前，我们仍然需要定义一个助手函数 `_pad_bert_inputs` 来将特殊的 “<mask>” 令牌附加到输入中。它的论点 `examples` 包含了帮助函数 `_get_nsp_data_from_paragraph` 和 `_get_mlm_data_from_tokens` 用于两个预训任务的输出。

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

将用于生成两个预训练任务的训练示例的帮助函数和用于填充输入的辅助函数放在一起，我们将以下 `_WikiTextDataset` 类定制为用于预训练 BERT 的 WikiText-2 数据集。通过实现 `__getitem__ ` 函数，我们可以任意访问来自 WikiText-2 语料库的一对句子生成的预训练（蒙面语言建模和下一句预测）示例。 

原来的 BERT 模型使用字体嵌入，其词汇量为 30000 :cite:`Wu.Schuster.Chen.ea.2016`。WordPiece 的词元化方法是对 :numref:`subsec_Byte_Pair_Encoding` 中原来的字节对编码算法的轻微修改。为简单起见，我们使用 `d2l.tokenize` 函数进行词元化。出现少于五次的罕见代币将被过滤掉。

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

通过使用 `_read_wiki` 函数和 `_WikiTextDataset` 类，我们定义了下载以下 `load_data_wiki` 和 WikiText-2 数据集并从中生成预训示例。

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

将批次大小设置为 512，并且 BERT 输入序列的最大长度为 64，我们会打印出一个小批量的 BERT 预训练示例的形状。请注意，在每个 BERT 输入序列中，为蒙版语言建模任务预测 $10$ ($64 \times 0.15$) 的位置。

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

最后，让我们来看看词汇量的大小。即使在过滤掉不常见的令牌之后，它仍比 PTB 数据集大两倍以上。

```{.python .input}
#@tab all
len(vocab)
```

## 摘要

* 与 PTB 数据集相比，Wikitext-2 日期集保留了原始标点符号、大小写和数字，并且大两倍以上。
* 我们可以任意访问来自 WikiText-2 语料库的一对句子生成的预训练（蒙面语言建模和下一句预测）示例。

## 练习

1. 为简单起见，句点被用作分割句子的唯一分隔符。尝试其他句子拆分技术，例如 SPacy 和 NLTK。以 NLTK 为例。你需要先安装 NLTK：`pip install nltk`。在代码中，首先是 `import nltk`。然后，下载 Punkt 句子分词器：`nltk.download('punkt')`。分割诸如 `句子 = '这太棒了！为什么不呢？'`, invoking `nltk.tokenize.sent_tokenize（句子）` will return a list of two sentence strings: ` [“这太棒了！”，“为什么不？”]`。
1. 如果我们不过滤掉任何不常见的令牌，词汇量是多少？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:
