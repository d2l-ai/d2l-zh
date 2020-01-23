# 微调BERT用于自然语言推理任务

在`通过下游任务微调BERT`一节中，我们介绍了BERT在预训练阶段完成后，要适用于广泛的任务时，如何添加一个额外的输出层，对预训练的 BERT 表示进行微调。在这一节我们将介绍一个例子，如何通过微调BERT进行自然语言推理。


## BERT预训练
在之前的章节中，我们已经写好了BERT的训练函数，我们首先加载“WikiText-103”数据集，并预处理成BERT所需要的形式，再预训练BERT模型。

```{.python .input  n=1}
import d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time

npx.set_np()

bert_train_set = d2l.WikiDataset('wikitext-2', 128)
batch_size, ctx = 512, d2l.try_all_gpus()
bert_train_iter = gluon.data.DataLoader(bert_train_set, batch_size, shuffle=True)

bert = d2l.BERTModel(len(bert_train_set.vocab), embed_size=128, hidden_size=256, num_heads=2,
                     num_layers=2, dropout=0.2)
bert.initialize(init.Xavier(), ctx=ctx)
nsp_loss = gluon.loss.SoftmaxCELoss()
mlm_loss = gluon.loss.SoftmaxCELoss()

d2l.train_bert(bert_train_iter, bert, nsp_loss, mlm_loss, len(bert_train_set.vocab), ctx, 20, 40000)
```

## 在自然语言推理任务上进行微调
我们以之前介绍过的自然语言推理任务为例。现在介绍如何将自然语言推理这个下游任务接入BERT，并在这个下游任务上微调BERT模型。

### 数据预处理

自然语言推理任务本质上是个句对分类任务，所以我们需要将前提句和假设句拼接成一个序列，并在序列开始位置加入"[CLS]"，在每个句子结束位置加入“[SEP]”标记，在片段标记中使用0和1区分两个句子。这里直接使用`BERT的数据预处理及模型训练`一节中定义的“get_tokens_and_segment”函数

我们加载在“自然语言推理及数据集”章节中所提到的斯坦福大学自然语言推理数据集，并重新定义一个自然语言推理数据集类`SNLIBERTDataset`。

```{.python .input  n=65}
# Saved in the d2l package for later use
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, vocab=None):
        self.dataset = dataset
        self.max_len = 50  # 将每条评论通过截断或者补0，使得长度变成50
        self.data = d2l.read_file_snli('snli_1.0_'+ dataset + '.txt')
        self.vocab = vocab
        self.tokens, self.segment_ids, self.valid_lengths, self.labels =  \
                                self.preprocess(self.data, self.vocab)
        print('read ' + str(len(self.tokens)) + ' examples')


    def preprocess(self, data, vocab):
        LABEL_TO_IDX = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        def pad(x):
            return x[:self.max_len] if len(x) > self.max_len \
                                    else x + [0] * (self.max_len - len(x))
        
        tokens, segment_ids, valid_lengths, labels = [], [], [], []
        
        for x in data:
            token, segment_id = d2l.get_tokens_and_segment(x[0][:self.max_len], x[1][:self.max_len])
            valid_length = len(token)
            tokens.append(vocab.to_indices(pad(token)))
            segment_ids.append(np.array(pad(segment_id)))
            valid_lengths.append(np.array(valid_length))
            labels.append(np.array(LABEL_TO_IDX[x[2]]))
            
        return tokens, segment_ids, valid_lengths, labels

    def __getitem__(self, idx):
        return (self.tokens[idx], self.segment_ids[idx], self.valid_lengths[idx]), self.labels[idx]

    def __len__(self):
        return len(self.tokens)
```

通过自定义的`SNLIBERTDataset`类来分别创建训练集和测试集的实例。我们指定最大文本长度为50。下面我们可以分别查看训练集和测试集所保留的样本个数。

```{.python .input  n=66}
d2l.download_snli()
train_set = SNLIBERTDataset("train", bert_train_set.vocab)
test_set = SNLIBERTDataset("test", bert_train_set.vocab)
```

设批量大小为256，分别定义训练集和测试集的迭代器。

```{.python .input  n=67}
batch_size = 256
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(test_set, batch_size)
```

### 用于微调的分类模型

刚才我们已经训练好了BERT模型，我们只需要附加一个额外的层来进行分类。 BERTClassifier类使用BERT模型对句子表示进行编码，然后使用第一个令牌“[CLS]”的编码通过全连接层进行分类。

```{.python .input  n=82}
class BERTClassifier(nn.Block):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(256, flatten=False, activation='relu'))
        self.classifier.add(gluon.nn.Dense(num_classes))

    def forward(self, X):
        inputs, segment_types, seq_len = X
        seq_encoding = self.bert(inputs, segment_types, seq_len)
        return self.classifier(seq_encoding[:, 0, :])
```

初始化网络时要注意的是我们只需要初始化分类层。 因为BERT是使用已经预训练好的权重。

```{.python .input  n=83}
net = BERTClassifier(bert, 3)
net.classifier.initialize(ctx=ctx)
```

现在就可以训练模型了。

```{.python .input  n=87}
lr, num_epochs = 0.00005, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx, d2l.split_batch_multi_inputs)
```

## 小结

- 只需在BERT的输出层上加简单的多层感知机或线性分类器即可接入下游任务。
- 单句分类任务和句对分类任务取“[CLS]”位置的输出表示接入全连接层作为输出。
- 问答任务取第二个句子每个位置的输出表示作为下游任务的输入。
- 序列标注任务取除了特殊标记外其他位置的输出表示接入全连接层作为输出。
