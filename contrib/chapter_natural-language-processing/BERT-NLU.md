# 微调BERT用于自然语言推理任务

在上一节中，我们介绍了BERT在预训练阶段完成后，要适用于广泛的任务时，如何添加一个额外的输出层，对预训练的 BERT 表示进行微调。即如何将BERT接入下游任务。在这一节我们将介绍，如何通过微调BERT进行自然语言推理。


## BERT预训练
在之前的章节中，我们已经写好了BERT的训练函数，我们首先加载“WikiText-103”数据集，并预处理成BERT所需要的形式，再预训练BERT模型。

```{.python .input  n=1}
import d2lzh as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import data as gdata, nn, utils as gutils
import time

npx.set_np()

bert_train_set = d2l.WikiDataset(128)
batch_size, ctx = 16, d2l.try_all_gpus()
bert_train_iter = gdata.DataLoader(bert_train_set, batch_size, shuffle=True)

bert = d2l.BERTModel(len(bert_train_set.vocab), embed_size=128, hidden_size=512, num_heads=2, num_layers=4, dropout=0.1)
bert.initialize(init.Xavier(), ctx=ctx)
nsp_loss = gluon.loss.SoftmaxCELoss()
mlm_loss = gluon.loss.SoftmaxCELoss()

d2l.train_bert(bert_train_iter, bert, nsp_loss, mlm_loss, len(bert_train_set.vocab), ctx, 20, 1)
```

```{.json .output n=1}
[
 {
  "ename": "UserWarning",
  "evalue": "Gradient of Parameter `embedding0_weight` on context gpu(1) has not been updated by backward since last `step`. This could mean a bug in your model that made it only use a subset of the Parameters (Blocks) for this iteration. If you are intentionally only using a subset, call step with ignore_stale_grad=True to suppress this warning and skip updating of Parameters with stale gradient",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mUserWarning\u001b[0m                               Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-1-ca9183ab8f83>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0mmlm_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgluon\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSoftmaxCELoss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 19\u001b[0;31m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_bert\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbert_train_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbert\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnsp_loss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmlm_loss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbert_train_set\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvocab\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m20\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m/data/home/ubuntu/d2l-zh/d2l/d2l.py\u001b[0m in \u001b[0;36mtrain_bert\u001b[0;34m(data_eval, net, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, max_step)\u001b[0m\n\u001b[1;32m   1746\u001b[0m                 \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1747\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1748\u001b[0;31m             \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1749\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1750\u001b[0m             \u001b[0mrunning_mlm_loss\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0ml\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0ml\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mls_mlm\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/data/anaconda/envs/py36/lib/python3.6/site-packages/mxnet/gluon/trainer.py\u001b[0m in \u001b[0;36mstep\u001b[0;34m(self, batch_size, ignore_stale_grad)\u001b[0m\n\u001b[1;32m    330\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    331\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_allreduce_grads\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 332\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_update\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mignore_stale_grad\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    333\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    334\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mallreduce_grads\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/data/anaconda/envs/py36/lib/python3.6/site-packages/mxnet/gluon/trainer.py\u001b[0m in \u001b[0;36m_update\u001b[0;34m(self, ignore_stale_grad)\u001b[0m\n\u001b[1;32m    414\u001b[0m                             \u001b[0;34m\"call step with ignore_stale_grad=True to suppress this \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    415\u001b[0m                             \u001b[0;34m\"warning and skip updating of Parameters with stale gradient\"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 416\u001b[0;31m                             %(param.name, str(data.context)))\n\u001b[0m\u001b[1;32m    417\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    418\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_kvstore\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_update_on_kvstore\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mUserWarning\u001b[0m: Gradient of Parameter `embedding0_weight` on context gpu(1) has not been updated by backward since last `step`. This could mean a bug in your model that made it only use a subset of the Parameters (Blocks) for this iteration. If you are intentionally only using a subset, call step with ignore_stale_grad=True to suppress this warning and skip updating of Parameters with stale gradient"
  ]
 }
]
```

## 在自然语言推理任务上进行微调
我们以之前介绍过的自然语言推理任务为例，介绍如何将下游任务接入BERT，并在下游任务上微调BERT模型。

### 数据预处理

自然语言推理任务本质上是个句对分类任务，所以我们需要将前提句和假设句拼接成一个序列，并在序列开始位置加入"[CLS]"，在每个句子结束位置加入“[SEP]”标记，在片段标记中使用0和1区分两个句子。这里可以直接使用上一节中定义的“get_tokens_and_segment”函数

我们加载在“自然语言推理及数据集”章节中所提到的斯坦福大学自然语言推理数据集，并重新定义一个自然语言推理数据集类`SNLIBERTDataset`。

```{.python .input  n=65}
# Save to the d2l package.
class SNLIBERTDataset(gdata.Dataset):
    def __init__(self, dataset, vocab=None):
        self.dataset = dataset
        self.max_len = 50 # 将每条评论通过截断或者补0，使得长度变成50
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
            labels.append(np.array([LABEL_TO_IDX[x[2]]]))
            
        return tokens, segment_ids, valid_lengths, labels

    def __getitem__(self, idx):
        return self.tokens[idx], self.segment_ids[idx], self.valid_lengths[idx], self.labels[idx]

    def __len__(self):
        return len(self.tokens)
```

通过自定义的`SNLIBERTDataset`类来分别创建训练集和测试集的实例。我们指定最大文本长度为50。下面我们可以分别查看训练集和测试集所保留的样本个数。

```{.python .input  n=66}
d2l.download_snli()
train_set = SNLIBERTDataset("train", bert_train_set.vocab)
test_set = SNLIBERTDataset("test", bert_train_set.vocab)
```

```{.json .output n=66}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "read 50 examples\nread 50 examples\n"
 }
]
```

设批量大小为64，分别定义训练集和测试集的迭代器。

```{.python .input  n=67}
batch_size = 64
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

### 用于微调的分类模型

刚才我们已经训练好了BERT模型，我们只需要附加一个额外的层来进行分类。 BERTClassifier类使用BERT模型对句子表示进行编码，然后使用第一个令牌“[CLS]”的编码通过全连接层进行分类。

```{.python .input  n=82}
class BERTClassifier(nn.Block):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = gluon.nn.Dense(num_classes)

    def forward(self, inputs, segment_types, seq_len):
        seq_encoding = self.bert(inputs, segment_types, seq_len)
        return self.classifier(seq_encoding[:, 0, :])
```

初始化网络时要注意的是我们只需要初始化分类层。 因为BERT是使用已经预训练好的权重。

```{.python .input  n=83}
net = BERTClassifier(bert, 2)
net.classifier.initialize(ctx=ctx)
```

我们修改“可分解注意力模型用于自然语言推理”中的“_get_batch_snli”方法，将小批量数据样本batch划分并复制到ctx变量所指定的各个显存上。

```{.python .input  n=84}
def _get_batch_snli_bert(batch, ctx):
    tokens, segment_ids, valid_lengths, labels = batch
    return (gutils.split_and_load(tokens, ctx),
            gutils.split_and_load(segment_ids, ctx),
            gutils.split_and_load(valid_lengths.astype('float32'), ctx),
            gutils.split_and_load(labels.astype('float32'), ctx), tokens.shape[0])
```

同样，我们也修改“evaluate_accuracy_snli”函数。

```{.python .input  n=85}
def evaluate_accuracy_snli_bert(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = np.array([0]), 0
    for batch in data_iter:
        tokens, segment_ids, valid_lengths, labels, batch_size = _get_batch_snli(batch, ctx)
        for t, s_i, v_l, y in zip(tokens, segment_ids, valid_lengths, labels):
            y = y.astype('float32')
            acc_sum += (net(t, s_i, v_l).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum / n
```

我们也需要修改“train_snli”函数。

```{.python .input  n=86}
def train_snli_bert(train_iter, dev_iter, net, loss, trainer, ctx, num_epochs):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            tokens, segment_ids, valid_lengths, labels, batch_size = _get_batch_snli(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(t, s_i, v_l) for t, s_i, v_l in zip(tokens, segment_ids, valid_lengths)]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, labels)]
            for l in ls:
                l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
            train_l_sum += sum([l.sum() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum()
                                 for y_hat, y in zip(y_hats, labels)])
            m += sum([y.size for y in labels])
        test_acc = evaluate_accuracy_snli(dev_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, dev acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
```

现在就可以训练模型了。

```{.python .input  n=87}
lr, num_epochs = 0.1, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCELoss()
train_snli_bert(train_iter, test_iter, net, loss, trainer,[mx.gpu(0)], num_epochs)
```

```{.json .output n=87}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 1, loss 0.6776, train acc 16.060, dev acc 17.000, time 0.1 sec\nepoch 2, loss 1.7697, train acc 16.000, dev acc 18.000, time 0.1 sec\nepoch 3, loss 1.9370, train acc 17.000, dev acc 17.000, time 0.1 sec\nepoch 4, loss 0.6695, train acc 16.100, dev acc 17.000, time 0.3 sec\nepoch 5, loss 0.9682, train acc 16.000, dev acc 17.000, time 0.1 sec\n"
 }
]
```
