# BERT的数据预处理及模型训练

在上一节中，我们介绍了双向语言表征模型（BERT）。在这一节我们将介绍如何预处理BERT在输入时所需要的数据，并使用这些数据训练一个BERT模型。

## 语言模型数据集

BERT中有一个任务是下一句预测，我们需要在文档中获得真实的“下一个句子”，所以需要文档级的训练语料。本节所用到的数据集是WikiText语言模型数据集，该数据集是从维基百科上的文章集中提取的。WikiText语言模型数据集分为“wikitext-2”和“wikitext-103”两种，前者的文章数和单词数都少于后者。
我们首先下载这个数据集。

```{.python .input  n=1}
import collections
import d2l
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
from mxnet.contrib import text
import os
import random
import time
import zipfile

npx.set_np()
```

```{.python .input}
# Saved in the d2l package for later use
def download_wiki(data_set='wikitext-2', data_dir='../data/'):
    if data_set=='wikitext-2':
        url = ('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')
        sha1 = '3c914d17d80b1459be871a5039ac23e752a53cbe'
    else:
        url = ('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip')
        sha1 = '0aec09a7537b58d4bb65362fee27650eeaba625a'
    fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall(data_dir)
        
download_wiki()
```

然后我们读取这一数据集文件。该文件的每一行是一段文本，我们只保留多于两个句子的文本段落。

```{.python .input  n=2}
# Saved in the d2l package for later use
def read_wiki(data_set='wikitext-2'):
    file_name = os.path.join('../data/', data_set, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        raw = f.readlines()
    data = [line.strip().lower().split(' . ')
            for line in raw if len(line.split(' . '))>=2]
    random.shuffle(data)
    return data

train_data = read_wiki()
```

## 数据预处理

之前介绍了BERT包含两个预训练任务：下一句预测、掩码语言模型。下面我们分别介绍该如何对这两个任务进行数据预处理。

### 下一句预测

下一句预测任务在选择文档库中的一个句子A后。以一半的概率将句子A的真实下一句作为句子B，标签设为True。一半概率使用来自语料库的随机句子作为句子B，标签设为False。

```{.python .input  n=3}
# Saved in the d2l package for later use
def get_next_sentence(sentence, next_sentence, all_documents):
    # 对于每一个句子，有50%的概率使用真实的下一句
    if random.random() < 0.5:
        tokens_a = sentence.split() 
        tokens_b = next_sentence.split()
        is_next = True
    # 对于每一个句子，有50%的概率使用随机选取的句子作为下一句
    else:
        random_sentence = random.choice(random.choice(all_documents))
        tokens_a = sentence.split() 
        tokens_b = random_sentence.split()
        is_next = False
    return tokens_a, tokens_b, is_next
```

我们在每个句子结束位置加入"[SEP]"标记，并连结这两个句子作为一个序列。然后，在序列开始位置插入“[CLS]”标记，同时在片段标记中使用0和1区分两个句子。

```{.python .input  n=4}
# Saved in the d2l package for later use
def get_tokens_and_segment(tokens_a, tokens_b):
    tokens = []  # 词片标记
    segment_ids = []  # 片段索引，使用0和1区分两个句子

    # 在序列开始插入“[CLS]”标记
    tokens.append('[CLS]')
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    # 在句子结束位置插入“[SEP]”标记
    tokens.append('[SEP]')
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append('[SEP]')
    segment_ids.append(1)
    
    return tokens, segment_ids
```

我们从原始的语料里建立下一句任务的输入，在这里我们舍弃超过最大长度的句子。对于句对输入，这里的最大长度是指在连结两句子后包含了特殊标记的序列的长度。

```{.python .input  n=5}
# Saved in the d2l package for later use
def create_next_sentence(document, all_documents, vocab, max_length):
    instances = []
    for i in range(len(document)-1):
        tokens_a, tokens_b, is_next = get_next_sentence(document[i], document[i+1], all_documents)
        
        # 舍弃超过最大长度的句子，注意这里计算长度时要考虑“[CLS]”标记和两个“[SEP]”标记
        if len(tokens_a) + len(tokens_b) + 3 > max_length:
             continue
        tokens, segment_ids = get_tokens_and_segment(tokens_a, tokens_b)
        instances.append((tokens, segment_ids, is_next))
    return instances
```

### 掩码语言模型


我们需要随机将一定比例的输入标记选择为掩码标记，在BERT原文的设定中，在每个序列随机掩码15%的标记。
在预训练阶段，我们使用了“[MASK]”这个掩码标记，但“[MASK]”在微调阶段并不会出现。这会带来预训练和微调之间的不匹配问题。BERT原文采用了一些策略来缓解这一问题。在选择了需要被掩码的标记后：
以80％概率：用“[MASK]“标记替换单词，例如，
> my dog is hairy → my dog is [MASK]

以10％的概率：用一个随机的单词替换该单词，例如，

> my dog is hairy → my dog is apple

以10％的概率：保持单词不变，例如，
> my dog is hairy → my dog is hairy

```{.python .input  n=6}
# Saved in the d2l package for later use
def choice_mask_tokens(tokens, cand_indexes, num_to_predict, vocab):
    output_tokens = list(tokens)
    masked_lms = []
    random.shuffle(cand_indexes)
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        for index in index_set:
            masked_token = None
            # 80%的概率替换成“[MASK]”标记
            if random.random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10%的概率保持不变
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10%的概率使用随机单词进行替换
                else:
                    masked_token = random.randint(0, len(vocab) - 1)

            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
    return output_tokens, masked_lms
```

通过掩码随机遮挡15%的标记，即允许每个样本中15%的标记被预测。将遮挡后的序列转换为索引表示。作为模型的输入，我们需要获得掩码位置和掩码位置真实标记的索引表示。

```{.python .input  n=7}
# Saved in the d2l package for later use
def create_masked_lm(tokens, vocab):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indexes.append([i])
        
    # 计算需要遮挡的标记数目
    num_to_predict = max(1, int(round(len(tokens) * 0.15)))
    
    # 遮挡标记，返回的是遮挡后句子的标记序列，以及掩码位置和掩码位置的真实标记。
    output_tokens, masked_lms = choice_mask_tokens(tokens, cand_indexes,
                                                   num_to_predict, vocab)
            
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_lm_positions = []  # 掩码位置
    masked_lm_labels = []  # 掩码位置的真实标记
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
        
    return vocab.to_indices(output_tokens), masked_lm_positions,
           vocab.to_indices(masked_lm_labels)
```

### 创建样本集
我们需要将每个样本填充为等长，同时转换成numpy形式。

```{.python .input  n=8}
# Saved in the d2l package for later use
def convert_numpy(instances, max_length):
    input_ids, segment_ids, masked_lm_positions, masked_lm_ids = [], [], [], []
    masked_lm_weights, next_sentence_labels, valid_lengths = [], [], []
    for instance in instances:
        # instance[0] 输入的索引表示
        # instance[1] 掩码位置
        # instance[2] 掩码位置的真实标记
        # instance[3] 片段标记
        # instance[4] 下一句预测标签
        input_id = instance[0] + [0] * (max_length - len(instance[0]))
        segment_id = instance[3] + [0] * (max_length - len(instance[3]))
        masked_lm_position = instance[1] + [0] * (20 - len(instance[1]))  # 对于每个样本最大掩码数量为20
        masked_lm_id = instance[2] + [0] * (20 - len(instance[2]))
        # 通过将非掩码位置的权重置为0，来避免预测非掩码位置的标记
        masked_lm_weight = [1.0] * len(instance[2]) + [0.0] * (20 - len(instance[1]))
        next_sentence_label = instance[4]
        valid_length = len(instance[0])

        input_ids.append(np.array(input_id, dtype='int32'))
        segment_ids.append(np.array(segment_id, dtype='int32'))
        masked_lm_positions.append(np.array(masked_lm_position, dtype='int32'))
        masked_lm_ids.append(np.array(masked_lm_id, dtype='int32'))
        masked_lm_weights.append(np.array(masked_lm_weight, dtype='float32'))
        next_sentence_labels.append(np.array(next_sentence_label))
        valid_lengths.append(np.array(valid_length))
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths
```

依次调用下一句预测任务和掩码语言模型数据预处理的方法，用来创建样本集。

```{.python .input  n=1}
# Saved in the d2l package for later use
def create_training_instances(train_data, vocab, max_length):
    # 创建下一句任务的样本
    instances = []
    for i, document in enumerate(train_data):
        instances.extend(create_next_sentence(document, train_data, vocab, max_length))
    # 进行掩码语言模型的预处理
    instances = [(create_masked_lm(tokens, vocab) + (segment_ids, is_random_next))
                 for (tokens, segment_ids, is_random_next) in instances]
    # 转换成numpy形式
    input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths = convert_numpy(instances, max_length)
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths
```

### 自定义数据集类
我们通过继承Gluon提供的`Dataset`类自定义了一个语言模型数据集类`WikiDataset`。同样可以通过`__getitem__`函数，任意访问数据集中索引为idx的样本。在这个数据集类中，我们读取“WikiText-103”语言建模数据集，分别调用下一句任务的数据生成方法和掩码语言模型的数据生成方法创建样本集。

```{.python .input  n=18}
# Saved in the d2l package for later use
class WikiDataset(gluon.data.Dataset):
    def __init__(self, data_set = 'wikitext-2', max_length = 128):
        train_data = read_wiki(data_set)
        self.vocab = self.get_vocab(train_data)
        self.input_ids, self.masked_lm_ids, self.masked_lm_positions,\
        self.masked_lm_weights, self.next_sentence_labels, self.segment_ids,\
        self.valid_lengths = create_training_instances(train_data, self.vocab, max_length)

    def get_vocab(self, data):
        # 过滤出现频度小于5的词
        counter = collections.Counter([w for st in data
                                       for tk in st
                                       for w in tk.split()])
        return text.vocab.Vocabulary(counter, min_freq=5, reserved_tokens=['[MASK]', '[CLS]', '[SEP]'])
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_lm_ids[idx], self.masked_lm_positions[idx], self.masked_lm_weights[idx],\
           self.next_sentence_labels[idx], self.segment_ids[idx], self.valid_lengths[idx]

    def __len__(self):
        return len(self.input_ids)
```

### 读取数据集

通过自定义的`WikiDataset`类来创建数据集的实例。

```{.python .input  n=11}
train_set = WikiDataset('wikitext-2', 128)
```

设批量大小为128，定义训练集的迭代器。

```{.python .input  n=12}
batch_size = 128
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
```

打印第一个小批量。这里的数据依次是词标记索引、掩码位置的真实标记、掩码位置、掩码位置的权重、下一句任务标签、片段索引以及有效句子长度。

```{.python .input  n=13}
for _, data_batch in enumerate(train_iter):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data_batch
    print(input_id.shape, masked_id.shape, masked_position.shape, masked_weight.shape,\
          next_sentence_label.shape, segment_id.shape, valid_length.shape)
    break
```

## 模型训练

在预处理好数据后，我们可以进行模型训练了。

### 初始化模型
我们将`双向语言表征模型（BERT）`一节中的BERT模型初始化，并设置嵌入层尺寸为256.
下一句预测任务和掩码语言模型任务都使用带有softmax的交叉熵作为损失函数。

```{.python .input  n=14}
net = d2l.BERTModel(len(train_set.vocab), embed_size=256, hidden_size=256, num_heads=4, num_layers=4, dropout=0.2)
ctx = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=ctx)
nsp_loss = mx.gluon.loss.SoftmaxCELoss()
mlm_loss = mx.gluon.loss.SoftmaxCELoss()
```

### 训练函数
_get_batch_bert 这个函数将小批量数据样本batch划分并复制到ctx变量所指定的各个显存上。

```{.python .input  n=15}
# Saved in the d2l package for later use
def _get_batch_bert(batch, ctx):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = batch
    
    return (gluon.utils.split_and_load(input_id, ctx, even_split=False),
            gluon.utils.split_and_load(masked_id, ctx, even_split=False),
            gluon.utils.split_and_load(masked_position, ctx, even_split=False),
            gluon.utils.split_and_load(masked_weight, ctx, even_split=False),
            gluon.utils.split_and_load(next_sentence_label, ctx, even_split=False),
            gluon.utils.split_and_load(segment_id, ctx, even_split=False),
            gluon.utils.split_and_load(valid_length.astype('float32'), ctx, even_split=False))
```

定义batch_loss函数，计算每个批量的损失。

```{.python .input  n=5}
# Saved in the d2l package for later use
def batch_loss_bert(net, nsp_loss, mlm_loss, input_id, masked_id, masked_position,
                    masked_weight, next_sentence_label, segment_id, valid_length, vocab_size):
    ls = []
    ls_mlm = []
    ls_nsp = []
    for i_id,\ m_id, m_pos, m_w, nsl, s_i, v_l in zip(input_id, masked_id, masked_position, masked_weight,\
                                                      next_sentence_label, segment_id, valid_length):
        num_masks = m_w.sum() + 1e-8
        _, classified, decoded = net(i_id, s_i, v_l.reshape(-1),m_pos)
        # 计算掩码语言模型的损失
        l_mlm = mlm_loss(decoded.reshape((-1, vocab_size)),m_id.reshape(-1), m_w.reshape((-1, 1)))
        l_mlm = l_mlm.sum() / num_masks
        # 计算下一句预测的损失
        l_nsp = nsp_loss(classified, nsl)
        l_nsp = l_nsp.mean()
        # 掩码语言模型和下一句预测的损失求和
        l = l_mlm + l_nsp
        ls.append(l)
        ls_mlm.append(l_mlm)
        ls_nsp.append(l_nsp)
        npx.waitall()
        return ls, ls_mlm, ls_nsp
```

定义train函数，使用多GPU训练模型。

```{.python .input  n=6}
# Saved in the d2l package for later use
def train_bert(data_eval, net, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, max_step):
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    step_num = 0
    while step_num < max_step:
        eval_begin_time = time.time()
        begin_time = time.time()
        
        running_mlm_loss = running_nsp_loss = 0
        total_mlm_loss = total_nsp_loss = 0
        running_num_tks = 0
        for _, data_batch in enumerate(data_eval):
            (input_id, masked_id, masked_position, masked_weight, \
             next_sentence_label, segment_id, valid_length) = _get_batch_bert(data_batch, ctx)
            
            step_num += 1
            with autograd.record():
                ls, ls_mlm, ls_nsp = batch_loss_bert(net, nsp_loss, mlm_loss, input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length, vocab_size)
            for l in ls:
                l.backward()
            
            trainer.step(1)
            
            running_mlm_loss += sum([l for l in ls_mlm])
            running_nsp_loss += sum([l for l in ls_nsp])

            if (step_num + 1) % (log_interval) == 0:
                total_mlm_loss += running_mlm_loss
                total_nsp_loss += running_nsp_loss
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = 0

        eval_end_time = time.time()
        if running_mlm_loss != 0:
            total_mlm_loss += running_mlm_loss
            total_nsp_loss += running_nsp_loss
        total_mlm_loss /= step_num
        total_nsp_loss /= step_num
        print('Eval mlm_loss={:.3f}\tnsp_loss={:.3f}\t'
                     .format(float(total_mlm_loss),
                             float(total_nsp_loss)))
        print('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))
```

现在就可以训练BERT模型了。我们在这里运行1步训练来验证训练函数。我们将在下一节完整运行BERT模型的预训练及介绍如何接入下游任务。

```{.python .input}
train_bert(bert_train_iter, bert, nsp_loss, mlm_loss, len(bert_train_set.vocab), ctx, 20, 1)
```

## 小结

- BERT模型需要在序列开始位置插入“[CLS]”标记，在每个句子结束位置加入"[SEP]"标记。
- 可以在片段标记中使用0和1区分两个句子。
- BERT采用了三种策略来缓解微调阶段不会出现“[MASK]”标记的问题。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![]()
