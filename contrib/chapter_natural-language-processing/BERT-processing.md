# BERT的数据预处理及模型训练

在上一节中，我们介绍了双向语言表征模型（BERT），在这一节我们将介绍如何预处理BERT所需要的数据，并训练一个BERT模型。

## 语言模型数据集

本节所用到的数据集“WikiText-103”语言建模数据集，该数据集是从维基百科上经过验证的优秀和精选文章集中提取的。
我们首先下载这个数据集。

```{.python .input  n=2}
import d2lzh as d2l
import os
import collections
import os
from mxnet import autograd, gluon, init, np, npx
from mxnet.contrib import text
from mxnet.gluon import Block
from mxnet.gluon import data as gdata, nn, utils as gutils
from mxnet.gluon.model_zoo import model_store
import mxnet as mx
import os
import random
import time
import math
import zipfile

npx.set_np()

# Save to the d2l package.
def download_wiki(data_dir='../data/'):
    url = ('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip')
    sha1 = '0aec09a7537b58d4bb65362fee27650eeaba625a'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall(data_dir)
        
download_wiki()
```
然后我们读取这一数据集。文件的每一行是一段文本，我们只保留大于两个句子的行。

```{.python .input  n=2}
# Save to the d2l package.
def read_wiki():  
    file_name = os.path.join('../data/wikitext-103/', 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        raw = f.readlines()
    data = [line.strip().lower().split(' . ') for line in raw if len(line.split(' . '))>=2]
    random.shuffle(data)
    return data

train_data = read_wiki()
```

## 数据预处理


之前介绍了BERT包含两个预训练任务：下一句预测、遮蔽语言模型。我们分别介绍如何预处理这两个任务的数据。

### 下一句预测

下一句预测任务需要在每个训练样本选择句子A和B时，50%的概率B是A真实的下一句，有一半的概率使用来自语料库的随机句子替换句子B。
```{.python .input  n=2}
# Save to the d2l package.
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
我们在序列开始位置插入“[CLS]”标记，在每个句子结束位置加入"[SEP]"标记。同时在片段标记中使用0和1区分两个句子。
```{.python .input  n=2}
# Save to the d2l package.
def get_tokens_and_segment(tokens_a, tokens_b):
    tokens = []
    segment_ids = []

    # 在序列开始插入“[CLS]”标记
    tokens.append('[CLS]')
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append('[SEP]')
    segment_ids.append(1)
    
    return tokens, segment_ids
```

我们从原始的语料里建立下一句任务的输入，在这里我们舍弃超过最大长度的句子。

```{.python .input  n=2}
# Save to the d2l package.
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

### 遮蔽语言模型


我们需要随机将一定比例的输入标记选择为被遮蔽标记，在BERT的设计中，在每个序列中随机遮蔽 15% 的标记。
由于在预训练阶段，我们使用了“[MASK]”这个遮蔽标记，但“[MASK]”在微调阶段并不会出现。这会带来预训练和微调之间的不匹配。BERT采用了一些策略来缓解这一问题。在选择了需要被遮蔽的标记后：
以80％概率：用[MASK]标记替换单词，例如，
> my dog is hairy → my dog is [MASK]

以10％的概率：用一个随机的单词替换该单词，例如，

> my dog is hairy → my dog is apple

以10％的概率：保持单词不变，例如，

> my dog is hairy → my dog is hairy


```{.python .input  n=2}
# Save to the d2l package.
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
                # 10%的概率使用随机令牌进行替换
                else:
                    masked_token = random.randint(0, len(vocab) - 1)

            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
    return output_tokens, masked_lms
```

随机遮蔽15%的标记，即允许每个样本中15%的标记被预测。将遮蔽后的序列转换为索引表示，同时返回被遮蔽的位置，以及被遮蔽位置真实标记的索引表示。

```{.python .input  n=2}
# Save to the d2l package.
def create_masked_lm(tokens, vocab):
    
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indexes.append([i])
   
    num_to_predict = max(1, int(round(len(tokens) * 0.15)))
    
    output_tokens, masked_lms = choice_mask_tokens(tokens, cand_indexes, num_to_predict, vocab)
            
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
        
    return vocab.to_indices(output_tokens), masked_lm_positions, vocab.to_indices(masked_lm_labels)
```

### 创建样本集
我们需要将每个样本填充为等长，同时转换成numpy形式。
```{.python .input  n=2}
# Save to the d2l package.
def convert_numpy(instances, max_length):
    input_ids, segment_ids, masked_lm_positions, masked_lm_ids = [], [], [], []
    masked_lm_weights, next_sentence_labels, valid_lengths = [], [], []
    for instance in instances:
        # instance[0] 输入的索引表示
        # instance[1] 被遮蔽位置
        # instance[2] 被遮蔽位置的真实标记
        # instance[3] 片段标记
        # instance[4] 下一句预测标签
        input_id = instance[0] + [0] * (max_length - len(instance[0]))
        segment_id = instance[3] + [0] * (max_length - len(instance[3]))
        masked_lm_position = instance[1] + [0] * (20 - len(instance[1]))
        masked_lm_id = instance[2] + [0] * (20 - len(instance[2]))
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

依次调用下一句预测任务数据预处理的方法和遮蔽语言模型数据预处理的方法，用来创建样本集。

```{.python .input  n=2}
# Save to the d2l package.
def create_training_instances(train_data, vocab, max_length):
    instances = []
    for i, document in enumerate(train_data):
        instances.extend(create_next_sentence(document, train_data, vocab, max_length))

    instances = [(create_masked_lm(tokens, vocab) + (segment_ids, is_random_next))
                 for (tokens, segment_ids, is_random_next) in instances]
    
    input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths = convert_numpy(instances, max_length)
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths
```
### 自定义数据集类
我们通过继承Gluon提供的`Dataset`类自定义了一个语言模型数据集类`WikiDataset`。同样可以通过`__getitem__`函数，任意访问数据集中索引为idx的样本。在这个数据集类中，我们读取“WikiText-103”语言建模数据集，分别调用下一句任务的数据生成方法和遮蔽语言模型的数据生成方法创建样本集。
```{.python .input  n=2}
# Save to the d2l package.
class WikiDataset(gdata.Dataset):
    def __init__(self, max_length):
        train_data = read_wiki()
        self.vocab = self.get_vocab(train_data)
        self.input_ids, self.masked_lm_ids, self.masked_lm_positions, self.masked_lm_weights,\
           self.next_sentence_labels, self.segment_ids, self.valid_lengths = create_training_instances(train_data, self.vocab, max_length)

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
```{.python .input  n=2}
train_set = WikiDataset(128)
```
设批量大小为16，定义训练集的迭代器。
```{.python .input  n=2}
batch_size = 16
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
```
打印第一个小批量。这里的数据依次是令牌索引、遮蔽词的标签、被遮蔽的位置、被遮蔽位置的权重、下一句任务标签、片段索引以及有效句子长度。
```{.python .input  n=2}
for data_batch in enumerate(train_iter):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data_batch
    print(input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length)
```

## 模型训练

在预处理好数据后，我们可以进行模型训练了。

### 初始化模型
我们将上一节中的BERT模型初始化，并设置嵌入层尺寸为100.
下一句预测任务和遮蔽语言模型任务都使用带有softmax的交叉熵作为损失函数。

```{.python .input  n=2}
net = d2l.BERTModel(len(train_set.vocab), embed_size=128, hidden_size=512, num_heads=2, num_layers=4, dropout=0.1)
ctx = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=ctx)
nsp_loss = mx.gluon.loss.SoftmaxCELoss()
mlm_loss = mx.gluon.loss.SoftmaxCELoss()
```

### 训练函数
_get_batch_bert 这个函数将小批量数据样本batch划分并复制到ctx变量所指定的各个显存上。
```{.python .input  n=2}
# Save to the d2l package.
def _get_batch_bert(batch, ctx):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = batch
    
    return (gutils.split_and_load(input_id, ctx),
            gutils.split_and_load(masked_id, ctx),
            gutils.split_and_load(masked_position, ctx),
            gutils.split_and_load(masked_weight, ctx),
            gutils.split_and_load(next_sentence_label, ctx),
            gutils.split_and_load(segment_id, ctx),
            gutils.split_and_load(valid_length.astype('float32'), ctx))
```

定义train函数，使用多GPU训练模型。

```{.python .input  n=2}
# Save to the d2l package.
def train_bert(data_eval, model, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, num_epochs):
    trainer = mx.gluon.Trainer(model.collect_params(), 'adam')
    for epoch in range(num_epochs):
        eval_begin_time = time.time()
        begin_time = time.time()
        step_num = 0
        running_mlm_loss = running_nsp_loss = 0
        total_mlm_loss = total_nsp_loss = 0
        running_num_tks = 0
        for _, data_batch in enumerate(data_eval):
            (input_id, masked_id, masked_position, masked_weight, \
             next_sentence_label, segment_id, valid_length) = _get_batch_bert(data_batch, ctx)
            
            step_num += 1
            with autograd.record():
                ls = []
                ls_mlm = []
                ls_nsp = []
                for i_id, m_id, m_pos, m_w, nsl, s_i, v_l in zip(input_id, \
                        masked_id, masked_position, masked_weight, \
                        next_sentence_label, segment_id, valid_length):
                    num_masks = m_w.sum() + 1e-8
                    _, classified, decoded = net(i_id, s_i, v_l.reshape(-1),
                                          m_pos)
                    l_mlm = mlm_loss(decoded.reshape((-1, vocab_size)),m_id.reshape(-1), m_w.reshape((-1, 1)))
                    l_nsp = nsp_loss(classified, nsl)
                    l_mlm = l_mlm.sum() / num_masks
                    l_nsp = l_nsp.mean()
                    l = l_mlm + l_nsp
                    ls.append(l)
                    ls_mlm.append(l_mlm)
                    ls_nsp.append(l_nsp)
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
现在就可以训练BERT模型了。我们将在下一节运行BERT模型的预训练及介绍如何接入下游任务。

## 小结

- BERT模型需要在序列开始位置插入“[CLS]”标记，在每个句子结束位置加入"[SEP]"标记。
- 可以在片段标记中使用0和1区分两个句子。
- BERT采用了三种策略来缓解微调阶段不会出现“[MASK]”标记的问题。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![]()