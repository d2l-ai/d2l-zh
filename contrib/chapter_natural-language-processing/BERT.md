# 来自Transformer的双向编码器表示（BERT）

在“词嵌入”章节中，我们提到了词向量是用来表示词的向量，也就是说词向量是能够反映出语义的特征。但常用的词嵌入模型在训练完成后，每个词向量就固定了。在之后使用的时候，无论出现这个词的上下文是什么，词向量都不会随着上下文发生变化。例如“apple”这个词既有水果的意思，又是一家公司的名字，这种词向量叫做静态词向量。而我们期待一个好的词向量应该能够随着不同上下文产生变化，这种能够随着上下文语境不同而变化的词向量叫做动态词向量。

既然需要随着不同的上下文产生变化，我们可以设计一种动态计算词向量的网络，这个网络的输入是每个词的静态词向量，输出是每个词在当前上下文中的词向量。而这个网络也类似于词嵌入模型，可以预先在大量的预料中进行训练。这种动态计算词向量的网络叫做语言表示模型。

来自Transformer的双向编码器表示（BERT）首先在大规模语料上来预训练上下文深度双向表示，这一阶段叫做预训练阶段。在适用于广泛的任务时，只需要一个额外的输出层，就可以对预训练的 BERT 表示进行微调，而无需对特定于任务进行大量模型结构的修改。

首先导入实验所需的包和模块。

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
import math
import zipfile

npx.set_np()
```
## 语言模型数据集

在介绍模型之前，介绍本节所用到的数据集“WikiText-103”语言建模数据集，该数据集是从维基百科上经过验证的优秀和精选文章集中提取的。我们首先下载这个数据集

```{.python .input  n=2}
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

## 模型结构

BERT的基础模型结构是在“Transformer”章节中描述的多层双向Transformer编码器。原始的Transformer包括编码器和解码器部分。由于BERT的目标是语言表示模型，因此只需要编码器机制。
BERT分为Base和Large两个版本。Base版本包含12层Transformer，有110M的参数。Large版本包含24层Transformer，有340M的参数。

### 输入表示

BERT的输入支持单个句子或一对句子。分别适用于单句任务（如文本分类任务）和句对任务（如自然语言推理任务）。BERT的输入包含三部分，分别是令牌嵌入、片段嵌入、位置嵌入。

令牌嵌入（Token Embeddings）是将各个词转换成固定维度的向量。首先在序列的开始位置加入特殊标记“[CLS]”，在序列的结束位置加入特殊标记“[SEP]”。如果有两个句子，直接拼接在一起，在每个句子序列的结束位置都加入“[SEP]”。在BERT中，每个词会被转换成768维的向量表示。

片段嵌入（Segment Embeddings）是为了使BERT能够处理句对的输入，句子对中的两个句子被简单的拼接在一起作为输入，因为我们需要使模型能够区分一个句子对中的两个句子，这就是片段嵌入的作用。片段嵌入只有两种向量表示，把向量0给第一个句子序列中的每个令牌，把向量1给第二个句子序列中的每个令牌。如果是输入仅仅有一个句子，那序列中的每个令牌的片段嵌入都是向量0。向量0和向量1都是在训练过程中更新得到的。每个向量都是768维度，所以片段嵌入层的大小是（2，768）。

位置嵌入（Position Embeddings）。为了解决Transformer无法编码序列的问题，我们引入了位置嵌入。在BERT中的位置嵌入与Transformer里的位置嵌入稍有不同，BERT中的位置嵌入是在各个位置上学习一个向量表示，从而来将顺序的信息编码进来。BERT最长能处理512个令牌的序列，所以位置嵌入层的大小是（512，768）。

对于一个长度为n的输入序列，我们将有令牌嵌入（n，768）用来表示词，片段嵌入（n，768）用来区分两个句子，位置嵌入（n，768）用来学习到顺序。将这三种嵌入按元素相加，得到一个（n，768）的表示，这一表示就是BERT的输入。

我们修改“Transformer”中的TransformerEncoder函数，加入BERT所需要的令牌嵌入、片段嵌入、位置嵌入。
```{.python .input  n=2}
class BERTEncoder(gluon.nn.Block):
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
## 预训练任务

BERT包含两个预训练任务：下一句预测、遮蔽语言模型。

### 下一句预测

在自然语言处理中有很多下游任务是建立在理解两个句子之间关系的基础上。比如自然语言推理任务。这并不是语言模型所能直接学习到的。为了能够学习到句子间关系，BERT设计了一个预测下一句的二分类任务，即预测输入的两个句子是否为连续的文本。具体就是为每个训练样本选择句子A和B时，50%的概率B是A真实的下一句，有一半的概率使用来自语料库的随机句子替换句子B。
> Input = [CLS] the man went to [MASK] store [SEP]
> he bought a gallon [MASK] milk [SEP]
> Label = IsNext


> Input = [CLS] the man [MASK] to the store [SEP]
> penguin [MASK] are flight ##less birds [SEP]
> Label = NotNext

然后将“[CLS]”标记的输出送入一个单层网络，并使用softmax计算IsNext标签的概率，以判断句子是否是当前句子的下一句。使用“[CLS]”是因为Transformer是可以把全局信息编码进每个位置，因此“[CLS]”位置的输出表示可以包含整个输入序列的特征。

我们从原始的语料里建立下一句任务的输入。
```{.python .input  n=2}
def create_next_sentence(document, all_documents, vocab):
    instances = []
    for i in range(len(document)-1):
        
        # 对于每一个句子，有50%的概率使用真实的下一句
        if random.random() < 0.5:
            tokens_a = document[i].split() 
            tokens_b = document[i+1].split()
            is_random_next = False
        # 对于每一个句子，有50%的概率使用随机选取的句子作为下一句
        else:
            random_sentence = random.choice(random.choice(all_documents))
            is_random_next = True
            tokens_a = document[i].split() 
            tokens_b = random_sentence.split()
        if len(tokens_a) + len(tokens_b) + 3 > max_length:
             continue
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
        
        instances.append((tokens, segment_ids, is_random_next))
    return instances
```

然后我们设计下一句预测任务的模型，我们将编码后的结果传递给NSClassifier以获得下一个句子预测。 
```{.python .input  n=2}
class NSClassifier(gluon.nn.Block):
    def __init__(self, units=768, **kwargs):
        super(NSClassifier, self).__init__(**kwargs)
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(units=units, flatten=False, activation='tanh'))
        self.classifier.add(gluon.nn.Dense(units=2, flatten=False))

    def forward(self, X, *args):
        X = X[:, 0, :]  # get the encoding of the first token
        return self.classifier(X)
```


### 遮蔽语言模型（mask-lm）
一般来说语言表示模型只能从左到右或者从右到左的单向训练。因为如果允许双向训练就意味着会使得每个词在多层的网络中间接地“看到自己”。
为了训练深度双向的表示，BERT设计了一种完形填空的猜词任务，名为遮蔽语言模型的任务。具体来说，就是随机将一定比例的输入标记替换为遮蔽标记“[MASK]”，然后预测这些被遮蔽的标记。即将遮蔽标记对应的输入隐藏向量输入一个单层网络，用softmax计算词汇表中每个单词的概率，以预测遮蔽标记应该对应哪个词。在BERT的设计中，在每个序列中随机遮蔽 15% 的标记。
由于在预训练阶段，我们使用了“[MASK]”这个遮蔽标记，但“[MASK]”在微调阶段并不会出现。这会带来预训练和微调之间的不匹配。BERT采用了一些策略来缓解这一问题。在选择了需要被遮蔽的标记后：
80％的时间：用[MASK]标记替换单词，例如，
> my dog is hairy → my dog is [MASK]

10％的时间：用一个随机的单词替换该单词，例如，

> my dog is hairy → my dog is apple

10％的时间：保持单词不变，例如，

> my dog is hairy → my dog is hairy

我们创建遮蔽语言模型的数据输入，接收从下一句预测任务得到的数据。
```{.python .input  n=2}
def create_masked_lm(tokens, vocab):
    masked_lms = []
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indexes.append([i])
    random.shuffle(cand_indexes)
    num_to_predict = max(1, int(round(len(tokens) * 0.15)))
    MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                              ['index', 'label'])
    output_tokens = list(tokens)
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        for index in index_set:
            masked_token = None
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
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
        
    return vocab.to_indices(output_tokens), masked_lm_positions, vocab.to_indices(masked_lm_labels)
```

创建遮蔽语言模型的预测模型，模型需要重建被掩蔽的单词，我们使用gather_nd来选择代表遮蔽位置令牌的向量。 然后在它们上通过一个前馈网络，以预测词汇表中所有单词的概率分布。

```{.python .input  n=2}
class MLMDecoder(gluon.nn.Block):
    def __init__(self, vocab_size, units, **kwargs):
        super(MLMDecoder, self).__init__(**kwargs)
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
## 模型训练

### 自定义数据集类
我们通过继承Gluon提供的`Dataset`类自定义了一个语言模型数据集类`WikiDataset`。同样可以通过`__getitem__`函数，任意访问数据集中索引为idx的样本。在这个数据集类中，我们读取“WikiText-103”语言建模数据集，分别调用下一句任务的数据生成方法和遮蔽语言模型的数据生成方法，并将所有的字符串转换成数字索引表示。
```{.python .input  n=2}
max_length = 128
class WikiDataset(gdata.Dataset):
    def __init__(self, train_data):
        self.vocab = self.get_vocab(train_data)
        self.input_ids, self.masked_lm_ids, self.masked_lm_positions, self.masked_lm_weights,\
           self.next_sentence_labels, self.segment_ids, self.valid_lengths = self.create_training_instances(train_data, self.vocab)

    def get_vocab(self, data):
        # 过滤出现频度小于5的词
        counter = collections.Counter([w for st in data
                                       for tk in st
                                       for w in tk.split()])
        return text.vocab.Vocabulary(counter, min_freq=5, reserved_tokens=['[MASK]', '[CLS]', '[SEP]'])
    def create_training_instances(self,train_data, vocab):
        def transform(instance, max_seq_length):
            input_ids = instance[0]
            assert len(input_ids) <= max_seq_length
            segment_ids = instance[3]
            masked_lm_positions = instance[1]
            valid_lengths = len(input_ids)

            masked_lm_ids = instance[2]
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            next_sentence_label = 1 if instance[4] else 0

            features = {}
            features['input_ids'] = input_ids
            features['segment_ids'] = segment_ids
            features['masked_lm_positions'] = masked_lm_positions
            features['masked_lm_ids'] = masked_lm_ids
            features['masked_lm_weights'] = masked_lm_weights
            features['next_sentence_labels'] = [next_sentence_label]
            features['valid_lengths'] = [valid_lengths]
            return features
        instances = []
        for i, document in enumerate(train_data):
            instances.extend(create_next_sentence(document, train_data, vocab))

        #(tokens, masked_lm_positions, masked_lm_labels)
        instances = [(create_masked_lm(tokens, vocab) + (segment_ids, is_random_next) ) for (tokens, segment_ids, is_random_next) in instances]

        input_ids = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_labels = []
        valid_lengths = []

        for inst_index, instance in enumerate(instances):
            features = transform(instance, max_length)
            input_id = features['input_ids'] + [0] * (max_length-features['valid_lengths'][0])
            segment_id = features['segment_ids'] + [0] * (max_length-features['valid_lengths'][0])
            masked_lm_position = features['masked_lm_positions'] + [0] * (20 - len(features['masked_lm_positions']))
            masked_lm_id = features['masked_lm_ids'] + [0] * (20 - len(features['masked_lm_positions']))
            masked_lm_weight = features['masked_lm_weights'] + [0.0] * (20 - len(features['masked_lm_positions']))
            next_sentence_label = features['next_sentence_labels'][0]
            valid_length = features['valid_lengths'][0]

            input_ids.append(np.array(input_id, dtype='int32'))
            segment_ids.append(np.array(segment_id, dtype='int32'))
            masked_lm_positions.append(np.array(masked_lm_position, dtype='int32'))
            masked_lm_ids.append(np.array(masked_lm_id, dtype='int32'))
            masked_lm_weights.append(np.array(masked_lm_weight, dtype='float32'))
            next_sentence_labels.append(next_sentence_label)
            valid_lengths.append(valid_length)
        return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
               next_sentence_labels, segment_ids, valid_lengths
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_lm_ids[idx], self.masked_lm_positions[idx], self.masked_lm_weights[idx],\
           self.next_sentence_labels[idx], self.segment_ids[idx], self.valid_lengths[idx]

    def __len__(self):
        return len(self.input_ids)
```

### 读取数据集

通过自定义的`WikiDataset`类来创建数据集的实例。
```{.python .input  n=2}
train_set = WikiDataset(train_data)
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

### 构建模型

我们将上面的从Transfomer中修改得到的BERTEncoder，以及下一句任务预测模型、遮蔽语言模型合并到一起，得到BERT模型。
```{.python .input  n=2}
class BERTModel(Block):
    def __init__(self, vocab_size=None, token_type_vocab_size=2, units=256,
                 embed_size=128):
        super(BERTModel, self).__init__()
        self._vocab_size = vocab_size
        self.encoder = BERTEncoder(vocab_size=vocab_size, units=128, hidden_size=512,
                      num_heads=2, num_layers=4, dropout=0.1)
        
        self.ns_classifier = NSClassifier()
        self.decoder = MLMDecoder(vocab_size=vocab_size, units=128)

    def forward(self, inputs, token_types, valid_length=None, masked_positions=None):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        outputs = []
        seq_out = self.encoder(inputs, token_types, valid_length)
        outputs.append(seq_out)

        next_sentence_classifier_out = self.ns_classifier(seq_out)
        outputs.append(next_sentence_classifier_out)
        
        decoder_out = self.decoder(seq_out, masked_positions)
        outputs.append(decoder_out)
        
        return tuple(outputs)
```
### 初始化模型
我们将BERT模型初始化，并设置嵌入层尺寸为100.
下一句预测任务和遮蔽语言模型任务都使用带有softmax的交叉熵作为损失函数。

```{.python .input  n=2}
net = BERTModel(len(train_set.vocab), embed_size = 100)
net.initialize(init.Xavier(), ctx=mx.gpu())
nsp_loss = mx.gluon.loss.SoftmaxCELoss()
mlm_loss = mx.gluon.loss.SoftmaxCELoss()
```

### 前向函数
在这个函数中，我们将数据送入模型中，并得到下一句预测任务和遮蔽语言模型的输出，并通过上一步定义的损失函数以计算损失。
```{.python .input  n=2}
def forward(data, model, mlm_loss, nsp_loss, vocab_size, dtype):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data
    num_masks = masked_weight.sum() + 1e-8
    valid_length = valid_length.reshape(-1)
    masked_id = masked_id.reshape(-1)
    valid_length_typed = valid_length.astype(dtype, copy=False)
    _, classified, decoded = model(input_id.as_in_context(mx.gpu()), segment_id.as_in_context(mx.gpu()), valid_length_typed.as_in_context(mx.gpu()),
                                      masked_position.as_in_context(mx.gpu()))
    decoded = decoded.reshape((-1, vocab_size))
    
    ls1 = mlm_loss(decoded.astype('float32', copy=False),
                   masked_id.as_in_context(mx.gpu()), masked_weight.as_in_context(mx.gpu()).reshape((-1, 1)))
    ls2 = nsp_loss(classified.astype('float32', copy=False), next_sentence_label.as_in_context(mx.gpu()))
    
    ls1 = ls1.sum() / num_masks.as_in_context(mx.gpu())
    ls2 = ls2.mean()
    ls = ls1 + ls2
    return ls, next_sentence_label, classified, masked_id, decoded, \
           masked_weight, ls1, ls2, valid_length.astype('float32', copy=False)
```
### 训练函数

```{.python .input  n=2}
def train(data_eval, model, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, num_epochs, dtype):
    """Evaluation function."""
    trainer = mx.gluon.Trainer(model.collect_params(), 'adam')
    for epoch in range(num_epochs):
        eval_begin_time = time.time()
        begin_time = time.time()
        step_num = 0
        running_mlm_loss = running_nsp_loss = 0
        total_mlm_loss = total_nsp_loss = 0
        running_num_tks = 0
        for _, data_batch in enumerate(data_eval):
            step_num += 1
            with mx.autograd.record():
                out = forward(data_batch, model, mlm_loss, nsp_loss, vocab_size, dtype)
                (ls, next_sentence_label, classified, masked_id,
                 decoded, masked_weight, ls1, ls2, valid_length) = out
            ls.backward()
            trainer.step(1)
            running_mlm_loss += ls1.as_in_context(mx.cpu())
            running_nsp_loss += ls2.as_in_context(mx.cpu())
            running_num_tks += valid_length.sum().as_in_context(mx.cpu())

            # logging
            if (step_num + 1) % (log_interval) == 0:
                total_mlm_loss += running_mlm_loss
                total_nsp_loss += running_nsp_loss
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = running_num_tks = 0

        eval_end_time = time.time()
        # accumulate losses from last few batches, too
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
现在就可以训练模型了。
```{.python .input  n=2}
train(train_iter, net, nsp_loss, mlm_loss, len(train_set.vocab), mx.gpu(), 20, 3, 'float32')
```
## 下游任务

在获得训练好的BERT后，最终只需在BERT的输出层上加简单的多层感知机或线性分类器即可。

对于单句和句对分类任务，直接取“[CLS]”位置的输出表示作为下游任务的输入。
对于问答这种抽取式任务，取第二个句子每个位置的输出表示作为下游任务的输入。
对于序列标注任务，取除了“[CLS]”位置外其他位置的输出表示作为下游任务的输入。


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