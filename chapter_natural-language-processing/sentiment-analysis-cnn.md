# 文本情感分类：使用卷积神经网络（textCNN）

在早先的章节中，我们探究了各式各样的计算机视觉任务，例如图像分类、物体检测、语义分割和样式迁移。由于图像中每个像素和它周围的像素高度相关，而卷积神经网络恰好善于抓取空间相关性，因此我们发现这类网络在计算机视觉领域有着出众的表现。

在自然语言处理领域中，文本序列可以被看作是时序数据。能不能将时序数据当作一维的图像，并使用卷积神经网络分析这类数据呢？答案是肯定的：Kim在2014年提出的基于卷积神经网络的短文本分类模型可谓这一领域的开山之作 [1]。该模型也称为textCNN。本节将基于textCNN介绍如何使用卷积神经网络对文本情感进行分类。

本节所有卷积操作的步幅均为1。


## 一维卷积层

在“卷积神经网络”篇章中，我们介绍了如何使用二维卷积层处理图像。既然我们将时序数据与一维图像做类比，不妨先介绍一维卷积层的计算。


### 一维互相关运算

和二维卷积一样，一维卷积通常也使用一维的互相关运算。图10.2演示了如何对一个宽为7的输入作用宽为2的核来计算输出。

![一维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$0\times1+1\times2=2$。](../img/conv1d.svg)

可以看到输出的宽度为$7-2+1=6$，且第一个元素是由输入的最左边的宽为2的子数组与核数组按元素相乘后再相加得来。设输入、核以及输出分别为`X`、`K`和`Y`，即`Y[0] = (X[0:2] * K).sum()`，这里`X`、`K`和`Y`的类型都是NDArray。接下来我们将输入中高亮部分的宽为2的窗口向右滑动一列来计算`Y`的第二个元素。输出中其他元素的计算以此类推。

下面我们将上述过程实现在`corr1`函数里，它接受`X`和`K`，输出`Y`。

```{.python .input  n=13}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
import random

def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i : i + w] * K).sum()
    return Y
```

让我们重现图10.2中一维互相关运算的结果。

```{.python .input  n=11}
X = nd.array([0, 1, 2, 3, 4, 5, 6])
K = nd.array([1, 2])
corr1d(X, K)
```

### 多输入通道的一维互相关运算

多输入通道的一维互相关运算也与多输入通道的二维互相关运算类似：在每个通道上，将核与相应的输入做一维互相关运算，并将通道之间的结果相加得到输出结果。图10.3展示了含3个输入通道的一维互相关运算。

![含3个输入通道的一维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$。](../img/conv1d-channel.svg)

让我们重现图10.3中多输入通道的一维互相关运算的结果。

```{.python .input  n=12}
def corr1d_multi_in(X, K):
    # 我们首先沿着 X 和 K 的第 0 维（通道维）遍历。然后使用 * 将结果列表 (list) 变成
    # add_n 的位置参数（positional argument）来进行相加。
    return nd.add_n(*[corr1d(x, k) for x, k in zip(X, K)])

X = nd.array([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = nd.array([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

由二维互相关运算的定义可知，多输入通道的一维互相关运算可以看作是单输入通道的二维互相关运算。如图10.4所示，我们也可以将图10.3中多输入通道的一维互相关运算以等价的单输入通道的二维互相关运算呈现。

![单输入通道的二维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$。](../img/conv1d-2d.svg)


图10.2和图10.3中的输出都只有一个通道。我们在[“多输入和输出通道”](../chapter_convolutional-neural-networks/channels.md)一节中介绍了如何在二维卷积层中指定多个输出通道。类似地，我们也可以在一维卷积层指定多个输出通道，从而拓展卷积层中的模型参数。

## 时序最大池化层

时序最大池化层（max-over-time pooling）实际上是最大池化层在时序数据上的应用：假设输入包含多个通道，各通道由不同时间步上的数值组成，各通道的输出即该通道所有时间步中最大的数值。因此，时序最大池化层的输入在各个通道上的时间步数可以不同。

为提升计算性能，我们常常将不同长度的时序样本组成一个小批量，并通过在较短序列后附加特殊字符（例如0）令批量中各时序样本长度相同。这些人为添加的特殊字符当然是无意义的。由于时序最大池化的主要目的是抓取时序中最重要的特征，它通常能使模型不受人为添加字符的影响。


## textCNN的设计

textCNN主要使用了一维卷积层和时序最大池化层。假设输入的文本序列由$n$个词组成，每个词用$d$维的词向量表示。那么输入序列的宽为$n$，输入通道数为$d$。textCNN的计算主要分为以下几步：

1. 定义多个一维卷积核，并使用这些卷积核对输入分别做卷积计算。
2. 对输出的所有通道分别做时序最大池化，再将这些通道的池化输出值连结为向量。
3. 通过全连接层将连结后的向量变换为有关各类别的输出。这一步可以使用丢弃层应对过拟合。

![textCNN的设计。](../img/textcnn.svg)

图10.5用一个例子解释了textCNN的设计。这里的输入是一个有11个词的句子，每个词用6维词向量表示。因此输入序列的宽为11，输入通道数为6。给定2个一维卷积核，核宽分别为2和4，输出通道数分别设为4和5。因此，一维卷积计算后，4个输出通道的宽为$11-2+1=10$，而其他5个通道的宽为$11-4+1=8$。尽管每个通道的宽不同，我们依然可以对各个通道做时序最大池化，并将9个通道的池化输出连结成一个9维向量。最终，我们使用全连接将9维向量变换为2维输出：正面情感和负面情感的预测。

下面我们来实现textCNN模型并用它对文本情感进行分类。


## 获取和处理IMDb数据集

我们依然使用和上一节中相同的IMDb数据集做情感分析。以下获取和处理数据集的步骤与上一节中的相同。

```{.python .input  n=2}
# 下载数据集。
gb.download_imdb()

# 读取训练和测试数据集。
train_data = gb.read_imdb('aclImdb', 'train')
test_data = gb.read_imdb('aclImdb', 'test')
random.shuffle(train_data)
random.shuffle(test_data)

# 使用空格分词。
train_tokenized, test_tokenized = gb.get_tokenized_imdb(train_data, test_data)

# 创建词典。
token_counter = gb.count_tokens(train_tokenized)
vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>',
                              reserved_tokens=None)

# 预处理数据。
train_features, test_features, train_labels, test_labels = gb.preprocess_imdb(
    train_tokenized, test_tokenized, train_data, test_data, vocab)
```

## 加载预训练的词向量

我们加载预训练的100维GloVe词向量。

```{.python .input  n=7}
# 加载预训练的词向量。
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

## 定义模型

下面我们来实现textCNN模型。实验中，每个词的词向量由两套词向量连结而成：`embedding_static`设为100维GloVe词向量且训练中不更新；`embedding_non_static`初始化为100维GloVe词向量并在训练中不断迭代。模型定义中的`Conv1D`即一维卷积层，`GlobalMaxPool1D`即时序最大池化层。

```{.python .input  n=10}
class TextCNN(nn.Block):
    def __init__(self, vocab, embedding_size, ngram_kernel_sizes,
                 nums_channels, num_outputs, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.ngram_kernel_sizes = ngram_kernel_sizes
        self.embedding_static = nn.Embedding(len(vocab), embedding_size)
        self.embedding_non_static = nn.Embedding(len(vocab), embedding_size)
        for i in range(len(ngram_kernel_sizes)):
            # 一维卷积层。
            conv = nn.Conv1D(nums_channels[i],
                             kernel_size=ngram_kernel_sizes[i], strides=1,
                             activation='relu')
            # 时序最大池化层。
            pool = nn.GlobalMaxPool1D()
            # 将 self.conv_{i} 置为第 i 个 conv。
            setattr(self, 'conv_{i}', conv)
            # 将 self.pool_{i} 置为第 i 个 pool。
            setattr(self, 'pool_{i}', pool)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(num_outputs)

    def forward(self, inputs):
        # 将 inputs 的形状由（批量大小，词数）变换为（词数，批量大小）。
        inputs = inputs.T
        # 根据 Conv1D 要求的输入形状，embeddings_static 和 embeddings_non_static
        # 的形状由（词数，批量大小，词向量维度）变换为（批量大小，词向量维度，词数）。
        embeddings_static = self.embedding_static(inputs).transpose((1, 2, 0))
        embeddings_non_static = self.embedding_non_static(
            inputs).transpose((1, 2, 0))
        # 将 embeddings_static 和 embeddings_non_static 按词向量维度连结。
        embeddings = nd.concat(embeddings_static, embeddings_non_static,
                               dim=1)
        # 对于第 i 个卷积核，在时序最大池化后会得到一个形状为
        # （批量大小，nums_channels[i]，1）的矩阵。使用 flatten 函数将它形状压成
        # （批量大小，nums_channels[i]）。
        encoding = [
            nd.flatten(self.get_pool(i)(self.get_conv(i)(embeddings)))
            for i in range(len(self.ngram_kernel_sizes))]
        # 将批量按各通道的输出连结。encoding的形状：
        # （批量大小，nums_channels 各元素之和）。
        encoding = nd.concat(*encoding, dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

    # 返回 self.conv_{i}。
    def get_conv(self, i):
        return getattr(self, 'conv_{i}')

    # 返回 self.pool_{i}。
    def get_pool(self, i):
        return getattr(self, 'pool_{i}')
```

## 实验设置

我们定义3个卷积核，它们的核宽分别为3、4和5，输出通道数均为100。

```{.python .input  n=11}
num_outputs = 2
lr = 0.001
num_epochs = 5
batch_size = 64
embed_size = 100
ngram_kernel_sizes = [3, 4, 5]
nums_channels = [100, 100, 100]
ctx = gb.try_all_gpus()
```

接下来，我们用预训练的100维GloVe词向量初始化`embedding_static`和`embedding_non_static`。其中只有`embedding_static`在训练中不更新模型参数。

```{.python .input}
net = TextCNN(vocab, embed_size, ngram_kernel_sizes, nums_channels,
              num_outputs)
net.initialize(init.Xavier(), ctx=ctx)
# embedding_static 和 embedding_non_static 均使用预训练的词向量。
net.embedding_static.weight.set_data(glove_embedding.idx_to_vec)
net.embedding_non_static.weight.set_data(glove_embedding.idx_to_vec)
# 训练中不更新 embedding_static 的词向量，即不更新 embedding_static 的模型参数。
net.embedding_static.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
```

## 训练并评价模型

现在我们可以加载数据并训练模型了。

```{.python .input  n=30}
train_set = gdata.ArrayDataset(train_features, train_labels)
test_set = gdata.ArrayDataset(test_features, test_labels)
train_loader = gdata.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True)
test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)
gb.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs)
```

下面我们使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input}
gb.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
gb.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## 小结

* 我们可以使用一维卷积来处理时序数据。
* 多输入通道的一维互相关运算可以看作是单输入通道的二维互相关运算。
* 时序最大池化层的输入在各个通道上的时间步数可以不同。
* textCNN主要使用了一维卷积层和时序最大池化层。


## 练习

* 动手调参，从准确率和运行效率比较情感分析的两类方法：使用循环神经网络和使用卷积神经网络。
* 使用上一节练习中介绍的三种方法：调节超参数、使用更大的预训练词向量和使用spacy分词工具，你能使模型在测试集上的准确率提高到0.87以上吗？
* 你还能将textCNN应用于自然语言处理的哪些任务中？


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![](../img/qr_sentiment-analysis-cnn.svg)


## 参考文献

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
