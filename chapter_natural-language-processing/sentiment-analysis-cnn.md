# 文本情感分类：使用卷积神经网络（textCNN）

在早先的章节中我们探究了使用二维卷积神经网来处理二维图像数据。但文本数据只有一个维度，通常我们将它当做时间维度，从而适合循环神经网络。但我们也可以将文本当做是一维图像，从而可以用一维卷积神经网来抓取词与临近词之间的关联。本节我们将介绍将卷积神经网络应用到文本数据的开创性工作之一：textCNN [1]。先导入本节需要的包和模块。

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
```

## 一维卷积层

在介绍模型前我们先来研究一维卷积层。和二维卷积层一样，一维卷积层使用一维的互相关运算。图10.4演示了如何对一个宽为7的输入作用宽为2的核来计算输出。

![一维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$0\times1+1\times2=2$。](../img/conv1d.svg)

可以看到输出的宽度为$7-2+1=6$，且第一个元素是由输入的最左边的宽为2的子数组与核数组按元素相乘后再相加得来。设输入、核以及输出分别为`X`、`K`和`Y`，即`Y[0] = (X[0:2] * K).sum()`，这里`X`、`K`和`Y`的类型都是NDArray。接下来我们将输入中高亮部分的宽为2的窗口向右滑动一列来计算`Y`的第二个元素。输出中其他元素的计算以此类推。

下面我们将上述过程实现在`corr1`函数里，它接受`X`和`K`，输出`Y`。

```{.python .input  n=3}
def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

让我们重现图10.4中一维互相关运算的结果。

```{.python .input  n=4}
X, K = nd.array([0, 1, 2, 3, 4, 5, 6]), nd.array([1, 2])
corr1d(X, K)
```

多输入通道的一维互相关运算也与多输入通道的二维互相关运算类似：在每个通道上，将核与相应的输入做一维互相关运算，并将通道之间的结果相加得到输出结果。图10.5展示了含3个输入通道的一维互相关运算。

![含3个输入通道的一维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$。](../img/conv1d-channel.svg)

让我们重现图10.5中多输入通道的一维互相关运算的结果。

```{.python .input  n=5}
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

由二维互相关运算的定义可知，多输入通道的一维互相关运算可以看作是单输入通道的二维互相关运算。如图10.6所示，我们也可以将图10.5中多输入通道的一维互相关运算以等价的单输入通道的二维互相关运算呈现。但这里核的高等于输入的高。

![单输入通道的二维互相关运算。高亮部分为第一个输出元素及其计算所使用的输入和核数组元素：$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$。](../img/conv1d-2d.svg)

由于这个等价关系，多输通道的一维互相关运算等价于单输出通道和多输出通道的二维互相关运算。可以参考[“多输入和输出通道”](../chapter_convolutional-neural-networks/channels.md)一节中得到详细的描述。

## 时序最大池化层

类似的我们有一维池化层。textCNN中使用的时序最大池化层（max-over-time pooling）对应全局一维最大池化层，它将每个输入通道里的值取最大作为输出。上一节我们看到了将不同长度的样本通过补0组成一个小批量。这些人为添加的特殊字符当然是无意义的。通过时序最大池化层，我们只考虑样本中的最大值，从而使得模型不受人为添加字符的影响。

## 获取和处理IMDb数据集

我们依然使用和上一节中相同的IMDb数据集做情感分析。以下获取和处理数据集的步骤与上一节中的相同。

```{.python .input  n=2}
batch_size = 64
gb.download_imdb()
train_data, test_data = gb.read_imdb('train'), gb.read_imdb('test')
vocab = gb.get_vocab_imdb(train_data)
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    *gb.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(
    *gb.preprocess_imdb(test_data, vocab)), batch_size)
```

## textCNN 模型

textCNN主要使用了一维卷积层和时序最大池化层。假设输入的文本序列由$n$个词组成，每个词用$d$维的词向量表示。那么输入样本的宽为$n$，高为1，输入通道数为$d$。textCNN使用多个并行的有着不同核大小的一维卷积层，这样每个卷积层能抓取不同上下文窗口大小的信息。对于每个卷积层的输出作用时序最大池化层。接着将所有时序最大池化层的输出在通道维上连结为向量，并最后使用全连接层来输出类别。

![textCNN的设计。](../img/textcnn.svg)

图10.7用一个例子解释了textCNN的设计。这里的输入是一个有11个词的句子，每个词用6维词向量表示。因此输入序列的宽为11，输入通道数为6。给定2个一维卷积核，核宽分别为2和4，输出通道数分别设为4和5。因此，一维卷积计算后，4个输出通道的宽为$11-2+1=10$，而其他5个通道的宽为$11-4+1=8$。尽管每个通道的宽不同，我们依然可以对各个通道做时序最大池化，并将9个通道的池化输出连结成一个9维向量。最终，我们使用全连接将9维向量变换为2维输出：正面情感和负面情感的预测。

下面我们来实现textCNN模型。跟上一节相比，除了用一维卷积层替换循环神经网络外，这里我们使用了两个嵌入层，一个的权重固定，另一个则参与训练。

```{.python .input  n=10}
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层。
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最大池化层没有权重，所以可以共用一个实例。
        self.pool = nn.GlobalMaxPool1D()  
        self.convs = nn.Sequential()  # 创建多个一维卷积层。
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))
            
    def forward(self, inputs):
        # 将两个嵌入层的输出，其形状是（批量大小，词数，词向量维度），在上连结。
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 然后将词向量维度，这是一维卷积层的通道维，调整到第二维。
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于第 i 个一维卷积层，在时序最大池化后会得到一个形状为（批量大小，通道大小，1）
        # 的矩阵。使用 flatten 函数去掉最后一个维度，然后在通道维上连结。
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 作用丢弃层后使用全连接层得到输出。
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

创建一个TextCNN类实例。它有3个卷积层，它们的核宽分别为3、4和5，输出通道数均为100。

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = gb.try_all_gpus()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)
```

### 加载预训练的词向量

同上一节一样加载预训练的100维GloVe词向量来初始化词嵌入层。

```{.python .input  n=7}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

### 训练并评价模型

现在我们可以训练模型了。

```{.python .input  n=30}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
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
