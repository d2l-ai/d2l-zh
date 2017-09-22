# 循环神经网络 --- 从0开始

前面的教程里我们使用的网络都属于**前馈神经网络**。为什么叫前馈是整个网络是一条链（回想下`gluon.nn.Sequential`），每一层的结果都是反馈给下一层。这一节我们介绍**循环神经网络**，这里每一层不仅输出给下一层，同时还输出一个**隐藏状态**，给当前层在处理下一个样本时使用。下图展示这两种网络的区别。

![](../img/rnn_1.png)

循环神经网络的这种结构使得它适合处理前后有依赖关系的样本。我们拿语言模型举个例子来解释这个是怎么工作的。语言模型的任务是给定句子的前*T*个字符，然后预测第*T+1*个字符。假设我们的句子是“你好世界”，使用前馈神经网络来预测的一个做法是，在时间1输入“你”，预测”好“，时间2向同一个网络输入“好”预测“世”。下图左边展示了这个过程。

![](../img/rnn_2.png)

注意到一个问题是，当我们预测“世”的时候只给了“好”这个输入，而完全忽略了“你”。直觉上“你”这个词应该对这次的预测比较重要。虽然这个问题通常可以通过**n-gram**来缓解，就是说预测第*T+1*个字符的时候，我们输入前*n*个字符。如果*n=1*，那就是我们这里用的。我们可以增大*n*来使得输入含有更多信息。但我们不能任意增大*n*，因为这样通常带来模型复杂度的增加从而导致需要大量数据和计算来训练模型。

循环神经网络使用一个隐藏状态来记录前面看到的数据来帮助当前预测。上图右边展示了这个过程。在预测“好”的时候，我们输出一个隐藏状态。我们用这个状态和新的输入“好”来一起预测“世”，然后同时输出一个更新过的隐藏状态。我们希望前面的信息能够保存在这个隐藏状态里，从而提升预测效果。

在更加正式的介绍这个模型前，我们先去弄一个比“你好世界“稍微复杂点的数据。

## 《时间机器》数据集

我们用《时间机器》这本书做数据集主要是因为[古登堡计划](http://www.gutenberg.org)计划使得可以免费下载，而且我们看了太多用莎士比亚作为例子的教程。下面我们读取这个数据并看看前面500个字符（char）是什么样的：

```{.python .input  n=1}
with open("../data/timemachine.txt") as f:
    time_machine = f.read()
print(time_machine[0:500])
```

接着我们稍微处理下数据集。包括全部改为小写，去除换行符，然后截去后面一段使得接下来的训练会快一点。

```{.python .input  n=2}
time_machine = time_machine.lower().replace('\n', '').replace('\r', '')
time_machine = time_machine[0:10000]
```

## 字符的数值表示

先把数据里面所有不同的字符拿出来做成一个字典：

```{.python .input  n=3}
character_list = list(set(time_machine))
character_dict = dict([(char,i) for i,char in enumerate(character_list)])

vocab_size = len(character_dict)

print('vocab size:', vocab_size)
print(character_dict)
```

然后可以把每个字符转成从0开始的指数(index)来方便之后的使用。

```{.python .input  n=4}
time_numerical = [character_dict[char] for char in time_machine]

sample = time_numerical[:40]

print('chars: \n', ''.join([character_list[idx] for idx in sample]))
print('\nindices: \n', sample)

```

## 数据读取

同前一样我们需要每次随机读取一些（`batch_size`个）样本和其对用的标号。这里的样本跟前面有点不一样，这里一个样本通常包含一系列连续的字符（前馈神经网络里可能每个字符作为一个样本）。

如果我们把序列长度（`seq_len`）设成10，那么一个可能的样本是`The Time T`。其对应的标号仍然是长为10的序列，每个字符是对应的样本里字符的后面那个。例如前面样本的标号就是`he Time Tr`。

下面代码每次从数据里随机采样一个批量：

```{.python .input  n=5}
import random
from mxnet import nd

def data_iter(batch_size, seq_len, ctx=None):
    num_examples = (len(time_numerical)-1) // seq_len
    num_batches = num_examples // batch_size
    # 随机化样本
    idx = list(range(num_examples))
    random.shuffle(idx)
    # 返回seq_len个数据
    def _data(pos):
        return time_numerical[pos:pos+seq_len]
    for i in range(num_batches):
        # 每次读取batch_size个随机样本
        examples = idx[i:i+batch_size]
        data = nd.array(
            [_data(j*seq_len) for j in examples], ctx=ctx)
        label = nd.array(
            [_data(j*seq_len+1) for j in examples], ctx=ctx)
        yield data, label
```

看下读出来长什么样：

```{.python .input  n=6}
for data, label in data_iter(batch_size=3, seq_len=8):
    print('data: ', data, '\n\nlabel:', label)
    break
```

## 循环神经网络

在对输入输出数据有了解后，我们来正式介绍循环神经网络。

首先回忆下单隐层的前馈神经网络的定义，假设隐层的激活函数是$\phi$，那么这个隐层的输出就是

$$H = \phi(X W_{wh} + b_h)$$

最终的输出是

$$\hat{Y} = \text{softmax}(H W_{hy} + b_y)$$

（跟[多层感知机](../chapter02_multilayer-neural-network/mlp-scratch.md)相比，这里我们把下标从$W_1$和$W_2$改成了意义更加明确的$W_{wh}$和$W_{hy}$)

将上面网络改成循环神经网络，我们首先对输入输出加上时间戳$t$。假设$X_t$是序列中的第$t$个输入，对应的隐层输出和最终输出是$H_t$和$\hat{Y}_t$。循环神经网络只需要在计算隐层的输出的时候加上跟前一时间输入的加权和，为此我们引入一个新的可学习的权重$W_{hh}$：

$$H_t = \phi(X_t  W_{xh} + H_{t-1} W_{hh} + b_h )$$

输出的计算跟前一致：

$$\hat{Y}_t = \text{softmax}(H_t W_{hy} + b_y)$$

一开始我们提到过，隐层输出（又叫隐藏状态）可以认为是这个网络的记忆。它存储前面时间里面的信息。我们的输出是完全只基于这个状态。最开始的状态，$H_{-1}$，通常会被初始为0.

## Onehot编码

注意到每个字符现在是用一个整数来表示，而输入进网络我们需要一个定长的向量。一个常用的办法是使用onehot来将其表示成向量。就是说，如果值是$i$, 那么我们创建一个全0的长为`vocab_size`的向量，并将其第$i$位表示成1.

```{.python .input  n=32}
nd.one_hot(nd.array([0,4]), vocab_size)
```

记得前面我们每次得到的数据是一个`batch_size x seq_len`的批量。下面这个函数将其转换成`seq_len`个可以输入进网络的`batch_size x vocba_size`的矩阵

```{.python .input  n=34}
def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]

inputs = get_inputs(data)
print('input length: ',len(inputs))
print('input[0] shape: ', inputs[0].shape)
```

## 初始化模型参数

模型的输入和输出维度都是`vocab_size`。

```{.python .input  n=35}
import mxnet as mx

# 尝试使用 GPU
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
print('Will use ', ctx)

num_hidden = 256
weight_scale = .01

# 隐含层
Wxh = nd.random_normal(shape=(vocab_size,num_hidden), ctx=ctx) * weight_scale
Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx) * weight_scale
bh = nd.zeros(num_hidden, ctx=ctx)
# 输出层
Why = nd.random_normal(shape=(num_hidden,vocab_size), ctx=ctx) * weight_scale
by = nd.zeros(vocab_size, ctx=ctx)

params = [Wxh, Whh, bh, Why, by]
for param in params:
    param.attach_grad()
```

## 定义模型

我们将前面的模型公式定义直接写成代码。

```{.python .input  n=23}
def rnn(inputs, H):
    # inputs: seq_len 个 batch_size x vocab_size 矩阵
    # H: batch_size x num_hidden 矩阵
    # outputs: seq_len 个 batch_size x vocab_size 矩阵
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, Wxh) + nd.dot(H, Whh) + bh)
        Y = nd.dot(H, Why) + by
        outputs.append(Y)
    return (outputs, H)
```

做个简单的测试：

```{.python .input  n=36}
state = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state)

print('output length: ',len(outputs))
print('output[0] shape: ', outputs[0].shape)
print('state shape: ', state_new.shape)
```

## 预测序列

在做预测时我们只需要给定时间0的输入和起始隐藏状态。然后我们每次将上一个时间的输出作为下一个时间的输入。

![](../img/rnn_3.png)

```{.python .input  n=37}
def predict(prefix, num_chars):
    # 预测以 prefix 开始的接下来的 num_chars 个字符
    prefix = prefix.lower()
    state = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    output = [character_dict[prefix[0]]]
    for i in range(num_chars+len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        Y, state = rnn(get_inputs(X), state)
        #print(Y)
        if i < len(prefix)-1:
            next_input = character_dict[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([character_list[i] for i in output])
```

## 梯度剪裁

在求梯度时，循环神经网络因为需要反复做`O(seq_len)`次乘法，有可能会有数值稳定性问题。（想想 $2^{40}$和$0.5^{40}$）。一个常用的做法是如果梯度特别大，那么就投影到一个比较小的尺度上。假设我们把所有梯度接成一个向量 $\boldsymbol{g}$，假设剪裁的阈值是$\theta$，那么我们这样剪裁使得$\|\boldsymbol{g}\|$不会超过$\theta$：

$$ \boldsymbol{g} = \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

```{.python .input}
def grad_clipping(params, theta):
    norm = nd.array([0.0], ctx)
    for p in params:
        norm += nd.sum(p.grad ** 2)
    norm = nd.sqrt(norm).asscalar()
    if norm > theta:
        for p in params:
            p.grad[:] *= theta/norm
```

## 训练模型

下面我们可以还是训练模型。跟前面前置网络的教程比，这里只有两个不同。

1. 通常我们使用Perplexit(PPL)这个指标。可以简单的认为就是对交叉熵做exp运算使得数值更好读。
2. 在更新前我们对梯度做剪裁

```{.python .input  n=81}
from mxnet import autograd
from mxnet import gluon
from math import exp

epochs = 200
seq_len = 35
learning_rate = .1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

for e in range(epochs+1):
    train_loss, num_examples = 0, 0
    state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for data, label in data_iter(batch_size, seq_len, ctx):
        with autograd.record():
            outputs, state = rnn(get_inputs(data), state)
            # reshape label to (batch_size*seq_len, )
            # concate outputs to (batch_size*seq_len, vocab_size)
            label = label.T.reshape((-1,))
            outputs = nd.concat(*outputs, dim=0)
            loss = softmax_cross_entropy(outputs, label)
        loss.backward()

        grad_clipping(params, 5)
        utils.SGD(params, learning_rate)

        train_loss += nd.sum(loss).asscalar()
        num_examples += loss.size

    if e % 20 == 0:
        print("Epoch %d. PPL %f" % (e, exp(train_loss/num_examples)))
        print(' - ', predict('The Time Ma', 100))
        print(' - ', predict("The Medical Man rose, came to the lamp,", 100), '\n')

```

可以看到一开始学到简单的字符，然后简单的词，接着是复杂点的词，然后看上去似乎像个句子了。

## 结论

通过隐藏状态，循环神经网络很够更好的使用数据里的时序信息。

## 练习

调调参数（数据集大小，模型复杂度，学习率），看看对Perplexity和预测的结果造成的区别。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/989)
