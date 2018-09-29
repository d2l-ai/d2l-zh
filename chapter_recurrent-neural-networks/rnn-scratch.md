# 循环神经网络的从零开始实现

在本节，我们将从零开始实现一个基于字符级循环神经网络的语言模型，并在周杰伦专辑歌词数据集上训练一个模型来进行歌词创作。首先，我们读取周杰伦专辑歌词数据集。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = gb.load_data_jay_lyrics()
```

## One-hot向量

为了将词表示成向量来输入进神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为$N$（即`vocab_size`），每个字符已经同一个从0到$N-1$的连续整数值索引一一对应。如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为$N$的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。

```{.python .input  n=2}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  0.  0. ...,  0.  0.  0.]\n [ 0.  0.  1. ...,  0.  0.  0.]]\n<NDArray 2x1027 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们每次采样的小批量的形状是（`batch_size`, `num_steps`）。下面的函数将这样的小批量转换成`num_steps`个可以输入进网络的形状为（`batch_size`, `vocab_size`）的矩阵。也就是总时间步$T=$ `num_steps`，时间步$t$的输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n=$ `batch_size`，$d=$ `vocab_size`（one-hot向量长度）。

```{.python .input  n=3}
def to_onehot(X, size):  # 本函数已保存在 gluonbook 包中方便以后使用。
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "(5, (2, 1027))"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 初始化模型参数

接下来，我们初始化模型参数。隐藏单元个数 `num_hiddens`是一个超参数。

```{.python .input  n=4}
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = gb.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数。
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数。
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度。
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "will use cpu(0)\n"
 }
]
```

## 定义模型

我们根据循环神经网络的计算表达式实现该模型。首先定义`init_rnn_state`函数来返回初始化的隐藏状态。它返回由一个形状为（`batch_size`，`num_hiddens`）的值为0的NDArray组成的元组。使用元组是为了更方便处理隐藏状态含有多个NDArray的情况。

```{.python .input  n=5}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

下面的`rnn`函数定义了在一个时间步里如何计算隐藏状态和输出。这里的激活函数使用了tanh函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，当元素在实数域上均匀分布时，tanh函数值的均值为0。

```{.python .input  n=6}
def rnn(inputs, state, params):
    # inputs 和 outputs 皆为 num_steps 个形状为（batch_size，vocab_size）的矩阵。
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

做个简单的测试来观察输出结果的个数（时间步个数），以及第一个时间步的输出层输出形状和隐藏状态形状。

```{.python .input  n=7}
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(5, (2, 1027), (2, 256))"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义预测函数

以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。这个函数稍显复杂，其中我们将循环神经单元`rnn`设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。

```{.python .input  n=8}
# 本函数已保存在 gluonbook 包中方便以后使用。
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]    
    for t in range(num_chars + len(prefix)):
        # 将上一时间步的输出作为当前时间步的输入。
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态。
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是 prefix 里的字符或者当前的最佳预测字符。
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

测试一下这个函数。因为模型参数为随机值，所以预测结果也是随机的。

```{.python .input  n=9}
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "'\u5206\u5f00\u6597\u4e24\u6696\u7238\u574a\u513f\u7b49\u4e0a\u5f77\u666f\u661f'"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 裁剪梯度

循环神经网络中较容易出现梯度衰减或爆炸，其原因我们会在[下一节](bptt.md)解释。为了应对梯度爆炸，我们可以裁剪梯度（clipping gradient）。假设我们把所有模型参数梯度的元素拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后梯度

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

的$L_2$范数不超过$\theta$。

```{.python .input  n=10}
# 本函数已保存在 gluonbook 包中方便以后使用。
def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## 困惑度

语言模型里我们通常使用困惑度（perplexity）来评价模型的好坏。回忆一下[“Softmax回归”](../chapter_deep-learning-basics/softmax-regression.md)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

* 最佳情况下，模型总是把标签类别的概率预测为1。此时困惑度为1。
* 最坏情况下，模型总是把标签类别的概率预测为0。此时困惑度为正无穷。
* 基线情况下，模型总是预测所有类别的概率都相同。此时困惑度为类别数。

显然，任何一个有效模型的困惑度必须小于类别数。在本例中，困惑度必须小于词典中不同的字符数`vocab_size`。相对于交叉熵损失，困惑度的值更大，使得模型比较时更加清楚。例如“模型一比模型二的困惑度小1”比“模型一比模型二的交叉熵损失小0.01”感官上更加清楚一些。

## 定义模型训练函数

跟之前章节的训练模型函数相比，这里有以下几个不同。

1. 使用困惑度（perplexity）评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。

同样这个函数由于考虑到后面将介绍的循环神经网络，所以实现更长一些。

```{.python .input  n=11}
# 本函数已保存在 gluonbook 包中方便以后使用。
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = gb.data_iter_random
    else:
        data_iter_fn = gb.data_iter_consecutive     
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在 epoch 开始时初始化隐藏变量。
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        loss_sum, start = 0.0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for t, (X, Y) in enumerate(data_iter):
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏变量。
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用 detach 函数从计算图分离隐藏状态变量。
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs 有 num_steps 个形状为（batch_size，vocab_size）的矩阵。
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为（num_steps * batch_size，vocab_size）。
                outputs = nd.concat(*outputs, dim=0)
                # Y 的形状是（batch_size，num_steps），转置后再变成长
                # batch * num_steps 的向量，这样跟输出的行一一对应。
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差。
                l = loss(outputs, y).mean()
            l.backward()
            # 裁剪梯度后使用 SGD 更新权重。
            grad_clipping(params, clipping_theta, ctx)
            gb.sgd(params, lr, 1)  # 因为已经误差取过均值，梯度不用再做平均。
            loss_sum += l.asscalar()

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(loss_sum / (t + 1)), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
```

## 训练模型并创作歌词

现在我们可以训练模型了。首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input  n=12}
num_epochs, num_steps, batch_size, lr, clipping_theta = 200, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
```

下面采用随机采样训练模型并创作歌词。

```{.python .input  n=13}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 50, perplexity 69.470822, time 1.64 sec\n - \u5206\u5f00 \u6211\u60f3\u8981\u8fd9\u60f3  \u6211\u6709\u4f60\u7684\u53ef\u7231 \u6211\u60f3\u8981\u4f60\u60f3  \u6211\u6709 \u4f60\u60f3 \u6211\u60f3 \u8fd9\u60f3 \u6211\u4e0d\u80fd \u60f3\u4f60 \u4f60\u7684\u4f60  \u4f60\u7684\u4f60 \u6211\n - \u4e0d\u5206\u5f00 \u6211\u60f3\u60f3\u4f60\u60f3  \u6211\u6709 \u4f60\u60f3 \u6211\u60f3 \u8fd9\u60f3 \u6211\u4e0d\u80fd \u60f3\u4f60 \u4f60\u7684\u4f60  \u4f60\u7684\u4f60 \u6211\u6709 \u8fd9\u751f  \u6211\u6709\u8981\u4f60 \u6211\u6709\u60f3\nepoch 100, perplexity 9.787908, time 1.66 sec\n - \u5206\u5f00 \u6211\u60f3\u60f3\u597d\u751f\u6d3b \u4e0d\u77e5\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u79bb\u5f00\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u4e86\u8fd9\u751f\u594f \u6211\u77e5\u9053\u597d\u751f\u5c0f \u4e0d\u77e5\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u79bb\u5f00\u6211 \n - \u4e0d\u5206\u5f00\u67f3 \u6211\u4e0d\u80fd\u518d\u60f3\u4f60 \u4e0d\u77e5\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u79bb\u5f00\u6211 \u4e0d\u77e5\u4e0d\u89c9 \u6211\u5df2\u4e86\u8fd9\u751f\u594f \u6211\u77e5\u9053\u597d\u751f\u5c0f \u4e0d\u77e5\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u79bb\u5f00\u6211\nepoch 150, perplexity 2.838760, time 1.70 sec\n - \u5206\u5f00 \u5feb\u4f7f\u7528\u53cc\u622a\u68cd \u54fc\u591a\u54c8 \u5a18\u624b\u8d70 \u6211\u60f3\u5c31\u8fd9\u6837\u7275\u7740\u4f60\u7684\u624b\u4e0d\u653e\u5f00 \u7231\u80fd\u4e0d\u80fd\u591f\u6c38\u8fdc\u5355\u7eaf\u6ca1\u6709\u60b2\u5bb3 \u6211 \u60f3\u5e26\u4f60\u9a91\u5355\n - \u4e0d\u5206\u5f00\u5417 \u6211\u53eb\u4f60\u7238 \u4f60\u6253\u6211\u6709 \u8fd9\u6837\u7b11\u4e00\u53ea\u534a\u4f1a \u6211\u4e0d\u80fd\u518d\u8fdc\u7275\u770b\u8457 \u6211\u7231 \u6211\u4e0d\u80fd \u60f3\u60c5\u5c31\u7684\u592a\u5feb\u5c31\u50cf\u9f99\u5377\u98ce \u4e0d\u80fd\u627f\nepoch 200, perplexity 1.563728, time 1.61 sec\n - \u5206\u5f00 \u5feb\u65f6\u7684\u5728\u7b49\u8457  \u6709\u4f60\u4e86 \u6709\u679c\u5e03\u542c\u4e86\u5427? \u6211\u7ed9\u800d\u7684\u8ba9\u6a21\u6709\u6837 \u4ec0\u4e48\u5175\u5668\u6700\u559c\u6b22 \u53cc\u622a\u68cd\u67d4\u4e2d\u5e26\u521a \u60f3\u8981\u53bb\u6cb3\u5357\n - \u4e0d\u5206\u5f00\u671f \u6211\u53eb\u4f60\u7238 \u4f60\u6253\u6211\u5988 \u8fd9\u6837\u5bf9\u5417\u5e72\u561b\u8fd9\u6837 \u4f55\u5fc5\u8ba9\u9152\u7275\u9f3b\u5b50\u8d70 \u778e \u8bf4\u5e95\u6211\u7684\u80a9\u8180 \u4f60 \u5728\u6211\u80f8\u53e3\u7761\u8457 \u50cf\u8fd9\u6837\n"
 }
]
```

接下来采用相邻采样训练模型并创作歌词。

```{.python .input  n=14}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 50, perplexity 59.698101, time 1.62 sec\n - \u5206\u5f00 \u6211\u4e0d\u8981\u518d\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231\n - \u4e0d\u5206\u5f00 \u6211\u4e0d\u8981\u8fd9\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231 \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u7231\n"
 },
 {
  "ename": "KeyboardInterrupt",
  "evalue": "",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-14-5dcd34b7b161>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m                       \u001b[0mchar_to_idx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_steps\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m                       \u001b[0mclipping_theta\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpred_period\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpred_len\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m                       prefixes)\n\u001b[0m",
   "\u001b[0;32m<ipython-input-11-54792c9ce232>\u001b[0m in \u001b[0;36mtrain_and_predict_rnn\u001b[0;34m(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)\u001b[0m\n\u001b[1;32m     36\u001b[0m             \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     37\u001b[0m             \u001b[0;31m# \u88c1\u526a\u68af\u5ea6\u540e\u4f7f\u7528 SGD \u66f4\u65b0\u6743\u91cd\u3002\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 38\u001b[0;31m             \u001b[0mgrad_clipping\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparams\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mclipping_theta\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     39\u001b[0m             \u001b[0mgb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msgd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparams\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# \u56e0\u4e3a\u5df2\u7ecf\u8bef\u5dee\u53d6\u8fc7\u5747\u503c\uff0c\u68af\u5ea6\u4e0d\u7528\u518d\u505a\u5e73\u5747\u3002\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     40\u001b[0m             \u001b[0mloss_sum\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-10-2034c1ff1fda>\u001b[0m in \u001b[0;36mgrad_clipping\u001b[0;34m(params, theta, ctx)\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mparam\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mparams\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m         \u001b[0mnorm\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mparam\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgrad\u001b[0m \u001b[0;34m**\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m     \u001b[0mnorm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnorm\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mnorm\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0mtheta\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mparam\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mparams\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masscalar\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1892\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1893\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"The current array is not a scalar\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1894\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1895\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1896\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masnumpy\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1874\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1875\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_as\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_void_p\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1876\u001b[0;31m             ctypes.c_size_t(data.size)))\n\u001b[0m\u001b[1;32m   1877\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1878\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
  ]
 }
]
```

## 小结

* 我们可以应用基于字符级循环神经网络的语言模型来创作歌词。
* 当训练循环神经网络时，为了应对梯度爆炸，我们可以裁剪梯度。
* 困惑度是对交叉熵损失函数做指数运算后得到的值。

## 练习

* 调调超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 不裁剪梯度，运行本节代码。结果会怎样？
* 如果变化梯度裁剪阈值，需要对学习率做怎样的相应变化？
* 将`pred_period`改为1，观察未充分训练的模型（困惑度高）是如何创作歌词的。你获得了什么启发？
* 将相邻采样改为不从计算图分离隐藏状态，运行时间有没有变化？
* 将本节中使用的激活函数替换成ReLU，重复本节的实验。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/989)

![](../img/qr_rnn-scratch.svg)
